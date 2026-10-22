"""SSW-specific Delta-Mem implementation.

This module keeps the common Delta-Mem kernels, projections, state management,
and adapter utilities in :mod:`delta_impl`, while providing the SSW wrapper
and attachment entry point.  SSW means one memory write per message: token
hidden states belonging to the same message are averaged before the memory
projections and affine scan are run.

Select this implementation by setting ``DELTA_MEM_IMPL=ssw`` before importing
``deltamem.core.delta``.  The default implementation remains ``delta_impl``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from deltamem.core import delta_impl as _base
from deltamem.core.delta_impl import *  # noqa: F401,F403 - preserve public API
from deltamem.core.delta_impl import (
    SUPPORTED_BASE_ATTENTION_TYPES,
    DeltaMemAttention as _BaseDeltaMemAttention,
    HFDeltaMemConfig,
    ensure_attention_compat_views,
)


class DeltaMemAttention(_BaseDeltaMemAttention):
    """Delta-Mem attention whose write policy is fixed to SSW.

    Checkpoints normally already contain ``memory_write_granularity`` set to
    ``message_mean``.  The replacement below also makes an explicitly selected
    SSW implementation safe when an older config says ``token``.
    """

    def __init__(self, base, config: HFDeltaMemConfig) -> None:
        if config.memory_write_granularity != "message_mean":
            config = replace(config, memory_write_granularity="message_mean")
        super().__init__(base, config)
        self.ssw_token_ce: torch.Tensor | None = None
        self.ssw_ce_alpha: float | None = None
        self.ssw_diagnostics_enabled = False
        self.reset_ssw_diagnostics()

    def set_ssw_token_ce(
        self,
        token_ce: torch.Tensor | None,
        alpha: float | None,
    ) -> None:
        """Set externally computed, token-aligned CE scores for the next write pass."""

        self.ssw_token_ce = token_ce
        self.ssw_ce_alpha = None if alpha is None else float(alpha)

    def reset_ssw_diagnostics(self) -> None:
        self.ssw_rep_cosine_sum = 0.0
        self.ssw_rep_cosine_count = 0
        self.ssw_write_count = 0
        self.ssw_overwrite_sum = 0.0
        self.ssw_overwrite_relative_sum = 0.0
        self.ssw_overwrite_count = 0

    def set_ssw_diagnostics_enabled(self, enabled: bool) -> None:
        self.ssw_diagnostics_enabled = bool(enabled)

    def get_ssw_diagnostics(self) -> dict[str, float | int]:
        return {
            "representation_cosine_sum": self.ssw_rep_cosine_sum,
            "representation_cosine_count": self.ssw_rep_cosine_count,
            "layer_write_count": self.ssw_write_count,
            "overwrite_sum": self.ssw_overwrite_sum,
            "overwrite_relative_sum": self.ssw_overwrite_relative_sum,
            "overwrite_count": self.ssw_overwrite_count,
        }

    # These two methods are the SSW-specific write path.  They deliberately
    # live here instead of relying on the token/sentence write implementations
    # in delta_impl.py: SSW groups valid tokens by message id and emits one
    # averaged proposal per message.
    def _message_write_inputs(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.write_enabled or self.write_message_ids is None:
            return None
        message_ids = self.write_message_ids
        if message_ids.dim() != 2:
            return None
        if message_ids.shape[:2] != hidden_states.shape[:2]:
            return None
        message_ids = message_ids.to(device=hidden_states.device)
        active_mask = message_ids.ge(0)
        if token_mask is not None:
            active_mask = active_mask & token_mask
        if not active_mask.any():
            return None
        return message_ids, active_mask

    def _build_message_write_means(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        message_inputs = self._message_write_inputs(hidden_states, token_mask)
        if message_inputs is None:
            return None, None, None
        message_ids, active_mask = message_inputs
        max_message_id = int(message_ids.masked_select(active_mask).max().item())
        num_messages = max_message_id + 1
        message_hidden = hidden_states.new_zeros(
            hidden_states.size(0), num_messages, hidden_states.size(-1)
        )
        message_mask = torch.zeros(
            hidden_states.size(0), num_messages,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        summary_message_ids = torch.full(
            (hidden_states.size(0), num_messages), -1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        for batch_idx in range(hidden_states.size(0)):
            sample_ids = message_ids[batch_idx]
            sample_active = active_mask[batch_idx]
            if not sample_active.any():
                continue
            for message_id in sample_ids.masked_select(sample_active).unique(sorted=True).tolist():
                message_id = int(message_id)
                selector = sample_active & sample_ids.eq(message_id)
                token_hidden = hidden_states[batch_idx, selector]
                mean_hidden = token_hidden.mean(dim=0)
                selected_hidden = mean_hidden
                if self.ssw_token_ce is not None and self.ssw_ce_alpha is not None:
                    if self.ssw_token_ce.shape[:2] != hidden_states.shape[:2]:
                        raise ValueError(
                            "SSW token CE shape must match hidden-state [batch, seq] dimensions: "
                            f"got {tuple(self.ssw_token_ce.shape)} for {tuple(hidden_states.shape[:2])}"
                        )
                    token_ce = self.ssw_token_ce[batch_idx, selector].to(
                        device=hidden_states.device,
                        dtype=torch.float32,
                    )
                    token_ce = torch.nan_to_num(token_ce, nan=0.0, posinf=1e4, neginf=0.0)
                    token_weights = torch.softmax(self.ssw_ce_alpha * token_ce, dim=0)
                    selected_hidden = torch.sum(
                        token_weights.to(dtype=token_hidden.dtype).unsqueeze(-1) * token_hidden,
                        dim=0,
                    )
                    if self.ssw_diagnostics_enabled:
                        cosine = F.cosine_similarity(
                            mean_hidden.float().unsqueeze(0),
                            selected_hidden.float().unsqueeze(0),
                            dim=-1,
                            eps=1e-8,
                        )[0]
                        self.ssw_rep_cosine_sum += float(cosine.detach().cpu())
                        self.ssw_rep_cosine_count += 1
                elif self.ssw_diagnostics_enabled:
                    # Baseline is identical to its own message mean.
                    self.ssw_rep_cosine_sum += 1.0
                    self.ssw_rep_cosine_count += 1
                message_hidden[batch_idx, message_id] = selected_hidden
                message_mask[batch_idx, message_id] = True
                summary_message_ids[batch_idx, message_id] = message_id
                if self.ssw_diagnostics_enabled:
                    self.ssw_write_count += 1
        return message_hidden, message_mask, summary_message_ids

    def _memory_affine_scan(
        self,
        state: torch.Tensor,
        memory_q_seq: torch.Tensor,
        memory_k_seq: torch.Tensor,
        memory_v_seq: torch.Tensor,
        beta_seq: torch.Tensor,
        lambda_seq: torch.Tensor,
        write_route_seq: torch.Tensor | None = None,
        read_route_seq: torch.Tensor | None = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.ssw_diagnostics_enabled or memory_q_seq.size(1) <= 1:
            return super()._memory_affine_scan(
                state,
                memory_q_seq,
                memory_k_seq,
                memory_v_seq,
                beta_seq,
                lambda_seq,
                write_route_seq=write_route_seq,
                read_route_seq=read_route_seq,
                token_mask=token_mask,
            )

        reads: list[torch.Tensor] = []
        for token_idx in range(memory_q_seq.size(1)):
            active = None if token_mask is None else token_mask[:, token_idx]
            state_before = state.detach().float()
            state, token_reads = super()._memory_affine_scan(
                state,
                memory_q_seq[:, token_idx : token_idx + 1],
                memory_k_seq[:, token_idx : token_idx + 1],
                memory_v_seq[:, token_idx : token_idx + 1],
                beta_seq[:, token_idx : token_idx + 1],
                lambda_seq[:, token_idx : token_idx + 1],
                write_route_seq=(
                    None if write_route_seq is None
                    else write_route_seq[:, token_idx : token_idx + 1]
                ),
                read_route_seq=(
                    None if read_route_seq is None
                    else read_route_seq[:, token_idx : token_idx + 1]
                ),
                token_mask=(
                    None if token_mask is None else token_mask[:, token_idx : token_idx + 1]
                ),
            )
            reads.append(token_reads)
            state_after = state.detach().float()
            before_flat = state_before.flatten(1)
            after_flat = state_after.flatten(1)
            before_norm = before_flat.norm(dim=-1)
            valid = before_norm > 1e-8
            if active is not None:
                valid = valid & active.bool()
            if valid.any():
                cosine = F.cosine_similarity(
                    before_flat[valid], after_flat[valid], dim=-1, eps=1e-8
                )
                relative = (
                    (after_flat[valid] - before_flat[valid]).norm(dim=-1)
                    / before_norm[valid].clamp_min(1e-8)
                )
                self.ssw_overwrite_sum += float((1.0 - cosine).sum().cpu())
                self.ssw_overwrite_relative_sum += float(relative.sum().cpu())
                self.ssw_overwrite_count += int(valid.sum().item())
        return state, torch.cat(reads, dim=1)


def attach_delta_mem(model: nn.Module, config: HFDeltaMemConfig) -> list[str]:
    """Wrap attention modules with the SSW Delta-Mem attention class."""

    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, SUPPORTED_BASE_ATTENTION_TYPES):
            continue
        if name.split(".")[-1] not in config.target_modules:
            continue
        if config.target_layers and module.layer_idx not in config.target_layers:
            continue
        module = ensure_attention_compat_views(module)
        parent, attr = _base._get_parent_module(model, name)
        wrapped = DeltaMemAttention(module, config).to(
            device=module.q_proj.weight.device,
            dtype=module.q_proj.weight.dtype,
        )
        setattr(parent, attr, wrapped)
        replaced.append(name)
    if not replaced:
        raise RuntimeError("No target modules were replaced")
    return replaced


__all__ = [
    name for name in dir(_base)
    if not name.startswith("_")
]
__all__ += ["DeltaMemAttention", "attach_delta_mem"]
