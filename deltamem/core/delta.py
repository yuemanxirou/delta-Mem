from __future__ import annotations

from dataclasses import dataclass
import os

from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from deltamem.core.backbone_compat import SmolLM3Attention, ensure_attention_compat_views
_DELTA_MEM_IMPL = os.environ.get("DELTA_MEM_IMPL", "default").strip().lower()
_SSW_IMPLS = {"ssw", "delta_impl_ssw"}
_SSW_CE_WEIGHTED_IMPLS = {"ssw_ce_weighted", "ce_ssw", "delta_impl_ce_ssw"}
_SSW_SURPRISAL_IMPLS = {"ssw_surprisal", "surprisal_ssw", "delta_impl_ssw_surprisal"}
_SENTENCE_SSW_IMPLS = {"sentence_ssw", "ssw_sentence", "delta_impl_sentence_ssw"}
if _DELTA_MEM_IMPL in {"", "default", "delta", "delta_impl"}:
    from deltamem.core import delta_impl as _implementation
elif _DELTA_MEM_IMPL in _SSW_IMPLS:
    from deltamem.core import delta_impl_ssw as _implementation
elif _DELTA_MEM_IMPL in _SSW_CE_WEIGHTED_IMPLS:
    from deltamem.core import delta_impl_ssw_ce_weighted as _implementation
elif _DELTA_MEM_IMPL in _SSW_SURPRISAL_IMPLS:
    from deltamem.core import delta_impl_ssw_surprisal as _implementation
elif _DELTA_MEM_IMPL in _SENTENCE_SSW_IMPLS:
    from deltamem.core import delta_impl_sentence_ssw as _implementation
else:
    raise ValueError(
            "Unsupported DELTA_MEM_IMPL={!r}; expected default, ssw, ssw_ce_weighted, ssw_surprisal, or sentence_ssw".format(_DELTA_MEM_IMPL)
    )

from deltamem.core.delta_impl import (
    VALID_DELTA_HEADS,
    VALID_MEMORY_PARTITION_BASIS,
    VALID_MEMORY_PARTITION_ROUTING,
    VALID_STATE_UPDATE_MODES,
    collect_delta_mem_gate_stats,
    collect_delta_mem_partition_route_stats,
    collect_delta_mem_state_stats,
    collect_delta_mem_weight_stats,
    diff_delta_mem_snapshots,
    freeze_non_delta_mem_params,
    get_delta_mem_online_state,
    get_delta_mem_partition_regularization,
    get_delta_mem_write_regularization,
    iter_delta_mem_modules,
    load_delta_mem_adapter,
    load_delta_mem_online_state,
    normalize_delta_heads,
    normalize_memory_partition_basis,
    normalize_memory_partition_routing,
    normalize_state_update_mode,
    reset_delta_mem_states,
    save_delta_mem_adapter,
    set_delta_mem_read_context_mask,
    set_delta_mem_write_enabled,
    set_delta_mem_write_message_ids,
    set_delta_mem_write_sentence_ids,
    snapshot_delta_mem_weights,
    HFDeltaMemConfig as ExperimentalHFDeltaMemConfig,
)

# The common helpers above intentionally remain imported from delta_impl for
# backwards compatibility.  The implementation-specific attention wrapper
# and attachment function are selected through DELTA_MEM_IMPL.
ExperimentalDeltaMemAttention = _implementation.DeltaMemAttention

VALID_MEMORY_READOUT_MODES = ("delta",)


def normalize_memory_readout_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized != "delta":
        raise ValueError(
            "Mainline Delta-Mem only supports memory_readout_mode='delta'. "
            "Use deltamem.core.delta_impl for experimental readouts."
        )
    return normalized


@dataclass(frozen=True)
class HFDeltaMemConfig(ExperimentalHFDeltaMemConfig):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.memory_readout_mode != "delta":
            raise ValueError(
                "Mainline HFDeltaMemConfig only supports memory_readout_mode='delta'. "
                "Archived synthetic_kv / latent_context / memory_branch readouts were removed."
            )


if _DELTA_MEM_IMPL in _SSW_IMPLS | _SSW_CE_WEIGHTED_IMPLS | _SSW_SURPRISAL_IMPLS | _SENTENCE_SSW_IMPLS:
    # Expose the selected implementation class so isinstance checks used by
    # training/evaluation utilities continue to match wrapped modules.
    DeltaMemAttention = ExperimentalDeltaMemAttention
else:
    class DeltaMemAttention(ExperimentalDeltaMemAttention):
        def __init__(self, base: Qwen3Attention | SmolLM3Attention, config: HFDeltaMemConfig) -> None:
            if config.memory_readout_mode != "delta":
                raise ValueError(
                    "Mainline DeltaMemAttention only supports memory_readout_mode='delta'."
                )
            super().__init__(base, config)


def _get_parent_module(root, module_name: str):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def attach_delta_mem(model, config: HFDeltaMemConfig) -> list[str]:
    if config.memory_readout_mode != "delta":
        raise ValueError(
            "attach_delta_mem in mainline only supports memory_readout_mode='delta'."
        )
    if _DELTA_MEM_IMPL in _SSW_IMPLS | _SSW_CE_WEIGHTED_IMPLS | _SSW_SURPRISAL_IMPLS | _SENTENCE_SSW_IMPLS:
        return _implementation.attach_delta_mem(model, config)
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, (Qwen3Attention, SmolLM3Attention)):
            continue
        if name.split(".")[-1] not in config.target_modules:
            continue
        if config.target_layers and module.layer_idx not in config.target_layers:
            continue
        module = ensure_attention_compat_views(module)
        parent, attr = _get_parent_module(model, name)
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
    "VALID_DELTA_HEADS",
    "VALID_MEMORY_PARTITION_BASIS",
    "VALID_MEMORY_PARTITION_ROUTING",
    "VALID_MEMORY_READOUT_MODES",
    "VALID_STATE_UPDATE_MODES",
    "DeltaMemAttention",
    "HFDeltaMemConfig",
    "attach_delta_mem",
    "collect_delta_mem_gate_stats",
    "collect_delta_mem_partition_route_stats",
    "collect_delta_mem_state_stats",
    "collect_delta_mem_weight_stats",
    "diff_delta_mem_snapshots",
    "freeze_non_delta_mem_params",
    "get_delta_mem_online_state",
    "get_delta_mem_partition_regularization",
    "get_delta_mem_write_regularization",
    "iter_delta_mem_modules",
    "load_delta_mem_adapter",
    "load_delta_mem_online_state",
    "normalize_delta_heads",
    "normalize_memory_partition_basis",
    "normalize_memory_partition_routing",
    "normalize_memory_readout_mode",
    "normalize_state_update_mode",
    "reset_delta_mem_states",
    "save_delta_mem_adapter",
    "set_delta_mem_read_context_mask",
    "set_delta_mem_write_enabled",
    "set_delta_mem_write_message_ids",
    "set_delta_mem_write_sentence_ids",
    "snapshot_delta_mem_weights",
]
