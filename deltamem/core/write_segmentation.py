from __future__ import annotations

import re

import torch

_SENTENCE_BOUNDARY_RE = re.compile(
    r"\n+|(?<=[.!?;:])(?:[\"')\]\u201d\u2019]+)?\s+"
)


def split_text_into_sentence_chunks(text: str) -> list[str]:
    if text == "":
        return [""]

    chunks: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.end()
        if end <= start:
            continue
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start = end
    if start < len(text):
        chunks.append(text[start:])
    if not chunks:
        return [text]
    return chunks


def split_text_into_sentence_token_chunks(text: str) -> list[str]:
    if text == "":
        return [""]

    chunks: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        boundary = match.start()
        if boundary > start:
            chunks.append(text[start:boundary])
        start = boundary
    if start < len(text):
        chunks.append(text[start:])
    if not chunks:
        return [text]
    return chunks


def build_surprisal_chunk_ids(
    message_ids: torch.Tensor,
    token_surprisal: torch.Tensor,
    *,
    token_mask: torch.Tensor | None = None,
    threshold: float,
    min_chunk_tokens: int = 1,
    max_chunk_tokens: int | None = None,
) -> torch.Tensor:
    """Segment each message into ordered cumulative-surprisal chunks.

    ``token_surprisal[b, t]`` is assumed to score ``input_ids[b, t]`` (not
    the preceding token).  ``-1`` in the returned tensor means "do not write".
    Chunk ids are contiguous and monotonically ordered *within each sample*.
    The threshold-crossing token belongs to the chunk it closes.
    """
    if message_ids.ndim != 2 or token_surprisal.ndim != 2:
        raise ValueError("message_ids and token_surprisal must be [batch, sequence]")
    if message_ids.shape != token_surprisal.shape:
        raise ValueError("message_ids and token_surprisal must have identical shapes")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if min_chunk_tokens <= 0:
        raise ValueError("min_chunk_tokens must be positive")
    if max_chunk_tokens is not None and max_chunk_tokens < min_chunk_tokens:
        raise ValueError("max_chunk_tokens must be >= min_chunk_tokens")
    if token_mask is not None and token_mask.shape != message_ids.shape:
        raise ValueError("token_mask must match message_ids")

    active = message_ids.ge(0)
    if token_mask is not None:
        active = active & token_mask.bool()
    chunk_ids = torch.full_like(message_ids, -1)
    safe_surprisal = torch.nan_to_num(
        token_surprisal.detach().float(), nan=0.0, posinf=1e4, neginf=0.0
    ).clamp_min_(0.0)

    # This intentionally uses a small CPU-style loop: segmentation is an
    # inference-only control path, while the heavy hidden/projection work stays
    # vectorized. It also makes recurrent write order explicit and testable.
    for batch_idx in range(message_ids.size(0)):
        next_chunk_id = 0
        current_message_id: int | None = None
        running_surprisal = 0.0
        current_length = 0
        current_chunk_id = -1
        close_current_chunk = False
        for token_idx in range(message_ids.size(1)):
            if not bool(active[batch_idx, token_idx]):
                continue
            message_id = int(message_ids[batch_idx, token_idx].item())
            if message_id != current_message_id:
                current_message_id = message_id
                running_surprisal = 0.0
                current_length = 0
                current_chunk_id = next_chunk_id
                next_chunk_id += 1
                close_current_chunk = False
            elif close_current_chunk:
                current_chunk_id = next_chunk_id
                next_chunk_id += 1
                running_surprisal = 0.0
                current_length = 0
                close_current_chunk = False

            chunk_ids[batch_idx, token_idx] = current_chunk_id
            running_surprisal += float(safe_surprisal[batch_idx, token_idx].item())
            current_length += 1
            reached_threshold = running_surprisal >= threshold
            reached_max = max_chunk_tokens is not None and current_length >= max_chunk_tokens
            if current_length >= min_chunk_tokens and (reached_threshold or reached_max):
                # The next active token in this same message starts a new
                # chunk. Delay id allocation so a final boundary does not
                # create an unused gap before the next message.
                close_current_chunk = True
    return chunk_ids


def group_hidden_by_ids(
    hidden_states: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Arithmetic-mean hidden summaries, preserving ascending group order."""
    if hidden_states.ndim != 3 or group_ids.shape != hidden_states.shape[:2]:
        raise ValueError("hidden_states must be [B,T,H] and group_ids must be [B,T]")
    active = group_ids.ge(0)
    if token_mask is not None:
        if token_mask.shape != group_ids.shape:
            raise ValueError("token_mask must match group_ids")
        active = active & token_mask.bool()
    if not active.any():
        return None, None, None
    max_group_id = int(group_ids.masked_select(active).max().item())
    num_groups = max_group_id + 1
    summaries = hidden_states.new_zeros(hidden_states.size(0), num_groups, hidden_states.size(-1))
    group_mask = torch.zeros(hidden_states.size(0), num_groups, dtype=torch.bool, device=hidden_states.device)
    summary_ids = torch.full((hidden_states.size(0), num_groups), -1, dtype=torch.long, device=hidden_states.device)
    for batch_idx in range(hidden_states.size(0)):
        sample_active = active[batch_idx]
        for group_id in group_ids[batch_idx, sample_active].unique(sorted=True).tolist():
            group_id = int(group_id)
            selector = sample_active & group_ids[batch_idx].eq(group_id)
            summaries[batch_idx, group_id] = hidden_states[batch_idx, selector].mean(dim=0)
            group_mask[batch_idx, group_id] = True
            summary_ids[batch_idx, group_id] = group_id
    return summaries, group_mask, summary_ids


def build_sentence_chunk_ids(
    message_ids: torch.Tensor,
    sentence_ids: torch.Tensor,
    *,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use each aligned sentence as one write chunk, with hard message edges."""
    if message_ids.shape != sentence_ids.shape or message_ids.ndim != 2:
        raise ValueError("message_ids and sentence_ids must have shape [batch, sequence]")
    active = message_ids.ge(0) & sentence_ids.ge(0)
    if token_mask is not None:
        if token_mask.shape != message_ids.shape:
            raise ValueError("token_mask must match message_ids")
        active = active & token_mask.bool()
    result = torch.full_like(message_ids, -1)
    for batch_idx in range(message_ids.size(0)):
        next_chunk = 0
        seen: dict[tuple[int, int], int] = {}
        for token_idx in range(message_ids.size(1)):
            if not bool(active[batch_idx, token_idx]):
                continue
            key = (int(message_ids[batch_idx, token_idx]), int(sentence_ids[batch_idx, token_idx]))
            if key not in seen:
                seen[key] = next_chunk
                next_chunk += 1
            result[batch_idx, token_idx] = seen[key]
    return result


def build_embedding_semantic_chunk_ids(
    message_ids: torch.Tensor,
    sentence_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    *,
    semantic_threshold: float,
    token_mask: torch.Tensor | None = None,
    percentile: float | None = None,
    max_chunk_tokens: int | None = 512,
    min_chunk_tokens: int = 1,
    respect_message_boundaries: bool = False,
) -> torch.Tensor:
    """Build token-aligned semantic chunks from sentence-level embeddings.

    ``token_embeddings`` is expected to contain the same sentence embedding at
    each token position in that sentence.  The older absolute-threshold mode
    remains available through ``semantic_threshold``.  When ``percentile`` is
    supplied, boundaries are placed at distances at or above that percentile
    (computed independently for each sample), which is the preferred mode for
    the Embedding-Semantic SSW experiment.

    Sentence ids are allowed to continue across message ids.  In that mode the
    sequence is treated as one chronological context, so a semantic chunk can
    cross an original message boundary.  ``respect_message_boundaries`` keeps
    the legacy hard-boundary behavior for ablations.
    """
    if token_embeddings.ndim != 3 or token_embeddings.shape[:2] != message_ids.shape:
        raise ValueError("token_embeddings must have shape [batch, sequence, hidden]")
    if semantic_threshold < 0:
        raise ValueError("semantic_threshold must be non-negative")
    if percentile is not None and not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    if max_chunk_tokens is not None and max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive or None")
    if min_chunk_tokens <= 0:
        raise ValueError("min_chunk_tokens must be positive")
    active = message_ids.ge(0) & sentence_ids.ge(0)
    if token_mask is not None:
        if token_mask.shape != message_ids.shape:
            raise ValueError("token_mask must match message_ids")
        active = active & token_mask.bool()
    result = torch.full_like(message_ids, -1)
    embeddings = token_embeddings.float()
    for batch_idx in range(message_ids.size(0)):
        # Recover sentence units in chronological token order.  The tuple key
        # avoids accidental collisions when callers use per-message sentence
        # numbering.
        sentence_keys: list[tuple[int, int]] = []
        sentence_embeddings: list[torch.Tensor] = []
        sentence_lengths: list[int] = []
        seen: set[tuple[int, int]] = set()
        for token_idx in range(message_ids.size(1)):
            if not bool(active[batch_idx, token_idx]):
                continue
            key = (int(message_ids[batch_idx, token_idx]), int(sentence_ids[batch_idx, token_idx]))
            if key in seen:
                continue
            seen.add(key)
            selector = active[batch_idx] & message_ids[batch_idx].eq(key[0]) & sentence_ids[batch_idx].eq(key[1])
            sentence_keys.append(key)
            sentence_embeddings.append(embeddings[batch_idx, selector].mean(dim=0))
            sentence_lengths.append(int(selector.sum().item()))
        if not sentence_keys:
            continue

        distances: list[float] = []
        for idx in range(len(sentence_embeddings) - 1):
            if respect_message_boundaries and sentence_keys[idx][0] != sentence_keys[idx + 1][0]:
                distances.append(float("inf"))
                continue
            similarity = torch.nn.functional.cosine_similarity(
                sentence_embeddings[idx].unsqueeze(0),
                sentence_embeddings[idx + 1].unsqueeze(0),
                dim=-1,
                eps=1e-8,
            )[0]
            distances.append(1.0 - float(similarity.item()))
        if percentile is not None and distances:
            cutoff = float(torch.tensor(distances, dtype=torch.float32).quantile(percentile / 100.0).item())
            cuts = [distance >= cutoff for distance in distances]
        else:
            cuts = [distance >= semantic_threshold for distance in distances]
        if respect_message_boundaries:
            cuts = [cut or sentence_keys[idx][0] != sentence_keys[idx + 1][0] for idx, cut in enumerate(cuts)]

        # First create provisional semantic units at candidate boundaries.
        provisional: list[list[int]] = []
        current: list[int] = []
        for idx, length in enumerate(sentence_lengths):
            if current and (
                cuts[idx - 1]
                or (respect_message_boundaries and sentence_keys[idx - 1][0] != sentence_keys[idx][0])
            ):
                provisional.append(current)
                current = []
            current.append(idx)
        if current:
            provisional.append(current)

        # Then pack adjacent provisional units toward the target size.  This
        # is intentionally independent of candidate cuts: a semantic cut is a
        # place where a chunk may end, not a mandate to emit a tiny write.
        groups: list[list[int]] = []
        current = []
        current_length = 0
        for unit in provisional:
            unit_length = sum(sentence_lengths[idx] for idx in unit)
            if current and max_chunk_tokens is not None and current_length + unit_length > max_chunk_tokens:
                groups.append(current)
                current = []
                current_length = 0
            current.extend(unit)
            current_length += unit_length
        if current:
            groups.append(current)

        # If a semantic cut produced a very short unit, merge it forward when
        # possible.  This avoids one-token writes without changing chronology.
        if min_chunk_tokens > 1 and len(groups) > 1:
            merged: list[list[int]] = []
            for group in groups:
                group_length = sum(sentence_lengths[idx] for idx in group)
                if merged and group_length < min_chunk_tokens:
                    previous_length = sum(sentence_lengths[idx] for idx in merged[-1])
                    same_message = sentence_keys[merged[-1][-1]][0] == sentence_keys[group[0]][0]
                    fits = max_chunk_tokens is None or previous_length + group_length <= max_chunk_tokens
                    if fits and (not respect_message_boundaries or same_message):
                        merged[-1].extend(group)
                        continue
                merged.append(group)
            groups = merged

        chunk_for_sentence: dict[tuple[int, int], int] = {}
        for chunk_id, group in enumerate(groups):
            for sentence_index in group:
                chunk_for_sentence[sentence_keys[sentence_index]] = chunk_id
        for token_idx in range(message_ids.size(1)):
            if bool(active[batch_idx, token_idx]):
                key = (int(message_ids[batch_idx, token_idx]), int(sentence_ids[batch_idx, token_idx]))
                result[batch_idx, token_idx] = chunk_for_sentence[key]
    return result


__all__ = [
    "build_surprisal_chunk_ids",
    "build_sentence_chunk_ids",
    "build_embedding_semantic_chunk_ids",
    "group_hidden_by_ids",
    "split_text_into_sentence_chunks",
    "split_text_into_sentence_token_chunks",
]
