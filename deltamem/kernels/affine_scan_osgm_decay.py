from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised through the fallback path.
    triton = None
    tl = None

import math
# osgm_decay new
# from .fla.ops.os_delta_rule.wy_fast_os import prepare_wy_repr_bwd, prepare_wy_repr_fwd, recompute_w_u_fwd
from .fla.ops.os_delta_rule.chunk_osgm_phase_decay import (
    compute_osgm_decay_phase1_fwd as compute_osgm_dd_decay_phase1_fwd_rec,
    compute_osgm_decay_phase1_bwd as compute_osgm_dd_decay_phase1_bwd_rec,
    # fused_osgm_bwd_mapping
)
'''
from .fla.ops.os_delta_rule.chunk_osgm_phase_wy import (
    compute_osgm_phase1_fwd_wy,
    compute_osgm_phase1_bwd_wy
)
'''

@dataclass(frozen=True)
class TritonScanSupport:
    supported: bool
    reason: str = ""


def _next_power_of_2(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def triton_scan_support(
    state: torch.Tensor, # [B, R, R]
    q: torch.Tensor, # [B, T, R]
    k_read: torch.Tensor, # [B, T, R]
    k_write: torch.Tensor, # [B, T, R]
    v: torch.Tensor, # [B, T, R]
    keep: torch.Tensor, # [B, T, R]
    erase: torch.Tensor, # [B, T, R]
    write: torch.Tensor, # [B, T, R]
    gamma_log: torch.Tensor, # [H], H=1
) -> TritonScanSupport:
    if triton is None or tl is None:
        return TritonScanSupport(False, "triton is unavailable")
    if not state.is_cuda:
        return TritonScanSupport(False, "state tensor is not on CUDA")
    if any(tensor.device != state.device for tensor in (q, k_read, k_write, v, keep, erase, write)):
        return TritonScanSupport(False, "all tensors must be on the same CUDA device")
    if state.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return TritonScanSupport(False, f"unsupported dtype {state.dtype}")
    if (
        state.ndim != 3
        or q.ndim != 3
        or k_read.ndim != 3
        or k_write.ndim != 3
        or v.ndim != 3
        or keep.ndim != 3
        or erase.ndim != 3
        or write.ndim != 3
    ):
        return TritonScanSupport(False, "expected 3D tensors")
    batch_size, rank, rank2 = state.shape
    if rank != rank2:
        return TritonScanSupport(False, "state must be square in the last two dims")
    if q.size(0) != batch_size or q.size(2) != rank:
        return TritonScanSupport(False, "q shape does not match state rank")
    if k_read.shape != q.shape or k_write.shape != q.shape or v.shape != q.shape or keep.shape != q.shape or erase.shape != q.shape or write.shape != q.shape:
        return TritonScanSupport(False, "q/k_read/k_write/v/keep/erase/write must share shape [batch, seq, rank]")
    if any(
        tensor.dtype != state.dtype
        for tensor in (q, k_read, k_write, v, keep, erase, write)
    ):
        return TritonScanSupport(
            False,
            "all floating-point tensors must share dtype",
        )
    if gamma_log.ndim != 1 or gamma_log.numel() != 1:
        return TritonScanSupport(
            False,
            "gamma_log must have shape [1] when affine scan uses H=1",
        )
    return TritonScanSupport(True)


if triton is not None and tl is not None:

    @triton.jit
    def _affine_scan_forward_kernel(
    state0_ptr,
    q_ptr,
    k_read_ptr,
    k_write_ptr,
    v_ptr,
    keep_ptr,
    erase_ptr,
    write_ptr,
    mask_ptr,
    state_hist_ptr,
    reads_ptr,
    state_out_ptr,
    batch_size,
    seq_len,
    rank,
    state_b_stride,
    state_row_stride,
    state_col_stride,
    q_b_stride,
    q_t_stride,
    q_r_stride,
    scalar_b_stride,
    scalar_t_stride,
    scalar_r_stride,
    mask_b_stride,
    mask_t_stride,
    hist_b_stride,
    hist_t_stride,
    hist_row_stride,
    hist_col_stride,
    reads_b_stride,
    reads_t_stride,
    reads_r_stride,
    HAS_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        batch_idx = pid // rank
        row_idx = pid % rank
        if batch_idx >= batch_size:
            return

        cols = tl.arange(0, BLOCK_R)
        col_mask = cols < rank

        state_offsets = (
            batch_idx * state_b_stride
            + row_idx * state_row_stride
            + cols * state_col_stride
        )
        state_vec = tl.load(state0_ptr + state_offsets, mask=col_mask, other=0.0).to(tl.float32)

        t = 0
        while t < seq_len:
            hist_offsets = (
                batch_idx * hist_b_stride
                + t * hist_t_stride
                + row_idx * hist_row_stride
                + cols * hist_col_stride
            )
            tl.store(state_hist_ptr + hist_offsets, state_vec, mask=col_mask) # 这是为了反向传播

            valid = tl.full((), 1, tl.int32)
            if HAS_MASK:
                valid = tl.load(mask_ptr + batch_idx * mask_b_stride + t * mask_t_stride)
            valid_f = valid.to(tl.float32)

            q_offsets = batch_idx * q_b_stride + t * q_t_stride + cols * q_r_stride
            q_vec = tl.load(q_ptr + q_offsets, mask=col_mask, other=0.0).to(tl.float32)
            read = tl.sum(state_vec * q_vec, axis=0) * valid_f
            tl.store(
                reads_ptr + batch_idx * reads_b_stride + t * reads_t_stride + row_idx * reads_r_stride,
                read,
            )

            k_read_vec = tl.load(k_read_ptr + q_offsets, mask=col_mask, other=0.0).to(tl.float32)
            k_write_vec = tl.load(k_write_ptr + q_offsets, mask=col_mask, other=0.0).to(tl.float32)
            scalar_offset = batch_idx * scalar_b_stride + t * scalar_t_stride + row_idx * scalar_r_stride
            v_scalar = tl.load(v_ptr + scalar_offset).to(tl.float32) # shape: [B, T, R] -> [B, T, R]，所以这里是一个标量
            keep_scalar = tl.load(keep_ptr + scalar_offset).to(tl.float32)
            erase_scalar = tl.load(erase_ptr + scalar_offset).to(tl.float32)
            write_scalar = tl.load(write_ptr + scalar_offset).to(tl.float32)
            dot = tl.sum(state_vec * k_read_vec, axis=0)
            updated = (
                keep_scalar * state_vec
                - erase_scalar * dot * k_write_vec
                + write_scalar * v_scalar * k_write_vec
            )
            state_vec = tl.where(valid != 0, updated, state_vec)
            t += 1

        tl.store(state_out_ptr + state_offsets, state_vec, mask=col_mask)


    @triton.jit
    def _affine_scan_backward_kernel(
    state_hist_ptr,
    q_ptr,
    k_read_ptr,
    k_write_ptr,
    v_ptr,
    keep_ptr,
    erase_ptr,
    write_ptr,
    mask_ptr,
    grad_reads_ptr,
    grad_state_out_ptr,
    grad_state0_ptr,
    grad_q_ptr,
    grad_k_read_ptr,
    grad_k_write_ptr,
    grad_v_ptr,
    grad_keep_ptr,
    grad_erase_ptr,
    grad_write_ptr,
    batch_size,
    seq_len,
    rank,
    hist_b_stride,
    hist_t_stride,
    hist_row_stride,
    hist_col_stride,
    q_b_stride,
    q_t_stride,
    q_r_stride,
    scalar_b_stride,
    scalar_t_stride,
    scalar_r_stride,
    mask_b_stride,
    mask_t_stride,
    grad_reads_b_stride,
    grad_reads_t_stride,
    grad_reads_r_stride,
    grad_state_b_stride,
    grad_state_row_stride,
    grad_state_col_stride,
    grad_q_b_stride,
    grad_q_t_stride,
    grad_q_r_stride,
    HAS_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        batch_idx = pid // rank
        row_idx = pid % rank
        if batch_idx >= batch_size:
            return

        cols = tl.arange(0, BLOCK_R)
        col_mask = cols < rank

        grad_state_offsets = (
            batch_idx * grad_state_b_stride
            + row_idx * grad_state_row_stride
            + cols * grad_state_col_stride
        )
        grad_state = tl.load(grad_state_out_ptr + grad_state_offsets, mask=col_mask, other=0.0).to(tl.float32)

        t = seq_len - 1
        while t >= 0:
            valid = tl.full((), 1, tl.int32)
            if HAS_MASK:
                valid = tl.load(mask_ptr + batch_idx * mask_b_stride + t * mask_t_stride)

            if valid != 0:
                hist_offsets = (
                    batch_idx * hist_b_stride
                    + t * hist_t_stride
                    + row_idx * hist_row_stride
                    + cols * hist_col_stride
                )
                state_prev = tl.load(state_hist_ptr + hist_offsets, mask=col_mask, other=0.0).to(tl.float32)

                q_offsets = batch_idx * q_b_stride + t * q_t_stride + cols * q_r_stride
                q_vec = tl.load(q_ptr + q_offsets, mask=col_mask, other=0.0).to(tl.float32)
                k_read_vec = tl.load(k_read_ptr + q_offsets, mask=col_mask, other=0.0).to(tl.float32)
                k_write_vec = tl.load(k_write_ptr + q_offsets, mask=col_mask, other=0.0).to(tl.float32)

                scalar_offset = batch_idx * scalar_b_stride + t * scalar_t_stride + row_idx * scalar_r_stride
                v_scalar = tl.load(v_ptr + scalar_offset).to(tl.float32)
                keep_scalar = tl.load(keep_ptr + scalar_offset).to(tl.float32)
                erase_scalar = tl.load(erase_ptr + scalar_offset).to(tl.float32)
                write_scalar = tl.load(write_ptr + scalar_offset).to(tl.float32)
                grad_read = tl.load(
                    grad_reads_ptr
                    + batch_idx * grad_reads_b_stride
                    + t * grad_reads_t_stride
                    + row_idx * grad_reads_r_stride
                ).to(tl.float32)

                tl.atomic_add(
                    grad_q_ptr + batch_idx * grad_q_b_stride + t * grad_q_t_stride + cols * grad_q_r_stride,
                    grad_read * state_prev,
                    mask=col_mask,
                )

                grad_state_dot_k_write = tl.sum(grad_state * k_write_vec, axis=0)
                state_prev_dot_k_read = tl.sum(state_prev * k_read_vec, axis=0)

                tl.atomic_add(
                    grad_k_write_ptr
                    + batch_idx * grad_q_b_stride
                    + t * grad_q_t_stride
                    + cols * grad_q_r_stride,
                    (
                        write_scalar * v_scalar
                        - erase_scalar * state_prev_dot_k_read
                    ) * grad_state,
                    mask=col_mask,
                )

                tl.atomic_add(
                    grad_k_read_ptr
                    + batch_idx * grad_q_b_stride
                    + t * grad_q_t_stride
                    + cols * grad_q_r_stride,
                    -erase_scalar
                    * grad_state_dot_k_write
                    * state_prev,
                    mask=col_mask,
                )

                tl.store(grad_v_ptr + scalar_offset, write_scalar * grad_state_dot_k_write)
                tl.store(grad_keep_ptr + scalar_offset, tl.sum(grad_state * state_prev, axis=0))
                tl.store(grad_erase_ptr + scalar_offset, -state_prev_dot_k_read * grad_state_dot_k_write)
                tl.store(grad_write_ptr + scalar_offset, v_scalar * grad_state_dot_k_write)

                grad_state = (
                    grad_read * q_vec
                    + keep_scalar * grad_state
                    - erase_scalar * grad_state_dot_k_write * k_read_vec
                )

            t -= 1

        tl.store(grad_state0_ptr + grad_state_offsets, grad_state, mask=col_mask)


else:

    _affine_scan_forward_kernel = None
    _affine_scan_backward_kernel = None

def affine_scan_osgm_decay_fwd(
    state0,
    q,
    k_read,
    k_write,
    v,
    keep,
    erase,
    write,
    token_mask,
):
    state0 = state0.contiguous()
    q = q.contiguous()
    k_read = k_read.contiguous()
    k_write = k_write.contiguous()
    v = v.contiguous()
    keep = keep.contiguous()
    erase = erase.contiguous()
    write = write.contiguous()
    mask_tensor = None if token_mask is None else token_mask.to(device=state0.device, dtype=torch.int32).contiguous()

    batch_size, rank, _ = state0.shape
    seq_len = q.size(1)
    reads = torch.empty_like(v)
    state_out = torch.empty_like(state0)
    state_hist = torch.empty((batch_size, seq_len, rank, rank), device=state0.device, dtype=state0.dtype) # （B, T, R, R）用于保存每个时间步的state，方便反向传播

    block_r = _next_power_of_2(rank)
    grid = (batch_size * rank,) # 启动batch_size * rank个program
    _affine_scan_forward_kernel[grid](
        state0,
        q,
        k_read,
        k_write,
        v,
        keep,
        erase,
        write,
        mask_tensor if mask_tensor is not None else state0,
        state_hist,
        reads,
        state_out,
        batch_size,
        seq_len,
        rank,
        state0.stride(0),
        state0.stride(1),
        state0.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        0 if mask_tensor is None else mask_tensor.stride(0),
        0 if mask_tensor is None else mask_tensor.stride(1),
        state_hist.stride(0),
        state_hist.stride(1),
        state_hist.stride(2),
        state_hist.stride(3),
        reads.stride(0),
        reads.stride(1),
        reads.stride(2),
        HAS_MASK=mask_tensor is not None,
        BLOCK_R=block_r,
    )

    return state_out, reads, state_hist

def affine_scan_osgm_decay_bwd(
    state_hist,
    q,
    k_read,
    k_write,
    v,
    keep,
    erase,
    write,
    token_mask,
    grad_state_out,
    grad_reads,
):
    if grad_reads is None:
        grad_reads = torch.zeros_like(v)
    else:
        grad_reads = grad_reads.contiguous()
    if grad_state_out is None:
        grad_state_out = torch.zeros(
            state_hist.size(0),
            state_hist.size(2),
            state_hist.size(3),
            device=state_hist.device,
            dtype=state_hist.dtype,
        )
    else:
        grad_state_out = grad_state_out.contiguous()

    batch_size, seq_len, rank, _ = state_hist.shape
    grad_state0 = torch.empty_like(grad_state_out)
    grad_q = torch.zeros_like(q, dtype=torch.float32)
    grad_k_read = torch.zeros_like(k_read, dtype=torch.float32)
    grad_k_write = torch.zeros_like(k_write, dtype=torch.float32)
    grad_v = torch.zeros_like(v, dtype=torch.float32)
    grad_keep = torch.zeros_like(keep, dtype=torch.float32)
    grad_erase = torch.zeros_like(erase, dtype=torch.float32)
    grad_write = torch.zeros_like(write, dtype=torch.float32)

    mask_tensor = (
        None
        if token_mask is None or token_mask.numel() == 0
        else token_mask.to(
            device=state_hist.device,
            dtype=torch.int32,
        ).contiguous()
    )

    block_r = _next_power_of_2(rank)
    grid = (batch_size * rank,)
    _affine_scan_backward_kernel[grid](
        state_hist,
        q,
        k_read,
        k_write,
        v,
        keep,
        erase,
        write,
        mask_tensor if mask_tensor is not None else q,
        grad_reads,
        grad_state_out,
        grad_state0,
        grad_q,
        grad_k_read,
        grad_k_write,
        grad_v,
        grad_keep,
        grad_erase,
        grad_write,
        batch_size,
        seq_len,
        rank,
        state_hist.stride(0),
        state_hist.stride(1),
        state_hist.stride(2),
        state_hist.stride(3),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        0 if mask_tensor is None else mask_tensor.stride(0),
        0 if mask_tensor is None else mask_tensor.stride(1),
        grad_reads.stride(0),
        grad_reads.stride(1),
        grad_reads.stride(2),
        grad_state0.stride(0),
        grad_state0.stride(1),
        grad_state0.stride(2),
        grad_q.stride(0),
        grad_q.stride(1),
        grad_q.stride(2),
        HAS_MASK=mask_tensor is not None,
        BLOCK_R=block_r,
    )

    return (
        grad_state0,
        grad_q,
        grad_k_read,
        grad_k_write,
        grad_v,
        grad_keep,
        grad_erase,
        grad_write,
    )


class _TritonAffineScanOSGMFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        state0: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        keep: torch.Tensor,
        erase: torch.Tensor,
        write: torch.Tensor,
        token_mask: torch.Tensor | None,
        # osgm_decay new
        gamma_log,
        initial_d,
        eta,
        use_denominator,
        d_min,
        d_max,
        output_final_state,
        #use_wy_fast
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None,]:
        state0 = state0.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        keep = keep.contiguous()
        erase = erase.contiguous()
        write = write.contiguous()
        mask_tensor = None if token_mask is None else token_mask.to(device=state0.device, dtype=torch.int32).contiguous()

        batch_size, seq_len, rank = k.shape

        # Phase 1 kernel要求 [B,T,H,K]，这里用 H=1
        k_4d = k.unsqueeze(2)
        if initial_d is not None:
            if initial_d.ndim == 2:
                initial_d = initial_d.unsqueeze(1)
            elif initial_d.ndim != 3:
                raise RuntimeError(
                    f"initial_d must have shape [B, R] or [B, 1, R], "
                    f"got {tuple(initial_d.shape)}"
                )
            initial_d = initial_d.to(
                device=k.device, dtype=torch.float32
            ).contiguous()

        gamma_log = gamma_log.to(
            device=k.device,
            dtype=torch.float32
        )

        d_4d, final_d = compute_osgm_dd_decay_phase1_fwd_rec(
            k=k_4d, gamma_log=gamma_log, eta=eta, use_denominator=use_denominator, d_min=d_min, d_max=d_max, cu_seqlens=None, initial_d=initial_d, output_final_state=output_final_state
        )

        # 映射
        d = d_4d.squeeze(2)  # [B, T, K], dtype=FP32
        k_write = (d * k.to(torch.float32)).to(dtype=k.dtype)

        state_out, reads, state_hist = affine_scan_osgm_decay_fwd(
            state0=state0,
            q=q,
            k_read=k,
            k_write=k_write,
            v=v,
            keep=keep,
            erase=erase,
            write=write,
            token_mask=token_mask,
        )

        ctx.save_for_backward(
            state_hist,
            q,
            k,
            d,
            k_write,
            v,
            keep,
            erase,
            write,
            gamma_log,
            mask_tensor if mask_tensor is not None else torch.empty(0, device=state0.device, dtype=torch.int32),
        )

        ctx.eta = eta
        ctx.use_denominator = use_denominator
        ctx.d_min = d_min
        ctx.d_max = d_max
        ctx.has_initial_d = initial_d is not None
        ctx.has_token_mask = token_mask is not None

        return state_out, reads, final_d

    @staticmethod
    def backward(
        ctx,
        grad_state_out,
        grad_reads,
        grad_final_d,
    ):
        (
            state_hist,
            q,
            k,
            d,
            k_write,
            v,
            keep,
            erase,
            write,
            gamma_log,
            token_mask,
        ) = ctx.saved_tensors

        (
            grad_state0,
            grad_q,
            grad_k_read,
            grad_k_write,
            grad_v,
            grad_keep,
            grad_erase,
            grad_write,
        ) = affine_scan_osgm_decay_bwd(
            state_hist=state_hist,
            q=q,
            k_read=k,
            k_write=k_write,
            v=v,
            keep=keep,
            erase=erase,
            write=write,
            token_mask=token_mask,
            grad_state_out=grad_state_out,
            grad_reads=grad_reads,
        )

        # k_write = (d * k_fp32).to(k.dtype)
        grad_k_write_fp32 = grad_k_write.to(torch.float32)
        k_fp32 = k.to(torch.float32)
        grad_d = grad_k_write_fp32 * k_fp32
        grad_k_phase2 = grad_k_read.to(torch.float32) + grad_k_write_fp32 * d
        grad_k_phase1_4d, grad_initial_d, grad_gamma_log = compute_osgm_dd_decay_phase1_bwd_rec(
            k=k.unsqueeze(2),
            gamma_log=gamma_log,
            d_out=d.unsqueeze(2),
            dd_in=grad_d.unsqueeze(2),
            eta=ctx.eta,
            use_denominator=ctx.use_denominator,
            d_min=ctx.d_min,
            d_max=ctx.d_max,
            cu_seqlens=None,
            dd_final=grad_final_d,
            output_initial_state_gradient=ctx.has_initial_d,
        )

        grad_k_phase1 = grad_k_phase1_4d.squeeze(2)
        grad_k = (grad_k_phase2 + grad_k_phase1.to(torch.float32)).to(dtype=k.dtype)

        return (
            grad_state0,                                # state0
            grad_q,                                     # q
            grad_k,                                     # k
            grad_v,                                     # v
            grad_keep,                                  # keep
            grad_erase,                                 # erase
            grad_write,                                 # write
            None,                                       # token_mask
            grad_gamma_log,                             # gamma_log
            grad_initial_d if ctx.has_initial_d else None,  # initial_d
            None,                                       # eta
            None,                                       # use_denominator
            None,                                       # d_min
            None,                                       # d_max
            None,                                       # output_final_state
        )

def triton_affine_scan_osgm_decay(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    gamma_log: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    initial_d: torch.Tensor | None = None,
    eta: float = 0.1,
    use_denominator: bool = True,
    d_min: float | None = None,
    d_max: float | None = None,
    output_final_state: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    if gamma_log is None:
        retention = 0.999
        gamma_log = torch.full(
            (1,),
            math.log(
                retention/(1-retention)
            ),
            device=k.device,
            dtype=torch.float32,
        )

    if token_mask is not None:
        expected_shape = (q.size(0), q.size(1))
        if token_mask.shape != expected_shape:
            raise RuntimeError(
                f"token_mask must have shape {expected_shape}, "
                f"got {tuple(token_mask.shape)}"
            )

    if initial_d is not None:
        valid_shapes = {
            (k.size(0), k.size(2)),
            (k.size(0), 1, k.size(2)),
        }

        if tuple(initial_d.shape) not in valid_shapes:
            raise RuntimeError(
                "initial_d must have shape "
                f"{(k.size(0), k.size(2))} or "
                f"{(k.size(0), 1, k.size(2))}, "
                f"got {tuple(initial_d.shape)}"
            )

    support = triton_scan_support(
        state,
        q,
        k,
        k,
        v,
        keep,
        erase,
        write,
        gamma_log
    )
    if not support.supported:
        raise RuntimeError(
            f"Triton affine scan OSGM is unavailable: "
            f"{support.reason}"
        )

    return _TritonAffineScanOSGMFn.apply(
        state,
        q,
        k,
        v,
        keep,
        erase,
        write,
        token_mask,
        gamma_log,
        initial_d,
        eta,
        use_denominator,
        d_min,
        d_max,
        output_final_state,
    )
