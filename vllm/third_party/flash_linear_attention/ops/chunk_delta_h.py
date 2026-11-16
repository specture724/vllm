# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices, prepare_chunk_offsets
from .op import exp, exp2
from .utils import FLA_CHUNK_SIZE, use_cuda_graph

NUM_WARPS = [2, 4, 8, 16]
# Triton's AMD backend fails to lower this kernel with num_stages=4.
_CHUNK_DELTA_H_NUM_STAGES = [2, 3] if torch.version.hip else [2, 3, 4]


class PagedState(NamedTuple):
    """In-place state I/O for one batch; tensors retain their original storage.

    Pool rows are `[H, V, K]`, contiguous within each row. Indices and the
    initial-state mask have shape `[N]`; false starts from zero. Rows <= 0 are
    skipped. Valid final rows and checkpoint rows must be unique and disjoint.
    Checkpoint offsets use the existing `checkpoint_offsets` argument.
    """

    pool: torch.Tensor
    indices: torch.Tensor
    has_initial_state: torch.Tensor
    checkpoint_indices: torch.Tensor | None = None


# Autotune replays must see the original rows before each in-place update.
_tuning_state_rows: list[torch.Tensor] = []


def _save_state_rows(nargs, reset_only=False):
    if not reset_only and nargs["state_indices"] is not None:
        _tuning_state_rows.append(nargs["ht"][nargs["state_indices"].clamp(min=0)])


def _restore_state_rows(nargs, exception=None):
    if nargs["state_indices"] is not None:
        nargs["ht"][nargs["state_indices"].clamp(min=0)] = _tuning_state_rows.pop()


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "STORE_CHECKPOINT": lambda args: args["checkpoint_offsets"] is not None,
        "USE_STATE_CACHE": lambda args: args["state_indices"] is not None,
        "INDEX_CHECKPOINT": lambda args: args["checkpoint_indices"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in _CHUNK_DELTA_H_NUM_STAGES
        for BV in [32, 64]
    ],
    key=["H", "K", "V", "BT"],
    pre_hook=_save_state_rows,
    post_hook=_restore_state_rows,
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(
    do_not_specialize=[
        "T", "stride_state", "stride_indices", "stride_initial",
        "stride_checkpoint", "stride_checkpoint_indices",
    ],
    do_not_specialize_on_alignment=[
        "cu_seqlens", "chunk_offsets", "checkpoint_offsets", "state_indices",
        "has_initial_state", "checkpoint_indices",
    ],
)
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    checkpoint_state,
    checkpoint_offsets,
    state_indices,
    has_initial_state,
    checkpoint_indices,
    stride_state,
    stride_indices,
    stride_initial,
    stride_checkpoint,
    stride_checkpoint_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STORE_CHECKPOINT: tl.constexpr,
    USE_STATE_CACHE: tl.constexpr,
    INDEX_CHECKPOINT: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BV, BK]
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # calculate offset
    # Cast before multiplying: the products overflow int32 for long batches.
    h += (boh * H + i_h).to(tl.int64) * V * K
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * Hg + i_h // (H // Hg)).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K
    load_initial = True
    store_final = True
    if USE_STATE_CACHE:
        i_s = tl.load(
            state_indices + i_n.to(tl.int64) * stride_indices
        ).to(tl.int64)
        store_final = i_s > 0
        load_initial = store_final & tl.load(
            has_initial_state + i_n.to(tl.int64) * stride_initial
        )
        h0 += i_s * stride_state + i_h * V * K
        ht += i_s * stride_state + i_h * V * K
    else:
        if USE_INITIAL_STATE:
            h0 = h0 + i_nh.to(tl.int64) * V * K
        if STORE_FINAL_STATE:
            ht = ht + i_nh.to(tl.int64) * V * K
    if STORE_CHECKPOINT:
        # The state after o_c tokens is the one entering chunk o_c / BT. Only
        # offsets on a chunk boundary strictly inside the sequence are stored.
        o_c = tl.load(checkpoint_offsets + i_n).to(tl.int32)
        i_c = tl.where((o_c > 0) & (o_c % BT == 0) & (o_c < T), o_c // BT, -1)
        if INDEX_CHECKPOINT:
            i_dst = tl.load(
                checkpoint_indices + i_n.to(tl.int64) * stride_checkpoint_indices
            ).to(tl.int64)
            i_c = tl.where(i_dst > 0, i_c, -1)
        else:
            i_dst = i_n.to(tl.int64)
        checkpoint_state += i_dst * stride_checkpoint + i_h * V * K

    # load initial state
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    if USE_INITIAL_STATE:
        p_h0_1 = h0 + o_v[:, None] * K + o_k1[None, :]
        m_h0_1 = m_v[:, None] & m_k1[None, :] & load_initial
        b_h1 += tl.load(p_h0_1, mask=m_h0_1, other=0.0).to(tl.float32)
        if K > 64:
            p_h0_2 = h0 + o_v[:, None] * K + o_k2[None, :]
            m_h0_2 = m_v[:, None] & m_k2[None, :] & load_initial
            b_h2 += tl.load(p_h0_2, mask=m_h0_2, other=0.0).to(tl.float32)
        if K > 128:
            p_h0_3 = h0 + o_v[:, None] * K + o_k3[None, :]
            m_h0_3 = m_v[:, None] & m_k3[None, :] & load_initial
            b_h3 += tl.load(p_h0_3, mask=m_h0_3, other=0.0).to(tl.float32)
        if K > 192:
            p_h0_4 = h0 + o_v[:, None] * K + o_k4[None, :]
            m_h0_4 = m_v[:, None] & m_k4[None, :] & load_initial
            b_h4 += tl.load(p_h0_4, mask=m_h0_4, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        o_t = (i_t * BT + tl.arange(0, BT)).to(tl.int64)
        m_t = o_t < T
        p_h1 = h + i_t.to(tl.int64) * stride_h + o_v[:, None] * K + o_k1[None, :]
        m_h1 = m_v[:, None] & m_k1[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_h2 = h + i_t.to(tl.int64) * stride_h + o_v[:, None] * K + o_k2[None, :]
            m_h2 = m_v[:, None] & m_k2[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_h3 = h + i_t.to(tl.int64) * stride_h + o_v[:, None] * K + o_k3[None, :]
            m_h3 = m_v[:, None] & m_k3[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_h4 = h + i_t.to(tl.int64) * stride_h + o_v[:, None] * K + o_k4[None, :]
            m_h4 = m_v[:, None] & m_k4[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)
        if STORE_CHECKPOINT:
            if i_t == i_c:
                # From the fp32 accumulator, unlike h.
                p_c = checkpoint_state + o_v[:, None] * K + o_k1[None, :]
                tl.store(p_c, b_h1.to(p_c.dtype.element_ty), mask=m_h1)
                if K > 64:
                    p_c = checkpoint_state + o_v[:, None] * K + o_k2[None, :]
                    tl.store(p_c, b_h2.to(p_c.dtype.element_ty), mask=m_h2)
                if K > 128:
                    p_c = checkpoint_state + o_v[:, None] * K + o_k3[None, :]
                    tl.store(p_c, b_h3.to(p_c.dtype.element_ty), mask=m_h3)
                if K > 192:
                    p_c = checkpoint_state + o_v[:, None] * K + o_k4[None, :]
                    tl.store(p_c, b_h4.to(p_c.dtype.element_ty), mask=m_h4)

        p_w = w + o_t[:, None] * stride_w + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * stride_w + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = w + o_t[:, None] * stride_w + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = w + o_t[:, None] * stride_w + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = v + o_t[:, None] * stride_v + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v

        if SAVE_NEW_VALUE:
            p_v = v_new + o_t[:, None] * stride_v + o_v[None, :]
            tl.store(
                p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :]
            )

        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos * H + i_h + o_t * H
            b_g = tl.load(p_g, mask=m_t, other=0.0)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            if USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = k + o_k1[:, None] + o_t[None, :] * stride_k
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            b_h4 += tl.trans(tl.dot(b_k, b_v))
    # epilogue
    if STORE_FINAL_STATE:
        p_ht = ht + o_v[:, None] * K + o_k1[None, :]
        m_ht = m_v[:, None] & m_k1[None, :] & store_final
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 64:
            p_ht = ht + o_v[:, None] * K + o_k2[None, :]
            m_ht = m_v[:, None] & m_k2[None, :] & store_final
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 128:
            p_ht = ht + o_v[:, None] * K + o_k3[None, :]
            m_ht = m_v[:, None] & m_k3[None, :] & store_final
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 192:
            p_ht = ht + o_v[:, None] * K + o_k4[None, :]
            m_ht = m_v[:, None] & m_k4[None, :] & store_final
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_ht)


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = FLA_CHUNK_SIZE,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_exp2: bool = False,
    checkpoint_state: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
    state_io: PagedState | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run the chunk-state recurrence.

    Args:
        checkpoint_state: Contiguous `[N, H, V, K]` buffer; row `n` receives
            sequence `n`'s state after `checkpoint_offsets[n]` tokens.
        checkpoint_offsets: `[N]` token offset into each sequence; `0` for
            none. Only multiples of `chunk_size` below the sequence length are
            stored, and other rows are left untouched.
        state_io: Optional in-place pool and row mapping. Replaces dense
            initial/final states; the returned final state is then None.

    Returns:
        The per-chunk states, the new values if `save_new_value`, and the final
        state if `output_final_state`.
    """
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, V, K)
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )
    h0, ht = initial_state, final_state
    indices = has_initial = checkpoint_indices = None
    if state_io is not None:
        pool, indices, has_initial, checkpoint_indices = state_io
        if (
            initial_state is not None
            or output_final_state
            or checkpoint_state is not None
        ):
            raise ValueError("state_io replaces dense initial/final/checkpoint buffers")
        if pool.shape[1:] != (H, V, K) or pool.stride()[1:] != (V * K, K, 1):
            raise ValueError("state_io.pool must have contiguous [H, V, K] rows")
        for name, tensor in (("indices", indices), ("has_initial_state", has_initial)):
            if tensor.shape != (N,) or tensor.device != k.device:
                raise ValueError(f"state_io.{name} must be [N] on the input device")
        if pool.device != k.device or not pool.is_floating_point():
            raise ValueError("state_io.pool must be floating point on the input device")
        if (
            indices.dtype not in (torch.int32, torch.int64)
            or has_initial.dtype != torch.bool
        ):
            raise ValueError("state_io needs integer indices and a bool initial-state mask")
        if (checkpoint_offsets is None) != (checkpoint_indices is None):
            raise ValueError("checkpoint_offsets and state_io.checkpoint_indices go together")
        if checkpoint_indices is not None:
            assert checkpoint_offsets is not None
            if checkpoint_indices.shape != (N,) or checkpoint_offsets.shape != (N,):
                raise ValueError("checkpoint indices and offsets must have shape [N]")
            if (
                checkpoint_indices.device != k.device
                or checkpoint_offsets.device != k.device
                or checkpoint_indices.dtype not in (torch.int32, torch.int64)
                or checkpoint_offsets.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("checkpoint metadata must be integers on the input device")
            checkpoint_state = pool
        h0 = ht = pool

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        checkpoint_state=checkpoint_state,
        checkpoint_offsets=checkpoint_offsets,
        state_indices=indices,
        has_initial_state=has_initial,
        checkpoint_indices=checkpoint_indices,
        stride_state=h0.stride(0) if state_io is not None else 0,
        stride_indices=indices.stride(0) if indices is not None else 0,
        stride_initial=has_initial.stride(0) if has_initial is not None else 0,
        stride_checkpoint=checkpoint_state.stride(0) if checkpoint_state is not None else 0,
        stride_checkpoint_indices=checkpoint_indices.stride(0) if checkpoint_indices is not None else 0,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
    )
    return h, v_new, final_state
