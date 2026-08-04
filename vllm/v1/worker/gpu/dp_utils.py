# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import init_logger
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.ubatch_utils import is_last_ubatch_empty

logger = init_logger(__name__)


def sync_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager | None,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    num_active_loras: int = 0,
    wants_ubatch: bool = False,
    num_ubatches: int = 1,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """
    Coordinates the batch descriptor and DP padding across all ranks.

    Returns (synced_batch_desc, num_tokens_across_dp).
    """
    assert dp_size > 1, "DP size must be greater than 1"
    group = get_dp_group().cpu_group
    tensor = torch.zeros(4, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = desired_batch_desc.cg_mode.value
    tensor[2][dp_rank] = uniform_token_count or 0  # (0 means None)
    tensor[3][dp_rank] = 1 if wants_ubatch else 0
    dist.all_reduce(tensor, group=group)

    num_tokens_across_dp = tensor[0]
    cg_mode_across_dp = tensor[1]
    uniform_token_counts_across_dp = tensor[2]
    wants_ubatch_across_dp = tensor[3]

    if torch.all(num_tokens_across_dp == 0).item():
        synced_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=0, num_reqs=0
        )
        return synced_desc, None

    synced_cg_mode = CUDAGraphMode(int(cg_mode_across_dp.min().item()))

    ubatch_desc = _maybe_ubatch_descriptor(
        num_tokens_across_dp,
        wants_ubatch_across_dp,
        num_reqs,
        num_ubatches,
        # If any rank has to run eager, no rank may replay a graph.
        cudagraph_manager=(
            cudagraph_manager if synced_cg_mode != CUDAGraphMode.NONE else None
        ),
        uniform_token_count=_synced_uniform_token_count(uniform_token_counts_across_dp),
        num_active_loras=num_active_loras,
    )
    if ubatch_desc is not None:
        # Microbatching needs every rank to run the same number of tokens, so
        # that each rank can assume the others' microbatches are the same size.
        num_tokens_across_dp = torch.full_like(
            num_tokens_across_dp, ubatch_desc.num_tokens
        )
        return ubatch_desc, num_tokens_across_dp

    # If any rank wants to run eager, all ranks run eager
    if synced_cg_mode == CUDAGraphMode.NONE:
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=desired_batch_desc.num_active_loras,
        ), num_tokens_across_dp

    assert cudagraph_manager is not None, (
        "cudagraph_manager should only be None during profile run, "
        "where synced_cg_mode must be NONE across all DP ranks"
    )
    synced_num_tokens = int(num_tokens_across_dp.max().item())
    synced_uniform_token_count = _synced_uniform_token_count(
        uniform_token_counts_across_dp
    )

    # Dispatch for the final synced values, use num_reqs instead of synced_num_reqs
    # so we don't perform request padding for PIECEWISE graphs.
    # num_active_loras is per-rank and doesn't need cross-rank agreement.
    synced_desc = cudagraph_manager.dispatch(
        num_reqs,
        synced_num_tokens,
        synced_uniform_token_count,
        num_active_loras=num_active_loras,
    )

    # Update num_tokens_across_dp to reflect padded size.
    num_tokens_across_dp[:] = synced_desc.num_tokens

    return synced_desc, num_tokens_across_dp


def _synced_uniform_token_count(
    uniform_token_counts_across_dp: torch.Tensor,
) -> int | None:
    """The token count per request, if every rank has the same uniform one."""
    count = uniform_token_counts_across_dp[0]
    # If ranks disagree on the uniform token count, or its 0 (means None) set to None
    if count == 0 or not torch.all(uniform_token_counts_across_dp == count):
        return None
    return int(count)


def _maybe_ubatch_descriptor(
    num_tokens_across_dp: torch.Tensor,
    wants_ubatch_across_dp: torch.Tensor,
    num_reqs: int,
    num_ubatches: int,
    cudagraph_manager: CudaGraphManager | None = None,
    uniform_token_count: int | None = None,
    num_active_loras: int = 0,
) -> BatchExecutionDescriptor | None:
    """Decide whether the group microbatches this step, and at what size.

    Microbatching is all-or-nothing: every rank has to split, because the
    expert all-to-all is collective. Returns the descriptor all ranks will run,
    or None to fall through to the regular (single batch) path.

    Every input this decides on is either synchronized across the ranks or
    (`num_reqs`) only used for padding this rank's own batch, so all ranks
    reach the same conclusion without a second all-reduce.
    """
    if num_ubatches <= 1 or not torch.all(wants_ubatch_across_dp == 1).item():
        return None

    # Every rank runs the largest rank's token count, so pad up to it.
    num_tokens = int(num_tokens_across_dp.max().item())
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        num_ubatches=num_ubatches,
    )
    if cudagraph_manager is not None:
        # A microbatched graph pads the batch further, up to its capture size.
        desc = cudagraph_manager.dispatch(
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras=num_active_loras,
            num_ubatches=num_ubatches,
        )

    if is_last_ubatch_empty(
        int(num_tokens_across_dp.min().item()), desc.num_tokens, num_ubatches
    ):
        # The smallest rank has too few tokens to fill every microbatch. Note
        # this is checked against the padded size: padding up to a capture size
        # can empty out the trailing microbatch on its own.
        logger.debug(
            "Skipping microbatching: %d tokens do not fill %d microbatches of %d",
            int(num_tokens_across_dp.min().item()),
            num_ubatches,
            desc.num_tokens,
        )
        return None
    return desc


def dispatch_cg_and_sync_dp(
    cudagraph_manager: CudaGraphManager | None,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    need_eager: bool = False,
    num_active_loras: int = 0,
    wants_ubatch: bool = False,
    num_ubatches: int = 1,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    if need_eager:
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
        )
    else:
        assert cudagraph_manager is not None, (
            "cudagraph_manager should only be None during profile run, "
            "where need_eager must be True"
        )
        batch_desc = cudagraph_manager.dispatch(
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras=num_active_loras,
        )

    if dp_size == 1:
        # Microbatching needs the DP handshake to agree on it, so it is only
        # available with more than one DP rank (as in the V1 runner).
        return batch_desc, None

    return sync_cudagraph_and_dp_padding(
        cudagraph_manager,
        batch_desc,
        num_tokens,
        num_reqs,
        uniform_token_count,
        dp_size,
        dp_rank,
        num_active_loras=num_active_loras,
        wants_ubatch=wants_ubatch,
        num_ubatches=num_ubatches,
    )
