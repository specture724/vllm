# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dual batch overlap (DBO) in the V2 model runner.

The V2 runner slices the batch *before* building attention metadata, where V1
builds metadata for the whole batch and slices it afterwards. Both must describe
the same microbatches, so the main test here pins V2's slicing against V1's
`split_attn_metadata`. The rest covers the decision to microbatch and the
threaded execution of the microbatches.
"""

import threading
from dataclasses import replace

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config import ModelConfig, ParallelConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import create_forward_context
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import _maybe_ubatch_descriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.ubatch_utils import (
    UBatchRunner,
    create_even_ubatch_slices,
    slice_input_batch,
    slice_model_inputs,
)
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    maybe_create_ubatch_slices,
    split_attn_metadata,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id, dbo_yield

MAX_NUM_REQS = 32
MAX_NUM_TOKENS = 128


def _make_buffers() -> InputBuffers:
    return InputBuffers(
        max_num_reqs=MAX_NUM_REQS,
        max_num_tokens=MAX_NUM_TOKENS,
        device=torch.device("cpu"),
        num_ubatches=2,
    )


def _make_input_batch(
    query_lens: list[int],
    seq_lens: list[int],
    buffers: InputBuffers,
    num_reqs_padded: int | None = None,
    num_tokens_padded: int | None = None,
) -> InputBatch:
    """Build an InputBatch shaped like the one `prepare_inputs` produces."""
    num_reqs = len(query_lens)
    num_tokens = int(sum(query_lens))
    num_reqs_padded = num_reqs_padded or num_reqs
    num_tokens_padded = num_tokens_padded or num_tokens

    # make_dummy writes the shared buffers, so fill them afterwards.
    base = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    query_start_loc_np = np.zeros(num_reqs_padded + 1, dtype=np.int32)
    np.cumsum(query_lens, out=query_start_loc_np[1 : num_reqs + 1])
    # Padded entries repeat the token count, as `prepare_inputs` does.
    query_start_loc_np[num_reqs + 1 :] = num_tokens
    buffers.query_start_loc[: num_reqs_padded + 1] = torch.from_numpy(
        query_start_loc_np
    )

    buffers.seq_lens[:num_reqs] = torch.tensor(seq_lens, dtype=torch.int32)
    buffers.seq_lens[num_reqs:num_reqs_padded] = 0

    seq_lens_upper_bound = np.zeros(num_reqs_padded, dtype=np.int32)
    seq_lens_upper_bound[:num_reqs] = seq_lens

    return replace(
        base,
        req_ids=[f"req_{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs_padded,
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32),
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
        num_scheduled_tokens=np.array(query_lens, dtype=np.int32),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens_padded,
        query_start_loc=buffers.query_start_loc[: num_reqs_padded + 1],
        query_start_loc_np=query_start_loc_np,
        seq_lens=buffers.seq_lens[:num_reqs_padded],
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_upper_bound),
        num_computed_tokens_np=np.array(seq_lens, dtype=np.int32)
        - np.array(query_lens, dtype=np.int32),
        prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
        input_ids=buffers.input_ids[:num_tokens_padded],
        positions=buffers.positions[:num_tokens_padded],
        is_padding=buffers.is_padding[:num_tokens_padded],
    )


# (query_lens, seq_lens): uniform decode, a boundary that lands inside a
# request, and a boundary that lands exactly on a request start.
BATCHES = {
    "uniform_decode": ([1] * 8, [128 + 4 * i for i in range(8)]),
    "split_request": ([1, 1, 10, 1, 1], [64, 96, 512, 32, 48]),
    "aligned_prefills": ([6, 2, 4, 4], [6, 130, 260, 4]),
    "split_first_and_last": ([3, 9, 3], [70, 300, 41]),
}


@pytest.mark.parametrize("batch_name", list(BATCHES))
def test_slicing_matches_v1_split_attn_metadata(batch_name: str):
    """V2's pre-sliced inputs describe the same microbatches as V1's."""
    query_lens, seq_lens = BATCHES[batch_name]
    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
    num_tokens = int(num_scheduled_tokens.sum())

    ubatch_slices, _ = maybe_create_ubatch_slices(
        True, num_scheduled_tokens, num_tokens, len(query_lens), num_ubatches=2
    )
    assert ubatch_slices is not None

    v1_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=16,
        device=torch.device("cpu"),
    )
    v1_ubatches = split_attn_metadata(ubatch_slices, v1_metadata)

    buffers = _make_buffers()
    input_batch = _make_input_batch(query_lens, seq_lens, buffers)

    for i, (ubatch_slice, v1_ubatch) in enumerate(zip(ubatch_slices, v1_ubatches)):
        v2_ubatch = slice_input_batch(input_batch, ubatch_slice, i, buffers)

        assert v2_ubatch.num_reqs == v1_ubatch.num_reqs
        assert v2_ubatch.num_tokens == v1_ubatch.num_actual_tokens
        assert int(v2_ubatch.num_scheduled_tokens.max()) == v1_ubatch.max_query_len
        torch.testing.assert_close(v2_ubatch.query_start_loc, v1_ubatch.query_start_loc)
        torch.testing.assert_close(v2_ubatch.seq_lens, v1_ubatch.seq_lens)
        torch.testing.assert_close(
            torch.from_numpy(v2_ubatch.query_start_loc_np),
            v1_ubatch.query_start_loc_cpu,
        )
        # `prepare_attn` derives max_seq_len from the upper bound. Compare the
        # bound itself rather than V1's max_seq_len: V1 floors it with the full
        # batch's max_seq_len to keep the value CUDA-graph capture installed,
        # which V2 gets from `prepare_attn(for_capture=True)` instead.
        torch.testing.assert_close(
            v2_ubatch.seq_lens_cpu_upper_bound, v1_ubatch.seq_lens_cpu_upper_bound
        )


def test_microbatches_do_not_share_buffers():
    """Both microbatches are live at once, so their buffers must be distinct."""
    query_lens, seq_lens = BATCHES["split_request"]
    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
    num_tokens = int(num_scheduled_tokens.sum())

    ubatch_slices, _ = maybe_create_ubatch_slices(
        True, num_scheduled_tokens, num_tokens, len(query_lens), num_ubatches=2
    )
    assert ubatch_slices is not None

    buffers = _make_buffers()
    input_batch = _make_input_batch(query_lens, seq_lens, buffers)

    first = slice_input_batch(input_batch, ubatch_slices[0], 0, buffers)
    first_query_start_loc = first.query_start_loc.clone()
    first_seq_lens = first.seq_lens.clone()

    # Slicing the second microbatch must not disturb the first.
    slice_input_batch(input_batch, ubatch_slices[1], 1, buffers)

    torch.testing.assert_close(first.query_start_loc, first_query_start_loc)
    torch.testing.assert_close(first.seq_lens, first_seq_lens)


def test_trailing_microbatch_absorbs_cudagraph_padding():
    """The padded microbatch keeps padded rows empty and query_start_loc flat."""
    query_lens, seq_lens = [1] * 6, [64] * 6
    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
    num_tokens_padded = 8

    _, ubatch_slices_padded = maybe_create_ubatch_slices(
        True,
        num_scheduled_tokens,
        num_tokens_padded,
        num_reqs_padded=num_tokens_padded,
        num_ubatches=2,
    )
    assert ubatch_slices_padded is not None

    buffers = _make_buffers()
    input_batch = _make_input_batch(
        query_lens,
        seq_lens,
        buffers,
        num_reqs_padded=num_tokens_padded,
        num_tokens_padded=num_tokens_padded,
    )

    last = slice_input_batch(input_batch, ubatch_slices_padded[-1], 1, buffers)

    # Two real decodes plus two padded rows.
    assert last.num_reqs == 2
    assert last.num_tokens == 2
    assert last.num_reqs_after_padding == 4
    assert last.num_tokens_after_padding == 4
    torch.testing.assert_close(
        last.query_start_loc, torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        last.seq_lens, torch.tensor([64, 64, 0, 0], dtype=torch.int32)
    )


def _ubatch_descriptor(
    num_tokens_per_rank: list[int],
    wants_ubatch: list[int],
    num_ubatches: int = 2,
    cudagraph_manager=None,
    uniform_token_count: int | None = None,
):
    # `_maybe_ubatch_descriptor` is the decision the DP all-reduce feeds; it is
    # tested directly so the cases below don't need a process group.
    return _maybe_ubatch_descriptor(
        torch.tensor(num_tokens_per_rank, dtype=torch.int32),
        torch.tensor(wants_ubatch, dtype=torch.int32),
        num_reqs=8,
        num_ubatches=num_ubatches,
        cudagraph_manager=cudagraph_manager,
        uniform_token_count=uniform_token_count,
    )


def test_every_dp_rank_must_agree_to_microbatch():
    """One rank declining is enough to keep the whole group on one batch."""
    assert _ubatch_descriptor([256, 256], [1, 0]) is None
    assert _ubatch_descriptor([256, 256], [0, 0]) is None

    desc = _ubatch_descriptor([256, 256], [1, 1])
    assert desc is not None
    assert desc.num_ubatches == 2


def test_microbatching_pads_all_ranks_to_the_largest():
    """Ranks must run the same token count so microbatch sizes line up."""
    desc = _ubatch_descriptor([200, 256], [1, 1])
    assert desc is not None
    assert desc.num_tokens == 256


def test_microbatching_skipped_when_a_rank_cannot_fill_it():
    """A rank whose real tokens all land in the first microbatch aborts it."""
    # Rank 0 has 100 real tokens but the split point is at 256 // 2 = 128, so
    # its second microbatch would be pure padding.
    assert _ubatch_descriptor([100, 256], [1, 1]) is None


def test_microbatching_off_when_not_configured():
    assert _ubatch_descriptor([256, 256], [1, 1], num_ubatches=1) is None


def _cudagraph_manager(
    descs: list[BatchExecutionDescriptor], captured: bool = True
) -> CudaGraphManager:
    """A manager holding `descs` as its captured graphs.

    Built field by field: a real one needs a process group and a CUDA graph
    pool, and none of the dispatch path under test touches either.
    """
    manager = CudaGraphManager.__new__(CudaGraphManager)
    manager._graphs_captured = captured
    manager._lora_dispatch_map, manager._max_lora_case = {}, 0
    manager._candidates = {}
    for desc in descs:
        for num_tokens in range(desc.num_tokens + 1):
            manager._candidates.setdefault((num_tokens, 0), []).append(desc)
    return manager


def _decode_desc(num_tokens: int, num_ubatches: int = 1) -> BatchExecutionDescriptor:
    return BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=num_tokens,
        num_reqs=num_tokens,
        uniform_token_count=1,
        num_ubatches=num_ubatches,
    )


def test_microbatched_step_replays_its_own_cudagraph():
    """A microbatched step dispatches to a graph captured for microbatching."""
    manager = _cudagraph_manager([_decode_desc(256), _decode_desc(256, num_ubatches=2)])

    desc = _ubatch_descriptor(
        [200, 250], [1, 1], cudagraph_manager=manager, uniform_token_count=1
    )
    assert desc is not None
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_ubatches == 2
    # Padded up to the capture size, past the largest rank's 250 tokens.
    assert desc.num_tokens == 256

    # A non-microbatched batch of the same shape must not land on that graph.
    plain = manager.dispatch(250, 250, 1, num_active_loras=0)
    assert plain.num_ubatches == 1
    assert plain.cg_mode == CUDAGraphMode.FULL


def test_microbatched_step_runs_eager_without_a_matching_graph():
    """Mixed batches have no microbatched graph, so they fall back to eager."""
    manager = _cudagraph_manager([_decode_desc(256, num_ubatches=2)])

    # uniform_token_count=None marks a mixed prefill/decode batch.
    desc = _ubatch_descriptor(
        [200, 250], [1, 1], cudagraph_manager=manager, uniform_token_count=None
    )
    assert desc is not None
    assert desc.cg_mode == CUDAGraphMode.NONE
    assert desc.num_ubatches == 2
    assert desc.num_tokens == 250


def test_microbatching_skipped_when_cudagraph_padding_empties_it():
    """Padding up to a capture size can empty the trailing microbatch."""
    manager = _cudagraph_manager(
        [_decode_desc(128, num_ubatches=2), _decode_desc(256, num_ubatches=2)]
    )

    # The batch is padded to 256 for the busiest rank, splitting at 128. The
    # quiet rank's 120 real tokens all land in the first microbatch, so the
    # group gives up on microbatching rather than run an empty one.
    assert (
        _ubatch_descriptor(
            [120, 250], [1, 1], cudagraph_manager=manager, uniform_token_count=1
        )
        is None
    )
    # The same ranks do microbatch when they fit a smaller capture size.
    desc = _ubatch_descriptor(
        [120, 128], [1, 1], cudagraph_manager=manager, uniform_token_count=1
    )
    assert desc is not None and desc.num_tokens == 128


def _can_capture_ubatched(
    desc: BatchExecutionDescriptor,
    dp_size: int = 2,
    num_ubatches: int = 2,
    decode_threshold: int = 32,
) -> bool:
    manager = CudaGraphManager.__new__(CudaGraphManager)
    manager.dp_size = dp_size
    manager.num_ubatches = num_ubatches
    manager.parallel_config = ParallelConfig(
        enable_dbo=True,
        all2all_backend="deepep_low_latency",
        dbo_decode_token_threshold=decode_threshold,
    )
    return manager._can_capture_ubatched(desc)


def test_microbatched_graphs_captured_for_splittable_decode_shapes():
    assert _can_capture_ubatched(_decode_desc(64))
    # Below the DBO threshold no step would ask to microbatch.
    assert not _can_capture_ubatched(_decode_desc(16))
    # Without a DP group there is nobody to agree to microbatch with.
    assert not _can_capture_ubatched(_decode_desc(64), dp_size=1)
    # Piecewise graphs cannot be microbatched.
    assert not _can_capture_ubatched(
        replace(_decode_desc(64), cg_mode=CUDAGraphMode.PIECEWISE)
    )
    # An odd shape has no even split.
    assert not _can_capture_ubatched(
        replace(_decode_desc(64), num_tokens=63, num_reqs=63)
    )
    assert not _can_capture_ubatched(replace(_decode_desc(64), num_reqs=33))


def test_even_slices_match_the_content_split_for_uniform_decode():
    """The fixed split a graph replays is the split a padded decode gets anyway.

    Microbatched graphs pin the request boundary to the padded shape so it
    survives replay. For the uniform decode batches they are captured for, that
    has to describe the same microbatches as slicing by batch content.
    """
    num_reqs_padded, num_tokens_padded = 16, 16
    # 12 real decodes padded out to the capture size; both microbatches still
    # hold real tokens, which is what the DP handshake guarantees.
    num_scheduled_tokens = np.ones(12, dtype=np.int32)

    _, content_slices = maybe_create_ubatch_slices(
        True,
        num_scheduled_tokens,
        num_tokens_padded,
        num_reqs_padded,
        num_ubatches=2,
    )
    even_slices = create_even_ubatch_slices(num_tokens_padded, num_reqs_padded, 2)
    assert even_slices == content_slices


def test_even_split_keeps_padded_requests_empty():
    """Padded rows in a microbatch decode nothing, whatever the split."""
    query_lens, seq_lens = [1] * 5, [64, 72, 80, 88, 96]
    buffers = _make_buffers()
    input_batch = _make_input_batch(
        query_lens, seq_lens, buffers, num_reqs_padded=8, num_tokens_padded=8
    )

    slices = create_even_ubatch_slices(8, 8, 2)
    last = slice_input_batch(input_batch, slices[-1], 1, buffers)

    # One real decode (the 5th request) followed by three padded rows.
    assert last.num_reqs == 1
    assert last.num_tokens == 1
    torch.testing.assert_close(
        last.query_start_loc, torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        last.seq_lens, torch.tensor([96, 0, 0, 0], dtype=torch.int32)
    )


def test_slice_model_inputs_handles_mrope_positions():
    model_inputs = {
        "input_ids": torch.arange(8),
        # M-RoPE positions carry a leading section dim.
        "positions": torch.arange(24).view(3, 8),
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }
    sliced = slice_model_inputs(model_inputs, slice(4, 8))

    torch.testing.assert_close(sliced["input_ids"], torch.arange(4, 8))
    torch.testing.assert_close(sliced["positions"], torch.arange(24).view(3, 8)[:, 4:8])
    assert sliced["inputs_embeds"] is None
    assert sliced["intermediate_tensors"] is None


class _YieldingModel(torch.nn.Module):
    """Toy model that hands off to the other microbatch mid-forward."""

    def __init__(self, trace: list[tuple[str, int]]):
        super().__init__()
        self.trace = trace

    def forward(self, input_ids, positions, intermediate_tensors=None, **kwargs):
        self.trace.append(("enter", dbo_current_ubatch_id()))
        out = input_ids.float().unsqueeze(-1) * 2 + positions.float().unsqueeze(-1)
        # Stands in for the expert all-to-all handoff point.
        dbo_yield()
        self.trace.append(("exit", dbo_current_ubatch_id()))
        return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DBO needs a GPU")
def test_ubatch_runner_overlaps_and_matches_single_batch():
    """Microbatches interleave at the yield point and produce the same output."""
    vllm_config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", dtype="float16", seed=0),
        parallel_config=ParallelConfig(
            enable_dbo=True, all2all_backend="deepep_low_latency"
        ),
    )
    device = torch.device("cuda:0")
    runner = UBatchRunner(vllm_config, device)

    num_tokens = 16
    model_inputs = {
        "input_ids": torch.arange(num_tokens, device=device),
        "positions": torch.arange(num_tokens, device=device),
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }
    ubatch_slices = [
        UBatchSlice(slice(0, 4), slice(0, 8)),
        UBatchSlice(slice(4, 8), slice(8, 16)),
    ]
    forward_contexts = [create_forward_context(None, vllm_config) for _ in range(2)]

    trace: list[tuple[str, int]] = []
    model = _YieldingModel(trace)
    output = runner.run(model, model_inputs, forward_contexts, ubatch_slices)

    expected = model_inputs["input_ids"].float().unsqueeze(-1) * 2 + model_inputs[
        "positions"
    ].float().unsqueeze(-1)
    torch.testing.assert_close(output, expected)

    # Both microbatches reach the handoff before either finishes.
    assert trace == [("enter", 0), ("enter", 1), ("exit", 0), ("exit", 1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DBO needs a GPU")
def test_microbatched_forward_replays_as_a_cudagraph():
    """A microbatched forward can be captured and replayed on new inputs.

    This is what `ModelCudaGraphManager` does for a microbatched descriptor:
    the threads are started outside the capture, the handoff between them is
    recorded as graph edges, and the replay reads whatever the persistent input
    buffers hold.
    """
    vllm_config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", dtype="float16", seed=0),
        parallel_config=ParallelConfig(
            enable_dbo=True, all2all_backend="deepep_low_latency"
        ),
    )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    runner = UBatchRunner(vllm_config, device)

    num_tokens = 16
    weight = torch.randn(8, 8, device=device)

    class _MatmulModel(torch.nn.Module):
        """Yields mid-forward and runs a cuBLAS op in each microbatch."""

        def forward(self, input_ids, positions, **kwargs):
            x = input_ids.float().unsqueeze(-1) * torch.ones(1, 8, device=device)
            x = x @ weight
            dbo_yield()
            return x + positions.float().unsqueeze(-1)

    # Persistent buffers, as the model runner keeps them.
    input_ids = torch.arange(num_tokens, device=device)
    positions = torch.arange(num_tokens, device=device)
    model_inputs = {
        "input_ids": input_ids,
        "positions": positions,
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }
    ubatch_slices = create_even_ubatch_slices(num_tokens, num_tokens, 2)
    model = _MatmulModel()

    def expected() -> torch.Tensor:
        x = input_ids.float().unsqueeze(-1) * torch.ones(1, 8, device=device)
        return x @ weight + positions.float().unsqueeze(-1)

    def launch():
        return runner.launch(
            model,
            model_inputs,
            [create_forward_context(None, vllm_config) for _ in range(2)],
            ubatch_slices,
            compute_stream=capture_stream,
            for_capture=True,
        )

    capture_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(capture_stream):
        # Warmup pass, then capture, as the cudagraph manager does.
        with launch() as finish:
            torch.testing.assert_close(finish(), expected())
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with launch() as finish, torch.cuda.graph(graph, None, capture_stream):
            captured_output = finish()
    torch.cuda.synchronize()

    input_ids.copy_(torch.randint(0, 100, (num_tokens,), device=device))
    positions.copy_(torch.randint(0, 100, (num_tokens,), device=device))
    want = expected()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured_output, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DBO needs a GPU")
def test_ubatch_runner_names_the_microbatch_that_failed():
    """A failing microbatch surfaces as a named error, not a downstream KeyError.

    The model here never yields, so the microbatches run back to back and no
    handoff is left outstanding when one of them dies. A microbatch that fails
    *while its sibling is parked at a yield* hangs the step instead -- the
    shared handoff protocol in `ubatching.py` has no way to unwind a parked
    microbatch, and the V1 runner has the same gap. Fixing that means changing
    `ubatching.py`, which is out of scope for the V2 runner.
    """
    vllm_config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", dtype="float16", seed=0),
        parallel_config=ParallelConfig(
            enable_dbo=True, all2all_backend="deepep_low_latency"
        ),
    )
    device = torch.device("cuda:0")
    runner = UBatchRunner(vllm_config, device)

    class _FailingModel(torch.nn.Module):
        def forward(self, input_ids, positions, **kwargs):
            if dbo_current_ubatch_id() == 0:
                raise ValueError("boom")
            return input_ids.float().unsqueeze(-1)

    model_inputs = {
        "input_ids": torch.arange(8, device=device),
        "positions": torch.arange(8, device=device),
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }
    ubatch_slices = [
        UBatchSlice(slice(0, 2), slice(0, 4)),
        UBatchSlice(slice(2, 4), slice(4, 8)),
    ]
    forward_contexts = [create_forward_context(None, vllm_config) for _ in range(2)]

    result: dict[str, BaseException] = {}

    def _call():
        try:
            runner.run(_FailingModel(), model_inputs, forward_contexts, ubatch_slices)
        except BaseException as e:  # noqa: BLE001
            result["error"] = e

    # Run behind a watchdog: `UBatchRunner.run` joins without a timeout, so a
    # regression here would wedge the suite rather than fail it.
    caller = threading.Thread(target=_call, daemon=True)
    caller.start()
    caller.join(timeout=60.0)
    assert not caller.is_alive(), "UBatchRunner.run hung on a failing microbatch"

    assert isinstance(result["error"], RuntimeError)
    assert "Microbatch 0" in str(result["error"])
    assert isinstance(result["error"].__cause__, ValueError)
