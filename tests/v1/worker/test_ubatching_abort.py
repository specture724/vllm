# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbatches must unwind when a sibling fails, instead of hanging.

Microbatching passes a baton: each microbatch wakes the next and then parks
until someone wakes it. A thread that dies mid-forward never passes the baton
on, so without an abort path its siblings park forever and the step never
returns -- a hang, with no traceback and no failed request.
"""

import threading
from typing import Any, cast

import pytest
import torch

from vllm.v1.worker.gpu_ubatch_wrapper import _raise_ubatch_error, _run_ubatch_guarded
from vllm.v1.worker.ubatching import (
    _THREAD_ID_TO_CONTEXT,
    UBatchAborted,
    dbo_yield,
    make_ubatch_contexts,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ubatch contexts need CUDA streams"
)

# Long enough that a slow machine does not flake, short enough that a
# regression fails the suite instead of wedging it.
JOIN_TIMEOUT_S = 30.0


class _Driver:
    """Mirrors UBatchWrapper._run_ubatches' thread structure."""

    def __init__(self, bodies):
        self.bodies = bodies
        self.errors: dict[int, BaseException] = {}
        self.trace: list[tuple[str, int]] = []
        self.barrier_broke = False
        num = len(bodies)
        self.barrier = threading.Barrier(num + 1)
        self.contexts = make_ubatch_contexts(
            num_micro_batches=num,
            compute_stream=torch.cuda.Stream(),
            comm_stream=torch.cuda.Stream(),
            # Only ever compared and reassigned, never dereferenced.
            forward_contexts=[cast(Any, object()) for _ in range(num)],
            ready_barrier=self.barrier,
        )

    def run(self) -> list[str]:
        """Returns the names of any threads still alive (i.e. hung)."""
        threads = []
        for ctx, body in zip(self.contexts, self.bodies):
            thread = threading.Thread(
                target=_run_ubatch_guarded,
                args=(
                    ctx,
                    self.contexts,
                    self.errors,
                    self.barrier,
                    lambda ctx=ctx, body=body: body(ctx, self.trace),
                ),
                # Daemon so a regression fails the assertion below rather than
                # leaving a parked thread that stops pytest from exiting.
                daemon=True,
            )
            threads.append(thread)
            thread.start()

        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            self.barrier_broke = True
        else:
            self.contexts[0].cpu_wait_event.set()

        for thread in threads:
            thread.join(timeout=JOIN_TIMEOUT_S)
        return [t.name for t in threads if t.is_alive()]


def _yielding_body(num_yields: int, fail_after: int | None = None):
    def body(ctx, trace):
        with ctx:
            trace.append(("enter", ctx.id))
            for i in range(num_yields):
                if fail_after is not None and i == fail_after:
                    raise ValueError(f"microbatch {ctx.id} blew up")
                dbo_yield()
            trace.append(("done", ctx.id))

    return body


@requires_cuda
def test_failing_microbatch_does_not_hang_its_sibling():
    """The regression: microbatch 0 dies after a handoff, 1 must not park.

    Note the failure lands on 1's *second* yield, not its first -- unwinding
    microbatch 0 runs `__exit__`, which passes the baton one last time. That
    courtesy handoff is why the hang used to surface far from its cause.
    """
    driver = _Driver(
        [
            _yielding_body(4, fail_after=1),
            _yielding_body(4),
        ]
    )
    hung = driver.run()

    assert hung == [], f"microbatch thread(s) hung: {hung}"
    assert isinstance(driver.errors[0], ValueError)
    assert isinstance(driver.errors[1], UBatchAborted)
    # The sibling unwound rather than running to completion.
    assert ("done", 1) not in driver.trace


@requires_cuda
def test_root_cause_is_reported_not_the_abort():
    driver = _Driver([_yielding_body(4, fail_after=1), _yielding_body(4)])
    assert driver.run() == []

    with pytest.raises(RuntimeError, match="Microbatch 0 of 2 failed") as excinfo:
        _raise_ubatch_error(driver.errors, len(driver.contexts))
    assert isinstance(excinfo.value.__cause__, ValueError)


@requires_cuda
def test_failure_before_the_barrier_breaks_it():
    """A thread that dies before the barrier must not strand the main thread.

    The main thread waits on the same barrier the microbatches do, so a
    participant that never arrives would block it indefinitely.
    """

    def dies_early(ctx, trace):
        raise RuntimeError("died before entering the context")

    driver = _Driver([dies_early, _yielding_body(4)])
    hung = driver.run()

    assert hung == [], f"microbatch thread(s) hung: {hung}"
    assert driver.barrier_broke
    assert isinstance(driver.errors[0], RuntimeError)
    assert isinstance(driver.errors[1], threading.BrokenBarrierError)

    # BrokenBarrierError is fallout, so the real error is what gets reported.
    with pytest.raises(RuntimeError, match="Microbatch 0 of 2 failed") as excinfo:
        _raise_ubatch_error(driver.errors, len(driver.contexts))
    assert str(excinfo.value.__cause__) == "died before entering the context"

    # A context that raised out of __enter__ still has to deregister itself,
    # or its dead thread id keeps dbo_enabled() true for whoever reuses it.
    assert _THREAD_ID_TO_CONTEXT == {}


@requires_cuda
def test_microbatches_still_alternate_when_nothing_fails():
    """Guard the guard: the happy path keeps its strict ping-pong."""
    driver = _Driver([_yielding_body(2), _yielding_body(2)])
    hung = driver.run()

    assert hung == []
    assert driver.errors == {}
    assert driver.trace == [
        ("enter", 0),
        ("enter", 1),
        ("done", 0),
        ("done", 1),
    ]
    assert _THREAD_ID_TO_CONTEXT == {}
