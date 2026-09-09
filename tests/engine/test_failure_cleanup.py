from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from minisgl.core import Batch
from minisgl.scheduler.lifecycle import LifecycleRegistry, RequestLifecycleState
from minisgl.scheduler.scheduler import (
    BatchPreparationError,
    ForwardInput,
    Scheduler,
)


class FakeReq:
    def __init__(self, uid: int, extend_len: int = 2) -> None:
        self.uid = uid
        self.extend_len = extend_len
        self.table_idx = 0
        self.cached_len = 0
        self.device_len = extend_len


class FakeCache:
    def __init__(self) -> None:
        self.freed = []

    def allocate(self, size: int) -> torch.Tensor:
        return torch.arange(size, dtype=torch.int32)

    def _free(self, indices: torch.Tensor) -> None:
        self.freed.append(indices.clone())


class FailingGraphRunner:
    def pad_batch(self, batch: Batch) -> int:
        raise RuntimeError("injected_prepare_failure")


class PassingGraphRunner:
    def pad_batch(self, batch: Batch) -> int:
        batch.padded_reqs = list(batch.reqs)
        return 0


def make_prepare_scheduler(graph_runner) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cache_manager = FakeCache()
    scheduler.engine = SimpleNamespace(
        graph_runner=graph_runner,
        dummy_page=99,
        attn_backend=SimpleNamespace(prepare_metadata=lambda batch: None),
        sampler=SimpleNamespace(prepare=lambda batch: object()),
    )
    scheduler.token_pool = torch.zeros((2, 8), dtype=torch.int32)
    scheduler.page_table = torch.zeros((2, 8), dtype=torch.int32)
    return scheduler


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_batch_prepare_exception_returns_new_kv_pages(phase: str) -> None:
    scheduler = make_prepare_scheduler(FailingGraphRunner())
    batch = Batch(reqs=[FakeReq(1)], phase=phase)

    with pytest.raises(BatchPreparationError) as exc_info:
        scheduler._prepare_batch(batch)

    assert exc_info.value.phase == phase
    assert scheduler.cache_manager.freed[0].tolist() == [0, 1]


def test_sampler_prepare_exception_returns_new_kv_pages(monkeypatch) -> None:
    scheduler = make_prepare_scheduler(PassingGraphRunner())
    scheduler.engine.sampler.prepare = lambda batch: (_ for _ in ()).throw(
        RuntimeError("sampler_prepare_failure")
    )
    monkeypatch.setattr(
        "minisgl.scheduler.scheduler._make_2d_indices",
        lambda table, ranges: torch.tensor([0, 1], dtype=torch.int64),
    )
    batch = Batch(reqs=[FakeReq(2)], phase="prefill")

    with pytest.raises(BatchPreparationError):
        scheduler._prepare_batch(batch)

    assert scheduler.cache_manager.freed[0].tolist() == [0, 1]


class FakeHealth:
    def __init__(self) -> None:
        self.fatal_reason = None

    def fatal(self, reason: str) -> None:
        self.fatal_reason = reason


class StagedFailure(RuntimeError):
    stage = "sampling_exception"


@pytest.mark.parametrize(
    ("exception", "expected_reason"),
    [
        (RuntimeError("forward"), "engine_forward_exception"),
        (StagedFailure("sample"), "sampling_exception"),
    ],
)
def test_engine_or_sampling_failure_sets_fatal_health(
    monkeypatch, exception: Exception, expected_reason: str
) -> None:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._selected_reqs = {}
    scheduler._inflight_reqs = {}
    scheduler.lifecycle_registry = LifecycleRegistry()
    lifecycle = scheduler.lifecycle_registry.create(
        3, input_tokens=2, requested_output_tokens=2
    )
    lifecycle.transition(RequestLifecycleState.WAITING)
    lifecycle.transition(RequestLifecycleState.PREFILL_SELECTED)
    scheduler.health_reporter = FakeHealth()
    scheduler.engine = SimpleNamespace(
        forward_batch=lambda batch, args: (_ for _ in ()).throw(exception)
    )
    scheduler.decode_manager = SimpleNamespace(add_reqs=lambda reqs: None)
    failed = []
    monkeypatch.setattr(scheduler, "_load_token_ids", lambda forward_input: None)
    monkeypatch.setattr(
        scheduler, "_fail_batch", lambda batch, reason: failed.append(reason)
    )
    req = FakeReq(3)
    batch = Batch(reqs=[req], phase="prefill")
    forward_input = ForwardInput(
        batch=batch,
        sample_args=object(),
        load_indices=torch.tensor([], dtype=torch.int64),
        write_indices=torch.tensor([], dtype=torch.int64),
    )

    with pytest.raises(type(exception)):
        scheduler._forward(forward_input)

    assert scheduler.health_reporter.fatal_reason == expected_reason
    assert failed == [expected_reason]
