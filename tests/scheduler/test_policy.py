from __future__ import annotations

import time

import pytest

from minisgl.core import Batch
from minisgl.scheduler.policy import (
    SchedulingContext,
    SchedulingMetrics,
    UpstreamDefaultPolicy,
    create_scheduling_policy,
)


class FakeReq:
    def __init__(self, uid: int, extend_len: int = 1) -> None:
        self.uid = uid
        self.extend_len = extend_len


class FakePending:
    def __init__(self, uid: int, waiting_ns: int = 0) -> None:
        self.uid = uid
        self.enqueued_at_ns = time.monotonic_ns() - waiting_ns


def make_context(waiting=(), running=()) -> SchedulingContext:
    return SchedulingContext(
        waiting_requests=tuple(waiting),
        running_requests=tuple(running),
        available_token_budget=32,
        available_kv_blocks=128,
        current_batch_state=None,
        current_timestamp_ns=time.monotonic_ns(),
    )


def test_upstream_policy_prefers_prefill() -> None:
    prefill = Batch(reqs=[FakeReq(1, 8)], phase="prefill")
    decode = Batch(reqs=[FakeReq(2)], phase="decode")
    decode_calls = 0

    def schedule_decode():
        nonlocal decode_calls
        decode_calls += 1
        return decode

    decision = UpstreamDefaultPolicy().select(
        make_context(), lambda budget: prefill if budget == 32 else None, schedule_decode
    )

    assert decision.batch is prefill
    assert [req.uid for req in decision.selected_prefill_requests] == [1]
    assert decision.prefill_chunk_sizes == ((1, 8),)
    assert decision.selected_decode_requests == ()
    assert decode_calls == 0


def test_upstream_policy_falls_back_to_decode_without_duplicates() -> None:
    reqs = [FakeReq(1), FakeReq(2)]
    decode = Batch(reqs=reqs, phase="decode")

    decision = UpstreamDefaultPolicy().select(
        make_context(running=reqs), lambda _: None, lambda: decode
    )

    selected_uids = [req.uid for req in decision.selected_decode_requests]
    assert selected_uids == [1, 2]
    assert len(selected_uids) == len(set(selected_uids))
    assert "selected_decode" in decision.reason_codes


def test_metrics_capture_overhead_waiting_and_starvation() -> None:
    waiting = [FakePending(7, waiting_ns=6_000_000_000)]
    batch = Batch(reqs=[FakeReq(7, 4)], phase="prefill")
    context = make_context(waiting=waiting)
    decision = UpstreamDefaultPolicy().select(
        context, lambda _: batch, lambda: None
    )
    metrics = SchedulingMetrics("upstream_default")

    metrics.record(context, decision)
    snapshot = metrics.snapshot()

    assert snapshot["decision_count"] == 1
    assert snapshot["prefill_tokens"] == 4
    assert snapshot["starvation_count"] == 1
    assert snapshot["decision_latency_mean_us"] >= 0


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported scheduling policy"):
        create_scheduling_policy("deadline")


def test_step_telemetry_records_fallback_and_terminal_counts() -> None:
    context = make_context(waiting=[FakePending(8)])
    context = SchedulingContext(
        **{**context.__dict__, "step_id": 12}
    )
    decision = UpstreamDefaultPolicy().select(
        context, lambda _: None, lambda: None
    )
    metrics = SchedulingMetrics("upstream_default")

    metrics.record(
        context,
        decision,
        fallback_reason="invalid_decision",
        cancelled_count=3,
        failed_count=1,
        maximum_waiting_age_ms=44.0,
    )
    snapshot = metrics.snapshot()

    assert snapshot["last_step_id"] == 12
    assert snapshot["fallback_count"] == 1
    assert snapshot["fallback_reasons"] == {"invalid_decision": 1}
    assert snapshot["cancelled_count"] == 3
    assert snapshot["failed_count"] == 1
    assert snapshot["max_waiting_age_ms"] == 44.0


def test_scheduler_metrics_do_not_export_request_ids() -> None:
    metrics = SchedulingMetrics("upstream_default")
    snapshot = metrics.snapshot()

    assert "uid" not in snapshot
    assert "request_id" not in snapshot
    assert "prompt" not in snapshot
