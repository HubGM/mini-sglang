from __future__ import annotations

from dataclasses import dataclass

from minisgl.core import Batch
from minisgl.scheduler import (
    DeadlineAgingPolicy,
    DeadlineAwarePolicy,
    RequestSchedulingInfo,
    SchedulingContext,
    SchedulingPolicyConfig,
    ServiceTimeEstimator,
    TokenBudgetPolicy,
)


@dataclass
class FakePending:
    uid: int
    enqueued_at_ns: int = 0


@dataclass
class FakeReq:
    uid: int
    extend_len: int = 1


def policy_config(**overrides) -> SchedulingPolicyConfig:
    values = {
        "max_step_tokens": 8,
        "max_prefill_chunk_tokens": 3,
        "decode_reserve_ratio": 0.5,
        "max_consecutive_prefill_steps": 1,
        "default_ttft_deadline_ms": 200.0,
        "default_e2e_deadline_ms": 1200.0,
        "initial_prefill_ms_per_token": 0.1,
        "initial_decode_step_ms": 10.0,
        "service_ewma_alpha": 0.5,
        "max_wait_ms": 400.0,
        "aging_start_ms": 100.0,
        "aging_rate": 1.0,
    }
    values.update(overrides)
    return SchedulingPolicyConfig(**values)


def info(
    uid: int,
    *,
    enqueue_ns: int = 0,
    remaining_input: int = 10,
    remaining_output: int = 10,
    ttft_ms: float = 200.0,
    e2e_ms: float = 1200.0,
    first_token_ns: int | None = None,
) -> RequestSchedulingInfo:
    return RequestSchedulingInfo(
        uid=uid,
        enqueue_time_ns=enqueue_ns,
        first_scheduled_time_ns=None,
        first_token_time_ns=first_token_ns,
        input_tokens=remaining_input,
        remaining_input_tokens=remaining_input,
        requested_output_tokens=remaining_output,
        generated_tokens=0,
        ttft_deadline_ms=ttft_ms,
        e2e_deadline_ms=e2e_ms,
    )


def context(
    *,
    waiting=(),
    running=(),
    request_info=(),
    now_ns: int = 0,
) -> SchedulingContext:
    return SchedulingContext(
        waiting_requests=tuple(waiting),
        running_requests=tuple(running),
        available_token_budget=32,
        available_kv_blocks=128,
        current_batch_state=None,
        current_timestamp_ns=now_ns,
        request_info=tuple(request_info),
    )


class Callbacks:
    def __init__(self, prefill_reqs=(), decode_reqs=()) -> None:
        self.prefill_reqs = tuple(prefill_reqs)
        self.decode_reqs = tuple(decode_reqs)
        self.prefill_calls = []
        self.decode_calls = []

    def prefill(self, budget, priority_uids=None, max_chunk_tokens=None):
        self.prefill_calls.append((budget, tuple(priority_uids or ()), max_chunk_tokens))
        if not self.prefill_reqs:
            return None
        by_uid = {req.uid: req for req in self.prefill_reqs}
        ordered = [by_uid[uid] for uid in priority_uids or by_uid]
        return Batch(reqs=ordered, phase="prefill")

    def decode(self, max_requests=None, priority_uids=None):
        self.decode_calls.append((max_requests, tuple(priority_uids or ())))
        if not self.decode_reqs:
            return None
        by_uid = {req.uid: req for req in self.decode_reqs}
        ordered = [by_uid[uid] for uid in priority_uids or by_uid]
        return Batch(reqs=ordered[:max_requests], phase="decode")


def test_token_budget_caps_step_and_per_request_prefill_chunk() -> None:
    pending = [FakePending(1), FakePending(2)]
    reqs = [FakeReq(1, 3), FakeReq(2, 3)]
    callbacks = Callbacks(prefill_reqs=reqs)
    policy = TokenBudgetPolicy(policy_config())

    decision = policy.select(
        context(waiting=pending), callbacks.prefill, callbacks.decode
    )
    policy.validate_decision(context(waiting=pending), decision)

    assert decision.token_budget == 8
    assert decision.token_budget_used == 6
    assert max(size for _, size in decision.prefill_chunk_sizes) == 3
    assert callbacks.prefill_calls == [(8, (1, 2), 3)]


def test_decode_reservation_and_prefill_steps_interleave() -> None:
    pending = [FakePending(1)]
    running = [FakeReq(2)]
    callbacks = Callbacks(prefill_reqs=[FakeReq(1, 3)], decode_reqs=running)
    policy = TokenBudgetPolicy(policy_config())
    ctx = context(waiting=pending, running=running)

    first = policy.select(ctx, callbacks.prefill, callbacks.decode)
    second = policy.select(ctx, callbacks.prefill, callbacks.decode)

    assert first.batch is not None and first.batch.is_prefill
    assert second.batch is not None and second.batch.is_decode
    assert second.token_budget_used == 1
    assert callbacks.decode_calls[-1][0] == 1


def test_deadline_policy_orders_negative_slack_first() -> None:
    pending = [FakePending(1), FakePending(2)]
    request_info = [
        info(1, ttft_ms=500.0),
        info(2, remaining_input=100, ttft_ms=5.0, e2e_ms=20.0),
    ]
    callbacks = Callbacks(prefill_reqs=[FakeReq(1, 2), FakeReq(2, 2)])
    policy = DeadlineAwarePolicy(policy_config())
    ctx = context(waiting=pending, request_info=request_info)

    decision = policy.select(ctx, callbacks.prefill, callbacks.decode)

    assert callbacks.prefill_calls[0][1] == (2, 1)
    assert decision.minimum_slack_ms is not None
    assert decision.minimum_slack_ms < 0


def test_deadline_policy_uses_only_current_state_and_completed_step_ewma() -> None:
    estimator = ServiceTimeEstimator(0.1, 10.0, 0.5)
    request = info(1, remaining_input=20, remaining_output=4)

    before = estimator.remaining_ms(request)
    estimator.observe(
        phase="prefill",
        elapsed_ms=8.0,
        prefill_tokens=20,
        decode_tokens=0,
    )
    after = estimator.remaining_ms(request)

    assert before == 42.0
    assert after > before
    assert not hasattr(request, "finish_time_ns")


def test_aging_priority_improves_monotonically_with_waiting_age() -> None:
    policy = DeadlineAgingPolicy(policy_config())
    request = info(1)

    at_start = policy.effective_slack_ms(request, 100_000_000)
    later = policy.effective_slack_ms(request, 300_000_000)

    assert later < at_start


def test_max_wait_promotes_old_request_ahead_of_new_arrival() -> None:
    now_ns = 500_000_000
    old = info(1, enqueue_ns=0, ttft_ms=1000.0)
    new = info(2, enqueue_ns=490_000_000, ttft_ms=50.0)
    pending = [FakePending(2), FakePending(1)]
    callbacks = Callbacks(prefill_reqs=[FakeReq(1, 2), FakeReq(2, 2)])
    policy = DeadlineAgingPolicy(policy_config())
    ctx = context(
        waiting=pending,
        request_info=[new, old],
        now_ns=now_ns,
    )

    decision = policy.select(ctx, callbacks.prefill, callbacks.decode)

    assert callbacks.prefill_calls[0][1][0] == 1
    assert decision.urgent_request_count == 1


def test_urgent_prefill_cannot_starve_decode_indefinitely() -> None:
    now_ns = 500_000_000
    old = info(1, enqueue_ns=0)
    decoding = info(
        2,
        enqueue_ns=0,
        remaining_input=0,
        first_token_ns=1,
    )
    pending = [FakePending(1)]
    running = [FakeReq(2)]
    callbacks = Callbacks(prefill_reqs=[FakeReq(1, 2)], decode_reqs=running)
    policy = DeadlineAgingPolicy(policy_config())
    ctx = context(
        waiting=pending,
        running=running,
        request_info=[old, decoding],
        now_ns=now_ns,
    )

    first = policy.select(ctx, callbacks.prefill, callbacks.decode)
    second = policy.select(ctx, callbacks.prefill, callbacks.decode)

    assert first.batch is not None and first.batch.is_prefill
    assert second.batch is not None and second.batch.is_decode
