from __future__ import annotations

from dataclasses import dataclass

from minisgl.core import Batch
from minisgl.scheduler import (
    DeadlineAgingPolicy,
    DeadlineAgingV2Policy,
    DeadlineAwarePolicy,
    RequestSchedulingInfo,
    SchedulingContext,
    SchedulingPolicyConfig,
    ServiceTimeEstimator,
    TokenBudgetPolicy,
    create_scheduling_policy,
    dual_slo_config_hash,
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
        "default_tpot_deadline_ms": 50.0,
        "default_e2e_deadline_ms": 1200.0,
        "initial_prefill_ms_per_token": 0.1,
        "initial_decode_step_ms": 10.0,
        "service_ewma_alpha": 0.5,
        "max_wait_ms": 400.0,
        "aging_start_ms": 100.0,
        "aging_rate": 1.0,
        "min_prefill_budget_per_step": 2,
        "max_decode_only_steps": 2,
        "prefill_urgent_threshold_ms": 150.0,
        "hard_max_wait_ms": 300.0,
        "min_decode_reserve_ratio": 0.25,
        "max_decode_reserve_ratio": 0.65,
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
    first_scheduled_ns: int | None = None,
    last_token_ns: int | None = None,
    tpot_ms: float = 50.0,
    generated_tokens: int = 0,
    is_chunk_continuation: bool = False,
) -> RequestSchedulingInfo:
    return RequestSchedulingInfo(
        uid=uid,
        enqueue_time_ns=enqueue_ns,
        first_scheduled_time_ns=first_scheduled_ns,
        first_token_time_ns=first_token_ns,
        input_tokens=remaining_input,
        remaining_input_tokens=remaining_input,
        requested_output_tokens=remaining_output,
        generated_tokens=generated_tokens,
        ttft_deadline_ms=ttft_ms,
        e2e_deadline_ms=e2e_ms,
        tpot_deadline_ms=tpot_ms,
        last_token_time_ns=last_token_ns,
        is_chunk_continuation=is_chunk_continuation,
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
        ordered = [
            by_uid[uid] for uid in priority_uids or by_uid if uid in by_uid
        ]
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


def test_v2_pre_first_token_uses_ttft_not_remaining_e2e_slack() -> None:
    policy = DeadlineAgingV2Policy(policy_config())
    request = info(1, remaining_input=100, ttft_ms=200.0, e2e_ms=5.0)

    assert policy.raw_slack_ms(request, 0) == 190.0


def test_v2_post_first_token_uses_e2e_and_tpot_cadence() -> None:
    policy = DeadlineAgingV2Policy(policy_config())
    request = info(
        1,
        remaining_input=0,
        remaining_output=4,
        first_token_ns=1,
        last_token_ns=100_000_000,
        generated_tokens=2,
        e2e_ms=1200.0,
        tpot_ms=50.0,
    )

    assert policy.raw_slack_ms(request, 160_000_000) == -10.0


def test_v2_negative_ttft_slack_preempts_positive_waiter() -> None:
    pending = [FakePending(1), FakePending(2)]
    positive = info(1, ttft_ms=500.0)
    negative = info(2, remaining_input=100, ttft_ms=5.0)
    callbacks = Callbacks(prefill_reqs=[FakeReq(1, 2), FakeReq(2, 2)])
    policy = DeadlineAgingV2Policy(policy_config())

    policy.select(
        context(waiting=pending, request_info=[positive, negative]),
        callbacks.prefill,
        callbacks.decode,
    )

    assert callbacks.prefill_calls[0][1][0] == 2


def test_v2_more_negative_ttft_slack_has_priority() -> None:
    pending = [FakePending(1), FakePending(2)]
    late = info(1, remaining_input=100, ttft_ms=5.0)
    later = info(2, remaining_input=200, ttft_ms=5.0)
    callbacks = Callbacks(prefill_reqs=[FakeReq(1, 2), FakeReq(2, 2)])
    policy = DeadlineAgingV2Policy(policy_config())

    policy.select(
        context(waiting=pending, request_info=[late, later]),
        callbacks.prefill,
        callbacks.decode,
    )

    assert callbacks.prefill_calls[0][1] == (2, 1)


def test_v2_max_decode_only_steps_guarantees_prefill_service() -> None:
    pending = [FakePending(1)]
    running = [FakeReq(2)]
    waiting_info = info(1, ttft_ms=1000.0)
    decode_info = info(
        2,
        remaining_input=0,
        first_token_ns=1,
        last_token_ns=1,
        e2e_ms=5.0,
    )
    callbacks = Callbacks(prefill_reqs=[FakeReq(1, 2)], decode_reqs=running)
    policy = DeadlineAgingV2Policy(
        policy_config(max_decode_only_steps=1, max_consecutive_prefill_steps=10)
    )
    ctx = context(
        waiting=pending,
        running=running,
        request_info=[waiting_info, decode_info],
        now_ns=20_000_000,
    )

    first = policy.select(ctx, callbacks.prefill, callbacks.decode)
    second = policy.select(ctx, callbacks.prefill, callbacks.decode)

    assert first.batch is not None and first.batch.is_decode
    assert second.batch is not None and second.batch.is_prefill
    assert callbacks.prefill_calls[-1][0] >= policy.config.min_prefill_budget_per_step
    assert "max_decode_only_prefill_guarantee" in second.reason_codes


def test_v2_hard_urgent_is_fifo_and_new_short_cannot_overtake() -> None:
    now_ns = 500_000_000
    oldest = info(1, enqueue_ns=0, ttft_ms=2000.0)
    newer = info(2, enqueue_ns=50_000_000, ttft_ms=5.0)
    pending = [FakePending(2), FakePending(1)]
    callbacks = Callbacks(prefill_reqs=[FakeReq(1, 2), FakeReq(2, 2)])
    policy = DeadlineAgingV2Policy(policy_config(hard_max_wait_ms=300.0))

    decision = policy.select(
        context(
            waiting=pending,
            request_info=[newer, oldest],
            now_ns=now_ns,
        ),
        callbacks.prefill,
        callbacks.decode,
    )

    assert callbacks.prefill_calls[0][1][0] == 1
    assert decision.hard_urgent_request_count == 2
    assert "hard_urgent_prefill" in decision.reason_codes


def test_v2_hard_urgent_beats_chunk_continuation() -> None:
    now_ns = 400_000_000
    continuation = info(
        1,
        enqueue_ns=0,
        first_scheduled_ns=1,
        is_chunk_continuation=True,
    )
    urgent_short = info(2, enqueue_ns=0)
    pending = [FakePending(1), FakePending(2)]
    callbacks = Callbacks(prefill_reqs=[FakeReq(2, 2)])
    policy = DeadlineAgingV2Policy(policy_config())

    decision = policy.select(
        context(
            waiting=pending,
            request_info=[continuation, urgent_short],
            now_ns=now_ns,
        ),
        callbacks.prefill,
        callbacks.decode,
    )

    assert callbacks.prefill_calls[0][1][0] == 2
    assert decision.chunk_continuation_deferred


def test_v2_dynamic_decode_reservation_stays_within_bounds() -> None:
    policy = DeadlineAgingV2Policy(
        policy_config(min_decode_reserve_ratio=0.2, max_decode_reserve_ratio=0.6)
    )
    pending = [FakePending(1)]
    running = [FakeReq(2)]
    request_info = [
        info(1, ttft_ms=1000.0),
        info(2, remaining_input=0, first_token_ns=1, last_token_ns=1),
    ]
    ctx = context(
        waiting=pending,
        running=running,
        request_info=request_info,
        now_ns=10_000_000,
    )

    ratio = policy._decode_reserve_ratio(ctx)

    assert 0.2 <= ratio <= 0.6


def test_v2_cancelled_hard_urgent_is_not_selected() -> None:
    policy = DeadlineAgingV2Policy(policy_config())
    policy.on_request_cancelled(1)
    remaining = info(2, enqueue_ns=0)
    callbacks = Callbacks(prefill_reqs=[FakeReq(2, 2)])

    decision = policy.select(
        context(
            waiting=[FakePending(2)],
            request_info=[remaining],
            now_ns=500_000_000,
        ),
        callbacks.prefill,
        callbacks.decode,
    )

    assert [req.uid for req in decision.selected_prefill_requests] == [2]


def test_v2_deterministic_stress_has_no_lost_duplicate_or_starved_request() -> None:
    policy = DeadlineAgingV2Policy(
        policy_config(
            max_decode_only_steps=2,
            max_consecutive_prefill_steps=2,
            hard_max_wait_ms=300.0,
        )
    )
    pending = [FakePending(uid) for uid in range(1, 7)]
    enqueue = {uid: 0 for uid in range(1, 7)}
    selected: list[int] = []
    decode_req = FakeReq(99)

    for step in range(20):
        now_ns = step * 100_000_000
        request_info = [
            info(uid, enqueue_ns=enqueue[uid], ttft_ms=250.0)
            for uid in (req.uid for req in pending)
        ]
        request_info.append(
            info(
                99,
                remaining_input=0,
                first_token_ns=1,
                last_token_ns=max(1, now_ns - 60_000_000),
                e2e_ms=10.0,
            )
        )

        def schedule_one_prefill(budget, priority_uids=None, max_chunk_tokens=None):
            _ = (budget, max_chunk_tokens)
            if not pending:
                return None
            by_uid = {req.uid: req for req in pending}
            uid = next(uid for uid in priority_uids if uid in by_uid)
            return Batch(reqs=[FakeReq(uid, 2)], phase="prefill")

        decision = policy.select(
            context(
                waiting=pending,
                running=[decode_req],
                request_info=request_info,
                now_ns=now_ns,
            ),
            schedule_one_prefill,
            lambda max_requests=None, priority_uids=None: Batch(
                reqs=[decode_req], phase="decode"
            ),
        )
        policy.validate_decision(
            context(
                waiting=pending,
                running=[decode_req],
                request_info=request_info,
                now_ns=now_ns,
            ),
            decision,
        )
        if decision.selected_prefill_requests:
            uid = decision.selected_prefill_requests[0].uid
            selected.append(uid)
            pending = [req for req in pending if req.uid != uid]
        if not pending:
            break

    assert selected == [1, 2, 3, 4, 5, 6]
    assert len(selected) == len(set(selected))
    assert not pending


def test_v1_v2_names_and_legacy_alias_are_stable() -> None:
    config = policy_config()

    legacy = create_scheduling_policy("deadline_aging", config)
    v1 = create_scheduling_policy("deadline_aging_v1", config)
    v2 = create_scheduling_policy("deadline_aging_v2", config)

    assert isinstance(legacy, DeadlineAgingPolicy)
    assert isinstance(v1, DeadlineAgingPolicy)
    assert isinstance(v2, DeadlineAgingV2Policy)
    assert legacy.name == v1.name == "deadline_aging_v1"


def test_dual_slo_config_hash_is_stable_and_sensitive() -> None:
    first = policy_config()
    same = policy_config()
    changed = policy_config(hard_max_wait_ms=301.0)

    assert dual_slo_config_hash(first) == dual_slo_config_hash(same)
    assert dual_slo_config_hash(first) != dual_slo_config_hash(changed)
