from __future__ import annotations

from dataclasses import dataclass

import pytest

from minisgl.core import Batch
from minisgl.scheduler.policy import (
    BaseSchedulingPolicy,
    PolicyController,
    PolicyHealth,
    PolicyValidationError,
    SchedulingContext,
    SchedulingDecision,
    UpstreamDefaultPolicy,
    UpstreamPolicyFatalError,
)


@dataclass
class FakeReq:
    uid: int
    extend_len: int = 1


class FailingPolicy(BaseSchedulingPolicy):
    name = "failing"

    def select(self, context, schedule_prefill, schedule_decode):
        raise RuntimeError("injected")


class DuplicatePolicy(BaseSchedulingPolicy):
    name = "duplicate"

    def select(self, context, schedule_prefill, schedule_decode):
        req = context.running_requests[0]
        batch = Batch(reqs=[req, req], phase="decode")
        return SchedulingDecision(
            batch=batch,
            selected_decode_requests=(req, req),
        )


class OverBudgetPolicy(BaseSchedulingPolicy):
    name = "over_budget"

    def select(self, context, schedule_prefill, schedule_decode):
        req = FakeReq(context.waiting_requests[0].uid, extend_len=9)
        batch = Batch(reqs=[req], phase="prefill")
        return SchedulingDecision(
            batch=batch,
            selected_prefill_requests=(req,),
            prefill_chunk_sizes=((req.uid, 9),),
        )


def make_context(*, waiting=(), running=(), budget=8) -> SchedulingContext:
    return SchedulingContext(
        waiting_requests=tuple(waiting),
        running_requests=tuple(running),
        available_token_budget=budget,
        available_kv_blocks=100,
        current_batch_state=None,
        current_timestamp_ns=1,
    )


def test_policy_exception_falls_back_to_upstream_default() -> None:
    req = FakeReq(1)
    fallback_batch = Batch(reqs=[req], phase="decode")
    rollback_calls = []
    controller = PolicyController(FailingPolicy())

    selection = controller.select(
        make_context(running=[req]),
        lambda _: None,
        lambda: fallback_batch,
        rollback=lambda: rollback_calls.append(True),
    )

    assert selection.policy_name == "upstream_default"
    assert selection.fallback_reason == "policy_exception"
    assert selection.decision.batch is fallback_batch
    assert rollback_calls == [True]


def test_duplicate_decision_is_rejected_and_falls_back() -> None:
    req = FakeReq(2)
    fallback_batch = Batch(reqs=[req], phase="decode")
    controller = PolicyController(DuplicatePolicy())

    selection = controller.select(
        make_context(running=[req]),
        lambda _: None,
        lambda: fallback_batch,
        rollback=lambda: None,
    )

    assert selection.fallback_reason == "invalid_decision"
    assert selection.decision.batch is fallback_batch


def test_prefill_token_budget_violation_is_rejected() -> None:
    pending = FakeReq(3)
    policy = OverBudgetPolicy()
    decision = policy.select(
        make_context(waiting=[pending], budget=8), lambda _: None, lambda: None
    )

    with pytest.raises(PolicyValidationError, match="budget"):
        policy.validate_decision(
            make_context(waiting=[pending], budget=8), decision
        )


def test_terminal_request_selection_is_rejected() -> None:
    req = FakeReq(4)
    batch = Batch(reqs=[req], phase="decode")
    decision = SchedulingDecision(
        batch=batch, selected_decode_requests=(req,)
    )

    with pytest.raises(PolicyValidationError, match="terminal"):
        UpstreamDefaultPolicy().validate_decision(
            make_context(running=[req]),
            decision,
            terminal_uids={4},
        )


def test_repeated_policy_failures_open_circuit() -> None:
    req = FakeReq(5)
    fallback_batch = Batch(reqs=[req], phase="decode")
    controller = PolicyController(FailingPolicy(), failure_threshold=2)

    for _ in range(2):
        controller.select(
            make_context(running=[req]),
            lambda _: None,
            lambda: fallback_batch,
            rollback=lambda: None,
        )

    assert controller.circuit_open
    assert controller.health == PolicyHealth.CIRCUIT_OPEN

    selection = controller.select(
        make_context(running=[req]),
        lambda _: None,
        lambda: fallback_batch,
        rollback=lambda: None,
    )
    assert selection.policy_name == "upstream_default"
    assert selection.fallback_reason is None


def test_policy_circuit_requires_manual_reset() -> None:
    controller = PolicyController(FailingPolicy())
    controller.circuit_open = True
    controller.consecutive_failures = 3

    controller.manual_reset()

    assert not controller.circuit_open
    assert controller.consecutive_failures == 0


def test_upstream_default_failure_is_fatal() -> None:
    controller = PolicyController(
        active_policy=UpstreamDefaultPolicy(),
        fallback_policy=UpstreamDefaultPolicy(),
    )

    with pytest.raises(UpstreamPolicyFatalError):
        controller.select(
            make_context(),
            lambda _: (_ for _ in ()).throw(RuntimeError("injected")),
            lambda: None,
            rollback=lambda: None,
        )


def test_decision_timing_can_be_disabled() -> None:
    context = make_context()
    context = SchedulingContext(
        **{**context.__dict__, "measure_decision_overhead": False}
    )

    decision = UpstreamDefaultPolicy().select(
        context, lambda _: None, lambda: None
    )

    assert decision.decision_latency_ns == 0
