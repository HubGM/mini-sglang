from __future__ import annotations

import pytest

from minisgl.scheduler.lifecycle import (
    LifecycleRegistry,
    LifecycleTransitionError,
    RequestLifecycle,
    RequestLifecycleState,
)


def make_lifecycle() -> RequestLifecycle:
    return RequestLifecycle(uid=1, input_tokens=8, requested_output_tokens=4)


def test_legal_request_lifecycle_reaches_completed() -> None:
    lifecycle = make_lifecycle()

    for state in (
        RequestLifecycleState.WAITING,
        RequestLifecycleState.PREFILL_SELECTED,
        RequestLifecycleState.PREFILL_RUNNING,
        RequestLifecycleState.DECODING,
        RequestLifecycleState.FINISHING,
        RequestLifecycleState.COMPLETED,
    ):
        assert lifecycle.transition(state)

    assert lifecycle.is_terminal
    assert lifecycle.finish_time_ns is not None


def test_illegal_transition_is_rejected() -> None:
    lifecycle = make_lifecycle()

    with pytest.raises(LifecycleTransitionError, match="created -> decoding"):
        lifecycle.transition(RequestLifecycleState.DECODING)


def test_terminal_state_cannot_change() -> None:
    lifecycle = make_lifecycle()
    lifecycle.fail("test_failure")

    with pytest.raises(LifecycleTransitionError, match="terminal state"):
        lifecycle.transition(RequestLifecycleState.WAITING)


def test_duplicate_terminal_operation_is_idempotent() -> None:
    lifecycle = make_lifecycle()

    assert lifecycle.fail("first")
    assert not lifecycle.fail("second")
    assert lifecycle.terminal_reason == "first"


def test_finish_callback_can_only_be_claimed_once() -> None:
    lifecycle = make_lifecycle()

    assert lifecycle.claim_finish_callback()
    assert not lifecycle.claim_finish_callback()


def test_sse_stop_can_only_be_claimed_once() -> None:
    lifecycle = make_lifecycle()

    assert lifecycle.claim_sse_stop()
    assert not lifecycle.claim_sse_stop()


def test_cancellation_records_monotonic_timestamp() -> None:
    lifecycle = make_lifecycle()
    lifecycle.transition(RequestLifecycleState.WAITING)

    assert lifecycle.request_cancellation("client_disconnected")
    assert lifecycle.cancellation_time_ns is not None
    assert lifecycle.finish_cancelled("client_disconnected")
    assert lifecycle.state == RequestLifecycleState.CANCELLED


def test_cancellation_loses_race_after_completion() -> None:
    lifecycle = make_lifecycle()
    lifecycle.transition(RequestLifecycleState.WAITING)
    lifecycle.transition(RequestLifecycleState.PREFILL_SELECTED)
    lifecycle.transition(RequestLifecycleState.PREFILL_RUNNING)
    lifecycle.transition(RequestLifecycleState.DECODING)
    lifecycle.complete()

    assert not lifecycle.request_cancellation()
    assert lifecycle.state == RequestLifecycleState.COMPLETED


def test_waiting_age_and_deadline_slack_use_monotonic_time() -> None:
    lifecycle = RequestLifecycle(
        uid=1,
        input_tokens=8,
        requested_output_tokens=4,
        deadline_ms=100.0,
        created_time_ns=1_000_000_000,
    )
    lifecycle.transition(RequestLifecycleState.WAITING, now_ns=1_010_000_000)

    assert lifecycle.age_ms(1_025_000_000) == 25.0
    assert lifecycle.waiting_time_ms(1_025_000_000) == 15.0
    assert lifecycle.deadline_slack_ms(1_025_000_000) == 75.0


def test_registry_reports_only_low_cardinality_terminal_counts() -> None:
    registry = LifecycleRegistry()
    completed = registry.create(
        10, input_tokens=3, requested_output_tokens=2
    )
    cancelled = registry.create(
        20, input_tokens=4, requested_output_tokens=2
    )
    completed.fail("failed")
    cancelled.request_cancellation()
    cancelled.finish_cancelled()

    assert registry.terminal_counts() == {
        "completed": 0,
        "cancelled": 1,
        "failed": 1,
    }
    assert set(registry.terminal_uids()) == {10, 20}
