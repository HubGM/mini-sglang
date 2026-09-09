from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Callable, Dict


class RequestLifecycleState(str, enum.Enum):
    CREATED = "created"
    WAITING = "waiting"
    PREFILL_SELECTED = "prefill_selected"
    PREFILL_RUNNING = "prefill_running"
    DECODING = "decoding"
    FINISHING = "finishing"
    COMPLETED = "completed"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL_STATES = frozenset(
    {
        RequestLifecycleState.COMPLETED,
        RequestLifecycleState.CANCELLED,
        RequestLifecycleState.FAILED,
    }
)

_LEGAL_TRANSITIONS = {
    RequestLifecycleState.CREATED: {
        RequestLifecycleState.WAITING,
        RequestLifecycleState.CANCELLATION_REQUESTED,
        RequestLifecycleState.FAILED,
    },
    RequestLifecycleState.WAITING: {
        RequestLifecycleState.PREFILL_SELECTED,
        RequestLifecycleState.CANCELLATION_REQUESTED,
        RequestLifecycleState.FAILED,
    },
    RequestLifecycleState.PREFILL_SELECTED: {
        RequestLifecycleState.WAITING,
        RequestLifecycleState.PREFILL_RUNNING,
        RequestLifecycleState.CANCELLATION_REQUESTED,
        RequestLifecycleState.FAILED,
    },
    RequestLifecycleState.PREFILL_RUNNING: {
        RequestLifecycleState.WAITING,
        RequestLifecycleState.DECODING,
        RequestLifecycleState.FINISHING,
        RequestLifecycleState.CANCELLATION_REQUESTED,
        RequestLifecycleState.FAILED,
    },
    RequestLifecycleState.DECODING: {
        RequestLifecycleState.FINISHING,
        RequestLifecycleState.CANCELLATION_REQUESTED,
        RequestLifecycleState.FAILED,
    },
    RequestLifecycleState.FINISHING: {
        RequestLifecycleState.COMPLETED,
        RequestLifecycleState.CANCELLATION_REQUESTED,
        RequestLifecycleState.FAILED,
    },
    RequestLifecycleState.CANCELLATION_REQUESTED: {
        RequestLifecycleState.CANCELLED,
        RequestLifecycleState.FAILED,
    },
}


class LifecycleTransitionError(RuntimeError):
    pass


@dataclass
class RequestLifecycle:
    uid: int
    input_tokens: int
    requested_output_tokens: int
    deadline_ms: float | None = None
    state: RequestLifecycleState = RequestLifecycleState.CREATED
    created_time_ns: int = field(default_factory=time.monotonic_ns)
    enqueue_time_ns: int | None = None
    first_scheduled_time_ns: int | None = None
    prefill_start_time_ns: int | None = None
    prefill_end_time_ns: int | None = None
    first_decode_time_ns: int | None = None
    finish_time_ns: int | None = None
    cancellation_time_ns: int | None = None
    generated_tokens: int = 0
    terminal_reason: str | None = None
    _finish_callback_claimed: bool = field(default=False, repr=False)
    _sse_stop_claimed: bool = field(default=False, repr=False)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def transition(
        self,
        new_state: RequestLifecycleState,
        *,
        now_ns: int | None = None,
        reason: str | None = None,
    ) -> bool:
        if new_state == self.state:
            return False
        if self.is_terminal:
            raise LifecycleTransitionError(
                f"Request {self.uid} cannot leave terminal state {self.state.value}"
            )
        if new_state not in _LEGAL_TRANSITIONS.get(self.state, set()):
            raise LifecycleTransitionError(
                f"Illegal request lifecycle transition: "
                f"{self.state.value} -> {new_state.value}"
            )

        now_ns = now_ns or time.monotonic_ns()
        self.state = new_state
        if new_state == RequestLifecycleState.WAITING and self.enqueue_time_ns is None:
            self.enqueue_time_ns = now_ns
        elif (
            new_state == RequestLifecycleState.PREFILL_SELECTED
            and self.first_scheduled_time_ns is None
        ):
            self.first_scheduled_time_ns = now_ns
        elif (
            new_state == RequestLifecycleState.PREFILL_RUNNING
            and self.prefill_start_time_ns is None
        ):
            self.prefill_start_time_ns = now_ns
        elif new_state == RequestLifecycleState.DECODING:
            pass
        elif new_state == RequestLifecycleState.CANCELLATION_REQUESTED:
            self.cancellation_time_ns = now_ns
        elif new_state in TERMINAL_STATES:
            self.finish_time_ns = now_ns
            self.terminal_reason = reason or new_state.value
        return True

    def request_cancellation(self, reason: str = "cancelled") -> bool:
        if self.is_terminal or self.state == RequestLifecycleState.CANCELLATION_REQUESTED:
            return False
        self.terminal_reason = reason
        return self.transition(RequestLifecycleState.CANCELLATION_REQUESTED)

    def finish_cancelled(self, reason: str = "cancelled") -> bool:
        if self.state == RequestLifecycleState.CANCELLED:
            return False
        if self.is_terminal:
            return False
        if self.state != RequestLifecycleState.CANCELLATION_REQUESTED:
            self.request_cancellation(reason)
        return self.transition(RequestLifecycleState.CANCELLED, reason=reason)

    def fail(self, reason: str) -> bool:
        if self.is_terminal:
            return False
        return self.transition(RequestLifecycleState.FAILED, reason=reason)

    def complete(self) -> bool:
        if self.is_terminal:
            return False
        if self.state != RequestLifecycleState.FINISHING:
            self.transition(RequestLifecycleState.FINISHING)
        return self.transition(RequestLifecycleState.COMPLETED, reason="completed")

    def claim_finish_callback(self) -> bool:
        if self._finish_callback_claimed:
            return False
        self._finish_callback_claimed = True
        return True

    def claim_sse_stop(self) -> bool:
        if self._sse_stop_claimed:
            return False
        self._sse_stop_claimed = True
        return True

    def add_generated_token(self) -> None:
        if not self.is_terminal:
            self.generated_tokens += 1

    def mark_prefill_end(self, now_ns: int | None = None) -> None:
        if self.prefill_end_time_ns is None:
            self.prefill_end_time_ns = now_ns or time.monotonic_ns()

    def mark_first_decode(self, now_ns: int | None = None) -> None:
        if self.first_decode_time_ns is None:
            self.first_decode_time_ns = now_ns or time.monotonic_ns()

    def age_ms(self, now_ns: int | None = None) -> float:
        now_ns = now_ns or time.monotonic_ns()
        return max(0, now_ns - self.created_time_ns) / 1_000_000

    def waiting_time_ms(self, now_ns: int | None = None) -> float:
        if self.enqueue_time_ns is None:
            return 0.0
        end_ns = self.first_scheduled_time_ns or now_ns or time.monotonic_ns()
        return max(0, end_ns - self.enqueue_time_ns) / 1_000_000

    def deadline_slack_ms(self, now_ns: int | None = None) -> float | None:
        if self.deadline_ms is None:
            return None
        return self.deadline_ms - self.age_ms(now_ns)

    def aggregate_snapshot(self, now_ns: int | None = None) -> Dict[str, int | float | str | None]:
        return {
            "state": self.state.value,
            "input_tokens": self.input_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "generated_tokens": self.generated_tokens,
            "scheduling_age_ms": self.age_ms(now_ns),
            "waiting_time_ms": self.waiting_time_ms(now_ns),
            "deadline_ms": self.deadline_ms,
            "deadline_slack_ms": self.deadline_slack_ms(now_ns),
            "terminal_reason": self.terminal_reason,
        }


@dataclass
class LifecycleRegistry:
    requests: Dict[int, RequestLifecycle] = field(default_factory=dict)

    def create(
        self,
        uid: int,
        *,
        input_tokens: int,
        requested_output_tokens: int,
        deadline_ms: float | None = None,
    ) -> RequestLifecycle:
        if uid in self.requests:
            raise LifecycleTransitionError(f"Duplicate request uid {uid}")
        lifecycle = RequestLifecycle(
            uid=uid,
            input_tokens=input_tokens,
            requested_output_tokens=requested_output_tokens,
            deadline_ms=deadline_ms,
        )
        self.requests[uid] = lifecycle
        return lifecycle

    def get(self, uid: int) -> RequestLifecycle | None:
        return self.requests.get(uid)

    def terminal_uids(self) -> frozenset[int]:
        return frozenset(uid for uid, req in self.requests.items() if req.is_terminal)

    def maximum_waiting_age_ms(self, now_ns: int | None = None) -> float:
        now_ns = now_ns or time.monotonic_ns()
        ages = [
            req.age_ms(now_ns)
            for req in self.requests.values()
            if req.state == RequestLifecycleState.WAITING
        ]
        return max(ages, default=0.0)

    def terminal_counts(self) -> Dict[str, int]:
        result = {"completed": 0, "cancelled": 0, "failed": 0}
        for req in self.requests.values():
            if req.state.value in result:
                result[req.state.value] += 1
        return result

    def for_each_active(self, callback: Callable[[RequestLifecycle], None]) -> None:
        for lifecycle in tuple(self.requests.values()):
            if not lifecycle.is_terminal:
                callback(lifecycle)
