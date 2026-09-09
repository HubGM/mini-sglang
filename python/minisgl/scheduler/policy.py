from __future__ import annotations

import json
import os
import enum
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

from minisgl.core import Batch, Req

from .utils import PendingReq

SchedulePrefill = Callable[[int], Batch | None]
ScheduleDecode = Callable[[], Batch | None]


@dataclass(frozen=True)
class SchedulingContext:
    waiting_requests: Tuple[PendingReq, ...]
    running_requests: Tuple[Req, ...]
    available_token_budget: int
    available_kv_blocks: int
    current_batch_state: str | None
    current_timestamp_ns: int
    step_id: int = 0
    measure_decision_overhead: bool = True


@dataclass(frozen=True)
class SchedulingDecision:
    batch: Batch | None
    selected_prefill_requests: Tuple[Req, ...] = ()
    selected_decode_requests: Tuple[Req, ...] = ()
    prefill_chunk_sizes: Tuple[Tuple[int, int], ...] = ()
    preempted_request_uids: Tuple[int, ...] = ()
    reason_codes: Tuple[str, ...] = ()
    decision_latency_ns: int = 0


class BaseSchedulingPolicy(ABC):
    name: str
    version: str = "1"

    @property
    def health(self) -> PolicyHealth:
        return PolicyHealth.HEALTHY

    @abstractmethod
    def select(
        self,
        context: SchedulingContext,
        schedule_prefill: SchedulePrefill,
        schedule_decode: ScheduleDecode,
    ) -> SchedulingDecision:
        """Select the next engine batch without executing model work."""

    def validate_decision(
        self,
        context: SchedulingContext,
        decision: SchedulingDecision,
        *,
        terminal_uids: Iterable[int] = (),
        released_uids: Iterable[int] = (),
    ) -> None:
        validate_scheduling_decision(
            context,
            decision,
            terminal_uids=terminal_uids,
            released_uids=released_uids,
        )

    def on_request_cancelled(self, uid: int) -> None:
        _ = uid

    def on_request_finished(self, uid: int) -> None:
        _ = uid


class PolicyHealth(str, enum.Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CIRCUIT_OPEN = "circuit_open"


class PolicyValidationError(RuntimeError):
    pass


class UpstreamPolicyFatalError(RuntimeError):
    pass


class UpstreamDefaultPolicy(BaseSchedulingPolicy):
    """Preserve the pinned upstream prefill-first, decode-fallback behavior."""

    name = "upstream_default"
    version = "1"

    def select(
        self,
        context: SchedulingContext,
        schedule_prefill: SchedulePrefill,
        schedule_decode: ScheduleDecode,
    ) -> SchedulingDecision:
        started_ns = (
            time.perf_counter_ns() if context.measure_decision_overhead else 0
        )
        batch = schedule_prefill(context.available_token_budget)
        reason_codes = ["prefill_first"]
        if batch is None:
            reason_codes.append("prefill_unavailable")
            batch = schedule_decode()

        selected_prefill: Tuple[Req, ...] = ()
        selected_decode: Tuple[Req, ...] = ()
        chunk_sizes: Tuple[Tuple[int, int], ...] = ()
        if batch is None:
            reason_codes.append("idle_no_runnable_batch")
        elif batch.is_prefill:
            reason_codes.append("selected_prefill")
            selected_prefill = tuple(batch.reqs)
            chunk_sizes = tuple((req.uid, req.extend_len) for req in batch.reqs)
        else:
            reason_codes.append("selected_decode")
            selected_decode = tuple(batch.reqs)

        return SchedulingDecision(
            batch=batch,
            selected_prefill_requests=selected_prefill,
            selected_decode_requests=selected_decode,
            prefill_chunk_sizes=chunk_sizes,
            reason_codes=tuple(reason_codes),
            decision_latency_ns=(
                time.perf_counter_ns() - started_ns
                if context.measure_decision_overhead
                else 0
            ),
        )


def create_scheduling_policy(name: str) -> BaseSchedulingPolicy:
    if name == UpstreamDefaultPolicy.name:
        return UpstreamDefaultPolicy()
    raise ValueError(f"Unsupported scheduling policy: {name}")


def validate_scheduling_decision(
    context: SchedulingContext,
    decision: SchedulingDecision,
    *,
    terminal_uids: Iterable[int] = (),
    released_uids: Iterable[int] = (),
) -> None:
    batch = decision.batch
    prefill_uids = tuple(req.uid for req in decision.selected_prefill_requests)
    decode_uids = tuple(req.uid for req in decision.selected_decode_requests)
    all_selected_uids = prefill_uids + decode_uids

    if len(all_selected_uids) != len(set(all_selected_uids)):
        raise PolicyValidationError("duplicate_request")
    if set(prefill_uids).intersection(decode_uids):
        raise PolicyValidationError("prefill_decode_overlap")
    forbidden = set(terminal_uids).union(released_uids)
    if forbidden.intersection(all_selected_uids):
        raise PolicyValidationError("terminal_or_released_request")

    waiting_uids = {req.uid for req in context.waiting_requests}
    running_uids = {req.uid for req in context.running_requests}
    if not set(prefill_uids).issubset(waiting_uids):
        raise PolicyValidationError("prefill_request_not_waiting")
    if not set(decode_uids).issubset(running_uids):
        raise PolicyValidationError("decode_request_not_running")

    if batch is None:
        if all_selected_uids or decision.prefill_chunk_sizes:
            raise PolicyValidationError("idle_decision_has_requests")
        return

    batch_uids = tuple(req.uid for req in batch.reqs)
    if len(batch_uids) != len(set(batch_uids)):
        raise PolicyValidationError("duplicate_batch_request")
    expected_uids = prefill_uids if batch.is_prefill else decode_uids
    if batch_uids != expected_uids:
        raise PolicyValidationError("decision_batch_mismatch")

    if batch.is_prefill:
        chunks = decision.prefill_chunk_sizes
        if tuple(uid for uid, _ in chunks) != prefill_uids:
            raise PolicyValidationError("prefill_chunk_request_mismatch")
        if any(size <= 0 for _, size in chunks):
            raise PolicyValidationError("non_positive_prefill_chunk")
        if sum(size for _, size in chunks) > context.available_token_budget:
            raise PolicyValidationError("prefill_token_budget_exceeded")
        expected_chunks = tuple((req.uid, req.extend_len) for req in batch.reqs)
        if chunks != expected_chunks:
            raise PolicyValidationError("invalid_prefill_chunk_size")
    elif decision.prefill_chunk_sizes:
        raise PolicyValidationError("decode_decision_has_prefill_chunks")


@dataclass(frozen=True)
class PolicySelection:
    decision: SchedulingDecision
    policy_name: str
    fallback_reason: str | None = None


@dataclass
class PolicyController:
    active_policy: BaseSchedulingPolicy
    fallback_policy: BaseSchedulingPolicy = field(default_factory=UpstreamDefaultPolicy)
    failure_threshold: int = 3
    consecutive_failures: int = 0
    total_failures: int = 0
    fallback_count: int = 0
    circuit_open: bool = False

    @property
    def health(self) -> PolicyHealth:
        if self.circuit_open:
            return PolicyHealth.CIRCUIT_OPEN
        if self.consecutive_failures:
            return PolicyHealth.DEGRADED
        return self.active_policy.health

    def select(
        self,
        context: SchedulingContext,
        schedule_prefill: SchedulePrefill,
        schedule_decode: ScheduleDecode,
        *,
        rollback: Callable[[], None],
        terminal_uids: Iterable[int] = (),
        released_uids: Iterable[int] = (),
    ) -> PolicySelection:
        policy = self.fallback_policy if self.circuit_open else self.active_policy
        try:
            decision = policy.select(context, schedule_prefill, schedule_decode)
            policy.validate_decision(
                context,
                decision,
                terminal_uids=terminal_uids,
                released_uids=released_uids,
            )
        except Exception as exc:
            rollback()
            if policy is self.fallback_policy or policy.name == self.fallback_policy.name:
                raise UpstreamPolicyFatalError("upstream_default_failed") from exc
            self.total_failures += 1
            self.consecutive_failures += 1
            self.fallback_count += 1
            if self.consecutive_failures >= self.failure_threshold:
                self.circuit_open = True
            fallback_reason = (
                "invalid_decision"
                if isinstance(exc, PolicyValidationError)
                else "policy_exception"
            )
            try:
                decision = self.fallback_policy.select(
                    context, schedule_prefill, schedule_decode
                )
                self.fallback_policy.validate_decision(
                    context,
                    decision,
                    terminal_uids=terminal_uids,
                    released_uids=released_uids,
                )
            except Exception as fallback_exc:
                rollback()
                raise UpstreamPolicyFatalError(
                    "upstream_default_failed"
                ) from fallback_exc
            return PolicySelection(
                decision=decision,
                policy_name=self.fallback_policy.name,
                fallback_reason=fallback_reason,
            )

        if policy is self.active_policy:
            self.consecutive_failures = 0
        return PolicySelection(decision=decision, policy_name=policy.name)

    def on_request_cancelled(self, uid: int) -> None:
        try:
            self.active_policy.on_request_cancelled(uid)
        except Exception:
            self.total_failures += 1

    def on_request_finished(self, uid: int) -> None:
        try:
            self.active_policy.on_request_finished(uid)
        except Exception:
            self.total_failures += 1

    def manual_reset(self) -> None:
        self.circuit_open = False
        self.consecutive_failures = 0


@dataclass
class SchedulingMetrics:
    policy_name: str
    starvation_threshold_ns: int = 5_000_000_000
    decision_count: int = 0
    total_decision_latency_ns: int = 0
    max_decision_latency_ns: int = 0
    prefill_batch_count: int = 0
    decode_batch_count: int = 0
    idle_decision_count: int = 0
    total_batch_requests: int = 0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0
    max_batch_requests: int = 0
    max_waiting_requests: int = 0
    max_running_requests: int = 0
    max_waiting_time_ns: int = 0
    max_waiting_age_ms: float = 0.0
    fallback_count: int = 0
    policy_failure_count: int = 0
    cancelled_count: int = 0
    failed_count: int = 0
    last_step_id: int = 0
    fallback_reasons: Dict[str, int] = field(default_factory=dict)
    _starved_uids: set[int] = field(default_factory=set, repr=False)

    def record(
        self,
        context: SchedulingContext,
        decision: SchedulingDecision,
        *,
        fallback_reason: str | None = None,
        cancelled_count: int = 0,
        failed_count: int = 0,
        maximum_waiting_age_ms: float = 0.0,
    ) -> None:
        self.decision_count += 1
        self.last_step_id = context.step_id
        self.total_decision_latency_ns += decision.decision_latency_ns
        self.max_decision_latency_ns = max(
            self.max_decision_latency_ns, decision.decision_latency_ns
        )
        self.max_waiting_requests = max(
            self.max_waiting_requests, len(context.waiting_requests)
        )
        self.max_running_requests = max(
            self.max_running_requests, len(context.running_requests)
        )
        self.max_waiting_age_ms = max(
            self.max_waiting_age_ms, maximum_waiting_age_ms
        )
        self.cancelled_count = cancelled_count
        self.failed_count = failed_count
        if fallback_reason:
            self.fallback_count += 1
            self.fallback_reasons[fallback_reason] = (
                self.fallback_reasons.get(fallback_reason, 0) + 1
            )

        for req in context.waiting_requests:
            waiting_ns = max(0, context.current_timestamp_ns - req.enqueued_at_ns)
            self.max_waiting_time_ns = max(self.max_waiting_time_ns, waiting_ns)
            if waiting_ns >= self.starvation_threshold_ns:
                self._starved_uids.add(req.uid)

        batch = decision.batch
        if batch is None:
            self.idle_decision_count += 1
            return

        batch_size = len(batch.reqs)
        self.total_batch_requests += batch_size
        self.max_batch_requests = max(self.max_batch_requests, batch_size)
        if batch.is_prefill:
            self.prefill_batch_count += 1
            self.total_prefill_tokens += sum(size for _, size in decision.prefill_chunk_sizes)
        else:
            self.decode_batch_count += 1
            self.total_decode_tokens += batch_size

    def snapshot(self) -> Dict[str, int | float | str]:
        count = self.decision_count
        return {
            "policy": self.policy_name,
            "decision_count": count,
            "decision_latency_mean_us": (
                self.total_decision_latency_ns / count / 1_000 if count else 0.0
            ),
            "decision_latency_max_us": self.max_decision_latency_ns / 1_000,
            "prefill_batch_count": self.prefill_batch_count,
            "decode_batch_count": self.decode_batch_count,
            "idle_decision_count": self.idle_decision_count,
            "mean_batch_requests": (
                self.total_batch_requests
                / max(1, self.prefill_batch_count + self.decode_batch_count)
            ),
            "max_batch_requests": self.max_batch_requests,
            "prefill_tokens": self.total_prefill_tokens,
            "decode_tokens": self.total_decode_tokens,
            "max_waiting_requests": self.max_waiting_requests,
            "max_running_requests": self.max_running_requests,
            "max_waiting_time_ms": self.max_waiting_time_ns / 1_000_000,
            "max_waiting_age_ms": self.max_waiting_age_ms,
            "starvation_count": len(self._starved_uids),
            "last_step_id": self.last_step_id,
            "fallback_count": self.fallback_count,
            "fallback_reasons": dict(sorted(self.fallback_reasons.items())),
            "policy_failure_count": self.policy_failure_count,
            "cancelled_count": self.cancelled_count,
            "failed_count": self.failed_count,
        }

    def write_json(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(
            f".{output_path.name}.tmp.{os.getpid()}"
        )
        temporary_path.write_text(
            json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, output_path)
