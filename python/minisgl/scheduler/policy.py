from __future__ import annotations

import json
import os
import enum
import hashlib
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple

from minisgl.core import Batch, Req

from .utils import PendingReq

SchedulePrefill = Callable[
    [int, Sequence[int] | None, int | None], Batch | None
]
ScheduleDecode = Callable[[int | None, Sequence[int] | None], Batch | None]


@dataclass(frozen=True)
class SchedulingPolicyConfig:
    max_step_tokens: int = 2048
    max_prefill_chunk_tokens: int = 512
    decode_reserve_ratio: float = 0.5
    max_consecutive_prefill_steps: int = 1
    default_ttft_deadline_ms: float = 200.0
    default_tpot_deadline_ms: float = 50.0
    default_e2e_deadline_ms: float = 1200.0
    initial_prefill_ms_per_token: float = 0.15
    initial_decode_step_ms: float = 25.0
    service_ewma_alpha: float = 0.2
    max_wait_ms: float = 400.0
    aging_start_ms: float = 100.0
    aging_rate: float = 1.0
    min_prefill_budget_per_step: int = 256
    max_decode_only_steps: int = 2
    prefill_urgent_threshold_ms: float = 150.0
    hard_max_wait_ms: float = 300.0
    min_decode_reserve_ratio: float = 0.25
    max_decode_reserve_ratio: float = 0.65


DUAL_SLO_CONFIG_FIELDS = (
    "max_consecutive_prefill_steps",
    "min_prefill_budget_per_step",
    "max_decode_only_steps",
    "prefill_urgent_threshold_ms",
    "hard_max_wait_ms",
    "min_decode_reserve_ratio",
    "max_decode_reserve_ratio",
)


def dual_slo_config_hash(config: SchedulingPolicyConfig) -> str:
    payload = {
        field: getattr(config, field) for field in DUAL_SLO_CONFIG_FIELDS
    }
    encoded = json.dumps(
        payload, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RequestSchedulingInfo:
    uid: int
    enqueue_time_ns: int
    first_scheduled_time_ns: int | None
    first_token_time_ns: int | None
    input_tokens: int
    remaining_input_tokens: int
    requested_output_tokens: int
    generated_tokens: int
    ttft_deadline_ms: float
    e2e_deadline_ms: float
    tpot_deadline_ms: float = 50.0
    last_token_time_ns: int | None = None
    is_chunk_continuation: bool = False

    def age_ms(self, now_ns: int) -> float:
        return max(0, now_ns - self.enqueue_time_ns) / 1_000_000

    @property
    def remaining_output_tokens(self) -> int:
        return max(0, self.requested_output_tokens - self.generated_tokens)


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
    request_info: Tuple[RequestSchedulingInfo, ...] = ()


@dataclass(frozen=True)
class SchedulingDecision:
    batch: Batch | None
    selected_prefill_requests: Tuple[Req, ...] = ()
    selected_decode_requests: Tuple[Req, ...] = ()
    prefill_chunk_sizes: Tuple[Tuple[int, int], ...] = ()
    preempted_request_uids: Tuple[int, ...] = ()
    reason_codes: Tuple[str, ...] = ()
    decision_latency_ns: int = 0
    token_budget: int = 0
    token_budget_used: int = 0
    minimum_slack_ms: float | None = None
    urgent_request_count: int = 0
    request_slack_ms: Tuple[Tuple[int, float], ...] = ()
    pre_first_token_count: int = 0
    post_first_token_count: int = 0
    prefill_urgent_request_count: int = 0
    hard_urgent_request_count: int = 0
    consecutive_decode_only_steps: int = 0
    dynamic_decode_reserve_ratio: float | None = None
    chunk_continuation_deferred: bool = False


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

    def on_step_completed(
        self,
        *,
        phase: str,
        elapsed_ms: float,
        prefill_tokens: int,
        decode_tokens: int,
    ) -> None:
        _ = (phase, elapsed_ms, prefill_tokens, decode_tokens)


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
            token_budget=context.available_token_budget,
            token_budget_used=(
                sum(size for _, size in chunk_sizes)
                if batch is not None and batch.is_prefill
                else len(selected_decode)
            ),
        )


def create_scheduling_policy(
    name: str,
    config: SchedulingPolicyConfig | None = None,
) -> BaseSchedulingPolicy:
    if name == UpstreamDefaultPolicy.name:
        return UpstreamDefaultPolicy()
    from .advanced_policy import (
        DeadlineAgingPolicy,
        DeadlineAgingV2Policy,
        DeadlineAwarePolicy,
        TokenBudgetPolicy,
    )

    policies = {
        TokenBudgetPolicy.name: TokenBudgetPolicy,
        DeadlineAwarePolicy.name: DeadlineAwarePolicy,
        DeadlineAgingPolicy.name: DeadlineAgingPolicy,
        "deadline_aging": DeadlineAgingPolicy,
        DeadlineAgingV2Policy.name: DeadlineAgingV2Policy,
    }
    if policy_type := policies.get(name):
        return policy_type(config or SchedulingPolicyConfig())
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

    actual_used = (
        sum(size for _, size in decision.prefill_chunk_sizes)
        if batch.is_prefill
        else len(decision.selected_decode_requests)
    )
    if decision.token_budget_used != actual_used:
        raise PolicyValidationError("token_budget_usage_mismatch")
    if decision.token_budget < 0 or actual_used > decision.token_budget:
        raise PolicyValidationError("step_token_budget_exceeded")


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

    def on_step_completed(
        self,
        *,
        phase: str,
        elapsed_ms: float,
        prefill_tokens: int,
        decode_tokens: int,
    ) -> None:
        try:
            self.active_policy.on_step_completed(
                phase=phase,
                elapsed_ms=elapsed_ms,
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
            )
        except Exception:
            self.total_failures += 1

    def manual_reset(self) -> None:
        self.circuit_open = False
        self.consecutive_failures = 0


@dataclass
class SchedulingMetrics:
    policy_name: str
    policy_config_hash: str | None = None
    starvation_threshold_ns: int = 400_000_000
    request_sample_rate: float = 0.1
    max_step_records: int = 10_000
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
    total_token_budget: int = 0
    total_token_budget_used: int = 0
    prefill_chunk_count: int = 0
    minimum_slack_ms: float | None = None
    max_urgent_requests: int = 0
    max_prefill_urgent_requests: int = 0
    max_hard_urgent_requests: int = 0
    max_consecutive_decode_only_steps: int = 0
    dynamic_decode_reserve_total: float = 0.0
    dynamic_decode_reserve_count: int = 0
    chunk_continuation_deferred_count: int = 0
    step_records: list[dict[str, Any]] = field(default_factory=list)
    request_samples: list[dict[str, Any]] = field(default_factory=list)
    dropped_step_records: int = 0
    _decision_latencies_us: list[float] = field(default_factory=list, repr=False)
    _recorded_request_uids: set[int] = field(default_factory=set, repr=False)
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
        policy_name: str | None = None,
    ) -> None:
        self.decision_count += 1
        self.last_step_id = context.step_id
        self.total_decision_latency_ns += decision.decision_latency_ns
        self.max_decision_latency_ns = max(
            self.max_decision_latency_ns, decision.decision_latency_ns
        )
        self._decision_latencies_us.append(decision.decision_latency_ns / 1_000)
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
        self.prefill_chunk_count += len(decision.prefill_chunk_sizes)
        if decision.minimum_slack_ms is not None:
            self.minimum_slack_ms = (
                decision.minimum_slack_ms
                if self.minimum_slack_ms is None
                else min(self.minimum_slack_ms, decision.minimum_slack_ms)
            )
        self.max_urgent_requests = max(
            self.max_urgent_requests, decision.urgent_request_count
        )
        self.max_prefill_urgent_requests = max(
            self.max_prefill_urgent_requests,
            decision.prefill_urgent_request_count,
        )
        self.max_hard_urgent_requests = max(
            self.max_hard_urgent_requests,
            decision.hard_urgent_request_count,
        )
        self.max_consecutive_decode_only_steps = max(
            self.max_consecutive_decode_only_steps,
            decision.consecutive_decode_only_steps,
        )
        if decision.dynamic_decode_reserve_ratio is not None:
            self.dynamic_decode_reserve_total += (
                decision.dynamic_decode_reserve_ratio
            )
            self.dynamic_decode_reserve_count += 1
        if decision.chunk_continuation_deferred:
            self.chunk_continuation_deferred_count += 1
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

        step_record = {
            "step_id": context.step_id,
            "timestamp_ns": context.current_timestamp_ns,
            "policy": policy_name or self.policy_name,
            "waiting_count": len(context.waiting_requests),
            "running_count": len(context.running_requests),
            "selected_prefill_count": len(decision.selected_prefill_requests),
            "selected_prefill_tokens": sum(
                size for _, size in decision.prefill_chunk_sizes
            ),
            "selected_decode_count": len(decision.selected_decode_requests),
            "selected_decode_tokens": len(decision.selected_decode_requests),
            "token_budget": decision.token_budget,
            "token_budget_used": decision.token_budget_used,
            "prefill_chunk_size": (
                max((size for _, size in decision.prefill_chunk_sizes), default=0)
            ),
            "max_waiting_age_ms": maximum_waiting_age_ms,
            "minimum_slack_ms": decision.minimum_slack_ms,
            "urgent_request_count": decision.urgent_request_count,
            "prefill_urgent_request_count": (
                decision.prefill_urgent_request_count
            ),
            "hard_urgent_request_count": decision.hard_urgent_request_count,
            "pre_first_token_count": decision.pre_first_token_count,
            "post_first_token_count": decision.post_first_token_count,
            "consecutive_decode_only_steps": (
                decision.consecutive_decode_only_steps
            ),
            "dynamic_decode_reserve_ratio": (
                decision.dynamic_decode_reserve_ratio
            ),
            "chunk_continuation_deferred": (
                decision.chunk_continuation_deferred
            ),
            "reason_codes": list(decision.reason_codes),
            "scheduler_decision_us": decision.decision_latency_ns / 1_000,
        }
        if len(self.step_records) < self.max_step_records:
            self.step_records.append(step_record)
        else:
            self.dropped_step_records += 1

        batch = decision.batch
        if batch is None:
            self.idle_decision_count += 1
            return

        self.total_token_budget += decision.token_budget
        self.total_token_budget_used += decision.token_budget_used
        batch_size = len(batch.reqs)
        self.total_batch_requests += batch_size
        self.max_batch_requests = max(self.max_batch_requests, batch_size)
        if batch.is_prefill:
            self.prefill_batch_count += 1
            self.total_prefill_tokens += sum(size for _, size in decision.prefill_chunk_sizes)
        else:
            self.decode_batch_count += 1
            self.total_decode_tokens += batch_size

    def record_request(self, lifecycle) -> None:
        if lifecycle.uid in self._recorded_request_uids:
            return
        self._recorded_request_uids.add(lifecycle.uid)
        if self.request_sample_rate <= 0:
            return
        sample_bucket = (lifecycle.uid * 2654435761) % 10_000
        if sample_bucket >= int(self.request_sample_rate * 10_000):
            return
        self.request_samples.append(lifecycle.terminal_snapshot())

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        rank = (len(ordered) - 1) * quantile
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)

    def snapshot(self) -> Dict[str, Any]:
        count = self.decision_count
        return {
            "policy": self.policy_name,
            "policy_config_hash": self.policy_config_hash,
            "decision_count": count,
            "decision_latency_mean_us": (
                self.total_decision_latency_ns / count / 1_000 if count else 0.0
            ),
            "decision_latency_max_us": self.max_decision_latency_ns / 1_000,
            "decision_latency_p50_us": self._percentile(
                self._decision_latencies_us, 0.50
            ),
            "decision_latency_p95_us": self._percentile(
                self._decision_latencies_us, 0.95
            ),
            "decision_latency_p99_us": self._percentile(
                self._decision_latencies_us, 0.99
            ),
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
            "token_budget_utilization": (
                self.total_token_budget_used / self.total_token_budget
                if self.total_token_budget
                else 0.0
            ),
            "average_prefill_chunk_tokens": (
                self.total_prefill_tokens / self.prefill_chunk_count
                if self.prefill_chunk_count
                else 0.0
            ),
            "minimum_slack_ms": self.minimum_slack_ms,
            "max_urgent_requests": self.max_urgent_requests,
            "max_prefill_urgent_requests": self.max_prefill_urgent_requests,
            "max_hard_urgent_requests": self.max_hard_urgent_requests,
            "max_consecutive_decode_only_steps": (
                self.max_consecutive_decode_only_steps
            ),
            "mean_dynamic_decode_reserve_ratio": (
                self.dynamic_decode_reserve_total
                / self.dynamic_decode_reserve_count
                if self.dynamic_decode_reserve_count
                else None
            ),
            "chunk_continuation_deferred_count": (
                self.chunk_continuation_deferred_count
            ),
            "step_records": self.step_records,
            "dropped_step_records": self.dropped_step_records,
            "request_samples": self.request_samples,
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
