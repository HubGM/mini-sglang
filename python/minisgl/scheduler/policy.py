from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Tuple

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

    @abstractmethod
    def select(
        self,
        context: SchedulingContext,
        schedule_prefill: SchedulePrefill,
        schedule_decode: ScheduleDecode,
    ) -> SchedulingDecision:
        """Select the next engine batch without executing model work."""


class UpstreamDefaultPolicy(BaseSchedulingPolicy):
    """Preserve the pinned upstream prefill-first, decode-fallback behavior."""

    name = "upstream_default"

    def select(
        self,
        context: SchedulingContext,
        schedule_prefill: SchedulePrefill,
        schedule_decode: ScheduleDecode,
    ) -> SchedulingDecision:
        started_ns = time.perf_counter_ns()
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
            decision_latency_ns=time.perf_counter_ns() - started_ns,
        )


def create_scheduling_policy(name: str) -> BaseSchedulingPolicy:
    if name == UpstreamDefaultPolicy.name:
        return UpstreamDefaultPolicy()
    raise ValueError(f"Unsupported scheduling policy: {name}")


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
    _starved_uids: set[int] = field(default_factory=set, repr=False)

    def record(self, context: SchedulingContext, decision: SchedulingDecision) -> None:
        self.decision_count += 1
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
            "starvation_count": len(self._starved_uids),
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
