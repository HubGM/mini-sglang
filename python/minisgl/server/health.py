from __future__ import annotations

import enum
import queue
import threading
import time
from dataclasses import asdict, dataclass
from multiprocessing.queues import Queue
from typing import Iterable, Protocol


class ProcessLike(Protocol):
    def is_alive(self) -> bool: ...


class HealthEventKind(str, enum.Enum):
    MODEL_LOADED = "model_loaded"
    SCHEDULER_READY = "scheduler_ready"
    TOKENIZER_READY = "tokenizer_ready"
    HEARTBEAT = "heartbeat"
    ENGINE_STEP = "engine_step"
    FATAL = "fatal"
    STOPPED = "stopped"


@dataclass(frozen=True)
class HealthEvent:
    kind: HealthEventKind
    source: str
    timestamp_ns: int
    step_id: int | None = None
    reason: str | None = None
    waiting_count: int | None = None
    running_count: int | None = None
    cancelled_count: int | None = None
    failed_count: int | None = None


@dataclass(frozen=True)
class ReadinessSnapshot:
    ready: bool
    api_process_alive: bool
    tokenizer_process_alive: bool
    scheduler_process_alive: bool
    scheduler_event_loop_alive: bool
    model_loaded: bool
    last_successful_engine_step_ns: int | None
    fatal_error: str | None
    accepting_requests: bool
    last_step_id: int | None
    waiting_count: int
    running_count: int
    cancelled_count: int
    failed_count: int

    def as_dict(self) -> dict:
        return asdict(self)


class HealthReporter:
    def __init__(
        self,
        event_queue: Queue | None,
        source: str,
        min_interval_s: float = 0.0,
    ) -> None:
        self.event_queue = event_queue
        self.source = source
        self.min_interval_ns = int(min_interval_s * 1_000_000_000)
        self._last_progress_event_ns = 0
        self._last_progress_kind: HealthEventKind | None = None

    def emit(
        self,
        kind: HealthEventKind,
        *,
        step_id: int | None = None,
        reason: str | None = None,
        waiting_count: int | None = None,
        running_count: int | None = None,
        cancelled_count: int | None = None,
        failed_count: int | None = None,
    ) -> None:
        if self.event_queue is None:
            return
        self.event_queue.put(
            HealthEvent(
                kind=kind,
                source=self.source,
                timestamp_ns=time.monotonic_ns(),
                step_id=step_id,
                reason=reason,
                waiting_count=waiting_count,
                running_count=running_count,
                cancelled_count=cancelled_count,
                failed_count=failed_count,
            )
        )

    def _progress(
        self,
        kind: HealthEventKind,
        step_id: int,
        *,
        force: bool = False,
        waiting_count: int | None = None,
        running_count: int | None = None,
        cancelled_count: int | None = None,
        failed_count: int | None = None,
    ) -> None:
        now_ns = time.monotonic_ns()
        if (
            not force
            and now_ns - self._last_progress_event_ns < self.min_interval_ns
            and not (
                kind == HealthEventKind.ENGINE_STEP
                and self._last_progress_kind == HealthEventKind.HEARTBEAT
            )
        ):
            return
        self._last_progress_event_ns = now_ns
        self._last_progress_kind = kind
        self.emit(
            kind,
            step_id=step_id,
            waiting_count=waiting_count,
            running_count=running_count,
            cancelled_count=cancelled_count,
            failed_count=failed_count,
        )

    def heartbeat(self, step_id: int, **stats) -> None:
        self._progress(HealthEventKind.HEARTBEAT, step_id, **stats)

    def engine_step(self, step_id: int, **stats) -> None:
        self._progress(HealthEventKind.ENGINE_STEP, step_id, **stats)

    def fatal(self, reason: str) -> None:
        self.emit(HealthEventKind.FATAL, reason=reason)


class HealthStateAggregator:
    def __init__(self, heartbeat_timeout_s: float = 5.0) -> None:
        self.heartbeat_timeout_ns = int(heartbeat_timeout_s * 1_000_000_000)
        self.api_process_alive = True
        self.model_loaded = False
        self.scheduler_ready = False
        self.tokenizer_ready = False
        self.last_heartbeat_ns: int | None = None
        self.last_successful_engine_step_ns: int | None = None
        self.fatal_error: str | None = None
        self.accepting_requests = False
        self.scheduler_process_alive = False
        self.tokenizer_process_alive = False
        self.last_step_id: int | None = None
        self.waiting_count = 0
        self.running_count = 0
        self.cancelled_count = 0
        self.failed_count = 0
        self._lock = threading.Lock()

    def apply(self, event: HealthEvent) -> None:
        with self._lock:
            if self.fatal_error is not None:
                return
            if event.kind == HealthEventKind.MODEL_LOADED:
                self.model_loaded = True
            elif event.kind == HealthEventKind.SCHEDULER_READY:
                self.scheduler_ready = True
                self.last_heartbeat_ns = event.timestamp_ns
            elif event.kind == HealthEventKind.TOKENIZER_READY:
                self.tokenizer_ready = True
            elif event.kind == HealthEventKind.HEARTBEAT:
                self.last_heartbeat_ns = event.timestamp_ns
            elif event.kind == HealthEventKind.ENGINE_STEP:
                self.last_heartbeat_ns = event.timestamp_ns
                self.last_successful_engine_step_ns = event.timestamp_ns
            elif event.kind == HealthEventKind.FATAL:
                self.fatal_error = event.reason or "scheduler_fatal"
                self.accepting_requests = False
            elif event.kind == HealthEventKind.STOPPED:
                self.accepting_requests = False
            if event.step_id is not None:
                self.last_step_id = event.step_id
            for field_name in (
                "waiting_count",
                "running_count",
                "cancelled_count",
                "failed_count",
            ):
                value = getattr(event, field_name)
                if value is not None:
                    setattr(self, field_name, value)

    def update_process_state(
        self,
        scheduler_processes: Iterable[ProcessLike],
        tokenizer_processes: Iterable[ProcessLike],
    ) -> None:
        scheduler = tuple(scheduler_processes)
        tokenizer = tuple(tokenizer_processes)
        with self._lock:
            self.scheduler_process_alive = bool(scheduler) and all(
                process.is_alive() for process in scheduler
            )
            self.tokenizer_process_alive = bool(tokenizer) and all(
                process.is_alive() for process in tokenizer
            )
            if self.scheduler_ready and not self.scheduler_process_alive:
                self.fatal_error = self.fatal_error or "scheduler_process_exited"
            if self.tokenizer_ready and not self.tokenizer_process_alive:
                self.fatal_error = self.fatal_error or "tokenizer_process_exited"
            if self.fatal_error:
                self.accepting_requests = False

    def snapshot(self, now_ns: int | None = None) -> ReadinessSnapshot:
        now_ns = now_ns or time.monotonic_ns()
        with self._lock:
            event_loop_alive = (
                self.last_heartbeat_ns is not None
                and now_ns - self.last_heartbeat_ns <= self.heartbeat_timeout_ns
            )
            ready = (
                self.api_process_alive
                and self.tokenizer_process_alive
                and self.scheduler_process_alive
                and event_loop_alive
                and self.model_loaded
                and self.scheduler_ready
                and self.tokenizer_ready
                and self.fatal_error is None
            )
            self.accepting_requests = ready
            return ReadinessSnapshot(
                ready=ready,
                api_process_alive=self.api_process_alive,
                tokenizer_process_alive=self.tokenizer_process_alive,
                scheduler_process_alive=self.scheduler_process_alive,
                scheduler_event_loop_alive=event_loop_alive,
                model_loaded=self.model_loaded,
                last_successful_engine_step_ns=self.last_successful_engine_step_ns,
                fatal_error=self.fatal_error,
                accepting_requests=self.accepting_requests,
                last_step_id=self.last_step_id,
                waiting_count=self.waiting_count,
                running_count=self.running_count,
                cancelled_count=self.cancelled_count,
                failed_count=self.failed_count,
            )


class BackendSupervisor:
    def __init__(
        self,
        *,
        event_queue: Queue,
        scheduler_processes: Iterable[ProcessLike],
        tokenizer_processes: Iterable[ProcessLike],
        heartbeat_timeout_s: float,
    ) -> None:
        self.event_queue = event_queue
        self.scheduler_processes = tuple(scheduler_processes)
        self.tokenizer_processes = tuple(tokenizer_processes)
        self.state = HealthStateAggregator(heartbeat_timeout_s)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._monitor,
            name="minisgl-backend-health",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _monitor(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self.event_queue.get(timeout=0.1)
            except queue.Empty:
                event = None
            if event is not None:
                self.state.apply(event)
            self.state.update_process_state(
                self.scheduler_processes, self.tokenizer_processes
            )

    def snapshot(self) -> ReadinessSnapshot:
        self.state.update_process_state(
            self.scheduler_processes, self.tokenizer_processes
        )
        return self.state.snapshot()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
