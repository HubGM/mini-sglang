from __future__ import annotations

import asyncio
import queue
import time
from types import SimpleNamespace

from minisgl.server.health import (
    BackendSupervisor,
    HealthEvent,
    HealthEventKind,
    HealthReporter,
    HealthStateAggregator,
)
from minisgl.server.api_server import FrontendManager, FrontendRequestState


class FakeProcess:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive

    def is_alive(self) -> bool:
        return self.alive


def event(
    kind: HealthEventKind,
    *,
    timestamp_ns: int = 1_000_000_000,
    reason: str | None = None,
) -> HealthEvent:
    return HealthEvent(
        kind=kind,
        source="test",
        timestamp_ns=timestamp_ns,
        reason=reason,
    )


def ready_state() -> tuple[HealthStateAggregator, FakeProcess, FakeProcess]:
    state = HealthStateAggregator(heartbeat_timeout_s=1.0)
    scheduler = FakeProcess()
    tokenizer = FakeProcess()
    state.apply(event(HealthEventKind.MODEL_LOADED))
    state.apply(event(HealthEventKind.SCHEDULER_READY))
    state.apply(event(HealthEventKind.TOKENIZER_READY))
    state.apply(event(HealthEventKind.HEARTBEAT))
    state.update_process_state([scheduler], [tokenizer])
    return state, scheduler, tokenizer


def test_readiness_requires_all_process_and_model_signals() -> None:
    state, _, _ = ready_state()

    snapshot = state.snapshot(now_ns=1_500_000_000)

    assert snapshot.ready
    assert snapshot.accepting_requests
    assert snapshot.scheduler_event_loop_alive


def test_scheduler_heartbeat_timeout_disables_readiness() -> None:
    state, _, _ = ready_state()

    snapshot = state.snapshot(now_ns=2_100_000_001)

    assert not snapshot.ready
    assert not snapshot.scheduler_event_loop_alive


def test_scheduler_child_exit_sets_fatal_error() -> None:
    state, scheduler, tokenizer = ready_state()
    scheduler.alive = False

    state.update_process_state([scheduler], [tokenizer])
    snapshot = state.snapshot(now_ns=1_100_000_000)

    assert not snapshot.ready
    assert snapshot.fatal_error == "scheduler_process_exited"


def test_tokenizer_child_exit_sets_fatal_error() -> None:
    state, scheduler, tokenizer = ready_state()
    tokenizer.alive = False

    state.update_process_state([scheduler], [tokenizer])
    snapshot = state.snapshot(now_ns=1_100_000_000)

    assert not snapshot.ready
    assert snapshot.fatal_error == "tokenizer_process_exited"


def test_fatal_ipc_is_irreversible_without_restart() -> None:
    state, scheduler, tokenizer = ready_state()
    state.apply(
        event(
            HealthEventKind.FATAL,
            timestamp_ns=1_100_000_000,
            reason="engine_forward_exception",
        )
    )
    state.apply(event(HealthEventKind.HEARTBEAT, timestamp_ns=1_200_000_000))
    state.update_process_state([scheduler], [tokenizer])

    snapshot = state.snapshot(now_ns=1_200_000_000)

    assert not snapshot.ready
    assert snapshot.fatal_error == "engine_forward_exception"
    assert not snapshot.accepting_requests


def test_engine_step_updates_last_successful_timestamp() -> None:
    state, _, _ = ready_state()
    state.apply(event(HealthEventKind.ENGINE_STEP, timestamp_ns=1_250_000_000))

    snapshot = state.snapshot(now_ns=1_300_000_000)

    assert snapshot.last_successful_engine_step_ns == 1_250_000_000


def test_health_reporter_uses_low_cardinality_reason() -> None:
    event_queue = queue.Queue()
    reporter = HealthReporter(event_queue, source="scheduler-0")

    reporter.fatal("engine_forward_exception")
    reported = event_queue.get_nowait()

    assert reported.kind == HealthEventKind.FATAL
    assert reported.reason == "engine_forward_exception"
    assert not hasattr(reported, "request_id")


def test_backend_supervisor_detects_mock_child_exit() -> None:
    event_queue = queue.Queue()
    scheduler = FakeProcess()
    tokenizer = FakeProcess()
    supervisor = BackendSupervisor(
        event_queue=event_queue,
        scheduler_processes=[scheduler],
        tokenizer_processes=[tokenizer],
        heartbeat_timeout_s=1.0,
    )
    supervisor.start()
    reporter = HealthReporter(event_queue, source="test")
    reporter.emit(HealthEventKind.MODEL_LOADED)
    reporter.emit(HealthEventKind.SCHEDULER_READY)
    reporter.emit(HealthEventKind.TOKENIZER_READY)
    reporter.heartbeat(0)
    time.sleep(0.15)

    assert supervisor.snapshot().ready
    scheduler.alive = False
    time.sleep(0.15)

    snapshot = supervisor.snapshot()
    supervisor.stop()
    assert not snapshot.ready
    assert snapshot.fatal_error == "scheduler_process_exited"


def test_readiness_snapshot_has_no_high_cardinality_request_fields() -> None:
    state, _, _ = ready_state()
    keys = set(state.snapshot(now_ns=1_100_000_000).as_dict())

    assert "request_id" not in keys
    assert "uid" not in keys
    assert "prompt" not in keys


def test_frontend_fails_inflight_request_after_scheduler_fatal() -> None:
    class FatalSupervisor:
        def snapshot(self):
            return SimpleNamespace(
                as_dict=lambda: {
                    "ready": False,
                    "accepting_requests": False,
                    "fatal_error": "scheduler_process_exited",
                }
            )

    manager = FrontendManager(
        config=SimpleNamespace(cancel_on_disconnect=True),
        send_tokenizer=SimpleNamespace(stop=lambda: None),
        recv_tokenizer=SimpleNamespace(stop=lambda: None),
        initialized=True,
        supervisor=FatalSupervisor(),
    )
    manager.ack_map[7] = []
    manager.event_map[7] = asyncio.Event()
    manager.request_states[7] = FrontendRequestState()

    async def observe_failure() -> None:
        monitor = asyncio.create_task(manager.monitor_health())
        stream = manager.wait_for_ack(7)
        try:
            reply = await asyncio.wait_for(stream.__anext__(), timeout=0.5)
            assert reply.finished
            assert reply.terminal_reason == "failed"
            assert reply.error == "scheduler_process_exited"
        finally:
            await stream.aclose()
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass

    asyncio.run(observe_failure())
