from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from minisgl.core import Batch, Req, SamplingParams
from minisgl.kvcache import BaseCacheHandle
from minisgl.message import AbortMsg
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.lifecycle import LifecycleRegistry, RequestLifecycleState
from minisgl.scheduler.policy import PolicyController, UpstreamDefaultPolicy
from minisgl.scheduler.prefill import PrefillManager
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.utils import PendingReq
from minisgl.server.api_server import FrontendManager


class FakeCacheManager:
    def __init__(self) -> None:
        self.release_calls = 0
        self.shared_prefix_pages = 7

    def free_and_cache_finished_req(self, handle, input_ids, indices) -> None:
        assert handle.cached_len <= len(input_ids)
        self.release_calls += 1


class FakeHealthReporter:
    def __init__(self) -> None:
        self.steps = []

    def engine_step(self, step_id: int, **stats) -> None:
        self.steps.append(step_id)


class FakeEvent:
    def __init__(self) -> None:
        self.synchronized = False

    def synchronize(self) -> None:
        self.synchronized = True


def make_scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cache_manager = FakeCacheManager()
    scheduler.table_manager = TableManager(
        8, torch.zeros((8, 32), dtype=torch.int32)
    )
    scheduler.decode_manager = DecodeManager()
    scheduler.prefill_manager = PrefillManager(
        scheduler.cache_manager,
        scheduler.table_manager,
        scheduler.decode_manager,
    )
    scheduler.lifecycle_registry = LifecycleRegistry()
    scheduler.policy_controller = PolicyController(UpstreamDefaultPolicy())
    scheduler._selected_reqs = {}
    scheduler._inflight_reqs = {}
    scheduler._cancelled_uids = set()
    scheduler._released_uids = set()
    scheduler.page_table = scheduler.table_manager.page_table
    scheduler.finished_reqs = set()
    scheduler.engine = SimpleNamespace(max_seq_len=32)
    scheduler.health_reporter = FakeHealthReporter()
    scheduler.step_id = 3
    scheduler.sent_replies = []
    scheduler.send_result = scheduler.sent_replies.append
    return scheduler


def make_req(scheduler: Scheduler, uid: int, cached_len: int = 1) -> Req:
    table_idx = scheduler.table_manager.allocate()
    return Req(
        input_ids=torch.tensor([10, 11], dtype=torch.int32),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=4,
        uid=uid,
        cache_handle=BaseCacheHandle(cached_len=cached_len),
        sampling_params=SamplingParams(max_tokens=4),
    )


def add_lifecycle(scheduler: Scheduler, uid: int, state: RequestLifecycleState):
    lifecycle = scheduler.lifecycle_registry.create(
        uid, input_tokens=2, requested_output_tokens=4
    )
    transitions = [
        RequestLifecycleState.WAITING,
        RequestLifecycleState.PREFILL_SELECTED,
        RequestLifecycleState.PREFILL_RUNNING,
        RequestLifecycleState.DECODING,
    ]
    for transition in transitions:
        if lifecycle.state == state:
            break
        lifecycle.transition(transition)
    assert lifecycle.state == state
    return lifecycle


def test_waiting_cancellation_removes_request_without_allocating_kv() -> None:
    scheduler = make_scheduler()
    lifecycle = add_lifecycle(scheduler, 1, RequestLifecycleState.WAITING)
    scheduler.prefill_manager.pending_list.append(
        PendingReq(
            1,
            torch.tensor([1, 2], dtype=torch.int32),
            SamplingParams(max_tokens=4),
        )
    )

    assert scheduler.abort_req(1)

    assert lifecycle.state == RequestLifecycleState.CANCELLED
    assert not scheduler.prefill_manager.pending_list
    assert scheduler.cache_manager.release_calls == 0
    assert len(scheduler.sent_replies) == 1


def test_selected_before_prefill_cancellation_releases_table_once() -> None:
    scheduler = make_scheduler()
    lifecycle = add_lifecycle(
        scheduler, 2, RequestLifecycleState.PREFILL_SELECTED
    )
    req = make_req(scheduler, 2)
    scheduler._selected_reqs[2] = req

    assert scheduler.abort_req(2)

    assert lifecycle.state == RequestLifecycleState.CANCELLED
    assert scheduler.cache_manager.release_calls == 1
    assert not scheduler.table_manager.owns(req.table_idx)


def test_running_decode_cancellation_removes_only_target() -> None:
    scheduler = make_scheduler()
    target_lifecycle = add_lifecycle(
        scheduler, 3, RequestLifecycleState.DECODING
    )
    survivor_lifecycle = add_lifecycle(
        scheduler, 4, RequestLifecycleState.DECODING
    )
    target = make_req(scheduler, 3)
    survivor = make_req(scheduler, 4)
    scheduler.decode_manager.add_reqs([target, survivor])

    assert scheduler.abort_req(3)

    assert target_lifecycle.state == RequestLifecycleState.CANCELLED
    assert survivor_lifecycle.state == RequestLifecycleState.DECODING
    assert not scheduler.decode_manager.contains_uid(3)
    assert scheduler.decode_manager.contains_uid(4)
    assert scheduler.table_manager.owns(survivor.table_idx)


def test_inflight_decode_cancellation_defers_release_until_event() -> None:
    scheduler = make_scheduler()
    lifecycle = add_lifecycle(
        scheduler, 5, RequestLifecycleState.DECODING
    )
    req = make_req(scheduler, 5)
    scheduler.decode_manager.add_reqs([req])
    scheduler._inflight_reqs[5] = req

    assert scheduler.abort_req(5)
    assert scheduler.cache_manager.release_calls == 0
    assert lifecycle.state == RequestLifecycleState.CANCELLATION_REQUESTED

    event = FakeEvent()
    batch = Batch(reqs=[req], phase="decode")
    scheduler._process_last_data(
        (SimpleNamespace(batch=batch), (None, torch.tensor([9]), event)),
        None,
    )

    assert event.synchronized
    assert lifecycle.state == RequestLifecycleState.CANCELLED
    assert scheduler.cache_manager.release_calls == 1
    assert not scheduler._inflight_reqs


def test_duplicate_cancellation_is_idempotent() -> None:
    scheduler = make_scheduler()
    add_lifecycle(scheduler, 6, RequestLifecycleState.WAITING)

    assert scheduler.abort_req(6)
    assert not scheduler.abort_req(6)
    assert scheduler.cache_manager.release_calls == 0
    assert len(scheduler.sent_replies) == 1


def test_cancel_does_not_clear_shared_prefix_cache() -> None:
    scheduler = make_scheduler()
    add_lifecycle(scheduler, 7, RequestLifecycleState.DECODING)
    req = make_req(scheduler, 7)
    scheduler.decode_manager.add_reqs([req])
    shared_before = scheduler.cache_manager.shared_prefix_pages

    scheduler.abort_req(7)

    assert scheduler.cache_manager.shared_prefix_pages == shared_before
    assert scheduler.cache_manager.release_calls == 1


def test_cancelled_request_cannot_remain_waiting_or_running() -> None:
    scheduler = make_scheduler()
    add_lifecycle(scheduler, 8, RequestLifecycleState.WAITING)
    scheduler.prefill_manager.pending_list.append(
        PendingReq(
            8,
            torch.tensor([1], dtype=torch.int32),
            SamplingParams(max_tokens=1),
        )
    )

    scheduler.abort_req(8)
    scheduler._assert_state_consistency()

    assert not scheduler.prefill_manager.contains_uid(8)
    assert not scheduler.decode_manager.contains_uid(8)


def test_table_manager_rejects_double_free() -> None:
    manager = TableManager(2, torch.zeros((2, 4), dtype=torch.int32))
    slot = manager.allocate()
    manager.free(slot)

    with pytest.raises(RuntimeError, match="not owned"):
        manager.free(slot)


class FakeAsyncQueue:
    def __init__(self) -> None:
        self.items = []

    async def put(self, item) -> None:
        self.items.append(item)

    def stop(self) -> None:
        pass


class ReadySupervisor:
    def snapshot(self):
        return SimpleNamespace(
            as_dict=lambda: {
                "ready": True,
                "accepting_requests": True,
                "fatal_error": None,
            }
        )

    def stop(self) -> None:
        pass


def test_client_disconnect_sends_one_abort_and_cleans_frontend_state() -> None:
    send_queue = FakeAsyncQueue()
    manager = FrontendManager(
        config=SimpleNamespace(cancel_on_disconnect=True),
        send_tokenizer=send_queue,
        recv_tokenizer=SimpleNamespace(stop=lambda: None),
        initialized=True,
        supervisor=ReadySupervisor(),
    )
    uid = manager.new_user()

    class DisconnectedRequest:
        async def is_disconnected(self) -> bool:
            return True

    async def generator():
        yield b"data"

    async def consume() -> None:
        with pytest.raises(asyncio.CancelledError):
            async for _ in manager.stream_with_cancellation(
                generator(), DisconnectedRequest(), uid
            ):
                pass

    asyncio.run(consume())

    assert len(send_queue.items) == 1
    assert isinstance(send_queue.items[0], AbortMsg)
    assert uid not in manager.ack_map
    assert uid not in manager.event_map


def test_frontend_duplicate_abort_is_idempotent() -> None:
    send_queue = FakeAsyncQueue()
    manager = FrontendManager(
        config=SimpleNamespace(cancel_on_disconnect=True),
        send_tokenizer=send_queue,
        recv_tokenizer=SimpleNamespace(stop=lambda: None),
        initialized=True,
        supervisor=ReadySupervisor(),
    )
    uid = manager.new_user()

    async def cancel_twice():
        return await manager.abort_user(uid), await manager.abort_user(uid)

    first, second = asyncio.run(cancel_twice())

    assert first
    assert not second
    assert len(send_queue.items) == 1
