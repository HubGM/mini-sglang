from __future__ import annotations

from types import SimpleNamespace

import torch

from minisgl.core import Req, SamplingParams
from minisgl.kvcache import BaseCacheHandle
from minisgl.kvcache.radix_manager import RadixCacheManager, RadixTreeNode
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import ChunkedReq, PrefillAdder, PrefillManager
from minisgl.scheduler.utils import PendingReq


class FakeReq:
    def __init__(self, uid: int, remain_len: int = 1, can_decode: bool = True) -> None:
        self.uid = uid
        self.remain_len = remain_len
        self._can_decode = can_decode

    def can_decode(self) -> bool:
        return self._can_decode


def test_decode_manager_deduplicates_and_removes_requests() -> None:
    manager = DecodeManager()
    req = FakeReq(1)

    manager.add_reqs([req, req])
    batch = manager.schedule_next_batch()

    assert batch is not None
    assert batch.reqs == [req]
    assert manager.inflight_tokens == 1
    manager.remove_req(req)
    assert manager.schedule_next_batch() is None


def test_prefill_manager_preserves_fifo_and_stops_at_first_blocked(
    monkeypatch,
) -> None:
    requests = [
        PendingReq(1, torch.tensor([1], dtype=torch.int32), SimpleNamespace(max_tokens=1)),
        PendingReq(2, torch.tensor([2], dtype=torch.int32), SimpleNamespace(max_tokens=1)),
        PendingReq(3, torch.tensor([3], dtype=torch.int32), SimpleNamespace(max_tokens=1)),
    ]
    manager = PrefillManager(
        cache_manager=SimpleNamespace(),
        table_manager=SimpleNamespace(),
        decode_manager=SimpleNamespace(inflight_tokens=0),
        pending_list=requests.copy(),
    )

    def try_add_one(_self, pending):
        return FakeReq(pending.uid) if pending.uid == 1 else None

    monkeypatch.setattr(PrefillAdder, "try_add_one", try_add_one)
    batch = manager.schedule_next_batch(prefill_budget=16)

    assert batch is not None
    assert [req.uid for req in batch.reqs] == [1]
    assert [req.uid for req in manager.pending_list] == [2, 3]


def test_finished_request_cache_release_returns_unshared_pages() -> None:
    class FakeBaseManager:
        def __init__(self) -> None:
            self.unlocked = False

        def insert_prefix(self, input_ids, indices):
            assert input_ids.tolist() == [10, 11, 12]
            assert indices.tolist() == [4, 5, 6]
            return 2

        def lock_handle(self, handle, unlock=False):
            self.unlocked = unlock

    manager = CacheManager.__new__(CacheManager)
    manager.manager = FakeBaseManager()
    manager._free_slots = torch.tensor([], dtype=torch.int32)
    handle = SimpleNamespace(cached_len=1)

    manager.free_and_cache_finished_req(
        handle,
        torch.tensor([10, 11, 12], dtype=torch.int32),
        torch.tensor([4, 5, 6], dtype=torch.int32),
    )

    assert manager._free_slots.tolist() == [5]
    assert manager.manager.unlocked is True


def test_request_completion_reaches_terminal_state() -> None:
    req = Req(
        input_ids=torch.tensor([10], dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=2,
        uid=1,
        cache_handle=BaseCacheHandle(cached_len=0),
        sampling_params=SamplingParams(max_tokens=2),
    )

    req.complete_one()
    assert req.can_decode()
    req.complete_one()

    assert req.cached_len == 2
    assert req.device_len == 3
    assert not req.can_decode()


def test_radix_cache_reports_prefix_hit(monkeypatch) -> None:
    def compare_key(node: RadixTreeNode, input_ids: torch.Tensor) -> int:
        limit = min(node.length, len(input_ids))
        for index in range(limit):
            if node._key[index].item() != input_ids[index].item():
                return index
        return limit

    monkeypatch.setattr(RadixTreeNode, "get_match_len", compare_key)
    manager = RadixCacheManager(torch.device("cpu"))
    manager.insert_prefix(
        torch.tensor([10, 11, 12], dtype=torch.int32),
        torch.tensor([4, 5, 6], dtype=torch.int32),
    )

    handle, indices = manager.match_prefix(
        torch.tensor([10, 11, 99], dtype=torch.int32)
    )

    assert handle.cached_len == 2
    assert indices.tolist() == [4, 5]


def test_prefill_adder_chunks_prompt_to_token_budget(monkeypatch) -> None:
    class FakeDestination:
        def __init__(self) -> None:
            self.copied = None

        def __getitem__(self, _slice):
            return self

        def copy_(self, source, non_blocking=False):
            assert non_blocking
            self.copied = source.clone()

    destination = FakeDestination()
    table_manager = SimpleNamespace(token_pool=[destination])
    pending = PendingReq(
        uid=9,
        input_ids=torch.arange(10, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=2),
    )
    adder = PrefillAdder(
        token_budget=4,
        reserved_size=0,
        cache_manager=SimpleNamespace(),
        table_manager=table_manager,
    )
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)

    req = adder._add_one_req(
        pending_req=pending,
        cache_handle=BaseCacheHandle(cached_len=0),
        table_idx=0,
        cached_len=0,
    )

    assert isinstance(req, ChunkedReq)
    assert req.extend_len == 4
    assert destination.copied.tolist() == [0, 1, 2, 3]
    assert adder.token_budget == 0
    assert adder.reserved_size == 12
