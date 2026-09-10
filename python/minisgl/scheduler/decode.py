from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    running_reqs: List[Req] = field(default_factory=list)

    def add_reqs(self, reqs: Iterable[Req]) -> None:
        existing = {req.uid for req in self.running_reqs}
        for req in reqs:
            if req.can_decode() and req.uid not in existing:
                self.running_reqs.append(req)
                existing.add(req.uid)

    def remove_req(self, req: Req) -> None:
        self.running_reqs = [item for item in self.running_reqs if item is not req]

    def abort_req(self, uid: int) -> Req | None:
        for index, req in enumerate(self.running_reqs):
            if req.uid == uid:
                return self.running_reqs.pop(index)
        return None

    def contains_uid(self, uid: int) -> bool:
        return any(req.uid == uid for req in self.running_reqs)

    @property
    def inflight_tokens(self) -> int:
        return sum(req.remain_len for req in self.running_reqs)

    def schedule_next_batch(
        self,
        max_requests: int | None = None,
        priority_uids: Sequence[int] | None = None,
    ) -> Batch | None:
        if not self.runnable:
            return None
        req_by_uid = {req.uid: req for req in self.running_reqs}
        ordered: list[Req] = []
        if priority_uids is not None:
            ordered.extend(
                req_by_uid[uid] for uid in priority_uids if uid in req_by_uid
            )
        selected_uids = {req.uid for req in ordered}
        ordered.extend(
            req for req in self.running_reqs if req.uid not in selected_uids
        )
        if max_requests is not None:
            ordered = ordered[:max_requests]
        return Batch(reqs=ordered, phase="decode") if ordered else None

    @property
    def runnable(self) -> bool:
        return bool(self.running_reqs)
