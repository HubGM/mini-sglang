import torch


class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self._allocated_slots: set[int] = set()
        self.page_table = page_table
        self.token_pool = torch.empty_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        slot = self._free_slots.pop()
        if slot in self._allocated_slots:
            raise RuntimeError(f"Table slot {slot} is already allocated")
        self._allocated_slots.add(slot)
        return slot

    def free(self, slot: int) -> None:
        if slot not in self._allocated_slots:
            raise RuntimeError(f"Table slot {slot} is not owned by an active request")
        self._allocated_slots.remove(slot)
        self._free_slots.append(slot)

    def owns(self, slot: int) -> bool:
        return slot in self._allocated_slots
