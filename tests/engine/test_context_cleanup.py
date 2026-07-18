from __future__ import annotations

import pytest

from minisgl.core import Context


def test_forward_context_resets_after_exception() -> None:
    context = Context.__new__(Context)
    context._batch = None
    batch = object()

    with pytest.raises(RuntimeError, match="boom"):
        with context.forward_batch(batch):
            assert context.batch is batch
            raise RuntimeError("boom")

    assert context._batch is None


@pytest.mark.xfail(
    strict=True,
    reason="Pinned upstream does not export AbortMsg or route cancellation to Scheduler",
)
def test_cancellation_message_reaches_scheduler_boundary() -> None:
    from minisgl.message import AbortMsg

    assert AbortMsg(uid=1).uid == 1
