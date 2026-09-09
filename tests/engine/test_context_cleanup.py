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


def test_cancellation_message_reaches_scheduler_boundary() -> None:
    from minisgl.message import AbortBackendMsg, AbortMsg

    assert AbortMsg(uid=1).uid == 1
    assert AbortBackendMsg(uid=1).uid == 1
