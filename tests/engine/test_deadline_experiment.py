from __future__ import annotations

import pytest

from benchmark.deadline_experiment import (
    RequestMetric,
    WorkItem,
    _trace_shape,
    summarize,
    trace_fingerprint,
)


def request_metric(offset: float, end: float = 1.0) -> RequestMetric:
    return RequestMetric(
        request_hash=f"request-{offset}",
        request_class="short",
        input_len=64,
        expected_output_tokens=4,
        output_tokens=4,
        sse_chunks=3,
        scheduled_offset_s=offset,
        send_offset_s=0.25 + offset,
        first_token_offset_s=0.30 + offset,
        end_offset_s=end + offset,
        queue_delay_ms=0.0,
        ttft_ms=50.0,
        tpot_ms=10.0,
        e2e_ms=80.0,
        output_hash="hash",
        status_code=200,
        terminal_reason="stop",
        error_code=None,
        slo_good=True,
    )


def health() -> dict:
    return {
        "ready": True,
        "waiting_count": 0,
        "running_count": 0,
        "cancelled_count": 0,
        "failed_count": 0,
        "fatal_error": None,
    }


def test_trace_shapes_are_deterministic_and_use_safe_lengths() -> None:
    for workload in (
        "mixed",
        "long-prefill-interference",
        "starvation-stress",
    ):
        first = _trace_shape(workload, 17)
        second = _trace_shape(workload, 17)
        assert first == second
        assert all(input_len <= 2048 for _, input_len, _, _ in first)
        assert all(output_len <= 96 for _, _, output_len, _ in first)


def test_trace_fingerprint_never_depends_on_prompt_contents() -> None:
    metadata = {
        "request_hash": "safe-hash",
        "request_class": "short",
        "input_len": 64,
        "output_len": 8,
        "scheduled_offset_s": 0.0,
    }
    first = WorkItem(prompt="private prompt one", **metadata)
    second = WorkItem(prompt="private prompt two", **metadata)

    assert trace_fingerprint([first]) == trace_fingerprint([second])


def test_summary_uses_measurement_window_and_actual_arrival_cadence() -> None:
    metrics = [
        request_metric(0.00, end=0.50),
        request_metric(0.03, end=0.53),
        request_metric(0.06, end=0.56),
    ]

    result = summarize(metrics, health(), health(), {})

    assert round(result["offered_rps"], 3) == 33.333
    assert result["duration_s"] == pytest.approx(0.37)
    assert round(result["completed_rps"], 3) == 8.108
    assert result["timeout_rate"] == 0.0
