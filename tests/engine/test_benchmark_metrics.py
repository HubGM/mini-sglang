from __future__ import annotations

from benchmark.engine_lab_baseline import RequestMetric, percentile, summarize
from benchmark.summarize_baseline import summarize_group


def metric(request_id: str, *, error: str | None = None) -> RequestMetric:
    return RequestMetric(
        request_id=request_id,
        input_len=10,
        expected_output_len=4,
        observed_chunks=4 if error is None else 0,
        scheduled_s=0.0,
        sent_s=0.0,
        first_token_s=0.1 if error is None else None,
        end_s=0.5,
        ttft_ms=100.0 if error is None else None,
        tpot_ms=25.0 if error is None else None,
        e2e_ms=500.0,
        queue_delay_ms=0.0,
        error=error,
    )


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([], 0.95) is None


def test_request_summary_excludes_failed_requests_from_throughput() -> None:
    summary = summarize([metric("ok"), metric("failed", error="timeout")], 2.0)

    assert summary["successful_requests"] == 1
    assert summary["errors"] == 1
    assert summary["completed_rps"] == 0.5
    assert summary["output_tokens_per_s"] == 2.0


def test_repeated_summary_reports_dispersion() -> None:
    records = []
    for repeat, rps in enumerate((1.0, 2.0, 3.0), start=1):
        records.append(
            {
                "repeat": repeat,
                "requests": 4,
                "successful_requests": 4,
                "errors": 0,
                "completed_rps": rps,
                "input_tokens_per_s": 10.0,
                "output_tokens_per_s": 5.0,
                "ttft_ms": {"p95": 100.0, "p99": 120.0},
                "tpot_ms": {"p95": 20.0, "p99": 25.0},
                "e2e_ms": {"p95": 300.0, "p99": 350.0},
                "queue_delay_ms": {"p95": 1.0},
            }
        )

    summary = summarize_group(records)

    assert summary["valid_runs"] == 3
    assert summary["metrics"]["completed_rps"]["median"] == 2.0
    assert summary["metrics"]["completed_rps"]["stdev"] == 1.0
