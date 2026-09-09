from __future__ import annotations

import json

from benchmark.lifecycle_stress import (
    RequestResult,
    parse_sse_line,
    percentile,
    summarize_results,
    synthetic_prompt,
)


def result(mode: str, outcome: str, cleanup_ms: float | None) -> RequestResult:
    return RequestResult(
        mode=mode,
        outcome=outcome,
        elapsed_ms=10.0,
        cancel_cleanup_ms=cleanup_ms,
        data_events=1,
        output_hash="redacted-hash",
        cancel_accepted=outcome == "cancelled",
        duplicate_cancel_rejected=None,
        error_code=None,
    )


def test_synthetic_prompt_is_fixed_length_and_deterministic() -> None:
    first = synthetic_prompt(7, 32)
    second = synthetic_prompt(7, 32)

    assert first == second
    assert len(first.split()) == 32


def test_sse_parser_extracts_content_and_terminal_reason() -> None:
    content = {
        "choices": [
            {
                "delta": {"content": "token"},
                "finish_reason": None,
            }
        ]
    }
    terminal = {
        "choices": [
            {
                "delta": {},
                "finish_reason": "cancelled",
            }
        ]
    }

    assert parse_sse_line(f"data: {json.dumps(content)}") == ("token", None)
    assert parse_sse_line(f"data: {json.dumps(terminal)}") == (
        "",
        "cancelled",
    )
    assert parse_sse_line("data: [DONE]") == ("", None)


def test_stress_summary_counts_terminal_outcomes_and_cleanup_latency() -> None:
    summary = summarize_results(
        [
            result("normal", "completed", None),
            result("waiting", "cancelled", 5.0),
            result("disconnect", "disconnect", None),
        ]
    )

    assert summary["requests"] == 3
    assert summary["terminal_or_disconnect"] == 3
    assert summary["outcomes"] == {
        "cancelled": 1,
        "completed": 1,
        "disconnect": 1,
    }
    assert summary["cancel_cleanup_ms"]["p95"] == 5.0
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
