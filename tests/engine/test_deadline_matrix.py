from __future__ import annotations

import json
from pathlib import Path

from benchmark.deadline_experiment import POLICIES, WORKLOADS
from benchmark.deadline_matrix import (
    aggregate_results,
    bootstrap_ci,
    build_manifest,
    compact_summary,
    repeated_stats,
    run_is_complete,
    summarize_correctness,
    validate_run_payload,
)


def scheduler_payload(policy: str, request_count: int = 2) -> dict:
    return {
        "policy": policy,
        "fallback_count": 0,
        "policy_failure_count": 0,
        "cancelled_count": 0,
        "failed_count": 0,
        "dropped_step_records": 0,
        "decision_latency_p50_us": 1.0,
        "decision_latency_p95_us": 2.0,
        "decision_latency_p99_us": 3.0,
        "token_budget_utilization": 0.5,
        "average_prefill_chunk_tokens": 128.0,
        "decode_batch_count": 5,
        "prefill_batch_count": 2,
        "minimum_slack_ms": -1.0,
        "max_urgent_requests": 1,
        "decision_count": 7,
        "step_records": [
            {
                "token_budget": 2048,
                "token_budget_used": 512,
                "prefill_chunk_size": 512,
            }
        ],
        "request_samples": [
            {
                "terminal_state": "completed",
                "requested_output_tokens": 4,
                "generated_tokens": 4,
                "waiting_time_ms": float(index + 1),
                "starvation": False,
            }
            for index in range(request_count)
        ],
    }


def run_payload(policy: str, workload: str, repeat: int) -> dict:
    return {
        "run_id": f"{workload}-{policy}-r{repeat}",
        "workload": workload,
        "policy": policy,
        "repeat": repeat,
        "trace_fingerprint": f"{workload}-r{repeat}",
        "valid": True,
        "requests": 2,
        "successful_requests": 2,
        "expected_output_tokens": 8,
        "output_tokens": 8,
        "completed_rps": float(repeat),
        "input_tokens_per_s": 100.0,
        "output_tokens_per_s": 20.0,
        "goodput_rps": float(repeat),
        "slo_attainment": 1.0,
        "ttft_ms": {"p50": 10.0, "p95": 20.0, "p99": 30.0},
        "tpot_ms": {"p50": 5.0, "p95": 6.0, "p99": 7.0},
        "e2e_ms": {"p50": 30.0, "p95": 40.0, "p99": 50.0},
        "long_request_completion_rate": 1.0,
        "gpu": {
            "utilization_mean_percent": 50.0,
            "utilization_max_percent": 90.0,
            "memory_mean_mib": 20_000.0,
            "memory_max_mib": 21_000.0,
        },
        "health": {"final_waiting": 0, "final_running": 0},
    }


def test_manifest_randomizes_policy_order_but_keeps_trace_seed(tmp_path: Path) -> None:
    repo = tmp_path
    paths = (
        "python/minisgl/scheduler/advanced_policy.py",
        "python/minisgl/scheduler/policy.py",
        "python/minisgl/scheduler/scheduler.py",
        "benchmark/deadline_experiment.py",
        "benchmark/deadline_matrix.py",
    )
    for relative in paths:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    manifest = build_manifest(repo, seed=99)
    orders = []
    for repeat in (1, 2, 3):
        for workload in WORKLOADS:
            block = [
                run
                for run in manifest["runs"]
                if run["repeat"] == repeat and run["workload"] == workload
            ]
            assert {run["policy"] for run in block} == set(POLICIES)
            assert len({run["trace_seed"] for run in block}) == 1
            orders.append(tuple(run["policy"] for run in block))
    assert len(set(orders)) > 1
    assert manifest["core_run_count"] == 36


def test_validity_gate_checks_scheduler_and_request_lifecycle() -> None:
    summary = run_payload("deadline_aging", "mixed", 1)

    result = validate_run_payload(
        summary, scheduler_payload("deadline_aging"), "deadline_aging"
    )

    assert result["formal_valid"]
    assert result["scheduler"]["waiting_time_ms"]["p99"] == 1.99
    assert result["scheduler"]["starvation_count"] == 0


def test_invalid_policy_or_budget_fails_validity_gate() -> None:
    summary = run_payload("deadline_aging", "mixed", 1)
    scheduler = scheduler_payload("token_budget")
    scheduler["step_records"][0]["token_budget_used"] = 4096

    result = validate_run_payload(summary, scheduler, "deadline_aging")

    assert not result["formal_valid"]
    assert not result["validity_gate"]["policy_matches"]
    assert not result["validity_gate"]["token_budget_respected"]


def test_repeated_statistics_and_bootstrap_are_deterministic() -> None:
    stats = repeated_stats([1.0, 2.0, 3.0], seed=7)

    assert stats["median"] == 2.0
    assert stats["stdev"] == 1.0
    assert bootstrap_ci([1.0, 2.0, 3.0], seed=7) == stats[
        "bootstrap_median_95ci"
    ]


def test_aggregate_requires_three_valid_repeats_per_condition(tmp_path: Path) -> None:
    for workload in WORKLOADS:
        for policy in POLICIES:
            for repeat in (1, 2, 3):
                payload = validate_run_payload(
                    run_payload(policy, workload, repeat),
                    scheduler_payload(policy),
                    policy,
                )
                path = tmp_path / f"{workload}-{policy}-r{repeat}.summary.json"
                path.write_text(json.dumps(payload), encoding="utf-8")

    result = aggregate_results(tmp_path)

    assert result["valid_core_runs"] == 36
    assert result["all_repeats_complete"]
    assert result["same_trace_within_condition"]
    assert result["groups"]["mixed"]["deadline_aging"]["metrics"][
        "completed_rps"
    ]["median"] == 2.0


def test_completion_and_correctness_summaries_are_privacy_safe(tmp_path: Path) -> None:
    summary_path = tmp_path / "run.summary.json"
    summary_path.write_text('{"formal_valid": true}', encoding="utf-8")
    assert run_is_complete(summary_path)

    paths = []
    for policy in POLICIES:
        path = tmp_path / f"{policy}.json"
        path.write_text(
            json.dumps(
                {
                    "policy": policy,
                    "prompt_count": 2,
                    "token_counts": [4, 4],
                    "token_sequence_hashes": ["a", "b"],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    result = summarize_correctness(paths)

    assert result["all_identical"]
    assert not result["raw_prompts_stored"]
    assert not result["raw_token_ids_stored"]
    assert "token_sequence_hashes" not in json.dumps(result)


def test_compact_summary_excludes_run_and_trace_identifiers(tmp_path: Path) -> None:
    for workload in WORKLOADS:
        for policy in POLICIES:
            for repeat in (1, 2, 3):
                payload = validate_run_payload(
                    run_payload(policy, workload, repeat),
                    scheduler_payload(policy),
                    policy,
                )
                path = tmp_path / f"{workload}-{policy}-r{repeat}.summary.json"
                path.write_text(json.dumps(payload), encoding="utf-8")

    aggregate = aggregate_results(tmp_path)
    correctness = {
        "all_identical": True,
        "raw_prompts_stored": False,
        "raw_token_ids_stored": False,
    }
    manifest = {
        "base_seed": 99,
        "policies": list(POLICIES),
        "workloads": list(WORKLOADS),
        "repeats": [1, 2, 3],
        "slo": {"ttft_ms": 200.0},
        "scheduler": {"max_step_tokens": 2048},
    }

    result = compact_summary(aggregate, correctness, manifest)
    serialized = json.dumps(result)

    assert result["validity"]["valid_core_runs"] == 36
    assert result["groups"]["mixed"]["deadline_aging"]["metrics"][
        "completed_rps"
    ]["median"] == 2.0
    assert "values" not in serialized
    assert "run_ids" not in serialized
    assert all(
        "trace_fingerprints" not in group
        for policies in result["groups"].values()
        for group in policies.values()
    )
    assert result["privacy"]["aggregate_only"]
