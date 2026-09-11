from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.deadline_v2_matrix import (
    DEV_CANDIDATES,
    DEV_WORKLOADS,
    HOLDOUT_WORKLOADS,
    POLICIES,
    aggregate_holdout,
    build_dev_manifest,
    build_holdout_manifest,
    compact_summary,
    config_hash,
    select_dev_config,
    summarize_correctness,
    validate_run_payload,
)


def make_repo(root: Path) -> Path:
    for relative in (
        "python/minisgl/scheduler/advanced_policy.py",
        "python/minisgl/scheduler/policy.py",
        "python/minisgl/scheduler/scheduler.py",
        "benchmark/deadline_experiment.py",
        "benchmark/deadline_v2_matrix.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    return root


def request_sample(request_class: str = "short") -> dict:
    return {
        "terminal_state": "completed",
        "request_class": request_class,
        "request_role": "prefill",
        "requested_output_tokens": 4,
        "generated_tokens": 4,
        "waiting_time_ms": 10.0,
        "starvation": False,
        "enqueue_to_first_schedule_ms": 10.0,
        "enqueue_to_prefill_start_ms": 12.0,
        "prefill_start_to_end_ms": 8.0,
        "prefill_end_to_first_token_ms": 3.0,
        "first_token_to_finish_ms": 20.0,
    }


def scheduler_payload(policy: str, config_hash_value: str | None = None) -> dict:
    return {
        "policy": policy,
        "policy_config_hash": config_hash_value,
        "fallback_count": 0,
        "policy_failure_count": 0,
        "cancelled_count": 0,
        "failed_count": 0,
        "dropped_step_records": 0,
        "decision_latency_p50_us": 1.0,
        "decision_latency_p95_us": 2.0,
        "decision_latency_p99_us": 3.0,
        "token_budget_utilization": 0.5,
        "step_records": [
            {
                "waiting_count": 1,
                "running_count": 1,
                "selected_prefill_count": 1,
                "selected_decode_count": 0,
                "token_budget": 2048,
                "token_budget_used": 128,
                "prefill_chunk_size": 128,
                "dynamic_decode_reserve_ratio": 0.4,
                "prefill_urgent_request_count": 0,
                "hard_urgent_request_count": 0,
                "consecutive_decode_only_steps": 0,
                "chunk_continuation_deferred": False,
                "reason_codes": ["prefill_selected"],
            }
        ],
        "request_samples": [request_sample()],
    }


def run_payload(policy: str, workload: str, repeat: int) -> dict:
    return {
        "run_id": f"holdout-{workload}-{policy}-r{repeat}",
        "workload": workload,
        "policy": policy,
        "repeat": repeat,
        "trace_fingerprint": f"{workload}-{repeat}",
        "valid": True,
        "requests": 1,
        "successful_requests": 1,
        "output_tokens": 4,
        "completed_rps": 10.0 + repeat,
        "goodput_rps": 8.0,
        "slo_attainment": 0.8,
        "ttft_ms": {"p50": 40.0, "p95": 50.0, "p99": 60.0},
        "tpot_ms": {"p50": 20.0, "p95": 25.0, "p99": 30.0},
        "e2e_ms": {"p50": 100.0, "p95": 120.0, "p99": 140.0},
        "per_class": {
            "short": {
                "ttft_ms": {"p50": 40.0, "p95": 50.0, "p99": 60.0}
            }
        },
        "health": {"final_waiting": 0, "final_running": 0},
    }


def test_dev_and_holdout_plans_are_disjoint_and_bounded(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    dev = build_dev_manifest(repo)
    frozen = {
        "config": DEV_CANDIDATES["balanced"],
        "FINAL_CONFIG_HASH": config_hash(DEV_CANDIDATES["balanced"]),
    }
    holdout = build_holdout_manifest(repo, frozen)

    assert dev["candidate_count"] == 3
    assert dev["run_count"] == 9
    assert holdout["run_count"] == 27
    assert {run["trace_seed"] for run in dev["runs"]}.isdisjoint(
        {run["trace_seed"] for run in holdout["runs"]}
    )
    assert len({run["policy"] for run in holdout["runs"]}) == 3


def test_v2_validity_requires_complete_phase_and_dual_slo_telemetry() -> None:
    summary = run_payload("deadline_aging_v2", "mixed", 1)
    result = validate_run_payload(
        summary,
        scheduler_payload("deadline_aging_v2", "frozen"),
        "deadline_aging_v2",
        expected_config_hash="frozen",
    )

    assert result["formal_valid"]
    assert result["scheduler"]["per_class"]["short"]["waiting_time_ms"]["p95"] == 10.0
    assert result["scheduler"]["phase_timing"]["prefill_start_to_end_ms"]["p95"] == 8.0
    assert result["FINAL_CONFIG_HASH"] == "frozen"

    broken = scheduler_payload("deadline_aging_v2")
    broken["request_samples"][0]["prefill_end_to_first_token_ms"] = None
    invalid = validate_run_payload(
        run_payload("deadline_aging_v2", "mixed", 1),
        broken,
        "deadline_aging_v2",
    )
    assert not invalid["validity_gate"]["phase_telemetry_complete"]


def test_upstream_control_is_not_subject_to_experimental_chunk_cap() -> None:
    scheduler = scheduler_payload("upstream_default")
    scheduler["step_records"][0]["prefill_chunk_size"] = 1536
    scheduler["step_records"][0]["token_budget"] = 4096
    scheduler["step_records"][0]["token_budget_used"] = 1536

    result = validate_run_payload(
        run_payload("upstream_default", "mixed", 1),
        scheduler,
        "upstream_default",
    )

    assert result["formal_valid"]
    assert result["validity_gate"]["prefill_chunk_respected"]


def test_dev_selection_is_lexicographic_and_immutable(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    for candidate_index, candidate in enumerate(DEV_CANDIDATES):
        for workload in DEV_WORKLOADS:
            payload = validate_run_payload(
                run_payload("deadline_aging_v2", workload, 1),
                scheduler_payload("deadline_aging_v2"),
                "deadline_aging_v2",
                candidate=candidate,
            )
            payload["scheduler"]["starvation_count"] = candidate_index
            (results / f"dev-{workload}-{candidate}.summary.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
    frozen_path = tmp_path / "frozen.json"

    frozen = select_dev_config(results, frozen_path)

    assert frozen["selected_candidate"] == "balanced"
    assert frozen["FINAL_CONFIG_HASH"] == config_hash(DEV_CANDIDATES["balanced"])
    assert select_dev_config(results, frozen_path) == frozen
    changed = json.loads(frozen_path.read_text(encoding="utf-8"))
    changed["FINAL_CONFIG_HASH"] = "changed"
    frozen_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be changed"):
        select_dev_config(results, frozen_path)


def test_holdout_aggregate_requires_27_valid_runs_and_is_privacy_safe(tmp_path: Path) -> None:
    frozen = {
        "config": DEV_CANDIDATES["balanced"],
        "FINAL_CONFIG_HASH": config_hash(DEV_CANDIDATES["balanced"]),
    }
    for workload in HOLDOUT_WORKLOADS:
        for policy in POLICIES:
            for repeat in (1, 2, 3):
                payload = validate_run_payload(
                    run_payload(policy, workload, repeat),
                    scheduler_payload(
                        policy,
                        frozen["FINAL_CONFIG_HASH"]
                        if policy == "deadline_aging_v2"
                        else None,
                    ),
                    policy,
                    expected_config_hash=(
                        frozen["FINAL_CONFIG_HASH"]
                        if policy == "deadline_aging_v2"
                        else None
                    ),
                )
                (tmp_path / f"holdout-{workload}-{policy}-r{repeat}.summary.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

    aggregate = aggregate_holdout(tmp_path, frozen)
    compact = compact_summary(aggregate)
    serialized = json.dumps(compact)

    assert aggregate["valid_runs"] == 27
    assert aggregate["all_repeats_complete"]
    assert aggregate["same_trace_within_repeat"]
    assert aggregate["final_config_hash_matches"]
    assert "values" not in serialized
    assert "trace_fingerprint" not in serialized


def test_three_policy_correctness_summary_does_not_export_hashes(tmp_path: Path) -> None:
    paths = []
    for policy in POLICIES:
        path = tmp_path / f"{policy}.json"
        path.write_text(
            json.dumps(
                {
                    "policy": policy,
                    "prompt_count": 3,
                    "token_counts": [16, 16, 16],
                    "token_sequence_hashes": ["a", "b", "c"],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    result = summarize_correctness(paths)

    assert result["all_identical"]
    assert "token_sequence_hashes" not in json.dumps(result)
