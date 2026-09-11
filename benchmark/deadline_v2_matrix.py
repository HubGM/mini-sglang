from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
DEV_BASE_SEED = 20261000
HOLDOUT_BASE_SEED = 20261100
POLICIES = (
    "upstream_default",
    "deadline_aging_v1",
    "deadline_aging_v2",
)
DEV_WORKLOADS = (
    "mixed",
    "long-prefill-interference",
    "deadline-induced-starvation",
)
HOLDOUT_WORKLOADS = (
    "mixed",
    "long-prefill-interference",
    "starvation-stress",
)
STARVATION_THRESHOLD_MS = 400.0
MAX_STEP_TOKENS = 2048
MAX_PREFILL_CHUNK_TOKENS = 512

V1_CONFIG = {
    "max_consecutive_prefill_steps": 1,
    "min_prefill_budget_per_step": 256,
    "max_decode_only_steps": 2,
    "prefill_urgent_threshold_ms": 150.0,
    "hard_max_wait_ms": 300.0,
    "min_decode_reserve_ratio": 0.25,
    "max_decode_reserve_ratio": 0.65,
}

# Candidate count is deliberately capped at three before any DEV data exists.
DEV_CANDIDATES: dict[str, dict[str, int | float]] = {
    "balanced": {
        "max_consecutive_prefill_steps": 2,
        "min_prefill_budget_per_step": 256,
        "max_decode_only_steps": 2,
        "prefill_urgent_threshold_ms": 150.0,
        "hard_max_wait_ms": 300.0,
        "min_decode_reserve_ratio": 0.25,
        "max_decode_reserve_ratio": 0.65,
    },
    "prefill_strict": {
        "max_consecutive_prefill_steps": 2,
        "min_prefill_budget_per_step": 512,
        "max_decode_only_steps": 1,
        "prefill_urgent_threshold_ms": 100.0,
        "hard_max_wait_ms": 250.0,
        "min_decode_reserve_ratio": 0.20,
        "max_decode_reserve_ratio": 0.60,
    },
    "decode_guarded": {
        "max_consecutive_prefill_steps": 1,
        "min_prefill_budget_per_step": 256,
        "max_decode_only_steps": 2,
        "prefill_urgent_threshold_ms": 150.0,
        "hard_max_wait_ms": 300.0,
        "min_decode_reserve_ratio": 0.30,
        "max_decode_reserve_ratio": 0.70,
    },
}

CONFIG_FLAGS = {
    "max_consecutive_prefill_steps": "max-consecutive-prefill-steps",
    "min_prefill_budget_per_step": "min-prefill-budget-per-step",
    "max_decode_only_steps": "max-decode-only-steps",
    "prefill_urgent_threshold_ms": "prefill-urgent-threshold-ms",
    "hard_max_wait_ms": "hard-max-wait-ms",
    "min_decode_reserve_ratio": "min-decode-reserve-ratio",
    "max_decode_reserve_ratio": "max-decode-reserve-ratio",
}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def config_hash(config: dict[str, int | float]) -> str:
    encoded = json.dumps(
        config, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def implementation_fingerprint(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "python/minisgl/scheduler/advanced_policy.py",
        "python/minisgl/scheduler/policy.py",
        "python/minisgl/scheduler/scheduler.py",
        "benchmark/deadline_experiment.py",
        "benchmark/deadline_v2_matrix.py",
    ):
        digest.update(relative.encode())
        digest.update((repo_root / relative).read_bytes())
    return digest.hexdigest()


def _randomized_runs(
    *,
    policies: Sequence[str],
    workloads: Sequence[str],
    repeats: Sequence[int],
    seed: int,
    prefix: str,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    sequence = 0
    for repeat in repeats:
        for workload_index, workload in enumerate(workloads):
            order = list(policies)
            order_seed = seed + repeat * 10_000 + workload_index * 101
            random.Random(order_seed).shuffle(order)
            for position, policy in enumerate(order, start=1):
                sequence += 1
                runs.append(
                    {
                        "sequence": sequence,
                        "run_id": f"{prefix}-{workload}-{policy}-r{repeat}",
                        "workload": workload,
                        "policy": policy,
                        "repeat": repeat,
                        "trace_seed": seed + repeat,
                        "order_seed": order_seed,
                        "position_in_block": position,
                    }
                )
    return runs


def build_dev_manifest(repo_root: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    sequence = 0
    for workload_index, workload in enumerate(DEV_WORKLOADS):
        candidates = list(DEV_CANDIDATES)
        order_seed = DEV_BASE_SEED + workload_index * 101
        random.Random(order_seed).shuffle(candidates)
        for position, candidate in enumerate(candidates, start=1):
            sequence += 1
            runs.append(
                {
                    "sequence": sequence,
                    "run_id": f"dev-{workload}-{candidate}",
                    "workload": workload,
                    "policy": "deadline_aging_v2",
                    "repeat": 1,
                    "trace_seed": DEV_BASE_SEED + 1,
                    "candidate": candidate,
                    "config_hash": config_hash(DEV_CANDIDATES[candidate]),
                    "order_seed": order_seed,
                    "position_in_block": position,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "DEV",
        "base_seed": DEV_BASE_SEED,
        "implementation_fingerprint": implementation_fingerprint(repo_root),
        "candidate_count": len(DEV_CANDIDATES),
        "candidates": {
            name: {"config": config, "config_hash": config_hash(config)}
            for name, config in DEV_CANDIDATES.items()
        },
        "workloads": list(DEV_WORKLOADS),
        "run_count": len(runs),
        "starvation_threshold_ms": STARVATION_THRESHOLD_MS,
        "raw_prompts_stored": False,
        "raw_token_ids_stored": False,
        "runs": runs,
    }


def build_holdout_manifest(
    repo_root: Path, frozen: dict[str, Any]
) -> dict[str, Any]:
    runs = _randomized_runs(
        policies=POLICIES,
        workloads=HOLDOUT_WORKLOADS,
        repeats=(1, 2, 3),
        seed=HOLDOUT_BASE_SEED,
        prefix="holdout",
    )
    for run in runs:
        if run["policy"] == "deadline_aging_v2":
            run["config_hash"] = frozen["FINAL_CONFIG_HASH"]
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "HOLDOUT",
        "base_seed": HOLDOUT_BASE_SEED,
        "implementation_fingerprint": implementation_fingerprint(repo_root),
        "policies": list(POLICIES),
        "workloads": list(HOLDOUT_WORKLOADS),
        "repeats": [1, 2, 3],
        "run_count": len(runs),
        "FINAL_CONFIG_HASH": frozen["FINAL_CONFIG_HASH"],
        "starvation_threshold_ms": STARVATION_THRESHOLD_MS,
        "slo": {"ttft_ms": 200.0, "tpot_ms": 50.0, "e2e_ms": 1200.0},
        "raw_prompts_stored": False,
        "raw_token_ids_stored": False,
        "runs": runs,
    }


def create_or_validate_manifest(
    path: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError("Existing manifest does not match the frozen plan")
        return existing
    atomic_write_json(path, expected)
    return expected


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def distribution(values: Iterable[float]) -> dict[str, float | None]:
    copied = [float(value) for value in values]
    return {
        "p50": percentile(copied, 0.50),
        "p95": percentile(copied, 0.95),
        "p99": percentile(copied, 0.99),
        "max": max(copied, default=None),
    }


def _normalized_class(sample: dict[str, Any]) -> str:
    value = sample.get("request_class")
    if value in {"short", "decode"}:
        return "short"
    if value in {"long", "long-prefill"}:
        return "long"
    return "medium"


PHASE_FIELDS = (
    "enqueue_to_first_schedule_ms",
    "enqueue_to_prefill_start_ms",
    "prefill_start_to_end_ms",
    "prefill_end_to_first_token_ms",
    "first_token_to_finish_ms",
)


def _phase_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        field: distribution(
            float(sample[field])
            for sample in samples
            if sample.get(field) is not None
        )
        for field in PHASE_FIELDS
    }


def _step_diagnosis(records: list[dict[str, Any]]) -> dict[str, Any]:
    waiting = [record for record in records if record.get("waiting_count", 0) > 0]
    no_prefill = [
        record
        for record in waiting
        if record.get("selected_prefill_count", 0) == 0
    ]
    decode_only = [
        record
        for record in no_prefill
        if record.get("selected_decode_count", 0) > 0
    ]
    reasons: dict[str, int] = {}
    ratios: list[float] = []
    for record in records:
        for reason in record.get("reason_codes", []):
            reasons[reason] = reasons.get(reason, 0) + 1
        if record.get("dynamic_decode_reserve_ratio") is not None:
            ratios.append(float(record["dynamic_decode_reserve_ratio"]))
    return {
        "steps_with_waiting_prefill": len(waiting),
        "steps_without_prefill_while_waiting": len(no_prefill),
        "decode_only_steps_while_waiting": len(decode_only),
        "max_consecutive_decode_only_steps": max(
            (int(record.get("consecutive_decode_only_steps", 0)) for record in records),
            default=0,
        ),
        "chunk_continuation_deferred_count": sum(
            bool(record.get("chunk_continuation_deferred")) for record in records
        ),
        "dynamic_decode_reserve_ratio": distribution(ratios),
        "reason_codes": dict(sorted(reasons.items())),
    }


def validate_run_payload(
    summary: dict[str, Any],
    scheduler: dict[str, Any],
    expected_policy: str,
    *,
    expected_config_hash: str | None = None,
    candidate: str | None = None,
) -> dict[str, Any]:
    samples = list(scheduler.get("request_samples", []))
    steps = list(scheduler.get("step_records", []))
    active_steps = [step for step in steps if int(step.get("token_budget_used", 0)) > 0]
    completed = [sample for sample in samples if sample.get("terminal_state") == "completed"]
    budget_respected = all(
        0 <= int(step.get("token_budget_used", 0))
        <= int(step.get("token_budget", 0))
        for step in active_steps
    )
    if expected_policy != "upstream_default":
        budget_respected = budget_respected and all(
            int(step.get("token_budget", 0)) <= MAX_STEP_TOKENS
            for step in active_steps
        )
    chunks_respected = expected_policy == "upstream_default" or all(
        int(step.get("prefill_chunk_size", 0)) <= MAX_PREFILL_CHUNK_TOKENS
        for step in active_steps
    )
    phase_complete = all(
        all(sample.get(field) is not None for field in PHASE_FIELDS)
        for sample in completed
    )
    gates = {
        "client_valid": bool(summary.get("valid")),
        "policy_matches": scheduler.get("policy") == expected_policy,
        "all_requests_sampled": len(samples) == summary.get("requests"),
        "all_terminal_completed": len(completed) == summary.get("requests"),
        "client_scheduler_token_count_matches": sum(
            int(sample.get("generated_tokens", 0)) for sample in samples
        )
        == summary.get("output_tokens"),
        "token_budget_respected": budget_respected,
        "prefill_chunk_respected": chunks_respected,
        "phase_telemetry_complete": phase_complete,
        "no_scheduler_fallback": scheduler.get("fallback_count", 0) == 0,
        "no_policy_failure": scheduler.get("policy_failure_count", 0) == 0,
        "no_cancellation": scheduler.get("cancelled_count", 0) == 0,
        "no_failure": scheduler.get("failed_count", 0) == 0,
        "telemetry_not_truncated": scheduler.get("dropped_step_records", 0) == 0,
        "final_waiting_zero": summary.get("health", {}).get("final_waiting") == 0,
        "final_running_zero": summary.get("health", {}).get("final_running") == 0,
    }
    if expected_policy == "deadline_aging_v2":
        gates["dual_slo_step_telemetry"] = bool(steps) and all(
            step.get("dynamic_decode_reserve_ratio") is not None
            and "prefill_urgent_request_count" in step
            and "hard_urgent_request_count" in step
            for step in steps
        )
        if expected_config_hash is not None:
            gates["frozen_config_hash_matches"] = (
                scheduler.get("policy_config_hash") == expected_config_hash
            )

    per_class: dict[str, Any] = {}
    for request_class in ("short", "medium", "long"):
        records = [sample for sample in completed if _normalized_class(sample) == request_class]
        if records:
            per_class[request_class] = {
                "requests": len(records),
                "waiting_time_ms": distribution(
                    float(sample["waiting_time_ms"]) for sample in records
                ),
                "phases": _phase_summary(records),
            }
    per_role = {
        role: {
            "requests": len(records),
            "waiting_time_ms": distribution(
                float(sample["waiting_time_ms"]) for sample in records
            ),
            "phases": _phase_summary(records),
        }
        for role in ("prefill", "decode", "mixed")
        if (records := [sample for sample in completed if sample.get("request_role") == role])
    }
    summary["scheduler"] = {
        "decision_latency_us": {
            "p50": scheduler.get("decision_latency_p50_us"),
            "p95": scheduler.get("decision_latency_p95_us"),
            "p99": scheduler.get("decision_latency_p99_us"),
        },
        "token_budget_utilization": scheduler.get("token_budget_utilization"),
        "waiting_time_ms": distribution(
            float(sample["waiting_time_ms"])
            for sample in completed
            if sample.get("waiting_time_ms") is not None
        ),
        "starvation_count": sum(bool(sample.get("starvation")) for sample in completed),
        "per_class": per_class,
        "per_role": per_role,
        "phase_timing": _phase_summary(completed),
        "step_diagnosis": _step_diagnosis(steps),
        "fallback_count": scheduler.get("fallback_count", 0),
        "policy_failure_count": scheduler.get("policy_failure_count", 0),
        "request_sample_count": len(samples),
        "step_record_count": len(steps),
    }
    if expected_config_hash is not None:
        key = "CONFIG_HASH" if candidate is not None else "FINAL_CONFIG_HASH"
        summary[key] = expected_config_hash
    if candidate is not None:
        summary["candidate"] = candidate
    summary["validity_gate"] = gates
    summary["formal_valid"] = all(gates.values())
    return summary


def validate_run_files(
    summary_path: Path,
    scheduler_path: Path,
    expected_policy: str,
    *,
    expected_config_hash: str | None = None,
    candidate: str | None = None,
) -> dict[str, Any]:
    result = validate_run_payload(
        json.loads(summary_path.read_text(encoding="utf-8")),
        json.loads(scheduler_path.read_text(encoding="utf-8")),
        expected_policy,
        expected_config_hash=expected_config_hash,
        candidate=candidate,
    )
    atomic_write_json(summary_path, result)
    return result


def run_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("formal_valid"))
    except (OSError, json.JSONDecodeError):
        return False


def select_dev_config(results_dir: Path, output: Path) -> dict[str, Any]:
    if len(DEV_CANDIDATES) > 3:
        raise ValueError("DEV candidate count exceeds frozen limit")
    scores: list[dict[str, Any]] = []
    for candidate, config in DEV_CANDIDATES.items():
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(results_dir.glob(f"dev-*-{candidate}.summary.json"))
        ]
        if len(records) != len(DEV_WORKLOADS) or not all(
            record.get("formal_valid") for record in records
        ):
            raise ValueError(f"DEV candidate {candidate} is incomplete")
        score = (
            sum(int(record["scheduler"]["starvation_count"]) for record in records),
            statistics.mean(float(record["ttft_ms"]["p95"]) for record in records),
            statistics.mean(float(record["tpot_ms"]["p95"]) for record in records),
            -statistics.mean(float(record["completed_rps"]) for record in records),
        )
        scores.append(
            {
                "candidate": candidate,
                "config_hash": config_hash(config),
                "score": list(score),
            }
        )
    selected = min(scores, key=lambda item: tuple(item["score"]))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "selection_order": [
            "starvation_count",
            "ttft_p95_ms",
            "tpot_p95_ms",
            "negative_completed_rps",
        ],
        "selected_candidate": selected["candidate"],
        "config": DEV_CANDIDATES[selected["candidate"]],
        "FINAL_CONFIG_HASH": selected["config_hash"],
        "dev_scores": scores,
        "holdout_examined": False,
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("Frozen final config cannot be changed")
        return existing
    atomic_write_json(output, payload)
    return payload


def bootstrap_ci(
    values: Sequence[float], *, seed: int, samples: int = 10_000
) -> list[float | None]:
    if not values:
        return [None, None]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(samples)
    ]
    return [percentile(medians, 0.025), percentile(medians, 0.975)]


def repeated_stats(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    copied = [float(value) for value in values]
    return {
        "values": copied,
        "median": statistics.median(copied) if copied else None,
        "mean": statistics.mean(copied) if copied else None,
        "stdev": statistics.stdev(copied) if len(copied) > 1 else 0.0,
        "min": min(copied, default=None),
        "max": max(copied, default=None),
        "bootstrap_median_95ci": bootstrap_ci(copied, seed=seed),
    }


METRICS = {
    "completed_rps": ("completed_rps",),
    "goodput_rps": ("goodput_rps",),
    "slo_attainment": ("slo_attainment",),
    "ttft_p95_ms": ("ttft_ms", "p95"),
    "ttft_p99_ms": ("ttft_ms", "p99"),
    "tpot_p95_ms": ("tpot_ms", "p95"),
    "tpot_p99_ms": ("tpot_ms", "p99"),
    "e2e_p95_ms": ("e2e_ms", "p95"),
    "e2e_p99_ms": ("e2e_ms", "p99"),
    "max_waiting_ms": ("scheduler", "waiting_time_ms", "max"),
    "starvation_count": ("scheduler", "starvation_count"),
    "scheduler_decision_p95_us": ("scheduler", "decision_latency_us", "p95"),
    "token_budget_utilization": ("scheduler", "token_budget_utilization"),
}


def _nested_number(payload: dict[str, Any], path: Sequence[str]) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value) if isinstance(value, (int, float)) else None


def _improvement(baseline: float | None, optimized: float | None) -> float | None:
    if baseline in (None, 0) or optimized is None:
        return None
    return (baseline - optimized) / baseline


def aggregate_holdout(results_dir: Path, final_config: dict[str, Any]) -> dict[str, Any]:
    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(results_dir.glob("holdout-*.summary.json"))
    ]
    valid = [run for run in runs if run.get("formal_valid")]
    groups: dict[str, dict[str, Any]] = {}
    metric_paths = dict(METRICS)
    for request_class in ("short", "medium", "long"):
        for quantile in ("p50", "p95", "p99"):
            metric_paths[f"{request_class}_ttft_{quantile}_ms"] = (
                "per_class", request_class, "ttft_ms", quantile
            )
        for quantile in ("p95", "p99"):
            metric_paths[f"{request_class}_waiting_{quantile}_ms"] = (
                "scheduler", "per_class", request_class, "waiting_time_ms", quantile
            )

    for workload_index, workload in enumerate(HOLDOUT_WORKLOADS):
        groups[workload] = {}
        for policy_index, policy in enumerate(POLICIES):
            records = sorted(
                (
                    run for run in valid
                    if run.get("workload") == workload and run.get("policy") == policy
                ),
                key=lambda run: run.get("repeat", 0),
            )
            metrics = {
                name: repeated_stats(
                    [
                        value for record in records
                        if (value := _nested_number(record, path)) is not None
                    ],
                    seed=HOLDOUT_BASE_SEED + workload_index * 1000 + policy_index * 100 + index,
                )
                for index, (name, path) in enumerate(metric_paths.items())
            }
            groups[workload][policy] = {
                "valid_repeats": [record.get("repeat") for record in records],
                "trace_fingerprints": [record.get("trace_fingerprint") for record in records],
                "metrics": metrics,
            }

    comparisons: dict[str, Any] = {}
    for workload in HOLDOUT_WORKLOADS:
        upstream = groups[workload]["upstream_default"]["metrics"]
        v1 = groups[workload]["deadline_aging_v1"]["metrics"]
        v2 = groups[workload]["deadline_aging_v2"]["metrics"]
        comparisons[workload] = {
            "v2_vs_upstream_ttft_p95_improvement": _improvement(
                upstream["ttft_p95_ms"]["median"], v2["ttft_p95_ms"]["median"]
            ),
            "v2_vs_upstream_tpot_p95_improvement": _improvement(
                upstream["tpot_p95_ms"]["median"], v2["tpot_p95_ms"]["median"]
            ),
            "v2_vs_upstream_e2e_p95_improvement": _improvement(
                upstream["e2e_p95_ms"]["median"], v2["e2e_p95_ms"]["median"]
            ),
            "v2_vs_upstream_completed_rps_change": (
                None
                if upstream["completed_rps"]["median"] in (None, 0)
                else (
                    v2["completed_rps"]["median"]
                    - upstream["completed_rps"]["median"]
                ) / upstream["completed_rps"]["median"]
            ),
            "v1_to_v2_starvation_reduction": (
                (v1["starvation_count"]["median"] or 0)
                - (v2["starvation_count"]["median"] or 0)
            ),
            "v2_starvation_zero": v2["starvation_count"]["max"] == 0,
        }

    same_trace = all(
        len(
            {
                run.get("trace_fingerprint")
                for run in valid
                if run.get("workload") == workload and run.get("repeat") == repeat
            }
        ) == 1
        for workload in HOLDOUT_WORKLOADS
        for repeat in (1, 2, 3)
    )
    all_complete = all(
        groups[workload][policy]["valid_repeats"] == [1, 2, 3]
        for workload in HOLDOUT_WORKLOADS
        for policy in POLICIES
    )
    v2_hashes = {
        run.get("FINAL_CONFIG_HASH")
        for run in valid
        if run.get("policy") == "deadline_aging_v2"
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "HOLDOUT",
        "valid_runs": len(valid),
        "expected_runs": 27,
        "all_repeats_complete": all_complete,
        "same_trace_within_repeat": same_trace,
        "final_config_hash_matches": v2_hashes == {final_config["FINAL_CONFIG_HASH"]},
        "FINAL_CONFIG_HASH": final_config["FINAL_CONFIG_HASH"],
        "starvation_threshold_ms": STARVATION_THRESHOLD_MS,
        "groups": groups,
        "comparisons": comparisons,
    }


def compact_summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    keep = ("median", "mean", "stdev", "min", "max", "bootstrap_median_95ci")
    groups: dict[str, Any] = {}
    for workload, policies in aggregate["groups"].items():
        groups[workload] = {}
        for policy, payload in policies.items():
            groups[workload][policy] = {
                "valid_repeat_count": len(payload["valid_repeats"]),
                "metrics": {
                    metric: {key: stats[key] for key in keep}
                    for metric, stats in payload["metrics"].items()
                },
            }
    return {
        key: value
        for key, value in aggregate.items()
        if key not in {"groups"}
    } | {
        "groups": groups,
        "privacy": {
            "aggregate_only": True,
            "raw_prompts_stored": False,
            "raw_token_ids_stored": False,
            "request_identifiers_stored": False,
        },
    }


def summarize_correctness(paths: Sequence[Path]) -> dict[str, Any]:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected = set(POLICIES)
    policies = {record.get("policy") for record in records}
    token_sequences = [record.get("token_sequence_hashes") for record in records]
    counts = [record.get("token_counts") for record in records]
    return {
        "policies": sorted(str(policy) for policy in policies),
        "expected_policy_set": policies == expected,
        "prompt_count": records[0].get("prompt_count") if records else 0,
        "token_counts": counts[0] if counts else [],
        "all_identical": bool(records)
        and policies == expected
        and len({json.dumps(value) for value in token_sequences}) == 1
        and len({json.dumps(value) for value in counts}) == 1,
        "raw_prompts_stored": False,
        "raw_token_ids_stored": False,
        "token_hashes_exported": False,
    }


def _emit_runs(manifest: dict[str, Any]) -> None:
    for run in manifest["runs"]:
        print(
            "\t".join(
                str(run.get(key, "-"))
                for key in (
                    "run_id", "workload", "policy", "repeat", "candidate", "config_hash"
                )
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("plan-dev", "plan-holdout"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--repo-root", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--final-config", type=Path)
        sub.add_argument("--emit-lines", action="store_true")

    select = subparsers.add_parser("select-dev")
    select.add_argument("--results-dir", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)

    emit = subparsers.add_parser("config-args")
    emit.add_argument("--candidate", choices=tuple(DEV_CANDIDATES))
    emit.add_argument("--final-config", type=Path)

    validate = subparsers.add_parser("validate-run")
    validate.add_argument("--summary", type=Path, required=True)
    validate.add_argument("--scheduler", type=Path, required=True)
    validate.add_argument("--policy", choices=POLICIES, required=True)
    validate.add_argument("--config-hash")
    validate.add_argument("--candidate")

    completed = subparsers.add_parser("completed")
    completed.add_argument("--summary", type=Path, required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--results-dir", type=Path, required=True)
    summarize.add_argument("--final-config", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--compact-output", type=Path, required=True)

    correctness = subparsers.add_parser("correctness")
    correctness.add_argument("paths", type=Path, nargs=3)
    correctness.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan-dev":
        manifest = create_or_validate_manifest(
            args.output, build_dev_manifest(args.repo_root)
        )
        if args.emit_lines:
            _emit_runs(manifest)
    elif args.command == "plan-holdout":
        if args.final_config is None:
            raise ValueError("--final-config is required for HOLDOUT")
        frozen = json.loads(args.final_config.read_text(encoding="utf-8"))
        manifest = create_or_validate_manifest(
            args.output, build_holdout_manifest(args.repo_root, frozen)
        )
        if args.emit_lines:
            _emit_runs(manifest)
    elif args.command == "select-dev":
        print(json.dumps(select_dev_config(args.results_dir, args.output), sort_keys=True))
    elif args.command == "config-args":
        if bool(args.candidate) == bool(args.final_config):
            raise ValueError("select exactly one config source")
        config = (
            DEV_CANDIDATES[args.candidate]
            if args.candidate
            else json.loads(args.final_config.read_text(encoding="utf-8"))["config"]
        )
        for key, flag in CONFIG_FLAGS.items():
            print(f"--{flag}={config[key]}")
    elif args.command == "validate-run":
        result = validate_run_files(
            args.summary,
            args.scheduler,
            args.policy,
            expected_config_hash=args.config_hash,
            candidate=args.candidate,
        )
        print(json.dumps({"formal_valid": result["formal_valid"]}))
        if not result["formal_valid"]:
            raise SystemExit(1)
    elif args.command == "completed":
        raise SystemExit(0 if run_is_complete(args.summary) else 1)
    elif args.command == "summarize":
        frozen = json.loads(args.final_config.read_text(encoding="utf-8"))
        aggregate = aggregate_holdout(args.results_dir, frozen)
        atomic_write_json(args.output, aggregate)
        atomic_write_json(args.compact_output, compact_summary(aggregate))
        print(json.dumps({"valid_runs": aggregate["valid_runs"]}))
        if not (
            aggregate["valid_runs"] == 27
            and aggregate["all_repeats_complete"]
            and aggregate["same_trace_within_repeat"]
            and aggregate["final_config_hash_matches"]
        ):
            raise SystemExit(1)
    elif args.command == "correctness":
        result = summarize_correctness(args.paths)
        atomic_write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        if not result["all_identical"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
