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

from benchmark.deadline_experiment import POLICIES, WORKLOADS


SCHEMA_VERSION = 1
DEFAULT_SEED = 20260909
MAX_STEP_TOKENS = 2048
MAX_PREFILL_CHUNK_TOKENS = 512
STARVATION_THRESHOLD_MS = 400.0


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def implementation_fingerprint(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative_path in (
        "python/minisgl/scheduler/advanced_policy.py",
        "python/minisgl/scheduler/policy.py",
        "python/minisgl/scheduler/scheduler.py",
        "benchmark/deadline_experiment.py",
        "benchmark/deadline_matrix.py",
    ):
        path = repo_root / relative_path
        digest.update(relative_path.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_manifest(repo_root: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    sequence = 0
    for repeat in (1, 2, 3):
        for workload_index, workload in enumerate(WORKLOADS):
            order = list(POLICIES)
            order_seed = seed + repeat * 10_000 + workload_index * 101
            random.Random(order_seed).shuffle(order)
            for position, policy in enumerate(order, start=1):
                sequence += 1
                run_id = f"{workload}-{policy}-r{repeat}"
                runs.append(
                    {
                        "sequence": sequence,
                        "run_id": run_id,
                        "workload": workload,
                        "policy": policy,
                        "repeat": repeat,
                        "trace_seed": seed + repeat,
                        "order_seed": order_seed,
                        "position_in_block": position,
                    }
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "deadline-token-scheduler",
        "base_seed": seed,
        "implementation_fingerprint": implementation_fingerprint(repo_root),
        "policies": list(POLICIES),
        "workloads": list(WORKLOADS),
        "repeats": [1, 2, 3],
        "core_run_count": len(runs),
        "slo": {"ttft_ms": 200.0, "tpot_ms": 50.0, "e2e_ms": 1200.0},
        "scheduler": {
            "max_step_tokens": MAX_STEP_TOKENS,
            "max_prefill_chunk_tokens": MAX_PREFILL_CHUNK_TOKENS,
            "decode_reserve_ratio": 0.5,
            "max_consecutive_prefill_steps": 1,
            "max_wait_ms": 400.0,
            "aging_start_ms": 100.0,
            "aging_rate": 1.0,
            "starvation_threshold_ms": STARVATION_THRESHOLD_MS,
            "request_sample_rate": 1.0,
        },
        "raw_prompts_stored": False,
        "raw_token_ids_stored": False,
        "runs": runs,
    }


def create_or_load_manifest(path: Path, repo_root: Path, seed: int) -> dict[str, Any]:
    expected = build_manifest(repo_root, seed)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable_fields = (
            "schema_version",
            "base_seed",
            "implementation_fingerprint",
            "runs",
        )
        if any(existing.get(field) != expected.get(field) for field in stable_fields):
            raise ValueError("Existing experiment manifest does not match this plan")
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


def _active_step_records(scheduler: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record
        for record in scheduler.get("step_records", [])
        if int(record.get("token_budget_used", 0)) > 0
    ]


def validate_run_payload(
    summary: dict[str, Any],
    scheduler: dict[str, Any],
    expected_policy: str,
) -> dict[str, Any]:
    request_samples = scheduler.get("request_samples", [])
    active_steps = _active_step_records(scheduler)
    waiting_values = [
        float(sample["waiting_time_ms"])
        for sample in request_samples
        if sample.get("waiting_time_ms") is not None
    ]
    terminal_completed = sum(
        sample.get("terminal_state") == "completed" for sample in request_samples
    )
    lifecycle_output_tokens = sum(
        int(sample.get("generated_tokens", 0)) for sample in request_samples
    )
    budget_respected = all(
        0 <= int(record.get("token_budget_used", 0))
        <= int(record.get("token_budget", 0))
        for record in active_steps
    )
    chunk_respected = True
    if expected_policy != "upstream_default":
        budget_respected = budget_respected and all(
            int(record.get("token_budget", 0)) <= MAX_STEP_TOKENS
            for record in active_steps
        )
        chunk_respected = all(
            int(record.get("prefill_chunk_size", 0))
            <= MAX_PREFILL_CHUNK_TOKENS
            for record in active_steps
        )

    gates = {
        "client_valid": bool(summary.get("valid")),
        "policy_matches": scheduler.get("policy") == expected_policy,
        "all_requests_sampled": len(request_samples) == summary.get("requests"),
        "all_terminal_completed": terminal_completed == summary.get("requests"),
        "client_scheduler_token_count_matches": lifecycle_output_tokens
        == summary.get("output_tokens"),
        "token_budget_respected": budget_respected,
        "prefill_chunk_respected": chunk_respected,
        "no_scheduler_fallback": scheduler.get("fallback_count", 0) == 0,
        "no_policy_failure": scheduler.get("policy_failure_count", 0) == 0,
        "no_cancellation": scheduler.get("cancelled_count", 0) == 0,
        "no_failure": scheduler.get("failed_count", 0) == 0,
        "telemetry_not_truncated": scheduler.get("dropped_step_records", 0) == 0,
        "final_waiting_zero": summary.get("health", {}).get("final_waiting") == 0,
        "final_running_zero": summary.get("health", {}).get("final_running") == 0,
    }
    summary["scheduler"] = {
        "decision_latency_us": {
            "p50": scheduler.get("decision_latency_p50_us"),
            "p95": scheduler.get("decision_latency_p95_us"),
            "p99": scheduler.get("decision_latency_p99_us"),
        },
        "token_budget_utilization": scheduler.get("token_budget_utilization"),
        "average_prefill_chunk_tokens": scheduler.get(
            "average_prefill_chunk_tokens"
        ),
        "decode_steps": scheduler.get("decode_batch_count"),
        "prefill_steps": scheduler.get("prefill_batch_count"),
        "waiting_time_ms": distribution(waiting_values),
        "starvation_count": sum(
            bool(sample.get("starvation")) for sample in request_samples
        ),
        "minimum_slack_ms": scheduler.get("minimum_slack_ms"),
        "max_urgent_requests": scheduler.get("max_urgent_requests"),
        "fallback_count": scheduler.get("fallback_count"),
        "policy_failure_count": scheduler.get("policy_failure_count"),
        "cancelled_count": scheduler.get("cancelled_count"),
        "failed_count": scheduler.get("failed_count"),
        "decision_count": scheduler.get("decision_count"),
        "step_record_count": len(scheduler.get("step_records", [])),
        "request_sample_count": len(request_samples),
    }
    summary["validity_gate"] = gates
    summary["formal_valid"] = all(gates.values())
    return summary


def validate_run_files(
    summary_path: Path, scheduler_path: Path, expected_policy: str
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    validated = validate_run_payload(summary, scheduler, expected_policy)
    atomic_write_json(summary_path, validated)
    return validated


def run_is_complete(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        return bool(
            json.loads(summary_path.read_text(encoding="utf-8")).get(
                "formal_valid"
            )
        )
    except (OSError, json.JSONDecodeError):
        return False


def bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int,
    samples: int = 10_000,
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


METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "completed_rps": ("completed_rps",),
    "input_tokens_per_s": ("input_tokens_per_s",),
    "output_tokens_per_s": ("output_tokens_per_s",),
    "ttft_p50_ms": ("ttft_ms", "p50"),
    "ttft_p95_ms": ("ttft_ms", "p95"),
    "ttft_p99_ms": ("ttft_ms", "p99"),
    "tpot_p50_ms": ("tpot_ms", "p50"),
    "tpot_p95_ms": ("tpot_ms", "p95"),
    "tpot_p99_ms": ("tpot_ms", "p99"),
    "e2e_p50_ms": ("e2e_ms", "p50"),
    "e2e_p95_ms": ("e2e_ms", "p95"),
    "e2e_p99_ms": ("e2e_ms", "p99"),
    "goodput_rps": ("goodput_rps",),
    "slo_attainment": ("slo_attainment",),
    "scheduler_decision_p50_us": ("scheduler", "decision_latency_us", "p50"),
    "scheduler_decision_p95_us": ("scheduler", "decision_latency_us", "p95"),
    "scheduler_decision_p99_us": ("scheduler", "decision_latency_us", "p99"),
    "token_budget_utilization": ("scheduler", "token_budget_utilization"),
    "average_prefill_chunk_tokens": ("scheduler", "average_prefill_chunk_tokens"),
    "decode_steps": ("scheduler", "decode_steps"),
    "prefill_steps": ("scheduler", "prefill_steps"),
    "waiting_p95_ms": ("scheduler", "waiting_time_ms", "p95"),
    "waiting_p99_ms": ("scheduler", "waiting_time_ms", "p99"),
    "max_waiting_ms": ("scheduler", "waiting_time_ms", "max"),
    "starvation_count": ("scheduler", "starvation_count"),
    "long_request_completion_rate": ("long_request_completion_rate",),
    "gpu_utilization_mean_percent": ("gpu", "utilization_mean_percent"),
    "gpu_utilization_max_percent": ("gpu", "utilization_max_percent"),
    "gpu_memory_mean_mib": ("gpu", "memory_mean_mib"),
    "gpu_memory_max_mib": ("gpu", "memory_max_mib"),
}

COMPACT_STAT_FIELDS = (
    "median",
    "mean",
    "stdev",
    "bootstrap_median_95ci",
)


def nested_number(payload: dict[str, Any], path: Sequence[str]) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _improvement(baseline: float | None, optimized: float | None, higher: bool) -> float | None:
    if baseline in (None, 0) or optimized is None:
        return None
    numerator = optimized - baseline if higher else baseline - optimized
    return numerator / baseline


def aggregate_results(results_dir: Path) -> dict[str, Any]:
    all_runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(results_dir.glob("*.summary.json"))
    ]
    valid_runs = [run for run in all_runs if run.get("formal_valid")]
    groups: dict[str, dict[str, Any]] = {}
    for workload in WORKLOADS:
        groups[workload] = {}
        for policy_index, policy in enumerate(POLICIES):
            records = sorted(
                (
                    run
                    for run in valid_runs
                    if run.get("workload") == workload
                    and run.get("policy") == policy
                ),
                key=lambda run: run.get("repeat", 0),
            )
            metrics: dict[str, Any] = {}
            for metric_index, (name, path) in enumerate(METRIC_PATHS.items()):
                values = [
                    value
                    for record in records
                    if (value := nested_number(record, path)) is not None
                ]
                metrics[name] = repeated_stats(
                    values,
                    seed=DEFAULT_SEED + policy_index * 100 + metric_index,
                )
            groups[workload][policy] = {
                "valid_repeats": [record["repeat"] for record in records],
                "run_ids": [record["run_id"] for record in records],
                "trace_fingerprints": [
                    record.get("trace_fingerprint") for record in records
                ],
                "metrics": metrics,
            }

    comparisons: dict[str, Any] = {}
    for workload in WORKLOADS:
        upstream = groups[workload]["upstream_default"]["metrics"]
        aging = groups[workload]["deadline_aging"]["metrics"]
        comparisons[workload] = {
            "ttft_p95_improvement": _improvement(
                upstream["ttft_p95_ms"]["median"],
                aging["ttft_p95_ms"]["median"],
                False,
            ),
            "ttft_p99_improvement": _improvement(
                upstream["ttft_p99_ms"]["median"],
                aging["ttft_p99_ms"]["median"],
                False,
            ),
            "tpot_p95_improvement": _improvement(
                upstream["tpot_p95_ms"]["median"],
                aging["tpot_p95_ms"]["median"],
                False,
            ),
            "e2e_p95_improvement": _improvement(
                upstream["e2e_p95_ms"]["median"],
                aging["e2e_p95_ms"]["median"],
                False,
            ),
            "goodput_improvement": _improvement(
                upstream["goodput_rps"]["median"],
                aging["goodput_rps"]["median"],
                True,
            ),
            "max_waiting_reduction": _improvement(
                upstream["max_waiting_ms"]["median"],
                aging["max_waiting_ms"]["median"],
                False,
            ),
            "starvation_count_reduction": (
                (upstream["starvation_count"]["median"] or 0)
                - (aging["starvation_count"]["median"] or 0)
            ),
            "completed_throughput_change": _improvement(
                upstream["completed_rps"]["median"],
                aging["completed_rps"]["median"],
                True,
            ),
        }

    trace_consistency = True
    for repeat in (1, 2, 3):
        for workload in WORKLOADS:
            fingerprints = {
                run.get("trace_fingerprint")
                for run in valid_runs
                if run.get("workload") == workload
                and run.get("repeat") == repeat
            }
            trace_consistency &= len(fingerprints) == 1

    repeat_complete = all(
        groups[workload][policy]["valid_repeats"] == [1, 2, 3]
        for workload in WORKLOADS
        for policy in POLICIES
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "deadline-token-scheduler",
        "expected_core_runs": 36,
        "observed_runs": len(all_runs),
        "valid_core_runs": len(valid_runs),
        "all_repeats_complete": repeat_complete,
        "same_trace_within_condition": trace_consistency,
        "starvation_threshold_ms": STARVATION_THRESHOLD_MS,
        "groups": groups,
        "deadline_aging_vs_upstream": comparisons,
    }


def compact_summary(
    aggregate: dict[str, Any],
    correctness: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for workload, policies in aggregate.get("groups", {}).items():
        groups[workload] = {}
        for policy, group in policies.items():
            metrics = {
                name: {
                    field: statistics_payload.get(field)
                    for field in COMPACT_STAT_FIELDS
                }
                for name, statistics_payload in group.get("metrics", {}).items()
            }
            groups[workload][policy] = {
                "valid_repeats": group.get("valid_repeats", []),
                "metrics": metrics,
            }

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": aggregate.get("experiment"),
        "methodology": {
            "base_seed": manifest.get("base_seed"),
            "policies": manifest.get("policies", []),
            "workloads": manifest.get("workloads", []),
            "repeats": manifest.get("repeats", []),
            "slo": manifest.get("slo", {}),
            "scheduler": manifest.get("scheduler", {}),
        },
        "validity": {
            "expected_core_runs": aggregate.get("expected_core_runs"),
            "observed_runs": aggregate.get("observed_runs"),
            "valid_core_runs": aggregate.get("valid_core_runs"),
            "all_repeats_complete": aggregate.get("all_repeats_complete"),
            "same_trace_within_repeat": aggregate.get(
                "same_trace_within_condition"
            ),
        },
        "gpu_correctness": correctness,
        "groups": groups,
        "deadline_aging_vs_upstream": aggregate.get(
            "deadline_aging_vs_upstream", {}
        ),
        "privacy": {
            "aggregate_only": True,
            "raw_prompts_stored": False,
            "raw_token_ids_stored": False,
            "request_identifiers_stored": False,
            "trace_fingerprints_stored": False,
        },
    }


def summarize_correctness(paths: Sequence[Path]) -> dict[str, Any]:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_policy = {record["policy"]: record for record in records}
    baseline = by_policy.get("upstream_default", {})
    baseline_hashes = baseline.get("token_sequence_hashes")
    per_policy = {
        policy: {
            "prompt_count": record.get("prompt_count"),
            "token_counts": record.get("token_counts"),
            "identical_to_upstream": (
                record.get("token_sequence_hashes") == baseline_hashes
            ),
        }
        for policy, record in sorted(by_policy.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "temperature": 0.0,
        "policy_count": len(by_policy),
        "expected_policies": list(POLICIES),
        "all_policies_present": set(by_policy) == set(POLICIES),
        "all_identical": bool(per_policy)
        and all(item["identical_to_upstream"] for item in per_policy.values()),
        "per_policy": per_policy,
        "raw_prompts_stored": False,
        "raw_token_ids_stored": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--repo-root", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--seed", type=int, default=DEFAULT_SEED)
    plan.add_argument("--emit-lines", action="store_true")

    completed = subparsers.add_parser("completed")
    completed.add_argument("--summary", type=Path, required=True)

    validate = subparsers.add_parser("validate-run")
    validate.add_argument("--summary", type=Path, required=True)
    validate.add_argument("--scheduler", type=Path, required=True)
    validate.add_argument("--policy", choices=POLICIES, required=True)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--results-dir", type=Path, required=True)
    summarize_parser.add_argument("--output", type=Path, required=True)

    correctness = subparsers.add_parser("correctness")
    correctness.add_argument("paths", type=Path, nargs="+")
    correctness.add_argument("--output", type=Path, required=True)

    compact = subparsers.add_parser("compact")
    compact.add_argument("--summary", type=Path, required=True)
    compact.add_argument("--correctness", type=Path, required=True)
    compact.add_argument("--manifest", type=Path, required=True)
    compact.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        manifest = create_or_load_manifest(
            args.output, args.repo_root, args.seed
        )
        if args.emit_lines:
            for run in manifest["runs"]:
                print(
                    "\t".join(
                        str(run[key])
                        for key in ("run_id", "workload", "policy", "repeat")
                    )
                )
        return
    if args.command == "completed":
        raise SystemExit(0 if run_is_complete(args.summary) else 1)
    if args.command == "validate-run":
        result = validate_run_files(args.summary, args.scheduler, args.policy)
        print(json.dumps({"run_id": result.get("run_id"), "valid": result["formal_valid"]}))
        raise SystemExit(0 if result["formal_valid"] else 1)
    if args.command == "summarize":
        result = aggregate_results(args.results_dir)
        atomic_write_json(args.output, result)
        print(
            json.dumps(
                {
                    "valid_core_runs": result["valid_core_runs"],
                    "all_repeats_complete": result["all_repeats_complete"],
                }
            )
        )
        raise SystemExit(0 if result["all_repeats_complete"] else 1)
    if args.command == "correctness":
        result = summarize_correctness(args.paths)
        atomic_write_json(args.output, result)
        print(json.dumps({"all_identical": result["all_identical"]}))
        raise SystemExit(0 if result["all_identical"] else 1)
    if args.command == "compact":
        result = compact_summary(
            json.loads(args.summary.read_text(encoding="utf-8")),
            json.loads(args.correctness.read_text(encoding="utf-8")),
            json.loads(args.manifest.read_text(encoding="utf-8")),
        )
        atomic_write_json(args.output, result)
        print(json.dumps(result["validity"]))
        raise SystemExit(
            0
            if result["validity"]["all_repeats_complete"]
            and result["gpu_correctness"].get("all_identical")
            else 1
        )


if __name__ == "__main__":
    main()
