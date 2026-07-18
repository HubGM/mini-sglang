from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize_group(records: list[dict]) -> dict:
    metrics = {
        "completed_rps": [record["completed_rps"] for record in records],
        "input_tokens_per_s": [record["input_tokens_per_s"] for record in records],
        "output_tokens_per_s": [record["output_tokens_per_s"] for record in records],
        "ttft_p95_ms": [record["ttft_ms"]["p95"] for record in records],
        "ttft_p99_ms": [record["ttft_ms"]["p99"] for record in records],
        "tpot_p95_ms": [record["tpot_ms"]["p95"] for record in records],
        "tpot_p99_ms": [record["tpot_ms"]["p99"] for record in records],
        "e2e_p95_ms": [record["e2e_ms"]["p95"] for record in records],
        "e2e_p99_ms": [record["e2e_ms"]["p99"] for record in records],
        "queue_delay_p95_ms": [record["queue_delay_ms"]["p95"] for record in records],
    }
    distributions = {}
    for name, values in metrics.items():
        distributions[name] = {
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return {
        "valid_runs": len(records),
        "total_requests": sum(record["requests"] for record in records),
        "successful_requests": sum(
            record["successful_requests"] for record in records
        ),
        "errors": sum(record["errors"] for record in records),
        "metrics": distributions,
    }


def aggregate(input_dir: Path) -> dict:
    groups = defaultdict(list)
    for path in sorted(input_dir.glob("*.summary.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        groups[(record["workload"], record["traffic"])].append(record)

    return {
        "input_directory": str(input_dir),
        "valid_run_count": sum(len(records) for records in groups.values()),
        "conditions": {
            f"{workload}/{traffic}": summarize_group(records)
            for (workload, traffic), records in sorted(groups.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = aggregate(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({"valid_run_count": result["valid_run_count"]}))


if __name__ == "__main__":
    main()
