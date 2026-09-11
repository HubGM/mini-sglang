from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


TERMINAL_REASONS = frozenset({"stop", "cancelled", "failed"})
SYNTHETIC_WORDS = (
    "engine",
    "scheduler",
    "request",
    "token",
    "cache",
    "latency",
    "batch",
    "runtime",
)


@dataclass(frozen=True)
class RequestResult:
    mode: str
    outcome: str
    elapsed_ms: float
    cancel_cleanup_ms: float | None
    data_events: int
    output_hash: str
    cancel_accepted: bool | None
    duplicate_cancel_rejected: bool | None
    error_code: str | None


def synthetic_prompt(seed: int, word_count: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(SYNTHETIC_WORDS) for _ in range(word_count))


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def parse_sse_line(line: str) -> tuple[str, str | None] | None:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return "", None
    data = json.loads(payload)
    choice = data["choices"][0]
    return choice.get("delta", {}).get("content", ""), choice.get("finish_reason")


async def get_health(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    response = await client.get(f"{base_url}/health")
    data = response.json()
    data["http_status"] = response.status_code
    return data


async def wait_until_idle(
    client: httpx.AsyncClient,
    base_url: str,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = await get_health(client, base_url)
        if (
            latest.get("ready")
            and latest.get("waiting_count") == 0
            and latest.get("running_count") == 0
        ):
            return latest
        await asyncio.sleep(0.1)
    raise TimeoutError(
        "scheduler_did_not_drain:"
        f"waiting={latest.get('waiting_count')},"
        f"running={latest.get('running_count')}"
    )


async def cancel_uid(
    client: httpx.AsyncClient,
    base_url: str,
    uid: str,
) -> bool:
    response = await client.post(f"{base_url}/v1/requests/{uid}/cancel")
    response.raise_for_status()
    return bool(response.json()["cancel_accepted"])


async def run_request(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    seed: int,
    word_count: int,
    output_tokens: int,
    mode: str,
    prefill_cancel_delay_s: float = 0.01,
    duplicate_cancel: bool = False,
) -> RequestResult:
    started = time.monotonic()
    cancel_started: float | None = None
    cancel_accepted: bool | None = None
    duplicate_rejected: bool | None = None
    data_events = 0
    output_parts: list[str] = []
    finish_reason: str | None = None
    error_code: str | None = None
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": synthetic_prompt(seed, word_count),
            }
        ],
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }

    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            uid = response.headers.get("x-minisgl-request-id")
            if uid is None:
                raise RuntimeError("missing_request_id_header")

            if mode == "waiting":
                cancel_started = time.monotonic()
                cancel_accepted = await cancel_uid(client, base_url, uid)
                if duplicate_cancel:
                    duplicate_rejected = not await cancel_uid(
                        client, base_url, uid
                    )
            elif mode == "prefill":
                await asyncio.sleep(prefill_cancel_delay_s)
                cancel_started = time.monotonic()
                cancel_accepted = await cancel_uid(client, base_url, uid)

            async for line in response.aiter_lines():
                parsed = parse_sse_line(line)
                if parsed is None:
                    continue
                content, reason = parsed
                if reason in TERMINAL_REASONS:
                    finish_reason = reason
                    continue
                if line.strip() == "data: [DONE]":
                    continue
                data_events += 1
                if content:
                    output_parts.append(content)

                if mode in {"decode", "disconnect"} and data_events == 1:
                    if mode == "disconnect":
                        break
                    cancel_started = time.monotonic()
                    cancel_accepted = await cancel_uid(client, base_url, uid)

        if mode == "disconnect":
            outcome = "disconnect"
        elif finish_reason == "cancelled":
            outcome = "cancelled"
        elif finish_reason == "failed":
            outcome = "failed"
        elif finish_reason == "stop":
            outcome = "completed"
        else:
            outcome = "missing_terminal"
    except Exception as exc:
        outcome = "failed"
        error_code = type(exc).__name__

    ended = time.monotonic()
    return RequestResult(
        mode=mode,
        outcome=outcome,
        elapsed_ms=(ended - started) * 1_000,
        cancel_cleanup_ms=(
            (ended - cancel_started) * 1_000
            if cancel_started is not None
            else None
        ),
        data_events=data_events,
        output_hash=hashlib.sha256("".join(output_parts).encode()).hexdigest(),
        cancel_accepted=cancel_accepted,
        duplicate_cancel_rejected=duplicate_rejected,
        error_code=error_code,
    )


def summarize_results(results: list[RequestResult]) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    modes: dict[str, int] = {}
    errors: dict[str, int] = {}
    for result in results:
        outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
        modes[result.mode] = modes.get(result.mode, 0) + 1
        if result.error_code is not None:
            errors[result.error_code] = errors.get(result.error_code, 0) + 1
    cleanup = [
        result.cancel_cleanup_ms
        for result in results
        if result.cancel_cleanup_ms is not None
    ]
    return {
        "requests": len(results),
        "outcomes": dict(sorted(outcomes.items())),
        "modes": dict(sorted(modes.items())),
        "error_codes": dict(sorted(errors.items())),
        "terminal_or_disconnect": sum(
            outcome in {"completed", "cancelled", "disconnect"}
            for outcome in (result.outcome for result in results)
        ),
        "cancel_cleanup_ms": {
            "mean": statistics.mean(cleanup) if cleanup else None,
            "p50": percentile(cleanup, 0.50),
            "p95": percentile(cleanup, 0.95),
            "max": max(cleanup, default=None),
        },
    }


async def run_stage_gate(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
) -> tuple[list[RequestResult], dict[str, Any]]:
    modes = ("waiting", "prefill", "decode", "disconnect")
    results = []
    for index, mode in enumerate(modes):
        result = await run_request(
            client,
            base_url=args.base_url,
            model=args.model,
            seed=args.seed + index,
            word_count=1536 if mode == "prefill" else 96,
            output_tokens=64,
            mode=mode,
            duplicate_cancel=mode == "waiting",
        )
        results.append(result)
        await wait_until_idle(client, args.base_url, args.drain_timeout)

    summary = summarize_results(results)
    summary["duplicate_cancel_rejected"] = bool(
        results[0].duplicate_cancel_rejected
    )
    summary["gate_passed"] = (
        results[0].outcome == "cancelled"
        and results[1].outcome == "cancelled"
        and results[2].outcome == "cancelled"
        and results[3].outcome == "disconnect"
        and summary["duplicate_cancel_rejected"]
    )
    return results, summary


async def run_survivor_gate(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline = await run_request(
        client,
        base_url=args.base_url,
        model=args.model,
        seed=args.seed + 100,
        word_count=128,
        output_tokens=32,
        mode="normal",
    )
    await wait_until_idle(client, args.base_url, args.drain_timeout)

    survivor, cancelled_peer = await asyncio.gather(
        run_request(
            client,
            base_url=args.base_url,
            model=args.model,
            seed=args.seed + 100,
            word_count=128,
            output_tokens=32,
            mode="normal",
        ),
        run_request(
            client,
            base_url=args.base_url,
            model=args.model,
            seed=args.seed + 101,
            word_count=192,
            output_tokens=96,
            mode="decode",
        ),
    )
    await wait_until_idle(client, args.base_url, args.drain_timeout)
    return {
        "baseline_outcome": baseline.outcome,
        "survivor_outcome": survivor.outcome,
        "peer_outcome": cancelled_peer.outcome,
        "output_hash_identical": baseline.output_hash == survivor.output_hash,
        "gate_passed": (
            baseline.outcome == "completed"
            and survivor.outcome == "completed"
            and cancelled_peer.outcome == "cancelled"
            and baseline.output_hash == survivor.output_hash
        ),
    }


async def run_stress_ratio(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    ratio: int,
    ratio_index: int,
) -> tuple[list[RequestResult], dict[str, Any]]:
    request_count = args.requests_per_ratio
    cancel_count = math.floor(request_count * ratio / 100 + 0.5)
    modes = ["normal"] * request_count
    cancellation_modes = ("waiting", "prefill", "decode", "disconnect")
    for index in range(cancel_count):
        modes[index] = cancellation_modes[index % len(cancellation_modes)]
    random.Random(args.seed + 1_000 + ratio).shuffle(modes)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(index: int, mode: str) -> RequestResult:
        async with semaphore:
            lengths = (32, 128, 512)
            word_count = lengths[index % len(lengths)]
            if mode == "prefill":
                word_count = 1024
            return await run_request(
                client,
                base_url=args.base_url,
                model=args.model,
                seed=args.seed + 10_000 * (ratio_index + 1) + index,
                word_count=word_count,
                output_tokens=(12, 24, 48)[index % 3],
                mode=mode,
            )

    results = await asyncio.gather(
        *(bounded(index, mode) for index, mode in enumerate(modes))
    )
    summary = summarize_results(results)
    summary.update(
        {
            "cancellation_ratio_percent": ratio,
            "planned_cancellations": cancel_count,
            "gate_passed": (
                summary["terminal_or_disconnect"] == request_count
                and not summary["error_codes"]
                and summary["outcomes"].get("missing_terminal", 0) == 0
            ),
        }
    )
    return results, summary


async def run(args: argparse.Namespace) -> dict[str, Any]:
    timeout = httpx.Timeout(args.request_timeout)
    limits = httpx.Limits(
        max_connections=args.concurrency + 8,
        max_keepalive_connections=args.concurrency + 8,
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        health_before = await get_health(client, args.base_url)
        if not health_before.get("ready"):
            raise RuntimeError("worker_not_ready")

        if args.smoke_requests:
            results, stress_summary = await run_stress_ratio(
                client,
                args,
                args.smoke_cancellation_ratio,
                0,
            )
            health_after = await wait_until_idle(
                client, args.base_url, args.drain_timeout
            )
            failed_delta = (
                health_after.get("failed_count", 0)
                - health_before.get("failed_count", 0)
            )
            scheduler_idle = (
                health_after.get("waiting_count") == 0
                and health_after.get("running_count") == 0
            )
            return {
                "schema_version": 1,
                "seed": args.seed,
                "policy": args.policy,
                "model": Path(args.model).name,
                "mode": "compatibility_smoke",
                "health_before": health_before,
                "stress": {
                    "aggregate": summarize_results(results),
                    "ratios": [stress_summary],
                },
                "health_after": health_after,
                "scheduler_failed_delta": failed_delta,
                "gates": {
                    "request_count_exact": len(results) == args.smoke_requests,
                    "stress_terminal_count": stress_summary["gate_passed"],
                    "final_waiting_zero": health_after.get("waiting_count") == 0,
                    "final_running_zero": health_after.get("running_count") == 0,
                    "scheduler_ready": health_after.get("ready") is True,
                    "scheduler_fatal_absent": health_after.get("fatal_error") is None,
                    "failed_delta_zero": failed_delta == 0,
                    "orphan_request_zero": scheduler_idle,
                },
                "raw_request_records_stored": False,
            }

        stage_results, stage_gate = await run_stage_gate(client, args)
        survivor_gate = await run_survivor_gate(client, args)
        stress_summaries = []
        stress_results: list[RequestResult] = []
        for ratio_index, ratio in enumerate((0, 10, 30, 50)):
            results, summary = await run_stress_ratio(
                client, args, ratio, ratio_index
            )
            stress_results.extend(results)
            stress_summaries.append(summary)
            await wait_until_idle(client, args.base_url, args.drain_timeout)

        health_after = await wait_until_idle(
            client, args.base_url, args.drain_timeout
        )

    cancelled_delta = (
        health_after.get("cancelled_count", 0)
        - health_before.get("cancelled_count", 0)
    )
    failed_delta = (
        health_after.get("failed_count", 0)
        - health_before.get("failed_count", 0)
    )
    stress_summary = summarize_results(stress_results)
    scheduler_idle = (
        health_after.get("waiting_count") == 0
        and health_after.get("running_count") == 0
    )
    all_stress_valid = all(item["gate_passed"] for item in stress_summaries)
    return {
        "schema_version": 1,
        "seed": args.seed,
        "policy": args.policy,
        "model": Path(args.model).name,
        "health_before": health_before,
        "stage_gate": stage_gate,
        "survivor_gate": survivor_gate,
        "stress": {
            "aggregate": stress_summary,
            "ratios": stress_summaries,
        },
        "health_after": health_after,
        "scheduler_cancelled_delta": cancelled_delta,
        "scheduler_failed_delta": failed_delta,
        "gates": {
            "stage_cancellation": stage_gate["gate_passed"],
            "survivor_output_unchanged": survivor_gate["gate_passed"],
            "stress_terminal_count": all_stress_valid,
            "final_waiting_zero": health_after.get("waiting_count") == 0,
            "final_running_zero": health_after.get("running_count") == 0,
            "scheduler_ready": health_after.get("ready") is True,
            "scheduler_fatal_absent": health_after.get("fatal_error") is None,
            "failed_delta_zero": failed_delta == 0,
            "orphan_request_zero": scheduler_idle,
            "kv_table_leak_signal_zero": scheduler_idle
            and health_after.get("fatal_error") is None,
        },
        "raw_request_records_stored": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--model", default="/models/Qwen3-8B")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--requests-per-ratio", type=int, default=25)
    parser.add_argument("--policy", default="upstream_default")
    parser.add_argument("--smoke-requests", type=int, default=0)
    parser.add_argument("--smoke-cancellation-ratio", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--drain-timeout", type=float, default=30.0)
    parser.add_argument("--total-timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_requests:
        args.requests_per_ratio = args.smoke_requests
    try:
        summary = asyncio.run(
            asyncio.wait_for(run(args), timeout=args.total_timeout)
        )
    except Exception as exc:
        summary = {
            "schema_version": 1,
            "seed": args.seed,
            "policy": args.policy,
            "fatal_error": type(exc).__name__,
            "raw_request_records_stored": False,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(summary, sort_keys=True))
    if "fatal_error" in summary or not all(summary["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
