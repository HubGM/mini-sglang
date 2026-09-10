from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
from minisgl.benchmark.client import generate_prompt
from transformers import AutoTokenizer


SLO_TTFT_MS = 200.0
SLO_TPOT_MS = 50.0
SLO_E2E_MS = 1200.0
POLICIES = (
    "upstream_default",
    "token_budget",
    "deadline_aware",
    "deadline_aging",
)
WORKLOADS = (
    "mixed",
    "long-prefill-interference",
    "starvation-stress",
)


@dataclass(frozen=True)
class WorkItem:
    request_hash: str
    request_class: str
    input_len: int
    output_len: int
    scheduled_offset_s: float
    prompt: str


@dataclass(frozen=True)
class RequestMetric:
    request_hash: str
    request_class: str
    input_len: int
    expected_output_tokens: int
    output_tokens: int
    sse_chunks: int
    scheduled_offset_s: float
    send_offset_s: float
    first_token_offset_s: float | None
    end_offset_s: float
    queue_delay_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_ms: float
    output_hash: str
    status_code: int | None
    terminal_reason: str | None
    error_code: str | None
    slo_good: bool


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def trace_fingerprint(items: Iterable[WorkItem]) -> str:
    trace = [
        {
            "request_hash": item.request_hash,
            "request_class": item.request_class,
            "input_len": item.input_len,
            "output_len": item.output_len,
            "scheduled_offset_s": item.scheduled_offset_s,
        }
        for item in items
    ]
    encoded = json.dumps(trace, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _trace_shape(workload: str, seed: int) -> list[tuple[str, int, int, float]]:
    rng = random.Random(seed)
    if workload == "mixed":
        shapes = (
            [("short", 96, 24)] * 10
            + [("medium", 512, 48)] * 10
            + [("long", 1536, 32)] * 10
        )
        rng.shuffle(shapes)
        return [(*shape, index * 0.125) for index, shape in enumerate(shapes)]

    if workload == "long-prefill-interference":
        short = [("decode", 96, 48, index * 0.125) for index in range(24)]
        long = [
            ("long-prefill", 2048, 16, 0.20 + index * 0.50)
            for index in range(6)
        ]
        return sorted(short + long, key=lambda item: (item[3], item[0]))

    if workload == "starvation-stress":
        short = [("short", 64, 24, index * 0.08) for index in range(36)]
        long = [
            ("long", 2048, 32, offset)
            for offset in (0.20, 0.90, 1.60, 2.30)
        ]
        return sorted(short + long, key=lambda item: (item[3], item[0]))

    raise ValueError(f"Unsupported workload: {workload}")


def build_workload(tokenizer, workload: str, seed: int) -> list[WorkItem]:
    shape = _trace_shape(workload, seed)
    random_state = random.getstate()
    random.seed(seed)
    try:
        return [
            WorkItem(
                request_hash=hashlib.sha256(
                    f"{workload}:{seed}:{index}".encode()
                ).hexdigest()[:16],
                request_class=request_class,
                input_len=input_len,
                output_len=output_len,
                scheduled_offset_s=offset,
                prompt=generate_prompt(tokenizer, input_len),
            )
            for index, (request_class, input_len, output_len, offset) in enumerate(
                shape
            )
        ]
    finally:
        random.setstate(random_state)


def parse_sse(line: str) -> tuple[str, str | None] | None:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return "", "done"
    data = json.loads(payload)
    choice = data["choices"][0]
    return choice.get("delta", {}).get("content", ""), choice.get(
        "finish_reason"
    )


async def get_health(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    response = await client.get(f"{base_url}/health")
    data = response.json()
    data["http_status"] = response.status_code
    return data


async def wait_for_idle(
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
        "worker_drain_timeout:"
        f"waiting={latest.get('waiting_count')},"
        f"running={latest.get('running_count')}"
    )


async def run_one(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    item: WorkItem,
    origin: float,
) -> RequestMetric:
    scheduled = origin + item.scheduled_offset_s
    await asyncio.sleep(max(0.0, scheduled - time.monotonic()))
    sent = time.monotonic()
    first_token: float | None = None
    end = sent
    token_times: list[float] = []
    output_parts: list[str] = []
    status_code: int | None = None
    terminal_reason: str | None = None
    error_code: str | None = None

    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": item.prompt}],
                "max_tokens": item.output_len,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": True,
                "ttft_deadline_ms": SLO_TTFT_MS,
                "e2e_deadline_ms": SLO_E2E_MS,
            },
        ) as response:
            status_code = response.status_code
            response.raise_for_status()
            async for line in response.aiter_lines():
                parsed = parse_sse(line)
                if parsed is None:
                    continue
                content, reason = parsed
                if reason == "done":
                    continue
                if reason is not None:
                    terminal_reason = reason
                    continue
                observed_at = time.monotonic()
                token_times.append(observed_at)
                if first_token is None:
                    first_token = observed_at
                if content:
                    output_parts.append(content)
        end = time.monotonic()
    except Exception as exc:
        end = time.monotonic()
        error_code = type(exc).__name__

    ttft_ms = (first_token - sent) * 1_000 if first_token is not None else None
    e2e_ms = (end - sent) * 1_000
    successful = (
        error_code is None
        and status_code == 200
        and terminal_reason == "stop"
        and first_token is not None
    )
    output_tokens = len(token_times)
    tpot_ms = (
        (end - first_token) * 1_000 / (output_tokens - 1)
        if first_token is not None and output_tokens > 1
        else None
    )
    slo_good = bool(
        successful
        and ttft_ms is not None
        and tpot_ms is not None
        and ttft_ms <= SLO_TTFT_MS
        and tpot_ms <= SLO_TPOT_MS
        and e2e_ms <= SLO_E2E_MS
    )
    return RequestMetric(
        request_hash=item.request_hash,
        request_class=item.request_class,
        input_len=item.input_len,
        expected_output_tokens=item.output_len,
        output_tokens=output_tokens,
        sse_chunks=len(token_times),
        scheduled_offset_s=item.scheduled_offset_s,
        send_offset_s=sent - origin,
        first_token_offset_s=(
            first_token - origin if first_token is not None else None
        ),
        end_offset_s=end - origin,
        queue_delay_ms=max(0.0, (sent - scheduled) * 1_000),
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_ms=e2e_ms,
        output_hash=hashlib.sha256("".join(output_parts).encode()).hexdigest(),
        status_code=status_code,
        terminal_reason=terminal_reason,
        error_code=error_code,
        slo_good=slo_good,
    )


class GpuSampler:
    def __init__(self, gpu_index: int, interval_s: float = 0.2) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.samples: list[tuple[float, float]] = []
        self._stop = asyncio.Event()

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                process = await asyncio.create_subprocess_exec(
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.gpu_index),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await process.communicate()
                if process.returncode == 0:
                    utilization, memory = stdout.decode().strip().split(",")
                    self.samples.append((float(utilization), float(memory)))
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()

    def summary(self) -> dict[str, float | int | None]:
        return {
            "samples": len(self.samples),
            "utilization_mean_percent": (
                statistics.mean(value for value, _ in self.samples)
                if self.samples
                else None
            ),
            "utilization_max_percent": max(
                (value for value, _ in self.samples), default=None
            ),
            "memory_mean_mib": (
                statistics.mean(value for _, value in self.samples)
                if self.samples
                else None
            ),
            "memory_max_mib": max(
                (value for _, value in self.samples), default=None
            ),
        }


def summarize(
    metrics: list[RequestMetric],
    health_before: dict[str, Any],
    health_after: dict[str, Any],
    gpu: dict[str, Any],
) -> dict[str, Any]:
    successful = [
        metric
        for metric in metrics
        if metric.error_code is None
        and metric.status_code == 200
        and metric.terminal_reason == "stop"
        and metric.first_token_offset_s is not None
    ]
    first_send_s = min((metric.send_offset_s for metric in metrics), default=0.0)
    last_end_s = max((metric.end_offset_s for metric in metrics), default=0.0)
    duration_s = max(0.0, last_end_s - first_send_s)

    def distribution(name: str) -> dict[str, float | None]:
        values = [
            float(value)
            for metric in successful
            if (value := getattr(metric, name)) is not None
        ]
        return {
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values, default=None),
        }

    long_requests = [metric for metric in metrics if metric.input_len >= 1024]
    long_success = [metric for metric in successful if metric.input_len >= 1024]
    errors: dict[str, int] = {}
    for metric in metrics:
        if metric.error_code:
            errors[metric.error_code] = errors.get(metric.error_code, 0) + 1
    good_requests = sum(metric.slo_good for metric in successful)

    def health_count(snapshot: dict[str, Any], key: str) -> int:
        value = snapshot.get(key, 0)
        return int(value) if isinstance(value, (int, float)) else 0

    scheduled_offsets = sorted(
        {metric.scheduled_offset_s for metric in metrics}
    )
    intervals = [
        later - earlier
        for earlier, later in zip(scheduled_offsets, scheduled_offsets[1:])
        if later > earlier
    ]
    tail_interval_s = statistics.median(intervals) if intervals else 0.001
    offered_window_s = (
        scheduled_offsets[-1] - scheduled_offsets[0] + tail_interval_s
        if scheduled_offsets
        else 0.0
    )
    timeout_count = sum(
        metric.error_code in {"ReadTimeout", "WriteTimeout", "PoolTimeout"}
        for metric in metrics
    )
    return {
        "requests": len(metrics),
        "successful_requests": len(successful),
        "expected_output_tokens": sum(
            metric.expected_output_tokens for metric in metrics
        ),
        "output_tokens": sum(metric.output_tokens for metric in successful),
        "errors": len(metrics) - len(successful),
        "error_codes": dict(sorted(errors.items())),
        "timeout_rate": timeout_count / len(metrics) if metrics else 0.0,
        "duration_s": duration_s,
        "offered_rps": len(metrics) / offered_window_s if offered_window_s else 0.0,
        "sent_rps": len(metrics) / duration_s if duration_s else 0.0,
        "completed_rps": len(successful) / duration_s if duration_s else 0.0,
        "input_tokens_per_s": (
            sum(metric.input_len for metric in successful) / duration_s
            if duration_s
            else 0.0
        ),
        "output_tokens_per_s": (
            sum(metric.output_tokens for metric in successful) / duration_s
            if duration_s
            else 0.0
        ),
        "goodput_rps": good_requests / duration_s if duration_s else 0.0,
        "slo_attainment": good_requests / len(metrics) if metrics else 0.0,
        "slo": {
            "ttft_ms": SLO_TTFT_MS,
            "tpot_ms": SLO_TPOT_MS,
            "e2e_ms": SLO_E2E_MS,
        },
        "ttft_ms": distribution("ttft_ms"),
        "tpot_ms": distribution("tpot_ms"),
        "e2e_ms": distribution("e2e_ms"),
        "client_queue_delay_ms": distribution("queue_delay_ms"),
        "long_request_completion_rate": (
            len(long_success) / len(long_requests) if long_requests else None
        ),
        "health": {
            "ready_before": health_before.get("ready"),
            "ready_after": health_after.get("ready"),
            "final_waiting": health_after.get("waiting_count"),
            "final_running": health_after.get("running_count"),
            "cancelled_delta": health_count(health_after, "cancelled_count")
            - health_count(health_before, "cancelled_count"),
            "failed_delta": health_count(health_after, "failed_count")
            - health_count(health_before, "failed_count"),
            "fatal_error": health_after.get("fatal_error"),
        },
        "gpu": gpu,
    }


async def warm_up(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    tokenizer,
    seed: int,
) -> None:
    items = build_workload(tokenizer, "mixed", seed)[:4]
    origin = time.monotonic() + 0.05
    await asyncio.gather(
        *(
            run_one(
                client,
                base_url=base_url,
                model=model,
                item=WorkItem(
                    request_hash=item.request_hash,
                    request_class="warmup",
                    input_len=item.input_len,
                    output_len=8,
                    scheduled_offset_s=0.0,
                    prompt=item.prompt,
                ),
                origin=origin,
            )
            for item in items
        )
    )
    await wait_for_idle(client, base_url, 60.0)
    response = await client.post(f"{base_url}/v1/scheduler/metrics/reset")
    response.raise_for_status()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.workload}-{args.policy}-r{args.repeat}"
    summary_path = output_dir / f"{run_id}.summary.json"
    if args.resume and summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("valid"):
            print(json.dumps({"run_id": run_id, "skipped": True}))
            return existing

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    trace_seed = args.seed + args.repeat
    items = build_workload(tokenizer, args.workload, trace_seed)
    limits = httpx.Limits(
        max_connections=args.max_concurrency,
        max_keepalive_connections=args.max_concurrency,
    )
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        health_before = await get_health(client, args.base_url)
        if not health_before.get("ready"):
            raise RuntimeError("worker_not_ready")
        await warm_up(
            client,
            base_url=args.base_url,
            model=args.model,
            tokenizer=tokenizer,
            seed=trace_seed + 1000,
        )
        health_before = await get_health(client, args.base_url)
        gpu_sampler = GpuSampler(args.gpu_index)
        sampler_task = asyncio.create_task(gpu_sampler.run())
        origin = time.monotonic() + 0.25
        metrics: list[RequestMetric] = []
        run_error: str | None = None
        health_after: dict[str, Any] = dict(health_before)
        try:
            metrics = await asyncio.gather(
                *(
                    run_one(
                        client,
                        base_url=args.base_url,
                        model=args.model,
                        item=item,
                        origin=origin,
                    )
                    for item in items
                )
            )
            health_after = await wait_for_idle(
                client, args.base_url, args.drain_timeout
            )
        except Exception as exc:
            run_error = type(exc).__name__
            try:
                health_after = await get_health(client, args.base_url)
            except Exception:
                health_after = {
                    "ready": False,
                    "waiting_count": None,
                    "running_count": None,
                    "failed_count": None,
                    "cancelled_count": None,
                    "fatal_error": "health_unavailable",
                }
        finally:
            gpu_sampler.stop()
            await sampler_task

    summary = summarize(
        metrics, health_before, health_after, gpu_sampler.summary()
    )
    summary.update(
        {
            "schema_version": 1,
            "run_id": run_id,
            "workload": args.workload,
            "policy": args.policy,
            "repeat": args.repeat,
            "trace_seed": trace_seed,
            "trace_fingerprint": trace_fingerprint(items),
            "scheduler_metrics_path": str(args.scheduler_metrics_path),
            "run_error": run_error,
            "raw_prompts_stored": False,
            "raw_token_ids_stored": False,
        }
    )
    health = summary["health"]
    summary["valid"] = bool(
        run_error is None
        and summary["requests"] > 0
        and summary["successful_requests"] == summary["requests"]
        and summary["errors"] == 0
        and health["ready_after"]
        and health["final_waiting"] == 0
        and health["final_running"] == 0
        and health["failed_delta"] == 0
        and health["fatal_error"] is None
    )

    records_path = output_dir / f"{run_id}.jsonl"
    temporary_records = records_path.with_suffix(".jsonl.tmp")
    temporary_records.write_text(
        "".join(
            json.dumps(asdict(metric), sort_keys=True) + "\n"
            for metric in metrics
        ),
        encoding="utf-8",
    )
    temporary_records.replace(records_path)
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--model", default="/models/Qwen3-8B")
    parser.add_argument("--tokenizer", default="/models/Qwen3-8B")
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--drain-timeout", type=float, default=60.0)
    parser.add_argument("--total-timeout", type=float, default=180.0)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--scheduler-metrics-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = asyncio.run(
        asyncio.wait_for(run(args), timeout=args.total_timeout)
    )
    if not summary.get("valid"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
