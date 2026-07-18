from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

from minisgl.benchmark.client import generate_prompt
from openai import AsyncOpenAI
from transformers import AutoTokenizer


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    input_len: int
    output_len: int
    prompt: str
    scheduled_offset_s: float


@dataclass(frozen=True)
class RequestMetric:
    request_id: str
    input_len: int
    expected_output_len: int
    observed_chunks: int
    scheduled_s: float
    sent_s: float
    first_token_s: float | None
    end_s: float
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_ms: float
    queue_delay_ms: float
    error: str | None


WORKLOADS = {
    "short-short": ((64, 128), (16, 32)),
    "long-short": ((1024, 2048), (16, 32)),
    "short-long": ((64, 128), (128, 256)),
    "mixed": ((64, 2048), (16, 256)),
}


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def build_workload(
    tokenizer,
    name: str,
    num_requests: int,
    seed: int,
    arrival_rate: float | None,
) -> List[WorkItem]:
    rng = random.Random(seed)
    if name == "starvation":
        lengths = [
            (1536 if i in {num_requests // 3, 2 * num_requests // 3} else 96, 32)
            for i in range(num_requests)
        ]
    else:
        input_range, output_range = WORKLOADS[name]
        lengths = [
            (rng.randint(*input_range), rng.randint(*output_range))
            for _ in range(num_requests)
        ]

    offset = 0.0
    items = []
    random_state = random.getstate()
    random.seed(seed)
    try:
        for index, (input_len, output_len) in enumerate(lengths):
            if arrival_rate is not None and index:
                offset += rng.expovariate(arrival_rate)
            items.append(
                WorkItem(
                    request_id=f"{name}-{seed}-{index:05d}",
                    input_len=input_len,
                    output_len=output_len,
                    prompt=generate_prompt(tokenizer, input_len),
                    scheduled_offset_s=offset,
                )
            )
    finally:
        random.setstate(random_state)
    return items


async def run_one(client, model: str, item: WorkItem, origin: float) -> RequestMetric:
    scheduled = origin + item.scheduled_offset_s
    await asyncio.sleep(max(0.0, scheduled - time.perf_counter()))
    sent = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=model,
            stream=True,
            messages=[{"role": "user", "content": item.prompt}],
            max_tokens=item.output_len,
            temperature=0.0,
            extra_body={"ignore_eos": True, "top_k": 1},
        )
        token_times = []
        async for chunk in response:
            if chunk.choices and chunk.choices[0].finish_reason is None:
                token_times.append(time.perf_counter())
        end = time.perf_counter()
        first = token_times[0] if token_times else None
        intervals = [
            token_times[index] - token_times[index - 1]
            for index in range(1, len(token_times))
        ]
        return RequestMetric(
            request_id=item.request_id,
            input_len=item.input_len,
            expected_output_len=item.output_len,
            observed_chunks=len(token_times),
            scheduled_s=scheduled,
            sent_s=sent,
            first_token_s=first,
            end_s=end,
            ttft_ms=(first - sent) * 1_000 if first is not None else None,
            tpot_ms=statistics.mean(intervals) * 1_000 if intervals else None,
            e2e_ms=(end - sent) * 1_000,
            queue_delay_ms=max(0.0, (sent - scheduled) * 1_000),
            error=None,
        )
    except Exception as exc:
        end = time.perf_counter()
        return RequestMetric(
            request_id=item.request_id,
            input_len=item.input_len,
            expected_output_len=item.output_len,
            observed_chunks=0,
            scheduled_s=scheduled,
            sent_s=sent,
            first_token_s=None,
            end_s=end,
            ttft_ms=None,
            tpot_ms=None,
            e2e_ms=(end - sent) * 1_000,
            queue_delay_ms=max(0.0, (sent - scheduled) * 1_000),
            error=f"{type(exc).__name__}: {exc}",
        )


def summarize(metrics: List[RequestMetric], duration_s: float) -> dict:
    valid = [metric for metric in metrics if metric.error is None]

    def distribution(name: str) -> dict:
        values = [
            float(value)
            for metric in valid
            if (value := getattr(metric, name)) is not None
        ]
        return {
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
        }

    output_tokens = sum(metric.observed_chunks for metric in valid)
    input_tokens = sum(metric.input_len for metric in valid)
    return {
        "requests": len(metrics),
        "successful_requests": len(valid),
        "errors": len(metrics) - len(valid),
        "duration_s": duration_s,
        "completed_rps": len(valid) / duration_s if duration_s else 0.0,
        "input_tokens_per_s": input_tokens / duration_s if duration_s else 0.0,
        "output_tokens_per_s": output_tokens / duration_s if duration_s else 0.0,
        "ttft_ms": distribution("ttft_ms"),
        "tpot_ms": distribution("tpot_ms"),
        "e2e_ms": distribution("e2e_ms"),
        "queue_delay_ms": distribution("queue_delay_ms"),
    }


async def main_async(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.workload}-{args.traffic}-r{args.repeat}"
    summary_path = output_dir / f"{run_id}.summary.json"
    if args.resume and summary_path.exists():
        print(f"skip completed {run_id}")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    arrival_rate = args.arrival_rate if args.traffic == "open" else None
    items = build_workload(
        tokenizer,
        args.workload,
        args.num_requests,
        args.seed + args.repeat,
        arrival_rate,
    )

    limits = __import__("httpx").Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    timeout = __import__("httpx").Timeout(args.timeout)
    async with AsyncOpenAI(
        base_url=args.base_url,
        api_key="not-used",
        timeout=timeout,
        max_retries=0,
        http_client=__import__("httpx").AsyncClient(limits=limits, timeout=timeout),
    ) as client:
        origin = time.perf_counter() + 0.25
        semaphore = asyncio.Semaphore(args.concurrency)

        async def bounded(item):
            async with semaphore:
                return await run_one(client, args.model, item, origin)

        metrics = await asyncio.gather(*(bounded(item) for item in items))
        duration_s = max(metric.end_s for metric in metrics) - origin

    records_path = output_dir / f"{run_id}.jsonl"
    temporary_records = records_path.with_suffix(".jsonl.tmp")
    temporary_records.write_text(
        "".join(json.dumps(asdict(metric), sort_keys=True) + "\n" for metric in metrics),
        encoding="utf-8",
    )
    temporary_records.replace(records_path)

    summary = summarize(metrics, duration_s)
    summary.update(
        {
            "run_id": run_id,
            "workload": args.workload,
            "traffic": args.traffic,
            "repeat": args.repeat,
            "seed": args.seed + args.repeat,
            "policy": args.policy,
            "model": Path(args.model).name,
        }
    )
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_summary.replace(summary_path)
    print(json.dumps(summary, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:19161/v1")
    parser.add_argument("--model", default="/models/Qwen3-8B")
    parser.add_argument("--tokenizer", default="/lfs1/users/chyan/models/Qwen3-8B")
    parser.add_argument(
        "--workload",
        choices=[*WORKLOADS, "starvation"],
        default="short-short",
    )
    parser.add_argument("--traffic", choices=["closed", "open"], default="closed")
    parser.add_argument("--num-requests", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--arrival-rate", type=float, default=4.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--output-dir",
        default="/lfs1/users/chyan/mini-sglang-engine-lab-runtime/results",
    )
    parser.add_argument("--policy", default="upstream_default")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
