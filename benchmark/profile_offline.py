from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM


CASES = {
    "concurrency-1": ([96], [32]),
    "moderate-8": ([128] * 8, [64] * 8),
    "saturation-32": ([128] * 32, [64] * 32),
    "mixed-prefill-decode": (
        [128, 128, 1024, 1536, 128, 128],
        [128, 128, 32, 32, 128, 128],
    ),
}


def make_prompts(lengths: list[int], seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randint(1, 10_000) for _ in range(length)] for length in lengths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--model", default="/models/Qwen3-8B")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/runtime/profiles"))
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--torch-profile", action="store_true")
    args = parser.parse_args()

    input_lengths, output_lengths = CASES[args.case]
    prompts = make_prompts(input_lengths, args.seed)
    sampling_params = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=length)
        for length in output_lengths
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.case}.scheduler.json"
    started = time.perf_counter()
    llm = LLM(
        args.model,
        dtype=torch.bfloat16,
        max_running_req=64,
        cuda_graph_max_bs=32,
        max_extend_tokens=4096,
        memory_ratio=0.75,
        attention_backend="fi",
        scheduling_policy="upstream_default",
        scheduler_metrics_path=str(metrics_path),
    )
    initialized = time.perf_counter()
    if args.torch_profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            generated = llm.generate(prompts, sampling_params)
            torch.cuda.synchronize()
        profiler.export_chrome_trace(
            str(args.output_dir / f"{args.case}.torch-trace.json")
        )
        (args.output_dir / f"{args.case}.torch-summary.txt").write_text(
            profiler.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=40,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        torch.cuda.nvtx.range_push(f"engine-profile-{args.case}")
        generated = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    finished = time.perf_counter()
    llm.shutdown()

    payload = {
        "case": args.case,
        "seed": args.seed,
        "request_count": len(prompts),
        "input_lengths": input_lengths,
        "output_lengths": output_lengths,
        "observed_output_tokens": [len(item["token_ids"]) for item in generated],
        "initialization_seconds": initialized - started,
        "generation_seconds": finished - initialized,
        "scheduler_metrics_path": str(metrics_path),
    }
    (args.output_dir / f"{args.case}.run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
