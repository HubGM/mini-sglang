from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM


def synthetic_prompts(seed: int) -> list[str]:
    words = (
        "engine",
        "scheduler",
        "request",
        "token",
        "cache",
        "latency",
        "batch",
        "runtime",
    )
    prompts = []
    for index in range(3):
        rng = random.Random(seed + index)
        prompts.append(" ".join(rng.choice(words) for _ in range(48)))
    return prompts


def token_hash(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/Qwen3-8B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    started = time.monotonic()
    llm = LLM(
        args.model,
        memory_ratio=0.75,
        cuda_graph_max_bs=8,
        max_running_req=8,
        max_extend_tokens=4096,
        scheduling_policy="upstream_default",
    )
    try:
        outputs = llm.generate(
            synthetic_prompts(args.seed),
            SamplingParams(max_tokens=16, temperature=0.0),
        )
    finally:
        llm.shutdown()

    result = {
        "seed": args.seed,
        "policy": "upstream_default",
        "temperature": 0.0,
        "max_tokens": 16,
        "prompt_count": len(outputs),
        "token_sequence_hashes": [
            token_hash(output["token_ids"]) for output in outputs
        ],
        "token_counts": [len(output["token_ids"]) for output in outputs],
        "elapsed_s": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "prompt_count": result["prompt_count"],
                "token_counts": result["token_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
