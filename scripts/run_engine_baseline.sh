#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${ENGINE_LAB_RUNTIME_ROOT:-/lfs1/users/chyan/mini-sglang-engine-lab-runtime}"
base_url="${ENGINE_LAB_BASE_URL:-http://127.0.0.1:19161/v1}"
tokenizer="${ENGINE_LAB_TOKENIZER:-/lfs1/users/chyan/models/Qwen3-8B}"
num_requests="${ENGINE_LAB_NUM_REQUESTS:-12}"
concurrency="${ENGINE_LAB_CONCURRENCY:-4}"
arrival_rate="${ENGINE_LAB_ARRIVAL_RATE:-4}"

for repeat in 1 2 3; do
  for workload in short-short long-short short-long mixed starvation; do
    for traffic in closed open; do
      python "${repo_root}/benchmark/engine_lab_baseline.py" \
        --base-url "${base_url}" \
        --tokenizer "${tokenizer}" \
        --workload "${workload}" \
        --traffic "${traffic}" \
        --num-requests "${num_requests}" \
        --concurrency "${concurrency}" \
        --arrival-rate "${arrival_rate}" \
        --repeat "${repeat}" \
        --output-dir "${runtime_root}/results" \
        --resume
    done
  done
done
