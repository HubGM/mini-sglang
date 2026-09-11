#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="llm-servelab-engine-6br2"
IMAGE_ID="${ENGINE_LAB_IMAGE_ID:-sha256:6e0b9f24bcf75cbe3ea72f22340eb9e85838220f3453a9044f419120e5ce79c6}"
GPU_ID="${ENGINE_LAB_GPU_ID:?Set ENGINE_LAB_GPU_ID to one confirmed-idle RTX A6000}"
PORT="${ENGINE_LAB_PORT:-19172}"
PHASE="${ENGINE_LAB_PHASE:-all}"
MAX_RUNS="${ENGINE_LAB_MAX_RUNS:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${ENGINE_LAB_MODEL_ROOT:-/lfs1/users/chyan/models/Qwen3-8B}"
RUNTIME_ROOT="${ENGINE_LAB_RUNTIME_ROOT:-/lfs1/users/chyan/mini-sglang-engine-lab-runtime-6br2}"
DEV_RESULTS="${RUNTIME_ROOT}/results/dev"
HOLDOUT_RESULTS="${RUNTIME_ROOT}/results/holdout"
SCHEDULER_DIR="${RUNTIME_ROOT}/results/scheduler"
CORRECTNESS_DIR="${RUNTIME_ROOT}/results/correctness"
LOG_DIR="${RUNTIME_ROOT}/logs"
DEV_MANIFEST="${RUNTIME_ROOT}/dev_manifest.json"
HOLDOUT_MANIFEST="${RUNTIME_ROOT}/holdout_manifest.json"
FINAL_CONFIG="${RUNTIME_ROOT}/FINAL_CONFIG.json"
BASE_URL="http://127.0.0.1:${PORT}"

[[ "$PHASE" =~ ^(all|dev|holdout)$ ]] || {
  echo "ENGINE_LAB_PHASE must be all, dev, or holdout" >&2
  exit 2
}
mkdir -p \
  "$DEV_RESULTS" "$HOLDOUT_RESULTS" "$SCHEDULER_DIR" \
  "$CORRECTNESS_DIR" "$LOG_DIR"

# shellcheck disable=SC1091
source /lfs1/users/chyan/miniconda3/bin/activate miles-dev
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

container_touched=0
server_started=0
executed_runs=0

verify_labels() {
  local metadata
  metadata="$(docker container inspect "$CONTAINER_NAME" --format \
    '{{index .Config.Labels "com.llm-servelab.project"}}|{{index .Config.Labels "com.llm-servelab.owner"}}|{{index .Config.Labels "com.llm-servelab.stage"}}')"
  [[ "$metadata" == "LLM-ServeLab|chyan|prompt-6br2" ]] || {
    echo "Refusing container with unconfirmed project labels: $metadata" >&2
    return 1
  }
}

stop_server() {
  [[ "$server_started" == 1 ]] || return 0
  verify_labels
  docker exec "$CONTAINER_NAME" bash -lc '
    pid_file=/runtime/engine_server.pid
    [[ -s "$pid_file" ]] || exit 0
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] || exit 1
    [[ -r "/proc/$pid/cmdline" ]] || exit 0
    command="$(tr "\0" " " < "/proc/$pid/cmdline")"
    [[ "$command" == *"python -m minisgl"* ]] || exit 1
    kill -INT -- "-$pid"
    for _ in $(seq 1 180); do
      [[ ! -d "/proc/$pid" ]] && exit 0
      state="$(sed -n "s/^State:[[:space:]]*\\([^[:space:]]*\\).*/\\1/p" "/proc/$pid/status" 2>/dev/null || true)"
      [[ "$state" == "Z" ]] && exit 0
      sleep 0.5
    done
    exit 2
  '
  server_started=0
}

stop_container() {
  [[ "$container_touched" == 1 ]] || return 0
  verify_labels
  local status
  status="$(docker container inspect "$CONTAINER_NAME" --format '{{.State.Status}}')"
  if [[ "$status" == "running" ]]; then
    docker stop -t 60 "$CONTAINER_NAME" >/dev/null
  fi
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if ! stop_server; then
    echo "Graceful server shutdown failed; stopping only $CONTAINER_NAME" >&2
    rc=1
  fi
  stop_container || rc=1
  exit "$rc"
}
trap cleanup EXIT INT TERM

check_gpu_idle() {
  local name processes memory
  name="$(nvidia-smi -i "$GPU_ID" --query-gpu=name --format=csv,noheader | xargs)"
  [[ "$name" == *"RTX A6000"* ]] || {
    echo "GPU $GPU_ID is not an RTX A6000: $name" >&2
    return 1
  }
  processes="$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits | xargs)"
  [[ -z "$processes" ]] || {
    echo "GPU $GPU_ID has a compute process; refusing shared-server use" >&2
    return 1
  }
  memory="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | xargs)"
  echo "Confirmed idle GPU $GPU_ID ($name, ${memory} MiB used)"
}

wait_gpu_released() {
  local processes memory
  for _ in $(seq 1 120); do
    processes="$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits | xargs)"
    memory="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | xargs)"
    if [[ -z "$processes" && "$memory" -eq 0 ]]; then
      return 0
    fi
    sleep 1
  done
  echo "GPU $GPU_ID did not return to 0 MiB" >&2
  return 1
}

ensure_container() {
  check_gpu_idle
  if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    verify_labels
    local actual_image source_mount model_mount runtime_mount status
    actual_image="$(docker container inspect "$CONTAINER_NAME" --format '{{.Image}}')"
    source_mount="$(docker container inspect "$CONTAINER_NAME" --format '{{range .Mounts}}{{if eq .Destination "/workspace/engine-lab"}}{{.Source}}|{{.RW}}{{end}}{{end}}')"
    model_mount="$(docker container inspect "$CONTAINER_NAME" --format '{{range .Mounts}}{{if eq .Destination "/models/Qwen3-8B"}}{{.Source}}|{{.RW}}{{end}}{{end}}')"
    runtime_mount="$(docker container inspect "$CONTAINER_NAME" --format '{{range .Mounts}}{{if eq .Destination "/runtime"}}{{.Source}}|{{.RW}}{{end}}{{end}}')"
    [[ "$actual_image" == "$IMAGE_ID" ]] || return 1
    [[ "$source_mount" == "${REPO_ROOT}|false" ]] || return 1
    [[ "$model_mount" == "${MODEL_ROOT}|false" ]] || return 1
    [[ "$runtime_mount" == "${RUNTIME_ROOT}|true" ]] || return 1
    status="$(docker container inspect "$CONTAINER_NAME" --format '{{.State.Status}}')"
    if [[ "$status" == "exited" ]]; then
      docker start "$CONTAINER_NAME" >/dev/null
    elif [[ "$status" != "running" ]]; then
      echo "Unsupported project-container state: $status" >&2
      return 1
    fi
  else
    docker image inspect "$IMAGE_ID" >/dev/null
    docker run -d \
      --name "$CONTAINER_NAME" \
      --label com.llm-servelab.project=LLM-ServeLab \
      --label com.llm-servelab.owner=chyan \
      --label com.llm-servelab.stage=prompt-6br2 \
      --gpus "device=${GPU_ID}" \
      --network host \
      --shm-size 32g \
      --log-opt max-size=10m \
      --log-opt max-file=3 \
      --mount "type=bind,src=${REPO_ROOT},dst=/workspace/engine-lab,readonly" \
      --mount "type=bind,src=${MODEL_ROOT},dst=/models/Qwen3-8B,readonly" \
      --mount "type=bind,src=${RUNTIME_ROOT},dst=/runtime" \
      "$IMAGE_ID" sleep infinity >/dev/null
  fi
  container_touched=1
  verify_labels
}

wait_ready() {
  for _ in $(seq 1 360); do
    if curl --silent --fail "${BASE_URL}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "Mini-SGLang readiness timed out" >&2
  return 1
}

config_args() {
  local source=$1
  if [[ "$source" == final ]]; then
    python -m benchmark.deadline_v2_matrix config-args \
      --final-config "$FINAL_CONFIG"
  elif [[ "$source" == v1 ]]; then
    printf '%s\n' \
      --max-consecutive-prefill-steps=1 \
      --min-prefill-budget-per-step=256 \
      --max-decode-only-steps=2 \
      --prefill-urgent-threshold-ms=150 \
      --hard-max-wait-ms=300 \
      --min-decode-reserve-ratio=0.25 \
      --max-decode-reserve-ratio=0.65
  else
    python -m benchmark.deadline_v2_matrix config-args --candidate "$source"
  fi
}

start_server() {
  local run_id=$1 policy=$2 config_source=$3 scheduler_path=$4 log_path=$5
  local joined_args
  check_gpu_idle
  if ss -ltn "sport = :${PORT}" | tail -n +2 | grep -q .; then
    echo "Host port $PORT is already in use" >&2
    return 1
  fi
  joined_args="$(config_args "$config_source" | tr '\n' ' ')"
  docker exec -d "$CONTAINER_NAME" bash -lc "
    : > /runtime/engine_server.pid
    nohup setsid env PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH=/workspace/engine-lab/python \
      python -m minisgl \
        --model-path /models/Qwen3-8B \
        --host 127.0.0.1 \
        --port ${PORT} \
        --memory-ratio 0.75 \
        --max-running-requests 64 \
        --max-extend-length 4096 \
        --cuda-graph-max-bs 32 \
        --attention-backend fi \
        --cache-type radix \
        --scheduling-policy ${policy} \
        --max-step-tokens 2048 \
        --max-prefill-chunk-tokens 512 \
        --decode-reserve-ratio 0.5 \
        --default-ttft-deadline-ms 200 \
        --default-tpot-deadline-ms 50 \
        --default-e2e-deadline-ms 1200 \
        --max-wait-ms 400 \
        --aging-start-ms 100 \
        --aging-rate 1 \
        --starvation-threshold-ms 400 \
        --scheduler-request-sample-rate 1 \
        --scheduler-max-step-records 10000 \
        ${joined_args} \
        --scheduler-metrics-path /runtime/results/scheduler/${run_id}.json \
        > /runtime/logs/${run_id}.log 2>&1 < /dev/null &
    printf '%s\n' \"\$!\" > /runtime/engine_server.pid
  "
  server_started=1
  wait_ready || {
    tail -n 80 "$log_path" >&2 || true
    return 1
  }
  [[ ! -e "$scheduler_path" ]] || {
    echo "Scheduler metrics already exist for incomplete run $run_id" >&2
    return 1
  }
}

archive_incomplete() {
  local path timestamp
  timestamp="$(date +%s)"
  for path in "$@"; do
    [[ ! -e "$path" ]] || mv "$path" "${path}.invalid.${timestamp}"
  done
}

run_one() {
  local root=$1 run_id=$2 workload=$3 policy=$4 repeat=$5 candidate=$6 hash=$7
  local summary_path scheduler_path log_path config_source
  summary_path="${root}/${run_id}.summary.json"
  scheduler_path="${SCHEDULER_DIR}/${run_id}.json"
  log_path="${LOG_DIR}/${run_id}.log"
  if python -m benchmark.deadline_v2_matrix completed --summary "$summary_path"; then
    echo "Skipping completed VALID run $run_id"
    return 0
  fi
  archive_incomplete "$summary_path" "$scheduler_path" "${root}/${run_id}.jsonl" "$log_path"
  if [[ "$policy" == "deadline_aging_v2" ]]; then
    config_source="${candidate:-final}"
  else
    config_source=v1
  fi
  echo "Running $run_id"
  start_server "$run_id" "$policy" "$config_source" "$scheduler_path" "$log_path"
  python -m benchmark.deadline_experiment \
    --base-url "$BASE_URL" \
    --model /models/Qwen3-8B \
    --tokenizer "$MODEL_ROOT" \
    --workload "$workload" \
    --policy "$policy" \
    --repeat "$repeat" \
    --run-id "$run_id" \
    --seed "$([[ "$root" == "$DEV_RESULTS" ]] && echo 20261000 || echo 20261100)" \
    --gpu-index "$GPU_ID" \
    --scheduler-metrics-path "$scheduler_path" \
    --output-dir "$root" \
    --resume
  stop_server
  wait_gpu_released
  validate_args=(
    --summary "$summary_path"
    --scheduler "$scheduler_path"
    --policy "$policy"
  )
  [[ -z "$candidate" ]] || validate_args+=(--candidate "$candidate")
  [[ -z "$hash" ]] || validate_args+=(--config-hash "$hash")
  python -m benchmark.deadline_v2_matrix validate-run "${validate_args[@]}"
  executed_runs=$((executed_runs + 1))
  if [[ "$MAX_RUNS" -gt 0 && "$executed_runs" -ge "$MAX_RUNS" ]]; then
    echo "Stopped at ENGINE_LAB_MAX_RUNS=$MAX_RUNS"
    exit 0
  fi
  sleep 3
}

run_dev() {
  python -m benchmark.deadline_v2_matrix plan-dev \
    --repo-root "$REPO_ROOT" --output "$DEV_MANIFEST"
  while IFS=$'\t' read -r run_id workload policy repeat candidate hash; do
    [[ "$candidate" != "-" ]] || candidate=""
    [[ "$hash" != "-" ]] || hash=""
    run_one "$DEV_RESULTS" "$run_id" "$workload" "$policy" \
      "$repeat" "$candidate" "$hash"
  done < <(
    python -m benchmark.deadline_v2_matrix plan-dev \
      --repo-root "$REPO_ROOT" --output "$DEV_MANIFEST" --emit-lines
  )
  python -m benchmark.deadline_v2_matrix select-dev \
    --results-dir "$DEV_RESULTS" --output "$FINAL_CONFIG"
}

run_correctness() {
  local policy output config_source
  for policy in "${POLICIES[@]}"; do
    output="${CORRECTNESS_DIR}/${policy}.json"
    [[ ! -s "$output" ]] || continue
    check_gpu_idle
    config_source=v1
    [[ "$policy" != "deadline_aging_v2" ]] || config_source=final
    mapfile -t extra_args < <(config_args "$config_source")
    docker exec "$CONTAINER_NAME" bash -lc \
      "cd /workspace/engine-lab && PYTHONDONTWRITEBYTECODE=1 \
       PYTHONPATH=/workspace/engine-lab/python python benchmark/gpu_correctness_gate.py \
       --model /models/Qwen3-8B --policy ${policy} \
       --max-step-tokens 2048 --max-prefill-chunk-tokens 512 \
       ${extra_args[*]} \
       --output /runtime/results/correctness/${policy}.json"
    wait_gpu_released
  done
  python -m benchmark.deadline_v2_matrix correctness \
    "$CORRECTNESS_DIR/upstream_default.json" \
    "$CORRECTNESS_DIR/deadline_aging_v1.json" \
    "$CORRECTNESS_DIR/deadline_aging_v2.json" \
    --output "$CORRECTNESS_DIR/summary.json"
}

run_cancellation_smoke() {
  local run_id="deadline-aging-v2-cancellation-smoke"
  local output="${RUNTIME_ROOT}/results/cancellation_smoke.json"
  local scheduler_path="${SCHEDULER_DIR}/${run_id}.json"
  local log_path="${LOG_DIR}/${run_id}.log"
  if [[ -s "$output" ]] && python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(not all(d.get("gates", {}).values()))' \
    "$output"; then
    return 0
  fi
  archive_incomplete "$output" "$scheduler_path" "$log_path"
  start_server "$run_id" deadline_aging_v2 final "$scheduler_path" "$log_path"
  python -m benchmark.lifecycle_stress \
    --base-url "$BASE_URL" \
    --model /models/Qwen3-8B \
    --policy deadline_aging_v2 \
    --seed 20261201 \
    --smoke-requests 50 \
    --smoke-cancellation-ratio 30 \
    --concurrency 12 \
    --output "$output"
  stop_server
  wait_gpu_released
}

run_holdout() {
  python -m benchmark.deadline_v2_matrix plan-holdout \
    --repo-root "$REPO_ROOT" --final-config "$FINAL_CONFIG" \
    --output "$HOLDOUT_MANIFEST"
  local final_hash
  final_hash="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["FINAL_CONFIG_HASH"])' "$FINAL_CONFIG")"
  while IFS=$'\t' read -r run_id workload policy repeat candidate hash; do
    [[ "$candidate" != "-" ]] || candidate=""
    [[ "$hash" != "-" ]] || hash=""
    [[ "$policy" != "deadline_aging_v2" ]] || hash="$final_hash"
    run_one "$HOLDOUT_RESULTS" "$run_id" "$workload" "$policy" \
      "$repeat" "" "$hash"
  done < <(
    python -m benchmark.deadline_v2_matrix plan-holdout \
      --repo-root "$REPO_ROOT" --final-config "$FINAL_CONFIG" \
      --output "$HOLDOUT_MANIFEST" --emit-lines
  )
  python -m benchmark.deadline_v2_matrix summarize \
    --results-dir "$HOLDOUT_RESULTS" \
    --final-config "$FINAL_CONFIG" \
    --output "$RUNTIME_ROOT/results/deadline_scheduler_v2_holdout.json" \
    --compact-output "$REPO_ROOT/benchmark/examples/deadline_scheduler_v2_holdout_summary.json"
}

POLICIES=(upstream_default deadline_aging_v1 deadline_aging_v2)
ensure_container
if [[ "$PHASE" == all || "$PHASE" == dev ]]; then
  run_dev
fi
if [[ "$PHASE" == all || "$PHASE" == holdout ]]; then
  [[ -s "$FINAL_CONFIG" ]] || {
    echo "Frozen FINAL_CONFIG is required before HOLDOUT" >&2
    exit 1
  }
  run_correctness
  run_cancellation_smoke
  run_holdout
fi
