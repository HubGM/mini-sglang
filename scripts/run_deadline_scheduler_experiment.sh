#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="llm-servelab-engine-6br"
IMAGE_ID="${ENGINE_LAB_IMAGE_ID:-sha256:6e0b9f24bcf75cbe3ea72f22340eb9e85838220f3453a9044f419120e5ce79c6}"
GPU_ID="${ENGINE_LAB_GPU_ID:?Set ENGINE_LAB_GPU_ID to one confirmed-idle RTX A6000}"
PORT="${ENGINE_LAB_PORT:-19171}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${ENGINE_LAB_MODEL_ROOT:-/lfs1/users/chyan/models/Qwen3-8B}"
RUNTIME_ROOT="${ENGINE_LAB_RUNTIME_ROOT:-/lfs1/users/chyan/mini-sglang-engine-lab-runtime-6br}"
RESULTS_DIR="${RUNTIME_ROOT}/results/core"
SCHEDULER_DIR="${RUNTIME_ROOT}/results/scheduler"
CORRECTNESS_DIR="${RUNTIME_ROOT}/results/correctness"
LOG_DIR="${RUNTIME_ROOT}/logs"
MANIFEST="${RUNTIME_ROOT}/experiment_manifest.json"
PID_FILE="${RUNTIME_ROOT}/engine_server.pid"
BASE_URL="http://127.0.0.1:${PORT}"
MAX_RUNS="${ENGINE_LAB_MAX_RUNS:-0}"

mkdir -p "$RESULTS_DIR" "$SCHEDULER_DIR" "$CORRECTNESS_DIR" "$LOG_DIR"

if [[ -f /lfs1/users/chyan/miniconda3/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /lfs1/users/chyan/miniconda3/bin/activate miles-dev
fi
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

container_touched=0
server_started=0

verify_labels() {
  local metadata
  metadata="$(docker container inspect "$CONTAINER_NAME" --format \
    '{{index .Config.Labels "com.llm-servelab.project"}}|{{index .Config.Labels "com.llm-servelab.owner"}}|{{index .Config.Labels "com.llm-servelab.stage"}}')"
  [[ "$metadata" == "LLM-ServeLab|chyan|prompt-6br" ]] || {
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
    echo "Graceful server shutdown did not complete; stopping only $CONTAINER_NAME" >&2
    rc=1
  fi
  stop_container || rc=1
  exit "$rc"
}
trap cleanup EXIT INT TERM

check_gpu_idle() {
  local name memory processes
  name="$(nvidia-smi -i "$GPU_ID" --query-gpu=name --format=csv,noheader | xargs)"
  [[ "$name" == *"RTX A6000"* ]] || {
    echo "GPU $GPU_ID is not an RTX A6000: $name" >&2
    return 1
  }
  processes="$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits | xargs)"
  [[ -z "$processes" ]] || {
    echo "GPU $GPU_ID has an existing compute process; refusing use" >&2
    return 1
  }
  memory="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | xargs)"
  echo "Using confirmed-idle GPU $GPU_ID ($name, ${memory} MiB used)"
}

ensure_container() {
  if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    verify_labels
    local actual_image source_mount model_mount runtime_mount status
    actual_image="$(docker container inspect "$CONTAINER_NAME" --format '{{.Image}}')"
    [[ "$actual_image" == "$IMAGE_ID" ]] || {
      echo "Existing container image ID does not match the pinned image" >&2
      return 1
    }
    source_mount="$(docker container inspect "$CONTAINER_NAME" --format '{{range .Mounts}}{{if eq .Destination "/workspace/engine-lab"}}{{.Source}}|{{.RW}}{{end}}{{end}}')"
    model_mount="$(docker container inspect "$CONTAINER_NAME" --format '{{range .Mounts}}{{if eq .Destination "/models/Qwen3-8B"}}{{.Source}}|{{.RW}}{{end}}{{end}}')"
    runtime_mount="$(docker container inspect "$CONTAINER_NAME" --format '{{range .Mounts}}{{if eq .Destination "/runtime"}}{{.Source}}|{{.RW}}{{end}}{{end}}')"
    [[ "$source_mount" == "${REPO_ROOT}|false" ]] || return 1
    [[ "$model_mount" == "${MODEL_ROOT}|false" ]] || return 1
    [[ "$runtime_mount" == "${RUNTIME_ROOT}|true" ]] || return 1
    status="$(docker container inspect "$CONTAINER_NAME" --format '{{.State.Status}}')"
    if [[ "$status" == "exited" ]]; then
      docker start "$CONTAINER_NAME" >/dev/null
    elif [[ "$status" != "running" ]]; then
      echo "Existing project container is in unsupported state: $status" >&2
      return 1
    fi
  else
    docker image inspect "$IMAGE_ID" >/dev/null
    docker run -d \
      --name "$CONTAINER_NAME" \
      --label com.llm-servelab.project=LLM-ServeLab \
      --label com.llm-servelab.owner=chyan \
      --label com.llm-servelab.stage=prompt-6br \
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

start_server() {
  local run_id=$1 policy=$2 scheduler_path=$3 log_path=$4
  check_gpu_idle
  if ss -ltn "sport = :${PORT}" | tail -n +2 | grep -q .; then
    echo "Host port $PORT is already in use" >&2
    return 1
  fi
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
        --max-consecutive-prefill-steps 1 \
        --default-ttft-deadline-ms 200 \
        --default-e2e-deadline-ms 1200 \
        --max-wait-ms 400 \
        --aging-start-ms 100 \
        --aging-rate 1 \
        --starvation-threshold-ms 400 \
        --scheduler-request-sample-rate 1 \
        --scheduler-max-step-records 10000 \
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

run_correctness_gate() {
  local complete=1 policy output
  for policy in upstream_default token_budget deadline_aware deadline_aging; do
    output="${CORRECTNESS_DIR}/${policy}.json"
    [[ -s "$output" ]] || complete=0
  done
  if [[ "$complete" == 1 ]] && python -m benchmark.deadline_matrix correctness \
      "$CORRECTNESS_DIR"/upstream_default.json \
      "$CORRECTNESS_DIR"/token_budget.json \
      "$CORRECTNESS_DIR"/deadline_aware.json \
      "$CORRECTNESS_DIR"/deadline_aging.json \
      --output "$CORRECTNESS_DIR/summary.json"; then
    return 0
  fi

  for policy in upstream_default token_budget deadline_aware deadline_aging; do
    check_gpu_idle
    output="${CORRECTNESS_DIR}/${policy}.json"
    docker exec "$CONTAINER_NAME" bash -lc \
      "cd /workspace/engine-lab && PYTHONDONTWRITEBYTECODE=1 \
       PYTHONPATH=/workspace/engine-lab/python python benchmark/gpu_correctness_gate.py \
       --model /models/Qwen3-8B --policy ${policy} \
       --max-step-tokens 2048 --max-prefill-chunk-tokens 512 \
       --output /runtime/results/correctness/${policy}.json"
  done
  python -m benchmark.deadline_matrix correctness \
    "$CORRECTNESS_DIR"/upstream_default.json \
    "$CORRECTNESS_DIR"/token_budget.json \
    "$CORRECTNESS_DIR"/deadline_aware.json \
    "$CORRECTNESS_DIR"/deadline_aging.json \
    --output "$CORRECTNESS_DIR/summary.json"
}

python -m benchmark.deadline_matrix plan \
  --repo-root "$REPO_ROOT" --output "$MANIFEST"
ensure_container
run_correctness_gate

executed_runs=0
while IFS=$'\t' read -r run_id workload policy repeat; do
  summary_path="${RESULTS_DIR}/${run_id}.summary.json"
  scheduler_path="${SCHEDULER_DIR}/${run_id}.json"
  log_path="${LOG_DIR}/${run_id}.log"
  if python -m benchmark.deadline_matrix completed --summary "$summary_path"; then
    echo "Skipping completed VALID run $run_id"
    continue
  fi

  attempt="$(date +%s)"
  [[ ! -e "$summary_path" ]] || mv "$summary_path" "${summary_path}.invalid.${attempt}"
  [[ ! -e "$scheduler_path" ]] || mv "$scheduler_path" "${scheduler_path}.invalid.${attempt}"
  echo "Running $run_id"
  start_server "$run_id" "$policy" "$scheduler_path" "$log_path"
  python -m benchmark.deadline_experiment \
    --base-url "$BASE_URL" \
    --model /models/Qwen3-8B \
    --tokenizer "$MODEL_ROOT" \
    --workload "$workload" \
    --policy "$policy" \
    --repeat "$repeat" \
    --seed 20260909 \
    --gpu-index "$GPU_ID" \
    --scheduler-metrics-path "$scheduler_path" \
    --output-dir "$RESULTS_DIR" \
    --resume
  stop_server
  python -m benchmark.deadline_matrix validate-run \
    --summary "$summary_path" \
    --scheduler "$scheduler_path" \
    --policy "$policy"
  executed_runs=$((executed_runs + 1))
  if [[ "$MAX_RUNS" -gt 0 && "$executed_runs" -ge "$MAX_RUNS" ]]; then
    echo "Stopped after requested pilot run count: $MAX_RUNS"
    exit 0
  fi
  sleep 3
done < <(
  python -m benchmark.deadline_matrix plan \
    --repo-root "$REPO_ROOT" --output "$MANIFEST" --emit-lines
)

python -m benchmark.deadline_matrix summarize \
  --results-dir "$RESULTS_DIR" \
  --output "$RUNTIME_ROOT/results/deadline_scheduler_summary.json"
