from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig
from minisgl.utils import cached_load_hf_config, init_logger


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False
    scheduler_heartbeat_timeout_s: float = 5.0
    cancel_on_disconnect: bool = True

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/minisgl_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/minisgl_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--num-pages",
        "--num-tokens",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=str,
        default=ServerArgs.attention_backend,
        choices=["fa3", "fi"],
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=["naive", "radix"],
        help="The KV cache management strategy.",
    )

    parser.add_argument(
        "--scheduling-policy",
        type=str,
        default=ServerArgs.scheduling_policy,
        choices=[
            "upstream_default",
            "token_budget",
            "deadline_aware",
            "deadline_aging",
            "deadline_aging_v1",
            "deadline_aging_v2",
        ],
        help="The engine-level token scheduling policy.",
    )

    parser.add_argument(
        "--max-step-tokens",
        type=int,
        default=ServerArgs.max_step_tokens,
        help="Maximum prefill or decode tokens selected in one engine step.",
    )
    parser.add_argument(
        "--max-prefill-chunk-tokens",
        type=int,
        default=ServerArgs.max_prefill_chunk_tokens,
        help="Maximum input tokens from one request in a prefill step.",
    )
    parser.add_argument(
        "--decode-reserve-ratio",
        type=float,
        default=ServerArgs.decode_reserve_ratio,
        help="Target temporal share of decode steps while both phases are runnable.",
    )
    parser.add_argument(
        "--max-consecutive-prefill-steps",
        type=int,
        default=ServerArgs.max_consecutive_prefill_steps,
        help="Maximum prefill steps while decode requests are runnable.",
    )
    parser.add_argument(
        "--default-ttft-deadline-ms",
        type=float,
        default=ServerArgs.default_ttft_deadline_ms,
        help="Default relative TTFT deadline used by deadline policies.",
    )
    parser.add_argument(
        "--default-e2e-deadline-ms",
        type=float,
        default=ServerArgs.default_e2e_deadline_ms,
        help="Default relative E2E deadline used by deadline policies.",
    )
    parser.add_argument(
        "--default-tpot-deadline-ms",
        type=float,
        default=ServerArgs.default_tpot_deadline_ms,
        help="Default decode-cadence target used by dual-SLO scheduling.",
    )
    parser.add_argument(
        "--initial-prefill-ms-per-token",
        type=float,
        default=ServerArgs.initial_prefill_ms_per_token,
        help="Conservative initial prefill service estimate.",
    )
    parser.add_argument(
        "--initial-decode-step-ms",
        type=float,
        default=ServerArgs.initial_decode_step_ms,
        help="Conservative initial decode-step service estimate.",
    )
    parser.add_argument(
        "--service-ewma-alpha",
        type=float,
        default=ServerArgs.service_ewma_alpha,
        help="EWMA weight for observed prefill/decode service time.",
    )
    parser.add_argument(
        "--max-wait-ms",
        type=float,
        default=ServerArgs.max_wait_ms,
        help="Waiting age that promotes a request to the urgent set.",
    )
    parser.add_argument(
        "--aging-start-ms",
        type=float,
        default=ServerArgs.aging_start_ms,
        help="Waiting age at which deadline priority begins improving.",
    )
    parser.add_argument(
        "--aging-rate",
        type=float,
        default=ServerArgs.aging_rate,
        help="Priority slack reduction per millisecond after aging starts.",
    )
    parser.add_argument(
        "--min-prefill-budget-per-step",
        type=int,
        default=ServerArgs.min_prefill_budget_per_step,
        help="Minimum budget offered when v2 guarantees a prefill step.",
    )
    parser.add_argument(
        "--max-decode-only-steps",
        type=int,
        default=ServerArgs.max_decode_only_steps,
        help="Maximum contended decode-only steps before prefill service.",
    )
    parser.add_argument(
        "--prefill-urgent-threshold-ms",
        type=float,
        default=ServerArgs.prefill_urgent_threshold_ms,
        help="Waiting age that makes an unserved prefill request urgent.",
    )
    parser.add_argument(
        "--hard-max-wait-ms",
        type=float,
        default=ServerArgs.hard_max_wait_ms,
        help="Waiting age that promotes an unserved request to HARD_URGENT.",
    )
    parser.add_argument(
        "--min-decode-reserve-ratio",
        type=float,
        default=ServerArgs.min_decode_reserve_ratio,
        help="Lower bound for v2 dynamic decode reservation.",
    )
    parser.add_argument(
        "--max-decode-reserve-ratio",
        type=float,
        default=ServerArgs.max_decode_reserve_ratio,
        help="Upper bound for v2 dynamic decode reservation.",
    )
    parser.add_argument(
        "--starvation-threshold-ms",
        type=float,
        default=ServerArgs.starvation_threshold_ms,
        help="Fixed waiting threshold used for starvation telemetry.",
    )
    parser.add_argument(
        "--scheduler-request-sample-rate",
        type=float,
        default=ServerArgs.scheduler_request_sample_rate,
        help="Fraction of terminal request lifecycle records retained.",
    )
    parser.add_argument(
        "--scheduler-max-step-records",
        type=int,
        default=ServerArgs.scheduler_max_step_records,
        help="Bounded number of low-cardinality scheduler step records.",
    )

    parser.add_argument(
        "--scheduler-metrics-path",
        type=str,
        default=ServerArgs.scheduler_metrics_path,
        help="Optional path for an atomic scheduler metrics summary.",
    )

    parser.add_argument(
        "--disable-scheduler-decision-timing",
        action="store_false",
        dest="scheduler_decision_timing",
        default=ServerArgs.scheduler_decision_timing,
        help="Disable per-step policy timing instrumentation.",
    )

    parser.add_argument(
        "--policy-failure-threshold",
        type=int,
        default=ServerArgs.policy_failure_threshold,
        help="Consecutive experimental policy failures before opening its circuit.",
    )

    parser.add_argument(
        "--scheduler-heartbeat-interval",
        type=float,
        dest="scheduler_heartbeat_interval_s",
        default=ServerArgs.scheduler_heartbeat_interval_s,
        help="Scheduler event-loop heartbeat interval in seconds.",
    )

    parser.add_argument(
        "--scheduler-heartbeat-timeout",
        type=float,
        dest="scheduler_heartbeat_timeout_s",
        default=ServerArgs.scheduler_heartbeat_timeout_s,
        help="Readiness timeout for scheduler heartbeats in seconds.",
    )

    parser.add_argument(
        "--disable-cancel-on-disconnect",
        action="store_false",
        dest="cancel_on_disconnect",
        default=ServerArgs.cancel_on_disconnect,
        help="Allow a disconnected streaming request to continue.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if (dtype_str := kwargs["dtype"]) != "auto":
        kwargs["dtype"] = DTYPE_MAP[dtype_str]
    else:
        dtype_or_str = cached_load_hf_config(kwargs["model_path"]).dtype
        if isinstance(dtype_or_str, str):
            kwargs["dtype"] = DTYPE_MAP[dtype_or_str]
        else:
            kwargs["dtype"] = dtype_or_str

    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    if result.max_step_tokens <= 0 or result.max_prefill_chunk_tokens <= 0:
        parser.error("scheduler token budgets must be positive")
    if not 0.0 <= result.decode_reserve_ratio <= 1.0:
        parser.error("--decode-reserve-ratio must be between 0 and 1")
    if result.max_consecutive_prefill_steps < 0:
        parser.error("--max-consecutive-prefill-steps must be non-negative")
    if not 0.0 < result.service_ewma_alpha <= 1.0:
        parser.error("--service-ewma-alpha must be in (0, 1]")
    if not 0.0 <= result.scheduler_request_sample_rate <= 1.0:
        parser.error("--scheduler-request-sample-rate must be between 0 and 1")
    if result.min_prefill_budget_per_step <= 0:
        parser.error("--min-prefill-budget-per-step must be positive")
    if result.max_decode_only_steps < 0:
        parser.error("--max-decode-only-steps must be non-negative")
    if result.hard_max_wait_ms <= 0:
        parser.error("--hard-max-wait-ms must be positive")
    if not 0.0 <= result.min_decode_reserve_ratio <= 1.0:
        parser.error("--min-decode-reserve-ratio must be between 0 and 1")
    if not 0.0 <= result.max_decode_reserve_ratio <= 1.0:
        parser.error("--max-decode-reserve-ratio must be between 0 and 1")
    if result.min_decode_reserve_ratio > result.max_decode_reserve_ratio:
        parser.error("dynamic decode reservation bounds are reversed")
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
