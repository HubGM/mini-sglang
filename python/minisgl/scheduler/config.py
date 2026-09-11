from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    scheduling_policy: str = "upstream_default"
    scheduler_metrics_path: str | None = None
    scheduler_decision_timing: bool = True
    policy_failure_threshold: int = 3
    max_step_tokens: int = 2048
    max_prefill_chunk_tokens: int = 512
    decode_reserve_ratio: float = 0.5
    max_consecutive_prefill_steps: int = 1
    default_ttft_deadline_ms: float = 200.0
    default_tpot_deadline_ms: float = 50.0
    default_e2e_deadline_ms: float = 1200.0
    initial_prefill_ms_per_token: float = 0.15
    initial_decode_step_ms: float = 25.0
    service_ewma_alpha: float = 0.2
    max_wait_ms: float = 400.0
    aging_start_ms: float = 100.0
    aging_rate: float = 1.0
    min_prefill_budget_per_step: int = 256
    max_decode_only_steps: int = 2
    prefill_urgent_threshold_ms: float = 150.0
    hard_max_wait_ms: float = 300.0
    min_decode_reserve_ratio: float = 0.25
    max_decode_reserve_ratio: float = 0.65
    starvation_threshold_ms: float = 400.0
    scheduler_request_sample_rate: float = 0.1
    scheduler_max_step_records: int = 10_000
    scheduler_heartbeat_interval_s: float = 1.0
    scheduler_health_queue: object | None = field(
        default=None, repr=False, compare=False
    )
    offline_mode: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return max(self.max_extend_tokens, self.max_step_tokens)

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
