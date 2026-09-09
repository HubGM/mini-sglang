from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
import torch.nn.functional as F
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger
from transformers import AutoTokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .lifecycle import (
    LifecycleRegistry,
    RequestLifecycle,
    RequestLifecycleState,
)
from .policy import (
    PolicyController,
    SchedulingContext,
    SchedulingMetrics,
    UpstreamDefaultPolicy,
    create_scheduling_policy,
)
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)


class BatchPreparationError(RuntimeError):
    def __init__(self, phase: str, batch: Batch, cause: Exception) -> None:
        super().__init__(f"{phase}_batch_preparation_failed")
        self.phase = phase
        self.batch = batch
        self.__cause__ = cause


def _make_2d_indices(table_2d: torch.Tensor, ranges: List[Tuple[int, int, int]]) -> torch.Tensor:
    """
    Return the 1D indices for the given 2D table and ranges.

    Example: The underlying indices of a 2D table (3, 4) are:
        [[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]]
    For ranges [(0, 1, 3), (2, 0, 2)], the returned indices are [1, 2, 8, 9].

    Args:
        table_2d (torch.Tensor): The 2D table tensor.
        ranges (List[Tuple[int, int, int]]): A list of tuples (entry, begin, end),
            where `entry` is the row index in the 2D table, and `begin` and `end`
            specify the range of column indices to include.
    Returns:
        torch.Tensor: A 1D tensor of indices.
    """
    assert table_2d.dim() == 2 and table_2d.is_contiguous()
    STRIDE = table_2d.stride(0)
    needed_size = sum(end - begin for _, begin, end in ranges)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for entry, begin, end in ranges:
        length = end - begin
        offset += length
        torch.arange(
            begin + entry * STRIDE,
            end + entry * STRIDE,
            dtype=torch.int32,
            out=indices_host[offset - length : offset],
        )
    return indices_host.to(table_2d.device, non_blocking=True)


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    load_indices: torch.Tensor
    write_indices: torch.Tensor


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)
        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(self.device, self.engine.num_pages, config.cache_type)
        self.decode_manager = DecodeManager()
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        self.scheduling_policy = create_scheduling_policy(config.scheduling_policy)
        self.policy_controller = PolicyController(
            active_policy=self.scheduling_policy,
            fallback_policy=UpstreamDefaultPolicy(),
            failure_threshold=config.policy_failure_threshold,
        )
        self.scheduling_metrics = SchedulingMetrics(self.scheduling_policy.name)
        self.scheduler_metrics_path = config.scheduler_metrics_path
        self.scheduler_decision_timing = config.scheduler_decision_timing
        self.current_batch_state: str | None = None
        self.lifecycle_registry = LifecycleRegistry()
        self._selected_reqs: dict[int, Req] = {}
        self._inflight_reqs: dict[int, Req] = {}
        self._cancelled_uids: set[int] = set()
        self._released_uids: set[int] = set()
        self.step_id = 0

        from minisgl.server.health import HealthReporter

        self.health_reporter = HealthReporter(
            config.scheduler_health_queue,
            source=f"scheduler-{config.tp_info.rank}",
            min_interval_s=config.scheduler_heartbeat_interval_s,
        )

        self.tp_info = config.tp_info
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.page_table = self.engine.page_table
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens

    def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
        if last_data is None:
            return
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply = BatchTokenizerMsg(data=[])
        ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
        ongoing_uids = {req.uid for req in ongoing_reqs}

        max_seq_len = self.engine.max_seq_len
        for i, req in enumerate(batch.reqs):
            lifecycle = self.lifecycle_registry.get(req.uid)
            if lifecycle is not None and batch.is_prefill:
                lifecycle.mark_prefill_end()
            if req.uid in self._cancelled_uids:
                if req.uid not in ongoing_uids:
                    self._free_req_resources(req)
                    self._finish_cancellation(lifecycle, reply)
                    self._inflight_reqs.pop(req.uid, None)
                continue
            if req in self.finished_reqs or isinstance(req, ChunkedReq):
                if req.uid not in ongoing_uids:
                    self._inflight_reqs.pop(req.uid, None)
                continue

            next_token_id = next_tokens_cpu[i]
            req.append_host(next_token_id.unsqueeze(0))
            if lifecycle is not None:
                lifecycle.add_generated_token()
            next_token = int(next_token_id.item())
            finished = req.remain_len <= 0
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id
            if req.device_len >= max_seq_len - 1:
                finished = True
                logger.warning_rank0(f"Request {req.uid} reached {max_seq_len = }, dropped.")
            if not finished or lifecycle is None or lifecycle.claim_finish_callback():
                reply.data.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                    )
                )

            # free resources if the req is finished and not ongoing
            if finished:
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)
                if lifecycle is not None:
                    lifecycle.complete()
                self.policy_controller.on_request_finished(req.uid)
                logger.debug_rank0("Request %s is finished", req)
            if req.uid not in ongoing_uids:
                self._inflight_reqs.pop(req.uid, None)

        # free resources for finished but not ongoing reqs
        for req in self.finished_reqs.difference(ongoing_reqs):
            self._free_req_resources(req)

        # keep only ongoing reqs in the finished set
        self.finished_reqs.intersection_update(ongoing_reqs)
        self.send_result(reply)
        self._report_health(engine_step=True)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            lifecycle = self.lifecycle_registry.create(
                msg.uid,
                input_tokens=input_len,
                requested_output_tokens=msg.sampling_params.max_tokens,
                deadline_ms=msg.sampling_params.deadline_ms,
            )
            if msg.uid in self._cancelled_uids:
                lifecycle.request_cancellation("cancelled_before_admission")
                reply = BatchTokenizerMsg(data=[])
                self._finish_cancellation(lifecycle, reply)
                self.send_result(reply)
                return
            if input_len >= max_seq_len:
                lifecycle.fail("input_too_long")
                self._send_terminal(lifecycle, "failed", error="input_too_long")
                return logger.warning_rank0(
                    f"Input sequence len {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            max_output_len = max_seq_len - input_len
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
            lifecycle.transition(RequestLifecycleState.WAITING)
        elif isinstance(msg, AbortBackendMsg):
            self.abort_req(msg.uid, reason=msg.reason)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _append_terminal(
        self,
        reply: BatchTokenizerMsg,
        lifecycle: RequestLifecycle,
        terminal_reason: str,
        *,
        error: str | None = None,
    ) -> None:
        if lifecycle.claim_finish_callback():
            reply.data.append(
                DetokenizeMsg(
                    uid=lifecycle.uid,
                    next_token=-1,
                    finished=True,
                    terminal_reason=terminal_reason,
                    error=error,
                )
            )

    def _send_terminal(
        self,
        lifecycle: RequestLifecycle,
        terminal_reason: str,
        *,
        error: str | None = None,
    ) -> None:
        reply = BatchTokenizerMsg(data=[])
        self._append_terminal(
            reply, lifecycle, terminal_reason, error=error
        )
        self.send_result(reply)

    def _finish_cancellation(
        self,
        lifecycle: RequestLifecycle | None,
        reply: BatchTokenizerMsg | None = None,
    ) -> None:
        if lifecycle is None:
            return
        lifecycle.finish_cancelled(lifecycle.terminal_reason or "cancelled")
        self.policy_controller.on_request_cancelled(lifecycle.uid)
        if reply is None:
            self._send_terminal(lifecycle, "cancelled")
        else:
            self._append_terminal(reply, lifecycle, "cancelled")

    def _free_req_resources(self, req: Req) -> bool:
        if req.uid in self._released_uids:
            return False
        if hasattr(self.table_manager, "owns") and not self.table_manager.owns(
            req.table_idx
        ):
            raise RuntimeError(
                f"Request {req.uid} no longer owns table slot {req.table_idx}"
            )
        self.cache_manager.free_and_cache_finished_req(
            req.cache_handle,
            req.host_ids[: req.cached_len],
            self.page_table[req.table_idx, : req.cached_len],
        )
        self.table_manager.free(req.table_idx)
        self._released_uids.add(req.uid)
        return True

    def abort_req(self, uid: int, *, reason: str = "client_cancelled") -> bool:
        lifecycle = self.lifecycle_registry.get(uid)
        if lifecycle is None:
            self._cancelled_uids.add(uid)
            return False
        if lifecycle.is_terminal:
            return False

        lifecycle.request_cancellation(reason)
        self._cancelled_uids.add(uid)
        _, chunked_req = self.prefill_manager.pop_req(uid)
        running_req = self.decode_manager.abort_req(uid)
        selected_req = self._selected_reqs.pop(uid, None)
        inflight_req = self._inflight_reqs.get(uid)
        req = inflight_req or selected_req or running_req or chunked_req

        if inflight_req is not None:
            return True
        if req is not None:
            self._free_req_resources(req)
        self._finish_cancellation(lifecycle)
        self._assert_state_consistency()
        return True

    def _fail_batch(self, batch: Batch, reason: str) -> None:
        for req in tuple(batch.reqs):
            self.prefill_manager.pop_req(req.uid)
            self.decode_manager.remove_req(req)
            self._selected_reqs.pop(req.uid, None)
            self._inflight_reqs.pop(req.uid, None)
            lifecycle = self.lifecycle_registry.get(req.uid)
            if req.uid not in self._released_uids:
                self._free_req_resources(req)
            if lifecycle is not None and lifecycle.fail(reason):
                self._send_terminal(lifecycle, "failed", error=reason)

    def _assert_state_consistency(self) -> None:
        waiting_uids = {req.uid for req in self.prefill_manager.pending_list}
        running_uids = {req.uid for req in self.decode_manager.running_reqs}
        overlap = waiting_uids.intersection(running_uids)
        if overlap:
            raise RuntimeError("request_in_waiting_and_running")
        terminal_uids = self.lifecycle_registry.terminal_uids()
        if terminal_uids.intersection(waiting_uids.union(running_uids)):
            raise RuntimeError("terminal_request_is_runnable")

    def _report_health(self, *, engine_step: bool = False) -> None:
        counts = self.lifecycle_registry.terminal_counts()
        stats = {
            "waiting_count": len(self.prefill_manager.pending_list),
            "running_count": len(self.decode_manager.running_reqs),
            "cancelled_count": counts["cancelled"],
            "failed_count": counts["failed"],
        }
        if engine_step:
            self.health_reporter.engine_step(self.step_id, **stats)
        else:
            self.health_reporter.heartbeat(self.step_id, **stats)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        needed_size = sum(r.extend_len for r in batch.reqs)
        allocated: torch.Tensor | None = None
        try:
            allocated = self.cache_manager.allocate(needed_size)
            batch.out_loc = allocated
            # NOTE: Pad the batch if needed
            if padding_size := self.engine.graph_runner.pad_batch(batch):
                batch.out_loc = F.pad(
                    batch.out_loc, (0, padding_size), value=self.engine.dummy_page
                )
            # NOTE: prepare 2d indices for token ids loading and writing
            load_indices = _make_2d_indices(
                self.token_pool,
                [
                    (r.table_idx, r.cached_len, r.device_len)
                    for r in batch.padded_reqs
                ],
            )
            write_indices = _make_2d_indices(
                self.token_pool,
                [
                    (r.table_idx, r.device_len, r.device_len + 1)
                    for r in batch.reqs
                ],
            )
            # NOTE: write out_loc to page_table before `prepare_metadata`
            self.page_table.view(-1)[load_indices] = batch.out_loc
            self.engine.attn_backend.prepare_metadata(batch)
            return ForwardInput(
                batch=batch,
                sample_args=self.engine.sampler.prepare(batch),
                load_indices=load_indices,
                write_indices=write_indices,
            )
        except Exception as exc:
            if allocated is not None:
                self.cache_manager._free(allocated)
            raise BatchPreparationError(batch.phase, batch, exc) from exc

    def _schedule_next_batch(self) -> ForwardInput | None:
        self.step_id += 1
        context = SchedulingContext(
            waiting_requests=tuple(self.prefill_manager.pending_list),
            running_requests=tuple(self.decode_manager.running_reqs),
            available_token_budget=self.prefill_budget,
            available_kv_blocks=self.cache_manager.available_size,
            current_batch_state=self.current_batch_state,
            current_timestamp_ns=time.monotonic_ns(),
            step_id=self.step_id,
            measure_decision_overhead=self.scheduler_decision_timing,
        )
        pending_snapshot = tuple(self.prefill_manager.pending_list)
        chunked_snapshot = {
            req.uid: req.chunked_req for req in pending_snapshot
        }
        scheduled_prefill_batches: list[Batch] = []

        def schedule_prefill(budget: int) -> Batch | None:
            batch = self.prefill_manager.schedule_next_batch(budget)
            if batch is not None:
                scheduled_prefill_batches.append(batch)
            return batch

        def rollback() -> None:
            old_chunked = {
                id(req)
                for req in chunked_snapshot.values()
                if req is not None
            }
            for batch in scheduled_prefill_batches:
                for req in batch.reqs:
                    if id(req) not in old_chunked and req.uid not in self._released_uids:
                        self.cache_manager.unlock(req.cache_handle)
                        self.table_manager.free(req.table_idx)
            self.prefill_manager.pending_list = list(pending_snapshot)
            for pending_req in pending_snapshot:
                pending_req.chunked_req = chunked_snapshot[pending_req.uid]
                lifecycle = self.lifecycle_registry.get(pending_req.uid)
                if (
                    lifecycle is not None
                    and lifecycle.state == RequestLifecycleState.PREFILL_SELECTED
                ):
                    lifecycle.transition(RequestLifecycleState.WAITING)
            scheduled_prefill_batches.clear()

        selection = self.policy_controller.select(
            context,
            schedule_prefill,
            self.decode_manager.schedule_next_batch,
            rollback=rollback,
            terminal_uids=self.lifecycle_registry.terminal_uids(),
            released_uids=self._released_uids,
        )
        decision = selection.decision
        terminal_counts = self.lifecycle_registry.terminal_counts()
        self.scheduling_metrics.policy_failure_count = (
            self.policy_controller.total_failures
        )
        self.scheduling_metrics.record(
            context,
            decision,
            fallback_reason=selection.fallback_reason,
            cancelled_count=terminal_counts["cancelled"],
            failed_count=terminal_counts["failed"],
            maximum_waiting_age_ms=self.lifecycle_registry.maximum_waiting_age_ms(
                context.current_timestamp_ns
            ),
        )
        batch = decision.batch
        if batch is not None:
            for req in batch.reqs:
                self._selected_reqs[req.uid] = req
                lifecycle = self.lifecycle_registry.get(req.uid)
                if (
                    batch.is_prefill
                    and lifecycle is not None
                    and lifecycle.state == RequestLifecycleState.WAITING
                ):
                    lifecycle.transition(RequestLifecycleState.PREFILL_SELECTED)
            batch.reqs = [
                req for req in batch.reqs if req.uid not in self._cancelled_uids
            ]
            if not batch.reqs:
                return None
        self.current_batch_state = batch.phase if batch else None
        if batch is None:
            return None
        try:
            return self._prepare_batch(batch)
        except BatchPreparationError as exc:
            reason = f"{exc.phase}_prepare_exception"
            logger.exception("%s", reason)
            self._fail_batch(batch, reason)
            return None

    def _load_token_ids(self, input: ForwardInput) -> None:
        batch, load_indices = input.batch, input.load_indices
        batch.input_ids = self.token_pool.view(-1)[load_indices]

    def _write_token_ids(self, input: ForwardInput, output: ForwardOutput) -> None:
        self.token_pool.view(-1)[input.write_indices] = output.next_tokens_gpu

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        self._load_token_ids(forward_input)
        batch, sample_args = forward_input.batch, forward_input.sample_args
        for req in batch.reqs:
            self._selected_reqs.pop(req.uid, None)
            self._inflight_reqs[req.uid] = req
            lifecycle = self.lifecycle_registry.get(req.uid)
            if (
                batch.is_prefill
                and lifecycle is not None
                and lifecycle.state == RequestLifecycleState.PREFILL_SELECTED
            ):
                lifecycle.transition(RequestLifecycleState.PREFILL_RUNNING)
            elif batch.is_decode and lifecycle is not None:
                lifecycle.mark_first_decode()
        try:
            forward_output = self.engine.forward_batch(batch, sample_args)
        except Exception as exc:
            reason = getattr(exc, "stage", "engine_forward_exception")
            self.health_reporter.fatal(reason)
            self._fail_batch(batch, reason)
            raise
        self._write_token_ids(forward_input, forward_output)
        self.decode_manager.add_reqs(forward_input.batch.reqs)
        for req in batch.reqs:
            lifecycle = self.lifecycle_registry.get(req.uid)
            if lifecycle is None or not batch.is_prefill:
                continue
            if isinstance(req, ChunkedReq):
                lifecycle.transition(RequestLifecycleState.WAITING)
            else:
                lifecycle.transition(RequestLifecycleState.DECODING)
        return forward_output

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self._report_health()
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        self._report_health()
        blocking = not (
            last_data  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data, ongoing_data)
        return ongoing_data

    def normal_loop(self) -> None:
        self._report_health()
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data, None)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        metrics = self.scheduling_metrics.snapshot()
        logger.info_rank0("Scheduler policy metrics: %s", metrics)
        if self.scheduler_metrics_path and self.tp_info.is_primary():
            self.scheduling_metrics.write_json(self.scheduler_metrics_path)
        from minisgl.server.health import HealthEventKind

        self.health_reporter.emit(HealthEventKind.STOPPED)
        self.engine.shutdown()
