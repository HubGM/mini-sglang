from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Sequence

from minisgl.core import Batch

from .policy import (
    BaseSchedulingPolicy,
    RequestSchedulingInfo,
    ScheduleDecode,
    SchedulePrefill,
    SchedulingContext,
    SchedulingDecision,
    SchedulingPolicyConfig,
)


@dataclass
class ServiceTimeEstimator:
    prefill_ms_per_token: float
    decode_step_ms: float
    alpha: float

    def observe(
        self,
        *,
        phase: str,
        elapsed_ms: float,
        prefill_tokens: int,
        decode_tokens: int,
    ) -> None:
        if elapsed_ms <= 0:
            return
        if phase == "prefill" and prefill_tokens > 0:
            observed = elapsed_ms / prefill_tokens
            self.prefill_ms_per_token = self._ewma(
                self.prefill_ms_per_token, observed
            )
        elif phase == "decode" and decode_tokens > 0:
            self.decode_step_ms = self._ewma(self.decode_step_ms, elapsed_ms)

    def _ewma(self, previous: float, observed: float) -> float:
        bounded = min(max(observed, previous * 0.25), previous * 4.0)
        return self.alpha * bounded + (1.0 - self.alpha) * previous

    def remaining_ms(self, info: RequestSchedulingInfo) -> float:
        return (
            info.remaining_input_tokens * self.prefill_ms_per_token
            + info.remaining_output_tokens * self.decode_step_ms
        )


class TokenBudgetPolicy(BaseSchedulingPolicy):
    name = "token_budget"
    version = "1"

    def __init__(self, config: SchedulingPolicyConfig) -> None:
        self.config = config
        self._decode_credit = 0.0
        self._consecutive_prefill_steps = 0

    def _waiting_order(self, context: SchedulingContext) -> tuple[int, ...]:
        return tuple(req.uid for req in context.waiting_requests)

    def _running_order(self, context: SchedulingContext) -> tuple[int, ...]:
        return tuple(req.uid for req in context.running_requests)

    def _priority_metadata(
        self, context: SchedulingContext
    ) -> tuple[float | None, int, tuple[tuple[int, float], ...]]:
        _ = context
        return None, 0, ()

    def _prefer_decode(
        self,
        context: SchedulingContext,
        *,
        reserve_due: bool,
        forced_decode: bool,
    ) -> bool:
        _ = context
        return reserve_due or forced_decode

    def _decode_reserve_ratio(self, context: SchedulingContext) -> float:
        _ = context
        return self.config.decode_reserve_ratio

    def _prefill_budget(self, context: SchedulingContext) -> int:
        return min(self.config.max_step_tokens, context.available_token_budget)

    def _make_decision(
        self,
        batch: Batch | None,
        *,
        started_ns: int,
        context: SchedulingContext,
        reason_codes: list[str],
        minimum_slack_ms: float | None,
        urgent_count: int,
        request_slack_ms: tuple[tuple[int, float], ...],
        decode_reserve_ratio: float,
    ) -> SchedulingDecision:
        prefill = tuple(batch.reqs) if batch is not None and batch.is_prefill else ()
        decode = tuple(batch.reqs) if batch is not None and batch.is_decode else ()
        chunks = tuple((req.uid, req.extend_len) for req in prefill)
        used = sum(size for _, size in chunks) if prefill else len(decode)
        return SchedulingDecision(
            batch=batch,
            selected_prefill_requests=prefill,
            selected_decode_requests=decode,
            prefill_chunk_sizes=chunks,
            reason_codes=tuple(reason_codes),
            decision_latency_ns=(
                time.perf_counter_ns() - started_ns
                if context.measure_decision_overhead
                else 0
            ),
            token_budget=self.config.max_step_tokens,
            token_budget_used=used,
            minimum_slack_ms=minimum_slack_ms,
            urgent_request_count=urgent_count,
            request_slack_ms=request_slack_ms,
            pre_first_token_count=sum(
                info.first_token_time_ns is None for info in context.request_info
            ),
            post_first_token_count=sum(
                info.first_token_time_ns is not None for info in context.request_info
            ),
            dynamic_decode_reserve_ratio=decode_reserve_ratio,
        )

    def select(
        self,
        context: SchedulingContext,
        schedule_prefill: SchedulePrefill,
        schedule_decode: ScheduleDecode,
    ) -> SchedulingDecision:
        started_ns = time.perf_counter_ns() if context.measure_decision_overhead else 0
        has_waiting = bool(context.waiting_requests)
        has_running = bool(context.running_requests)
        minimum_slack, urgent_count, request_slack = self._priority_metadata(context)
        reasons = ["step_level_interleaving", "token_budget_enforced"]

        decode_reserve_ratio = self._decode_reserve_ratio(context)
        if has_running:
            self._decode_credit += decode_reserve_ratio
        else:
            self._decode_credit = 0.0
        reserve_due = has_running and self._decode_credit >= 1.0
        forced_decode = (
            has_running
            and has_waiting
            and self._consecutive_prefill_steps
            >= self.config.max_consecutive_prefill_steps
        )

        if has_running and not has_waiting:
            phase = "decode"
        elif has_waiting and not has_running:
            phase = "prefill"
        elif has_running and has_waiting:
            phase = (
                "decode"
                if self._prefer_decode(
                    context,
                    reserve_due=reserve_due,
                    forced_decode=forced_decode,
                )
                else "prefill"
            )
        else:
            phase = "idle"

        batch: Batch | None = None
        if phase == "decode":
            reasons.append("decode_reserved" if reserve_due else "decode_selected")
            batch = schedule_decode(
                min(self.config.max_step_tokens, len(context.running_requests)),
                self._running_order(context),
            )
            if batch is None and has_waiting:
                reasons.append("decode_unavailable_prefill_fallback")
                phase = "prefill"
        if phase == "prefill":
            reasons.append("prefill_selected")
            batch = schedule_prefill(
                self._prefill_budget(context),
                self._waiting_order(context),
                self.config.max_prefill_chunk_tokens,
            )
            if batch is None and has_running:
                reasons.append("prefill_unavailable_decode_fallback")
                phase = "decode"
                batch = schedule_decode(
                    min(self.config.max_step_tokens, len(context.running_requests)),
                    self._running_order(context),
                )

        if batch is None:
            reasons.append("idle_no_runnable_batch")
            self._consecutive_prefill_steps = 0
        elif batch.is_prefill:
            self._consecutive_prefill_steps = (
                self._consecutive_prefill_steps + 1 if has_running else 0
            )
        else:
            self._consecutive_prefill_steps = 0
            self._decode_credit = max(0.0, self._decode_credit - 1.0)

        return self._make_decision(
            batch,
            started_ns=started_ns,
            context=context,
            reason_codes=reasons,
            minimum_slack_ms=minimum_slack,
            urgent_count=urgent_count,
            request_slack_ms=request_slack,
            decode_reserve_ratio=decode_reserve_ratio,
        )


class DeadlineAwarePolicy(TokenBudgetPolicy):
    name = "deadline_aware"
    version = "1"

    def __init__(self, config: SchedulingPolicyConfig) -> None:
        super().__init__(config)
        self.estimator = ServiceTimeEstimator(
            prefill_ms_per_token=config.initial_prefill_ms_per_token,
            decode_step_ms=config.initial_decode_step_ms,
            alpha=config.service_ewma_alpha,
        )

    def _info_by_uid(
        self, context: SchedulingContext
    ) -> dict[int, RequestSchedulingInfo]:
        return {info.uid: info for info in context.request_info}

    def raw_slack_ms(
        self, info: RequestSchedulingInfo, now_ns: int
    ) -> float:
        age_ms = info.age_ms(now_ns)
        e2e_slack = (
            info.e2e_deadline_ms
            - age_ms
            - self.estimator.remaining_ms(info)
        )
        if info.first_token_time_ns is not None:
            return e2e_slack
        ttft_service = (
            info.remaining_input_tokens * self.estimator.prefill_ms_per_token
        )
        ttft_slack = info.ttft_deadline_ms - age_ms - ttft_service
        return min(ttft_slack, e2e_slack)

    def effective_slack_ms(
        self, info: RequestSchedulingInfo, now_ns: int
    ) -> float:
        return self.raw_slack_ms(info, now_ns)

    def is_urgent(self, info: RequestSchedulingInfo, now_ns: int) -> bool:
        _ = (info, now_ns)
        return False

    def _ordered(
        self, context: SchedulingContext, uids: Sequence[int]
    ) -> tuple[int, ...]:
        info_by_uid = self._info_by_uid(context)
        return tuple(
            sorted(
                uids,
                key=lambda uid: (
                    self.effective_slack_ms(
                        info_by_uid[uid], context.current_timestamp_ns
                    ),
                    info_by_uid[uid].enqueue_time_ns,
                    uid,
                ),
            )
        )

    def _waiting_order(self, context: SchedulingContext) -> tuple[int, ...]:
        return self._ordered(
            context, tuple(req.uid for req in context.waiting_requests)
        )

    def _running_order(self, context: SchedulingContext) -> tuple[int, ...]:
        return self._ordered(
            context, tuple(req.uid for req in context.running_requests)
        )

    def _priority_metadata(
        self, context: SchedulingContext
    ) -> tuple[float | None, int, tuple[tuple[int, float], ...]]:
        values = tuple(
            (
                info.uid,
                self.effective_slack_ms(info, context.current_timestamp_ns),
            )
            for info in context.request_info
        )
        urgent = sum(
            self.is_urgent(info, context.current_timestamp_ns)
            for info in context.request_info
        )
        return min((value for _, value in values), default=None), urgent, values

    def _prefer_decode(
        self,
        context: SchedulingContext,
        *,
        reserve_due: bool,
        forced_decode: bool,
    ) -> bool:
        if reserve_due or forced_decode:
            return True
        info_by_uid = self._info_by_uid(context)
        waiting_slack = min(
            self.effective_slack_ms(
                info_by_uid[req.uid], context.current_timestamp_ns
            )
            for req in context.waiting_requests
        )
        running_slack = min(
            self.effective_slack_ms(
                info_by_uid[req.uid], context.current_timestamp_ns
            )
            for req in context.running_requests
        )
        return running_slack <= waiting_slack

    def on_step_completed(
        self,
        *,
        phase: str,
        elapsed_ms: float,
        prefill_tokens: int,
        decode_tokens: int,
    ) -> None:
        self.estimator.observe(
            phase=phase,
            elapsed_ms=elapsed_ms,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        )


class DeadlineAgingPolicy(DeadlineAwarePolicy):
    name = "deadline_aging_v1"
    version = "1"

    def effective_slack_ms(
        self, info: RequestSchedulingInfo, now_ns: int
    ) -> float:
        age_ms = info.age_ms(now_ns)
        aging_bonus = max(0.0, age_ms - self.config.aging_start_ms) * (
            self.config.aging_rate
        )
        return self.raw_slack_ms(info, now_ns) - aging_bonus

    def is_urgent(self, info: RequestSchedulingInfo, now_ns: int) -> bool:
        return info.first_token_time_ns is None and (
            info.age_ms(now_ns) >= self.config.max_wait_ms
        )

    def _ordered(
        self, context: SchedulingContext, uids: Sequence[int]
    ) -> tuple[int, ...]:
        info_by_uid = self._info_by_uid(context)
        return tuple(
            sorted(
                uids,
                key=lambda uid: (
                    not self.is_urgent(
                        info_by_uid[uid], context.current_timestamp_ns
                    ),
                    self.effective_slack_ms(
                        info_by_uid[uid], context.current_timestamp_ns
                    ),
                    info_by_uid[uid].enqueue_time_ns,
                    uid,
                ),
            )
        )

    def _prefer_decode(
        self,
        context: SchedulingContext,
        *,
        reserve_due: bool,
        forced_decode: bool,
    ) -> bool:
        if forced_decode:
            return True
        info_by_uid = self._info_by_uid(context)
        if any(
            self.is_urgent(info_by_uid[req.uid], context.current_timestamp_ns)
            for req in context.waiting_requests
        ):
            return False
        return super()._prefer_decode(
            context,
            reserve_due=reserve_due,
            forced_decode=forced_decode,
        )


class DeadlineAgingV2Policy(DeadlineAwarePolicy):
    name = "deadline_aging_v2"
    version = "2"

    def __init__(self, config: SchedulingPolicyConfig) -> None:
        super().__init__(config)
        self._consecutive_decode_only_steps = 0

    def ttft_slack_ms(
        self, info: RequestSchedulingInfo, now_ns: int
    ) -> float:
        return (
            info.ttft_deadline_ms
            - info.age_ms(now_ns)
            - info.remaining_input_tokens * self.estimator.prefill_ms_per_token
        )

    def decode_slack_ms(
        self, info: RequestSchedulingInfo, now_ns: int
    ) -> float:
        e2e_slack = (
            info.e2e_deadline_ms
            - info.age_ms(now_ns)
            - info.remaining_output_tokens * self.estimator.decode_step_ms
        )
        if info.last_token_time_ns is None:
            return e2e_slack
        cadence_age_ms = max(0, now_ns - info.last_token_time_ns) / 1_000_000
        cadence_slack = info.tpot_deadline_ms - cadence_age_ms
        return min(e2e_slack, cadence_slack)

    def raw_slack_ms(
        self, info: RequestSchedulingInfo, now_ns: int
    ) -> float:
        if info.first_token_time_ns is None:
            return self.ttft_slack_ms(info, now_ns)
        return self.decode_slack_ms(info, now_ns)

    def effective_slack_ms(
        self, info: RequestSchedulingInfo, now_ns: int
    ) -> float:
        slack = self.raw_slack_ms(info, now_ns)
        if info.first_token_time_ns is not None:
            return slack
        aging_bonus = max(
            0.0, info.age_ms(now_ns) - self.config.aging_start_ms
        ) * self.config.aging_rate
        return slack - aging_bonus

    def is_hard_urgent(
        self, info: RequestSchedulingInfo, now_ns: int
    ) -> bool:
        return (
            info.first_token_time_ns is None
            and info.first_scheduled_time_ns is None
            and info.age_ms(now_ns) >= self.config.hard_max_wait_ms
        )

    def is_prefill_urgent(
        self, info: RequestSchedulingInfo, now_ns: int
    ) -> bool:
        return info.first_token_time_ns is None and (
            self.ttft_slack_ms(info, now_ns) <= 0
            or (
                info.first_scheduled_time_ns is None
                and info.age_ms(now_ns)
                >= self.config.prefill_urgent_threshold_ms
            )
        )

    def is_urgent(self, info: RequestSchedulingInfo, now_ns: int) -> bool:
        return self.is_hard_urgent(info, now_ns)

    def _waiting_order(self, context: SchedulingContext) -> tuple[int, ...]:
        info_by_uid = self._info_by_uid(context)
        now_ns = context.current_timestamp_ns

        def priority(
            uid: int,
        ) -> tuple[bool, bool, float, int, bool, int]:
            info = info_by_uid[uid]
            hard_urgent = self.is_hard_urgent(info, now_ns)
            ttft_slack = self.ttft_slack_ms(info, now_ns)
            negative_ttft = ttft_slack <= 0
            return (
                not hard_urgent,
                False if hard_urgent else not negative_ttft,
                0.0 if hard_urgent or not negative_ttft else ttft_slack,
                info.enqueue_time_ns,
                info.is_chunk_continuation,
                uid,
            )

        return tuple(
            sorted(
                (req.uid for req in context.waiting_requests),
                key=priority,
            )
        )

    def _running_order(self, context: SchedulingContext) -> tuple[int, ...]:
        info_by_uid = self._info_by_uid(context)
        now_ns = context.current_timestamp_ns
        return tuple(
            sorted(
                (req.uid for req in context.running_requests),
                key=lambda uid: (
                    self.decode_slack_ms(info_by_uid[uid], now_ns),
                    info_by_uid[uid].enqueue_time_ns,
                    uid,
                ),
            )
        )

    def _decode_reserve_ratio(self, context: SchedulingContext) -> float:
        lower = self.config.min_decode_reserve_ratio
        upper = self.config.max_decode_reserve_ratio
        if not context.running_requests:
            return lower
        if not context.waiting_requests:
            return upper

        info_by_uid = self._info_by_uid(context)
        now_ns = context.current_timestamp_ns
        waiting = [info_by_uid[req.uid] for req in context.waiting_requests]
        running = [info_by_uid[req.uid] for req in context.running_requests]
        if any(self.is_hard_urgent(info, now_ns) for info in waiting):
            return lower

        min_ttft_slack = min(self.ttft_slack_ms(info, now_ns) for info in waiting)
        min_decode_slack = min(
            self.decode_slack_ms(info, now_ns) for info in running
        )
        count_share = len(running) / (len(running) + len(waiting))
        scale = max(
            self.config.default_ttft_deadline_ms,
            self.config.default_tpot_deadline_ms,
            1.0,
        )
        relative = max(
            -1.0,
            min(1.0, (min_ttft_slack - min_decode_slack) / scale),
        )
        slack_share = (relative + 1.0) / 2.0
        pressure = 0.5 * count_share + 0.5 * slack_share
        return max(lower, min(upper, lower + (upper - lower) * pressure))

    def _prefer_decode(
        self,
        context: SchedulingContext,
        *,
        reserve_due: bool,
        forced_decode: bool,
    ) -> bool:
        if forced_decode:
            return True
        info_by_uid = self._info_by_uid(context)
        now_ns = context.current_timestamp_ns
        waiting = [info_by_uid[req.uid] for req in context.waiting_requests]
        if (
            self._consecutive_decode_only_steps
            >= self.config.max_decode_only_steps
            or any(self.is_hard_urgent(info, now_ns) for info in waiting)
            or any(self.is_prefill_urgent(info, now_ns) for info in waiting)
        ):
            return False
        running = [info_by_uid[req.uid] for req in context.running_requests]
        decode_is_late = min(
            self.decode_slack_ms(info, now_ns) for info in running
        ) <= 0
        prefill_is_late = min(
            self.ttft_slack_ms(info, now_ns) for info in waiting
        ) <= 0
        return reserve_due or (decode_is_late and not prefill_is_late)

    def _prefill_budget(self, context: SchedulingContext) -> int:
        available = min(
            self.config.max_step_tokens, context.available_token_budget
        )
        return max(0, available)

    def select(
        self,
        context: SchedulingContext,
        schedule_prefill: SchedulePrefill,
        schedule_decode: ScheduleDecode,
    ) -> SchedulingDecision:
        previous_decode_streak = self._consecutive_decode_only_steps
        decision = super().select(context, schedule_prefill, schedule_decode)
        if (
            decision.batch is not None
            and decision.batch.is_decode
            and context.waiting_requests
        ):
            self._consecutive_decode_only_steps += 1
        elif decision.batch is not None and decision.batch.is_prefill:
            self._consecutive_decode_only_steps = 0
        elif not context.waiting_requests:
            self._consecutive_decode_only_steps = 0

        info_by_uid = self._info_by_uid(context)
        now_ns = context.current_timestamp_ns
        waiting_info = [
            info_by_uid[req.uid]
            for req in context.waiting_requests
            if req.uid in info_by_uid
        ]
        hard_count = sum(
            self.is_hard_urgent(info, now_ns) for info in waiting_info
        )
        prefill_urgent_count = sum(
            self.is_prefill_urgent(info, now_ns) for info in waiting_info
        )
        selected_uids = {
            req.uid for req in decision.selected_prefill_requests
        }
        chunk_deferred = bool(selected_uids) and any(
            info.is_chunk_continuation and info.uid not in selected_uids
            for info in waiting_info
        )
        reasons = list(decision.reason_codes)
        if hard_count and decision.selected_prefill_requests:
            reasons.append("hard_urgent_prefill")
        elif prefill_urgent_count and decision.selected_prefill_requests:
            reasons.append("prefill_urgent")
        if (
            previous_decode_streak >= self.config.max_decode_only_steps
            and decision.selected_prefill_requests
        ):
            reasons.append("max_decode_only_prefill_guarantee")
        if chunk_deferred:
            reasons.append("chunk_continuation_deferred")
        return replace(
            decision,
            reason_codes=tuple(reasons),
            prefill_urgent_request_count=prefill_urgent_count,
            hard_urgent_request_count=hard_count,
            urgent_request_count=hard_count,
            consecutive_decode_only_steps=self._consecutive_decode_only_steps,
            chunk_continuation_deferred=chunk_deferred,
        )
