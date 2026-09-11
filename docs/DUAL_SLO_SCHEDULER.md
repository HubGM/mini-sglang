# Dual-SLO Deadline Scheduler

## Policy Boundary

`deadline_aging_v2` is an opt-in engine-level policy. It receives immutable
snapshots, selects one prefill or decode batch, and returns a decision that is
validated before execution. The scheduler retains ownership of manager
mutation, KV allocation/release, model execution, lifecycle transitions,
cancellation, rollback, and the `upstream_default` fallback.

The implementation preserves `deadline_aging_v1` for direct comparison and
keeps `deadline_aging` as its legacy alias.

## Phase-Specific Slack

V2 separates requests by observable lifecycle state.

For `PRE_FIRST_TOKEN` requests:

```text
estimated_time_to_first_token_ms =
    remaining_input_tokens * ewma_prefill_ms_per_token

ttft_slack_ms =
    ttft_deadline_ms - request_age_ms
    - estimated_time_to_first_token_ms
```

For `POST_FIRST_TOKEN` requests:

```text
e2e_slack_ms =
    e2e_deadline_ms - request_age_ms
    - remaining_output_tokens * ewma_decode_step_ms

tpot_slack_ms =
    tpot_deadline_ms - time_since_last_generated_token_ms

decode_slack_ms = min(e2e_slack_ms, tpot_slack_ms)
```

The estimate uses only current token counts, request `max_tokens`, monotonic
timestamps, and EWMA observations from completed engine steps. It never uses
future output length or actual completion time. The conservative initial EWMA
is the fallback until observations exist.

## Prefill Service Guarantee

When both phases are runnable, decode still receives a bounded temporal
reservation. Prefill receives service when any of these conditions holds:

- the configured `max_decode_only_steps` has been reached;
- a waiting request has non-positive TTFT slack;
- an unserved request reaches `prefill_urgent_threshold_ms`;
- an unserved request reaches `hard_max_wait_ms`.

The selected prefill is offered the available step budget, capped by
`max_step_tokens`; each request remains capped by
`max_prefill_chunk_tokens`. Thus the guarantee creates an execution
opportunity without exceeding token or KV safety limits. A separate
`max_consecutive_prefill_steps` bound preserves finite decode progress.

## Hard Max-Wait

An unserved PRE_FIRST_TOKEN request whose age reaches `hard_max_wait_ms`
enters `HARD_URGENT`. Hard-urgent requests precede all ordinary deadline
scores and are ordered by enqueue time, which is also their hard-urgent entry
order because the threshold is fixed. New requests cannot overtake this FIFO.
Cancellation removes a request through the existing lifecycle path; the
policy never owns or frees its KV state.

The formal starvation threshold is independently frozen at 400 ms before DEV
and HOLDOUT. It is not changed to manufacture a zero-starvation result.

## Chunk Ordering

Waiting prefill priority is:

1. hard-urgent FIFO;
2. negative TTFT slack, most negative first;
3. oldest enqueue time;
4. chunk continuation as a tie-breaker.

This permits an urgent short prefill to precede an already-started long
chunk. The bounded telemetry records each deferred continuation and its reason
without exporting request IDs, prompts, or token IDs.

## Dynamic Decode Reservation

The decode reservation is computed from active decode count, waiting prefill
count, minimum decode slack, and minimum TTFT slack. It is clamped to
`min_decode_reserve_ratio` and `max_decode_reserve_ratio`. TTFT pressure moves
the reservation toward the lower bound; decode E2E/TPOT pressure moves it
toward the upper bound. Hard urgency immediately uses the lower bound.

No branch is workload-specific. The same frozen configuration is used for
all HOLDOUT workloads.

## Safety And Fallback

Every decision must satisfy request membership, uniqueness, phase separation,
chunk accounting, and total token-budget invariants. Invalid decisions or
policy exceptions roll manager state back and invoke `upstream_default`.
Repeated failures open the existing circuit. No v2 code changes model kernels,
sampling, tokenization, or cache release.

## DEV And HOLDOUT Protocol

The candidate set is declared in `benchmark/deadline_v2_matrix.py` and capped
at three. DEV uses mixed, long-prefill interference, and a deadline-induced
starvation trace with seed 20261001. Selection is lexicographic:

1. starvation count;
2. TTFT p95;
3. TPOT p95;
4. negative completed RPS.

The selected configuration is atomically frozen with `FINAL_CONFIG_HASH`.
HOLDOUT uses untouched seeds 20261101 through 20261103 and compares only
upstream, v1, and v2 across three workloads and three repeats: 27 formal runs.
The final summary is rejected unless all 27 runs are VALID, matched policies
share each trace, and all v2 runs carry the frozen hash.

Runtime results, logs, per-request hashes, and correctness hashes remain under
`/lfs1/users/chyan/mini-sglang-engine-lab-runtime-6br2`. Only aggregate,
de-identified HOLDOUT statistics may enter Git.
