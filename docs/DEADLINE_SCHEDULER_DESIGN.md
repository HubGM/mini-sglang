# Deadline-Aware Token-Budget Scheduler Design

## Scope

This change adds three opt-in engine scheduling policies while preserving
`upstream_default` as the default and correctness control:

- `token_budget`
- `deadline_aware`
- `deadline_aging`

The policy boundary selects local Mini-SGLang prefill or decode work. It does
not route requests across Workers, execute model kernels, or own KV pages.
The scheduler still owns admission, manager mutation, KV allocation/release,
cancellation, and model execution.

## Upstream Control

The pinned upstream behavior tries a prefill batch first and falls back to
decode only when prefill cannot run. Its behavior and output semantics are
unchanged. Every experimental decision is validated; an invalid decision or
policy exception rolls manager state back and falls back to
`upstream_default`. Three consecutive policy failures open the policy circuit.

## Token-Budget Scheduling

`token_budget` enforces these defaults:

| Parameter | Value |
| --- | ---: |
| `max_step_tokens` | 2048 |
| `max_prefill_chunk_tokens` | 512 |
| `decode_reserve_ratio` | 0.5 |
| `max_consecutive_prefill_steps` | 1 |

A prefill token and one decode position each consume one unit of the step
budget. The scheduler records actual use and rejects a decision whose use is
negative, inconsistent with the selected batch, or above the declared budget.

Long prompts remain in the prefill manager and advance in chunks of at most
512 tokens. Mini-SGLang's current batch representation does not support a
single batch containing both prefill and decode requests, so this
implementation does not claim same-batch mixed execution. It performs real
prefill/decode interleaving at engine-step granularity.

When waiting and running requests coexist, decode credit accumulates according
to the reserve ratio. A decode step is selected when credit reaches one. The
consecutive-prefill cap independently forces decode after one contended
prefill step. These two gates preserve decode progress without bypassing the
token budget.

## Deadline-Aware Ordering

Each request carries TTFT and E2E deadlines. Defaults used by the formal
experiment are 200 ms and 1200 ms. The policy computes only from state
available at the current scheduling step:

```text
estimated_remaining_service_ms =
    remaining_input_tokens * ewma_prefill_ms_per_token
  + remaining_output_tokens * ewma_decode_step_ms

deadline_slack_ms =
    deadline_ms - request_age_ms - estimated_remaining_service_ms
```

Before the first token, priority uses the minimum of TTFT slack and E2E slack.
After the first token, it uses E2E slack. Lower slack has higher priority;
negative slack is valid and means the request is predicted to miss its target.

The service estimator starts conservatively at 0.15 ms per prefill token and
25 ms per decode step. Completed GPU steps update it with an EWMA using
`alpha=0.2`; one observation is clipped to 0.25x through 4x of the previous
estimate. It never reads a request's future completion time or actual future
token count.

## Aging And Starvation Prevention

`deadline_aging` subtracts a monotonic waiting-age bonus from raw deadline
slack:

```text
aging_bonus_ms = max(0, age_ms - 100) * 1.0
effective_slack_ms = raw_slack_ms - aging_bonus_ms
```

An unserved request with age at least `max_wait_ms=400` enters the urgent set.
Urgent waiting requests are ordered before non-urgent requests. Decode can
still be forced by the consecutive-prefill cap, so urgent prefill does not
starve existing decode indefinitely.

The experiment fixed starvation before execution as waiting at least 400 ms,
which is twice the 200 ms TTFT SLO. A request is marked once at terminal
recording based on enqueue-to-first-schedule time.

## Lifecycle And KV Safety

Scheduling contexts contain immutable tuples and immutable per-request
snapshots. A decision cannot select a duplicate request, overlap prefill and
decode sets, select a terminal or released request, or name a request outside
the corresponding manager. The scheduler snapshots pending and chunk state
before invoking a policy and restores those snapshots before fallback.

Cancellation remains scheduler-owned. Waiting, selected, in-flight prefill,
decode, disconnect, and duplicate cancellation paths use the lifecycle
registry and release request-owned table/KV resources at most once. Policy
finish and cancellation callbacks do not free KV directly.

## Telemetry

Bounded step records include policy, queue sizes, selected prefill/decode
counts and tokens, budget/use, chunk size, maximum waiting age, minimum slack,
urgent count, reason codes, and decision latency. Sampled terminal request
records include token counts and lifecycle timestamps, but not prompts or raw
token IDs.

The API exposes a metrics reset message used after warm-up. Orderly API
shutdown forwards a shutdown message through tokenizer to scheduler so the
final aggregate is written before the process exits.

## Known Limitations

- The cost model is a small online EWMA, not a calibrated per-shape predictor.
- Deadlines are request-local and do not model cluster routing or preemption.
- Step-level interleaving cannot realize the utilization benefits of true
  same-batch prefill/decode.
- The formal results show a TTFT versus TPOT tradeoff and do not establish a
  generally superior default policy. See `DEADLINE_SCHEDULER_EXPERIMENT.md`.
