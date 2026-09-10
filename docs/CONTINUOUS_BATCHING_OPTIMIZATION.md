# Continuous Batching Optimization

## What Changed

The pinned Mini-SGLang engine used prefill-first scheduling. This phase adds a
bounded policy path around the existing managers rather than replacing model
execution or KV ownership:

1. Capture immutable waiting, running, capacity, lifecycle, and deadline
   state at the start of a scheduling step.
2. Ask the active policy to choose exactly one prefill or decode batch.
3. Validate request membership, uniqueness, chunk sizes, and token use.
4. Roll manager mutations back before invoking the upstream fallback if an
   experimental policy fails.
5. Dispatch the selected batch through the unchanged model runner.
6. Feed completed-step timing into the deadline policy EWMA.

## Step-Level Interleaving

The engine's `Batch` has one phase, so a step is either prefill or decode.
The new policies therefore implement temporal continuous batching:

```text
prefill chunk -> decode step -> prefill chunk -> decode step -> ...
```

This is not presented as same-batch mixed prefill/decode. The concrete gains
are bounded long-prefill work and guaranteed decode opportunities under
contention.

## Budget And Chunk Mechanics

The default experimental budget is 2048 tokens per step and a single prompt
contributes at most 512 prefill tokens to a step. Priority-aware manager APIs
accept an ordered UID list without transferring ownership to the policy.
Decode selection accepts a deterministic order and a maximum request count.

Across all 36 formal runs, every non-idle experimental step respected the
2048-token budget and the 512-token prefill-chunk cap. The validity gate also
verified that all admitted requests reached a terminal completed state and
that client and scheduler generated-token totals matched.

## Policy Ablation

The table reports medians across three matched-trace repeats. Throughput is
completed requests/s; latency is milliseconds.

| Workload | Policy | Completed RPS | TTFT p95 | TPOT p95 | E2E p95 | Avg prefill chunk |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mixed | Upstream | 5.936 | 689.7 | 131.3 | 3970.7 | 719.7 |
| Mixed | Token budget | 5.797 | 1149.3 | 122.8 | 4042.2 | 308.4 |
| Mixed | Deadline | 5.160 | 1976.5 | 69.7 | 3448.8 | 308.4 |
| Mixed | Deadline + aging | 5.563 | 1367.7 | 104.2 | 3891.5 | 308.4 |
| Long prefill | Upstream | 6.991 | 347.9 | 113.8 | 3555.0 | 491.4 |
| Long prefill | Token budget | 6.869 | 561.6 | 100.8 | 3520.0 | 273.0 |
| Long prefill | Deadline | 6.568 | 943.7 | 76.3 | 3228.2 | 273.0 |
| Long prefill | Deadline + aging | 6.680 | 959.3 | 82.8 | 3407.6 | 273.0 |
| Starvation stress | Upstream | 11.258 | 355.9 | 86.4 | 2137.9 | 267.4 |
| Starvation stress | Token budget | 10.876 | 597.5 | 87.2 | 2143.2 | 191.0 |
| Starvation stress | Deadline | 10.873 | 602.6 | 81.3 | 2236.3 | 191.0 |
| Starvation stress | Deadline + aging | 10.473 | 741.5 | 64.5 | 1844.9 | 191.0 |

Token budgeting alone bounded prefill work but did not improve every latency.
Deadline ordering traded admission/TTFT delay for decode cadence. Relative to
upstream, deadline + aging reduced median TPOT p95 by 20.7%, 27.3%, and 25.4%
for mixed, long-prefill, and starvation-stress workloads. It increased TTFT
p95 by 98.3%, 175.7%, and 108.4%, respectively.

## Scheduler Cost

Median scheduler decision p95 was:

| Workload | Upstream us | Token budget us | Deadline us | Deadline + aging us |
| --- | ---: | ---: | ---: | ---: |
| Mixed | 326.2 | 406.2 | 465.4 | 488.8 |
| Long prefill | 331.9 | 264.9 | 436.6 | 458.5 |
| Starvation stress | 379.1 | 401.3 | 488.8 | 501.4 |

These numbers include policy ordering, manager scheduling, validation, and
instrumentation. They are not an isolated policy-call microbenchmark and must
not be described as pure framework overhead.

## Correctness And Recovery

- Four policies produced identical token-ID sequences for all three fixed
  Qwen3-8B prompts at temperature 0.
- The formal matrix completed 1200/1200 requests with no client error,
  scheduler failure, fallback, or cancellation.
- Every run ended with waiting=0 and running=0.
- The prior two-seed, 200-request cancellation stress remains the lifecycle
  gate: no orphan, failed request, or progressive KV/table leak was observed.

The optimization is therefore a validated scheduling mechanism, but the
current policy parameters are not a production tuning recommendation.
