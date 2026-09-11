# TTFT Regression Diagnosis

## Frozen Evidence

This diagnosis uses the immutable Prompt 6B-R scheduler telemetry under
`mini-sglang-engine-lab-runtime-6br`. It does not modify or reclassify any of
the 36 formal runs. The old `deadline_aging` implementation is now named
`deadline_aging_v1`; the legacy CLI spelling remains an alias.

The v1 three-repeat medians versus `upstream_default` were:

| Workload | Upstream TTFT p95 ms | v1 TTFT p95 ms | Upstream TPOT p95 ms | v1 TPOT p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Mixed | 689.7 | 1367.7 | 131.3 | 104.2 |
| Long-prefill interference | 347.9 | 959.3 | 113.8 | 82.8 |
| Starvation stress | 355.9 | 741.5 | 86.4 | 64.5 |

The result is a real phase tradeoff: v1 improved decode cadence by 20.7% to
27.3%, but delayed requests that had not produced their first token.

## Step-Level Finding

The following counts are recomputed from all three v1 scheduler traces. A
decode-only step here means that at least one prefill was waiting, no prefill
was selected, and decode work was selected.

| Workload | Steps with waiting prefill | Decode-only while waiting | Longest decode-only streak by repeat |
| --- | ---: | ---: | --- |
| Mixed | 195 | 133 | 12 / 9 / 13 |
| Long-prefill interference | 177 | 114 | 6 / 6 / 6 |
| Starvation stress | 187 | 126 | 15 / 14 / 15 |

This rules out prompt arrival alone as the explanation. The fixed 0.5 decode
reservation, deadline comparison, and remaining-service estimate repeatedly
selected active decode while prefill remained queued. V1's aging reduced the
mixed-workload starvation count from deadline-only's 21 events to 4, but it
was a soft priority and could not provide a finite service bound.

## Waiting Distribution

Waiting is enqueue to first scheduler selection. Values below pool the three
matched v1 repeats and retain the predeclared short/medium/long classes.

| Workload / class | Requests | p50 ms | p95 ms | p99 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed / short | 30 | 27.42 | 594.74 | 598.39 | 599.87 |
| Mixed / medium | 30 | 0.78 | 221.76 | 278.43 | 298.70 |
| Mixed / long | 30 | 12.90 | 299.59 | 300.79 | 301.11 |
| Long-prefill / short | 72 | 0.68 | 125.22 | 154.70 | 154.74 |
| Long-prefill / long | 18 | 0.67 | 111.39 | 111.58 | 111.62 |
| Starvation / short | 108 | 1.07 | 278.15 | 337.49 | 366.45 |
| Starvation / long | 12 | 75.06 | 398.08 | 398.37 | 398.44 |

Short requests have the largest mixed-workload waiting tail, while the
starvation trace pushes long prefill close to the frozen 400 ms threshold.
The pattern is consistent with active decode suppressing admission, not with
one long chunk always winning prefill ordering.

## Phase Attribution Limits

Prompt 6B-R stored enqueue, first schedule, first token, and finish timing. It
did not store exact prefill start/end timestamps, request role, TPOT cadence
state, or chunk-deferral decisions. Therefore the old traces cannot honestly
split every request into enqueue-to-prefill-start, prefill execution,
prefill-end-to-first-token, and first-token-to-finish phases. They also cannot
prove whether a particular old delay came from a continued chunk or a fresh
prefill selection.

Prompt 6B-R2 adds these phase timestamps plus request class/role, PRE/POST
counts, prefill/hard urgency, dynamic reservation, decode-only streak, and
chunk-deferral reason codes. The HOLDOUT validity gate requires complete phase
telemetry for every completed request.

## Root Cause

V1 used one remaining-service slack model for phase competition. A running
request's decode remainder could become more urgent than an unserved request's
time-to-first-token target, and the fixed reservation reinforced that choice.
Because aging was only a continuously improving score, new deadline pressure
and repeated decode reservations could still defer old prefill. In short, v1
optimized the POST_FIRST_TOKEN tail without an explicit PRE_FIRST_TOKEN
service guarantee.

The v2 correction is deliberately narrow: phase-specific slack, finite
decode-only service, a hard FIFO wait bound, urgent-over-continuation ordering,
and bounded dynamic decode reservation. It does not change model execution,
KV ownership, output semantics, or the upstream fallback.
