# TTFT Fairness Final Experiment

## Executive Summary

`deadline_aging_v2` fixes the admission fairness failure measured in
`deadline_aging_v1`, but it is not a universally better replacement for the
upstream scheduler. Across the three-repeat Qwen3-8B HOLDOUT, v2 reduced
short-request TTFT p95 by 17.8% to 34.2% relative to upstream and reduced v1's
aggregate TTFT p95 by 31.5% to 51.9%. It also reduced v1 maximum waiting by
17.5% to 84.5% and eliminated all five request-level v1 starvation events in
the new mixed HOLDOUT.

The cost is visible in the class aggregate. Long-request TTFT increased enough
that overall v2 TTFT p95 remained 39.3% to 41.1% worse than upstream. V2 did
not preserve v1's 15% or greater TPOT improvement. Completed-throughput loss
was bounded to 0.9% to 2.6%, within the predeclared 5% guardrail. The honest
conclusion is therefore improved admission fairness and short-request
responsiveness, not improved overall latency.

## Environment And Protocol

| Item | Value |
| --- | --- |
| GPU | GPU 9, NVIDIA RTX A6000, 49,140 MiB |
| Container | `llm-servelab-engine-6br2` |
| Image | `sha256:6e0b9f24bcf75cbe3ea72f22340eb9e85838220f3453a9044f419120e5ce79c6` |
| Model | Local Qwen3-8B, read-only mount |
| Policies | `upstream_default`, `deadline_aging_v1`, `deadline_aging_v2` |
| HOLDOUT seeds | 20261101, 20261102, 20261103 |
| SLO | TTFT 200 ms, TPOT 50 ms, E2E 1200 ms |
| Starvation threshold | 400 ms, frozen before DEV |
| Runtime output | `/lfs1/users/chyan/mini-sglang-engine-lab-runtime-6br2` |

The matrix contains 3 policies x 3 workloads x 3 repeats. Policy order is
deterministically randomized within each workload/repeat block. Policies in a
block use the same trace fingerprint; all v2 runs carry the frozen config
hash. Warm-up is excluded from formal statistics. All 27 runs passed policy,
token-budget, chunk, phase-telemetry, client/scheduler token reconciliation,
fallback, lifecycle, and final-queue validity gates. The 900/900 requests
completed successfully with 28,715 generated tokens, zero client errors, zero
policy failures, zero fallbacks, and final waiting/running equal to zero.

Reported centers are medians of three HOLDOUT repeats. Brackets are bootstrap
95% confidence intervals for the median. With only three repeats, these
intervals frequently span the observed repeat range and do not establish
high-powered statistical significance.

## Frozen Configuration

DEV compared exactly three declared candidates on separate seed 20261001.
Selection was lexicographic by starvation count, TTFT p95, TPOT p95, then
negative completed RPS. `prefill_strict` was selected without examining
HOLDOUT:

| Parameter | Value |
| --- | ---: |
| `max_consecutive_prefill_steps` | 2 |
| `min_prefill_budget_per_step` | 512 |
| `max_decode_only_steps` | 1 |
| `prefill_urgent_threshold_ms` | 100 |
| `hard_max_wait_ms` | 250 |
| `min_decode_reserve_ratio` | 0.20 |
| `max_decode_reserve_ratio` | 0.60 |

`FINAL_CONFIG_HASH`:
`3007c3ac046d7c85842d0254162d36ddef8277befe9b6747165a93ef1d5b20b0`.
The hash matched all nine v2 HOLDOUT runs.

## Why V1 Regressed TTFT

V1 applied one remaining-service slack model to both unserved prefill and
active decode. Decode requests repeatedly became more urgent under that score,
while the fixed 0.5 decode reservation supplied no finite service guarantee
for PRE_FIRST_TOKEN work. The frozen Prompt 6B-R traces contain 114 to 133
decode-only steps while prefill was waiting per workload aggregate, with
longest streaks of 6 to 15 steps. Aging softened the priority but could still
be overtaken by continuing deadline pressure.

V2 separates TTFT slack from E2E/TPOT slack, bounds decode-only streaks,
reserves a minimum prefill budget, promotes old requests into hard-urgent
FIFO, lets urgent new prefill precede chunk continuation, and dynamically
clamps decode reservation. Full attribution and the limits of the old phase
telemetry are documented in `TTFT_REGRESSION_DIAGNOSIS.md`; the policy
mechanics are in `DUAL_SLO_SCHEDULER.md`.

## Primary HOLDOUT Results

| Workload | Policy | Completed RPS | TTFT p95 / p99 ms | TPOT p95 / p99 ms | E2E p95 / p99 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Long-prefill interference | upstream | 6.956 [6.952, 6.969] | 367.7 [365.4, 368.1] / 379.5 [376.1, 380.8] | 112.6 [112.1, 114.2] / 115.4 [115.1, 117.0] | 3577.3 [3564.7, 3578.1] / 3595.8 [3587.4, 3600.0] |
| Long-prefill interference | v1 | 6.720 [6.705, 6.726] | 926.2 [925.2, 940.0] / 1007.3 [972.7, 1008.3] | 86.9 [79.8, 87.0] / 96.6 [96.5, 98.1] | 3416.5 [3413.2, 3430.4] / 3430.7 [3430.1, 3440.5] |
| Long-prefill interference | v2 | 6.892 [6.890, 6.902] | 512.4 [492.1, 528.4] / 528.5 [516.1, 529.5] | 112.5 [108.7, 116.8] / 115.3 [111.6, 120.3] | 3592.6 [3588.0, 3620.5] / 3634.9 [3629.2, 3662.3] |
| Mixed | upstream | 5.892 [5.821, 6.122] | 487.2 [478.5, 600.3] / 573.1 [492.8, 653.2] | 130.8 [130.7, 140.3] / 146.0 [131.2, 154.3] | 4173.2 [4005.8, 4301.2] / 4359.1 [4115.7, 4436.5] |
| Mixed | v1 | 5.660 [5.343, 5.887] | 1415.3 [1189.9, 1436.5] / 1534.0 [1340.7, 1561.0] | 102.5 [92.0, 111.1] / 117.0 [93.4, 127.3] | 4126.1 [3910.3, 4170.4] / 4234.7 [4038.1, 4270.1] |
| Mixed | v2 | 5.760 [5.657, 5.998] | 680.7 [641.8, 841.2] / 771.5 [756.9, 906.5] | 139.7 [133.7, 148.2] / 155.0 [150.3, 158.8] | 4353.1 [4210.3, 4554.4] / 4601.2 [4312.1, 4613.6] |
| Starvation stress | upstream | 11.232 [11.167, 11.252] | 367.0 [364.4, 370.6] / 371.5 [369.3, 380.0] | 89.2 [87.5, 89.7] / 91.3 [89.5, 92.7] | 2163.6 [2135.7, 2184.9] / 2562.9 [2559.8, 2591.2] |
| Starvation stress | v1 | 10.420 [10.370, 10.424] | 755.6 [754.4, 770.3] / 1067.9 [1013.4, 1070.3] | 68.3 [66.6, 69.1] / 68.6 [68.4, 69.3] | 1921.9 [1917.9, 1958.5] / 2325.7 [2324.6, 2336.9] |
| Starvation stress | v2 | 10.941 [10.937, 10.995] | 517.8 [508.9, 519.5] / 532.9 [528.4, 560.6] | 88.7 [87.6, 89.8] / 90.7 [90.4, 91.4] | 2165.8 [2136.4, 2172.1] / 2644.9 [2620.8, 2646.1] |

Relative to upstream, v2 completed RPS changed by -0.9%, -2.2%, and -2.6%
for long-prefill, mixed, and starvation stress. TPOT p95 changed by -0.1%,
+6.8%, and -0.5%, where a negative latency change is an improvement. E2E p95
regressed by 0.4%, 4.3%, and 0.1%. Thus throughput met the 5% guardrail, while
the aggregate TTFT and 15% TPOT targets did not.

## Per-Class Fairness

Classes and the long-prefill short-request metric were declared before
HOLDOUT. Values are three-repeat medians.

| Workload | Policy | Short TTFT p50/p95/p99 ms | Medium TTFT p50/p95/p99 ms | Long TTFT p50/p95/p99 ms |
| --- | --- | ---: | ---: | ---: |
| Long-prefill interference | upstream | 193.0 / 346.6 / 376.9 | n/a | 354.3 / 366.8 / 368.2 |
| Long-prefill interference | v1 | 220.4 / 326.2 / 349.8 | n/a | 892.1 / 1010.3 / 1025.6 |
| Long-prefill interference | v2 | 175.4 / 228.1 / 276.7 | n/a | 491.5 / 528.6 / 528.8 |
| Mixed | upstream | 236.7 / 428.6 / 479.3 | 287.0 / 401.3 / 421.0 | 349.9 / 550.0 / 595.0 |
| Mixed | v1 | 300.3 / 775.0 / 909.2 | 595.6 / 880.3 / 926.7 | 1144.2 / 1513.7 / 1559.4 |
| Mixed | v2 | 218.8 / 352.3 / 353.0 | 398.7 / 552.7 / 565.1 | 608.8 / 746.8 / 802.4 |
| Starvation stress | upstream | 111.4 / 349.7 / 371.4 | n/a | 361.0 / 367.7 / 367.8 |
| Starvation stress | v1 | 237.1 / 457.1 / 568.7 | n/a | 760.7 / 1186.7 / 1246.1 |
| Starvation stress | v2 | 163.5 / 262.3 / 300.6 | n/a | 522.5 / 534.9 / 535.9 |

The corresponding enqueue-to-first-schedule waiting tails are:

| Workload | Policy | Short wait p95/p99 ms | Medium wait p95/p99 ms | Long wait p95/p99 ms |
| --- | --- | ---: | ---: | ---: |
| Long-prefill interference | upstream | 0.8 / 1.0 | n/a | 0.5 / 0.5 |
| Long-prefill interference | v1 | 96.0 / 99.6 | n/a | 18.3 / 22.9 |
| Long-prefill interference | v2 | 82.6 / 82.9 | n/a | 0.6 / 0.6 |
| Mixed | upstream | 0.7 / 0.7 | 1.0 / 1.1 | 0.9 / 1.0 |
| Mixed | v1 | 460.5 / 577.2 | 173.1 / 182.9 | 365.5 / 403.7 |
| Mixed | v2 | 92.6 / 93.7 | 93.0 / 93.8 | 90.6 / 91.2 |
| Starvation stress | upstream | 1.1 / 1.1 | n/a | 0.6 / 0.6 |
| Starvation stress | v1 | 276.0 / 374.0 | n/a | 342.3 / 383.5 |
| Starvation stress | v2 | 92.9 / 94.8 | n/a | 0.7 / 0.7 |

Under long-prefill interference, the predeclared short-request TTFT p95 fell
from 346.6 ms upstream and 326.2 ms in v1 to 228.1 ms in v2: improvements of
34.2% and 30.1%, respectively. Across all workloads, v2 improved short TTFT
p95 relative to upstream by 17.8% to 34.2%. Long TTFT p95 nevertheless
regressed by 35.8% to 45.5%, which explains why the aggregate TTFT target was
not met.

## Waiting, Starvation, And Scheduler Cost

Starvation is a request-level enqueue-to-first-schedule wait of at least the
pre-frozen 400 ms. Values are median [bootstrap 95% CI].

| Workload | Policy | Max wait ms | Starvation count | Decision p95 us | Token-budget use |
| --- | --- | ---: | ---: | ---: | ---: |
| Long-prefill interference | upstream | 1.1 [0.8, 1.3] | 0 [0, 0] | 380.7 [360.6, 467.0] | 3.9% |
| Long-prefill interference | v1 | 100.7 [100.6, 125.4] | 0 [0, 0] | 494.9 [444.1, 521.2] | 7.1% |
| Long-prefill interference | v2 | 83.0 [82.9, 83.4] | 0 [0, 0] | 380.6 [376.2, 430.5] | 7.0% |
| Mixed | upstream | 1.1 [1.1, 1.2] | 0 [0, 0] | 393.5 [379.6, 466.3] | 5.8% |
| Mixed | v1 | 606.3 [347.0, 610.3] | 2 [0, 3] | 631.4 [561.9, 735.0] | 10.4% |
| Mixed | v2 | 94.0 [92.4, 159.6] | 0 [0, 0] | 493.4 [467.6, 497.8] | 10.5% |
| Starvation stress | upstream | 1.1 [1.1, 1.3] | 0 [0, 0] | 388.1 [387.2, 524.4] | 3.2% |
| Starvation stress | v1 | 393.8 [367.3, 394.1] | 0 [0, 0] | 554.7 [539.6, 574.0] | 5.7% |
| Starvation stress | v2 | 95.1 [95.0, 95.1] | 0 [0, 0] | 498.2 [485.6, 505.0] | 5.8% |

The mixed v1 repeats contained 3, 2, and 0 starvation events; v2 contained
0, 0, and 0. This is a total reduction from 5 to 0 and a repeat-median
reduction from 2 to 0. Upstream had zero starvation, so the valid claim is
that v2 removed starvation introduced by v1, not that it improved upstream.
V2 reduced median maximum waiting versus v1 by 17.5%, 84.5%, and 75.9% for
long-prefill, mixed, and starvation stress.

V2 decision p95 was 380.6 to 498.2 us. It was 10.2% to 23.1% lower than v1,
but up to 28.4% above upstream. These measurements include snapshotting,
selection, manager scheduling, validation, and telemetry; they are not pure
policy-call microbenchmarks.

## Goodput And SLO Attainment

| Workload | Upstream goodput / attainment | V1 goodput / attainment | V2 goodput / attainment |
| --- | ---: | ---: | ---: |
| Long-prefill interference | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |
| Mixed | 0.000 / 0.0% | 0.000 / 0.0% | 0.192 / 3.3% |
| Starvation stress | 1.966 / 17.5% | 0.261 / 2.5% | 2.188 / 20.0% |

The strict joint SLO makes goodput sparse in mixed and zero in long-prefill.
The starvation-stress gain is reproducible across three v2 repeats, but it
does not justify a general goodput claim across workloads.

## Correctness And Cancellation

GPU correctness compared all three policies at temperature 0 on three fixed
prompts. Each produced the same 14-token compared sequence for every prompt;
the tracked aggregate exports neither token IDs nor token-sequence hashes.

The 50-request v2 cancellation compatibility smoke completed 35 requests,
cancelled 12, and disconnected 3. It covered waiting, prefill, decode, and
disconnect outcomes. Scheduler failed delta, orphan requests, final waiting,
and final running were all zero, and readiness remained true.

## Goal Assessment

| Predeclared criterion | Result |
| --- | --- |
| At least one aggregate workload TTFT p95 improves by 10% vs upstream | **Not met**; all three regress 39.3% to 41.1% |
| V1-induced request-level starvation reaches zero | **Met**; all nine v2 runs have zero, mixed total 5 to 0 |
| Preserve at least 15% TPOT p95 improvement | **Not met**; changes vs upstream are -0.1%, +6.8%, -0.5% latency |
| Completed RPS regression no more than 5% | **Met**; regression is 0.9% to 2.6% |
| Token and lifecycle correctness | **Met** |

## Claims And Resume Boundary

Safe HOLDOUT-based claims:

- designed a phase-aware Dual-SLO scheduler with hard max-wait FIFO, minimum
  prefill service, urgent-over-continuation ordering, and bounded dynamic
  decode reservation;
- validated 27/27 randomized matched-trace Qwen3-8B GPU runs and 900/900
  successful requests with exact fixed-seed output equivalence;
- reduced predeclared short-request TTFT p95 by 17.8% to 34.2% versus upstream,
  including 34.2% under long-prefill interference;
- removed five v1-induced mixed-workload starvation events and reduced v1
  maximum waiting by up to 84.5%, while keeping completed-RPS loss within 2.6%.

Not safe to claim:

- an aggregate TTFT, TPOT, E2E, or universal goodput improvement over upstream;
- that upstream starved requests;
- production-optimal SLO parameters or statistical significance beyond three
  controlled repeats;
- same-batch mixed prefill/decode, preemptive scheduling, or isolated policy
  overhead.

The tracked, de-identified aggregate is
`benchmark/examples/deadline_scheduler_v2_holdout_summary.json`. Manifests,
per-request data, step traces, logs, correctness hashes, and invalid archived
attempts remain outside Git under the runtime directory.

## Safety Closure

Only `llm-servelab-engine-6br2` was used. It was stopped with
`docker stop -t 60` after label verification and finished in `exited` state.
GPU 9 returned to 0 MiB. No other container, image, volume, network, or Docker
cache was modified or deleted.
