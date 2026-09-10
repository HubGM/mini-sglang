# Deadline Scheduler Experiment

## Environment

| Field | Value |
| --- | --- |
| Date | 2026-09-10 |
| GPU | GPU 0, NVIDIA RTX A6000, 49,140 MiB |
| Driver | 535.146.02 |
| Container | `llm-servelab-engine-6br` |
| Image | `sha256:6e0b9f24bcf75cbe3ea72f22340eb9e85838220f3453a9044f419120e5ce79c6` |
| Model | Local Qwen3-8B, read-only mount |
| Source | Engine Lab, read-only container mount |
| Runtime output | `/lfs1/users/chyan/mini-sglang-engine-lab-runtime-6br` |
| SLO | TTFT 200 ms, TPOT 50 ms, E2E 1200 ms |

The image and model were already local. No image/model was pulled or built.
Only the labeled project container was used. It was stopped with
`docker stop -t 60`; its final state is `exited`, and GPU 0 returned to 0 MiB.

## Method

The matrix contains four policies, three workloads, and three repeats for 36
core runs. Policy order is deterministically randomized within each
workload/repeat block. Policies in the same block use an identical trace;
repeat seeds are 20260910, 20260911, and 20260912. Four warm-up requests run
before metrics are reset and do not enter formal statistics.

| Workload | Trace |
| --- | --- |
| Mixed | 10 x 96/24, 10 x 512/48, 10 x 1536/32 tokens; 8 offered RPS |
| Long-prefill interference | 24 x 96/48 plus 6 periodic 2048/16; 10 offered RPS |
| Starvation stress | 36 x 64/24 plus 4 x 2048/32; 13.89 offered RPS |

Goodput requires a successful request to meet all three fixed SLOs. TTFT is
send to first SSE token, TPOT is the mean interval after the first generated
token, and E2E is send to terminal SSE event. Formal validity requires client
success, scheduler/client token-count agreement, correct policy, bounded
budget/chunk, complete telemetry, no fallback/failure/cancellation, and final
waiting/running equal to zero.

All reported center values are medians of three VALID repeats. Brackets in the
primary table are bootstrap 95% confidence intervals for the median. With only
three repeats, intervals often span the repeat range and should not be treated
as high-powered significance tests.

## Validity And Correctness

| Gate | Result |
| --- | --- |
| Core runs | 36/36 VALID |
| Requests | 1200/1200 successful |
| Generated tokens reconciled | 38,324 client tokens; exact scheduler match per run |
| Policy fallback/failure | 0/0 |
| Formal-run cancellation/failure | 0/0 |
| Final non-empty scheduler | 0/36 |
| Same trace within workload/repeat | PASS |
| Raw prompt/token IDs in aggregate | none |
| GPU token correctness | all four policies identical, 3/3 prompts each |

The correctness helper records 14 compared token IDs per prompt because the
pinned offline helper excludes its finishing token. This is consistent across
all policies and is not a semantic mismatch.

## Primary Results

| Workload / policy | Completed RPS | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | Goodput RPS | Max wait ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed / upstream | 5.936 [5.819, 6.004] | 689.7 [531.9, 778.7] | 131.3 [128.9, 134.4] | 3970.7 [3741.1, 4130.3] | 0.198 [0, 0.200] | 1.1 [0.9, 1.1] |
| Mixed / token budget | 5.797 [5.666, 5.836] | 1149.3 [951.8, 1349.3] | 122.8 [121.0, 130.2] | 4042.2 [3896.1, 4198.6] | 0 [0, 0] | 152.0 [144.6, 299.6] |
| Mixed / deadline | 5.160 [5.123, 5.267] | 1976.5 [1489.2, 2068.6] | 69.7 [64.8, 110.7] | 3448.8 [3284.7, 3596.2] | 0 [0, 0] | 1321.5 [593.7, 1500.1] |
| Mixed / deadline + aging | 5.563 [5.506, 5.766] | 1367.7 [1179.9, 1374.0] | 104.2 [98.4, 117.4] | 3891.5 [3653.6, 4108.5] | 0 [0, 0] | 594.8 [299.1, 599.9] |
| Long prefill / upstream | 6.991 [6.964, 6.993] | 347.9 [345.3, 364.6] | 113.8 [108.9, 114.0] | 3555.0 [3548.7, 3556.3] | 0 [0, 0] | 1.2 [1.0, 1.8] |
| Long prefill / token budget | 6.869 [6.866, 6.872] | 561.6 [559.8, 564.2] | 100.8 [100.8, 100.9] | 3520.0 [3516.5, 3521.1] | 0 [0, 0] | 72.9 [72.9, 75.2] |
| Long prefill / deadline | 6.568 [6.559, 6.573] | 943.7 [924.7, 951.7] | 76.3 [76.1, 81.7] | 3228.2 [3225.3, 3237.6] | 0 [0, 0] | 185.3 [156.4, 185.5] |
| Long prefill / deadline + aging | 6.680 [6.679, 6.687] | 959.3 [957.8, 959.8] | 82.8 [82.7, 82.8] | 3407.6 [3406.0, 3407.8] | 0 [0, 0] | 154.7 [154.3, 154.7] |
| Starvation / upstream | 11.258 [11.237, 11.267] | 355.9 [354.0, 363.4] | 86.4 [86.4, 87.1] | 2137.9 [2134.3, 2142.6] | 2.252 [2.247, 2.253] | 1.0 [1.0, 1.2] |
| Starvation / token budget | 10.876 [10.874, 10.890] | 597.5 [584.7, 599.5] | 87.2 [82.7, 87.3] | 2143.2 [2081.6, 2145.8] | 2.175 [1.903, 2.450] | 92.0 [91.6, 92.0] |
| Starvation / deadline | 10.873 [10.869, 10.895] | 602.6 [600.5, 607.7] | 81.3 [81.2, 81.4] | 2236.3 [2232.6, 2238.5] | 1.903 [1.902, 1.907] | 203.8 [203.8, 203.8] |
| Starvation / deadline + aging | 10.473 [10.393, 10.499] | 741.5 [732.1, 769.1] | 64.5 [64.4, 65.5] | 1844.9 [1839.7, 1903.0] | 0.524 [0.520, 0.787] | 397.8 [366.5, 398.4] |

## Tail, Throughput, And Scheduler Metrics

| Workload / policy | Input tok/s | Output tok/s | TTFT p99 ms | TPOT p99 ms | E2E p99 ms | Decision p95 us | Budget use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed / upstream | 4241.9 | 200.2 | 767.6 | 138.1 | 4254.2 | 326.2 | 5.8% |
| Mixed / token budget | 4142.8 | 195.5 | 1158.0 | 131.5 | 4333.7 | 406.2 | 10.5% |
| Mixed / deadline | 3687.8 | 174.1 | 2133.9 | 69.7 | 3648.8 | 465.4 | 8.6% |
| Mixed / deadline + aging | 3975.7 | 188.2 | 1427.8 | 114.5 | 4053.8 | 488.8 | 9.9% |
| Long prefill / upstream | 3400.3 | 284.1 | 355.7 | 115.9 | 3573.9 | 331.9 | 3.9% |
| Long prefill / token budget | 3341.0 | 279.6 | 569.5 | 101.3 | 3530.6 | 264.9 | 6.9% |
| Long prefill / deadline | 3194.8 | 266.9 | 1367.5 | 82.3 | 3237.6 | 436.6 | 6.7% |
| Long prefill / deadline + aging | 3249.1 | 271.9 | 1011.1 | 86.7 | 3427.8 | 458.5 | 7.0% |
| Starvation / upstream | 2954.0 | 269.3 | 358.0 | 88.4 | 2539.1 | 379.1 | 3.1% |
| Starvation / token budget | 2853.9 | 261.3 | 608.5 | 90.0 | 2672.6 | 401.3 | 5.8% |
| Starvation / deadline | 2853.0 | 261.5 | 626.6 | 81.3 | 2590.1 | 488.8 | 6.0% |
| Starvation / deadline + aging | 2748.0 | 252.4 | 1058.9 | 67.1 | 2305.7 | 501.4 | 5.7% |

Median GPU utilization ranged from 71.3% to 79.9%; peak utilization was 100%
for every condition. Median peak allocated GPU memory ranged from 36,950 to
37,334 MiB. The new policies did not introduce a progressive memory signal.

## Aging Versus Upstream

Using three-repeat medians and `(baseline - optimized) / baseline` for
latency, deadline + aging produced:

| Workload | TTFT p95 | TTFT p99 | TPOT p95 | E2E p95 | Completed RPS | Goodput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed | -98.3% | -86.0% | +20.7% | +2.0% | -6.3% | -100% |
| Long prefill | -175.7% | -184.2% | +27.3% | +4.1% | -4.4% | n/a (both zero) |
| Starvation stress | -108.4% | -195.8% | +25.4% | +13.7% | -7.0% | -76.7% |

The policy met a tail-latency threshold only for TPOT: all three workloads
improved TPOT p95 by more than 20%. It did not meet the mixed TTFT target,
degraded completed throughput by more than 5% in two workloads, and reduced
strict goodput where the baseline had nonzero goodput. Confidence intervals
for the large TTFT changes do not suggest a hidden TTFT win.

## Waiting And Starvation

The fixed starvation threshold was 400 ms. Long-prefill and
starvation-stress recorded zero starvation for every policy in every repeat,
and long-request completion was 100% in all 36 runs. The stress trace therefore
did not create baseline admission starvation and cannot support a claim that
aging eliminated an existing starvation problem.

Mixed recorded zero upstream and token-budget starvation, 21 total events for
deadline-aware, and 4 for deadline + aging across three repeats. Aging reduced
the deadline-only count, but its median was still 1 event versus 0 upstream.
The current slack estimator can prioritize already-running requests strongly
enough to delay first scheduling; aging mitigates but does not remove that
failure mode.

## Cancellation Compatibility

The preceding lifecycle gate remains applicable. It covered waiting,
prefill-selected, in-flight prefill, decode, disconnect, and duplicate
cancellation; two 100-request stress seeds ended with waiting=0, running=0,
orphan=0, failed=0, readiness=true, and survivor output equal to isolated
execution. New policies only select requests and receive lifecycle callbacks;
they do not own resource release. CPU tests combine deadline/chunk selection
with cancellation and verify no duplicate release.

## Claims

Safe claims from this phase:

- implemented and validated bounded token-budget/chunked-prefill scheduling,
  decode reservation, online deadline slack, aging, rollback, and telemetry;
- completed a randomized matched-trace 36-run Qwen3-8B matrix with 1200/1200
  requests successful and exact fixed-seed token equality across four policies;
- measured a repeat-median TPOT p95 reduction of 20.7% to 27.3% for deadline +
  aging, with 4.4% to 7.0% completed-throughput cost and explicit TTFT tradeoff.

Not safe to claim:

- better overall latency, goodput, or starvation than upstream;
- same-batch prefill/decode;
- hard-capacity scaling or production-optimal parameters;
- statistical significance beyond these three controlled repeats;
- isolated policy overhead from the recorded decision latency.

The tracked aggregate is
`benchmark/examples/deadline_scheduler_summary.json`. Per-request summaries,
step records, runtime logs, trace fingerprints, and correctness hashes remain
outside Git under the runtime directory.
