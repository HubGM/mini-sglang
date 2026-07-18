# Engine Baseline And Profile

## Environment

| Field | Value |
| --- | --- |
| Date | 2026-07-18 |
| Host GPU | GPU 0, NVIDIA RTX A6000, 49,140 MiB |
| Topology | NUMA node 0; no NVLink reported |
| Container | `llm-servelab-engine-6a` |
| Image ID | `sha256:6e0b9f24bcf75cbe3ea72f22340eb9e85838220f3453a9044f419120e5ce79c6` |
| Model | Local Qwen3-8B, read-only mount |
| Python | 3.12.3 |
| PyTorch | 2.9.1+cu129 |
| `torch.version.cuda` | 12.9 |
| Driver | 535.146.02 |
| SGLang | 0.5.6.post2 |
| Triton | 3.5.1 |
| FlashInfer | 0.5.3 |
| Mini-SGLang baseline | `144024ee9cb96adf5fb6efba8898e6a61b42ad9f` |
| Engine policy | `upstream_default` |

Source and model mounts were read-only. Results, logs, and profiler traces were
written below `/lfs1/users/chyan/mini-sglang-engine-lab-runtime`. No model,
runtime log, JSONL result, or profiler trace is tracked by Git.

## Baseline Stabilization

The first long-short run caused a scheduler child failure:

`torch.AcceleratorError: CUDA error: an illegal memory access was encountered`

The API process continued answering `/v1/models`, exposing a false-readiness
condition. The run did not produce a summary and was not counted as VALID.
The container was stopped with `docker stop -t 60`; GPU memory returned to
0 MiB.

Official upstream commit `20fcd7f` later fixed a FlashInfer planning race:
FlashInfer reuses a pinned host staging buffer while the prior asynchronous H2D
copy may still be in flight. Engine Lab backports only that official event
synchronization. A 12-request long-short pilot then passed, and the corrected
30-point matrix completed without another CUDA fault.

## Correctness Gate

Before the policy refactor, three fixed prompts were run with temperature 0,
seed `20260718`, and `max_tokens=16`. After introducing
`UpstreamDefaultPolicy`, all three token-ID sequences were exactly identical:

| Check | Result |
| --- | --- |
| Requests compared | 3 |
| Token IDs per stored output | 14 |
| Exact per-request equality | 3/3 |
| Sampling mode | deterministic argmax |

The pinned offline helper excludes the finishing token from its returned token
list, which explains the stored count of 14 rather than treating it as a model
correctness mismatch.

## CPU Tests

The Engine Lab CPU suite has 15 passing tests and one strict expected failure
covering:

- policy prefill precedence and decode fallback;
- no duplicate decode selection;
- FIFO waiting admission and blocked-request behavior;
- request completion, removal, and KV page release;
- radix prefix-cache matching;
- token-budget-bounded chunked prefill;
- exception-safe context reset;
- deterministic temperature-zero sampling;
- decision latency/waiting/starvation telemetry;
- benchmark percentiles, failure filtering, and repeated-run dispersion.

The strict expected failure is the cancellation path: pinned upstream defines
`AbortMsg` in its tokenizer message module but does not export it or route it to
the scheduler. If cancellation becomes wired, the strict `xfail` will turn
into a test failure until it is replaced by a resource-release assertion.

Pinned upstream's original tests are not a usable single-GPU pytest gate:
coverage flags require an absent plugin, several files execute during import,
and `tests/kernel/test_tensor.py` hard-codes `cuda:1`. Those tests were not
silently reported as passing.

## Benchmark Method

The final result directory is:

`/lfs1/users/chyan/mini-sglang-engine-lab-runtime/results/baseline_final`

Each condition has 12 requests and three seeds/repeats. The 30 conditions are:

- short-short, long-short, short-long, mixed, and starvation;
- closed-loop concurrency 4 and open-loop Poisson at 4 RPS;
- repeats 1, 2, and 3.

All 30 summary files and all 30 request JSONL files were atomically written.
All 360 requests completed without reported client error. Results before the
FlashInfer fix and results from the first metric implementation remain in
separate audit directories and are not mixed into the final aggregate.

The final harness counts token chunks only when `finish_reason is None`.
The terminating SSE frame is excluded from output-token and TPOT metrics.

Closed-loop `queue_delay_ms` includes time waiting on the client concurrency
semaphore because all requests have offset zero. Open-loop queue delay includes
arrival backlog. Neither value is a direct measurement of Mini-SGLang's
internal waiting queue.

## Repeated Results

Values below are medians of three VALID runs. Latencies are milliseconds.

| Workload / traffic | Completed RPS | Output tok/s | TTFT p95 | TTFT p99 | TPOT p95 | E2E p95 | Client queue p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short-short / closed | 4.865 | 117.112 | 181.238 | 241.092 | 27.770 | 823.985 | 1698.699 |
| short-short / open | 2.951 | 66.152 | 153.156 | 225.479 | 26.401 | 820.769 | 20.496 |
| long-short / closed | 2.495 | 58.583 | 822.320 | 913.408 | 63.471 | 2003.226 | 3736.282 |
| long-short / open | 2.896 | 64.914 | 161.979 | 236.330 | 27.159 | 851.990 | 38.599 |
| short-long / closed | 0.730 | 148.080 | 163.162 | 240.241 | 24.174 | 5735.875 | 10527.011 |
| short-long / open | 0.722 | 142.518 | 151.260 | 225.297 | 24.211 | 5760.506 | 7450.277 |
| mixed / closed | 0.836 | 119.940 | 490.958 | 572.987 | 28.356 | 6456.782 | 9461.182 |
| mixed / open | 0.910 | 133.182 | 168.356 | 238.497 | 25.060 | 5858.707 | 5314.331 |
| starvation / closed | 3.909 | 121.165 | 302.875 | 304.199 | 26.201 | 1030.935 | 2043.307 |
| starvation / open | 2.838 | 88.228 | 158.549 | 226.404 | 26.277 | 924.443 | 102.848 |

These small runs establish lifecycle stability, not hard capacity. Closed/open
values must not be compared as if their offered-load semantics were identical.
Long-short closed also shows substantial run-to-run variation:
completed RPS median 2.495, range 2.466-2.673; TTFT p99 median 913.4 ms,
range 882.9-960.7 ms.

## Scheduler Decision Overhead

The unprofiled fixed-seed correctness run recorded 17 decisions with mean
28.5 us and max 418.9 us. The successful Nsight concurrency-1 run recorded
33 decisions with mean 17.3 us and max 411.5 us.

`torch.profiler` changes timing and must not be used as a clean overhead
measurement:

| Profile case | Decisions | Mean decision us | Max decision us | Max batch | Starvation |
| --- | ---: | ---: | ---: | ---: | ---: |
| concurrency 1 | 33 | 114.2 | 3647.6 | 1 | 0 |
| moderate 8 | 65 | 25.7 | 1413.5 | 8 | 0 |
| near-saturation 32 | 65 | 57.4 | 3425.7 | 32 | 0 |
| mixed prefill/decode | 129 | 13.6 | 1270.9 | 6 | 0 |

The interface overhead cannot yet be isolated from manager scheduling and
Python timing calls. Prompt 6B should add an A/B microbenchmark around the
policy call before claiming a scheduler-overhead regression percentage.

## Profiler Results

Nsight Systems 2025.6 generated a valid concurrency-1 report. Repeated Nsight
collection later left its report exporter stuck after target exit; bounded
cleanup stopped the project container. The remaining cases used the existing
`torch.profiler`, with no package installation.

Profiler outputs remain in:

`/lfs1/users/chyan/mini-sglang-engine-lab-runtime/profiles`

Key observations:

1. Concurrency-1's profiled generation range was about 783 ms. Its PyTorch
   summary reported 762.9 ms self CUDA time.
2. GEMM/CUTLASS kernels dominate GPU time. In the Nsight concurrency-1 kernel
   summary, the largest BF16 GEMM family used 30.4% of GPU kernel time.
3. FlashInfer paged-attention kernels were approximately 1.1% of
   concurrency-1 GPU kernel time; the workload is dominated by model GEMM, not
   KV indexing.
4. Batch 32 had about 2.152 s self CUDA time; mixed prefill/decode had about
   3.435 s. Larger batches shift GEMM kernel selection and increase total
   compute per engine step.
5. The mixed trace exposes substantial `cudaEventSynchronize` CPU time. This
   includes deliberate correctness synchronization and profiler effects, so it
   is a candidate for later decomposition, not proof that the race fix alone
   causes the full delay.
6. Pinned H2D and D2H copies are individually small in the PyTorch summaries.
   Nsight's CUDA API totals include initialization and graph capture, so their
   aggregate `cudaMemcpyAsync` share is not an inference-only copy ratio.
7. CUDA Graph launch is present in the Nsight API summary (31 launches for the
   concurrency-1 decode path). Prefill remains eager.
8. Sampling/argmax and KV store kernels are visible but are not top GPU-time
   consumers in these cases.

GPU idle-gap and kernel-launch-gap distributions require timeline analysis of
the saved traces; the summary tables alone do not support a precise idle
percentage.

## Prompt 6B Gate

| Gate | State |
| --- | --- |
| Upstream-derived baseline stable after documented fix | PASS |
| CPU Engine Lab tests | PASS, 15 tests; 1 strict expected cancellation failure |
| Fixed-seed token IDs unchanged | PASS, 3/3 |
| Three-repeat matrix | PASS, 30/30 VALID |
| No observed client task leak | PASS for completed matrix |
| Scheduler metrics available | PASS |
| Waiting/starvation metric available | PASS in policy telemetry |
| Cancellation releases KV | FAIL: cancellation is not wired to scheduler |
| Exception releases KV | NOT PROVEN |
| Scheduler crash invalidates readiness | FAIL: false-ready API observed |
| Original upstream single-GPU pytest suite | FAIL as a pytest gate |
| Container exits and GPU releases | PASS during bounded cleanups |

The lab is not ready to implement the final deadline-aware scheduler. The next
safe step is a 6A stabilization patch: wire cancellation, propagate scheduler
failure to readiness, add resource-release tests, and establish deterministic
decode ordering. Only then should Prompt 6B compare new policies.
