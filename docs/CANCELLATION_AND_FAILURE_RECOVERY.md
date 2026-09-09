# Cancellation And Failure Recovery

## Cancellation Semantics

| Case | Scheduler action | Resource action |
| --- | --- | --- |
| Before admission | Remember UID, cancel when `UserMsg` arrives | No KV or table allocation |
| Waiting | Remove `PendingReq` | No new KV allocation |
| Selected before prefill | Remove selected request | Unlock prefix and release table row |
| Prefill or decode in flight | Mark cancellation requested and exclude from next batch | Defer release until CUDA completion |
| Running decode | Remove only the target from `DecodeManager` | Release target ownership once |
| Client disconnect | Send the same abort path when configured | Same scheduler cleanup |
| Duplicate cancel | Return not accepted | No second callback or free |

Cancellation does not clear the global radix cache. A request holds a locked
handle to a shared prefix and owns its newly allocated pages and table row.
Cleanup inserts only valid computed prefix state, returns duplicate pages,
unlocks the matched handle, and releases the request row. Shared radix nodes
remain governed by normal cache reference and eviction rules.

## Exception Classes

Policy-plane errors and data-plane errors have different containment:

- An experimental policy exception or invalid decision rolls scheduler-manager
  mutations back and retries the same step through `upstream_default`.
- Repeated experimental policy failures open a circuit after the configured
  threshold. Recovery requires an explicit manual reset.
- Failure of `upstream_default` is fatal.
- Prefill/decode preparation exceptions return newly allocated pages and fail
  the affected batch with a low-cardinality reason.
- Engine-forward and sampler exceptions fail the affected batch, emit fatal
  health, and are re-raised. The process supervisor then rejects new work.
- Tokenizer or scheduler child exit marks readiness false and records a fatal
  reason. In-flight frontend waiters receive a terminal failure instead of
  waiting indefinitely.

The implementation does not automatically restart a failed scheduler. A
restart could hide corrupted device state and invalidate benchmark results.

## Health Contract

API liveness is not Worker readiness. `/health` and `/ready` return HTTP 200
only when all of these conditions hold:

- API process state is available;
- tokenizer and scheduler children are alive;
- tokenizer and scheduler readiness events were received;
- the model is loaded;
- the scheduler event-loop heartbeat is within its timeout;
- no fatal error is latched.

The response also reports aggregate waiting/running/cancelled/failed counts,
the last scheduler step, and the last successful engine-step timestamp. A
fatal state is irreversible without process restart.

## Prompt 6B-0 Validation

The CPU suite covers lifecycle transitions, duplicate terminal handling,
waiting/selected/running cancellation, deferred in-flight cleanup, shared
prefix preservation, table double-free detection, prepare/forward/sampler
failure, policy fallback/validation/circuit breaking, heartbeat timeout, mock
child exit, and prompt-free telemetry.

GPU validation used one RTX A6000, the same image ID and local Qwen3-8B as
Prompt 6A, `upstream_default`, and read-only source/model mounts.

| Gate | Result |
| --- | --- |
| Fixed-seed token sequence hashes | 3/3 identical to `b5a699e` |
| Immediate waiting cancellation | PASS |
| Cancellation issued during prefill/TTFT | PASS |
| Decode cancellation after first event | PASS |
| Streaming disconnect cancellation | PASS |
| Duplicate cancellation | PASS, second request rejected |
| Survivor output with cancelled peer | Identical SHA-256 |
| Stress | 2 repeats, 200 matrix requests |
| Stress terminal outcomes | 200/200 |
| Scheduler cancellation accounting | 58/58 across complete gates |
| Final waiting/running | 0/0 |
| Failed/orphan requests | 0/0 |
| CUDA, table ownership, or integrity errors | 0 observed |
| Scheduler readiness after stress | ready=true |

The matrix uses 25 requests at each 0%, 10%, 30%, and 50% planned
cancellation ratio per repeat. The two independent seeds produced the same
outcome counts. Cancellation cleanup p95 was 8.6 ms and 12.7 ms. GPU memory
rose once from 36,798 MiB to 37,174 MiB when larger batches first exercised
runtime workspaces, then remained exactly 37,174 MiB after the second repeat.
This supports a no-progressive-leak result; it is not a byte-level KV allocator
proof.

Raw request records, prompt text, request IDs, and token IDs were not stored.

## Remaining Limits

- GPU prefill kernels are not preempted mid-kernel. A cancel issued during
  prefill is applied at the next scheduler message boundary.
- The pinned decode manager still uses a Python set, so FIFO decode ordering is
  not a formal contract.
- Real CUDA fault recovery is intentionally not tested. Fatal GPU state should
  invalidate the Worker and require a controlled restart.
- Deadline-aware scheduling is not part of 6B-0.

