# Request Lifecycle

## Scope

Prompt 6B-0 adds an explicit request lifecycle around the pinned
Mini-SGLang scheduler. The state machine is scheduler-owned and records only
low-cardinality state and aggregate timing. It never records prompt text or
token values.

## States

```text
CREATED
  -> WAITING
  -> PREFILL_SELECTED
  -> PREFILL_RUNNING
  -> DECODING
  -> FINISHING
  -> COMPLETED

Any non-terminal state
  -> CANCELLATION_REQUESTED
  -> CANCELLED

Any non-terminal state
  -> FAILED
```

`PREFILL_SELECTED -> WAITING` and `PREFILL_RUNNING -> WAITING` are legal for
policy rollback and chunked prefill. The only terminal states are
`COMPLETED`, `CANCELLED`, and `FAILED`.

`RequestLifecycle.transition` rejects illegal transitions and any attempt to
leave a terminal state. Duplicate terminal operations are idempotent.

## Ownership And Invariants

The scheduler enforces these invariants:

1. A request has exactly one terminal state.
2. A terminal request cannot re-enter waiting or running state.
3. A request cannot be present in both `PrefillManager.pending_list` and
   `DecodeManager.running_reqs`.
4. A cancelled UID is filtered from future batches.
5. Every allocated request-table row has one owner and is released once.
6. Every request-owned KV allocation is released or transferred to radix-cache
   ownership.
7. A shared prefix handle is unlocked, not globally cleared.
8. The backend terminal callback and frontend SSE stop are each claimed once.

`TableManager` tracks allocated rows explicitly and raises on double free.
`Scheduler._released_uids` makes request resource release idempotent while
retaining the table ownership assertion.

## Timestamps

Lifecycle timestamps use `time.monotonic_ns()`:

- creation and enqueue;
- first schedule;
- prefill start and end;
- first decode;
- finish;
- cancellation.

The aggregate API exposes scheduling age, waiting time, generated-token count,
and optional deadline slack. `deadline_ms` is telemetry input only in 6B-0.
It does not change scheduling order.

## Completion

On normal completion, the scheduler:

1. removes the request from the decode manager;
2. transitions through `FINISHING` to `COMPLETED`;
3. emits one terminal detokenize message;
4. transfers valid computed prefix pages to the radix cache where applicable;
5. returns duplicate pages, unlocks the matched prefix, and frees the table row.

The frontend removes acknowledgement/event state when the terminal reply is
consumed. The streaming wrapper emits one final finish frame and one `[DONE]`.

## Cancellation Path

```text
HTTP cancel or client disconnect
  -> FrontendManager.abort_user
  -> AbortMsg
  -> tokenizer AbortBackendMsg
  -> Scheduler._process_one_msg
  -> Scheduler.abort_req
  -> waiting/selected/running lookup
  -> resource cleanup
  -> CANCELLED
```

An in-flight CUDA batch cannot be freed before its completion event. Such a
request enters `CANCELLATION_REQUESTED`, is excluded from the next batch, and
is released when `_process_last_data` observes completion. This prevents reuse
of pages or table state while a kernel may still reference them.

## Telemetry Boundary

The health channel carries aggregate waiting, running, cancelled, and failed
counts. Scheduler metrics carry aggregate batch, token, age, decision-time,
fallback, and failure counts. Neither channel uses request IDs as metric labels
or stores prompts.

Detailed runtime summaries remain outside Git under
`/lfs1/users/chyan/mini-sglang-engine-lab-runtime-6b0`.

