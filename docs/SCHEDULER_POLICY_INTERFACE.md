# Scheduler Policy Interface

## Boundary

`BaseSchedulingPolicy` chooses the next local engine batch. It does not execute
model work, allocate KV directly, route among Workers, or own HTTP behavior.

Each policy provides:

- `name` and `version`;
- `health`;
- `select(context, schedule_prefill, schedule_decode)`;
- `validate_decision(...)`;
- `on_request_cancelled(uid)`;
- `on_request_finished(uid)`.

The context contains immutable waiting/running snapshots, token and KV
budgets, current batch phase, monotonic timestamp, step ID, and an optional
decision-timing switch.

## Decision Validation

Every decision is validated before model work. Validation rejects:

- duplicate request selection;
- prefill/decode overlap;
- requests absent from the corresponding waiting/running set;
- terminal or already released requests;
- batch and decision-list mismatch;
- non-positive or inconsistent chunk sizes;
- prefill token-budget overflow;
- prefill metadata on a decode decision.

The policy can ask scheduler-owned callbacks to build a prefill or decode
batch. It cannot mutate KV or table managers directly.

## Fallback

`PolicyController` wraps an experimental policy and
`UpstreamDefaultPolicy`.

1. Run and validate the active policy.
2. On exception or invalid output, roll back manager mutations.
3. Re-run the current step with `upstream_default`.
4. Record `policy_exception` or `invalid_decision`.
5. Open the experimental-policy circuit after the configured consecutive
   failure threshold.
6. Keep the circuit open until explicit manual reset.

An `upstream_default` exception is `UpstreamPolicyFatalError`; it is not
silently retried as if the data plane were healthy.

The configured default remains `upstream_default`. Prompt 6B-0 does not add or
enable a deadline-aware policy.

## Telemetry

Per scheduler step, aggregate telemetry includes:

- step and monotonic timestamp;
- available token budget;
- selected prefill/decode requests and token totals;
- waiting/running counts and maximum waiting age;
- policy name and measured decision time;
- fallback reason;
- cumulative cancelled and failed counts.

Request lifecycle telemetry includes enqueue, schedule, prefill, decode,
finish, cancellation, age, waiting, optional deadline, and optional slack.
Prometheus-style dimensions must remain low cardinality. Request UID, prompt,
raw token ID, and block hash are not labels.

Decision timing can be disabled through configuration so policy experiments can
measure instrumentation overhead separately.

## Extension Checklist

Before a new policy can be benchmarked:

1. preserve `upstream_default` as the control;
2. add CPU tests for valid, invalid, exception, rollback, and circuit paths;
3. pass fixed-seed GPU token correctness;
4. pass cancellation and fatal-health gates;
5. run matched-trace performance repeats;
6. report fallback and decision overhead;
7. avoid changing cluster-level routing behavior.

