# Engine Lab Origin

## Repository Identity

| Field | Value |
| --- | --- |
| Upstream repository | `https://github.com/sgl-project/mini-sglang.git` |
| Pinned upstream baseline | `144024ee9cb96adf5fb6efba8898e6a61b42ad9f` |
| Upstream HEAD at lab creation | `9a91cfafe754aa85daee49998176275667eb58f2` |
| Fork repository | `git@github.com:HubGM/mini-sglang.git` |
| Development branch | `feature/deadline-token-scheduler` |
| License | MIT |

The pinned baseline matches the previously validated local Mini-SGLang
environment. The official upstream had advanced by the time this lab was
created, so upstream HEAD is recorded separately and is not silently mixed
into the baseline.

The pinned commit declares the MIT license in `pyproject.toml`, but predates
the standalone `LICENSE` file now present on `upstream/main`. This lab carries
that official license file verbatim. Copyright and license notices must remain
intact in derived work.

## Provenance

Mini-SGLang's engine, scheduler, model implementations, kernels, cache
managers, API server, and distributed runtime come from the upstream
`sgl-project/mini-sglang` project. This lab does not claim those components as
original work.

Lab-owned changes are intentionally narrow:

- architecture and experiment documentation;
- a minimal backport of official upstream FlashInfer race fix `20fcd7f`;
- a pluggable scheduler-policy boundary;
- an upstream-equivalent default policy;
- scheduler decision telemetry and CPU correctness tests;
- reproducible correctness, benchmark, and profiler harnesses.

The appropriate project description is:

> Based on Mini-SGLang, modified the token scheduler and implemented
> experimental scheduling policies with correctness and performance gates.

It is inaccurate to claim:

> Implemented a complete LLM inference engine from scratch.
