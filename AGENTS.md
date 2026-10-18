# Working in this repository

This is a scientific kernel experiment repository. Favor reproducible evidence over peak numbers.

1. Read README.md, docs/methodology.md, docs/precision.md and docs/h3-coverage.md before changing benchmark semantics.
2. Work from the existing checkout unless isolation is needed; use codex/ branches for subsequent changes. Preserve other agents' changes. Do not run concurrent GPU benchmarks.
3. Use `uv sync --locked`. Run `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest -q`. On a non-Metal host report GPU validation as unavailable, never as passing.
4. Every kernel needs an independent numerical reference, finite-value checks, tail shapes, zeros and outlier coverage appropriate to its contract. Report quantization error separately from implementation error. Never relax a tolerance to conceal a bug.
5. Preserve compile failures and unsupported paths as explicit results. No silent dtype/backend fallback. Distinguish storage, API execution support, native arithmetic, and measured acceleration.
6. Measure compilation-free synchronized latency with warmup and multiple repetitions. Separate offline weight preparation, online activation quantization, kernel-only time and end-to-end time. Include layout conversion and required padding when comparing deployed paths.
7. Keep raw JSON records immutable. Record code revision/dirty state, source digest, device, versions, seeds, shapes, raw timings and references. New best results require a second seed and independent rerun on identical hardware. Do not extrapolate a microbenchmark into model throughput or quality.
8. Keep source/config provenance pinned. Generated workloads must match `scripts/generate_workloads.py`. Do not redistribute model weights or upstream implementation code without its license and attribution.
9. Never enable a network-accessible self-hosted GPU runner for untrusted pull requests. GPU experiments run locally; hosted CI covers CPU tests.
10. Update docs/h3-coverage.md and docs/initial-results.md when evidence changes. `unsupported`, `not measured`, `incorrect` and `quality rejected` are different states.

A new workload only needs the JSON schema used in workloads/*.json. A new operation adds a prepare function returning callable, reference extraction, implementation and original references, timed preparation stages and metadata in src/m5kernels/engine.py. Metal kernels live under src/m5kernels/shaders/. Keep model integration separate from kernel microbenchmarks.
