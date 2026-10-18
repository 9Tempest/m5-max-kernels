# Contributing experiments

Open an issue describing the workload, shape range, precision contract and expected deployment path. For a kernel PR include:

- Baseline and candidate on identical inputs, output dtype and timing boundary.
- Correctness against an independent reference, including edge tiles and representative outliers.
- Raw JSON with provenance, multiple timing samples, quantization costs and errors.
- A repeated measurement after tuning with a fresh seed; disclose regressions and unsupported shapes.
- A clear coverage change and links to upstream config/paper revisions.

Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q`. Regenerate manifests with `uv run python scripts/generate_workloads.py` and review the diff. Hosted CI does not verify Apple GPU execution. Attach local GPU evidence for shader changes.

Do not call a kernel SOTA based on an unmatched paper number. Match device core count, precision, shape, layout, accuracy target, inclusion of quantization, timing source and software versions first. A synthetic operator error threshold is not a model-quality certification.
