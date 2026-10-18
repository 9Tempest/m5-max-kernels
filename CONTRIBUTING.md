# Contributing experiments

Open an issue describing the workload, shape range, precision contract and expected deployment path. For a kernel PR include:

- Baseline and candidate on identical inputs, output dtype and timing boundary.
- Correctness against an independent reference, including edge tiles and representative outliers.
- Raw JSON with provenance, multiple timing samples, quantization costs and errors.
- A repeated measurement after tuning with a fresh seed; disclose regressions and unsupported shapes.
- A clear coverage change and links to upstream config/paper revisions.

Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q`. Regenerate manifests with `uv run python scripts/generate_workloads.py` and review the diff. Hosted CI does not verify Apple GPU execution. Attach local GPU evidence for shader changes.

Do not call a kernel SOTA based on an unmatched paper number. Match device core count, precision, shape, layout, accuracy target, inclusion of quantization, timing source and software versions first. A synthetic operator error threshold is not a model-quality certification.

For a recorded bottleneck hypothesis include a Metal capture summary or Instruments summary using [docs/profiling.md](docs/profiling.md). Confirm the expected shader dispatches before drawing conclusions. Keep raw tensor/process captures local, and compare unprofiled timings separately from replay counters.

After tuning, reproduce selected candidates with `uv run python scripts/validate_selections.py TUNE.json --output results/local/validation-NEW.json`. This alternates a fresh BF16 baseline and each selected candidate using a new seed. Repeat with `--distribution outliers` before proposing a precision change. The native MXINT8 probe can be rerun with `uv run python scripts/probe_capabilities.py results/local/mx-probe-NEW.json` when the compiler or OS changes.
