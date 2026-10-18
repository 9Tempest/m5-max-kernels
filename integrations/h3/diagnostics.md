# Optional h3.c diagnostics

`diagnostics.patch` is an opt-in diagnostic addition to
[antirez/h3.c at 8974cc055ea9c02fcd14cc27dfda3e1027c05153](https://github.com/antirez/h3.c/tree/8974cc055ea9c02fcd14cc27dfda3e1027c05153).
It is not part of the unmodified integration baseline. It adds no GPU command
submission, synchronization, kernel, or model arithmetic. It does add host
clock reads, logging, and a few latent-buffer reads/scans; all timings from this
build must be labelled diagnostic and kept separate from baseline latency.

Upstream h3.c is Copyright (c) 2026 Salvatore Sanfilippo, under the
[MIT license](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/LICENSE).
The verbatim upstream license is retained in [UPSTREAM-LICENSE](UPSTREAM-LICENSE).
This repository distributes only the small modification patch, under its own
MIT license, and does not include the upstream implementation or weights.

## Apply after collecting the unmodified baseline

The following checks and application are run from this repository's root.
Use a separate checkout so the baseline executable and evidence stay available.
The diagnostic runner accepts only the pinned revision with this exact patch;
any other tracked change or untracked source file is rejected. It records the
patch, full tracked-source, changed-source, and rebuilt executable digests.

```sh
git clone --no-hardlinks .cache/h3.c .cache/h3.c-diag
git -C .cache/h3.c-diag checkout --detach 8974cc055ea9c02fcd14cc27dfda3e1027c05153
git -C .cache/h3.c-diag apply --check ../../integrations/h3/diagnostics.patch
git -C .cache/h3.c-diag apply ../../integrations/h3/diagnostics.patch
make -C .cache/h3.c-diag -j8
uv run python integrations/h3/run.py --mode bf16 --diagnostics \
  --source-dir .cache/h3.c-diag \
  --output-dir results/local/h3/bf16-seed42-diagnostics-001
```

Use the same request arguments as the baseline and a new evidence directory.
The explicit `--diagnostics` flag sets `H3_DIAGNOSTICS=1` after rejecting
inherited `H3_*` settings, and holds the repository GPU lock. Do not set the
variable externally or run alongside another GPU workload. The ordinary
runner continues to require a clean, unmodified upstream checkout.

Pinned SHA-256 values are patch
`42f3c0ea22782aa71afebe3acfa399435a6d08e0091dcaddd197290036962e7f`,
patched `h3_dit.c`
`552a071ccfcbd4923599368f67154012f47bef730431f51952e14c65b9861953`,
and the full tracked-source digest using `run.py:git_provenance`
`63dd83536296ceacf6240bc168dcb693dcd9f1201e152cc4b4cd9d61c69bd283`.

The patch was checked for application to the pinned unmodified source and
compiled with `clang -fsyntax-only` and the upstream warning flags. No model
or GPU validation is implied by these checks. A temporary CPU-only helper
check with a stubbed buffer reader verified finite/nonfinite/zero counts,
histogram boundaries, RMS and explicit read failure. Before interpreting diagnostic
results, run a matched-seed comparison against the unmodified executable and
check output hashes or decoded output equivalence. If that comparison differs,
retain both results and investigate before using this build for analysis.

## Output contract

Each record is one JSON object after the literal prefix `h3 diag: `. Preserve
the complete raw stdout/stderr first; parse only lines with that prefix. The
runner retains parsed records in `diagnostic_records`, requires matching
geometry and at least one latent sample, and rejects malformed JSON or nonfinite
JSON numbers without coercion. Explicit nonfinite latent counts mark the run
`incorrect`; failed readbacks or pending work mark it `failed`. Successful
finite checks cover sampled post-Euler latents only, not all-layer parity.

| Record | Meaning |
| --- | --- |
| `geometry` | Actual packed rows, modality rows, latent dimensions, model widths, active layers, schedule length, selected load-time INT8 availability, final-head precision, RoPE scale and approximation controls. |
| `operation` | Actual branch label, step, block, rows, one high-level API call, host API time, and changes in direct/MPS-linear/MPS-SDPA dispatch counters. |
| `latent` | Full target-latent finite count, nonfinite count, zeros, finite minimum/maximum/RMS and absolute-value histogram, after an Euler update at an already completed command-buffer window. |

Indices are zero-based. Operations sample steps `0`, `steps / 2`, `steps - 1`
and blocks `0`, `24`, `49`; missing blocks or steps remain missing if an
approximation skips them. These are representative calls, not full-run totals.
INT8 configuration flags establish what was selected at load; operation labels
and dispatch deltas describe branches actually exercised, including environment
overrides. The low-level shader name and Metal instruction mix are not captured.
For example, a BF16 operation with an MPS dispatch is distinct from a direct
Metal dispatch, but the direct counter alone does not prove TensorOps usage.

`host_api_ms` measures the CPU call that validates and encodes an operation.
It can include first-use graph construction/compilation and allocation; it is
**not synchronized operation latency or GPU execution time**. It cannot rank
GPU arithmetic bottlenecks. Fused operations remain fused groups: INT8 MLP
contains activation quantization, FC1/SwiGLU, FC2 quantization and FC2; QKV
contains projection, normalization and RoPE as selected upstream. Counter
`calls` counts these API groups, whereas dispatch counters count encoded work.

No per-operation GPU duration is emitted. Upstream `gpu_seconds` is updated
only at submit/wait, and MPSGraph may schedule child command buffers, so
subtracting it around an asynchronous call would misattribute work. Keep the
existing whole-phase wall/encode/wait/root-GPU statistics. A separately
captured Metal System Trace can establish scheduling intervals, but this
runtime does not label Metal encoders with its C operation names. Existing
trace exports may contain only generic command-buffer/compute-command IDs.
Per-operation device-time attribution needs GPU Capture/replay and verified
mapping of shader or fused MPS graph groups to operations, or separate
instrumentation. Do not turn CPU enqueue times into GPU percentages.

The default M5 GPU sampler completes one evaluation per window. Latent samples
occur only when the existing loop actually completes a window at the selected
step. With longer/unbounded windows, early/middle samples can be absent. This
patch does not force a window boundary. CPU-sampler executions do not produce
these intermediate latent records. Each successful sample scans all target
elements; finite statistics describe finite entries only and are undefined
when `finite` is zero, even though their numeric placeholders are zero.
`read_failed` is explicit and must not be treated as a successful finite check.
Histogram counts use the matching inclusive upper bounds in
`abs_upper_inclusive`; the first bin contains zero, subsequent bins exclude
the previous upper bound, and the final `null` bound means greater than 32.
The histogram includes only finite values, and its counts must sum to `finite`.

These are **post-Euler latents**, not layer-wise Q/K/V or FFN activations.
They can detect trajectory explosions and compare scale evolution, but cannot
calibrate FC2 quantization or prove an implementation error bound. Per-layer
activation capture needs a separate design with buffer lifetime protection and
an explicit readback/overhead contract. Do not read aliased scratch tensors
while GPU work is pending.

## Precision and geometry checks

The default M5 runtime uses INT8 for QKV, MLP and attention-output projections.
Use all three `--use-slower-bf16-*` flags for the h3.c BF16 comparison. Its final
heads are also converted to BF16 unless `H3_DIT_F32_FINAL=1` is set; record that
distinction from the official checkpoint's mixed precision. Changing this
setting defines a separate baseline, not a correction to previous records.

For a 5-second request the native temporal shape is 124 frames at 24 fps,
37 video latent timesteps and 207 audio timesteps per stereo channel. Thus a
512-square text-only request has 9,472 video rows + 414 audio rows + prompt
rows. At 256 square it has 2,368 video rows + 414 audio rows + prompt rows.
The emitted geometry is the authority for each run, including conditioning.
INT8 kernels pad their row workspace to a multiple of 128 where selected;
the operation record's `rows` is the logical row count, not the padded workspace.

## Audit of upstream `tests/bench_dit.c`

The pinned benchmark is useful for reciprocal, warm, real-weight **whole DiT
forward** comparisons. Several optional A/B modes warm both paths, alternate
variants, check output bytes or report velocity relative-L2, and print raw
per-forward seconds. Examples include command-buffer splitting, graph-data
caching, INT8 projections, MLP and fusion variants. They are not isolated
per-operation GPU benchmarks, even when one operation is changed.

Its default shape is fixed at 512x512, 22 frames, 7 video latent timesteps,
37 audio timesteps per channel and six text rows: 1,872 packed rows. The
`h3_dit_bench_864` target changes spatial dimensions to 864x480, yielding 2,915
rows. Neither represents a 5-second request. The default prompt fixture is
`misc/fixtures/h3_real_prompt_bf16.safetensors`, exactly six BF16 rows of width
5,120 named `x.output`; it is not generated by the integration runner.

Default inputs are deterministic synthetic ramps, not live generation
activations. The default seven-forward path uses steps 0/3/6/9/12/15/18,
updates latents by `velocity * 0.001`, and does not exclude the first forward
as warmup. It is not the real Euler trajectory. Some A/B modes use a serving
schedule and a full denoise, so preserve their environment and exact mode.
`H3_BENCH_ALL_BF16=1` selects BF16 projection flags; final-head precision still
needs separate tracking. Output-byte equality tests and reported drift do not
constitute an independent float64 numerical reference.

To use it for this project, first supply a licensed, provenance-recorded
fixture and separately record the actual benchmark geometry. A new 5-second
per-op benchmark would need captured real tensors, preserved fused-layout
contracts, explicit compile-free warmup, synchronized repeated samples and
independent implementation/quantization references. No upstream benchmark
numbers are promoted into this repository's measured results by this patch.
