# Native projection gate epilogues

The experiment adds two opt-in native dispatches to pinned `antirez/h3.c`
`8974cc055ea9c02fcd14cc27dfda3e1027c05153`, on top of the frozen H256 FC2
experiment. Both flags default to off. Two independent twelve-case native GPU
campaigns pass their implementation and paired-output contracts. H256 FC2
shows no consistent timing benefit across seeds. Attention-output at M9914
has lower medians in both seeds and both campaigns; M2049 includes a reversal.
Full-model parity and end-to-end acceleration remain **pending**.

| Flag | Changed projection | Existing downstream operation |
|---|---|---|
| `H3_FUSE_FC2_GATE=1` | FC2, K14336 → N5376, gate slot 5 | Next block's AdaLN slots 0/1 and INT8 packing |
| `H3_FUSE_ATTN_OUT_GATE=1` | Attention output, K7168 → N5376, gate slot 2 | Current block's MLP AdaLN slots 3/4 and INT8 packing |

The native backend already fuses gate, AdaLN and INT8 packing. These changes
move the gate into the preceding GEMM epilogue and use an otherwise matching
AdaLN/packing kernel on the updated hidden tensor. They do not add a previously
missing gate/AdaLN fusion. Native QKV/RMS/RoPE and attention head-major gather/
quantization stay in their existing dispatches. For rows above 2048, the
attention output baseline and candidate both use the dynamic-K 128×128 kernel;
the native small-row K7168 specialization is outside this experiment.

The numerical boundary is:

```
projection = BF16(INT8 GEMM with original scale arithmetic)
hidden_next = BF16(FP32(hidden) + FP32(projection) * FP32(gate))
next_input = INT8(BF16(AdaLN(hidden_next)))
```

FC2 H256 rotation, weight codes/scales and activation packing are unchanged.
The group1024 policy remains available as a separate matched check. The gate
uses the existing per-token modality/timestep row map. FC2 uses the current
block's gate table and the next block's normalization table. Attention output
uses separate slots in the current block's table. Each output element reads
its own residual before writing the same location; the GEMM input and scratch
storage are separate from hidden. The last block skips its old standalone gate
when the FC2 epilogue has already applied it.

Most block transitions still use three dispatches: online activation packing,
projection, and next AdaLN/packing. Removing the BF16 branch write/read also
introduces a separate read of the updated residual in the next kernel. The
net logical traffic reduction is therefore one BF16 M×5376 transfer, about
106.60 MB (101.66 MiB) at M9914, per enabled transition. At a final block without
next AdaLN, FC2 also removes one standalone gate dispatch. These are source
accounting estimates, not measured memory traffic or promised speedups. The
activation allocations remain unchanged.

## Frozen source and development finding

`fused.patch` is the complete patch against the upstream pin; do not apply it
after `hadamard.patch`. `fused-overlay.patch` applies only the additional
fusion changes to the frozen H256 source. Both retain upstream MIT attribution
in [UPSTREAM-LICENSE](UPSTREAM-LICENSE). Original harness/reference files use
the repository MIT license. No weights are included.

| Artifact | SHA256 |
|---|---|
| Combined patch | `cbd7239d98dffac8feb469caa7fdeee504a564dda51f625f95406284b2a806fe` |
| Overlay patch | `9fea68ccc3c1994b7737f49855c289debe819b27186e9150176c367cd99640f4` |
| Tracked runtime source | `83ef61ace55a0ca9d8c2735d627e8c0ca96692bfebaae2376fae3b6fd9dbef7b` |
| Numerical driver source | `6105dad5dd401aea70f4db1ab153b49bb6817a315c1ff314741d2e24e22bce30` |
| CPU reference header | `18e1a880800384ec20a5279f16ef6cacdee70a2b61b037cf5d04d5886e5684d7` |

The initial CPU-only source digest
`a9c7a9046ffa9ec20c2755b66387297cf783a6a230594c56e39f82eb7c9bf6cc`
(patch `dc4376bca0ff0a063aff9eca1b22b8fab2b8ed6dfaa5df33ce346804714d3197`)
compiled but omitted four new names from the eager pipeline registry. Static
review identified that it would fail at dispatch. It was never run on GPU.
The current source registers all four names under `tensorOpsEnabled`; a CPU
source-contract test checks new shader definitions against this registry.
This finding illustrates why offline shader compilation is not execution
validation. The upstream initializer eagerly creates these pipelines before
the driver reaches numerical checks or timed samples.

The overlay passes `git apply --check` against the frozen H256 runtime. The
prepared-checkpoint loader has adjacent additions in the DiT struct, so a
later combination needs an explicit three-way merge and a new source pin.
Neither existing runtime clone is changed by this experiment.

## Reproduce numerical and online checks

Use a separate runtime checkout:

```sh
git clone https://github.com/antirez/h3.c .cache/h3.c-fused
git -C .cache/h3.c-fused checkout -b codex/h3-fused-epilogues 8974cc055ea9c02fcd14cc27dfda3e1027c05153
git -C .cache/h3.c-fused apply ../../integrations/h3/fused.patch
uv run python integrations/h3/run_fused_campaign.py \
  --source-dir .cache/h3.c-fused \
  --model-dir .cache/models/MiniMax-H3 \
  --output-dir results/local/h3/fused-campaign-001 \
  --gpu-wait-seconds 7200
```

The wrapper acquires the shared GPU lock before source/model verification,
rebuilds the runtime and a private driver, then holds the lock for the entire
campaign. It rejects inherited H3/Metal overrides, changed source/patch/driver
digests, unsupported hardware and silent TensorOps fallback. It verifies the
original checkpoint against the pinned model manifest. Each of the twelve
cases has immutable raw output, a log and a result; failures are retained and
the other cases continue.

The cases are seeds 17/29 × logical rows 2049/9914 (padded 2176/9984) × FC2
group1024/H256 and attention output. They use real block-0 weights and
deterministic synthetic BF16 activations, including an exact zero row and a
sparse-outlier last row. Three synthetic modulation rows exercise row-map
addressing, with independent FC2 next-block modulation. Correctness output
buffers are poisoned before execution. Numerical checks require:

* Every branch/hidden/scale value finite; paired hidden, INT8 codes and FP32
  scales bit-identical across the entire output, including padding.
* All hidden elements agree exactly with an independent CPU `fmaf` gate
  reference using the original stored BF16 projection.
* Three complete hidden rows (zero, middle and last) use all K values and all
  5376 output channels in independent float64 decoded-INT8 and original-BF16
  dot products. The reference then follows BF16 branch, FP32 gate/BF16 hidden,
  independent float64 RMS/AdaLN, BF16 and INT8 boundaries. Each selected row
  must pass the existing 0.008 BF16 implementation relative-L2 screen.
* Quantization drift against original BF16 operands is reported separately
  at projection, hidden and next packed-input stages. It is not a fusion
  quality pass/fail criterion because the paired precision policy is fixed.

The independent RMS uses higher precision rather than mirroring the Metal
reduction tree; implementation metrics and exact paired checks serve different
purposes. M2049's selected rows cover all three synthetic maps. M9914's middle
and last sampled reference rows both select map 1, although the full-buffer
paired and CPU gate checks cover every map. These microchecks do not exercise
every real timestep map, all 50 weights, combined full-model buffer lifetimes
or the final block's no-next-AdaLN endpoint.

Timing starts only after numerical checks pass. Each path gets three untimed
warmups and ten alternating-order repetitions, with the starting order
reversed across seeds. Each repetition restores the same residual outside
timing. The primary interval is synchronized host wall time from command
creation through submission/completion. It includes online activation
rotation/packing, GEMM, gate and next AdaLN/packing; excludes input/reference
construction, readback, model loading, offline weight preparation and prior
shader/pipeline creation. CPU encode and submit/wait intervals are separate.
Command-buffer GPU duration is supplemental and is never substituted for wall
time or interpreted as individual shader attribution. Raw samples are kept;
the reported median ratio is an observed operator-path result, not model speed.

## First native GPU campaign

[Public raw driver/case/request/campaign JSON](../../results/published/h3-native-eval/fused-campaign-001-v1/README.md)
is copied byte-for-byte after privacy review. The
[summary](../../results/published/h3-native-eval/fused-campaign-001-v1/summary.json)
retains every raw wall/encode/wait/GPU sample, immutable source checksums,
hardware/software and independent-reference metrics. No model weights,
activation arrays, executables or private logs are published.

All twelve cases completed on Apple M5 Max, macOS Darwin 27.0.0, Xcode 27.1.
Every paired hidden/code/scale mismatch count, nonfinite count, zero-row and
padding error count is zero. The independent full-K selected-row reference
passes without changing the 0.008 implementation limit. The maximum aggregate
hidden implementation relative L2 is approximately 0.0000314. Quantization
error against original BF16 remains separately recorded and is not a claim
that fusion changes model precision or quality.

| Policy | Logical / padded rows | Seed | Native wall median ms | Fused wall median ms | Native / fused |
|---|---:|---:|---:|---:|---:|
| H256 FC2 | 2049 / 2176 | 17 | 3.5140 | 3.5270 | 0.99631 |
| H256 FC2 | 2049 / 2176 | 29 | 3.5150 | 3.5250 | 0.99716 |
| H256 FC2 | 9914 / 9984 | 17 | 14.9160 | 14.9080 | 1.00054 |
| H256 FC2 | 9914 / 9984 | 29 | 15.3790 | 15.4130 | 0.99779 |
| Attention output | 2049 / 2176 | 17 | 1.9790 | 1.9390 | 1.02063 |
| Attention output | 2049 / 2176 | 29 | 2.3765 | 2.3665 | 1.00423 |
| Attention output | 9914 / 9984 | 17 | 8.3460 | 8.1570 | 1.02317 |
| Attention output | 9914 / 9984 | 29 | 8.0410 | 7.5375 | 1.06680 |
| Group1024 FC2 | 2049 / 2176 | 17 | 4.4390 | 4.3965 | 1.00967 |
| Group1024 FC2 | 2049 / 2176 | 29 | 4.6060 | 4.5030 | 1.02287 |
| Group1024 FC2 | 9914 / 9984 | 17 | 18.1370 | 18.0325 | 1.00580 |
| Group1024 FC2 | 9914 / 9984 | 29 | 17.4060 | 16.9490 | 1.02696 |

These medians summarize three warmups and ten alternating-order samples per
variant, with starting order reversed for seed 29. H256 FC2's range straddles
parity and offers no useful demonstrated gain. Timing variation is material
relative to the smaller effects; the independent rerun below tests the observed
directions. These intervals are not model throughput.

Both paths report **three timed GPU dispatches**: activation packing/rotation,
GEMM, and next AdaLN/packing. The original native path already fuses gate into
the final operation. CPU residual restoration happens before the timing and
statistics boundaries; it is not a timed GPU dispatch. Full-model checks are
still needed for all real timestep maps, combined flags, buffer lifetimes and
the final block endpoint. No default is enabled from these microbenchmarks.

The package can be regenerated into a new output directory using the CPU-only
`publish_fused_campaign.py --campaign-dir ... --output-dir ...` exporter. It
revalidates all twelve raw numerical/timing records, source identities and
driver digests before copying them; existing output directories are rejected.

## Independent second campaign and resident candidate

The [second raw campaign](../../results/published/h3-native-eval/fused-campaign-002-v1/README.md)
also passes all twelve numerical cases. The
[two-campaign comparison](../../results/published/h3-native-eval/fused-comparison-001-002-v1/README.md)
revalidates all 24 raw case/driver records and recomputes the medians from raw
samples. Source/model/wrapper/environment and the linked executable hashes
match. Library archive hashes differ and are preserved separately. Campaign
intervals do not overlap. Every case's independent numerical reference metrics
are identical across the two runs; this does not assert unrecorded tensor
hash equality between separate processes.

| Policy | Rows | Seeds with lower fused median in both campaigns | Native / fused range over both seeds and campaigns |
|---|---:|---:|---:|
| H256 FC2 | 2049 | 0 / 2 | 0.99541–1.01949 |
| H256 FC2 | 9914 | 1 / 2 | 0.99252–1.00229 |
| Attention output | 2049 | 1 / 2 | 0.99482–1.02063 |
| Attention output | 9914 | 2 / 2 | 1.00788–1.06680 |
| Group1024 FC2 | 2049 | 2 / 2 | 1.00967–1.12034 |
| Group1024 FC2 | 9914 | 2 / 2 | 1.00580–1.02875 |

These are descriptive median directions, not significance tests. H256 FC2
does not support enabling its gate epilogue: at M9914, seed17's reductions
are only about 0.05%/0.23%, while seed29 is slower in both campaigns.
Attention-output M9914 is the best next resident candidate under the fixed
H256 checkpoint policy, conditional on whole-model exact parity. Its observed
wall reductions span about 0.78–6.26%; the magnitude is not stable enough to
predict a full-generation change. M2049 seed29 reverses in the second campaign.

Group1024 FC2 has repeated positive directions, but its small-shape magnitude
changes substantially across campaigns. It is a different FC2 quantization
policy and cannot be substituted for H256 or used to justify the H256 flag.
Prioritize matched fused-source `none` versus `attention` resident requests
after all Euler/final tensors and media agree. Test `both` for complete-model
correctness, while keeping FC2 off for the primary performance comparison.
No default enablement follows from these operator measurements.

The CPU-only `compare_fused_campaigns.py` creates a new comparison directory
from two source campaign directories, rejects reused/overlapping records or
mismatched executable/environment identities, and retains all raw timings and
source hashes without modifying either original campaign.

## Full-model trajectory checks

After microchecks pass, compare individual and combined flags with the same
original H256 capture. This wrapper adds readbacks/synchronization and is a
diagnostic run, not a performance sample:

```sh
uv run python integrations/h3/run_fused_experiment.py \
  --fusion both --seed 42 \
  --reference-run results/local/h3/fc2-hadamard-capture-seed42-001 \
  --source-dir .cache/h3.c-fused \
  --model-dir .cache/models/MiniMax-H3 \
  --output-dir results/local/h3/fused-both-capture-seed42-001 \
  --gpu-wait-seconds 7200
```

Repeat with `--fusion fc2`, `attention` and `none`, using new output paths.
The wrapper checks both runtime flag markers and all 40 packed Euler tensors
plus the final unpacked video/audio VAE inputs against the immutable reference.
Any byte difference fails this exact fusion parity contract. MP4 byte parity
is also recorded; visual/audio quality assessment remains a separate task.
The resident end-to-end harness is `run_native_session.py`; use one full warmup
and three repetitions with the same frozen weights/source, then repeat with
a second seed before selecting a backend.

For the matched resident comparison, use this fused source for **both** `none`
and the enabled flags. Its eager pipeline registry also affects the per-request
audio VAE setup. Comparing the older H256 source directly with an enabled
fusion would mix that setup difference into the full-request measurement.

```sh
uv run python integrations/h3/run_native_session.py \
  --mode hadamard-fc2 --fusion none --seed 42 \
  --source-dir .cache/h3.c-fused \
  --output-dir results/local/h3/fused-none-resident-seed42-001 \
  --gpu-wait-seconds 7200
```

Use a new process and output directory for `--fusion both`, `fc2` or
`attention`. The wrapper keeps one precision/fusion policy fixed per process;
the upstream cache key omits the experimental environment flags. It saves the
exact harness sources, compiles from that snapshot, and checks every returned
seed and denoise callback interval. Warmup initialization markers establish
the enabled flags; subsequent generations must retain the DiT, decoder and
conditioning caches. The denoise interval uses native monotonic callback
timestamps, including its logging/check overhead. Native cached DiT profile
wall marks include time across requests and must not replace this interval.

CPU-only checks:

```sh
uv run pytest -q tests/test_h3_fused_reference.py tests/test_h3_fused_campaign.py tests/test_h3_fused_experiment.py
```
