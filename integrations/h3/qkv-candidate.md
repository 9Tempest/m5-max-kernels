# Native grouped QKV candidate

This opt-in experiment connects our TM64/TN128/four-SIMD/swizzle-8 Q/K tile to
H3's actual grouped QKV API. The baseline remains the pinned native grouped INT8
QKV/RMS/RoPE kernel with its normal full-K, local-scale and epilogue choices.
`H3_M5_QKV_ROPE=0` remains the default. C compilation, offline Metal compilation
and CPU reference tests do not establish GPU correctness or acceleration.

This is the same tile family as the earlier grouped-QKV experiment that was
slower on large native shapes. Its previous negative result remains relevant;
connecting the native API or adding H256 operands is not itself evidence of an
improvement. GPU campaigns and full-model checks must determine adoption.

## Source and native interface

`qkv_candidate.metal` is our MIT-licensed original port from
`src/m5kernels/shaders/qk_rope_projection.metal` and `qk_rope_body.metal`.
`qkv-candidate-overlay.patch` applies to the frozen fusion source; the combined
`qkv-candidate.patch` starts at upstream
`8974cc055ea9c02fcd14cc27dfda3e1027c05153`. Upstream attribution and license are
in [UPSTREAM-LICENSE](UPSTREAM-LICENSE). The experiment changes four runtime
files: `h3_gpu.h`, `h3_gpu.m`, `h3_shaders.metal` and `h3_dit.c`.

The exported `h3_gpu_m5_grouped_qkv_linear_rope_int8` API preserves the native
argument order through `input_is_quantized` and omits its three slower-path
arguments. With prepacked input it accepts native or H256 operands unchanged:

- `A`: INT8 `[ceil128(M),5376]`; FP32 row scales `[ceil128(M)]`.
- `W`: INT8 `[56,3,128,5376]`; FP32 output scales `[56,3,128]`.
- Separate Q/K gamma: BF16 `[128]`; trig: BF16 `[M,48]`.
- Separate Q/K/V outputs: BF16 `[56,M,128]`, ready for native head-major attention.

Integer projection uses the same left-associated FP32 decoding and a BF16
projection boundary. Q/K then compute full-head RMS, gamma, split-half RoPE on
channels 0–95, and BF16 output. Channels 96–127 retain normalized values. V
bypasses RMS and RoPE. The candidate's SIMD RMS sum has a different reduction
order from the native kernel; Q/K bit equality is reported, not assumed.

The native API requires `M>=128`. `M=17` is explicitly unsupported; `M=129`
exercises a real row tail and padding without substituting a different logical
shape. Runtime overrides that change the native baseline are rejected. No
BF16 or backend fallback is allowed.

## Independent validation driver

`qkv_validate.c` runs against real block-0 grouped QKV weights and Q/K gamma.
The original BF16 synthetic input represents already-computed AdaLN output.
Synthetic trig values exercise the split-half mapping, not a recorded H3
position sequence. `native` and `h256` operand policies are separate cases.
Both kernels in a case receive identical codes/scales and physical layouts.

```text
qkv_validate MODEL_DIT_DIRECTORY SHADER ROWS SEED POLICY NEW_REPORT.json [WARMUPS REPEATS]
```

`POLICY` is `native` or `h256`. Defaults are three warmups and ten repetitions.
The report must not exist. Exit codes are 0 for completed, 2 for explicit
unsupported, 3 for numerical rejection, and 1 for execution failure. The driver
requires a caller-held shared GPU lock; use the campaign wrapper for real runs.

The combined runtime that includes H256 QKV/FC1 preparation is pinned by
`kernel-integration-source.json` and `kernel-integration.patch`. Its digest must
be supplied explicitly rather than inferred from an arbitrary working tree:

```sh
uv run python integrations/h3/run_qkv_campaign.py \
  --source-manifest integrations/h3/kernel-integration-source.json \
  --source-manifest-sha256 SHA256_OF_APPROVED_MANIFEST \
  --source-dir .cache/h3.c-kernel-integration \
  --model-dir .cache/models/MiniMax-H3 \
  --rows 17 129 2049 9914 17455 --seeds 17 29 \
  --policies native h256 --warmups 3 --repeats 10 \
  --output-dir results/local/h3/qkv-campaign-NEW
```

`qkv_reference.h` is independent scalar CPU code. It computes every K term and
all 128 channels for heads 0, 28 and 55 at four selected rows: zero, epsilon
dominated, ordinary, and an outlier tail row. It separately reports:

- Implementation error against FP64 decoding of the actual INT8 operands and
  their actual FP32 scales, with the required BF16 projection boundary.
- Quantization error between that decoded reference and original unrotated
  BF16 input/weight references, including H256 preparation error where used.
- Each kernel's total error against original BF16 references.

Each row-class/stream implementation metric must satisfy the existing BF16
relative-L2 limit `0.008`. This limit is not enlarged for the new reduction.
Packing checks independently reproduce row codes/scales on the four input
rows and 1,152 weight rows; H256 uses an independent scalar FP32 butterfly.
All output elements and all scales must be finite, padded input codes must be
zero with scale one, the zero row must produce zero outputs, and V must be bit
identical between kernels. Q/K full-output bit differences are retained.
This samples numerical reference rows and heads; it is not an independent
FP64 reference for every layer/head/token in the full model.

## JSON and timing contract

The driver emits `h3_native_qkv_v1`; the strict CPU checker is
`validate_qkv_report.validate(report, rows, seed, policy, warmups, repeats)`.
It returns `completed` or `unsupported`, or raises on invalid/incomplete or
numerically rejected evidence. The campaign preserves failed raw records.
Source, patch, compiler, device, model inventory, executable and exact harness
snapshots are recorded by the campaign wrapper rather than invented by C.

Completed reports contain `shape`, `operand_policy`, source-derived baseline
selection, `packing`, full-output `paired`, sampled independent `reference`,
and raw `timings`. Each reference row contains three stream records with
`native_implementation`, `candidate_implementation`, `native_total_error`,
`candidate_total_error` and `quantization_error`. Nonfinite theoretical metrics
are explicit `null`; a completed implementation result cannot contain them.

Two stages are measured with warmup and alternating paired order:

| Stage | Included online work | Dispatches |
| --- | --- | ---: |
| `prepacked_qkv` | Grouped projection, Q/K RMS/RoPE, all three head-major outputs | 1 |
| `online_activation_plus_qkv` | Policy activation packing, required padding, then the same QKV | 2 |

The H256 online stage includes H256 and INT8 packing but starts from BF16
AdaLN output; it does not include AdaLN itself. Offline weight quantization is
reported separately and excluded from both stages. There are no extra output
layout conversions in either path. Correctness/readback/reference work is
outside warmed intervals.

Each raw sample has synchronized host `wall_seconds`, `cpu_encode_seconds`,
`submit_wait_seconds`, supplemental Metal `supplemental_gpu_seconds`, and
`direct_dispatches`. CPU encoding is not GPU operation attribution. Host wall
latency is the primary paired metric. Repetition order reverses within pairs
and begins with the candidate for seed 29. An independent reverse campaign is
required before any new-best claim. Microbenchmarks do not imply model
throughput or quality; full-model latent/media comparisons and resident
measurements are separate evidence.
