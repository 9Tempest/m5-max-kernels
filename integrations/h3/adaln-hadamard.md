# Native QKV / FC1 AdaLN + H256 preparation

This experiment rotates QKV and FC1 input channels using normalized, unsigned
H256, then uses symmetric whole-row INT8 packing. It preserves the original
BF16 checkpoint and the existing FC2-H256 checkpoint. It does not turn the
whole model into INT8 or move a Hadamard transform through RMSNorm, learned
gamma, modulation, SwiGLU, or a residual update.

The independent prep overlay is
[`adaln-hadamard-overlay.patch`](adaln-hadamard-overlay.patch), based on upstream
`8974cc055ea9c02fcd14cc27dfda3e1027c05153` plus the frozen
[`fused.patch`](fused.patch) source
`83ef61ace55a0ca9d8c2735d627e8c0ca96692bfebaae2376fae3b6fd9dbef7b`.
Its SHA256 is
`403b2d431789c64b88ce0365e3c33bcab8ec09092774fd8741ca72efe6961456`.
The complete patch against upstream is
[`adaln-hadamard.patch`](adaln-hadamard.patch), SHA256
`5e05d0f2a4c6d305cb5c71203c0c1613ad0bab8f910a4ac71d40ce8aab45d501`.
The standalone prep source digest is
`06060a65dee021eaad585b98eef6c190de2ca14526984332a07ca7b37c8c58ae`.
It changes `h3_dit.c`, `h3_gpu.h`, `h3_gpu.m`, and `h3_shaders.metal`.
The combined QKV-kernel experiment has a separate source manifest and digest;
these standalone pins must not be substituted for that combined source.

## Policy and exact boundaries

`H3_HADAMARD_QKV=0|1` and `H3_HADAMARD_FC1=0|1` are independently selectable,
default to zero, and reject other strings. `H3_HADAMARD_PREP_STAGED=0|1`
selects an independent staged implementation for numerical comparison and
also defaults to zero. Staged mode requires at least one H family. These flags
are distinct from `H3_M5_QKV_ROPE`, which selects a different QKV projection
kernel while retaining the same INT8 storage layout.

For each enabled family, original BF16 W is rotated once during model loading,
in FP32, and quantized per output row. QKV W remains
`[56 heads, 3 streams, 128 channels, 5376]`; FC1 W remains
`[14336 gate rows, 14336 up rows, 5376]`. Original files are read-only.
Prepared checkpoint v1, streaming, token reduction, clip overrides, and mixed
MLP-stage overrides are rejected for the new H policies. There is no implicit
old-cache import or fallback to unrotated weights. Existing FC2 H256 behavior
remains separately controlled.

Online preparation follows these boundaries:

1. If requested, compute the native gate/residual expression and round the
   updated hidden residual to BF16.
2. Use that BF16 residual for RMSNorm, learned BF16 gamma, and the original
   BF16 modulation vectors; preserve the native post-AdaLN BF16 rounding.
3. Apply 21 independent normalized H256 blocks to the 5376-wide BF16 row in
   FP32, with no additional BF16 rounding after rotation.
4. Quantize using the maximum absolute value of the entire rotated row,
   nearest-even rounding, and signed codes in `[-127,127]`.
   A zero row has codes zero and scale one. Padded rows use the same convention.

The 256-thread fused kernel preserves the native standalone AdaLN RMS reduction
order. The independent staged path calls the existing gate and AdaLN kernels,
then the separate H256 packer. This paired equality test is distinct from the
independent FP64 reference, whose reduction order differs. Subsequent native
Q/K norm, RoPE, FC1/SwiGLU, FC2, and residual operations retain their existing
rounding contracts.

## Native API and buffer contract

```c
int h3_gpu_adaln_h256_int8(
    h3_gpu *gpu,
    h3_gpu_tensor *gated_residual,
    h3_gpu_tensor *codes, h3_gpu_tensor *scales,
    h3_gpu_tensor *bf16_scratch,
    const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
    const h3_gpu_tensor *gamma,
    const h3_gpu_tensor *gate_modulation,
    const h3_gpu_tensor *norm_modulation,
    const h3_gpu_tensor *gate_row_map,
    const h3_gpu_tensor *norm_row_map,
    uint32_t rows, uint32_t padded_rows, uint32_t width,
    uint32_t slots, uint32_t gate_slot,
    uint32_t shift_slot, uint32_t scale_slot,
    float epsilon, int apply_gate, int staged);
```

Width is exactly 5376. Codes are row-major `[padded_rows,5376]`, scales are
FP32 `[padded_rows]`; native GEMMs use `padded_rows=ceil(rows/128)*128`.
Residual/branch/scratch are BF16 `[rows,5376]`, gamma is BF16 `[5376]`.
Each modulation table has shape `[conditions,slots,5376]`. Gate and norm maps
are independent U32 arrays of length `rows`; the caller guarantees finite
inputs and in-range map indices before encoding. The API validates dimensions,
capacity, dtype, epsilon, and the explicit boolean controls.

In-place residual update is supported. With `apply_gate=0`, output residual
must alias the input residual and is left unchanged. Scratch cannot alias
the residual or branch. Other input/output aliasing is outside the contract.
Staged mode requires scratch and performs two or three dispatches. Fused mode
performs one dispatch and normally uses null scratch; supplied scratch receives
the post-AdaLN BF16 row for diagnostic comparison. The API does not synchronize
or read back GPU buffers.

`h3_gpu_quantize_h256_int8` accepts K5376 and the existing FC2 K14336. It is
also the on-load weight rotation/packing API. Reusing the same physical A/W
layout keeps the new QKV kernel and the new precision policy independent.

## Real activation capture and independent analysis

The runner's `--capture-adaln` option is diagnostic only. It enables
`H3_ADALN_CAPTURE_DIR`; normal resident/performance runs leave that variable
unset. Captures use steps 0/10/19 and logical target blocks 0/24/49 for each
enabled family. For carried QKV input, block 24 or 49 is prepared after the
preceding block's FC2 gate, using the next block's gamma/modulation. Each case
selects 16 distinct full-K rows, including first/middle/last rows of every
actual packed modality segment. The metadata retains row IDs and segment
boundaries. This is selected-row evidence on the current H-policy trajectory,
not a claim about all model activations or an all-BF16 trajectory.

Before reading inputs, the diagnostic helper explicitly completes queued GPU
work; after preparation it completes that work again before reading outputs.
It starts a new command buffer after each readback. Capture adds waits and an
optional diagnostic scratch write, so capture timings are excluded from
resident-performance conclusions. Normal preparation has neither operation.

Each `qkv-stepNN-blockNN` or `fc1-stepNN-blockNN` stem has one JSON metadata
file and 12 binary payloads:

| Payloads | Shape / storage |
|---|---|
| `residual`, `branch`, `gate`, `shift`, `scale`, `gated`, `adaln` with `.bf16` suffix | `[16,5376]`, little-endian BF16 |
| `gamma.bf16` | `[5376]`, little-endian BF16 |
| `codes.i8` | `[16,5376]`, signed INT8 |
| `scales.f32` | `[16]`, little-endian FP32 |
| `gate-map.u32`, `norm-map.u32` | `[16]`, little-endian U32 |

Gate/shift/scale vectors are gathered using the independently captured maps.
The complete conditioning tables are not exported. Unused branch/gate/map
payloads are zero in pregated mode. All files use exclusive creation; partial
failed captures remain on disk rather than being relabeled as complete.

[`adaln_capture.py`](adaln_capture.py) provides a light `validate_capture`
function for the runner and a separate CPU analyzer. The analyzer checks the
gate and AdaLN against independent FP64 calculations with BF16 boundaries,
then independently verifies H256 packing cells/scales. It reports quantization
error separately from those implementation checks. The implementation relative
L2 limit is fixed at 0.008; exact staged/fused BF16/codes/scales equality belongs
to the native paired driver, not the FP64 reduction reference.

Optional projection analysis loads the safetensors index/header and reads only
selected original BF16 W rows with a memmap. It compares ordinary row INT8 and
H256 row INT8 on the same frozen post-AdaLN X/W with full K. QKV samples complete
Q/K/V streams of heads 0/28/55 (1152 output channels). FC1 samples 128 matching
gate/up channel pairs, including endpoints (256 output channels). These are
pre-Q/K-RMS/RoPE or pre-SwiGLU projection results. Both W quantizers are CPU
references; a separate metric combines captured GPU X codes with CPU W codes.
Neither metric certifies the actual GPU weight packing or GEMM implementation.

The analyzer binds capture hashes/cases to a completed run record, verifies the
original model identity and index SHA, and records selected tensor-byte hashes.
It inherits full shard hashes from the run's preflight inventory and explicitly
does not claim to rehash those large shards. The native loader reads shard
headers and its inventory does not necessarily include the safetensors index.
In that case, the analyzer checks the current index against the pinned model
manifest's Git-blob identity and records this as an analysis-time check; it
does not retroactively claim that the native run read or verified that index.
Run this CPU matrix-product analysis
outside GPU performance windows:

```sh
uv run python integrations/h3/adaln_capture.py RUN/adaln \
  --hadamard-qkv 1 --hadamard-fc1 1 \
  --model .cache/models/MiniMax-H3 --run-record RUN/run.json \
  --output NEW-ANALYSIS.json
```

Use `--staged 1` for a staged capture. Analysis outputs are exclusive and failed
analyses save an explicit failure record. Original captures remain unchanged.

## Validation and promotion boundary

The standalone source passes native CPU compilation and offline Metal
compilation. The CPU capture/reference tests cover corrupt flags/layouts,
missing modality endpoints, zero and outlier rows, nonfinite values, invalid
INT8 cells, header/index mismatches, and immutable failure output.
The root-controlled [`run_prep_campaign.py`](run_prep_campaign.py) tests the
native API with M17/2049/9914/17455, two seeds, independent maps, in-place
residuals, zeros/outliers, and explicit padding. Its raw GPU results and the
combined model experiments are separate evidence with their own source pins.

QKV and FC1 H policies remain opt-in. Lower preparation time than a staged H
path alone does not establish a speedup over native preparation without H.
Likewise, sampled precision improvements do not establish final video/audio
quality. Default promotion requires the combined source's implementation tests,
real-activation precision evidence, complete denoiser trajectory/media checks,
and repeated resident performance on the intended workload.
