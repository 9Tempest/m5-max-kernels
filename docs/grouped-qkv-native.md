# Grouped QKV against the native H3 path

This experiment compares complete grouped QKV projection, Q/K RMSNorm and RoPE, including the V projection and all three head-major outputs. The primary comparison uses **unrotated W8A8 with the native row-quantization policy**. It is distinct from the earlier single-Q/K experiment with Hadamard. Native and candidate must consume identical packed activation/weight bytes, FP32 scales, BF16 gamma and BF16 trigonometric tables. The repeated large-shape comparison favors the native kernel; the custom kernel remains an experiment, not a replacement dispatch policy.

## Layout and execution contract

The pinned native source is `antirez/h3.c` revision `8974cc055ea9c02fcd14cc27dfda3e1027c05153`. Its [QKV entry point](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L3885) and [fused projection kernel](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_shaders.metal#L1754) establish the following:

- Activations are `[M,K]`, with H3 K=5376. Grouped weights are `[H,3,128,K]`, with H3 H=56. The flattened output-channel index is `(head*3 + stream)*128 + channel`, where streams are Q, K and V. This is not the layout of three concatenated full projection matrices.
- Each output is a separate contiguous BF16 `[H,M,128]` tensor. All three outputs must be materialized and evaluated in both implementations, including V's layout conversion.
- Native activation allocation/packing rounds M up to a multiple of 128; outputs retain logical M. Weight rows are not padded. Native requires M≥128, K divisible by 128, heads divisible by four, head width128 and 48 rotary pairs. Mathematical references also test smaller and tail shapes; they do not imply native execution support for those shapes.
- The default fused kernel changes at M>2048 to a variant caching scales in threadgroup memory. K5376 also selects a specialized K tile. The actual selected symbols and mode bits belong in benchmark metadata; disabling these defaults would change the baseline.

The [native compiler setup](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L362) selects safe math mode. The common driver compiles both implementations with explicit Metal 4.1 and safe math, records these options and preserves the pinned native source. It reproduces the native default dispatch, rather than calling the full native runtime API; runtime scheduling and whole-model performance are outside this comparison. Upstream implementation code is not copied into this repository's original numerical reference.

## Quantization audit

The [native row quantizers](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_shaders.metal#L1320) start from actual BF16 input, widened to FP32. For a nonzero row, with QKV clipping factor1:

`peak = max(abs(x)); scale = FP32(peak / 127); inverse = FP32(127 / peak); code = clamp(RNE(FP32(x * inverse)), -127, 127)`.

The reciprocal multiplication is part of the declared policy. It should not silently be replaced by division by the already rounded stored scale. Zero-row behavior also differs from this repository's generic quantizer:

| Row kind | INT8 codes | Stored FP32 scale |
|---|---|---|
| Actual nonzero row | Symmetric nearest-even quantization | `peak/127` |
| Actual all-zero row | All zero | `1/127` |
| Added row beyond logical M | All zero | `1` |

The [host quantizer selection](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L2779) defaults to scalar loads for QKV inputs and weights; `H3_INT8_VECTOR_QUANT` selects the four-element load variant. Both use the same expressions and zero/padding policies. [Weight quantization](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L2932) uses the same row quantizer without padding and with clipping factor1. Quantization is per input token and per grouped output channel, not per head or per 32-element block.

The independent `native_int8_quantize_reference` models these FP32 expressions, returns codes/scales/decoded operands and records quantization error against the actual BF16 inputs. Input BF16 conversion error is excluded. Invalid values and finite-range overflow remain explicit errors; they are not silently replaced or saturated into a claimed valid result.

## Projection, normalization and rounding

Every stream first accumulates INT8 products into INT32, applies its FP32 activation scale and output-channel scale, then rounds to BF16. V is precisely this rounded projection in head-major layout. V is not normalized, rotated or scaled by either gamma vector.

Q and K independently compute RMS statistics over all 128 projected BF16 channels of each token/head, with epsilon `1e-5`. Their learned BF16 gamma vectors each have length128 and are shared across heads/tokens. FP32 normalization and gamma multiplication precede the split-half RoPE pairs `(j,j+48)` for j=0…47. Channels96…127 are normalized but unrotated. There is no BF16 normalization boundary before RoPE; Q/K round to BF16 only after rotation.

The [scalar and four-element BF16 converters](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_shaders.metal#L15) both implement nearest-even rounding. The vector converter does **not** truncate. The native RMS reduction uses ordered scalar FMA across128 channels; its vector-load mode retains that order. A candidate SIMD reduction changes rounding order. This can change final BF16 values near a rounding midpoint, so native-versus-candidate byte equality is diagnostic rather than a universal requirement. A final BF16 difference is not evidence of a truncating converter.

## Independent references and acceptance

[grouped_qkv_reference.py](../src/m5kernels/grouped_qkv_reference.py) computes original NumPy float64 projection dots, explicitly rounds the projection to BF16 and applies the independent [Q/K reference](../src/m5kernels/qk_rope_reference.py). Supplied operands are not implicitly rounded: original references consume decoded actual BF16 A/W, while implementation references consume decoded packed INT8 A/W. Both use the same actual BF16 gamma and trigonometric tables. Optional returned pre-round and BF16 projections allow V and GEMM errors to be examined separately.

Sampled references include complete128-channel Q, K and V streams for every sampled head, with full K. The matched comparison must report errors for **each of Q, K and V**, plus the combined outputs; one stream's failure cannot be hidden by the magnitude of another. Keep the existing BF16-output implementation relative-L2 limit `0.008` for each stream and combined output. This is a numerical acceptance limit, not a promised maximum ULP count or a reason to relax an observed failure. Full-output finiteness and native/candidate comparison complement the independent sampled references.

Report quantization error separately by comparing the independently decoded-operand result with the original-BF16 result. A shared preparation error cannot be accepted merely because both GPU kernels consume the same erroneous packed buffers: activation and weight preparation need their own reference checks. Q/K RMS can hide uniform projection-scale errors at ordinary magnitudes; tiny nonzero epsilon-dominated inputs and unnormalized V make those errors observable.

[CPU tests](../tests/test_grouped_qkv_reference.py) independently form scalar full-K dots, scalar RMS sums and 2×2 rotations. They cover normal, zero, outlier and tiny values; odd rows and K tails; grouped stream/head ordering; independent gamma vectors; V bypass behavior; projection rounding; quantizer ties and padding; BF16 payload conversion; invalid values and finite-range overflow. The helpers use no MLX or GPU calls.

Kernel-only timing starts from common prepared INT8 buffers. Complete online timing must include activation packing and required padding, all three projections, Q/K postprocessing and all three final outputs. Offline weight preparation is recorded separately. These measurements do not cover SDPA, a complete H3 block, denoiser trajectory or generated-output quality.


## Measurements and decision

Apple M5 Max, 40 GPU cores, 128 GiB; macOS27.0.1, Xcode27.1. All rows use K5376, 56 heads, D128 and N21504. Synthetic normal inputs; synchronized host wall medians in milliseconds. Each independent campaign uses 25 warmups and 40 repetitions. Seed43 reverses native/candidate execution order. The candidate is TM64/TN128, four SIMD groups, M traversal band8, dynamic K, threadgroup BF16 scratch and SIMD RMS.

| M | Native online, seeds29 /43 | Candidate online, seeds29 /43 | Candidate change |
|---:|---:|---:|---|
| 128 | 0.582 /0.673 | 0.541 /0.560 | 7.0–16.8% lower in separate blocks; see alternating cross-check below |
| 4096 | 8.173 /7.977 | 8.640 /8.576 | **5.7–7.5% slower** |
| 9914 | 19.688 /19.572 | 20.690 /20.633 | **5.1–5.4% slower** |

Online includes native BF16 activation quantization, M padding, all Q/K/V projections, Q/K RMS/gamma/RoPE and direct final head-major stores. Both paths use two dispatches in one command buffer. Input weights are already in the native grouped layout in both paths, so no extra weight reorder is needed; checkpoint loading/reordering is outside this resident contract. Offline weight quantization is measured separately, about 0.80–0.89ms. Activation quantization alone is about 0.12–0.13ms at M128,0.25–0.26ms at M4096 and 0.48–0.49ms at M9914. Standalone stage medians cannot be added to reconstruct the fused command-buffer median.

The kernel-only native/candidate medians are 7.93/8.50–8.58ms at M4096 and 19.18–19.33/20.29–20.40ms at M9914. The complete QKV experiment therefore does **not** reproduce the previous single-Q/K improvement over our own staged path. The unchanged upstream baseline already has grouped QKV/RMS/RoPE fusion. Keep native for these large H3 shapes. No full-attention, denoiser or generated-media speedup follows from these measurements.

[Seed29 raw records](../results/published/grouped-qkv-final-seed29-001.json), [independent reversed seed43](../results/published/grouped-qkv-final-seed43-001.json).

### Small-shape alternating cross-check

Separate timing blocks have visible submission noise at M128. A [seed29 same-buffer run](../results/published/grouped-qkv-paired-m128-seed29-001.json) and [independent reversed seed43](../results/published/grouped-qkv-paired-m128-seed43-001.json) prepare one common runner, warm both paths 30 times and alternate the first backend in each of60 measured pairs. Native/candidate wall medians are 0.592/0.553ms and 0.588/0.548ms: **6.7–6.8% lower candidate latency**. Supplemental command-buffer GPU medians are 0.430/0.407ms and 0.430/0.405ms, about 5.3–5.8% lower. GPU timestamps are read after the wall timer and remain supplemental. Individual paired ratios still overlap1; this is a small-shape median result, not a per-invocation guarantee or a large-token H3 improvement.

### Numerical evidence

All 24 final normal/outlier implementation cases pass the independent and paired gates, including complete-buffer finiteness and missing-store detection. Every V element is bit-identical to native. Full Q/K paired relative L2 is at most **2.44e-5**; for M4096 seed29 normal,753 Q elements and 743 K elements differ out of29,360,128 elements per stream. The largest independent per-stream implementation relative L2 across both campaigns is 8.53e-5, below the unchanged0.008 limit.

An [ordered-RMS diagnostic](../results/published/grouped-qkv-ordered-rms-001.json) produces bit-identical Q/K/V at M4096 seed43, including the projection and output BF16 rounds. Its online 8.674ms is still slower than native 7.927ms. The `rms_order`, `k_specialization` and `scratch` experiment options apply to the candidate; native always retains its pinned dispatch/reduction policy.

Normal-input total quantization error is about 1.27–1.31% relative L2. Both paths reject **all six outlier conditions** under the per-stream5% quality gate: Q/K errors are about 7.49–8.56%, while V is about 4.77–5.45%. M4096 seed29 has combined error4.96%, which would falsely pass an aggregate-only gate; its Q/K errors correctly reject it. Implementation correctness and quantization quality remain separate states. These unrotated W8A8 results do not evaluate Hadamard's potential quality improvement or real H3 activations.

### Tuning and trace evidence

The [12-tile search](../results/published/grouped-qkv-sweep-001.json) spans TM32/64/128, TN64/128 and four/eight SIMD groups. The fastest online candidate is 8.652ms versus native 8.058ms. Smaller tiles are often substantially slower. A [native-style static full-K5376 variant](../results/published/grouped-qkv-fullk-001.json) gives 8.675ms versus native 7.931ms. Removing threadgroup projection scratch and using the final Q/K buffers as temporary storage also loses: [TM64](../results/published/grouped-qkv-device64-001.json)8.778 versus 7.895ms, [TM128](../results/published/grouped-qkv-device128-001.json)8.963 versus 7.975ms. This does not support a simple claim that shared-memory capacity alone causes the gap.

A [traversal search](../results/published/grouped-qkv-traversal-001.json) tests bands1/4/8/16/32 on the two competitive tile sizes. Band8 remains the best sampled choice; band1 can roughly double latency. This establishes scheduling sensitivity, but does not by itself measure cache miss rates or DRAM bandwidth. All tuning outcomes, including losses, are retained.

The [native Metal trace](../results/published/grouped-qkv-native-m4096-profile-001.json) and [initial shared-scratch trace](../results/published/grouped-qkv-shared-m4096-profile-001.json) contain exactly **one command buffer, two encoders and two dispatches** each. Serial replay attributes98.82%/98.96% of cost to QKV and 1.18%/1.04% to quantization. Native/candidate replay duration is 11.447/12.378ms; these profiler durations include replay overhead and are not benchmark wall latency. The candidate reduces static instruction count1251→711 and device-store instructions 22→11, but both report272 spilled bytes and it remains slower. Temp-register counts are 114/108. These are profiler/compiler observations, not measured dynamic byte traffic. The report exposes no occupancy or bandwidth limiter sufficient to label the path definitively compute-bound or memory-bound. The measured bottleneck is the QKV dispatch; reducing epilogue instruction count alone has not improved it.

The final source was captured again: [native](../results/published/grouped-qkv-native-final-profile-001.json) and [candidate](../results/published/grouped-qkv-candidate-final-profile-001.json) still contain exactly two dispatches, with QKV cost 98.84%/98.93% and replay11.447/12.395ms. Final candidate static instructions are 706, with 108 temporary registers and 272 spilled bytes; the conclusion is unchanged.

Large binary captures remain local under `results/local/`; public counter JSON contains source hashes and numerical checks. Public exports omit only the compiler installation path, identify the changed metadata field and retain the SHA-256 of the original immutable local record; all measurements are unchanged. The [initial default-language compilation failure](../results/published/grouped-qkv-smoke-001.json) is retained; explicit Metal4.1 fixes the common Python-hosted driver without changing upstream shader bytes. The [successful tail smoke](../results/published/grouped-qkv-smoke-002.json) and GPU tests cover supported tail rows, zeros, outliers, tiny values, all Q/K/V layouts, static-K loops and device/threadgroup scratch.

## Reproduce the source and experiment

Run from this repository. The fetch helper clones only the [MIT-licensed h3.c source](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/LICENSE), copyright Salvatore Sanfilippo, at the exact revision above. It verifies the shader, host and license bytes against Git and prints their SHA-256 hashes. An existing destination is checked without modification; a revision or source mismatch fails. `--destination /path/to/checkout` selects a different source directory. No weights are downloaded and no upstream code is built.

```sh
uv sync --locked
uv run python scripts/fetch_native_qkv.py
uv run pytest -q tests/test_fetch_native_qkv.py tests/test_native_qkv_cpu.py

# Run only when the GPU is available; each output path must be new.
uv run python scripts/grouped_qkv_campaign.py --filter m4096 --seed 29 \
  --output results/local/grouped-qkv-reproduction-seed29.json
uv run python scripts/grouped_qkv_paired.py --seed 29 \
  --output results/local/grouped-qkv-paired-reproduction-seed29.json
uv run python -m m5kernels.profile --suite h3-grouped-qkv \
  --case h3.grouped_qkv.m4096 --backend native-qkv --precision int8 \
  --tile 64 128 4 8 --online --output results/local/grouped-native-online.gputrace
```

The benchmark runner defaults to `.cache/h3.c`; a custom fetched location can be supplied to `NativeQKVRunner(upstream_root=...)` in Python. Capture the candidate with `--backend candidate-qkv` and a new output path using the same shape and tile. Captures use the common native command queue; see [profiling](profiling.md) for counter export. The bridge explicitly selects Metal 4.1 and safe math for both libraries, because the Python-hosted default language version did not expose TensorOps.
