# Q/K projection, RMSNorm and RoPE fusion

H3 applies RoPE after per-head Q/K RMS normalization in the main DiT blocks. The independent implementation in this repository now has staged/fused measurements: seed29's complete online path is 13.0% lower at M4096 and 17.6% lower at M9914 for synthetic normal inputs. These are preliminary results pending the independent seed43 rerun. The pinned native runtime already fuses INT8 QKV projection, Q/K normalization and RoPE; this experiment does not establish a new native capability or a performance advantage over that runtime.

## Pinned sources and layouts

The model configuration has 56 heads of width 128, projection input width 5376 and Q/K normalization epsilon `1e-5`. Q and K have independent learned gamma vectors of length 128, shared across heads and tokens. The text refiner uses Q/K normalization without RoPE. The official implementation defines three position axes, with 16 frequencies per axis: the resulting 48 angles rotate 96 of the 128 channels. It uses split-half pairing, rather than adjacent channel pairs. See the pinned [Diffusers attention and rotary definitions](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/models/transformers/transformer_minimax_h3.py#L50) and [local configuration](../sources/transformer-config.json).

Native source is `antirez/h3.c` revision `8974cc055ea9c02fcd14cc27dfda3e1027c05153`:

- [DiT call site](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_dit.c#L1901) selects the grouped QKV projection/norm/RoPE path before attention.
- [INT8 projection epilogue](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_shaders.metal#L1754) maps output columns as `[head, stream, channel]`, with stream order Q, K, V. Each head therefore contributes Q128, K128, V128 before the next head. Three destination tensors each use contiguous `[H,M,128]` layout: offset `(head*M + row)*128 + channel`.
- [Native dispatch and mode selection](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L3885) requires width 128, 48 rotary pairs and heads divisible by four for this specialized path. It also selects a local-scale variant at larger row counts, and marks the attention inputs as head-major. These are native implementation restrictions, not mathematical requirements.
- [Position-table preparation](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_dit.c#L883) forms the time, height and width angles, with spatial scaling on height/width. Native cosines and sines are stored as BF16 `[M,48]`; tables are shared across heads.

The official Diffusers fused-projection expression splits a concatenated Q/K/V output into three streams before introducing heads. That layout differs from the native grouped checkpoint ordering. Any fused-QKV benchmark or adapter must declare and account for its weight layout explicitly. A single Q or K projection has ordinary `[H*128,K]` weights and avoids this distinction.

## Native numerical contract

For one row and head, let `p[d]` denote the scaled projection result. The native path computes:

1. `z[d] = BF16(p[d])`. INT8 projection uses INT32 accumulation followed by FP32 activation and weight scales; BF16 rounding happens before normalization.
2. `r = rsqrt(sum(z[d]^2)/128 + 1e-5)`, using all 128 channels of that head. Statistics precede gamma and do not mix heads or tokens.
3. `u[d] = float32(z[d]) * r * float32(gamma[d])`. There is **no BF16 round after normalization** in the native fused epilogue.
4. For each `j` from 0 to 47, rotate the pair `(j,j+48)`: `v[j] = u[j]*cos[j] - u[j+48]*sin[j]` and `v[j+48] = u[j+48]*cos[j] + u[j]*sin[j]`. Channels 96–127 retain `u[d]`.
5. Round Q/K outputs to BF16 and store in head-major layout. V has projection rounding and layout conversion only; it is neither normalized nor rotated.

The native kernel temporarily writes the rounded projection into its final device output, synchronizes, obtains each row's RMS factor and overwrites Q/K with normalized/rotated values. Its RMS summation follows a scalar channel order, including the vector-load variant. A SIMD reduction may change FP32 summation order. Matching boundaries and passing an independent numerical gate do not establish byte equality with that implementation.

Diffusers expresses normalization and rotation as separate framework operations. This native-style contract does not claim eager Diffusers byte parity. Adding a BF16 normalization boundary is an explicit alternate policy, not a harmless implementation detail. Likewise, gamma cannot generally move after RoPE: a channelwise diagonal scaling and a rotation do not commute when the two paired gamma values differ. BF16 cosine/sine tables also need not satisfy `cos²+sin²=1` exactly.

## Independent reference and fusion design

[qk_rope_reference.py](../src/m5kernels/qk_rope_reference.py) is original NumPy code, with float64 arithmetic between named rounding boundaries. Its separate BF16 helper implements FP32-to-BF16 round-to-nearest-even without importing MLX. Gamma and cosine/sine operands are consumed as supplied; native comparisons must pass their actual decoded BF16 values. The optional normalization round exists for policy diagnostics, not as an eager-framework emulator.

The reference accepts a single Q or K projection as `[M,H,D]` and returns contiguous head-major `[H,M,D]` output plus projection, inverse RMS, normalization and rotation intermediates. It supports zero rotary pairs and smaller tail dimensions for mathematical tests. A GPU kernel may explicitly support a narrower domain. [CPU tests](../tests/test_qk_rope_reference.py) use scalar RMS accumulation and dense rotation matrices, and cover independent arithmetic BF16 rounding, zeros, outliers, constant rows, odd row/head counts, noncontiguous inputs, invalid values and shapes, normalization axes, split-half pairing, untouched rotary tails and both rounding-boundary counterexamples.

The implemented [projection kernel](../src/m5kernels/shaders/qk_rope_projection.metal) assigns a complete 128-channel head to each threadgroup. TN128 uses one TensorOps matrix product; TN64 uses two sequential half-head products. Both halves write into the same BF16 threadgroup tile using the accumulator's reported multidimensional indices. A uniform barrier precedes normalization, then each SIMD group owns one row/head at a time. Invalid tail rows are never consumed. No fixed lane-to-accumulator layout is assumed.

This BF16 scratch preserves the projection round without a device projection buffer. TM32, TM64 and TM128 require 8, 16 and 32 KiB respectively. Register use and threadgroup-memory occupancy can offset the saved buffer traffic, so larger tiles are candidates rather than automatic improvements. The shared [postprocessing body](../src/m5kernels/shaders/qk_rope_body.metal) uses FP32 per-head reduction and rotation, with no BF16 norm intermediate. The [standalone comparator](../src/m5kernels/shaders/qk_rope_norm.metal) reads the device projection and directly writes the same head-major output; no layout conversion is omitted from the comparison.

The `qk_rope` operation currently benchmarks one Q or K projection with ordinary `[H*128,K]` weights. It does not pack native grouped QKV weights, produce V, or run attention. The lower-level adapter supports INT8 and BF16 projection; the campaign described here uses INT8 only. Both candidates share FP32 H256 rotation and symmetric per-token/per-output-channel INT8 preparation. Weight rotation and packing are offline; activation rotation and packing are online. Gamma and trig tables are actual BF16 operands. Their values and the inputs/weights are synthetic; the angles are independent random samples rather than an H3 position-table trace.

Correctness references include **all 128 channels of each sampled head and full K** before normalization. Sampling isolated projection columns does not define a valid RMS reference. The harness samples up to 32 rows and two complete heads, checks all outputs for finiteness, and requires exact full-buffer agreement between paired INT8 staged/fused paths. Implementation error against decoded quantized operands is separate from quality error against original BF16 operands, using the same native-style normalization/rotation policy for both. Projection and postprocess errors are also recorded independently. GPU tests include epsilon-dominated small nonzero projections, where an erroneous projection scale cannot hide behind RMS normalization, and explicit rounding/pairing/tail regressions.

`timing` measures resident INT8 projection through normalization, RoPE and head-major output. `online_activation_plus_gemm` additionally includes activation H256 and INT8 packing. The staged online path has four dispatches: rotation, packing, projection and postprocessing. The fused path has three: rotation, packing and projection with its epilogue. Isolated projection/postprocess stages are diagnostics for the staged components; they are not separate pieces of the fused timing.

## Tile selection and repeated measurements — 2026-10-18

The [M4096 seed17 sweep](../results/published/2026-10-18-qk-rope-m4096-sweep-s17.json) compares the staged `(128,64,4,8)` tile once per distribution with 12 fused configurations: TM32/64/128, TN64/128, four/eight SIMD groups, and M swizzle eight. Its 26 records cover normal and outlier distributions; all pass implementation checks and report exact paired outputs. This sweep selects `(64,128,4,8)` for the independent comparison. It is configuration-selection evidence, not a promotion run.

The [seed29 comparison](../results/published/2026-10-18-qk-rope-final-s29.json) uses 30 warmups and 40 synchronized host-wall repetitions on the same M5 Max. The staged tile remains `(128,64,4,8)` and the fused tile is `(64,128,4,8)`. The table reports complete online medians, including H256, activation packing, native-style projection/RMS/RoPE and head-major BF16 output. Seed43 independently repeats the experiment with the candidate order reversed.

| Rows M | Distribution | Seed29 staged → fused ms | Latency reduction | Seed43 staged → fused ms | Latency reduction |
|---|---|---:|---:|---:|---:|
| 128 | normal | 0.301 → 0.297 | 1.3% | 0.298 → 0.301 | -1.1% |
| 128 | outliers | 0.368 → 0.286 | 22.3% | 0.296 → 0.302 | -2.0% |
| 4096 | normal | 3.977 → 3.460 | 13.0% | 4.012 → 3.456 | 13.9% |
| 4096 | outliers | 3.995 → 3.447 | 13.7% | 4.022 → 3.456 | 14.1% |
| 9914 | normal | 9.659 → 7.959 | 17.6% | 9.765 → 7.944 | 18.6% |
| 9914 | outliers | 9.863 → 8.125 | 17.6% | 9.685 → 7.952 | 17.9% |

All 24 final records pass the implementation and synthetic quality gates, with exact full-buffer paired outputs. Total original-operand relative L2 is 1.11–1.31%, identical between the paired paths. These values measure the complete synthetic Q/K expression after quantization; they do not measure model output quality. M9914 matches a separately recorded packed-token workload scenario, while tensor values remain synthetic.

The M128 results need particular care before drawing a performance conclusion. In this run the fused normal-input resident core median is 0.493 ms, while its separately timed complete online median is 0.297 ms even though the latter includes that core. The separate timing intervals therefore do not establish a consistent small-M decomposition. The reversed-order rerun gives about 0.30 ms for both M128 paths, with the fused path slightly slower. There is no repeatable small-M win. Raw samples are preserved; these medians alone do not identify a cache or hardware cause.

The [seed43 reversed-order rerun](../results/published/2026-10-18-qk-rope-final-s43.json) confirms 13–14% lower online latency at M4096 and 18–19% at M9914. This compares complete single-projection expressions, not the upstream grouped QKV implementation. The [baseline profile](../results/published/2026-10-18-qk-rope-baseline-profile.json) confirms four online dispatches; the [fused profile](../results/published/2026-10-18-qk-rope-fused-profile.json) confirms three. These captures use M4096, seed29 and the benchmark's selected tiles.

| GPU debugger metric | Staged | Fused |
|---|---:|---:|
| Projection shader cost | 86.69% | 88.96%, includes RMS/RoPE |
| Separate RMS/RoPE cost | 4.01% | Included in projection |
| Input H256 cost | 5.00% | 5.79% |
| Input INT8 packing cost | 4.30% | 5.25% |
| Projection temporary registers | 112 | 116 |
| Projection spilled bytes reported | 272 | 272 |
| Online dispatches | 4 | 3 |
| Serial replay duration | 5.6930 ms | 4.7048 ms |

The dominant cost remains the projection, and the fused shader's register increase is small without increased reported spills. Removing a device projection store/read and the postprocess dispatch helps, but the chosen GEMM tile also changes; this experiment does not attribute the whole gain to dispatch removal alone. The source allocates 16 KiB of BF16 threadgroup scratch for the selected fused tile. The exported shader summary nevertheless reports threadgroup memory as zero, so that field is not treated as reliable evidence of actual occupancy. No DRAM-bandwidth or cache-bottleneck claim follows from these counters.

Replay durations include profiling overhead and are separate from the synchronized wall medians above. Cost percentages are normalized within each capture. Binary traces remain local under `results/local/qk-rope-{baseline,fused}.gputrace`; sanitized public JSON includes capture settings, source/environment metadata and counters. The next performance target is the projection's TensorOps tiling/register scheduling, with any improvement checked against the full online path.


## Reproduce

```sh
uv sync --locked
uv run python scripts/qk_rope_campaign.py --filter m4096 --seed 17 \
  --warmup 10 --repeat 20 --sweep \
  --output results/local/qk-rope-sweep-new.json
uv run python scripts/qk_rope_campaign.py --filter '' --seed 29 \
  --warmup 30 --repeat 40 --tile 64 128 4 8 \
  --output results/local/qk-rope-final-new-s29.json
uv run python scripts/qk_rope_campaign.py --filter '' --seed 43 --reverse \
  --warmup 30 --repeat 40 --tile 64 128 4 8 \
  --output results/local/qk-rope-final-new-s43.json

uv run python -m m5kernels.profile --kind gpu --suite h3-qk-rope \
  --case h3.qk_rope.m4096 --backend qk-staged --precision int8 \
  --tile 128 64 4 8 --online --output results/local/qk-rope-baseline-new.gputrace
uv run python scripts/profile_report.py results/local/qk-rope-baseline-new.gputrace \
  --expect-dispatches 4 --output results/local/qk-rope-baseline-new-profile.json
uv run python -m m5kernels.profile --kind gpu --suite h3-qk-rope \
  --case h3.qk_rope.m4096 --backend qk-fused --precision int8 \
  --tile 64 128 4 8 --online --output results/local/qk-rope-fused-new.gputrace
uv run python scripts/profile_report.py results/local/qk-rope-fused-new.gputrace \
  --expect-dispatches 3 --output results/local/qk-rope-fused-new-profile.json
```

Run GPU experiments serially and use fresh filenames. The campaign records code/environment provenance, tile order, seeds, raw samples and explicit failures, and saves completed records if interrupted. `--tile` changes only the fused candidate; the staged baseline always uses `(128,64,4,8)`.

No attention throughput, denoiser trajectory or generated-output quality claim follows from this component experiment. Native grouped QKV parity, full-model integration and performance promotion require their own recorded evidence.
