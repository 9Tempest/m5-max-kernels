# AdaLN + Hadamard + dynamic INT8 fusion

The operation can be one Metal dispatch. The experiment retains each row on chip while computing normalization statistics, channelwise modulation, block Hadamard and the final **whole-row** INT8 scale. It writes packed INT8 activations and one FP32 scale per row. Weight rotation and quantization remain offline. FP8 is a separate precision policy and is not combined with Hadamard here.

## Semantics and where this applies

H3 uses RMSNorm with learned channel weights, not mean-centered LayerNorm, before attention projections and before the FFN gate/up projections. The tested expression is

```
r = rsqrt(mean(x*x) + epsilon)
y = BF16((x * r * gamma) * (1 + modulation_scale) + modulation_shift)
z = float32(y) D H256
s = max(abs(z)) / 127            # over the complete K row
q = clamp(round_to_even(z / s), -127, 127)
```

Zero rotated rows use scale 1. The BF16 cast is a real numerical boundary, retained inside the fused kernel. This matches the single final AdaLN rounding in the repository's native-style Metal baseline. It is not a byte-parity claim with eager Diffusers, which expresses separate tensor operations, or with upstream h3.c, whose reduction tree and quantization arithmetic differ. The optional FP32-boundary mode has an explicitly different contract.

In real H3, modulation is selected by timestep and modality for each packed token. The kernel API accepts a modulation table plus a row map; current performance experiments use one shared modulation vector and synthetic tensor values. FFN-down consumes **SwiGLU output**, so this AdaLN fusion belongs at Q/K/V and FFN gate/up, not at FFN-down. Q/K/V can reuse one prepared activation, as can gate/up; do not count the same preparation three times in model latency.

The pinned [Diffusers H3 source](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/models/transformers/transformer_minimax_h3.py) establishes the model ordering. The pinned [h3.c Metal source](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_shaders.metal) already contains gate/AdaLN/INT8 fusion without Hadamard. Our kernels are independently implemented; this experiment compares the staged operations in this repository, not a measured win over h3.c's existing fused implementation.

## Passes, reductions and synchronization

Two global passes, two reductions and two dispatches are different quantities. Keeping a row in registers can eliminate global rereads without eliminating the dependencies between reductions. For H3 RMSNorm, the two row reductions are the sum of squares and the maximum **after** modulation and rotation. The centered LayerNorm variant uses mean, centered variance and post-rotation maximum: three row reductions. Its variance calculation avoids subtracting two large moments. [Triton's LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) is another example of retaining a row while reducing its statistics.

For width 5376 and H256, a row contains 21 blocks. A 256-thread workgroup has eight SIMD groups; each SIMD group owns complete blocks, with eight values per lane. Up to three blocks per group require 24 FP32 value slots per thread. Five Hadamard stages exchange values across the 32 SIMD lanes, and three stages exchange the eight registers within each lane. Hadamard therefore needs no threadgroup-memory exchange. Small shared arrays synchronize the row statistics and peak across SIMD groups.

`one_read` retains the initial input values. `two_read` reloads X after the first statistic reduction and then retains intermediate values. Both are one-dispatch algorithms; the source-level read counts do not establish measured DRAM traffic. Register spilling and occupancy can reverse the expected benefit, so both variants are measured rather than assuming fewer reads always wins.

Ignoring channel parameters, caches, scale writes and spills, the staged RMS pipeline has a logical 21 bytes/element of input/intermediate/output traffic: two BF16 X reads, a BF16 AdaLN write/read, a FP32 rotation write and two reads by the quantizer, and the INT8 output. The buffered fused path needs one BF16 X read plus the INT8 output (3 bytes/element). This is an access-count comparison, not a measured bandwidth or speedup claim.

## Why normalization cannot generally disappear

For an exact positive row scalar g, symmetric dynamic quantization has identical integer codes for g*z and z, with its dequantization scale multiplied by g. A pure RMSNorm, or an ideal zero-shift AdaLN, can exploit that identity. General AdaLN has a nonzero shift:

```
H(g*v + shift) = g*H(v) + H(shift)
```

The resulting integer codes depend on g. The retained BF16 rounding also prevents general exact cancellation. Likewise, channelwise modulation cannot simply commute through Hadamard. The implemented fusion preserves the operation order.

## Measured results — 2026-10-18

The fixed selected configuration is H256, 256 threads, `one_read`, with a BF16 boundary after modulation and the existing INT8 GEMM tile `(128,64,4,8)`. An initial seed17 sweep compared 128/256 threads and one/two input reads. Seed29 and an independent seed43 rerun then compared the staged and fixed fused paths on the same M5 Max. Each result uses five warmups and 30 synchronized timed repetitions. The BF16 comparison is **Metal AdaLN + MLX BF16 GEMM**, not an entirely MLX pipeline.

Normal synthetic inputs, seed29:

| Projection | Staged preparation ms | Fused preparation ms | Preparation speedup | Staged online ms | Fused online ms | Online speedup |
|---|---:|---:|---:|---:|---:|---:|
| Q, M=128 | 0.1667 | 0.1451 | 1.15× | 0.3010 | 0.2766 | 1.09× |
| Q, M=4096 | 0.8071 | 0.4396 | 1.84× | 3.9734 | 3.6338 | 1.09× |
| FFN-up, M=4096 | 0.8506 | 0.5163 | 1.65× | 7.2478 | 6.7800 | 1.07× |
| Q, M=9914 | 1.7982 | 0.9168 | 1.96× | 9.4527 | 8.5475 | 1.11× |

Seed43 independently repeats the Q4096 improvement at 3.9812 → 3.6626 ms (1.087×) and Q9914 at 9.3878 → 8.5381 ms (1.100×). FFN-up's full online gain varies from 1.026× to 1.069× across these reruns; M128 also varies from 1.026× to 1.088× and is sensitive to host overhead. The preparation result should not be presented as the projection or model speedup.

In seed29 synthetic outliers, fused Q4096 / FFN-up4096 / Q9914 total relative L2 is 0.592% / 0.792% / 0.910%; normal-input errors are approximately 1.2–1.4%. These include quantization and BF16 GEMM output rounding. Ideal quantization, actual preparation, and GEMM implementation errors are recorded separately. Staged/fused results need not be bit-identical because reduction ordering can change rounding at thresholds. The gates are unchanged and all published candidates pass. These errors are against this AdaLN expression and its synthetic values, so they are not interchangeable with the earlier raw-activation Hadamard experiment.

M9914 follows the integration task's 512×512, 124-frame request (9472 video + 414 audio + 28 text tokens); these records still use synthetic activation/weight/modulation values. LayerNorm and mapped modulation have GPU correctness coverage; the performance table measures shared-vector RMSNorm/AdaLN only.

Evidence: [configuration sweep](../results/published/2026-10-18-norm-hadamard-sweep.json), [seed29 validation](../results/published/2026-10-18-norm-hadamard-independent.json), [independent seed43 rerun](../results/published/2026-10-18-norm-hadamard-rerun.json).

The [staged capture](../results/published/2026-10-18-norm-hadamard-staged-profile.json) contains four dispatches: AdaLN, Hadamard, quantization and GEMM. The [fused capture](../results/published/2026-10-18-norm-hadamard-fused-profile.json) confirms two: fused preparation and GEMM, each invoked once. The selected preparation shader has **48 temporary registers and zero spilled bytes**, and accounts for 8.72% of this serial replay's shader cost. GEMM accounts for 91.28%; its reported spills are separate from the preparation shader. These are profiling counters, not synchronized-wall timing or proof of a specific bottleneck.

The slower [128-thread single-read variant](../results/published/2026-10-18-norm-hadamard-128-profile.json) reports 64 temporary registers and **208 spilled bytes**, versus zero spills at 256 threads. Its seed17 normal-input preparation is 0.647 ms versus 0.447 ms for the selected 256-thread variant; reloading X with 128 threads gives 0.453 ms. The counter difference supports register pressure as a cause worth investigating, but does not isolate it from instruction count and occupancy. Raw captures stay under ignored `results/local/norm-hadamard-{staged,fused,128}-001.gputrace`.

## Reproduction commands

```sh
uv sync --locked
uv run python scripts/norm_hadamard_campaign.py \
  --output results/local/norm-hadamard-sweep-001.json
uv run m5bench bench --suite h3-norm-hadamard --filter norm_q.m4096 \
  --backends norm-staged,norm-fused --precisions int8 \
  --tile 128 64 4 8 --output results/local/norm-hadamard-001.json
uv run python -m m5kernels.profile --kind gpu --suite h3-norm-hadamard \
  --case h3.norm_q.m4096 --backend norm-fused --precision int8 --online \
  --tile 128 64 4 8 --output results/local/norm-hadamard-001.gputrace
uv run python scripts/profile_report.py results/local/norm-hadamard-001.gputrace \
  --expect-dispatches 2 --output results/local/norm-hadamard-profile-001.json
```

The operation `norm_hadamard_gemm` follows the regular workload JSON schema. Its `fusion` configuration specifies norm, epsilon, block size, rounding boundary, thread count, read strategy and optional sign seed. The harness records independent float64 normalization/rotation references, ideal quantization error, actual preparation error and GEMM implementation error separately. `activation_prepare` measures the complete norm/modulation/rotation/quantization stage; `online_activation_plus_gemm` includes this stage and GEMM. Compilation and offline weight preparation are excluded.

Supported rotation blocks are exact divisors of K in 32–1024 powers of two; no hidden padding is used. Modulation-table bounds and finite inputs are caller preconditions. The [gated residual extension](gated-residual-fusion.md) now has separate/fused synthetic measurements and GPU captures. Full-model activation distributions, multimodal-table performance and generated-video quality remain separate validation tasks.
