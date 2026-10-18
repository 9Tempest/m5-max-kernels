# Moving AdaLN shift through the projection

Moving the AdaLN shift to a cached output bias is algebraically valid before rounding, and H3 has enough repeated conditioning rows to make caching plausible. **The initial implementation does not improve measured latency.** At Q4096, the cached split path is approximately tied or slower than the full FP32 AdaLN path; its initial MLX-based preparation of one conditioning row costs another 0.91–1.03 ms. Native BF16 bias projection subsequently reduces condition preparation to 0.408–0.471 ms in complete-path reruns, but the condition-inclusive split remains slower than the full FP32 control. The original and follow-up evidence are retained separately below.

## Three different numerical policies

Write the learned RMS weight as gamma, modulation scale as alpha, shift as beta, and the positive RMS factor of token i as r_i. Let c_p = gamma * (1 + alpha_p), where p selects the token's conditioning row. In exact arithmetic:

```
y_i = r_i * (x_i * c_p) + beta_p
y_i @ W.T = r_i * ((x_i * c_p) @ W.T) + b_p
b_p = beta_p @ W.T
```

The experiment compares these policies with the same original BF16 inputs/weights, H256, per-row INT8 quantization, BF16 GEMM output and GEMM tile `(128,64,4,8)`:

| Policy | Activation preparation | Shift projection |
|---|---|---|
| Native BF16 boundary: `norm-fused`, `round_output=input` | FP32 norm/modulation, then BF16 completed AdaLN, FP32 H256 and INT8 quantization | Included in the quantized activation |
| Full FP32 AdaLN: `norm-fused`, `round_output=fp32` | FP32 norm/modulation/H256 and INT8 quantization; no intermediate BF16 AdaLN cast | Included in the quantized activation |
| Split: `norm-shift-split`, `round_output=fp32` | Compute `(x*c_p) D H256`, quantize it, and multiply the positive inverse RMS into its row scale | Cached `beta_p @ original_BF16_W.T`, accumulated/stored in FP32 and added in GEMM's epilogue |

For the split path, the row maximum must follow channelwise c and Hadamard. RMS statistics still use original x. Zero dynamic rows retain zero codes and unit scale; their output can still be the nonzero cached bias. Bias is added after INT32 GEMM accumulation and row-scale dequantization, before the final BF16 output round. It must precede downstream Q/K normalization and RoPE, or the FC1 SwiGLU nonlinearity.

The implemented split policy deliberately mixes original BF16 weights for the shift branch with rotated INT8 weights for the dynamic branch. It retains the original weights to build the bias; that memory and setup cost are not free. It is not the same effective weight matrix as a single W8A8 projection.

An alternative would cache `(beta_p D H) @ decoded_rotated_INT8_W.T`. That uses the same effective quantized weight for both branches, but is not the policy measured here. Quantizing beta itself to accelerate the precompute would introduce another approximation. These choices must have different cache keys and numerical references.

## The BF16 boundary prevents an exact rewrite

The native contract contains `BF16(r*x*c + beta)`. Its rounding correction depends on x, so it cannot be represented by a constant projected beta. For example, adding beta=1/256 to the two BF16 values 1 and 1+1/128 produces rounding corrections 0 and 1/128 respectively. A single cached shift cannot reproduce both.

Exact reconstruction would require the activation-dependent correction `e(x) @ W.T`, where `e(x) = BF16(r*x*c + beta) - (r*x*c + beta)`. That restores a full contraction and defeats the simple cache proposal. The split implementation is therefore an explicit alternate precision policy. Its `native_bf16_boundary_diagnostic` reports the difference between ideal native-boundary and FP32 reference projections; this difference is about 0.158–0.163% relative L2 in the current synthetic cases, not a model-quality bound.

Separating beta can also worsen quantization when the dynamic term and beta cancel. The test with `u=[1,1000]` and `beta=[-0.99,-1000]` retains a small joint signal; quantizing u first loses the first component and causes a much larger error after adding beta. The algebra remains correct, but the quantized approximation changes. These and the gamma-folding counterexample are preserved in [test_affine_rewrites.py](../tests/test_affine_rewrites.py).

## Initial measured result — 2026-10-18

These are synthetic Q4096 workloads, K=5376, N=7168, **one shared conditioning row P=1**, on the same 40-core M5 Max. Both campaigns use five warmups and 30 synchronized host-wall repetitions. Seed43 reverses candidate order. No full-model speedup or quality conclusion follows from this experiment.

| Seed / distribution | Native BF16-boundary online ms | Full FP32 AdaLN online ms | Split, cached condition online ms | Split condition preparation ms | Split including condition preparation ms |
|---|---:|---:|---:|---:|---:|
| 29 / normal | 3.640 | 3.654 | 3.682 | 1.028 | 4.576 |
| 29 / outliers | 3.600 | 3.597 | 3.646 | 0.910 | 4.409 |
| 43 / normal | 3.637 | 3.612 | 4.048 | 0.984 | 4.593 |
| 43 / outliers | 3.896 | 3.650 | 3.666 | 0.911 | 4.394 |

`online_activation_plus_gemm` reuses the combined channel factors and projected bias. `condition_prepare` computes both, including the current implementation's FP32 conversion/projection of retained original BF16 weights. `online_with_conditioning` executes the condition preparation and complete projection together. Its median is measured directly; it is not the sum of independently measured medians.

Even with conditioning cached, split activation preparation takes 0.510–0.683 ms versus 0.467–0.488 ms for full FP32 AdaLN. The split path is slower than its full FP32 comparator in every pair above. Seed43 normal has substantial timing spread, and the seed43 native-boundary outlier case is also slower than its seed29 result; an apparent win against that one native run does not establish a repeatable improvement. No performance promotion is justified.

All 12 records pass the implementation and synthetic quality gates. Split total relative L2 is 1.237% / 1.246% for normal seeds29/43, and 0.609% / 0.796% for outliers, against the FP32-policy reference. The native-boundary policy has a different reference, so its error numbers are not interchangeable. Independent decoded-operand GEMM checks, ideal quantization error, actual preparation error and projected-bias error remain separate. Cancellation and real activation distributions still need evaluation.

Evidence: [seed29 comparison](../results/published/2026-10-18-shift-split-independent.json), [reversed-order seed43 rerun](../results/published/2026-10-18-shift-split-rerun.json). Raw samples and code/environment provenance are retained.

The [cached split GPU capture](../results/published/2026-10-18-shift-split-profile.json) contains two dispatches, each invoked once: folded RMS/Hadamard/quantization and INT8 GEMM with its projected-bias epilogue. Preparation reports **75 temporary registers, zero spilled bytes and 9.68% of shader cost**. GEMM reports **112 temporary registers, 272 spilled bytes and 90.32% of shader cost**. These percentages belong to a serial replay with profiling overhead; they are not synchronized benchmark-wall fractions. This capture reuses the condition cache and does not include bias construction. Absence of preparation spills does not prove that the folded preparation is faster, and GEMM remains the dominant recorded cost.

## How much conditioning reuse H3 actually offers

The pinned h3.c revision is `8974cc055ea9c02fcd14cc27dfda3e1027c05153`. Its [schedule constants](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_dit_schedule.h#L11) define 50 blocks, 3 modalities, 6 AdaLN slots and hidden width 5376. [Per-block projections](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_dit_schedule.c#L264) already materialize all schedule-dependent AdaLN values in BF16 before denoising.

For plain text-to-video/audio, the [row mapping](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_dit_schedule.c#L449) uses video/tag0, text/tag1 at the video timestep, and audio/tag2 at the audio timestep: three active rows per step. The [serving schedule](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_host.c#L150) and [time-row allocation](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_dit_schedule.c#L100) produce 39 time rows for 20 steps: one shared initial audio/video time and two rows for each later step. The full table therefore has 117 modality rows, while this plain request uses 60 distinct row indices. Conditioning images/audio or mixed text tags change the accessed set; collect it from actual maps rather than assuming P=3 universally.

Beta and the projection weights vary by block and branch. The cache is per block, projection and conditioning row; it cannot be shared across all layers. Attention QKV uses slots 0/1, while the MLP input uses slots 3/4 after the attention gate: [attention path](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_dit.c#L1897), [MLP preparation](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_dit.c#L1960). Reference slots should be consumed through their existing semantics, not reordered from an inferred name.

The following are **theoretical arithmetic/storage counts, not measured latency or memory peaks**:

| Q projection quantity | M=4096 | M=9914 |
|---|---:|---:|
| Main GEMM, 2MNK | 315.680 GFLOP | 764.075 GFLOP |
| Three-row bias projection, 2PNK | 0.231 GFLOP | 0.231 GFLOP |
| Bias projection/main GEMM arithmetic ratio | 0.0732% | 0.0303% |
| Bias additions in fused epilogue | 29.36 million | 71.06 million |
| Extra BF16 output read/write if bias were a separate pass | 112 MiB | 271.09 MiB |

Q weights occupy 73.5 MiB in BF16 or 36.75 MiB in INT8. Three FP32 projected bias rows occupy 84 KiB. A tiny FLOP ratio therefore does not imply tiny setup latency: low-P GEMMs can be limited by weight reads, conversion and dispatch, as the measured P=1 result demonstrates.

For 20 steps and all 50 blocks, retaining FP32 bias for QKV and complete FC1 (both gate and up) requires **0.561 GiB for the 60 used rows**, or **1.093 GiB for all 117 stored rows**. The associated projections require approximately **1.618 / 3.156 TFLOP** once per conditioning schedule. These counts exclude combined channel factors, weight copies, quantization scales, allocator overhead and temporary buffers. They are not evidence that this cache has been implemented or will be fast.

## Measured bias-cache construction follow-up

An isolated microbenchmark now compares `beta @ W.T` using resident BF16 operands and FP32 output. `mlx-fp32-convert` explicitly converts both inputs to FP32 inside timing. `mpp-bf16-to-fp32` uses native BF16 input tensors with FP32 accumulation/output and tile `(32,64,4,1)`, avoiding a materialized FP32 weight copy. Every required conversion for each path is included. This benchmark uses ten warmups and 30 synchronized host-wall repetitions, synthetic beta rows and one resident Q projection; it does not load weights or build a full model cache.

| Conditioning rows P | MLX conversion, seed29 ms | Native BF16, seed29 ms | MLX conversion, seed43 ms | Native BF16, seed43 ms |
|---|---:|---:|---:|---:|
| 1 | 1.600 | 0.395 | 0.917 | 0.358 |
| 3 | 0.986 | 0.365 | 0.975 | 0.356 |
| 60 | 0.976 | 0.402 | 0.979 | 0.381 |
| 117 | 0.981 | 0.712 | 0.972 | 0.660 |

Native P60 construction takes approximately the same time as P1: 0.402 versus 0.395 ms at seed29, and 0.381 versus 0.358 ms at seed43. This supports batching all used step/modality rows to reuse the resident weights; it does not measure physical DRAM reads or prove that only one weight read occurred. P117 costs more, making selection of actually used rows worthwhile to test. Seed29 P1 has substantial spread, including 0.895–2.013 ms p10–p90 for MLX, so the independent seed43 result is necessary context for that median.

If one P60 cache were used across 20 steps, dividing these measured setup medians by 20 gives a **theoretical amortized allocation of about 0.020 / 0.019 ms per step for this one projection**. No per-step or model latency of that size was measured. Cache lookup, actual conditioning rows, integration, first-use compilation/loading and all other model projections remain outside this microbenchmark. The initial complete-path table is unchanged: a faster condition builder alone cannot reverse its cached-online slowdown.

All 16 bias-construction records pass the independent float64 reference gate over all rows, sampled output columns and full K, with finite full outputs. Evidence: [seed29 bias cache comparison](../results/published/2026-10-18-bias-cache-independent.json), [seed43 rerun](../results/published/2026-10-18-bias-cache-rerun.json). These records time per-request conditioning preparation, not reusable static weight work.

## Native bias projection in the complete path

The next Q4096 P=1 comparison switches condition projection to `mpp-bf16-to-fp32` while retaining the original-W bias policy, combined channel-factor preparation and bias epilogue. These new records use **50 warmups and 30 timed repetitions**; the earlier five-warmup records and initial table remain unchanged. Seed43 again reverses candidate order. Condition-inclusive timing rebuilds the cache within each timed call, but inputs remain resident and compilation is warmed: it is not cold process/model startup.

| Seed / distribution | Full FP32 AdaLN online ms | Split, cached condition online ms | Native condition preparation ms | Split including condition preparation ms |
|---|---:|---:|---:|---:|
| 29 / normal | 3.844 | 3.844 | 0.450 | 4.132 |
| 29 / outliers | 3.646 | 3.802 | 0.471 | 4.094 |
| 43 / normal | 3.790 | 3.818 | 0.408 | 4.114 |
| 43 / outliers | 3.788 | 3.832 | 0.471 | 4.105 |

Native condition preparation now takes 0.408–0.471 ms, including the combined gamma/modulation factor. This improves the setup component, but cached online execution remains tied or slower, and the condition-inclusive path takes 4.094–4.132 ms against 3.646–3.844 ms for its full FP32 control. All 12 new records pass the numerical gates. The faster bias builder therefore does not justify promoting this split policy; schedule-wide batching and further preparation/epilogue changes still need complete measurements.

Evidence: [native bias seed29 comparison](../results/published/2026-10-18-shift-split-native-bias.json), [native bias seed43 rerun](../results/published/2026-10-18-shift-split-native-bias-rerun.json). The complete timings are directly measured rather than reconstructed from the isolated cache benchmark.

## Ranked follow-ups

1. **Apply the measured native batching approach to actual conditioning rows before denoising, then remeasure the complete path.** Reuse the finite, already-rounded BF16 AdaLN tables, rather than regenerate ideal FP32 coefficients. Project all required beta rows together while each original projection weight is loaded, before the runtime frees its BF16 copy after quantization. Cache combined c separately with its declared FP32 arithmetic. The synthetic bias-cache result above supports this setup optimization, but it does not fix the observed cached-online slowdown by itself. Record cold setup, cache-hit execution and amortization over requests/steps; include checkpoint, block/projection, actual table values, rotation, quantization and rounding policy in the cache key. The current [weight lifetime](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_dit.c#L694) makes the load boundary the practical location.

2. **Fuse the preceding GEMM's gated residual epilogue.** Compute the branch, apply the declared branch-output BF16 round, then perform the gate/residual addition and store BF16 R_next. This could avoid materializing and rereading a separate branch tensor. It does not make full row normalization free: ordinary GEMM tiles split output columns among threadgroups, so a complete row RMS needs cross-threadgroup coordination or another dispatch. Preserve the branch round and residual round explicitly; bypassing either creates a different precision policy. Measure extra epilogue inputs/register pressure as well as removed traffic.

3. **Fold the H256 normalization factor 1/16 into the output scale.** A uniform positive scalar cancels from symmetric dynamic INT8 codes, so unnormalized butterflies can use `S = peak/127 * inverse_rms/16` in the zero-shift path. The zero-row rule remains Q=0/S=1. This removes only normalization arithmetic, not Hadamard exchanges or reductions. Intermediate range grows by 16; retain finite/range tests and nearest-even threshold tests. The benefit is an unmeasured small-instruction optimization, not an assumed bandwidth improvement.

4. **Treat static gamma folding as an ablation, never an automatic rewrite.** Although `(x*gamma)W.T = x(W*gamma).T` before quantization, the quantizer sees different ranges. With x=[1,1000], gamma=[1000,1] and W=[1,1], separately quantized `(x*gamma)` and W produce 2000; separately quantized x and `(W*gamma)` produce 0. Any rebalanced weights need new independent error gates and real calibration. The dynamic modulation factor still varies with conditioning.

5. **Do not keep every residual permanently in a rotated basis without a full graph design.** General channelwise gates and modulation do not commute with H: applying a gate in rotated coordinates requires the generally dense `T.T @ diag(gate) @ T`. SwiGLU and componentwise BF16 rounding also do not commute with the transform. RMS norm preserves Euclidean length under an ideal orthogonal transform, but that isolated identity does not remove these other barriers. A permanent rotated representation is a separate model/precision experiment, not a free kernel optimization.

The next useful decision is whether a batched bias cache plus a better folded preparation/epilogue can beat the existing fused full-AdaLN path under the same complete timing scope. Current measurements answer that question negatively for the implemented P=1 path; they do not justify replacing the default kernel.

## Reproduce the initial comparison

```sh
uv sync --locked
uv run python scripts/shift_split_campaign.py --filter m4096 --seed 29 --repeat 30 \
  --output results/local/shift-split-independent-new.json
uv run python scripts/shift_split_campaign.py --filter m4096 --seed 43 --repeat 30 --reverse \
  --output results/local/shift-split-rerun-new.json
uv run python scripts/bias_cache_campaign.py --seed 29 --repeat 30 \
  --output results/local/bias-cache-independent-new.json
uv run python scripts/bias_cache_campaign.py --seed 43 --repeat 30 \
  --output results/local/bias-cache-rerun-new.json
uv run python scripts/shift_split_campaign.py --filter m4096 --seed 29 --repeat 30 \
  --bias-backend mpp-bf16-to-fp32 --warmup 50 \
  --output results/local/shift-split-native-bias-new.json
uv run python scripts/shift_split_campaign.py --filter m4096 --seed 43 --repeat 30 --reverse \
  --bias-backend mpp-bf16-to-fp32 --warmup 50 \
  --output results/local/shift-split-native-bias-rerun-new.json
```

Run GPU experiments serially and choose fresh filenames. The suite also declares a Q9914 scenario, but the initial complete-path result table above measures Q4096 only. Projection bias tables and maps are supported by the GEMM API; the complete-path experiment uses shared P=1 while isolated bias construction covers P=1/3/60/117. Model-level integration, batching of actual schedule coefficients, cached quantized-W bias construction and generated audiovisual quality remain unmeasured.
