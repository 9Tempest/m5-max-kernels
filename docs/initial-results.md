# Initial measurements — 2026-10-18

Apple M5 Max (40 GPU cores, 128 GiB), macOS 27.0.1, Xcode 27.1, MLX 0.32.4. Synthetic normal inputs; resident weights and activations. All timings below are synchronized host wall medians, not GPU counter durations.

RMS/affine follow-up: folding inverse RMS into INT8 scales reduces synchronization but does not show a stable full-path improvement after paired seeds, scheduling changes and 256/512-thread trials. Moving AdaLN shift into a projected GEMM bias is implemented as an explicit alternate precision policy; its shared-P1 conditioning-inclusive path remains slower. Native BF16-to-FP32 projection reduces condition preparation, and building 60 synthetic Q bias rows costs 0.38–0.40 ms in isolation. This is a setup microbenchmark, not measured whole-model acceleration. See [RMS scale folding](rms-scale-folding.md) and [shift splitting](affine-shift-splitting.md), including quality rejections and negative results.

## Independently repeated candidates

Each pair below was rerun with seed 29, 5 warmups and 20 timed repetitions after the 8-tile search (seed 17) and its holdout (seed 18). Baseline and candidate runs are interleaved. **Candidate latency includes online activation quantization and the GEMM**, with offline weight quantization excluded; all outputs are BF16.

| Component (M=4096) | MLX BF16 ms | Tuned INT8 online ms | Speedup | Total relative L2 | Tile M/N/groups/swizzle |
|---|---:|---:|---:|---:|---|
| h3.attn_q.m4096 | 5.185 | 3.489 | 1.49× | 1.27% | 128 / 64 / 4 / 8 |
| h3.ffn_up.m4096 | 10.135 | 6.582 | 1.54× | 1.32% | 128 / 64 / 4 / 8 |
| h3.ffn_down.m4096 | 13.402 | 7.998 | 1.68× | 1.33% | 128 / 128 / 8 / 8 |

INT8 means per-token activation / per-output-channel weight scaling, INT32 TensorOps accumulation and a fused FP32 scale epilogue. This is W8A8, not weight-only compression. Numerical errors are against float64 dot products of the original BF16 operands, sampled over rows/columns with full K. They are **not** denoiser output quality scores.

BF16 tuning improves FFN-down from 13.153 to 11.068 ms (1.19×), but Q and FFN-up are essentially tied with MLX in this rerun. FP8 improves FFN-down from 13.794 to 10.647 ms (1.30×); Q and FFN-up show no useful gain. FP8 executes on this stack, but is not the preferred path for these three large projections. Small-shape results are more sensitive to host overhead and require separate deployment validation.

Evidence: [independent rerun](../results/published/2026-10-18-independent-validation.json), [Q sweep](../results/published/2026-10-18-attn-q-tune.json), [FFN-up sweep](../results/published/2026-10-18-ffn-up-tune.json), [FFN-down sweep](../results/published/2026-10-18-ffn-down-tune.json). Raw JSON retains every candidate, individual sample, error metric and code/environment provenance.

## Outlier check changes the precision decision

The repeated outlier campaign multiplies periodically selected activation/weight features by 20. All implementations remain numerically correct against their decoded operands, but total quantization error changes substantially:

| Component (M=4096) | INT8 total relative L2 | FP8 total relative L2 | INT8 5% quality gate |
|---|---:|---:|---|
| Q projection | 4.69% | 3.49% | pass, near limit |
| FFN-up | 4.95% | 3.43% | pass, near limit |
| FFN-down | **8.84%** | 3.78% | **reject** |

The harness retains this rejected result and does not produce a speedup recommendation for it. This is a concrete reason to preserve BF16/FP8 options and collect real H3 activations before deployment. The normal-input INT8 winners above remain experimental candidates. [Outlier evidence](../results/published/2026-10-18-outliers-validation.json).

## Follow-up: Hadamard before INT8

The [Hadamard experiment](hadamard.md) keeps weight rotation/quantization offline and includes activation rotation in online latency. At seed29, plain block256 reduces FFN-down outlier relative L2 from 8.84% to **1.22%**; the independently validated complete path takes **8.701 ms** versus **12.707 ms** BF16 (1.46×). Q and FFN-up error fall to 0.70%/0.72% at 3.789/6.895 ms. These pass the same unchanged synthetic gate; real H3 activation and output-quality validation remains pending. The active comparison is INT8 + Hadamard versus unrotated FP8; FP8 + Hadamard is outside scope.

The subsequent [AdaLN/Hadamard/INT8 fusion](norm-hadamard-fusion.md) preserves the native-style BF16 AdaLN boundary and fuses three preparation stages into one dispatch. For Q4096, seed29 preparation is 0.807 → 0.440 ms and the complete projection is 3.973 → 3.634 ms; seed43 repeats 3.981 → 3.663 ms. Q9914 repeats approximately 1.10× total-path acceleration. These compare the same staged and fused INT8 policy, with full online preparation timed. The selected preparation shader reports 48 temporary registers and zero spills; this is not an end-to-end H3 inference result.

The [gated residual extension](gated-residual-fusion.md) compares standalone gate-add plus that existing fused preparation with a single fused gated preparation. Both write and retain BF16 R_next; online timing includes this output. Normal Q4096 repeats 3.942 → 3.698 ms at seed29 and 3.875 → 3.713 ms at seed43, about 4–6% lower latency. FFN-up and Q9914 gains vary, and Q9914 outliers are slightly slower in both repeats. The initial sweep's 6.507 → approximately 3.71 ms difference was not reproduced and is retained as a timing anomaly, not promoted as a stable gain. All 36 independent records pass the unchanged implementation and synthetic quality gates; full-model quality remains unmeasured.

## Follow-up: projection epilogues

Moving gated residual add into the preceding GEMM preserves the branch BF16 round, residual BF16 round and following AdaLN BF16 round. Both placements retain the updated residual and contain the same three core/five online dispatches. Two independent seeds with reversed order give M4096 attention-to-FFN online 10.846 → 10.710 / 10.670 → 10.619 ms, and FFN-to-Q 12.723 → 12.426 / 12.488 → 12.096 ms. Large-shape reductions range from 0.3–4.6%; small-M timing is unstable. All 48 final records pass, with exact paired residual/codes/scales and 0.67–1.37% total chain relative L2. [Chain results](gate-epilogue-fusion.md).

The independent Q/K projection/RMSNorm/RoPE kernel preserves native H3 BF16 projection/output boundaries, rotates 96 of 128 channels per head and writes head-major BF16 directly. With common online H256/INT8 preparation, M4096 normal latency is 3.977 → 3.460 ms at seed29 and 4.012 → 3.456 ms at seed43; M9914 is 9.659 → 7.959 and 9.765 → 7.944 ms. Normal/outlier repeats support 13–14% lower online latency at M4096 and about 18% at M9914, with exact paired Q/K outputs and 1.11–1.31% total relative L2. M128 shows no repeatable win. The pinned native runtime already fuses grouped QKV/RMS/RoPE; this experiment compares our single-projection fused and staged implementations, not upstream runtime throughput. [Q/K results](qk-rope-fusion.md).

## Trace findings

The new gated-chain captures both contain five online dispatches; combined GEMM cost is 92.25% / 92.94% before/after. The gate epilogue retains 112 temporary registers and 272 reported spilled bytes, while the transition drops from 52 to 48 registers. Q/K fusion changes four online dispatches to three; its projection/RMS/RoPE shader accounts for 88.96% of replay cost, with 116 registers and 272 reported spilled bytes versus the staged GEMM's 112/272. The projection remains the main measured cost. [Gate counters](gate-epilogue-fusion.md), [Q/K counters and limitations](qk-rope-fusion.md). Replay durations are not mixed with benchmark timings.

Both FFN-down GPU captures contain **one activation-quantization dispatch and one GEMM dispatch**, verified from shader invocation counts. The standalone records are [before](../results/published/2026-10-18-ffn-int8-before-profile.json) and [tuned](../results/published/2026-10-18-ffn-int8-tuned-profile.json).

| GPU debugger metric | Initial 64/64/4/1 | Tuned 128/128/8/8 |
|---|---:|---:|
| Serial replay GPU duration | 13.233 ms | 10.309 ms |
| GEMM share of shader cost | 96.20% | 95.05% |
| Quantizer share | 3.80% | 4.95% |
| GEMM temporary registers | 72 | 114 |
| GEMM spilled bytes (tool report) | 144 | 272 |

The dominant recorded cost is GEMM, so tiling/traversal is the first optimization target for this shape. The faster tile spills **more**, illustrating why register/spill counts alone do not establish a bottleneck. Larger tiles and changed traversal improve measured throughput here, but cache/traffic causality has not been isolated. These replay durations include profiling overhead and are not substituted into the benchmark speedup table.

An actual Instruments Metal System Trace also contains 1,581 target-process GPU intervals. After unioning nested/duplicate intervals, GPU activity spans 4,886.87 ms of a 5,185.01 ms observed launch window (94.25%). This includes setup and warmup; it is not a hot-loop-only utilization number. See the [sanitized summary](../results/published/2026-10-18-instruments-summary.json) and [capture/replay instructions](profiling.md). Raw binary traces remain local.

## Interpretation and remaining coverage

The MLX BF16 baseline covers all 55 cases in `h3`: projection alternatives, SDPA, RMS modulation and SwiGLU at synthetic token counts 128/1024/4096, plus small time/AdaLN projections. The native precision sweep covers all 46 projection cases in BF16 and INT8, and 40 in FP8; 6 FP8 input-projection shapes explicitly require a future padding adapter. The FP16 sweep also covers all 55 MLX cases and 46 MPP GEMMs. The LLM campaign covers 15 Qwen3-VL-32B projection shapes (decode/prefill). Other workload records are indexed in [results/published](../results/published).

MLX BF16 SDPA takes 38.89 / 167.86 / 742.99 ms at sequence lengths 8192 / 16384 / 32768 (56 heads, dimension 128; 5 timed samples). QDQ attention error at S=4096 is 1.17% INT8, 4.53% FP8 and 1.51% MXINT8; those timings are BF16 SDPA after QDQ preprocessing and cannot demonstrate a low-bit attention speedup.

Native scaled MXINT8→FP32 TensorOps did not compile through the tested API path. `mx-emulated` measures block-32 MXINT8 numerical effects by restoring BF16 before matmul; it is not native compressed execution. A separate outlier run (seed 17) gives 1.61% MXINT8 error for FFN-down, with 18.47 ms online latency through the emulated path. Low-bit attention currently uses QDQ sensitivity with BF16 SDPA, not a faster INT8/FP8 attention implementation. FP8 input rows require K divisible by 128; video/audio input projections requiring padding are explicitly unsupported.

The ~60 TFLOP/s FP16 and ~110 effective TOPS INT8 numbers in [Draw Things’ report](https://releases.drawthings.ai/p/metal-quantized-attention-pulling) use different kernels, fusion and measurement conditions. They are context, not matched pass/fail targets. No claim of SOTA is made until a pinned MFA comparison is run on matched workloads.

Real H3 generation, real activation calibration, fused low-bit attention, visual/audio VAE and full encoder execution remain outside this operator campaign. Q/K projection with normalization/RoPE now has an independent component experiment; grouped QKV and attention integration remain unmeasured. The [coverage ledger](h3-coverage.md) makes these gaps explicit. Do not infer video latency, memory fit or perceptual quality from these microbenchmarks.
