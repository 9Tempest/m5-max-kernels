# H3 coverage ledger

Projection-epilogue follow-up: gated residual placement now has a full two-projection chain comparison, retaining the BF16 branch/residual/AdaLN boundaries and residual side output. Two seeds with reversed order give 0.3–4.6% lower online latency across M4096/M9914 chains; full residual/codes/scales agree exactly. Q/K projection + per-head RMS + RoPE + head-major output has a separate synthetic INT8/H256 comparison, with exact paired outputs and 13–14% / about 18% online reductions at M4096/M9914. Small-M timings do not support promotion. The native upstream runtime already fuses grouped QKV/RMS/RoPE; our single-projection measurements do not establish native runtime improvement. [Gate epilogue](gate-epilogue-fusion.md), [Q/K epilogue](qk-rope-fusion.md).

RMS algebra follow-up: zero-shift RMS/weighted RMS/H256 scale folding is implemented and measured at Q-sized M4096 across two seeds and 256/512-thread candidates. Fewer synchronizations did not establish a stable full-path speedup. Weighted RMS without H exceeds the synthetic outlier quality budget. Projecting nonzero shift into an output bias is also measured, including condition preparation, for shared P=1 only; no performance promotion. Bias-table maps have correctness coverage; actual H3 schedule-wide cache integration and model quality remain unmeasured. [Scale folding](rms-scale-folding.md), [shift splitting and cache costs](affine-shift-splitting.md).

Pinned official configuration revision: `42ed227ee7df40d41602854ae760620d6eb651fe`. Projection semantics were checked against the pinned Diffusers transformer source linked in sources/README.md. This ledger distinguishes **operator microbenchmarks** from a complete H3 deployment.

| Component | Shape / semantics | Coverage |
|---|---|---|
| DiT Q/K/V | [T,5376] × [7168,5376]ᵀ each, no bias | MLX and custom MPP; fused QKV alternative also listed |
| DiT attention | B=1, H=56, D=128, noncausal packed sequence | MLX fused SDPA; low-bit QDQ sensitivity; native low-bit attention pending |
| Attention output | [T,7168] × [5376,7168]ᵀ | MLX and custom MPP |
| SwiGLU gate / up | [T,5376] × [14336,5376]ᵀ each | Separate and fused-projection alternatives; compiled activation baseline |
| FFN down | [T,14336] × [5376,14336]ᵀ | MLX and custom MPP |
| Pre-norm / AdaLN application | Width 5376, FP32 RMS reduction | Compiled MLX and fused Metal; one shared modulation vector microbenchmark |
| Gated residual into next AdaLN | BF16 residual update, then RMS/modulation, H256 and per-token INT8 packing | Separate gate + prior fused preparation compared with fused gate/preparation; both retain updated residual; synthetic Q/FFN-up results |
| Q/K normalization and RoPE | Per-head RMS width 128; 48 split-half rotary pairs | Independent fused projection/RMS/RoPE and staged comparator, native BF16 boundaries and head-major output; single Q/K synthetic microbenchmark, grouped QKV integration pending |
| Gated preceding projection | Attention output or FFN-down into next AdaLN/H256/INT8 projection | Gate in GEMM epilogue versus gate in next preparation; both projections and residual output included |
| Text refiner | Two transformer blocks, same projection widths | Shape-equivalent GEMMs covered; block integration pending |
| Text input projection | 5120→5376 | GEMM core; bias not fused |
| Video input / output | 96↔5376 (24×1×2×2 patch) | GEMM core; patchification/bias pending |
| Audio input / output | 32↔5376 | GEMM core; bias pending |
| AdaLN generation | 2688→96768; final 2688→10752 | M=1 GEMM core; cached outside per-token path; SiLU/bias pending |
| Time embeddings | 256→5376→2688 | GEMM core; sinusoidal construction and activations pending |
| H3 encoder / Qwen3-VL-32B | Hidden 5120; Q width 8192; KV width 1024; FFN 25600 | `llm` projection suite; grouped-query attention, vision encoder and layer-50 extraction not integrated |
| Visual VAE | 3D conv encoder, 24 latents; 36-layer ViT decoder, 32×64 heads | Config pinned; convolution/decoder pipeline not benchmarked |
| Audio VAE | Stereo 32kHz, 40 latent tokens/s/channel | Config pinned; convolution/resampling pipeline not benchmarked |
| Sparse attention | Upstream initial release exposes full attention | No sparse-kernel claim |
| Context-IR / 2K regeneration | Hosted portions of official system | Outside local kernel coverage |
| Full generation | Real weights, activation distributions, scheduler and AV output | Not run; no end-to-end performance/quality claim |

`h3` contains 55 cases across T=128,1024,4096 and small AdaLN/time projections. These are unique operator shapes and alternatives, not all calls in a generation. The 50 DiT layers share dimensions but differ in values. Do not multiply a peak TOPS number into a video latency estimate.

Real packed token counts must be captured for a chosen request. As a planning approximation, video tokens scale with causal temporal latents and H/32 × W/32, and stereo audio adds about 80 tokens/s; text/reference conditioning and padding add more. We deliberately keep runtime scenario lengths explicit instead of pretending a duration uniquely determines a shape.

Next model-level milestones: (1) trace packed tensor shapes and module counts for an official request, (2) capture representative activations across denoising steps, (3) integrate pinned MFA/native low-bit attention, (4) profile encoder and VAE, (5) compare complete outputs and latency against a BF16 pipeline. Changes to this ledger must reflect measured evidence, not planned support.

First campaign evidence is now published for all 55 MLX BF16 baseline cases, 46 projection shapes in BF16/INT8/MXINT8 emulation, 40 FP8-compatible projection shapes, QDQ attention at three lengths, RMS fusion, and 15 Qwen projection shapes. Long BF16 SDPA has been measured through S=32768. The independently repeated INT8 projection wins are conditional on the input distribution: FFN-down fails the synthetic outlier quality gate. See [initial-results.md](initial-results.md).

Hadamard follow-up: identical normalized block rotations on A/W are implemented for INT8 projection experiments. Q, FFN-up and FFN-down at M=4096 have block/sign sweeps, tile tuning, a second-seed holdout and independent normal/outlier reruns. Weight rotation is offline; activation rotation is timed. INT8 + Hadamard and unrotated FP8 are separate candidates; the historical FP8 + Hadamard ablation is retained but will not be pursued. This has not been inserted into a complete H3 block or commuted through SwiGLU. [Results and contract](hadamard.md).

AdaLN fusion follow-up: RMSNorm/modulation, H256 and per-token INT8 packing now share one Metal dispatch. Shared-vector RMS microbenchmarks cover Q at M=128/4096/9914 and FFN-up at M=4096 with independent seeds; Q4096's full online path improves about 9% over staged INT8. M9914 matches a deployment-request token count, but operands remain synthetic. LayerNorm and modality/timestep row-map selection have correctness tests; mapped-table performance and full model integration are not measured here. FFN-down remains a separate post-SwiGLU path. [Fusion results and trace](norm-hadamard-fusion.md).

Gated residual follow-up: both the separate and fused INT8 paths materialize BF16 R_next before the next normalization and retain it for a later skip connection. Seed29/43 normal Q4096 online medians improve 3.942 → 3.698 ms and 3.875 → 3.713 ms over standalone gate-add plus the existing fused AdaLN/Hadamard/quantizer. Gains vary by shape and distribution; Q9914 outliers show no benefit in either rerun. Gate and upcoming modulation are independent operands/maps. Shared-vector synthetic performance and correctness do not establish mapped-table performance or model quality. [Contract and repeated evidence](gated-residual-fusion.md).
