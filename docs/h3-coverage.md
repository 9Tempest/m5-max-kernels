# H3 coverage ledger

Pinned official configuration revision: `42ed227ee7df40d41602854ae760620d6eb651fe`. Projection semantics were checked against the pinned Diffusers transformer source linked in sources/README.md. This ledger distinguishes **operator microbenchmarks** from a complete H3 deployment.

| Component | Shape / semantics | Coverage |
|---|---|---|
| DiT Q/K/V | [T,5376] × [7168,5376]ᵀ each, no bias | MLX and custom MPP; fused QKV alternative also listed |
| DiT attention | B=1, H=56, D=128, noncausal packed sequence | MLX fused SDPA; low-bit QDQ sensitivity; native low-bit attention pending |
| Attention output | [T,7168] × [5376,7168]ᵀ | MLX and custom MPP |
| SwiGLU gate / up | [T,5376] × [14336,5376]ᵀ each | Separate and fused-projection alternatives; compiled activation baseline |
| FFN down | [T,14336] × [5376,14336]ᵀ | MLX and custom MPP |
| Pre-norm / AdaLN application | Width 5376, FP32 RMS reduction | Compiled MLX and fused Metal; one shared modulation vector microbenchmark |
| Q/K normalization | Per-head RMS width 128 | Fused RMS kernel supports width, but isolated QK norm/RoPE composition not benchmarked |
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
