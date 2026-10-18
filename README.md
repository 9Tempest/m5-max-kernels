# M5 Max Kernels

A reproducible kernel laboratory for Apple Silicon: measure latency **and numerical error**, keep raw evidence, and promote kernels only on the shapes where they win. The first target is MiniMax H3; the same harness includes LLM projection workloads.

Initial machine: **Apple M5 Max, 40 GPU cores, 128 GiB unified memory**, macOS 27 / Xcode 27.1, MLX 0.32.4. This is an experimental kernel repository, not a claim of universal state-of-the-art performance or an end-to-end H3 runtime.

## First measured outcome

At M=4096, tuned W8A8 projections take **3.49 ms Q**, **6.58 ms FFN-up**, and **8.00 ms FFN-down**, including online activation quantization: **1.49× / 1.54× / 1.68×** faster than interleaved MLX BF16 baselines. Normal-input total relative L2 is about 1.3%. **FFN-down INT8 fails the synthetic outlier quality gate (8.84% error)**; FP8 passes that check (3.78%). These are kernel experiments, not a validated model precision policy.

Follow-up: **Hadamard before INT8 reduces FFN-down outlier error from 8.84% to 1.22%**, with 8.70 ms total online latency versus 12.71 ms BF16 (1.46×). Weights are rotated offline; activation rotation is included. See the [Hadamard experiment](docs/hadamard.md).

Further fusion combines **AdaLN + Hadamard + dynamic INT8 quantization** in one Metal dispatch. Q-projection preparation at M=4096 falls from 0.81 to 0.44 ms; the complete projection improves about 9% over the staged INT8 path in two independent seeds. [Fusion experiment and trace](docs/norm-hadamard-fusion.md).

Adding the preceding **gated residual update** to that fused preparation lowers Q4096 online latency by another 4–6% versus a separate gate-add dispatch, while preserving the updated BF16 residual output. The benefit varies by shape; Q9914 outliers show no improvement. [Repeated gated-fusion results](docs/gated-residual-fusion.md).

Moving the gate further into the **preceding GEMM epilogue** preserves the BF16 branch/residual boundaries and yields a smaller 0.3–4.6% reduction across repeated large-shape two-projection chains. [Complete-chain experiment](docs/gate-epilogue-fusion.md). Independently, **Q/K projection + RMSNorm + RoPE** fusion reduces complete online latency by 13–14% at M4096 and about 18% at M9914, with exact paired outputs. This single-Q/K comparison includes input packing and head-major output; the upstream native runtime already has grouped QKV fusion. [Q/K results and contract](docs/qk-rope-fusion.md).

The repository contains raw baseline/precision/tuning/holdout results, real before/after GPU counter reports and a target-filtered Instruments trace summary. See [results and limitations](docs/initial-results.md).

RMS factors can be folded into INT8 dequant scales, including after Hadamard. The [scale-folding experiment](docs/rms-scale-folding.md) reduces synchronization but finds no stable overall speedup here. [Moving AdaLN shift into a projected output bias](docs/affine-shift-splitting.md) is also implemented as an alternate precision policy, with conditioning costs and negative results retained.

## Run

Install [uv](https://docs.astral.sh/uv/), Xcode and the Metal toolchain. From a source checkout:

```sh
uv sync --locked
uv run m5bench doctor
uv run pytest -q
uv run m5bench bench --suite smoke --filter gemm --backends mlx,mpp \
  --precisions bf16,int8,fp8 --output results/local/smoke-001.json
uv run m5bench bench --suite h3 --backends mlx --precisions bf16 \
  --output results/local/h3-baseline-001.json
uv run m5bench tune --suite h3 --filter ffn_down.m4096 --precisions bf16,int8,fp8 \
  --output results/local/ffn-tune-001.json
uv run m5bench report results/local/ffn-tune-001.json
```

Output paths are exclusive: choose a new filename for each experiment. Performance measurements require a real Apple Silicon GPU; CPU CI checks references, manifests and result handling. Run one GPU experiment at a time. Close competing GPU workloads for publishable measurements.

## What is implemented

| Operation | Baselines / experiments | Contract |
|---|---|---|
| GEMM / linear projections | MLX; custom Metal TensorOps tiled GEMM | BF16, FP16, FP32; explicit tail handling |
| W8A8 GEMM | Native INT8 TensorOps + INT32 accumulation + fused scale epilogue | Symmetric per-token activations, per-output-channel weights; BF16 output |
| FP8 GEMM | E4M3 TensorOps + FP32 accumulator + fused scale epilogue | Per-token/channel scaling; K/row stride divisible by 128 |
| Hadamard + INT8 GEMM | Custom Metal or MLX block rotation + existing TensorOps | Identical normalized rotation for A/W; offline W, online A; FP32 transform; FP8 is a separate candidate |
| MXINT8 | 32-element quantizer + explicit software dequantization | S1.6 factor folded into E8M0 scale; **not native MX acceleration** |
| Attention | MLX fused SDPA; INT8/FP8/MXINT8 QDQ sensitivity experiment | QDQ runs BF16 SDPA; its timing is **not quantized attention throughput** |
| RMSNorm + modulation | Compiled MLX; custom fused Metal | FP32 reduction, BF16/FP16/FP32 output |
| AdaLN + Hadamard + INT8 packing | One Metal dispatch; staged comparison | RMS/LayerNorm, explicit post-modulation rounding, whole-row scales, optional modulation row map |
| Gated residual + AdaLN + Hadamard + INT8 | Fused preparation versus separate gate + prior fused preparation | Preserves and outputs BF16 residual before next norm; independent gate/modulation maps; online output cost included |
| SwiGLU | Compiled MLX | Baseline for fusion experiments |
| RMS + INT8 scale folding | Direct/folded Metal, optional H, 128/256/512 threads | Zero shift, FP32 intermediate; norm factor enters row scale |
| Projected AdaLN shift | FP32 bias in INT8 GEMM epilogue | Shared vector or mapped table; explicit rounding-policy change and conditioning cost |
| Gated GEMM epilogue | Gate in preceding projection versus gate in following norm | Preserves BF16 branch/residual boundaries; both projections and next preparation timed |
| Q/K projection + RMSNorm + RoPE | Fused Metal versus GEMM plus standalone postprocess | Complete 128-wide heads; BF16 projection/output, FP32 normalization/rotation, direct head-major output |

Native scaled INT8→FP32 TensorOps is not working in the tested stack. FP8 execution support alone does not establish a dedicated FP8 hardware speedup. FP32 TensorOps may use reduced-mantissa multiplication: the strict FP32 correctness gate must pass before promotion.

For GPU counters, shader resource usage and CPU/GPU timelines, see the executable [profiling workflow](docs/profiling.md). Both Metal captures and an Instruments trace have been collected on the target machine.

See [initial measured results](docs/initial-results.md), [methodology](docs/methodology.md), [H3 coverage](docs/h3-coverage.md), and [precision contracts](docs/precision.md). Raw records live in [results/published](results/published).

## Workload coverage

`workloads/h3.json` is generated from pinned official configs: hidden width **5376**, attention width **7168 = 56 × 128**, SwiGLU width **14336**, 50 transformer layers and 2 text refiner layers. It includes individual and fused projections, attention, modulation, input/output projections and small precomputable AdaLN branches. Individual/fused alternatives must never be summed together as a model latency estimate.

Token counts 128 / 1024 / 4096 are synthetic microbenchmark scenarios. `h3-long` extends attention to 8192 / 16384 / 32768; `llm` covers Qwen3-VL-32B decode/prefill projections. A real H3 packed sequence depends on resolution, duration, prompt and conditioning. These are not traced generation shapes. See the coverage ledger for VAE, multimodal encoder and full-generation work still needed.

## Extend and maintain

Add a versioned JSON workload or pass `--suite /path/to/workload.json`. Follow [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) for new backends, precision modes and promotion evidence. Source revisions and config hashes are pinned in [sources/manifest.json](sources/manifest.json). Dependencies are locked in `uv.lock`; CI checks CPU correctness and formatting, and Dependabot proposes dependency updates.

Original code is MIT licensed. Upstream model configuration snapshots retain their original terms; see [sources/README.md](sources/README.md). Model weights, private data and credentials do not belong in this repository.
