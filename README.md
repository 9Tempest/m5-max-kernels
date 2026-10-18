# M5 Max Kernels

A reproducible kernel laboratory for Apple Silicon: measure latency **and numerical error**, keep raw evidence, and promote kernels only on the shapes where they win. The first target is MiniMax H3; the same harness includes LLM projection workloads.

Initial machine: **Apple M5 Max, 40 GPU cores, 128 GiB unified memory**, macOS 27 / Xcode 27.1, MLX 0.32.4. This is an experimental kernel repository, not a claim of universal state-of-the-art performance or an end-to-end H3 runtime.

## First measured outcome

At M=4096, tuned W8A8 projections take **3.49 ms Q**, **6.58 ms FFN-up**, and **8.00 ms FFN-down**, including online activation quantization: **1.49× / 1.54× / 1.68×** faster than interleaved MLX BF16 baselines. Normal-input total relative L2 is about 1.3%. **FFN-down INT8 fails the synthetic outlier quality gate (8.84% error)**; FP8 passes that check (3.78%). These are kernel experiments, not a validated model precision policy.

The repository contains raw baseline/precision/tuning/holdout results, real before/after GPU counter reports and a target-filtered Instruments trace summary. See [results and limitations](docs/initial-results.md).

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
| MXINT8 | 32-element quantizer + explicit software dequantization | S1.6 factor folded into E8M0 scale; **not native MX acceleration** |
| Attention | MLX fused SDPA; INT8/FP8/MXINT8 QDQ sensitivity experiment | QDQ runs BF16 SDPA; its timing is **not quantized attention throughput** |
| RMSNorm + modulation | Compiled MLX; custom fused Metal | FP32 reduction, BF16/FP16/FP32 output |
| SwiGLU | Compiled MLX | Baseline for fusion experiments |

Native scaled INT8→FP32 TensorOps is not working in the tested stack. FP8 execution support alone does not establish a dedicated FP8 hardware speedup. FP32 TensorOps may use reduced-mantissa multiplication: the strict FP32 correctness gate must pass before promotion.

For GPU counters, shader resource usage and CPU/GPU timelines, see the executable [profiling workflow](docs/profiling.md). Both Metal captures and an Instruments trace have been collected on the target machine.

See [initial measured results](docs/initial-results.md), [methodology](docs/methodology.md), [H3 coverage](docs/h3-coverage.md), and [precision contracts](docs/precision.md). Raw records live in [results/published](results/published).

## Workload coverage

`workloads/h3.json` is generated from pinned official configs: hidden width **5376**, attention width **7168 = 56 × 128**, SwiGLU width **14336**, 50 transformer layers and 2 text refiner layers. It includes individual and fused projections, attention, modulation, input/output projections and small precomputable AdaLN branches. Individual/fused alternatives must never be summed together as a model latency estimate.

Token counts 128 / 1024 / 4096 are synthetic microbenchmark scenarios. `h3-long` extends attention to 8192 / 16384 / 32768; `llm` covers Qwen3-VL-32B decode/prefill projections. A real H3 packed sequence depends on resolution, duration, prompt and conditioning. These are not traced generation shapes. See the coverage ledger for VAE, multimodal encoder and full-generation work still needed.

## Extend and maintain

Add a versioned JSON workload or pass `--suite /path/to/workload.json`. Follow [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) for new backends, precision modes and promotion evidence. Source revisions and config hashes are pinned in [sources/manifest.json](sources/manifest.json). Dependencies are locked in `uv.lock`; CI checks CPU correctness and formatting, and Dependabot proposes dependency updates.

Original code is MIT licensed. Upstream model configuration snapshots retain their original terms; see [sources/README.md](sources/README.md). Model weights, private data and credentials do not belong in this repository.
