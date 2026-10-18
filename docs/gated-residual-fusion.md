# Gated residual + AdaLN + Hadamard + INT8 fusion

The gated variant extends the existing fused AdaLN/Hadamard/INT8 preparation with the preceding residual update. It returns **both packed activations and the updated residual**. On the repeated synthetic Q4096 workload, the complete INT8 projection improves by about 4–6% over a separate gate-add dispatch followed by the existing fused preparation. Other shapes show smaller or inconsistent gains; Q9914 outliers show no benefit.

## Contract and comparison

The measured native-style BF16 contract is:

```
R_next = BF16(float32(R) + float32(gate) * float32(branch))
inv = rsqrt(mean(float32(R_next)^2) + epsilon)
y = BF16((float32(R_next) * inv * gamma) * (1 + scale) + shift)
z = float32(y) D H256
s = max(abs(z)) / 127                 # over the entire K row
q = clamp(round_to_even(z / s), -127, 127)
```

Zero rotated rows use q=0 and scale 1. The gate product has no intermediate BF16 round. The residual addition is rounded to BF16 before the next normalization, and the completed modulation is separately rounded to BF16 before rotation. The implementation may contract FP32 multiply/add; this is not a byte-parity claim with eager framework expressions that introduce additional casts.

The gate belongs to the preceding branch. Gamma, scale and shift belong to the upcoming normalization/modulation and remain separate operands. The API supports a gate table with its own `gate_row_map` and a modulation table with an independent `row_map`; the measurements here use shared per-channel vectors. Both maps are caller-validated int32 indices into their respective tables. Weight rotation uses the same D/H as activations and remains offline, together with weight quantization.

The fair INT8 comparison is:

| Backend | Online preparation | Outputs retained |
|---|---|---|
| `residual-separate` | Standalone gate-add, then the **previous fused** norm/modulation/Hadamard/quantizer | Q, row scales and BF16 R_next |
| `residual-fused` | Gate-add through quantization in one preparation dispatch | Q, row scales and BF16 R_next |
| `mlx` / BF16 | Metal gate-add and norm/modulation, then MLX BF16 GEMM | BF16 projection and R_next |

The BF16 comparator is therefore not an entirely MLX pipeline. Both INT8 candidates use the same GEMM and preserve the residual needed by a later skip connection. `activation_prepare` includes writing R_next; `online_activation_plus_gemm` includes preparation, that residual output and GEMM. The kernel-only `timing` uses already prepared operands and an already materialized residual, so it does not represent the gated online path.

The harness computes an independent float64 residual expression from the actual BF16 inputs, applies the explicit FP32/BF16 boundary and requires an exact match on sampled rows with full K. It checks all output values for finiteness. Small GPU tests cover full tensors, tails, gate-zero/unit cases and independent row maps. Normalization, weight rotation, quantization and GEMM have separate references: ideal quantization error, actual preparation error and GEMM implementation error are distinct records.

## Repeated measurements — 2026-10-18

Hardware is the same 40-core M5 Max used by the earlier campaigns. The selected configuration is RMS, H256, 256 threads, `one_read`, the BF16 boundaries above, and INT8 GEMM tile `(128,64,4,8)`. Each independent measurement uses five warmups and 30 synchronized host-wall repetitions. The table gives complete online medians, including activation preparation and the updated residual output. Each cell compares separate → fused in milliseconds.

| Projection | Synthetic distribution | Seed 29: separate → fused ms | Seed 43: separate → fused ms |
|---|---|---:|---:|
| Q, M=4096 | normal | 3.942 → 3.698 | 3.875 → 3.713 |
| Q, M=4096 | outliers | 3.834 → 3.718 | 3.868 → 3.731 |
| FFN-up, M=4096 | normal | 7.170 → 7.005 | 7.124 → 6.795 |
| FFN-up, M=4096 | outliers | 6.943 → 6.992 | 6.934 → 6.817 |
| Q, M=9914 | normal | 9.135 → 9.096 | 9.340 → 9.004 |
| Q, M=9914 | outliers | 9.146 → 9.155 | 9.142 → 9.206 |

Q4096 normal repeats a 1.066× / 1.044× online speedup. Its preparation alone changes from 0.777 → 0.692 ms and 0.739 → 0.678 ms. FFN-up normal improves in both seeds, but FFN-up outliers alternate between a small slowdown and a small gain. Q9914 normal varies from approximately tied to a modest gain, while its outlier case is slightly slower in both repeats. Fusion is not a universal dispatch recommendation.

The initial seed17 normal sweep recorded 6.507 ms for `residual-separate` and approximately 3.71 ms for selected fused variants. That large difference did **not** reproduce: the independent separate baselines are 3.94/3.87 ms. The initial record is retained as a timing anomaly; it does not support a stable 6.5 → 3.7 ms fusion claim. No hardware cause is inferred from those timings alone.

All 36 records in the two independent campaigns pass their implementation and synthetic quality gates. INT8 total relative L2 ranges from 1.21–1.33% for normal inputs and 0.65–0.94% for the synthetic outliers in this table. Separate and fused records have matching reported errors here, and all INT8 residual checks match the independently rounded reference. These are errors for this synthetic gated expression, not measured H3 output quality or calibration on model activations.

M9914 is a workload scenario matching the separately recorded integration request's packed token count; values, weights, gate and modulation remain synthetic. Shared-vector RMS performance has been measured. Mapped gate/modulation performance, full-block integration and generated audiovisual quality are not established by this campaign. In particular, this work does not move normalization across SwiGLU or replace the FFN-down input contract.

Evidence: [seed17 configuration sweep](../results/published/2026-10-18-gated-norm-sweep.json), [seed29 independent comparison](../results/published/2026-10-18-gated-norm-independent.json), [seed43 rerun](../results/published/2026-10-18-gated-norm-rerun.json). These immutable records retain raw samples, source digests, dirty revision state, input seeds, settings and numerical metrics.

## GPU trace evidence

The Q4096 [separate-path capture](../results/published/2026-10-18-gated-norm-separate-profile.json) contains three dispatches: gate addition, the existing norm/Hadamard/quantizer, and GEMM. The [fused-path capture](../results/published/2026-10-18-gated-norm-fused-profile.json) contains two: gated preparation and GEMM. Each recorded shader is invoked once, and both captured workloads retain the updated BF16 residual output.

| Path | Shader | Temporary registers | Spilled bytes | Serial-replay shader cost |
|---|---|---:|---:|---:|
| Separate | Gate addition | 7 | 0 | 3.83% |
| Separate | Norm/Hadamard/quantizer | 48 | 0 | 8.37% |
| Separate | GEMM | 112 | 272 | 87.80% |
| Fused | Gated norm/Hadamard/quantizer | 52 | 0 | 10.14% |
| Fused | GEMM | 112 | 272 | 89.86% |

Folding gate addition into preparation increases its temporary-register count from 48 to 52 without introducing reported spills. GEMM remains the dominant recorded cost, consistent with a modest complete-projection benefit rather than the initial sweep's large apparent improvement. Cost percentages are normalized within each serial GPU replay and include profiling overhead; they are neither benchmark-wall fractions nor standalone latency comparisons. The counters do not establish a specific cache or bandwidth cause. The 272 spilled bytes belong to GEMM, not preparation.

Raw captures remain local under `results/local/gated-norm-{separate,fused}-001.gputrace`. Published JSON preserves invocation counts and resource counters without redistributing the binary captures. See [profiling.md](profiling.md) for capture/replay commands; use `h3-gated-norm-hadamard`, case `h3.gated_q.m4096`, backend `residual-separate` or `residual-fused`, and `--online` to include residual preparation.

## Reproduce

```sh
uv sync --locked
uv run python scripts/norm_hadamard_campaign.py --gated \
  --cases h3.gated_q.m4096 --seed 17 \
  --output results/local/gated-norm-sweep-new.json
uv run python scripts/norm_hadamard_campaign.py --gated \
  --cases h3.gated_q.m4096,h3.gated_ffn_up.m4096,h3.gated_q.m9914 \
  --threads 256 --passes one_read --seed 29 --repeat 30 \
  --output results/local/gated-norm-independent-new.json
uv run python scripts/norm_hadamard_campaign.py --gated \
  --cases h3.gated_q.m4096,h3.gated_ffn_up.m4096,h3.gated_q.m9914 \
  --threads 256 --passes one_read --seed 43 --repeat 30 \
  --output results/local/gated-norm-rerun-new.json
```

Run GPU experiments serially and use fresh output filenames. The workload operation is `gated_norm_hadamard_gemm` in `h3-gated-norm-hadamard`; its `fusion` configuration records the normalization, epsilon, block size, rounding boundary, thread count, read strategy and optional sign seed. See the [underlying AdaLN fusion](norm-hadamard-fusion.md) for block ownership and reductions. Source-level dispatch/read counts alone do not prove a device bandwidth or cache benefit.
