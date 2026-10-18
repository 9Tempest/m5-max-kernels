# Moving the gated residual update into the preceding GEMM

The `gate-in-gemm` candidate moves the gated residual update from the following AdaLN/Hadamard/INT8 preparation into the preceding GEMM epilogue. It preserves the BF16 branch and residual boundaries and returns the updated residual needed by later skip connections. The complete chain still contains **three core dispatches and five online dispatches**. The intended saving is an intermediate-buffer access, rather than a reduction in dispatch count.

For the INT8 projection, the contract is:

```text
branch = BF16(float32(INT32_dot) * activation_scale * weight_scale)
R_next = BF16(float32(R) + float32(branch) * float32(gate))
inv = rsqrt(mean(float32(R_next)^2) + epsilon)
modulated = BF16((float32(R_next) * inv * gamma) * (1 + scale) + shift)
rotated = float32(modulated) H256
q, s = symmetric_per_token_INT8(rotated)
output = BF16(INT8_GEMM(q, next_weight) * s * next_weight_scale)
```

The branch rounds to BF16 **before** its gate product, even when it exists only in epilogue registers. The gate product has no intermediate BF16 round. The residual addition rounds to BF16 before RMS, and completed AdaLN rounds to BF16 before H256. The RMS denominator covers the unweighted updated residual. INT8 scales cover the entire rotated row; a zero row uses codes zero and scale one. These are native-style operation boundaries, with implementation error checked separately from quantization error.

`metal.gemm` accepts BF16 `residual[M,N]` and either a shared BF16 `gate[N]` or a BF16 gate table selected by an independent int32 row map. It requires explicit BF16 output and supports INT8 or BF16 projection. A simultaneous GEMM bias and gated update is explicitly unsupported. Inputs are not mutated; the returned matrix is `R_next`. The measurements below use INT8 and shared gates. Upcoming gamma, scale and shift remain separate from the preceding gate. Mapped-table performance and full H3 integration are not established here.

The two measured paths retain the same final projection and updated residual:

| Stage | `gate-in-norm` baseline | `gate-in-gemm` candidate |
|---|---|---|
| Preceding projection | GEMM writes BF16 branch | GEMM locally rounds the branch, applies gate/residual and writes BF16 `R_next` |
| Transition | Fused gate + AdaLN + H256 + INT8; writes `R_next` | Existing AdaLN + H256 + INT8 reads `R_next` |
| Next projection | INT8 GEMM | Same INT8 GEMM |
| Online input preparation | H256 followed by INT8 packing | Same H256 and INT8 packing |

`timing` includes both projections and the transition with resident preceding INT8 activations. `online_activation_plus_gemm` additionally includes preceding activation rotation and packing. Both timings materialize the final projection and `R_next`. The isolated `previous_projection`, `transition` and `next_projection` stages diagnose where time moves; adding their independently measured medians is not a substitute for timing the complete chain. Both weight rotations and quantizers are recorded as offline preparation and excluded from online latency.

The harness constructs both complete paths from identical inputs and requires exact full-tensor agreement for updated residuals, packed transition activations and their scales. Independent NumPy float64 references cover the preceding projection, its BF16 branch and residual boundaries, AdaLN/Hadamard/quantization, and the next projection. References sample rows and final output columns while retaining the full contraction dimensions. The next-GEMM implementation target uses actual decoded packed operands; the original-operand target measures complete-chain quantization effects separately. All materialized outputs are checked for finiteness. These checks establish the synthetic operation contract, not generated-video quality.

For the current `one_read` transition, the affected BF16 buffer accesses per residual element are:

| Path | Affected buffer accesses | Logical bytes per element |
|---|---|---:|
| Gate in normalization | Branch store + branch read + old residual read + updated residual store | 8 |
| Gate in GEMM | Old residual read + updated residual store + updated residual read | 6 |

The candidate eliminates the branch buffer, but the following preparation must now read the updated residual instead of retaining it from its own gate computation. The net reduction is **2 bytes per residual element**: at M=4096 and width 5376, this is 44,040,192 bytes, or 44.04 MB decimal / 42 MiB. This accounting excludes common GEMM operands, gate/modulation parameters, packed outputs, scales, cache effects and spills. It is a logical source-level traffic comparison; it does not measure DRAM bytes or establish a bandwidth speedup.

The initial local pilot, `results/local/gate-epilogue-m4096-s29.json`, contains eight M4096 records for attention-to-FFN and FFN-to-Q chains, normal/outlier inputs and both gate placements. All eight pass implementation checks and report exact paired residual/code/scale parity. Both projections used the uniform `(128,64,4,8)` tile. This is an initial, unpromoted experiment: it does not establish the best component configurations or a repeatable performance gain. The subsequent `gate-epilogue-final*.json` campaigns use the selected preceding and following projection tiles and independent seeds, as reported below. Raw pilot records remain unchanged.

## Repeated complete-chain measurements — 2026-10-18

The final campaigns use 30 warmups and 40 synchronized host-wall samples per stage, with seed 29 followed by an independent seed 43 run in reversed candidate order. The previous FFN-down uses tile `(128,128,8,8)`; previous attention-output and following projections use `(128,64,4,8)`. Both placements use the same component tiles. The table reports the complete online path, including initial H256/INT8, both GEMMs, transition and the retained residual; offline weight preparation is excluded.

| Chain | Distribution | Seed29 baseline → epilogue ms | Latency reduction | Seed43 baseline → epilogue ms | Latency reduction |
|---|---|---:|---:|---:|---:|
| attention_to_ffn, M=4096 | normal | 10.846 → 10.710 | 1.3% | 10.670 → 10.619 | 0.5% |
| attention_to_ffn, M=4096 | outliers | 10.766 → 10.626 | 1.3% | 10.683 → 10.615 | 0.6% |
| ffn_to_q, M=4096 | normal | 12.723 → 12.426 | 2.3% | 12.488 → 12.096 | 3.1% |
| ffn_to_q, M=4096 | outliers | 12.823 → 12.227 | 4.6% | 12.500 → 12.147 | 2.8% |
| attention_to_ffn, M=9914 | normal | 26.094 → 25.746 | 1.3% | 26.252 → 25.667 | 2.2% |
| attention_to_ffn, M=9914 | outliers | 26.099 → 25.855 | 0.9% | 25.973 → 25.895 | 0.3% |
| ffn_to_q, M=9914 | normal | 30.490 → 29.554 | 3.1% | 30.337 → 29.112 | 4.0% |
| ffn_to_q, M=9914 | outliers | 30.070 → 29.446 | 2.1% | 30.681 → 29.628 | 3.4% |

The large-shape gains are small: attention-to-FFN is about 0.3–2.2% lower latency and FFN-to-Q about 2.1–4.6%. The M128 data have substantial host-wall variability, including unreproduced large differences and occasional regressions; no small-shape dispatch policy is promoted. These named chains cover one preceding and one following projection, not a whole attention/FFN block or denoiser.

All 48 final records pass the unchanged implementation and synthetic 5% quality gates. Full-buffer residuals, transition codes and transition scales are identical between placements in every record. Thus the moved gate introduces no difference in these paired outputs; overall quantization error still exists. Sampled final-output total relative L2 is 0.67–1.37%, with implementation error at most 0.18%. Zeros, tails, independent maps, exact rounding-boundary counterexamples and finite outputs have separate tests. Independent activation/weight preparation checks participate in correctness, not just diagnostics.

Evidence: [seed29](../results/published/2026-10-18-gate-epilogue-final-s29.json), [seed43 reversed](../results/published/2026-10-18-gate-epilogue-final-s43.json), [uniform-tile pilot](../results/published/2026-10-18-gate-epilogue-m4096-s29.json). Every raw sample is retained, including the unstable small-M timings.

The [baseline profile](../results/published/2026-10-18-gate-epilogue-baseline-profile.json) and [epilogue profile](../results/published/2026-10-18-gate-epilogue-fused-profile.json) capture the attention-to-FFN M4096 online chain. Each contains five workload dispatches: input rotation, input quantization, two GEMMs and one transition. The baseline has five encoders and the candidate four; encoder grouping is distinct from dispatch count.

| Recorded component | Baseline replay cost | Candidate replay cost | Baseline → candidate temporary registers | Spilled bytes |
|---|---:|---:|---:|---:|
| Both GEMMs, combined | 92.25% | 92.94% | 112 → 112 per shader | 272 per GEMM shader |
| Transition preparation | 3.58% | 2.93% | 52 → 48 | 0 |
| Initial H256 | 2.34% | 2.32% | 7 → 7 | 0 |
| Initial INT8 packing | 1.83% | 1.82% | 16 → 16 | 0 |

The baseline's ordinary GEMM shader is invoked twice. The candidate separates that cost into a gated preceding GEMM at 31.80% and an ordinary next GEMM at 61.14%. Gate placement reduces the transition's reported register count without introducing transition spills; both GEMM variants still report 112 temporary registers and 272 spilled bytes. The combined GEMM cost remains dominant. Static instruction/load/store counters do not translate directly into dynamic memory transactions or DRAM bandwidth.

The captured serial-replay frame durations are **15.6874 ms → 15.5392 ms**. These are profiler replay measurements with profiling overhead and default GPU state, **not synchronized benchmark medians**. Cost percentages are normalized separately within each replay and cannot establish a cross-path latency improvement. The captures support resource and dispatch analysis; promotion depends on the independent benchmark campaigns.

Raw binary captures remain local at `results/local/gate-epilogue-{baseline,fused}.gputrace`; their capture sidecars and `results/local/gate-epilogue-{baseline,fused}-profile.json` preserve settings and counters. The commands below use fresh output names and run serially:

```sh
uv sync --locked
uv run python scripts/gate_epilogue_campaign.py --filter m4096 \
  --seed 29 --warmup 30 --repeat 40 \
  --output results/local/gate-epilogue-final-new-s29.json
uv run python scripts/gate_epilogue_campaign.py --filter m4096 \
  --seed 43 --warmup 30 --repeat 40 --reverse \
  --output results/local/gate-epilogue-final-new-s43.json

uv run python -m m5kernels.profile --kind gpu --suite h3-gate-epilogue \
  --case h3.chain_attention_to_ffn.m4096 --backend gate-in-norm \
  --precision int8 --tile 128 64 4 8 --online \
  --output results/local/gate-epilogue-baseline-new.gputrace
uv run python scripts/profile_report.py results/local/gate-epilogue-baseline-new.gputrace \
  --expect-dispatches 5 --output results/local/gate-epilogue-baseline-new-profile.json
uv run python -m m5kernels.profile --kind gpu --suite h3-gate-epilogue \
  --case h3.chain_attention_to_ffn.m4096 --backend gate-in-gemm \
  --precision int8 --tile 128 64 4 8 --online \
  --output results/local/gate-epilogue-fused-new.gputrace
uv run python scripts/profile_report.py results/local/gate-epilogue-fused-new.gputrace \
  --expect-dispatches 5 --output results/local/gate-epilogue-fused-new-profile.json
```

The generated workload records `previous_tile` separately from the following projection's `tile`; inspect both in a campaign record. The component configurations for the final campaigns use `(128,128,8,8)` for a preceding FFN-down projection, `(128,64,4,8)` for a preceding attention-output projection, and `(128,64,4,8)` for the next projection. The attention-to-FFN profile above uses `(128,64,4,8)` for both. Use an empty filter (`--filter ''`) to include all generated M128/M4096/M9914 scenarios. To publish the existing reviewed counter exports without replacing prior evidence:

```sh
cp -n results/local/gate-epilogue-baseline-profile.json \
  results/published/2026-10-18-gate-epilogue-baseline-profile.json
cp -n results/local/gate-epilogue-fused-profile.json \
  results/published/2026-10-18-gate-epilogue-fused-profile.json
```

See [gated residual preparation](gated-residual-fusion.md) for the preceding fusion baseline and [profiling](profiling.md) for capture requirements. Shared-vector synthetic measurements do not establish a full-model speedup or a new precision policy.
