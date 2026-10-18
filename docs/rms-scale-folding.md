# RMS factors in dynamic quantization scales

The original AdaLN/Hadamard fusion preserves the sequence of operations: reduce normalization statistics, normalize and modulate, perform H, reduce the transformed maximum, then pack INT8. It removes dispatches and intermediate memory traffic, but does not eliminate the dependency between the two reductions.

For zero-shift RMS with FP32 intermediates, the positive row normalization factor can instead enter the emitted dequantization scale. This is a separate numerical contract from H3's BF16-rounded AdaLN output.

## Algebra and implementation

For a row `x`, shared channel multiplier `c`, optional sign matrix `D` and normalized block Hadamard `H`, define:

```text
r = rsqrt(mean(x*x) + epsilon)
u = (x*c) D H
p = max(abs(u))
q = round_even(127*u/p)
S = r*p/127
```

Zero transformed rows use `q=0, S=1`. Decoding gives `q*S`, an approximation to `r*u`. Plain RMS uses `c=1` and identity `D,H`; a learned gamma or zero-shift modulation uses `c=gamma*(1+scale)`. The RMS statistic always uses original `x`, while the maximum follows the channel multiplier and H. Using `maxabs(x)` in the weighted or rotated cases would be incorrect.

The [RMSNorm paper](https://arxiv.org/abs/1910.07467) defines the normalization; the scale-factor cancellation above follows directly from positive homogeneity of symmetric dynamic quantization. FP32 evaluation order can move values across quantization midpoints, so real-number equivalence does not imply bit-identical codes.

`rms_quantize(..., fold_scale=True)` retains each thread's original-input sum of squares while it applies gamma/signs/H. It jointly reduces `float2(sum_of_squares, transformed_peak)` in one cross-SIMD synchronization phase. It applies inverse RMS only to the emitted row scale. The `False` baseline first computes inverse RMS, normalizes each element, then transforms and reduces the peak. Both are one-dispatch kernels with resident inputs and FP32 intermediates; neither materializes an intermediate normalized tensor.

The implementation lives in [rms_quant.py](../src/m5kernels/rms_quant.py) and [rms_quant.metal](../src/m5kernels/shaders/rms_quant.metal). Plain RMS supports arbitrary K tails. H supports whole power-of-two blocks from 32 through 1024, without hidden padding. Gamma is FP32. Callers must supply finite values with representable FP32 intermediates and positive representable scales; the benchmark checks those conditions before timing. Epsilon must remain finite and positive in FP32.

## Numerical evidence

An independent float64 implementation checks normalization, gamma, signed H, dynamic quantization and decoded values. Tests cover zeros, signed/zero/outlier gamma, tiny inputs where epsilon dominates, exact ties to even, all supported input dtypes, arbitrary plain-RMS tails, inactive SIMD groups and H3 width 5376. Quantization error is reported separately from decoded implementation error. The projection harness independently checks weight rotation and quantization, float64 dot preservation, sampled output rows/columns with full K, and finite values across complete outputs.

`bf16_intermediate_diagnostic` records the difference introduced by a hypothetical BF16 round after RMS*gamma. This diagnostic is outside timing and does not change the active contract. It is not a visual/audio model-quality evaluation.

## Measured result: fewer synchronizations did not establish a speedup

The initial folded H256 kernel used 96 temporary registers versus 72 for the direct kernel, with zero spills in both. Moving gamma multiplication next to its butterflies and completing the SIMD-local square sum earlier reduced folded register usage to 75. It still uses one cross-SIMD barrier. Both shader versions are retained under [experiments/rms-scale-folding](../experiments/rms-scale-folding/README.md).

These M4096/K5376/N7168 measurements use 256 preparation threads, BF16 inputs/output, synthetic learned gamma and INT8 GEMM. Medians are synchronized host wall time. The independent seed43 campaign reverses backend order.

| Variant / seed43 distribution | Direct prep ms | Folded prep ms | Direct online ms | Folded online ms |
|---|---:|---:|---:|---:|
| Initial / H256 normal | 0.511 | 0.531 | 3.659 | 3.684 |
| Initial / H256 outliers | 0.508 | 0.533 | 3.674 | 3.691 |
| Scheduled / H256 normal | 0.504 | 0.501 | 3.658 | 3.657 |
| Scheduled / H256 outliers | 0.517 | 0.504 | 3.678 | 3.659 |
| Scheduled / no-H weighted normal | 0.286 | 0.285 | 3.449 | 3.446 |

The scheduled variant is approximately tied overall. Seed29 contains slow first-case intervals, including H256 direct preparation around 0.707 ms; the apparent large wins do not repeat in seed43 and are not promoted. Both campaigns retain all seeds/distributions. A 512-thread campaign with 50 warmups also failed to establish a repeatable overall win; H256 preparation was about 0.57–0.62 ms. The default remains 256 threads.

Serial GPU replay assigns approximately 9.7% of the complete H256 path to preparation and 90.3% to GEMM. The unchanged GEMM reports 112 temporary registers and 272 spilled bytes. Preparation counters are 72/96/75 registers for direct/initial-folded/scheduled-folded, all with zero spills. These expose a resource tradeoff; they do not independently establish the bottleneck or a host-wall speedup.

All RMS records pass implementation checks. **Weighted RMS without H fails the 5% synthetic outlier quality budget**, at 5.43% and 5.62% in seeds29/43. H256 passes at about 0.81%/0.83%. Folding is not a replacement for the rotation's numerical effect.

Evidence: [initial seed29](../results/published/2026-10-18-rms-quant-initial.json), [initial seed43](../results/published/2026-10-18-rms-quant-initial-rerun.json), [scheduled seed29](../results/published/2026-10-18-rms-quant-scheduled.json), [scheduled seed43](../results/published/2026-10-18-rms-quant-scheduled-rerun.json), [512 seed29](../results/published/2026-10-18-rms-quant-512-independent.json), [512 seed43](../results/published/2026-10-18-rms-quant-512-rerun.json), [direct counters](../results/published/2026-10-18-rms-direct-profile.json), [initial folded counters](../results/published/2026-10-18-rms-folded-initial-profile.json), [scheduled folded counters](../results/published/2026-10-18-rms-folded-scheduled-profile.json).

## Reproduce

```sh
uv run python scripts/rms_quant_campaign.py --seed 29 --repeat 30 \
  --output results/local/rms-fold-new-seed29.json
uv run python scripts/rms_quant_campaign.py --seed 43 --repeat 30 --reverse \
  --output results/local/rms-fold-new-seed43.json
```

The `rms-quant` suite uses Q-sized projections with K=5376, N=7168 and synthetic inputs. It compares plain RMS, learned-gamma RMS and learned-gamma RMS+H256. Record activation preparation, prequantized GEMM, complete online preparation+GEMM and offline weight transformation/packing separately. The second campaign reverses backend order to check order sensitivity. Offline weight preparation is excluded from online latency and retained as its own timed stage.

## Applicability to H3

Nonzero AdaLN shift gives `y=r*(x*c)+beta`. After H, the shift is still additive: `y D H = r*((x*c) D H) + beta D H`. Factoring out `r` leaves `beta D H/r`, which depends on the normalization result. It does not permit a joint original-input norm and final-peak reduction.

H3's native-style contract also rounds completed AdaLN output to BF16 before H. BF16 rounding does not commute with factoring out `r`. This zero-shift experiment therefore cannot silently replace the existing full-AdaLN kernel. The harness explicitly rejects additive shift, LayerNorm and reduced-precision intermediate options.

Moving shift through the subsequent GEMM gives a more general alternate policy. See the [affine splitting study](affine-shift-splitting.md), including measured conditioning costs and counterexamples.
