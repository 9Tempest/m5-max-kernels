# Native QKV and AdaLN/Hadamard integration results

Both requested kernel families are connected to actual H3 inference. They
remain opt-in: the custom grouped QKV tile is slower than the already-fused
native comparator, and adding H256 to QKV/FC1 increases whole-model drift in
both measured seeds. Correct implementation and improved local quantization
do not establish a better deployment policy.

[Play the six new samples](gallery.html) · [Immutable file inventory](manifest.json)
· [Default decision](raw/decisions/kernel-default-combined-001.json)
· [Full contracts and reproduction](../../../../integrations/h3/kernel-integration.md)

## Connected policies and controls

| Policy | Custom QKV/RMS/RoPE | QKV H256 | FC1 H256 | Existing FC2 H256 |
|---|---:|---:|---:|---:|
| baseline | 0 | 0 | 0 | 1 |
| qkv-only | 1 | 0 | 0 | 1 |
| adaln-h-only | 0 | 1 | 1 | 1 |
| combined | 1 | 1 | 1 | 1 |

The two older gate-in-GEMM flags are zero throughout. The original BF16
checkpoint and prepared FC2-H256 checkpoint are retained. These experiments
prepare the new weight policy on load from the original checkpoint; they do
not relabel the older prepared checkpoint as QKV/FC1-H256.

All captures use the same five-second 512×512 fox prompt, seed42 or43,
20 Euler steps, all 50 blocks, reference RoPE and no reuse/token reduction.
The exact exports have 120 frames at 24 fps; native 124-frame originals remain
local. With the new flags disabled, both seeds exactly reproduce all 42
historical FC2-H256 latent tensors and both MP4 exports.

## Matched resident performance

| Seed | Synchronized interval | Baseline median, s | Combined median, s | Baseline / combined | Latency change |
|---:|---|---:|---:|---:|---:|
| 42 | Complete cached generation | 147.659017 | 149.364160 | 0.988584× | +1.155% |
| 42 | Denoise callback | 118.525176 | 120.390033 | 0.984510× | +1.573% |
| 43 | Complete cached generation | 147.807164 | 149.768661 | 0.986903× | +1.327% |
| 43 | Denoise callback | 118.640971 | 120.575822 | 0.983953× | +1.631% |

Positive latency change means slower than the existing FC2-H256 INT8 baseline.

| Seed | Interval | Baseline raw seconds | Combined raw seconds |
|---:|---|---|---|
| 42 | Complete cached generation | 149.216220 / 147.646479 / 147.659017 | 149.428955 / 149.364160 / 149.317661 |
| 42 | Denoise callback | 120.058543 / 118.455572 / 118.525176 | 120.443899 / 120.390033 / 120.381656 |
| 43 | Complete cached generation | 147.824244 / 147.807164 / 147.701214 | 149.983710 / 149.768661 / 149.755262 |
| 43 | Denoise callback | 118.640971 / 118.655992 / 118.593650 | 120.794514 / 120.575822 / 120.554058 |

Each arm has one full warmup and three timed requests. Order is
baseline42→combined42→combined43→baseline43. No latent/AdaLN capture is enabled
and CPU matrix/media analysis completed before timing began. Same-seed output
identity is checked across all four requests within each process.

Cached generation includes decode and original AV export, but excludes
preflight/build, initial weight preparation and exact-five-second trimming.
Repeated audio-VAE setup prevents a compilation-free end-to-end claim.
Denoise callbacks are fresh synchronized boundaries and include online
activation preparation. Two seeds and three repetitions provide descriptive
evidence, not confidence intervals or universal performance guarantees.
The shared GPU lock coordinates our jobs; it cannot exclude unrelated
external device activity. [Raw timings, ranges and provenance](raw/resident/kernel-resident-baseline-combined-001.json).

## Independent implementation and operator timing

- Preparation: 48/48 cases pass across M17/2049/9914/17455, seeds17/29,
  no-gate/shared-map/independent-map cases. Staged/fused BF16 AdaLN, INT8 codes
  and FP32 scales are exact. Maximum independent implementation relative L2 is
  6.087501e-7; finite/zero/padding and in-place residual checks pass.
- QKV: 32/32 supported cases pass across M129/2049/9914/17455 and native/H256
  operands, including zero, epsilon-dominated, ordinary and outlier rows.
  All eight M17 cases remain explicitly unsupported by the native API.
  Largest sampled independent implementation relative L2 is 0.000350711,
  below the unchanged 0.008 limit. V is exact; maximum full-output paired
  Q/K relative L2 is 0.0000348474 because the RMS reduction order differs.
- Both campaigns independently repeat in reversed order. Warmed,
  synchronized operator timing includes layout/padding and activation
  preparation for online comparisons. Offline weight quantization is separate.
- At M9914/17455, native/custom QKV online latency ratios are 0.9352–0.9428:
  **the custom tile is 6.1–6.9% slower**. Fused H256 preparation is 1.075–2.176×
  faster than staged H256 for the gate/shared-map cases, but is slower than
  native preparation without H256 (native/fused ratios 0.586–0.916).
  Visible variation and occasional other-case reversals are retained.

[QKV round1](raw/campaigns/qkv-r1/campaign.json) /
[round2](raw/campaigns/qkv-r2/campaign.json) ·
[Prep round1](raw/campaigns/prep-r1/campaign.json) /
[round2](raw/campaigns/prep-r2/campaign.json).

## Whole-model numerical evaluation

Each policy is compared with the same-seed BF16 capture. Lower relative L2
and higher SNR/SSIM mean closer numerical agreement, not certified perceptual
quality. Every one of the 42 captured tensors per run is finite. All20 Euler
video/audio states, final VAE inputs, absolute-error quantiles and decoded
RGB/PCM are retained in the reports.

| Seed | Policy | Final video L2 | Final audio L2 | RGB SSIM | RGB SNR, dB | PCM SNR, dB |
|---:|---|---:|---:|---:|---:|---:|
|42|baseline|0.335791|0.115167|0.704100|13.2407|15.2909|
|42|qkv-only|0.288042|0.138090|0.748341|14.8141|15.4685|
|42|adaln-h-only|0.351247|0.188343|0.680491|12.9583|12.0830|
|42|combined|0.403528|0.197490|0.651836|11.7464|10.9949|
|43|baseline|0.203205|0.035488|0.848317|19.6092|13.5560|
|43|qkv-only|0.178927|0.040522|0.873979|21.4069|13.5082|
|43|adaln-h-only|0.329118|0.049874|0.765097|13.3996|11.3390|
|43|combined|0.373968|0.065682|0.732474|12.1459|11.1246|

Combined video L2 worsens at 14/20 and 19/20 Euler steps; audio L2 worsens
at 18/20 and 16/20. Final audio absolute-error p999 grows 0.272951→0.542040
and 0.056762→0.172448. The repeated aggregate/tail/media regression rejects
combined default selection. QKV-only improves video agreement at both seeds,
but final audio latent error and tails worsen; it is a tradeoff, not uniformly
better numerics, and its operator performance is slower.

[Seed 42 complete evaluation](raw/evaluations/kernel-all-policies-eval-seed42-001/kernel-evaluation.json) ·
[Seed 43 complete evaluation](raw/evaluations/kernel-all-policies-eval-seed43-001/kernel-evaluation.json).

## Real activation calibration and limits

Across the two combined captures, all 36 QKV/FC1 cases and all 576 sampled rows
have lower projection relative L2 under H256 on identical X/W. Three rows have
higher maximum absolute error (4.86%, 16.41%, 1.81%); they are retained, not used
as automatic single-row vetoes. The complete model regression drives the
decision. Additional H-only captures and their analyses are also included.

Each capture samples 16 real rows at steps 0/10/19 and blocks 0/24/49, including
modality boundaries. Projection references sample complete Q/K/V heads or
paired FC1 gate/up channels with full K5376. Weight packing in this calibration
is a CPU reference; it is not an extra actual-row GPU FC1 GEMM proof.
Independent GPU implementation checks are the separate operator campaigns.
Index verification when absent from the native inventory is explicitly an
analysis-time pinned-index check, not retroactive native-loader provenance.

Six extracted frames per candidate retain the fox and snowy forest, with pose
and timing changes. Continuous motion and audio listening remain unassessed.
No prompt adherence, perceptual acceptance, official-Python parity or broader
prompt generalization is claimed. M17455 is an operator case matching the
earlier 15-second request's token count, not a new 15-second end-to-end result.

## Next optimization directions

1. Match the native grouped QKV's 128-row/full-K/local-scale layout before
   sweeping tile/traversal variants. Its source differences from our 64-row
   swizzle-8 tile suggest an ablation; they do not prove a hardware bottleneck.
2. Separate QKV and FC1 rotation, then calibrate layer/step selection. Preserve
   the same BF16 boundaries and judge full trajectories, media and tails.
3. Reduce H256 preparation traffic/reduction overhead under actual modality
   maps while retaining the residual output. Validate each variant against
   the staged exact comparator and independent reference.

The manifest preserves original JSON bytes and copied source snapshots.
Weights, raw latent/activation arrays, executable binaries, private logs and
the user's reference image are excluded. This authored interpretation is
versioned with the repository; it does not modify the raw reports.
