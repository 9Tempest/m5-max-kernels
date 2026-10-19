# Native QKV and AdaLN/Hadamard integration

This follow-up connects the repository's Q/K projection/RMSNorm/RoPE and
AdaLN/H256/INT8 kernels to the actual pinned H3 runtime. The comparator is
h3.c's existing fused implementation. Earlier single-projection fusion
measurements do not establish a gain over that comparator.

The combined source is distributed as `kernel-integration.patch` against
upstream `8974cc055ea9c02fcd14cc27dfda3e1027c05153` (MIT; see
`UPSTREAM-LICENSE`). `kernel-integration-source.json` binds the complete source
digest and patch digest. The source manifest's SHA256 is an explicit argument
to every new runner. Historical runners, patches, records and checkpoints
retain their own frozen contracts.

## Connected paths

| Policy | Custom grouped QKV | QKV H256 | FC1 H256 | FC2 H256 |
|---|---:|---:|---:|---:|
| `baseline` | 0 | 0 | 0 | 1 |
| `qkv-only` | 1 | 0 | 0 | 1 |
| `adaln-h-only` | 0 | 1 | 1 | 1 |
| `combined` | 1 | 1 | 1 | 1 |

The custom QKV kernel uses the repository's TM64/TN128/four-SIMD-group,
swizzle-8 tile with the native grouped weight layout. It writes all three
streams directly in head-major layout. Projection rounds to BF16 before
Q/K RMSNorm; gamma precedes 48 split-half rotary pairs. V bypasses RMSNorm
and rotation. The SIMD RMS reduction may differ from the original reduction
tree, so numerical validation uses independent decoded-operand references.

QKV/FC1 H256 rotates and quantizes weights during offline preparation.
Online preparation preserves the gated residual BF16 boundary, computes
RMSNorm and the complete AdaLN affine expression, rounds to BF16, then
applies normalized H256 and whole-row INT8 packing. Gate and normalization
tables/maps are independent operands. Rotation is not moved through AdaLN
shifts or SwiGLU. The existing FC2 transform remains after SwiGLU.

The two older gate-in-GEMM flags remain zero throughout this experiment.
`H3_HADAMARD_PREP_STAGED=1` selects an explicit diagnostic implementation of
the same H256 policy; it is not a fallback. Unsupported source, dtype,
geometry or old prepared-checkpoint combinations fail explicitly.

## Operator evidence

`run_prep_campaign.py` compares native preparation without H256, explicit
staged H256 preparation, and fused H256 preparation. Synchronized online
intervals include required padding; CPU input restoration and correctness
readbacks are outside timing. References independently compute gate, RMS,
AdaLN and H256 on CPU. Quantization error is reported separately from
implementation error, with the unchanged 0.008 relative-L2 implementation
gate applied to complete sampled rows.

The first two campaigns cover M17/2049/9914/17455, seeds17/29, and no-gate,
shared-map gate, and independent-map gate cases. All 48 cases pass. Staged
and fused BF16 AdaLN, INT8 codes and FP32 scales are bit-identical. Maximum
independent quantized-output relative L2 is 6.087501e-7. All gated elements
match the CPU FP32-FMA/BF16 reference; full output finite, zero and padding
checks pass. The second campaign reverses case order. Raw reports remain in
`results/local/h3/kernel-prep-campaign-r1` and `-r2`.

For the gate/shared-map path at M9914/17455, fused H256 is 1.075–2.176 times
as fast as staged H256 across those seeds and rounds. Compared with the
native preparation that does not perform H256, fused H256 costs more:
native/fused latency ratios are 0.586–0.916. These are preparation-only
observations with visible timing variation, not model speedups.

`run_qkv_campaign.py` compares native and custom kernels on identical actual
INT8 operands under both native and H256 packing policies. It separates
prepacked QKV from activation-packing-plus-QKV timing; the input is already
AdaLN output. It checks four row classes, three complete heads, all 128
channels and K5376, plus full-buffer finite/zero/padding checks. M17 is
explicitly unsupported by the native QKV API; M129 supplies a supported
tail case. No padded M17 result is presented as native M17 execution.

Both independent campaigns pass all 16 supported cases each. Across
M9914/17455, seeds17/29 and native/H256 operands, native/custom online latency
ratios are 0.9361–0.9418 in the first campaign and 0.9352–0.9428 in the reversed
rerun. Thus the custom implementation is 6.1–6.9% slower in these deployed
lengths. V is bit-identical; Q/K differences from the RMS reduction order are
retained and pass the unchanged independent implementation gate. This is a
negative performance result against the already-fused native comparator.

The largest sampled independent QKV implementation relative L2 across both
native and custom paths is 0.000350711, below the unchanged 0.008 limit.
The largest full-output paired Q/K relative L2 is 0.0000348474; V remains
exact. Small single-call differences can amplify through the iterative model,
so whole-model equality is neither assumed nor used as an implementation gate.

## Whole-model evidence and defaults

The independent runner exposes all four policies with exact runtime flag
logging and source guards. Captures preserve all 20 Euler video/audio
states and final VAE inputs. Optional AdaLN capture gathers 16 real rows
at steps0/10/19 and blocks0/24/49 for each enabled family; it adds diagnostic
waits and is excluded from latency measurements.

The resident recipe is the same 512-square, five-second fox request,
20 steps, 50 blocks, reference RoPE and no reuse/token reduction. Each arm
has one full warmup and three timed requests. Denoise timing uses fresh
synchronized callbacks. Complete-request timing includes decoding and
original 124-frame AV export; repeated audio-VAE setup means it is not
labelled compilation-free end-to-end latency.

All four policies have completed captures at seeds42/43. With the new flags
disabled, both seeds reproduce all 42 historical FC2-H256 latent tensors and
both original/exact-duration MP4s byte-for-byte. Every captured tensor is
finite. The following scores compare each policy with the same-seed BF16 run;
lower latent L2 and higher SSIM/SNR indicate closer numerical agreement.

| Seed | Policy | Final video L2 | Final audio L2 | RGB SSIM | RGB SNR, dB | PCM SNR, dB |
|---:|---|---:|---:|---:|---:|---:|
| 42 | baseline | 0.335791 | 0.115167 | 0.704100 | 13.2407 | 15.2909 |
| 42 | qkv-only | 0.288042 | 0.138090 | 0.748341 | 14.8141 | 15.4685 |
| 42 | adaln-h-only | 0.351247 | 0.188343 | 0.680491 | 12.9583 | 12.0830 |
| 42 | combined | 0.403528 | 0.197490 | 0.651836 | 11.7464 | 10.9949 |
| 43 | baseline | 0.203205 | 0.035488 | 0.848317 | 19.6092 | 13.5560 |
| 43 | qkv-only | 0.178927 | 0.040522 | 0.873979 | 21.4069 | 13.5082 |
| 43 | adaln-h-only | 0.329118 | 0.049874 | 0.765097 | 13.3996 | 11.3390 |
| 43 | combined | 0.373968 | 0.065682 | 0.732474 | 12.1459 | 11.1246 |

The combined policy regresses in video L2 at 14/20 and 19/20 Euler steps,
and in audio L2 at 18/20 and 16/20 steps. Its final audio absolute-error p999
increases from 0.272951 to 0.542040 at seed42 and from approximately 0.05676
to 0.172448 at seed43. This aggregate and tail regression rejects the combined
policy for default selection. The individual QKV policy has better video
agreement at both seeds but larger final audio latent errors, and its operator
performance is slower; it also stays opt-in. This is a numeric policy decision,
not a claim of perceptual inferiority.

The two combined captures each supply 18 real QKV/FC1 cases. All 36 cases and
all 576 selected rows improve same-X/W projection relative L2 with H256, yet
three rows have larger maximum absolute error (4.86%, 16.41% and 1.81%). Those
individual exceptions are retained rather than treated as automatic vetoes.
The decisive regression is the complete trajectory/media comparison above.
Projection calibration samples output channels with full K and uses CPU
weight quantization; it is separate from the independent GPU implementation
evidence. Reduced local projection error does not guarantee reduced final
generation drift.

Six extracted frames per candidate retain the fox/snow scene with pose and
timing differences. Continuous motion and audio listening remain unassessed.
Neither SNR nor SSIM establishes prompt adherence or perceptual acceptance.
Original BF16 and the prior FC2-H256 checkpoint are preserved; no rejected
QKV/FC1-H checkpoint replaces them. New flags remain off by default.

The resident performance campaign uses two seeds in reversed policy order,
one full warmup and three timed requests in each of four processes. Its
matched baseline/combined comparison is reported separately from diagnostic
capture process walls.

| Seed | Interval | Existing FC2-H256 baseline, s | Combined, s | Latency increase |
|---:|---|---:|---:|---:|
| 42 | Cached complete generation | 147.659017 | 149.364160 | 1.155% |
| 43 | Cached complete generation | 147.807164 | 149.768661 | 1.327% |
| 42 | Synchronized denoise callback | 118.525176 | 120.390033 | 1.573% |
| 43 | Synchronized denoise callback | 118.640971 | 120.575822 | 1.631% |

These are medians of three timed requests. Order is baseline42, combined42,
combined43, baseline43. Each process's four original exports are byte-identical
and match the same-seed/policy diagnostic capture's original video. This binds
the timed media to the numerical evaluation but does not prove latent equality
without a readback. The complete request includes decode/export and repeated
audio-VAE setup; initial weight preparation and exact-duration trimming are
excluded. No confidence interval, compilation-free end-to-end result or
cross-prompt performance claim is made.

The [public results and six playable samples](../../results/published/h3-native-eval/kernel-integration-v1/results.md)
include the raw arrays, ranges, 499 checksum-bound evidence/source files and
the completed default decision. Implementation checks pass; the combined
numeric review, two-seed speed benefit and denoise non-regression gates fail.
The decision is `keep_experimental`, with all three new flags remaining zero.

## Reproduction

Prepare a separate checkout, apply the complete patch once, and use a new
output directory for each command. Do not apply the older patches on top
of this complete patch.

```sh
git clone https://github.com/antirez/h3.c.git .cache/h3.c-kernel-integration
git -C .cache/h3.c-kernel-integration checkout --detach 8974cc055ea9c02fcd14cc27dfda3e1027c05153
git -C .cache/h3.c-kernel-integration apply ../../integrations/h3/kernel-integration.patch
uv sync --locked
```

Read and use the SHA256 of the reviewed source manifest as `PIN_SHA` below.
The runner verifies its patch list and complete native source before running.

```sh
uv run python integrations/h3/run_prep_campaign.py \
  --source-manifest integrations/h3/kernel-integration-source.json \
  --source-manifest-sha256 "$PIN_SHA" --output-dir results/local/h3/prep-NEW
uv run python integrations/h3/run_qkv_campaign.py \
  --source-manifest integrations/h3/kernel-integration-source.json \
  --source-manifest-sha256 "$PIN_SHA" --output-dir results/local/h3/qkv-NEW
uv run python integrations/h3/run_kernel_integration.py \
  --kind capture --policy combined --seed 42 --capture-adaln \
  --source-manifest-sha256 "$PIN_SHA" --output-dir results/local/h3/combined-capture-NEW
uv run python integrations/h3/run_kernel_integration.py \
  --kind resident --policy combined --seed 42 \
  --source-manifest-sha256 "$PIN_SHA" --output-dir results/local/h3/combined-resident-NEW
```

All GPU work uses the shared repository lock. Never run concurrent GPU
benchmarks. Optional CPU projection calibration also consumes substantial
memory bandwidth and should be scheduled outside GPU timing campaigns.

## Follow-up experiments suggested by this evidence

1. Tune against the actual grouped comparator. The native large-shape kernel
   uses 128-row tiles, full-K5376 specialization and local scales with Morton-4
   traversal; our candidate uses 64-row tiles and swizzle-8. A matched tile,
   traversal and scale-cache ablation can isolate the slowdown. These source
   differences are hypotheses, not device-counter proof of a bottleneck.
2. Calibrate QKV and FC1 rotation separately, then by layer/step if needed.
   The native flags already separate the two families. Better same-X/W
   projection error is insufficient when iterative trajectory and audio
   drift increase. Retain the BF16 boundaries and compare aggregate/tail
   outcomes over more prompts before choosing a mixed precision policy.
3. Optimize the H256 preparation overhead under the deployed modality maps.
   Compare shared-memory traffic, reductions and required padding while
   preserving the residual side output and BF16 AdaLN boundary. Reuse the
   staged/fused exact-pair check and independent reference for every variant.

The M17455 operator cases cover the earlier 15-second request's sequence
length. They are not a new 15-second end-to-end video benchmark, and no such
throughput claim is inferred from them.
