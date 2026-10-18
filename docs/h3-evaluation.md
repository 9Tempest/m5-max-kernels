# H3 evaluation: paired drift and model quality

`integrations/h3/evaluate.py` measures numerical differences between matched
outputs. It supports captured activations/latents (`.npy` or matching `.npz`
keys) and decoded video/audio. It is independent of `m5kernels`, `run.py` and
repository layout, so the same file can be copied into the pinned `h3.c` fork.
Requirements are Python 3.11+, NumPy, and FFmpeg/FFprobe for media. It never
downloads evaluation models or runs a GPU backend.

BF16 is the comparison baseline, not perceptual ground truth. Same-prompt,
same-seed generation can diverge spatially after small numerical changes.
Low pixel similarity therefore identifies output drift; it does not by itself
show that the candidate video is worse. A completed metric calculation is not
a quality pass. Every record keeps `quality_decision: not_evaluated`.

## Metric choices and the available evidence

| Metric | Appropriate scope | Interpretation and boundary |
| --- | --- | --- |
| SNR, relative L2, RMSE, maximum absolute error | Matched projection outputs, latents, RGB frames or PCM samples | Numerical drift at the exact capture boundary; sensitive to alignment and signal energy, not semantic quality |
| PSNR | Matched arrays with an explicit data range | Global MSE followed by a logarithm; not the average of frame PSNR values; signal fidelity, not a generative-quality score |
| FFmpeg SSIM | Corresponding decoded frames | Structural similarity under a named implementation/color contract; no temporal or semantic guarantee |
| LPIPS | Corresponding RGB images/frames | Learned image perceptual distance; lower is more similar; frame averages do not measure temporal consistency |
| VBench custom-input dimensions | A suitable custom clip or clip set | Dimension-specific visual/temporal measures; a single clip cannot stand in for the standard prompt-suite aggregate |
| VBench 2.0 | Dimension-specific benchmark inputs and, for some dimensions, collections | Intrinsic realism evaluation; not a generic score applicable to one arbitrary fox clip |
| FVD | Collections of videos from two distributions | Feature-distribution comparison; one clip per variant is insufficient for a meaningful estimate |

The original [SSIM paper and author implementation](https://ece.uwaterloo.ca/~z70wang/research/ssim/)
introduce a structural fidelity measure. The
[FFmpeg PSNR](https://ffmpeg.org/ffmpeg-filters.html#psnr) and
[SSIM filter documentation](https://ffmpeg.org/ffmpeg-filters.html#ssim) describe
their own frame-based implementations. Preserve the implementation, pixel
format, range and frame correspondence; do not interchange FFmpeg SSIM with
another library's SSIM or report RGB SSIM as the historical YUV score.

[LPIPS](https://github.com/richzhang/PerceptualSimilarity), based on
[Zhang et al.](https://arxiv.org/abs/1801.03924), accepts RGB tensors shaped
`N×3×H×W`, normalized to `[-1,1]`. A future adapter must pin the package/revision,
backbone, metric version and checkpoint hash. For aligned frames it should
save every distance and the mean/median/p95/maximum. These remain image-distance
statistics, without a claim about prompt adherence, motion or audio.

The [VBench official repository](https://github.com/Vchitect/VBench) and
[paper](https://arxiv.org/abs/2311.17982) define 16 dimensions. Its documented
custom-input mode supports subject consistency, background consistency,
motion smoothness, dynamic degree, aesthetic quality and imaging quality.
Report those dimensions separately as `custom_input`; do not call a fox-clip
evaluation the complete VBench score. Dynamic degree measures motion and is
not a universal higher-is-better quality target.

[VBench 2.0's official documentation](https://github.com/Vchitect/VBench/tree/master/VBench-2.0)
and [paper](https://arxiv.org/abs/2503.21755) evaluate intrinsic realism. Its
documented custom-input support covers human anatomy/identity/clothes,
diversity and multi-view consistency; diversity requires at least 20 videos
per prompt. The human-specific dimensions are not applicable to this fox
request, and one pair cannot support a full benchmark claim. Pin the selected
dimension and current evaluation protocol when implementing an adapter.

The [original FVD paper, section 4.2](https://arxiv.org/pdf/1812.01717) compares
feature means and covariances and shows finite-sample estimation effects:
even samples from the same distribution generally yield nonzero distance,
and compared experiments must use equal sample counts. The 120 frames of one
clip are not 120 independent video samples. Overlapping windows do not fix
this limitation. [Ge et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ge_On_the_Content_Bias_in_Frechet_Video_Distance_CVPR_2024_paper.html)
also identify I3D-FVD's content bias and limited temporal sensitivity.
[Beyond FVD](https://arxiv.org/html/2410.05203v2) gives dataset-specific
examples of slow sample-size convergence; those results are not a universal
minimum sample count. A future collection study must pin its feature model,
preprocessing and frame sampling, equalize video counts, and report repeated
subsampling/uncertainty alongside the sample count.

## Exact numerical contract

For reference `r`, candidate `c`, error `e = c - r`, and `N` scalar elements:

- `SNR_dB = 10 log10(sum(r²) / sum(e²))`.
- `relative_L2 = sqrt(sum(e²) / sum(r²))`.
- `RMSE = sqrt(sum(e²) / N)`; maximum absolute error is `max(abs(e))`.
- `PSNR_dB = 20 log10(data_range / RMSE)`.

All elements are checked for finiteness and included in the metrics; there is
no sampling, clipping or automatic range estimation. The implementation uses
float64 with scaled norms to avoid overflowing a sum of squares. Subtraction
or conversion overflow rejects the numerical result. Integer inputs outside
the exactly representable float64 range are rejected. BF16 bit patterns must
be decoded to numeric floating arrays by the producer, not disguised as
`uint16` values. Inputs must have the same nonempty shape; `.npz` keys must
match exactly, and pickle/object/complex inputs are rejected.

Arrays require `--data-range`, chosen and recorded before comparing variants.
RGB is decoded as unsigned 8-bit RGB with `data_range=255`. PCM uses floating
sample values without amplitude normalization, with default `data_range=2`
for the full nominal `[-1,1]` span. `--audio-data-range` can explicitly select a
different convention; this is not the peak-amplitude-1 convention in the older
`compare.py`. SNR is independent of this PSNR range choice. PCM outside its
nominal full-scale interval is preserved, not clipped.

Each scalar is `{status, value}`. JSON never contains NaN or Infinity:

| Condition | SNR | Relative L2 | PSNR |
| --- | --- | --- | --- |
| Nonzero reference, zero error | `positive_infinity`, null | finite 0 | `positive_infinity`, null |
| Zero reference, zero error | `undefined`, null | `undefined`, null | `positive_infinity`, null |
| Zero reference, nonzero error | `negative_infinity`, null | `positive_infinity`, null | finite if representable |
| Nonfinite input | `invalid_nonfinite`, null | `invalid_nonfinite`, null | `invalid_nonfinite`, null |

Reporting-range overflow/underflow has an explicit state; it is never replaced
with a large finite sentinel. Numeric failure retains the input hashes and
failure record and makes the command fail.

## Strict media correspondence

The media path requires exactly one video and one already-32-kHz stereo audio
stream. It compares dimensions, nominal/average FPS, exact rational frame
timestamps, audio frame timestamps and sample counts, declared duration,
pixel/color metadata and aspect ratio. Each video must have a contiguous
constant-rate timeline; audio blocks must have no gaps or overlaps and must
start with video. Declared audio/video durations must match. Unsupported
rotations are rejected.

Frame durations must also match the declared FPS/audio sample counts; changing
channel layouts, frame cropping, interlacing and repeated fields are rejected.
Frame-level color metadata is preserved and compared as well as stream metadata.

FFprobe records the decoded frame timeline before evaluation. FFmpeg then
uses software decoding (`-hwaccel none`), no autorotation, passthrough video
timing, `rgb24`, and `pcm_f32le`. There is no scale, frame-rate conversion,
frame duplication, audio rate/channel conversion, shift search or time warp.
Raw byte/sample counts must agree with the probe. Decoder priming/padding
handling is FFmpeg's documented decode behavior; the harness does not trim
the decoded sample arrays to force agreement. Global stereo metrics and
separate left/right metrics are recorded.

Matching unknown color metadata is retained as unknown, not evidence of a
calibrated color space. RGB conversion follows the recorded FFmpeg version
and input metadata. MP4 metrics include codec error and therefore do not
isolate model quantization. Prefer corresponding pre-encode frames or
lossless outputs for a future precision experiment.

`--ssim` compares the same decoded RGB frames, converted to planar RGB `gbrp`
inside the FFmpeg SSIM filter. It retains every frame score, their arithmetic
mean and raw stdout/stderr. This is an explicitly named RGB FFmpeg SSIM
backend. Missing frames, nonfinite SSIM or backend failure produce failure,
with no alternate metric fallback.

## Run and preserve evidence

```sh
uv run python integrations/h3/evaluate.py reference-latents.npz candidate-latents.npz \
  --kind arrays --data-range 1 --output-dir results/local/h3/eval-latents-001

uv run python integrations/h3/evaluate.py bf16/video-5s.mp4 int8/video-5s.mp4 \
  --kind media --ssim --output-dir results/local/h3/eval-media-001

uv run python integrations/h3/evaluate.py bf16/video-5s.mp4 bf16/video-5s.mp4 \
  --kind media --ssim --output-dir results/local/h3/eval-self-001

uv run pytest -q tests/test_h3_evaluate.py
```

In a standalone fork, use `python path/to/evaluate.py` with NumPy installed;
no repository-relative imports are required. Use a fresh output directory.
Each evaluation retains `evaluation.json`, input hashes, source hash/revision
and dirty state when Git is available, library/tool versions, tool binary
hashes, raw probe/process logs, decoded media and artifact hashes. Existing
records cannot be overwritten. Input hashes are checked again at the end.
Evaluation wall time is CPU evaluation cost, not generation latency.

LPIPS and VBench currently have no configured adapter or checkpoint. Without
their flags they are `not_measured`; explicitly requesting `--lpips` or
`--vbench` records `unavailable`, leaves the numeric metrics intact, and makes
the overall result `partial` with a nonzero exit code. There is no download or
silent substitute. FVD is `not_measured` for this single-pair interface.

## First paired media results

The first strict evaluation compares the completed seed42 BF16 and upstream
INT8 exports. Both pass geometry, exact timestamp/duration, color-metadata and
audio alignment checks: 120 frames at 512×512/24 fps and 160000 stereo samples
at 32 kHz. All decoded samples are finite.

The kernel-library result package now contains
[all six playable fox samples](../results/published/h3-native-eval/fox-two-seeds-v1/README.md),
a [video gallery](../results/published/h3-native-eval/fox-two-seeds-v1/gallery.html)
and a [source/checksum manifest](../results/published/h3-native-eval/fox-two-seeds-v1/manifest.json).
It includes byte-identical immutable evaluation JSON; model weights and
latent arrays remain local.

| Global paired metric | RGB (`data_range=255`) | Stereo PCM (`data_range=2`) |
| --- | ---: | ---: |
| SNR, dB | 12.712753 | 13.961846 |
| Relative L2 | 0.23139946 | 0.20040461 |
| RMSE | 29.858543 | 0.001710915 |
| Maximum absolute error | 233 | 0.020317929 |
| PSNR, dB | 18.629431 | 61.356032 |

Mean **RGB FFmpeg SSIM is 0.690500567**. This differs from the earlier
`compare.py` YUV SSIM (0.806325) because the color/metric contract differs; it
is not a performance or quality regression between repeated evaluations.
BF16 self-comparison gives zero RGB/PCM error and SSIM 1, with positive-infinite
SNR/PSNR represented by status plus null. The full results are
[paired evaluation](../results/published/h3-native-eval/fox-two-seeds-v1/evaluations/eval-bf16-int8-seed42-002-evaluation.json)
and [self-pair sanity check](../results/local/h3/eval-bf16-self-seed42-002/evaluation.json).
Both were generated by evaluator SHA256
`e54704bba58a6c85bf2bddd6825f7ecbb39b4a50440f6a350708fdd3e5e62f8a`.
Earlier `-001` records are retained; `-002` additionally enforces each frame's
duration, cropping/field flags, audio layout and frame-level color metadata.
The numerical results are unchanged. These are drift measurements from one
seed and do not authorize checkpoint promotion.

## Two-seed native FC2-H256 follow-up

The frozen FC2 capture producer records all 20 post-Euler packed video/audio
states and the final unpacked VAE inputs. `evaluate_latents.py` independently
checks completed run status, prompt/seed/recipe, source patch, original model
identity, exact dtype/layout/shapes, metadata and every raw tensor SHA256,
byte count and finite value. It writes one 42-key NPZ per policy, per-state
metrics and an immutable `latent-evaluation.json`. No reshaping across capture
boundaries or temporal alignment is applied beyond the declared raw layout.
`data_range=1` is only an explicit latent PSNR convention; SNR and relative L2
are the primary numeric comparisons, and latent PSNR is not a quality score.

| Final unpacked latent | Native INT8 relative L2 | H256-FC2 relative L2 | Native SNR dB | H256-FC2 SNR dB |
| --- | ---: | ---: | ---: | ---: |
| Seed42 video | 0.372733234 | 0.335791037 | 8.572038 | 9.478618 |
| Seed42 audio | 0.123749626 | 0.115167462 | 18.149122 | 18.773404 |
| Seed43 video | 0.221757357 | 0.203205321 | 13.082439 | 13.841298 |
| Seed43 audio | 0.062230552 | 0.035487680 | 24.119927 | 28.998448 |

H256 lowers video relative L2 in 18/20 steps at seed42 and 17/20 at seed43;
audio improves in 12/20 and 19/20. The regressing video steps are 0/5 and
1/3/4 respectively. Audio regresses at 5/8/9/10/11/14/15/16 and at 0. These
states are correlated denoiser trajectory samples, not independent repeats.
Final RMSE and maximum absolute error also decrease in all four cases.
[Seed42 provenance and every state](../results/published/h3-native-eval/fox-two-seeds-v1/evaluations/latent-threeway-seed42-001-latent-evaluation.json),
[seed43 provenance and every state](../results/published/h3-native-eval/fox-two-seeds-v1/evaluations/latent-threeway-seed43-001-latent-evaluation.json).

| Decoded media drift against BF16 | Native INT8 | H256-FC2 |
| --- | ---: | ---: |
| Seed42 RGB SNR dB | 12.712753 | 13.240715 |
| Seed42 RGB FFmpeg SSIM | 0.690500567 | 0.704099592 |
| Seed42 stereo PCM SNR dB | 13.961846 | 15.290948 |
| Seed43 RGB SNR dB | 17.374020 | 19.609208 |
| Seed43 RGB FFmpeg SSIM | 0.833956283 | 0.848317450 |
| Seed43 stereo PCM SNR dB | 11.811912 | 13.556003 |

Evidence: [seed42 H256](../results/published/h3-native-eval/fox-two-seeds-v1/evaluations/eval-bf16-hadamard-seed42-root-001-evaluation.json),
[seed43 native](../results/published/h3-native-eval/fox-two-seeds-v1/evaluations/eval-bf16-native-seed43-001-evaluation.json),
[seed43 H256](../results/published/h3-native-eval/fox-two-seeds-v1/evaluations/eval-bf16-hadamard-seed43-001-evaluation.json),
with the original seed42 native evaluation above. The previously recorded
strict every-row gate remains immutable. The user's updated decision allows
individual row/step regressions to be weighed against aggregate improvement,
tail statistics and complete video/audio results across seeds; a historical
strict-gate failure is not itself the current candidate rejection. These
metrics support that comparison without imposing a fabricated SNR threshold.

```sh
uv run python integrations/h3/evaluate_latents.py \
  --bf16 results/local/h3/fc2-bf16-capture-seed42-001 \
  --native results/local/h3/fc2-native-capture-seed42-001 \
  --hadamard results/local/h3/fc2-hadamard-capture-seed42-001 \
  --output-dir results/local/h3/latent-threeway-seed42-NEW
```

## Additional fixed benchmark pairs

The versioned [catalog](../integrations/h3/benchmarks.json) fixes person-turn-v1
(seed71), rainy-street-v1 (seed73), and water-pour-v1 (seed79), including the
exact prompts and temporal/texture/audio inspection questions. Every request
uses 512×512, 5 seconds, 24 fps, 20 Euler passes and all 50 blocks. The catalog
reader binds its exact bytes to SHA256
`a82e6544fc15c59beeca87a81e8e36f0d75ab2913a7eb3e4924ee62a5815d762`.

`run_benchmark.py` uses original BF16 or the separately retained, verified
imported INT8/H256 checkpoint. H256 requires the derived checkpoint directory
and its explicit manifest SHA; there is no original-weight or on-load
preparation fallback. Both paths retain source guards, full weight hashes,
the shared GPU lock and exact-duration media checks. Default latent capture
makes their process times diagnostic; `--no-capture-latents` is available for
a separate performance measurement. Video/audio shapes remain fixed, while
new-prompt text and total packed row counts are not assumed to equal 9914.

The two-seed quality captures above use the frozen on-load H256 implementation.
The separate 200-projection offline export and derived-checkpoint assembly
have completed, with manifest SHA256
`6840c91f6e5b9fcb4e8fe25292a1b1e2209d2f7f9c02e3427cfbbcbc71da4cda`.
The [prepared seed42 replay](../results/local/h3/prepared-replay-seed42-001/run.json)
has completed: all 42 captured latent tensors and both MP4 files are
byte-identical to the on-load H256 reference, with 200 verified projections
loaded and zero runtime weight rotation/quantization. This is loader replay
parity for one request, not a quality approval. The first
[native fusion campaign](../results/published/h3-native-eval/fused-campaign-001-v1/README.md)
passes all 12 operator numerical cases; independent timing repetition and
whole-model fusion parity/E2E remain pending, as do the additional
five-/fifteen-second pairs. The original BF16
and completed derived checkpoint are retained locally. The separate
[resident H256 measurement](../results/published/h3-native-eval/performance/hadamard-resident-seed42-001.json)
has one warmup plus three timed same-seed requests and byte-identical videos;
it is performance evidence for its recorded scope, not a perceptual quality gate.

```sh
uv run python integrations/h3/run_benchmark.py \
  --benchmark-id person-turn-v1 --mode bf16 \
  --output-dir results/local/h3/person-turn-bf16-NEW --gpu-wait-seconds 3600

uv run python integrations/h3/run_benchmark.py \
  --benchmark-id person-turn-v1 --mode hadamard-fc2 \
  --checkpoint-dir CHECKPOINT --checkpoint-manifest-sha256 DERIVED_MANIFEST_SHA256 \
  --output-dir results/local/h3/person-turn-h256-NEW --gpu-wait-seconds 3600
```

Use the same ID for each pair, then run the strict media evaluator. These
three tasks expand coverage; they do not constitute VBench/FVD evaluation.

For the new BF16/prepared-H256 pairs, use the separate loader-aware evaluator:

```sh
uv run python integrations/h3/evaluate_benchmark_pair.py \
  --bf16 results/local/h3/person-turn-bf16-NEW \
  --hadamard results/local/h3/person-turn-h256-NEW \
  --media --output-dir results/local/h3/person-turn-eval-NEW
```

It accepts the frozen BF16 observation source and the separately approved
prepared loader, rather than demanding that the loaders have identical
patches. The H256 run preserves an exact derived-manifest snapshot and a
stable `model.source_model` identity; evaluation verifies that snapshot's
SHA, original model identity, 200-projection import policy, no-requantization
marker, identical catalog/prompt/seed/recipe, and all 42 tensor hashes and
capture boundaries. `--media` runs the strict media comparison, while
`--media-evaluation PATH` joins an existing strict record only after matching
both video hashes and evaluator provenance. The original three-way fox helper
and its published results remain unchanged.

## Fifteen-second first-frame pair

The separate [long catalog](../integrations/h3/long_benchmark.json) fixes
`deepseek-whale-maid-15s-v1`: seed 83, 512×288, 24 fps, 20 Euler passes and
the supplied first frame. Its SHA256 is
`ee6fb7632e4b65858c9f0ec2944f5ae2a95eb85ae0e3219e308fb6fd1295e21a`.
The entire 2,233-byte Chinese prompt, including its final newline, and the
208,682-byte reference JPEG have independent pinned hashes. Each run stores
exclusive byte copies as `prompt.txt` and `conditioning-input.jpg`.
The native first-frame path scales the original 1280×720 image to 512×288;
this model preprocessing is recorded explicitly and is not an evaluation
alignment operation.

The native temporal contract generates 362 frames and retains `original.mp4`;
the separately timed export produces `video-15s.mp4` with 360 frames. Dynamic
capture shapes are final video `[24,107,18,32]`, final audio `[32,2,603]`,
per-step video `[15408,96]`, and per-step audio `[1206,32]`. Per-step video
capture excludes the 144 first-frame conditioning rows. These dimensions
follow the pinned native source; successful run metadata must independently
confirm them before evaluation.

```sh
uv run python integrations/h3/evaluate_long_pair.py \
  --bf16 results/local/h3/long-bf16-NEW \
  --hadamard results/local/h3/long-prepared-h256-NEW \
  --media --output-dir results/local/h3/long-pair-eval-NEW
```

This separate evaluator verifies the frozen catalog, verbatim prompt, image
bytes and preprocessing, seed/geometry/temporal boundary, approved source
pins, original model identity through the derived-manifest snapshot, and all
20 steps plus final tensors. It writes two NPZ files and an immutable
`long-evaluation.json`; optional media comparison uses the unchanged strict
RGB/PCM evaluator and binds the exact videos by hash. The original five-second
evaluators and published two-seed records keep their contracts. The new CPU
fixture tests validate rejection behavior, not GPU output or model quality.

## Promotion requirements

1. Match source/model hashes, prompt, seed, sampler, steps, dimensions and
   conditioning. Capture identical projection boundaries and early/middle/late
   denoiser states. Separate implementation error against decoded operands
   from total error against the BF16 baseline. Require finite tensors.
2. Run the paired numerical harness and a self-pair sanity check. Review
   projection and latent trajectories, not only final pixels. Record drift
   without inventing a universal SNR/PSNR threshold for model quality.
3. Use a preregistered diverse prompt set and multiple seeds, then independent
   reruns on the same hardware. Inspect visual fidelity, temporal consistency,
   prompt adherence and audio quality/synchronization; use applicable
   perceptual/VBench dimensions and blinded human comparisons. Visual metrics
   do not evaluate stereo audio or audio/video synchronization by themselves.
4. Promote a checkpoint/backend only when implementation contracts pass and
   the agreed aggregate/tail and video/audio criteria support the decision,
   with complete deployment performance measured. The original BF16 and
   separate derived INT8/H256 checkpoints remain locally retained. Report
   startup, preparation, resident inference and export scopes
   separately. Run CPU evaluation outside timed performance windows, and do
   not multiply a projection microbenchmark speedup into model throughput.

The current harness supplies the numeric/structural evidence for this
protocol. It intentionally does not implement a model-quality approval gate
or claim that a single five-second clip validates a checkpoint.
