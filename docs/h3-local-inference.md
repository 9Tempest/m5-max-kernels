# H3 local inference experiment

This experiment connects a real-weight deployment to the kernel laboratory.
The pinned runtime, model acquisition, exact prompt and commands are in
[integrations/h3](../integrations/h3/README.md). The source is
[h3.c 8974cc0](https://github.com/antirez/h3.c/tree/8974cc055ea9c02fcd14cc27dfda3e1027c05153),
with the original FL2VA checkpoint at MiniMax revision
`42ed227ee7df40d41602854ae760620d6eb651fe`.

## Request and evidence status

The request is a red fox walking in snowy woods, 512×512, 24 fps, five seconds,
seed 42, 20 Euler passes, all 50 DiT blocks, no reuse, token reduction or
upscaling. The native temporal contract produces 124 frames (5.1667 seconds);
a separately timed export retains 120 frames for exactly five seconds.

The pinned tokenizer has executed on the real tokenizer file and produces
28 text tokens. Source temporal/layout arithmetic gives 37 video latent
timesteps, 9472 video tokens and 414 audio tokens, for **9914 packed tokens**.
The later native FC2 capture confirms **9914 actual packed rows**, with text
`[0,28)`, audio `[28,442)` and video `[442,9914)`. This upgrades the original
CPU/source prediction to runtime observation for the fox request; other
prompts have different text lengths. See the [captured metadata](../results/local/h3/fc2-native-capture-seed42-001/fc2/fc2-step00-block00.json).

The first full **BF16 generation completed** with the verified checkpoint and
unmodified pinned runtime. The generation process took **252.9383 seconds**.
`ffprobe` verifies the retained 124-frame original and the separately exported
120-frame, 5.000-second video: both are 512×512 at 24 fps with 32 kHz stereo
audio. Exact-duration export took another 0.1922 seconds. Native upstream INT8
also completed in **182.4550 seconds**. A successful media check establishes
the output contract, not official-pipeline parity or perceptual quality.

| BF16 phase | Upstream wall seconds | Fraction of generation process |
| --- | ---: | ---: |
| DiT Euler denoising, 20 passes | 205.297 | 81.16% |
| Video VAE decoder | 30.218 | 11.95% |
| DiT loading/preparation | 8.707 | 3.44% |
| Qwen text encoder | 5.355 | 2.12% |
| Audio VAE decoder | 2.165 | 0.86% |

These phase scopes are not an exhaustive process decomposition. The separate
DiT total (214.436 seconds) overlaps loading and denoising and must not be added
to them. Denoising is the first component to investigate, followed by video
decoding; operator-level device attribution within either phase is unmeasured.

The [native INT8 run](../results/local/h3/int8-seed42-001/run.json) reports
126.556 s denoising, 19.633 s DiT loading/preparation, 30.148 s video VAE,
4.761 s text encoding and 0.473 s audio VAE. Its smaller denoise time is partly
offset by weight preparation. These are sequential single-process samples
with warmed caches, not an independently repeated speedup or an audio-VAE
precision improvement. Both paths retain BF16 SDPA.

The 245.0886-second preflight includes a documented **190.1877-second CPU
preflight pause** for coordination with the other GPU task. The pause ended
before generation began and is excluded from the 252.9383-second generation
measurement. Full checkpoint hashing had already read the weights, and an
earlier shader probe had exercised compilation: this is a first full request
with warmed caches, **not a cold-start measurement**. It is also only one
request, not a resident-session latency distribution.

The [progress analysis](../results/local/h3/bf16-progress-001.json) excludes the
first 13.2507-second step interval and reports a 10.2579-second median over the
remaining 19 intervals. These are pipe-receipt intervals across different
denoising steps, not device timestamps or proven compilation-free identical
repetitions. Raw evidence: [run record](../results/local/h3/bf16-seed42-001/run.json),
[coordination pause](../results/local/h3/bf16-seed42-001/coordination-pause.json),
[media probe](../results/local/h3/bf16-seed42-001/exact-probe.log),
[five-second video](../results/published/h3-native-eval/fox-two-seeds-v1/videos/fox-winter-t2va-v1-seed42-bf16.mp4).

The kernel-library [public result package](../results/published/h3-native-eval/README.md)
now includes all six BF16/native INT8/H256 fox videos at seeds 42/43, a
[playable gallery](../results/published/h3-native-eval/fox-two-seeds-v1/gallery.html)
and immutable per-step latent/media evaluation JSON with source/checksum links.

## Contracts that affect interpretation

- The M5 runtime defaults to INT8 QKV, MLP and attention-output projections.
  Three explicit `--use-slower-bf16-*` flags establish the BF16 comparison.
- Both modes retain the upstream BF16 SDPA. INT8 projections do not imply
  low-bit attention. The BF16 comparison is the runtime's close-reference path,
  including BF16 final-head conversion; official pipeline parity is unverified.
- First invocation includes runtime compilation, component loading, weight
  preparation, encoding, denoising, both VAEs and MP4 output. This is deployment
  latency, not compilation-free kernel latency.
- A separate H256 resident session completed one warmup and three timed
  requests. Cached generation takes 153.602430, 153.325268 and 153.235321 s;
  synchronized denoise callbacks take 123.723213, 123.509807 and 123.386680 s.
  The corresponding medians are 153.325268 s and 123.509807 s. All four
  124-frame videos are byte-identical. These are same-seed descriptive
  repetitions, not an interleaved cross-mode speedup estimate.
  The audio VAE still creates a GPU context/compiles shader source and loads
  weights per request, so compilation-free end-to-end timing is not claimed.
  [Public resident record](../results/published/h3-native-eval/performance/hadamard-resident-seed42-001.json).
- The upstream cached DiT profile wall mark is not reset between generations.
  Its next `GPU Euler denoise wall` includes intervening decode/export time.
  The session driver therefore independently times synchronized denoise callback
  boundaries. Do not use the stale upstream wall field for cached denoise.
- Root-command GPU timing may not cover MPSGraph child buffers. Tracked Metal
  tensor peak excludes MPS internal allocations and CPU media buffers.
- Denoising's `encode=167.986s` includes host API work and does not establish a
  CPU bottleneck. Its `root-gpu=0.104s` is not total GPU time. Attribute device
  work with a trace before selecting a host-overhead or kernel optimization.

A later [target-process Metal trace summary](../results/local/h3/native-capture-seed42-001-trace-summary.json)
contains 1665 compute intervals in a partial diagnostic DiT window. Their
observed span is 10.870592458 s and their union is 10.773800753 s (99.1096% of
that observed span). The capture requested 15 seconds, but this ratio is not
utilization over that full requested window. It includes capture/profiling
overhead and has no QKV/SDPA/MLP labels. It supports the warning that root-GPU
counters omit substantial work; it does not establish a compute bottleneck
or rank operators. Raw traces/XML with process identifiers stay local.

## Source-derived candidates for profiling

The component wall times above are measured. The following operator-level
explanations and optimization choices remain hypotheses to test.

| Component at 9914 logical rows | Pinned runtime path | Experiment |
| --- | --- | --- |
| BF16 QKV and attention output | MPSGraph; specialized QKV stops at 2048 rows and automatic split-linear at 3072 | Test larger shapes/tails or an explicit 2048-row split, counting all dispatch and conversion costs |
| BF16 MLP | Fused MPSGraph FC1→SwiGLU→FC2 in default `qkv-attn` TensorOps mode | Compare with native tensor GEMMs at real shapes; preserve fused activation semantics |
| INT8 FC2 | 14 groups of 1024 activation features, group-scaled accumulation | Compare upstream grouped INT8, upstream row INT8, H256+row INT8, standalone FP8 and BF16 |
| SDPA | Dense noncausal BF16 MPSGraph, 56 heads × 128 | Matched MPSGraph/MLX/pinned-MFA comparison with real layout and total conversion cost |
| Video VAE | FP32 transformer decoder, tiled/chunked | Profile and sweep tile geometry with output-difference checks; then consider precision separately |
| Repeated requests | Exact prompt/DiT/video-decoder cache, per-request audio VAE setup | Measure startup, cached denoise, decode and encoding independently; extend caching where measured setup cost warrants it |

The full dense QK/PV arithmetic is about 2.818 TFLOP per block at this shape.
Projection arithmetic is QKV 2.292, output 0.764, FC1 3.056 and FC2 1.528 TFLOP.
Attention is about 26.9% of this limited arithmetic total, **not** a measured
latency fraction. Do not assume it dominates the low-resolution request simply
because dense attention dominates at larger resolutions.

The SDPA source uses Apple's attention API and does not explicitly allocate
a full `[56,9914,9914]` score matrix. Its hypothetical BF16 size is 10.25 GiB;
that is not evidence of actual allocation. A trace is required to identify the
internal algorithm and materialized layout conversions.

The video decoder selects 288-pixel tiles at 512 square: four spatial tiles,
seven temporal chunks, and 36 FP32 blocks per tile/chunk, giving 1008 SDPA API
calls with sequence 2273, matching the full run's 1008 reported attention calls.
The decoder phase takes 30.218 seconds, about 12% of the process; the share
inside its attention, projections and other operations remains unmeasured.
See the pinned
[VAE source](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_video_vae.c)
and [dispatch source](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m).

## Alignment with the kernel experiment

Model inference and the companion kernel experiment share one GPU lock to avoid concurrent GPU measurements.
Its existing [Hadamard evidence](hadamard.md) uses synthetic T=4096 operands.
H256 reduces FFN-down outlier relative L2 from 8.84% to 1.22%; its complete
online path is 8.701 ms versus 12.707 ms BF16 on the recorded machine.
These numbers justify a candidate, not a prediction for this 9914-token video.

The newer [Q/K RMS/RoPE epilogue](qk-rope-fusion.md) repeats about 18% lower
online latency at synthetic M9914 with exact paired outputs. The
[gated projection epilogue](gate-epilogue-fusion.md) compares complete
two-projection chains and reports 0.3–4.6% lower latency across M4096/M9914,
preserving the BF16 rounding boundaries and residual side output. These are
matched candidates for a native integration experiment. The pinned runtime
already fuses grouped QKV/RMS/RoPE; the single-Q/K experiment is not a new
upstream capability or proof of a native speedup. Its grouped layout, real
activation distributions and complete block costs still need validation.

Real post-SwiGLU FC2 inputs have now been captured at steps 0/10/19 and blocks
0/24/49, with all 14336 features for each selected row. Real QKV input remains
to be captured. Keep implementation
error against decoded operands separate from total error against BF16 inputs.
The Hadamard candidate rotates weights offline and activations online; FC2
rotation must be after SwiGLU. FP8 is a separate candidate, without Hadamard.
An AdaLN/RMSNorm+rotation+quantizer fusion must reproduce the reference's BF16
rounding boundary or be declared a distinct precision policy.

The optional [diagnostic patch](../integrations/h3/diagnostics.md) can capture
actual branch/shape and post-Euler latent finite statistics without adding
GPU waits. It was not enabled for the original BF16 run. The later separate
FC2 capture patch now supplies actual fox shapes and finite checks for all
20 post-Euler video/audio states and final VAE inputs across seeds 42/43.
Its captured BF16/native seed42 exports match the original MP4 hashes exactly.
The diagnostic patch's per-operation `host_api_ms` is
CPU enqueue time, not GPU time.
Post-Euler latents cannot substitute for FC2 activation calibration. Use a
separate Instruments capture for device-time attribution, then promote only
after matched quality checks, another seed and an independent same-hardware run.

The [paired evaluation report](h3-evaluation.md) now includes both seeds:
H256 FC2 reduces final video/audio latent relative L2 against BF16 and improves
decoded RGB/PCM similarity versus native INT8 in both cases. Some intermediate
steps and sampled FC2 rows regress; these remain recorded. The historical
strict every-row gate is not the user's current adoption rule. The current
decision considers aggregate error, tail risks, multiple seeds and complete
video/audio quality together, without inventing a universal SNR threshold.
Seed42/43 RGB SNR improves from 12.7128/17.3740 dB to 13.2407/19.6092 dB;
RGB SSIM improves from 0.690501/0.833956 to 0.704100/0.848317. Stereo PCM SNR
improves from 13.9618/11.8119 dB to 15.2909/13.5560 dB. These are matched
BF16-relative drift measurements, not perceptual acceptance or new throughput
numbers. Original BF16 is retained, and the derived INT8/H256 checkpoint has
been exported and assembled for permanent local retention. Its manifest SHA256
is `6840c91f6e5b9fcb4e8fe25292a1b1e2209d2f7f9c02e3427cfbbcbc71da4cda`.
All 200 projections have persisted codes/scales; H256 applies to FC2, while
QKV/output/FC1 keep their native INT8 policy. Offline preparation itself takes
7.366674 s, separately from weight loading (1.365344 s), original tensor
hashing (45.582728 s), and readback/hash/write (12.995065 s). Full-model
preflight and payload verification are additional costs. The completed
[prepared seed42 replay](../results/local/h3/prepared-replay-seed42-001/run.json)
loads all 200 projections with zero runtime weight rotation/quantization.
All 42 captured latent tensors and both original/exact-duration MP4s are
byte-identical to the on-load H256 reference. This establishes replay parity
for that request, not official-pipeline or perceptual quality. The first
[native fusion campaign](../results/published/h3-native-eval/fused-campaign-001-v1/README.md)
passes all 12 operator numerical cases. H256 FC2 has no stable speed gain;
independent timing repetition, whole-model fusion parity/E2E and the additional
five-/fifteen-second prompt pairs remain pending.

## Integration plans after profiling

1. **Dispatch at the real sequence length.** Add a measured shape dispatch
   table in the native backend. Test QKV/output row splitting, then port the
   winning TensorOps tile to the same C/Metal buffer contract. Include the
   9914-row tail and any 9984-row INT8 workspace, launches and layout changes.
   Accept only if decoded implementation checks pass and the complete warm
   projection/block latency improves in an independent repeat.
2. **H256 INT8 with preparation fusion.** Rotate/quantize immutable weights
   once when preparing the resident DiT, or persist a separately versioned
   derived-weight cache keyed by original hashes, rotation, dtype and scale
   contract. Port the repository's Metal shader into the native runtime;
   calling the Python/MLX wrapper across a buffer-copy boundary is not a
   zero-copy integration. For QKV and gate/up, fuse the appropriate
   residual/norm/modulation boundary, then rotation and dynamic quantization.
   For FC2, capture and rotate post-SwiGLU values instead. Preserve residual
   output and the runtime's BF16 rounding/scaling semantics. Compare against
   its existing group1024 INT8 rather than only plain row-INT8.
3. **Attention and decoder optimization.** If the measured device-time share
   warrants it, integrate a pinned attention kernel with the exact packed
   noncausal layout and scaling contract. Independently sweep VAE tiles and
   staging buffers. Treat BF16/FP8 VAE or low-bit attention as new precision
   policies, with finite/reference and visual/audio checks, not transparent
   dispatch substitutions.
4. **Serving overhead.** Cache prepared contexts where repeated-request
   evidence shows setup cost, especially audio VAE. Retain prompt/seed
   reproducibility, memory limits and explicit cache keys. Report first
   request and resident request separately; exclude runtime compilation only
   from a verified warm timing scope.
