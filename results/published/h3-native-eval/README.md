# Native H3 evaluation results

[Six playable fox samples, two seeds](fox-two-seeds-v1/README.md) · [Local video gallery](fox-two-seeds-v1/gallery.html) · [Sample/config/checksum manifest](fox-two-seeds-v1/manifest.json)

[Fifteen-second whale-maid BF16/H256 pair](whale-maid-15s-v1/README.md) · [16:9 video gallery](whale-maid-15s-v1/gallery.html)

The whale-maid pair uses the complete Chinese prompt and the same first-frame
reference, seed83, 512×288, 24 fps, 20 Euler steps and all 50 DiT blocks. Both
exact-duration exports contain 360 frames. The original reference image remains
local; its digest and conditioning identity are retained. All 42 captured
latent tensors are finite in both paths.

| Fifteen-second measurement | BF16 | Prepared INT8 + H256 FC2 | BF16 / H256 |
|---|---:|---:|---:|
| Generation process, seconds | 506.275859 | 364.855037 | 1.387608× |
| Denoise phase, seconds | 443.527 | 293.568 | 1.510815× |
| Video VAE decoder, seconds | 43.918 | 43.951 | 0.999249× |
| Tracked DiT allocation peak, GiB | 38.199 | 20.967 | — |

This saves 141.420822 seconds (27.93%) in this single diagnostic pair. Process
time includes loading, generation, latent capture and native media export;
checkpoint preflight, build and the separate exact-duration export are outside
it. This is not a warmed repeated speedup or a compilation-free end-to-end
measurement. The tracked allocation peak is not process RSS.

Against BF16, H256's final video latent relative L2 is 0.211950 and final audio
latent relative L2 is 0.315935. Decoded RGB SNR is 15.6932 dB and RGB SSIM is
0.818705. PCM SNR is only 0.4392 dB: waveform drift is substantial despite
strict duration/sample alignment. No audio listening score or perceptual pass
is claimed. Sparse frames show the thinking/eating/return poses in both clips,
with different hand/spoon trajectories; they do not verify precise comic timing
or continuous motion. See the package's immutable paired evaluations and the
[evaluation protocol](../../../docs/h3-evaluation.md).

The fox sample package contains BF16, native INT8, and INT8 + H256 FC2 videos for seeds 42 and 43, alongside immutable media and 20-step latent evaluation JSON. Each video is 512×512, 120 frames, 24 fps and exactly five seconds. The native 124-frame output is retained locally. No weights or latent arrays are redistributed.

For both seeds, H256 improves final video/audio latent drift and decoded media similarity relative to the same-seed BF16 run. BF16 here is the pinned h3.c close-reference path; this is not evidence of official-pipeline parity or perceptual superiority.

| Seed | Candidate | Final video latent SNR (dB) | Final audio latent SNR (dB) | RGB SNR (dB) | RGB SSIM | PCM SNR (dB) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | Native INT8 | 8.5720 | 18.1491 | 12.7128 | 0.690501 | 13.9618 |
| 42 | INT8 + H256 FC2 | 9.4786 | 18.7734 | 13.2407 | 0.704100 | 15.2909 |
| 43 | Native INT8 | 13.0824 | 24.1199 | 17.3740 | 0.833956 | 11.8119 |
| 43 | INT8 + H256 FC2 | 13.8413 | 28.9984 | 19.6092 | 0.848317 | 13.5560 |

Higher SNR/SSIM means closer to BF16 under the specified metric, not necessarily better perceived video. RGB SSIM is FFmpeg's planar-RGB implementation. Audio is decoded 32 kHz stereo PCM; PSNR in the raw records uses full-scale range 2.0. Latent SNR and relative L2 are primary; latent PSNR's explicit range 1 is only a scaling convention.

The full trajectories are mixed: H256 video latent relative L2 is lower at 18/20 steps for seed42 and 17/20 for seed43; audio improves at 12/20 and 19/20. The projection calibration likewise retains individual-row and maximum-error regressions. The user accepted aggregate benefit with these risks visible; implementation tolerances remain unchanged.

All sample runs contain diagnostic readback and are **not performance samples**. See the [deployment report](../../../docs/h3-local-inference.md), [precision and checkpoint contract](../../../docs/h3-hadamard-native.md), and [metric definitions and research](../../../docs/h3-evaluation.md) for the separate performance scope, source pins, references and limitations. LPIPS, VBench and FVD are explicitly unmeasured.

A separate resident H256 run completes one full warmup and three repeated generations: **153.602430 / 153.325268 / 153.235321 s** for the cached full request, and **123.723213 / 123.509807 / 123.386680 s** for the synchronized denoising callback. All four native videos have identical bytes. These are one mode and one seed, without an interleaved performance comparison. Full-request time still includes repeated audio VAE setup; compilation-free full-request latency is not claimed. [Raw timing summary](performance/hadamard-resident-seed42-001-v2.json).

Both the original BF16 checkpoint and a separate prepared INT8 + H256 checkpoint are retained locally. Reloading all 200 prepared projections requires no weight rotation or requantization. A seed42 replay matches all 42 captured latent tensors and both video exports byte-for-byte. Verified payload import takes 12.721376 s in this single diagnostic run; it is separate from offline preparation and full checkpoint hash verification. [Checkpoint and replay evidence](prepared-checkpoint-summary-v1.json). Model weights are not included in this repository.

Three additional paired scenes—person turning, rainy street and pouring water—are defined in the [pinned catalog](../../../integrations/h3/benchmarks.json). Their result packages will be added after actual generation and evaluation. The [first native fused-GEMM campaign](fused-campaign-001-v1/README.md) passes all 12 numerical cases and records alternating online timings. H256 FC2 has no stable gain; whole-model parity/E2E and independent timing repetition remain pending; upstream grouped QKV and gate→AdaLN→quant fusion remain the baseline.
