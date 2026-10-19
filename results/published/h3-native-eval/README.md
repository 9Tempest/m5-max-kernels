# Native H3 evaluation results

The [QKV and AdaLN/H256 integration package](kernel-integration-v1/README.md)
adds six candidate videos, two-seed all-policy numerical evaluation, independent
operator campaigns and matched resident timing. Both requested kernel families
are connected to actual H3 call sites. The new flags remain off: the custom
QKV operator is slower than native grouped fusion, and QKV/FC1 H256 increases
whole-model drift despite improving sampled local projection error.
[Measured results and limitations](kernel-integration-v1/results.md).

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

The [three additional paired scenes](diversity-three-scenes-v1/README.md)—person turning, rainy street and pouring water—are complete, with six playable videos and immutable latent/media evaluations. Each uses its [pinned prompt and seed](../../../integrations/h3/benchmarks.json), 512×512, 120 frames, 24 fps, 20 steps and all 50 blocks. H256 uses the separately prepared checkpoint. Values below compare H256 against the corresponding BF16 output.

| Scene / seed | Final video latent SNR (dB) | RGB SNR (dB) | RGB SSIM | PCM SNR (dB) |
|---|---:|---:|---:|---:|
| Person turning / 71 | 20.8408 | 20.6638 | 0.912221 | 12.2206 |
| Rainy street / 73 | 10.3929 | 6.3140 | 0.638439 | 7.0216 |
| Pouring water / 79 | 8.0859 | 11.7725 | 0.718942 | 3.7082 |

Quantization drift varies substantially by scene. Six sampled frames per video show the broad person-turn and water-filling stages, with pose, framing and detail differences. Neither rainy clip clearly satisfies the requested left-to-right car crossing. Full motion and audio listening remain unscored; these records retain `quality_decision=not_evaluated`.

[Two independent native fused-GEMM campaigns](fused-comparison-001-002-v1/README.md) pass all 24 numerical cases and record alternating online timings. H256 FC2 has no consistent gain; attention output at M9914 improves in both seeds and both campaigns. The [full-model parity check](fused-model-parity-seed42-v1/README.md) confirms that attention-only and combined attention/FC2 fusion each match all 42 original H256 latent tensors and both video exports byte-for-byte. These experiments use the on-load H256 path; prepared-checkpoint loading and fusion have not been combined. Upstream grouped QKV and gate→AdaLN→quant fusion remain the baseline.

The [matched resident comparison](fused-resident-seed42-v1/README.md) keeps the same frozen runtime and H256 policy, with attention-output gate fusion disabled/enabled. Each arm completes one full warmup and three timed requests; all eight native 124-frame videos are byte-identical.

| Resident median, seed42 | Fusion off (s) | Attention fusion (s) | Baseline / candidate | Lower latency |
|---|---:|---:|---:|---:|
| Complete cached request | 151.126785 | 147.752793 | 1.022835× | 2.23% |
| Synchronized denoise callback | 121.909995 | 118.566479 | 1.028200× | 2.74% |

These are descriptive results from one sequential process per policy, one seed and three repetitions. A second seed, reversed process order and independent reruns are still required before stable performance promotion. Full-request timing includes repeated audio VAE setup and is not compilation-free. Both fusion flags remain off by default.

The [runtime-config audit](performance/runtime-config-audit-15s-v1.json) checks the current original and derived VAE configs against the pinned model inventory. The [audio-duration audit](performance/audio-duration-audit-v1.json) checks the stored decoded sample counts for all five published media evaluations against their declared durations. These are explicit posthoc checks; the historical raw records remain unchanged. The updated runners and evaluators verify runtime VAE configs, bind decoded media to its recorded digest, and require exact sample duration.
