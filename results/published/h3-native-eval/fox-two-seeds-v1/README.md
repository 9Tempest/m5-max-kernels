# H3 generated samples and paired evaluation

[Open local video gallery](gallery.html) · [Machine-readable manifest](manifest.json)

512×512, 5 seconds, 24 fps, 20 Euler passes and all 50 DiT blocks. Each comparison uses the same prompt and seed. Original native output has 124 frames; these separate exports contain exactly 120 frames.

BF16 is the pinned h3.c close-reference implementation, not official-pipeline or perceptual ground truth. H256 rotates only post-SwiGLU FC2 before INT8 packing; QKV, attention output and FC1 retain native INT8. Other tensors retain their original precision.

| Benchmark / seed | Mode | Preview / video |
|---|---|---|
| fox-winter-t2va-v1 / 42 | BF16 | [![Preview](posters/fox-winter-t2va-v1-seed42-bf16.jpg)](videos/fox-winter-t2va-v1-seed42-bf16.mp4) |
| fox-winter-t2va-v1 / 42 | Native INT8 | [![Preview](posters/fox-winter-t2va-v1-seed42-upstream-int8.jpg)](videos/fox-winter-t2va-v1-seed42-upstream-int8.mp4) |
| fox-winter-t2va-v1 / 42 | INT8 + H256 FC2 | [![Preview](posters/fox-winter-t2va-v1-seed42-hadamard-fc2.jpg)](videos/fox-winter-t2va-v1-seed42-hadamard-fc2.mp4) |
| fox-winter-t2va-v1 / 43 | BF16 | [![Preview](posters/fox-winter-t2va-v1-seed43-bf16.jpg)](videos/fox-winter-t2va-v1-seed43-bf16.mp4) |
| fox-winter-t2va-v1 / 43 | Native INT8 | [![Preview](posters/fox-winter-t2va-v1-seed43-upstream-int8.jpg)](videos/fox-winter-t2va-v1-seed43-upstream-int8.mp4) |
| fox-winter-t2va-v1 / 43 | INT8 + H256 FC2 | [![Preview](posters/fox-winter-t2va-v1-seed43-hadamard-fc2.jpg)](videos/fox-winter-t2va-v1-seed43-hadamard-fc2.mp4) |

## Evaluation records

All linked evaluation JSON files are copied byte-for-byte. Their checksums, sample associations, original run digests and video hashes are in the manifest. Raw latent arrays and weights are not included.

- [media: evaluations/eval-bf16-int8-seed42-002-evaluation.json](evaluations/eval-bf16-int8-seed42-002-evaluation.json)
- [media: evaluations/eval-bf16-hadamard-seed42-root-001-evaluation.json](evaluations/eval-bf16-hadamard-seed42-root-001-evaluation.json)
- [media: evaluations/eval-bf16-native-seed43-001-evaluation.json](evaluations/eval-bf16-native-seed43-001-evaluation.json)
- [media: evaluations/eval-bf16-hadamard-seed43-001-evaluation.json](evaluations/eval-bf16-hadamard-seed43-001-evaluation.json)
- [latents: evaluations/latent-threeway-seed42-001-latent-evaluation.json](evaluations/latent-threeway-seed42-001-latent-evaluation.json)
- [latents: evaluations/latent-threeway-seed43-001-latent-evaluation.json](evaluations/latent-threeway-seed43-001-latent-evaluation.json)

SNR/relative L2 quantify drift from BF16; RGB SSIM is FFmpeg's planar-RGB variant. These metrics do not establish human preference, prompt adherence or motion quality. FVD/LPIPS/VBench remain explicitly unmeasured. Diagnostic capture runs are not latency samples; use the separately reported warm performance experiment.
