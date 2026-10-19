# Native H3 FC2 Hadamard experiment

The optional native implementation rotates **FC2 only**, after SwiGLU. It does not implement a fully Hadamard-rotated H3 model. Across two seeds, all 18 real-activation samples improve aggregate relative L2 and RMSE over native group1024. Seed42 has one sampled-row L2 regression; seed43 improves every sampled row but has one aggregate maximum-error regression. The user accepted evaluating these aggregate benefits together with tail risk and full-model output, so neither observation is an automatic veto. The latest checkpoint requirement is to keep both the BF16 original and a separately loadable INT8 + Hadamard checkpoint permanently. No original checkpoint deletion is planned.

Upstream is [`antirez/h3.c` at `8974cc055ea9c02fcd14cc27dfda3e1027c05153`](https://github.com/antirez/h3.c/tree/8974cc055ea9c02fcd14cc27dfda3e1027c05153). Model files are [`MiniMaxAI/MiniMax-H3` at `42ed227ee7df40d41602854ae760620d6eb651fe`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe). The runtime's MIT license and model's separate license still apply; the repository does not distribute weights.

## Precision contract

For every active DiT block, the experimental FC2 path uses the following operations:

1. At model loading, rotate each original BF16 FC2 weight row using the normalized, plain Sylvester H256 transform. Butterflies and normalization use FP32, without signs or a transformed BF16 rounding boundary. Quantize each complete rotated row to symmetric INT8 with a single FP32 maximum-absolute-value scale.
2. At inference, read the existing **BF16 SwiGLU output**, apply the same FP32 H256 transform, then quantize each complete activation row. A single Metal dispatch implements this preparation through SIMD shuffles and register butterflies, without Python/MLX transfers or a temporary FP32 device tensor.
3. Reuse the native FC2 INT8 TensorOps projection: INT32 dot products, FP32 scales, then BF16 projection output. Continue with the original residual gate and normalization.

FC1, post-AdaLN BF16 rounding, SwiGLU BF16 storage, residual updates, and Q/K normalization/RoPE are unchanged. Rotations do not move through SwiGLU. The path supports the actual FC2 width K=14336 and preserves zero padding to 128-row tiles. Zero H256 rows use codes 0 and scale 1. The native quantizer uses scale 1/127 for genuine zero rows and scale 1 for padded rows; decoded zeros are identical.

The baseline is the deployed native **activation group1024** FC2, with 14 activation scales per row and one scale per output weight row. An additional native plain-row baseline is retained as an ablation. H256 whole-row must outperform group1024 to be considered for adoption; beating plain-row alone is insufficient. A future H256+group1024 policy would require a distinct mode, corresponding reference, and measured evidence.

`H3_HADAMARD_FC2=1` opts in. The default is unchanged. Unsupported TensorOps, invalid K, conflicting row/stage/clipping overrides, and compilation failures are explicit failures. No fallback is promoted as an experiment result. This version does not persist rotated weights: their preparation is included in model loading.

## Pinned implementation and build

The integration uses a separate checkout, preserving the original deployment. The patch changes `h3_dit.c`, `h3_gpu.h`, `h3_gpu.m`, and `h3_shaders.metal`.

| Artifact | SHA256 |
|---|---|
| [hadamard.patch](../integrations/h3/hadamard.patch) | `84065e6139c842aee2cb6d037a6c72ecfa9895e07b1696e9bf638bde3a5a8d1e` |
| Tracked source digest, using `run.git_provenance` | `833076975eeaaf14b0a9db01f6237d7de77930034222b1dd3b85cca7721f04ec` |

From the kernel repository root:

```sh
git clone https://github.com/antirez/h3.c .cache/h3.c-hadamard
git -C .cache/h3.c-hadamard switch -c codex/h3-hadamard 8974cc055ea9c02fcd14cc27dfda3e1027c05153
git -C .cache/h3.c-hadamard apply ../../integrations/h3/hadamard.patch
make -C .cache/h3.c-hadamard -j8
uv sync --locked
```

Use a new path if the checkout already exists. The standalone runner validates the complete patched source digest and patch digest, rejects untracked source additions, then rebuilds the executable from the pinned source. It also verifies every required model weight against the pinned manifest. This reads the weights before timing and warms the filesystem cache.

Build the independent projection readback driver:

```sh
mkdir -p .build
clang -O2 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Werror \
  -I .cache/h3.c-hadamard integrations/h3/fc2_validate.c \
  .cache/h3.c-hadamard/libh3.a -framework Foundation -framework Metal \
  -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
  -framework Accelerate -licucore -lm -o .build/h3-fc2-validate
```

The model component directory for the driver is **`FL2VA/transformer`**, and its weight key is `blocks.<block>.mlp.fc2.weight`.

## Full generation and diagnostic capture

[run_fc2_experiment.py](../integrations/h3/run_fc2_experiment.py) fixes the fox-in-snow prompt, 512×512 canvas, 20 Euler passes, all 50 blocks, seed, reference RoPE, and reuse/core-reuse of 1. Five requested seconds align to 124 native frames at 24 fps; the original is retained and a separate 120-frame/5-second video is exported and verified. The GPU lock covers source/model preflight, compilation, the entire process, capture checks, and media validation.

Use new immutable output directories for every invocation:

```sh
uv run python integrations/h3/run_fc2_experiment.py \
  --mode upstream-int8 --seed 42 --capture-fc2 --capture-latents \
  --output-dir results/local/h3/fc2-native-capture-seed42-NEW

uv run python integrations/h3/run_fc2_experiment.py \
  --mode bf16 --seed 42 --capture-latents \
  --output-dir results/local/h3/fc2-bf16-capture-seed42-NEW

uv run python integrations/h3/run_fc2_experiment.py \
  --mode hadamard-fc2 --seed 42 --capture-latents \
  --output-dir results/local/h3/fc2-hadamard-capture-seed42-NEW
```

`--gpu-wait-seconds 900` permits waiting for another experiment. The three BF16 projection overrides are applied automatically in BF16 mode. Do not inherit arbitrary `H3_*` variables; the runner rejects them. `--capture-fc2` is restricted to an INT8 FC1 trajectory and is incompatible with BF16 mode.

FC2 captures sample steps 0/10/19 and blocks 0/24/49. Each record contains complete K=14336 rows: each packed modality's first/middle/last row, plus seven uniformly spread rows, with duplicates removed. Metadata records row IDs and modality boundaries. These are BF16-stored post-SwiGLU activations on the native INT8 FC1 trajectory, not activations from a fully BF16 model. Capturing adds nine explicit submit/wait/restart points.

Latent capture records every completed Euler step in packed FP32 model coordinates and the final unpacked VAE inputs. For this recipe, final video is `[24,37,32,32]` and final audio is `[32,2,207]`. It uses existing completed GPU windows, adding CPU readback/I/O. Metadata contains shapes, precision modes, completed step count, final sigma, timestamps, and the caller-declared seed, which the runner validates against its request manifest. Every binary is checked for expected size and finite values. All files are written exclusively.

**Capture runs are diagnostic and cannot serve as latency samples.** A performance invocation omits both capture flags. Its timing still includes shader setup, loading, weight preparation, text encoding, denoising, VAE, and export; it is not compilation-free resident or kernel-only latency. Build and full-weight hash preflight are recorded separately.

## Independent FC2 references

[fc2_validate.c](../integrations/h3/fc2_validate.c) calls the same native FC2 dispatcher as full inference. It accepts the original component directory, block number, frozen input BF16 file, a new empty output directory, logical M, and shader path. It evaluates native group1024, H256 whole-row, and native plain-row against the **same frozen X/W**. Run the driver only while holding `m5kernels.locking.gpu_lock()` for the complete child process; the full-generation runner does not hold a lock after it returns.

The driver computes all 5376 output channels. It dumps all output values and activation codes/scales, plus 32 uniformly selected weight/output channels including both boundaries. Full K is retained for each reference. When logical M exceeds captured rows, the remaining input rows are zero; synthetic validation additionally exercises nonzero last rows of partial tiles.

[fc2_analyze.py](../integrations/h3/fc2_analyze.py) runs on CPU:

```sh
uv run python integrations/h3/fc2_analyze.py NEW_READBACK_DIRECTORY \
  --output NEW_ANALYSIS.json
```

The analyzer separates:

- **Implementation error:** GPU output versus independent NumPy float64 products of the actual decoded INT8 operands. BF16-output relative-L2 gate is 0.008.
- **Quantization error:** decoded products versus float64 products of the original BF16 X/W.
- **Total error:** actual BF16 GPU output versus the original-operand reference.

It independently checks normalized H256 dot preservation, maximum-absolute-value scales, quantization cells, finite outputs, exact zeros, and padded rows. Exact CPU-reference code mismatches are also reported; a predefined 0.0001 quantization-cell allowance covers FP32 divide/reciprocal rounding. Every output is scanned for finite values, but independent projection error covers **32 selected output channels**, not every channel or the full model. The analyzer records its own source hash and all readback hashes.

## Promotion gate and evidence

The original predeclared policy required every sampled step×block to improve aggregate relative L2, RMSE and maximum absolute error, with no sampled-row L2 regression. The immutable reports retain that historical gate and its failure. After reviewing the results, the user explicitly relaxed the decision policy: aggregate numerical benefit may justify adoption when tail/per-row risk, final latent/media quality and an independent seed are considered together. A single sampled-row regression is no longer an automatic veto. Implementation tolerances remain unchanged; this is a user-approved adoption-policy change, not an altered correctness threshold.

Projection sampling is necessary but insufficient. The remaining evaluation includes independent seeds, same-seed denoiser trajectories and final video/audio latents against BF16, and visual/audio quality. Final numerical comparisons use exported FP32 latents rather than MP4 pixels to avoid conflating model error with VAE/media encoding. The analyzer always reports `checkpoint_replacement_eligible=false` because its scope is only projection sampling; no analysis or campaign script deletes weights. The original BF16 checkpoint remains permanently available under the user's latest requirement, including after derived loading and numerical evaluation pass.

The first GPU correctness campaign passed all three policies on actual block-0 FC2 weights with synthetic BF16 inputs: seeds 17/29, M=129/257, explicit zero rows, outliers, and nonzero final logical rows. [Raw campaign](../results/local/h3/fc2-synthetic-001/campaign.json). Implementation relative L2 was about 0.0016–0.0017, within the unchanged BF16-output gate. H256 total error was 1.35% for each seed, versus native group1024's 1.75% and 1.86%. These are synthetic distribution checks.

The [seed42 native capture](../results/local/h3/fc2-native-capture-seed42-001/run.json) completed with all 20 Euler states and nine activation samples. Its five-second MP4 was byte-identical to the original native INT8 run (`993e73621f4028c9759058457f0f41ee2a06dd865bf8691de28f06baddcc3020`), supporting that the default-off instrumentation preserved that request's output. It does not establish a timing equivalence.

The [real-activation calibration campaign](../results/local/h3/fc2-calibration-seed42-001/campaign.json) evaluates each frozen sample with identical X/W. All three policies passed packing/implementation checks. Each of nine samples contains 16 real rows and 32 selected output columns, with full K=14336. The projection driver uses logical M=129, filling the remaining rows with zeros and padding to 256; this numerical experiment is **not** a 9914-row projection performance measurement.

| Euler step | DiT block | Native group1024 total relL2 | H256-row total relL2 |
|---|---:|---:|---:|
| 0 | 0 | 1.6364% | 1.3874% |
| 0 | 24 | 2.7985% | 1.1384% |
| 0 | 49 | 1.7331% | 0.7984% |
| 10 | 0 | 1.7523% | 1.3658% |
| 10 | 24 | 2.8230% | 1.3132% |
| 10 | 49 | 2.3042% | 0.7780% |
| 19 | 0 | 1.8729% | 1.1293% |
| 19 | 24 | 2.6001% | 1.2423% |
| 19 | 49 | 1.7438% | 0.6201% |

All nine aggregate relative-L2/RMSE/max-absolute metrics improve. Across their 144 input rows, 143 row L2 errors improve and one worsens. An independent CPU recomputation from the immutable BF16 operands and GPU outputs identifies that row as **step19/block49, sample index14, packed row7434**, within video segment `[442,9914)`. It is local video row6992, or zero-based temporal index27 and spatial patch `(y=5,x=0)` on the 37×16×16 packed video grid.

For its 32 sampled output channels, native group1024 → H256-row total relative L2 is **0.71093% → 0.81929%**, RMSE **19.58976 → 22.57586** (15.24% higher), and maximum absolute error **61.10614 → 107.06275**. This is the recorded tail-risk example; it must remain visible alongside the aggregate gains. These observations do not establish full-channel or full-model numerical dominance.

The independent seed43 campaign is recorded in `results/local/h3/fc2-calibration-seed43-001/campaign.json`, using the same frozen implementation and sampling contract. All three policies again pass implementation/reference, finite-value and padding checks in all nine cases. H256 improves aggregate relative L2 and RMSE in all nine samples, and improves L2 on **144/144 sampled rows**.

| Step | Block | Seed43 native group1024 relative L2 | Seed43 H256-row relative L2 |
|---|---|---:|---:|
| 0 | 0 | 1.6529% | 1.3511% |
| 0 | 24 | 3.0256% | 1.3032% |
| 0 | 49 | 1.9599% | 0.7856% |
| 10 | 0 | 1.9405% | 1.1958% |
| 10 | 24 | 2.8076% | 1.2398% |
| 10 | 49 | 1.4982% | 0.7848% |
| 19 | 0 | 1.9000% | 1.1504% |
| 19 | 24 | 2.6563% | 1.0121% |
| 19 | 49 | 1.4837% | 0.7145% |

The seed43 tail exception is **step0/block0 maximum absolute error: 65.295151 → 79.223693** (21.33% higher), even though this sample's aggregate relative L2 improves 1.65288% → 1.35107% and RMSE improves 6.31817 → 5.16450. The other eight maximum-error metrics improve. Consequently, the historical all-metrics strict gate also fails on seed43; the current aggregate-benefit review preserves this evidence without changing any implementation tolerance. Two seeds cover 288 sampled rows and 32 output columns per row with full K, not every token/output channel or a broad generative-quality benchmark.

## Reproduce numerical campaigns

[fc2_campaign.py](../integrations/h3/fc2_campaign.py) packages both scenarios. It validates source/patch hashes, verifies the original model files, rebuilds a private numerical driver, holds the GPU lock through the entire campaign, and records each failed case without overwriting earlier evidence. Synthetic correctness uses the default seed17/M129 and seed29/M257 scenarios above. Calibration verifies that captured tensors still match the original run's hashes before reading them.

```sh
uv run python integrations/h3/fc2_campaign.py --scenario synthetic \
  --output-dir results/local/h3/fc2-synthetic-NEW

uv run python integrations/h3/fc2_campaign.py --scenario calibration \
  --capture-run results/local/h3/fc2-native-capture-seed42-001 \
  --output-dir results/local/h3/fc2-calibration-seed42-NEW
```

The optional repeatable `--synthetic-case SEED:ROWS` selects explicit synthetic cases; it cannot be combined with calibration. A campaign can complete successfully while numerical adoption remains pending. The current campaign records the historical strict gate separately from the user's aggregate-benefit review policy.

Host C compilation, offline Metal 4 shader compilation, and 13 CPU reference/runner tests passed before the generation capture. The portable campaign adds 11 CPU parameter/provenance/failure-preservation tests. Source compilation alone was not counted as GPU correctness.

## Persistent mixed-precision checkpoint

The user now requires permanent retention of **both** the original BF16 checkpoint and the derived INT8 + Hadamard checkpoint. The prepared checkpoint contains all 200 native DiT projection matrices: QKV, attention output and FC1 use the original native per-output-row INT8 quantizer, and FC2 uses normalized plain H256 followed by per-output-row INT8. All online activation policies stay unchanged. Only FC2 rotates; this is not a fully Hadamard model. Norms, modulation, residual boundaries, final projections, text encoder and VAE retain their original precision.

[prepared_checkpoint.py](../integrations/h3/prepared_checkpoint.py) exports with the frozen H256 producer above, through [prepared_export.c](../integrations/h3/prepared_export.c). It does not route weights through Python/MLX. Each projection saves native INT8 codes and FP32 scales with file SHA256, original BF16 tensor SHA256, original shard SHA256, tensor key/dtype/shape/offset, transform/rounding/layout, and the model/source/patch pins. Payloads are checked for finite positive scales and the symmetric `[-127,127]` code range. The manifest is published only when all 200 projections pass. Partial exports remain explicit failures.

[derived_checkpoint.py](../integrations/h3/derived_checkpoint.py) then creates `model/` alongside `weights/`. It removes those 200 BF16 projection entries from newly rewritten transformer shards and updates their index; every retained tensor is copied byte for byte, with its dtype, shape, original/new offset and tensor-byte SHA256 recorded. Unchanged locally downloaded model files are hardlinked where possible, otherwise copied. The original files are never modified or deleted. The resulting `derived-checkpoint.json` binds the entire model file inventory and prepared manifest; it is written after full readback verification. This format is specific to the patched native loader, not a generic Transformers checkpoint.

The native import extension is isolated in `.cache/h3.c-hadamard-cache`. Reconstruct it from the same upstream revision using **only** [prepared-checkpoint.patch](../integrations/h3/prepared-checkpoint.patch), which already contains the H256 changes. Its patch SHA256 is `1cad08a5bddc2dfa37c29d2c00551fe467a032d358d1287072f93bfd9dc74c67`; source digest is `054ec73eb454e44dc75d71bf83fa53a7129e5bd1e30fb304236f545f92f9cd7e`. The default remains disabled. Native import requires `H3_HADAMARD_FC2=1`, `H3_PREPARED_DIT_CACHE` pointing to the prepared manifest, and its exact digest in `H3_PREPARED_DIT_MANIFEST_SHA256`; conflicting precision overrides are rejected. It validates every manifest entry and each loaded payload, skips the removed BF16 projection weights, and performs no second weight transform or quantization.

```sh
uv run python integrations/h3/prepared_checkpoint.py \
  --output-dir checkpoints/MiniMax-H3/int8-hadamard-fc2-v1

uv run python integrations/h3/derived_checkpoint.py \
  --checkpoint-dir checkpoints/MiniMax-H3/int8-hadamard-fc2-v1
```

[run_prepared_checkpoint.py](../integrations/h3/run_prepared_checkpoint.py) verifies the complete derived inventory using the explicit `--checkpoint-manifest-sha256` value recorded in `derive.json`, rebuilds the pinned loader, and holds the GPU lock through the entire run. For import replay, pass `--capture-latents --parity-reference PATH_TO_ORIGINAL_H256_CAPTURE` and the same seed. It checks all 20 packed Euler video/audio states, final unpacked VAE-input latents, the original 124-frame MP4 and the exported 120-frame MP4 for byte-exact equality. Replay parity tests serialization/loading, not model quality or superiority over BF16.

```sh
uv run python integrations/h3/run_prepared_checkpoint.py \
  --checkpoint-dir checkpoints/MiniMax-H3/int8-hadamard-fc2-v1 \
  --checkpoint-manifest-sha256 SHA256_FROM_DERIVE_JSON \
  --source-dir .cache/h3.c-hadamard-cache \
  --seed 42 --capture-latents \
  --parity-reference results/local/h3/fc2-hadamard-capture-seed42-001 \
  --output-dir results/local/h3/prepared-replay-seed42-NEW
```

Replace `SHA256_FROM_DERIVE_JSON` with the exact `derived_manifest_sha256` field in `derive.json`. Each tool accepts `--gpu-wait-seconds` to wait for the shared local GPU lock. Export, rewrite and inference are deliberately separate commands so their costs and failure records remain independently attributable.

Offline export reports original load/hash, synchronized native rotation/quantization and readback/hash/write phases separately. Imported generation reports verified payload-loading wall time and requires a native marker confirming all 200 projections with zero weight rotation/quantization time. Capture I/O is diagnostic and excluded from throughput claims. Host compilation and 39 checkpoint/schema/rewrite/runner CPU tests passed, including compile/export/partial-export/import-digest failure fixtures that retain diagnostics and block manifest publication. The main task schedules all export and inference GPU work serially.

The complete export, derived model rewrite and first native import replay have now passed. The local prepared checkpoint is `checkpoints/MiniMax-H3/int8-hadamard-fc2-v1`, with derived manifest SHA256 `6840c91f6e5b9fcb4e8fe25292a1b1e2209d2f7f9c02e3427cfbbcbc71da4cda` and prepared payload manifest SHA256 `a2b3bb2d64954fb48bccaea138558183c09d721417147615e49cbebfc2888f1b`. Its 400 code/scale payloads cover all 200 projections and contain 19279769600 bytes. The 56-file model tree keeps all unquantized tensors in their original precision and removes the replaced projection copies from its rewritten shards. The original BF16 checkpoint remains separately available.

`results/local/h3/prepared-replay-seed42-001/run.json` records **byte-exact equality of all 42 latent tensors and both MP4s** against the original H256 seed42 capture: all 20 packed Euler video/audio states, both final unpacked latents, the 124-frame original MP4 and 120-frame exact export. The native loader confirms 200 imported projections, 12.721376 seconds of verified payload loading, and zero weight rotation/quantization. This proves that export/import preserved this H256 trajectory; it does not establish equivalence to BF16 or general video quality.

The one offline export measured 7.366674 seconds of synchronized native weight preparation, separately from 1.365344 seconds loading weights, 45.582728 seconds hashing original tensor bytes, and 12.995065 seconds reading back, hashing and writing payloads. Its process wall time was 69.649356 seconds, with full model preflight and subsequent payload verification accounted separately. These are one-shot preparation/import costs, not warm inference latency or a repeated speedup estimate.

## Fifteen-second first-frame benchmark

[run_long_benchmark.py](../integrations/h3/run_long_benchmark.py) is a separate frozen entry point for the user's full Chinese whale-maid prompt and reference image. [long_benchmark.json](../integrations/h3/long_benchmark.json) fixes seed83, 512×288 (exact16:9), 24fps, 20Euler steps, all50DiT blocks, reuse1 and referenceRoPE. Its SHA256 is `ee6fb7632e4b65858c9f0ec2944f5ae2a95eb85ae0e3219e308fb6fd1295e21a`. The prompt is kept verbatim in UTF8, including its final newline, and each run saves `prompt.txt`; SHA256 `fccc21a2e05295b2920af3b6a3a58e720d7d8f5f15f2e057986faecd4f95f0cc` binds all2233bytes. There is no prompt override.

The original1280×720 JPEG stays in ignored local storage and is copied byte for byte into each run as `conditioning-input.jpg`; SHA256 is `b1b4c9330f8e27fd95c770e610210b366ad30933a651f71d1e39401bf63cf24d`, size208682bytes. The command uses the actual upstream `--first-frame` option. Upstream stretches the raw image to the512×288 render canvas; both have16:9aspect, and the harness does no prior image processing. `conditioning` and `inputcondition.json` record that preprocessing explicitly. Do not publish the original image with public numerical summaries.

The CPU-only [tokenizer_probe.c](../integrations/h3/tokenizer_probe.c) uses the pinned native tokenizer and checks full decode roundtrip: the prompt is545tokens, the `<Picture 1>: ` prefix6tokens, the visual presentation144tokens plus2boundary tokens, giving697text-encoder presentation rows. The runner rebuilds and reruns this check before generation and rejects changed counts, failed roundtrip, or exceeded limits instead of truncating. The pinned tokenizer has no truncation branch; native text allocation checks cap the presentation at167772rows (configured tokenizer maximum262144). Header-only validation found every required351QwenVision BF16tensor and118videoVAE-encoder F32tensor in the already-downloaded FL2VA files. This proves component presence and layout, not successful GPU execution of these newly exercised encoders.

Upstream aligns 360 requested frames to 362 native frames, then the harness exports 360 frames and 15 seconds. The pinned implementation's parameter validator (`h3.c:514–517`) rejects `h3_align_frame_count(params->frames) > 362`; this is the limit of this native implementation, not a claim about the model's theoretical maximum duration. The harness validates dimensions, decoded frame counts, 24fps and 32kHz stereo audio. The existing capture implementation supports this geometry without changing either frozen native source: final video `[24,107,18,32]`, final audio `[32,2,603]`, packed Euler target video `[15408,96]` and target audio `[1206,32]`. The 144 first-frame VAE condition rows are excluded from captured target latents. Including 697 presentation rows gives 17455 source-derived total DiT rows; the runner labels this as source-derived rather than runtime-traced.

The 42 FP32 tensor dumps occupy 127491840 bytes before metadata. At 17455 rows, the explicit common DiT BF16 activation arena plus activated FFN accounts for 2564907520 bytes (2.565GB); actual peak memory also includes all model weights, temporary attention/encoder/VAE buffers and outputs. The prepared mode adds a 252378112 byte (0.252GB) INT8 activation/scales arena, including padding to 17536 rows. These figures assume the pinned default activation aliases and fused NAX/INT8 MLP. This partial source accounting is not a peak-memory measurement. Sequence length is 1.76 times the original 9914 row fox recipe, so dense attention pair count is approximately 3.10 times; neither ratio predicts end-to-end throughput because resolution, image encoding, text length and decoding also change. The 15 second path's runtime and quality remain unmeasured until its serial GPU runs finish.

```sh
uv run python integrations/h3/run_long_benchmark.py --mode bf16 \
  --output-dir results/local/h3/whale-maid-15s-bf16-NEW

uv run python integrations/h3/run_long_benchmark.py --mode hadamard-fc2 \
  --checkpoint-dir checkpoints/MiniMax-H3/int8-hadamard-fc2-v1 \
  --checkpoint-manifest-sha256 SHA256_FROM_DERIVE_JSON \
  --output-dir results/local/h3/whale-maid-15s-hadamard-NEW
```

The default input path is `.cache/h3-inputs/deepseek-whale-maid-v1.jpg`; `--input-image` may locate the same exact bytes elsewhere, but a different hash is rejected. Latent capture is enabled by default and these runs are diagnostic rather than performance samples. OriginalBF16 and derivedH use the same image, prompt and recipe, with frozen native source digests833076… and054ec7… respectively. The prepared run retains an exact snapshot of its derived checkpoint manifest for independent pairing. The originalBF16 checkpoint remains permanently retained. The long runner adds31CPU tests; runner, new evaluator and existing5second benchmark tests passed together (76tests). The independent reviewer found no blocking source/geometry/input issue. None of these CPU checks is a GPU quality claim.
