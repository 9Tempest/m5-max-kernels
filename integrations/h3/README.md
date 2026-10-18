# H3 local inference integration

This directory keeps full-model deployment separate from `m5bench` kernel
microbenchmarks. It uses [antirez/h3.c](https://github.com/antirez/h3.c/tree/8974cc055ea9c02fcd14cc27dfda3e1027c05153),
commit `8974cc055ea9c02fcd14cc27dfda3e1027c05153` (MIT), and the original
[MiniMaxAI/MiniMax-H3 FL2VA checkpoint](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA),
revision `42ed227ee7df40d41602854ae760620d6eb651fe`. Weights retain their own
MiniMax community license; they are downloaded locally, not redistributed.

## Install

From the repository root on an Apple M5 Mac with Xcode, Metal and FFmpeg:

```sh
uv sync --locked
git clone https://github.com/antirez/h3.c.git .cache/h3.c
git -C .cache/h3.c checkout --detach 8974cc055ea9c02fcd14cc27dfda3e1027c05153
make -C .cache/h3.c -j8
uv run python integrations/h3/download.py --workers 16
```

`model-files.json` snapshots the pinned Hugging Face FL2VA tree metadata,
including LFS SHA-256 values. Downloads resume through `.partial` files and
verify hashes before promotion. The complete original family occupies about
144 GB. The text encoder, DiT and VAEs execute in separate stages; download
size is not simultaneous GPU residency. Downloaded weights stay local and are
ignored by Git. The completed experiment retains both variants under
`checkpoints/MiniMax-H3/`, with a compatibility symlink at the original cache
location; see the [checkpoint contract](../../docs/h3-hadamard-native.md).
Do not run two downloaders against the same destination.

## Benchmark recipe

`fox-winter-t2va-v1` uses this exact prompt:

> A red fox walks through fresh snow in a pine forest. Medium tracking shot,
> natural winter light, realistic fur, soft footsteps and wind.

The default request is 512×512, 5 seconds, 24 fps, seed 42, 20 Euler passes,
all 50 DiT blocks, no velocity/core reuse, no token reduction and no internal
upscaling. This is a low-resolution development request. Twenty steps are the
upstream serving default, not its 50-step quality oracle.

The upstream temporal contract rounds 120 requested frames to 124 frames
(`5 + 17n`), giving 5.1667 seconds. Keep `original.mp4` intact; the runner
separately exports `video-5s.mp4` with exactly 120 frames and verifies both
files with FFprobe. Export time is not charged to denoiser kernel time.

```sh
uv run python integrations/h3/run.py --mode bf16 \
  --output-dir results/local/h3/bf16-seed42-001
uv run python integrations/h3/run.py --mode upstream-int8 \
  --output-dir results/local/h3/int8-seed42-001
```

Every run needs a new directory. The runner holds the repository's GPU lock
throughout generation and media validation. It records source/binary/model
digests, revisions, environment, request, timestamped raw log offsets, phase
profiles, and output checksums in immutable evidence files. Full model digest
verification precedes timing and can warm the filesystem cache. The result
is not an untouched cold-disk measurement.

Use `--gpu-wait-seconds 1800` to queue behind another local experiment after
model verification. The default is an immediate lock check. Queue time is
recorded separately and excluded from generation process time; the acquired
lock stays held through generation and media validation.

`bf16` explicitly disables M5's default INT8 QKV, MLP and attention-output
projections. It is the upstream BF16 close-reference path, not a proof of
parity with the official Python pipeline. `upstream-int8` retains upstream
grouped FC2 and its existing fusion; it is not this repository's per-row
INT8 or Hadamard candidate. MXINT8 and FP8 are not substituted silently.
TensorOps compilation failure is recorded as unsupported and the automatic
upstream fallback is rejected. Both modes explicitly use reference RoPE.

## Evidence interpretation

One process includes runtime shader compilation, loading, weight preparation,
text encoding, denoising, audio/video VAE and MP4 creation. `--profile` reports
component phase walls, encode/wait counters, root GPU duration and allocation
counters. A component's `total` overlaps its phase marks; do not sum both.
Tracked tensor peak is not process RSS or whole-machine memory consumption.

The 124-frame geometry has 37 video latent timesteps. At 512 square this is
9472 video tokens + 414 stereo-audio tokens + actual text tokens. The initial
formula is source-derived, not an observed full runtime shape; use the optional
diagnostic patch to confirm real packed geometry. Existing T=4096 synthetic
kernel results cannot be multiplied into this request's latency.

Media structure validation establishes decodability, dimensions, frame count,
and duration. It does not establish numerical parity or perceptual quality.
Compare candidate outputs against the same seed/configuration BF16 output,
inspect motion and audio, and evaluate another seed. Any speedup promotion
requires independent reruns, synchronized warm timings and numerical/quality
checks, including online activation transformation and quantization overhead.

Candidate work is deliberately separate: INT8 + offline-weight/online-activation
Hadamard, standalone FP8, attention/layout/fusion, and resident-session caching.
No FP8 + Hadamard deployment claim is made.

## Resident requests and diagnostics

`session.c` is a small driver using the pinned public H3 API. Build it against
the unmodified runtime, then measure one complete warmup and three cached
generations with the same recipe:

```sh
mkdir -p .build
clang -O2 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Werror \
  -I .cache/h3.c integrations/h3/session.c .cache/h3.c/libh3.a \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph -framework Accelerate \
  -licucore -lm -o .build/h3-session
uv run python integrations/h3/run_session.py --mode upstream-int8 \
  --output-dir results/local/h3/int8-cached-seed42-001
```

The C driver owns the GPU lock for the complete session; do not wrap it in a
second lock. The Python wrapper rejects nonzero `--gpu-wait-seconds` and `--diagnostics`.
Audio VAE context creation and loading still repeat, so cached end-to-end
latency must not be labelled compilation-free. The wrapper reports raw
generation times and separately instrumented synchronized denoise callback
times. Upstream cached DiT phase-wall marks span intervening decode/export
work and cannot be used as per-generation denoise measurements.

For the pinned Hadamard implementation, `run_native_session.py` rebuilds its
own driver with `H3_SESSION_EXPERIMENT` and accepts `bf16`, `upstream-int8`, or
`hadamard-fc2`. This wrapper owns the GPU lock across preflight, compilation,
all four generations, and validation; the experimental driver does not acquire
a second lock. It supports waiting for the GPU and records every raw request
and denoise callback time. The fixed recipe remains 512×512, 124 source frames,
20 Euler passes, all 50 blocks, reference RoPE, and the fox prompt.

```sh
uv run python integrations/h3/run_native_session.py --mode hadamard-fc2 \
  --seed 42 --gpu-wait-seconds 1800 \
  --output-dir results/local/h3/hadamard-resident-seed42-NEW
```

Three same-seed repetitions describe within-session spread; independent
reruns and another seed are still required for a new performance claim.
`cached_denoise_callback` includes online activation preparation, required
padding, synchronization, and output unpacking, after the complete warmup.
`cached_generation` additionally includes decoder/export and repeated audio
VAE setup. There is no kernel-only or compilation-free full-request claim.
The wrapper verifies cache retention and reports whether all four MP4 files
are byte-identical. Fusion options remain explicitly unsupported until the
corresponding source version and numerical checks are frozen.

The optional [diagnostic build](diagnostics.md) records actual geometry,
selected branches and sampled post-Euler finite statistics. Keep its overhead
separate from baseline timing and verify same-seed output equivalence.

`download_ranges.py` is an optional, independently tested HTTP Range helper
for a remaining single-file download. Stop the original downloader and all
its children first. It preserves the existing prefix, validates response
ranges and final pinned SHA256, and retains its extra chunks for inspection.
Its tests use mocked HTTP; throughput improvement is not established.
