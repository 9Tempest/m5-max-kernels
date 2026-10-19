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

## Completed 15-second first-frame pair

The whale-maid request completed in both BF16 and the persisted INT8/H256
checkpoint: seed83, 512×288 (16:9), 24 fps, 20 Euler passes and all 50 blocks.
The full 2233-byte Chinese prompt round-trips through the native tokenizer
without truncation: 545 prompt tokens and 697 presentation tokens after image
prefix/vision/boundary tokens. Both runs use the same original reference-image
bytes and upstream stretch-to-canvas preprocessing. The runtime produces
362 frames; a separate export retains 360 frames for exactly 15 seconds with
32 kHz stereo audio. [Playable pair and immutable evaluations](../results/published/h3-native-eval/whale-maid-15s-v1/README.md).

| Captured request measurement | BF16 | Prepared INT8/H256 |
| --- | ---: | ---: |
| Generation subprocess wall, seconds | 506.275859 | 364.855037 |
| DiT Euler denoise wall, seconds | 443.527 | 293.568 |
| DiT load phase, seconds | 7.825 | 15.719 |
| Video VAE decoder wall, seconds | 43.918 | 43.951 |
| DiT tracked Metal tensor peak, GiB | 38.199 | 20.967 |

The process ratio is **1.387608×**, and the denoise ratio is **1.510815×** for
this pair. These are single-seed diagnostic observations, not stable resident
speedup estimates. Both runs capture all 20 Euler states plus final video/audio
VAE inputs, adding synchronization, readback and file writes. Generation time
includes runtime context/shader setup, component loading, conditioning,
denoising, both decoders and the original 362-frame AV export. It excludes
model verification/build/tokenizer preflight, GPU queue wait, offline checkpoint
export, later validation and exact-duration trimming (0.335590/0.333519 s).
Full preflight hashing warms the file cache; BF16 ran before H without
interleaved repetitions. The memory figure counts tracked DiT Metal tensors,
not RSS, all unified memory or MPS internal scratch. DiT total/load/denoise
scopes overlap and must not be summed. The
[sanitized performance summary](../results/published/h3-native-eval/performance/whale-maid-15s-diagnostic-v1.json)
preserves the original run-record SHA256s, timings and source/config identities;
it is a selected-field derivative, not a byte-identical raw-record copy.

The source digests intentionally differ: BF16 uses the capture-capable
`83307697…` source with all INT8/H flags disabled; H uses the `054ec73e…`
prepared loader. All recorded harness file hashes match, despite the checkout
being rebased between runs. The candidate loads 200 persisted projections,
with a 12.411358 s verified-payload load marker and zero runtime weight
rotation/quantization. That marker is part of its 15.719 s DiT load phase,
not an extra cost to add. QKV/output/FC1 use native INT8 and FC2 uses H256;
this compares the complete precision/checkpoint policies, not Hadamard alone.
Both retain the same BF16 attention and original encoder/VAE precision.

Runtime capture verifies final video `[24,107,18,32]` and audio `[32,2,603]`,
with per-step target rows `[15408,96]` and `[1206,32]`; the 144 conditioning
video rows are excluded from those target dumps. The total 17455 packed rows
remains a tokenizer/source-derived count, not a per-operation trace. Against
BF16, final video/audio relative L2 is 0.21195048/0.31593459; decoded RGB SSIM
is 0.81870478 and RGB SNR is 15.69317 dB. PCM SNR is only 0.439247 dB
(relative L2 0.950687), so substantial waveform drift remains explicit.
These are drift metrics, not perceptual acceptance. Review so far covers
limited sampled still frames; full motion/prompt timing and audio listening
remain unevaluated. The original BF16 checkpoint remains retained alongside
the derived checkpoint. Attention-only and combined fusion now have seed42
whole-model parity evidence; the matched resident comparison below measures
attention-only fusion with H256 fixed in both arms.

## Additional five-second prompts

The person-turn (seed71), rainy-street (seed73) and water-pour (seed79)
BF16/prepared-H256 pairs have completed strict latent/media evaluation.
The [public three-scene package](../results/published/h3-native-eval/diversity-three-scenes-v1/README.md)
contains all six videos and the three paired media/latent records, with a
[playable gallery](../results/published/h3-native-eval/diversity-three-scenes-v1/gallery.html).
Each uses the fixed 512×512, 24 fps, 20-step, 50-block recipe and retains both
checkpoints' source identities. These are different prompts and seeds, not
independent repetitions of the fox request or matched latency samples.

| BF16-relative diagnostic | Person turn, seed71 | Rainy street, seed73 | Water pour, seed79 |
| --- | ---: | ---: | ---: |
| Final video latent relative L2 | 0.09077403 | 0.30224053 | 0.39418745 |
| Final audio latent relative L2 | 0.03419717 | 0.09127944 | 0.18162600 |
| Decoded RGB SNR, dB | 20.663847 | 6.314006 | 11.772501 |
| Decoded RGB SSIM | 0.91222101 | 0.63843918 | 0.71894218 |
| Decoded stereo PCM SNR, dB | 12.220583 | 7.021580 | 3.708196 |

Sources: [person evaluation](../results/published/h3-native-eval/diversity-three-scenes-v1/evaluations/person-turn-eval-001-benchmark-evaluation.json),
[rain evaluation](../results/published/h3-native-eval/diversity-three-scenes-v1/evaluations/rainy-street-eval-001-benchmark-evaluation.json)
and [water evaluation](../results/published/h3-native-eval/diversity-three-scenes-v1/evaluations/water-pour-eval-001-benchmark-evaluation.json).
All records retain `quality_decision=not_evaluated`. Six still frames per
mode show the broad person walk/turn/wave stages, with different turn timing
and later face/coat details; the first frame partially crops the person.
In the rainy scene both cars appear to travel away along the street rather
than clearly cross left to right as requested. Night lighting/wet pavement
are present, with differences in signs and reflections. Both water samples
show an empty glass, an entering stream, bubbles/splash and a rising water
level. Glass framing, pitcher angle and stream details differ; the H256 rim
moves lower in later sampled frames. This does not establish fixed-camera
behavior or fluid realism. These observations do not assess full motion or
audio; no audio has been listened to.
[Person review](../results/local/h3/person-turn-review-001/README.md),
[rain review](../results/local/h3/rainy-street-review-001/README.md),
[water review](../results/local/h3/water-pour-review-001/README.md).

## Review fixes and retrospective checks

Cross-review tightened the sample publisher's binding of original model
identity across BF16 and prepared records, required copied video/evaluation
bytes to match their already-validated SHA256s, and made poster generation
read the verified destination video. Public-text checks now reject temporary
absolute paths as well as user-home paths. Fusion campaign comparison checks
each run's positive UTC interval before checking serial execution. Regression
tests preserve the formerly accepted invalid cases; these changes do not
rewrite existing raw records or published JSON.

The [VAE configuration audit](../results/published/h3-native-eval/performance/runtime-config-audit-15s-v1.json)
identifies a historical preflight gap: the original-model inventory omitted
the video/audio VAE configuration files. The current original and derived
files match the pinned model inventory, and the derived manifest already
binds those digests. This is retrospective supporting evidence, not proof
that the original BF16 files stayed unchanged throughout earlier generation.
Both the long-runner preflight and the shared original-model inventory now
verify these configs. The shared inventory fix was applied after all six scene
generations completed, before the matched resident comparison. Historical run
hashes and timing evidence remain unchanged.

The [audio-duration audit](../results/published/h3-native-eval/performance/audio-duration-audit-v1.json)
rechecks saved, hash-bound decoded-frame reports for the fox and 15s pairs:
each side has exactly 160000 or 480000 audio samples per channel, respectively,
and no unresolved decoded-frame skip data. It does not re-decode media or
recompute the historical metrics. FFmpeg's normal decoder skip/discard handling
is already reflected in decoded frame sample counts; the new gate rejects
unresolved side data or a duration mismatch without trimming/padding it away.

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
- Denoising's `encode=167.986s` is elapsed wall time between beginning a command
  buffer and the harness's commit, not CPU-active time. It includes API work
  and any waiting inside those calls. The final `wait=37.304s` measures the
  remaining commit-to-fence interval; it is not all GPU work. The
  `root-gpu=0.104s` counter is also not total GPU time. These definitions follow
  the pinned [begin/continue/submit accounting](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L874).

A later [target-process Metal trace summary](../results/local/h3/native-capture-seed42-001-trace-summary.json)
contains 1665 compute intervals in a partial diagnostic DiT window. Their
observed span is 10.870592458 s and their union is 10.773800753 s (99.1096% of
that observed span). The capture requested 15 seconds, but this ratio is not
utilization over that full requested window. It includes capture/profiling
overhead and has no QKV/SDPA/MLP labels. It supports the warning that root-GPU
counters omit substantial work; it does not establish a compute bottleneck
or rank operators. This is an **INT8 projection run**, so its 99.1096% observed
active fraction cannot be transferred to the BF16 MPSGraph path. Raw traces/XML
with process identifiers stay local.

## BF16 source audit and diagnostic experiments

The original BF16 denoise reports 4000 MPS-linear operations, 1000 MPS-SDPA
operations and 3180 direct dispatches; native INT8 reports zero MPS-linear
operations, 1000 MPS-SDPA operations and 8200 direct dispatches. Each has 20
explicit submissions. At 50 blocks × 20 steps, the BF16 linear count represents
QKV, attention output and both MLP matrix products. It is not 4000 independent
graph encodes: the fused MLP encodes one graph and increments the linear count
by two. See the [BF16 record](../results/local/h3/bf16-seed42-001/run.json),
[INT8 record](../results/local/h3/int8-seed42-001/run.json),
[block dispatch](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_dit.c#L1875)
and [MLP dispatch accounting](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L2596).

With the recorded `H3_NAX=qkv-attn`, all four large BF16 projections use
MPSGraph at 9914 rows. The specialized grouped-QKV path stops at 2048 rows;
the general BF16 linear path automatically splits only 2049–3072 rows in this
mode. The BF16 NAX MLP requires the separate `mlp` mode, which the comparison
does not select. This is an explicit dispatch policy, not a TensorOps compile
failure. The relevant guards are
[grouped QKV](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L3780),
[linear shape selection](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L2430)
and [MLP capability selection](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L537).

The runtime already caches each [linear graph by shape, bias and dtype](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L2351),
the [MLP graph by its dimensions](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L2546)
and the [SDPA graph by geometry/layout/type](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L1448).
The fixed full-token block loop therefore reuses two large linear graphs and
one MLP graph across layers and steps; it does not reconstruct them per layer.
[Immutable weight bindings are cached](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L137)
as well. Activation/output tensor-data wrappers and feed/result dictionaries
remain transient, and every call invokes `encodeToCommandBuffer` with a nil
execution descriptor. The code has no explicit `MPSGraphExecutable` cache, but
that does not prove the framework recompiles a cached graph on every call.
The current source also sets [MPS command-wrapper reuse on](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_gpu.m#L337)
by default; caching that wrapper is not an unimplemented optimization.

The local Xcode SDK's `MPSGraph.h` documents that graph encoding may call
`commitAndContinue`; upstream consequently refreshes `gpu.command` from
`command.rootCommandBuffer` after encoding. The [profile documentation](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/README.md#L690)
also explicitly excludes internal child buffers from complete root-GPU
coverage. There is no explicit per-projection `waitUntilCompleted` in H3's
linear/MLP functions. Graph compilation, framework scheduling/back-pressure,
allocation and metadata work are possible contributors inside the 167.986s
interval; only a BF16 CPU/GPU timeline can separate them. The almost complete
shift from encode time to final wait time in INT8 is evidence that the API
paths behave differently, not evidence that 168 seconds of removable CPU work
has been found.

The following matrix keeps the original BF16 and current 15-second reference
runs unchanged. Variant environment controls belong in a new isolated,
source-pinned diagnostic harness: the frozen benchmark runners deliberately
pin their runtime environment and must not be bypassed. Use the same 512×512
fox request, 9914 rows, all 50 blocks, 20 steps, reuse1 and seeds42/43; hold the
shared GPU lock for every full run. First record a cold-in-context invocation,
then one warmup and at least three measured resident repetitions with ABBA
ordering and an independent rerun. Trace runs are separate diagnostics.

| Experiment | Controlled change | Evidence needed before promotion |
| --- | --- | --- |
| Locate BF16 API cost | Existing diagnostic patch samples QKV, SDPA, output and MLP at steps0/10/19 × blocks0/24/49; separately attach Time Profiler and Metal System Trace to the BF16 process | Host API elapsed distributions, sampled active CPU stacks versus blocked intervals, and target-process GPU timeline; distinguish first use from repeated same-shape calls |
| Cache diagnosis | Count graph-cache misses and binding-cache hits in a new diagnostic clone; time graph lookup, binding creation and graph encode separately; compare an explicit precompiled executable only after locating compilation cost | Separate compile/setup time from steady-state encode and synchronized block time; absence of an application executable cache alone is insufficient |
| Native BF16 QKV/output at full sequence | Keep `H3_NAX=qkv-attn`; explicitly set `H3_NAX_SPLIT_ROWS=2048`, giving four full chunks plus a1722-row tail; separately compare broader `H3_NAX=1` | Include every split launch, row-major QKV/RoPE conversion and attention boundary. `H3_NAX=1` does not enable the separate fused NAX MLP |
| Native BF16 MLP | Select `H3_NAX=mlp`, with `H3_DISABLE_NAX_MLP=1` as a same-context MPS-MLP control | Real-weight full-K reference, BF16 activation/branch boundaries, all-step latent comparison and whole-block/runtime latency; fused arithmetic may differ from MPSGraph |
| Command scheduling | Fix sampler window1 and vary `H3_DIT_COMMAND_BLOCKS=0,10,25,30,50`; retain the current no-explicit-split run as baseline | Count explicit and internal submissions, host/GPU overlap, memory pressure and byte-exact latent parity; more encoding ahead is not assumed faster |
| Wrapper negative controls | `H3_DISABLE_GRAPH_DATA_CACHE=1` and `H3_REUSE_MPS_COMMAND=0` separately disable existing caches | Establish whether those caches matter on this OS/device; do not advertise re-enabling the current default as a new gain |

Both Time Profiler and Metal System Trace templates are available in the
installed Xcode. A reproducible observer command is
`xcrun xctrace record --template 'Time Profiler' --attach TARGET_H3_PID --time-limit 30s --output NEW.trace`;
repeat separately with `'Metal System Trace'`, recording the owned process,
source/run digest and exact observed denoising span. Do not compare traced wall
times to untraced performance runs or report generic compute intervals as
named operator percentages. If finer attribution is needed, add CPU signposts
and API-level wall/process/thread CPU clocks in another isolated source pin,
without inserting per-operation GPU fences into the production path.

Native BF16 dispatch is the candidate with the broadest direct effect on this
MPS-linear-heavy path, but its benefit is unmeasured here. The pinned upstream
[NAX notes](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/README.md#L571)
report that earlier microbenchmark gains sometimes reversed in complete
forwards on different OS versions. Neither a faster tile nor less host API
elapsed time guarantees lower end-to-end latency. Explicit graph preparation
is worthwhile only if compilation contributes materially after warmup;
wrapper/split tuning is a smaller-priority hypothesis because caching and
bounded submission already exist. SDPA and the measured 30.218s video VAE
remain separate profiling targets, not inferred beneficiaries of this BF16
audit. These tests offer a better chance of finding a substantial bottleneck
than further tuning a gate epilogue whose H256 native campaign showed no
stable speed gain, while making no percentage-speedup promise.

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
The diagnostic patch's per-operation `host_api_ms` is host API elapsed time,
which can include framework waiting; it is neither CPU-active nor GPU time.
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
for that request, not official-pipeline or perceptual quality. Two
[native fusion campaigns](../results/published/h3-native-eval/fused-comparison-001-002-v1/README.md)
pass all 24 operator numerical cases. H256 FC2 has no cross-seed stable speed
gain; attention output at M9914 has lower paired medians in both seeds and
rounds, without a statistical-significance or model-speedup claim.

The [whole-model seed42 fusion check](../results/published/h3-native-eval/fused-model-parity-seed42-v1/README.md)
now passes for attention-only and both attention/FC2 flags. Each candidate
matches all 42 finite latent tensors and both the original 124-frame and
exact-duration 120-frame MP4s byte-for-byte against the on-load H256 reference.
The publisher re-read all 126 tensors across the reference and two candidates,
and independently checked media hashes, source/patch/model identities, policy
markers and recipe. This is diagnostic parity for one fox request, not a
latency or perceptual-quality result. Prepared import plus fusion has not
been tested; separate prepared-loader parity does not establish that
combination. Neither fusion flag is enabled by default.

## Matched resident attention-fusion comparison

The [completed seed42 resident comparison](../results/published/h3-native-eval/fused-resident-seed42-v1/README.md)
uses the same frozen `83ef61ac…` runtime, pinned harness, original checkpoint,
fox prompt and environment in both arms. H256 FC2 is fixed; only
attention-output gate fusion changes. Each separate process completes one
full warmup and three timed requests, with 512×512, 20 steps, all 50 blocks
and no reuse. The unfused process runs first, followed by attention-only.

| Synchronized host measurement | None median, s | Attention median, s | None / attention | Median seconds saved |
| --- | ---: | ---: | ---: | ---: |
| Cached generation, including decode and original AV export | 151.126785 | 147.752793 | 1.022835× | 3.373992 |
| Denoise callback interval | 121.909995 | 118.566479 | 1.028200× | 3.343516 |

The generation median is 2.232557% lower and the denoise median is 2.742610%
lower in this round. Generation samples range from 149.808414–151.210439 s
without fusion and 147.617907–149.613678 s with it; denoise ranges are
120.727940–121.987383 s and 118.564946–120.543138 s. All eight original
124-frame MP4s, including warmups, are byte-identical. The exporter verifies
the same recipe/model/source/harness/environment, cache and policy markers,
raw timing events, callback boundaries, media checksums and nonoverlapping
process intervals. [Raw samples and provenance](../results/published/h3-native-eval/fused-resident-seed42-v1/comparison.json).

There is no latent capture in these timing runs. Online activation preparation,
padding and synchronized callback overhead are included; offline weight
preparation is in warmup. Cached generation includes the original 124-frame
AV export, not the later exact-duration trim. Audio VAE setup/compilation APIs
and loading still repeat, so compilation-free end-to-end latency is not
claimed. Denoise uses native callback offsets, not the stale cached profile
wall mark. The earlier standalone H256 resident result above is context,
not the baseline for these ratios.

These are descriptive ratios of separate three-sample medians at one seed
and a fixed process order, without interleaving or independent replication.
The shared lock and interval checks exclude overlap between these arms, not
all possible external GPU activity. A second seed with reversed order and
independent repeated comparisons is needed before stable promotion. Neither
fusion flag is enabled by default, and prepared-checkpoint import combined
with fusion remains untested. This result is not an INT8/H256-versus-BF16
resident comparison or a new-best claim.

## Integration plans after profiling

1. **Dispatch at the real sequence length.** Add a measured shape dispatch
   table in the native backend. Test QKV/output row splitting, then port the
   winning TensorOps tile to the same C/Metal buffer contract. Include the
   9914-row tail and any 9984-row INT8 workspace, launches and layout changes.
   Accept only if decoded implementation checks pass and the complete warm
   projection/block latency improves in an independent repeat.
2. **Extend the completed FC2 H256 integration selectively.** Native C/Metal
   FC2 rotation and INT8 packing, real post-SwiGLU capture, comparison against
   native group1024 INT8, and the loadable 200-projection prepared checkpoint
   are complete. Original BF16 remains retained; FC2 has H256, while QKV,
   attention output and FC1 retain native INT8. Possible next experiments are
   FC1-output/SwiGLU/H256-packing fusion and H256 for QKV or gate/up. The latter
   changes the precision policy and needs new real-activation calibration and
   full-model comparison. Any fusion must retain the current BF16 branch,
   post-SwiGLU, residual and normalization boundaries, include all required
   layout/padding work, and avoid a Python/MLX buffer-copy boundary. The native
   runtime already fuses gated residual into AdaLN/quantization; that existing
   fusion is not a new optimization. Neither further fusion nor broader H256
   deployment has measured performance or quality evidence here.
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
