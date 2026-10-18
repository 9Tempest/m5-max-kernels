# Hadamard before quantized GEMM

For row-major activation `A[M,K]` and weight `W[N,K]`, this experiment uses one shared orthogonal right rotation `R = D H`:

```
A' = A R
W' = W R
A' W'^T = A R R^T W^T = A W^T
```

`H` is a block-diagonal normalized Sylvester Hadamard matrix. `D` is either identity (plain Hadamard) or a fixed diagonal of ±1 signs generated with a separate PCG64 seed. Exactly the same signs apply to every activation row and weight row; they are never regenerated per token or data seed. The block size, sign seed and SHA256 of the sign vector are recorded.

Weights are rotated and quantized **offline**. Online inference rotates the activation, quantizes each rotated row, runs the existing INT8/FP8 TensorOps GEMM, then produces BF16 output. Both transforms accumulate and output FP32; there is no additional BF16 rounding between rotation and quantization. INT8 uses INT32 accumulation with the existing scale epilogue. The quantization scale remains per complete K row, not per Hadamard block.

This is a three-stage GPU composition (rotation → quantizer → GEMM), not a fused Hadamard/GEMM kernel. The custom rotation fuses the optional sign multiplication into its load. Its first five butterfly stages use SIMD shuffles; larger strides use threadgroup scratch and barriers. The comparison implementation uses MLX's built-in transform on FP32 blocks and includes its cast/sign multiplication in timing.

## Measured results — 2026-10-18

Block256, no random signs, same synthetic outlier definition and seed 29 as the original INT8 stress test. The block/sign sweep used seed 17; the fixed rotation then received an 8-tile GEMM sweep and holdout at seed 18. The final table uses **independent interleaved BF16/candidate runs**, 5 warmups and 20 timed repetitions. All candidate times include online activation rotation + quantization + GEMM. Weight rotation and quantization remain offline.

| M=4096 component | Unrotated INT8 relative L2 | Hadamard INT8 relative L2 | BF16 ms | Hadamard INT8 online ms | Speedup vs BF16 |
|---|---:|---:|---:|---:|---:|
| h3.attn_q.m4096 | 4.69% | 0.70% | 5.165 | 3.789 | 1.36× |
| h3.ffn_up.m4096 | 4.95% | 0.72% | 10.097 | 6.895 | 1.46× |
| h3.ffn_down.m4096 | 8.84% | 1.22% | 12.707 | 8.701 | 1.46× |

All three rotated candidates pass the unchanged 5% synthetic quality gate, including FFN-down which unrotated INT8 fails. Normal-input errors remain about 1.2–1.4%; rotation gives no meaningful accuracy benefit on those normal inputs and adds work. On the independent outlier run, standalone activation rotation takes about 0.41 ms for Q/FFN-up and 0.87 ms for FFN-down. Combined rotation+quantization takes about 0.68–0.69 ms and 1.47 ms respectively. These standalone medians must not be summed to estimate complete latency.

The same-tile FFN-down implementation comparison (128/128/8/8, seed29) isolates the added transform: unrotated INT8 is 8.059 ms with 8.84% error; Metal H256+INT8 is 9.051 ms with 1.22% error. The MLX Hadamard composition gives the same error at 10.795 ms. The retuned complete path in the table is 8.701 ms. Thus the benefit is improved INT8 accuracy while retaining a BF16 speed advantage, not a faster GEMM merely from rotation.

FP8 does not benefit in this experiment: its FFN-down outlier error is 3.78% unrotated versus 3.88% with H256, while online latency increases from 11.335 to 12.223 ms on the same tile. Larger Hadamard blocks (512/1024 for FFN-down) and fixed random signs did not give a compelling improvement over plain H256 in this sweep; they remain explicit configurable experiments. No universal optimal-block or model-quality claim is made.

Evidence: [block/sign sweep](../results/published/2026-10-18-hadamard-block-sweep.json), [larger blocks](../results/published/2026-10-18-hadamard-large-blocks.json), [Q tuning](../results/published/2026-10-18-hadamard-q-tune.json), [FFN-up tuning](../results/published/2026-10-18-hadamard-up-tune.json), [FFN-down tuning](../results/published/2026-10-18-hadamard-down-tune.json), [independent outlier rerun](../results/published/2026-10-18-hadamard-independent-outliers.json), [normal rerun](../results/published/2026-10-18-hadamard-independent-normal.json), [same-tile Metal/MLX/FP8 ablation](../results/published/2026-10-18-hadamard-implementation-comparison.json). Unrotated seed29 errors are preserved in the [earlier outlier record](../results/published/2026-10-18-outliers-validation.json); the direct same-tile comparison also reproduces FFN-down’s error.

Actual H3 activations across denoising steps and output quality have not been evaluated. The block transform cannot guarantee bounded quantization error for arbitrary inputs. Reuse of one rotated activation across shared-input projections (e.g. Q/K/V) has not been benchmarked.

## GPU trace

The [captured GPU report](../results/published/2026-10-18-hadamard-gpu-profile.json) verifies exactly one Hadamard, one quantizer and one GEMM dispatch. Serial replay shader cost is 6.07% rotation, 5.19% quantization and 88.74% GEMM. The Hadamard shader reports 7 temporary registers and zero spilled bytes. Replay duration is 12.495 ms with profiling overhead; it is not the 8.701 ms benchmark latency above. Raw capture: `results/local/ffn-int8-hadamard256.gputrace`.

```sh
uv run python -m m5kernels.profile --case h3.ffn_down.m4096 \
  --backend mpp-hadamard --precision int8 --online --tile 64 64 4 8 \
  --hadamard-block 256 --output results/local/hadamard-001.gputrace
uv run python scripts/profile_report.py results/local/hadamard-001.gputrace \
  --expect-dispatches 3 --output results/local/hadamard-profile-001.json
```

## Supported dimensions

The custom adapter supports block sizes 32, 64, 128, 256, 512 and 1024, with K exactly divisible by the block. It does not pad implicitly. H3 widths 5376, 7168 and 14336 all divide by 256. The latter two allow larger blocks; K=5376 cannot use block512/1024 without an additional explicitly timed adapter.

These are **block rotations**, not a full-K Hadamard or a complete model-wide QuaRot conversion. MLX advertises full transforms for `n=m*2^k`, `m∈{1,12,20,28}`; the block implementation here only uses powers of two. For FFN-down model integration, rotate the actual input **after SwiGLU**. The rotation cannot be moved through SiLU or elementwise gating.

## Reproduce

```sh
uv run m5bench bench --suite h3 --filter ffn_down.m4096 \
  --backends mpp-hadamard --precisions int8 --distribution outliers \
  --tile 128 128 8 8 --hadamard-block 256 \
  --output results/local/hadamard-001.json
# Optional fixed random signs; omit or use "none" for ordinary Hadamard.
uv run m5bench bench --suite h3 --filter attn_q.m4096 \
  --backends mpp-hadamard --precisions int8 --hadamard-sign-seed 0 \
  --hadamard-block 128 --hadamard-implementation mlx \
  --output results/local/hadamard-mlx-001.json
uv run python scripts/hadamard_campaign.py \
  --output results/local/hadamard-sweep-001.json
```

Custom workload JSON can include `"hadamard": {"block_size": 256, "sign_seed": null, "implementation": "metal"}` on GEMM cases. Explicit CLI flags override these fields; unspecified flags preserve the workload settings. Defaults are block256, unsigned, Metal. `m5bench tune --tune-backend mpp-hadamard` tunes GEMM tiles while keeping the requested rotation fixed; it does not choose a rotation size or signs.

## Measurement and numerical gates

- `timing`: GEMM with both rotated/quantized operands already prepared.
- `activation_hadamard`: activation rotation alone, including input conversion/sign application.
- `activation_quantization`: quantization of an already rotated resident activation.
- `activation_hadamard_plus_quantization`: complete online preparation.
- `online_activation_plus_gemm`: **activation rotation + quantization + GEMM**, including Python submission and synchronization. Use this for deployment comparisons.

Do not add the standalone medians to estimate the combined path. Offline weight rotation/quantization is excluded from every online timing. The existing GEMM byte lower bound excludes transformation traffic and is explicitly labeled as such.

The float64 NumPy reference uses independent butterflies, cross-checked against an explicit parity-defined Sylvester matrix. Small tests cover full tensors, multiple blocks, odd row counts, zeros/outliers and noncontiguous layouts. Large experiments check all K values in sampled activation/weight rows, full-output finiteness and preservation of the original dot product. They record transform implementation error, quantization error against actual rotated operands, total error against the original BF16 dot product, and GEMM implementation error against decoded quantized operands separately. Incorrect transforms cannot pass the benchmark or capture gate.

References: [MLX Hadamard API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.hadamard_transform.html), [pinned MLX Metal implementation](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/hadamard.h), [QuaRot paper](https://arxiv.org/html/2404.00456v2). These references motivate the experiment; their model-level claims are not results of this repository.
