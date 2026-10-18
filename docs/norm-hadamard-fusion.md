# AdaLN + Hadamard + dynamic INT8 fusion

The operation can be one Metal dispatch. The experiment retains each row on chip while computing normalization statistics, channelwise modulation, block Hadamard and the final **whole-row** INT8 scale. It writes packed INT8 activations and one FP32 scale per row. Weight rotation and quantization remain offline. FP8 is a separate precision policy and is not combined with Hadamard here.

## Semantics and where this applies

H3 uses RMSNorm with learned channel weights, not mean-centered LayerNorm, before attention projections and before the FFN gate/up projections. The tested expression is

```
r = rsqrt(mean(x*x) + epsilon)
y = BF16((x * r * gamma) * (1 + modulation_scale) + modulation_shift)
z = float32(y) D H256
s = max(abs(z)) / 127            # over the complete K row
q = clamp(round_to_even(z / s), -127, 127)
```

Zero rotated rows use scale 1. The BF16 cast is a real numerical boundary, retained inside the fused kernel. This matches the single final AdaLN rounding in the repository's native-style Metal baseline. It is not a byte-parity claim with eager Diffusers, which expresses separate tensor operations, or with upstream h3.c, whose reduction tree and quantization arithmetic differ. The optional FP32-boundary mode has an explicitly different contract.

In real H3, modulation is selected by timestep and modality for each packed token. The kernel API accepts a modulation table plus a row map; current performance experiments use one shared modulation vector and synthetic tensor values. FFN-down consumes **SwiGLU output**, so this AdaLN fusion belongs at Q/K/V and FFN gate/up, not at FFN-down. Q/K/V can reuse one prepared activation, as can gate/up; do not count the same preparation three times in model latency.

The pinned [Diffusers H3 source](https://github.com/huggingface/diffusers/blob/b800f828f3951c4d206e154411bcbfa693cd322e/src/diffusers/models/transformers/transformer_minimax_h3.py) establishes the model ordering. The pinned [h3.c Metal source](https://github.com/antirez/h3.c/blob/8974cc055ea9c02fcd14cc27dfda3e1027c05153/h3_shaders.metal) already contains gate/AdaLN/INT8 fusion without Hadamard. Our kernels are independently implemented; this experiment compares the staged operations in this repository, not a measured win over h3.c's existing fused implementation.

## Passes, reductions and synchronization

Two global passes, two reductions and two dispatches are different quantities. Keeping a row in registers can eliminate global rereads without eliminating the dependencies between reductions. For H3 RMSNorm, the two row reductions are the sum of squares and the maximum **after** modulation and rotation. The centered LayerNorm variant uses mean, centered variance and post-rotation maximum: three row reductions. Its variance calculation avoids subtracting two large moments. [Triton's LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) is another example of retaining a row while reducing its statistics.

For width 5376 and H256, a row contains 21 blocks. A 256-thread workgroup has eight SIMD groups; each SIMD group owns complete blocks, with eight values per lane. Up to three blocks per group require 24 FP32 value slots per thread. Five Hadamard stages exchange values across the 32 SIMD lanes, and three stages exchange the eight registers within each lane. Hadamard therefore needs no threadgroup-memory exchange. Small shared arrays synchronize the row statistics and peak across SIMD groups.

`one_read` retains the initial input values. `two_read` reloads X after the first statistic reduction and then retains intermediate values. Both are one-dispatch algorithms; the source-level read counts do not establish measured DRAM traffic. Register spilling and occupancy can reverse the expected benefit, so both variants are measured rather than assuming fewer reads always wins.

Ignoring channel parameters, caches, scale writes and spills, the staged RMS pipeline has a logical 21 bytes/element of input/intermediate/output traffic: two BF16 X reads, a BF16 AdaLN write/read, a FP32 rotation write and two reads by the quantizer, and the INT8 output. The buffered fused path needs one BF16 X read plus the INT8 output (3 bytes/element). This is an access-count comparison, not a measured bandwidth or speedup claim.

## Why normalization cannot generally disappear

For an exact positive row scalar g, symmetric dynamic quantization has identical integer codes for g*z and z, with its dequantization scale multiplied by g. A pure RMSNorm, or an ideal zero-shift AdaLN, can exploit that identity. General AdaLN has a nonzero shift:

```
H(g*v + shift) = g*H(v) + H(shift)
```

The resulting integer codes depend on g. The retained BF16 rounding also prevents general exact cancellation. Likewise, channelwise modulation cannot simply commute through Hadamard. The implemented fusion preserves the operation order.

## Reproduce

```sh
uv sync --locked
uv run python scripts/norm_hadamard_campaign.py \
  --output results/local/norm-hadamard-sweep-001.json
uv run m5bench bench --suite h3-norm-hadamard --filter norm_q.m4096 \
  --backends norm-staged,norm-fused --precisions int8 \
  --tile 128 64 4 8 --output results/local/norm-hadamard-001.json
uv run python -m m5kernels.profile --kind gpu --suite h3-norm-hadamard \
  --case h3.norm_q.m4096 --backend norm-fused --precision int8 --online \
  --tile 128 64 4 8 --output results/local/norm-hadamard-001.gputrace
uv run python scripts/profile_report.py results/local/norm-hadamard-001.gputrace \
  --expect-dispatches 2 --output results/local/norm-hadamard-profile-001.json
```

The operation `norm_hadamard_gemm` follows the regular workload JSON schema. Its `fusion` configuration specifies norm, epsilon, block size, rounding boundary, thread count, read strategy and optional sign seed. The harness records independent float64 normalization/rotation references, ideal quantization error, actual preparation error and GEMM implementation error separately. `activation_prepare` measures the complete norm/modulation/rotation/quantization stage; `online_activation_plus_gemm` includes this stage and GEMM. Compilation and offline weight preparation are excluded.

Supported rotation blocks are exact divisors of K in 32–1024 powers of two; no hidden padding is used. Modulation-table bounds and finite inputs are caller preconditions. Full-model activation distributions, residual-gate fusion, multimodal-table performance and generated-video quality remain separate validation tasks.
