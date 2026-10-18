# Precision and backend contracts

| Mode | Storage / scales | Accumulation | Execution |
|---|---|---|---|
| BF16, FP16 | Dense row-major A[M,K], W[N,K] | FP32 in custom MPP | Native TensorOps, output matches input dtype |
| FP32 | Dense FP32 | FP32 | MPP may reduce multiplication mantissas; strict gate can reject |
| INT8 | Signed symmetric [-127,127]; FP32 max-abs/127 per A row and W row | INT32 | TensorOps integer dot, fused FP32 dequant epilogue |
| FP8 | E4M3 finite values, subnormals, round-to-nearest-even; FP32 max-abs/448 per row | FP32 | TensorOps FP8 inputs, fused dequant epilogue |
| MXINT8 emulated | 32 contiguous K values, signed S1.6, shared E8M0 | BF16 matmul backend | Quantize, restore to BF16, ordinary MLX matmul |

The MXINT8 encoder chooses `e=floor(log2(amax))`, `q=clip(round(x*64/2^e),-127,127)`. For ordinary Metal signed integer tensors the plane scale is stored as `2^(e-6)` instead of OCP's `2^e`: the S1.6 `/64` is folded into the exponent. The logical decoded values match; this buffer is **not an unmodified OCP interchange payload**. Zero blocks use q=0 and effective scale 1. Finite normal inputs are the intended domain; exponent clipping at E8M0 limits is explicit.

Quantization is symmetric, round-to-nearest-even, and has no zero point. Finite-value references reject nonfinite inputs. General NaN/Inf propagation is not part of the quantizer's contract. INT32 dot products require a safe reduction length: for worst-case [-127,127] operands K must not exceed 133144. Current H3 shapes are below that limit.

Native scaled INT8 TensorOps with floating accumulation currently fails compilation in the tested macOS 27.0.1 / Xcode 27.1 stack. This is a failed **specific API path**, not proof that every possible MXINT8 implementation or M5 hardware feature is unavailable. The explicit software fallback measures numerical effects; its decoded weights consume BF16 memory and it is not a compressed deployment.

FP8 tensors have alignment constraints. The current adapter requires K (the contiguous row stride) divisible by 128. Unsupported shapes are reported; no hidden padding or conversion is excluded from latency. Use `int8` or `bf16` for other shapes until a timed padding adapter is added.

Attention QDQ quantizes Q/K/V, restores BF16, then calls MLX SDPA. This is a quantization-sensitivity experiment only: preprocessing is outside its timing and there is no INT8/FP8 attention speedup claim. A fused low-bit attention kernel needs correct scale handling for both QKᵀ and PV; V scales along the contraction dimension cannot simply be factored out.

The optional Hadamard path retains original BF16 inputs/output but rotates into FP32 before INT8 quantization. It uses the same fixed normalized block transform on A and W, preserving the unquantized dot product mathematically. Per-token/channel scales are computed over the full rotated K row. Weight rotation is offline; online latency includes activation rotation. This changes the quantized approximation, not the declared INT8 storage/accumulation mode. INT8 + Hadamard and unrotated FP8 are separate experiment policies; FP8 + Hadamard is outside the active scope. Historical FP8 + Hadamard records and backend compatibility remain available for reproducibility. See [hadamard.md](hadamard.md).

The fused AdaLN path computes normalization and channel modulation in FP32, rounds the completed AdaLN value to the input dtype once, then widens for Hadamard and INT8 packing. Removing that boundary is an explicit alternate mode, not an invisible optimization. RMSNorm uses mean-square; LayerNorm uses centered population variance. Quantization scales cover the entire rotated row even when Hadamard uses smaller blocks. See [the fusion contract](norm-hadamard-fusion.md).
