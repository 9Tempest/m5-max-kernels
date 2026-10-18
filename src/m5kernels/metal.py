"""Original Metal kernels, invoked through MLX's public custom-kernel API.

Inputs are never mutated. MPP's type traits require non-const tensor element
views, hence the casts in the shader (the underlying allocations remain read-only).
"""

from functools import lru_cache
from pathlib import Path
import mlx.core as mx

KERNELS = Path(__file__).parent / "shaders"
HEADER = (
    "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
)


@lru_cache(None)
def _quant_kernel(mode, tag="packed"):
    src = (KERNELS / "quantize.metal").read_text()
    src = src.replace("$BLOCK", "true" if mode == "mxint8" else "false")
    src = src.replace("$FP8", "true" if mode == "fp8" else "false")
    return mx.fast.metal_kernel(
        name="quant_" + mode + "_" + tag, input_names=["X"], output_names=["Q", "S"], source=src
    )


def quantize(x, mode, tag="packed"):
    """Return packed bytes and scales; MX scales include the S1.6 /64 adjustment."""
    rows, k = x.shape
    if mode == "mxint8" and k % 32:
        raise ValueError("MXINT8 requires K divisible by 32")
    blocks = rows * k // 32 if mode == "mxint8" else rows
    return tuple(
        _quant_kernel(mode, tag)(
            inputs=[x],
            grid=(blocks * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape, (rows, k // 32) if mode == "mxint8" else (rows,)],
            output_dtypes=[
                mx.uint8 if mode == "fp8" else mx.int8,
                mx.uint8 if mode == "mxint8" else mx.float32,
            ],
        )
    )


@lru_cache(None)
def _gemm_kernel(mode, tm, tn, groups, swizzle, bias_kind="none", gate_kind="none"):
    if bias_kind not in {"none", "vector", "table"}:
        raise ValueError("unknown GEMM bias kind")
    if bias_kind != "none" and mode != "int8":
        raise NotImplementedError("optional GEMM bias currently supports INT8 only")
    if gate_kind not in {"none", "vector", "table"}:
        raise ValueError("unknown GEMM gate kind")
    if gate_kind != "none" and mode not in {"int8", "bf16"}:
        raise NotImplementedError("gated GEMM supports INT8 and BF16 only")
    if gate_kind != "none" and bias_kind != "none":
        raise NotImplementedError("combining GEMM bias and gated residual is unsupported")
    dtype = {
        "fp16": "half",
        "bf16": "bfloat",
        "fp32": "float",
        "int8": "int8_t",
        "mxint8": "int8_t",
        "fp8": "metal_fp8_e4m3_format",
    }[mode]
    quant = mode in {"int8", "fp8", "mxint8"}
    planes = mode == "mxint8"
    row_scaled = mode in {"int8", "fp8"}
    src = (KERNELS / "gemm.metal").read_text()
    declarations = f"""
    auto a = tensor((device {dtype}*)A, dextents<int,2>(K,M), array<int,2>{{1,K}});
    auto b = tensor((device {dtype}*)B, dextents<int,2>(K,N), array<int,2>{{1,K}});
    """
    if mode == "fp8":
        declarations = """
        using T = tensor<device metal_fp8_e4m3_format,dextents<int,2>,tensor_inline>;
        T a((device uint8_t*)A,dextents<int,2>(K,M),array<int,2>{1,K});
        T b((device uint8_t*)B,dextents<int,2>(K,N),array<int,2>{1,K});
        """
    if planes:
        declarations = f"""
        using Scale = tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format,32,1>;
        using T = tensor<device {dtype},dextents<int,2>,tensor_inline,Scale>;
        T a((device {dtype}*)A,dextents<int,2>(K,M),array<int,2>{{1,K}},Scale((device uint8_t*)AS));
        T b((device {dtype}*)B,dextents<int,2>(K,N),array<int,2>{{1,K}},Scale((device uint8_t*)BS));
        """
    epilogue = ""
    if row_scaled:
        epilogue = """
        #pragma unroll
        for (ushort i=0; i<acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                auto ij=acc.get_multidimensional_index(i);
                int row=int(ij[1])+by*TM, col=int(ij[0])+bx*TN;
                if (row<M && col<N) C[row*N+col]=O(float(acc[i])*AS[row]*BS[col]);
            }
        }
        """
    else:
        epilogue = """
        for(ushort i=0;i<acc.get_capacity();++i) if(acc.is_valid_element(i)) {
            auto ij=acc.get_multidimensional_index(i);
            int row=int(ij[1])+by*TM,col=int(ij[0])+bx*TN;
            if(row<M && col<N) C[row*N+col]=O(float(acc[i]));
        }
        """
    if bias_kind != "none":
        bias_index = "col" if bias_kind == "vector" else "BiasRowMap[row]*N+col"
        epilogue = epilogue.replace(
            "O(float(acc[i])*AS[row]*BS[col])",
            f"O(float(acc[i])*AS[row]*BS[col]+Bias[{bias_index}])",
        )
    if gate_kind != "none":
        branch_value = "float(acc[i])*AS[row]*BS[col]" if mode == "int8" else "float(acc[i])"
        gate_index = "col" if gate_kind == "vector" else "GateRowMap[row]*N+col"
        old_store = f"C[row*N+col]=O({branch_value});"
        # The original GEMM output is BF16 before it participates in the
        # gated residual update. Keep that rounding boundary even though
        # the branch is now held locally instead of materialized globally.
        new_store = (
            "{\n"
            f"    bfloat branch=bfloat({branch_value});\n"
            "    C[row*N+col]=O(float(Residual[row*N+col])"
            f"+float(branch)*float(Gate[{gate_index}]));\n"
            "}"
        )
        epilogue = epilogue.replace(old_store, new_store)
    for key, value in {
        "$DECL": declarations,
        "$TM": str(tm),
        "$TN": str(tn),
        "$GROUPS": str(groups),
        "$SWIZZLE": str(swizzle),
        "$ACC": "int" if mode == "int8" else "float",
        "$STORE": epilogue,
    }.items():
        src = src.replace(key, value)
    input_names = ["A", "B", "AS", "BS"] if quant else ["A", "B"]
    if bias_kind != "none":
        input_names += ["Bias"] + (["BiasRowMap"] if bias_kind == "table" else [])
    if gate_kind != "none":
        input_names += ["Residual", "Gate"] + (["GateRowMap"] if gate_kind == "table" else [])
    name = f"gemm_{mode}_{tm}_{tn}_{groups}_{swizzle}"
    if bias_kind != "none":
        name += f"_bias_{bias_kind}"
    if gate_kind != "none":
        name += f"_gate_{gate_kind}"
    return mx.fast.metal_kernel(
        name=name,
        input_names=input_names,
        output_names=["C"],
        source=src,
        header=HEADER,
    )


def gemm(
    a,
    b,
    mode="bf16",
    tile=(64, 64, 4, 1),
    a_scales=None,
    b_scales=None,
    output_dtype=None,
    bias=None,
    bias_row_map=None,
    residual=None,
    gate=None,
    gate_row_map=None,
):
    """A[M,K] @ B[N,K].T -> FP32, with float or int32 accumulation.

    INT8 optionally adds FP32 bias after row-scale dequantization and before
    the output cast. Bias is a shared vector [N], or a table [P,N] selected
    by int32 bias_row_map[M]. Table indices must lie in [0,P); finite values
    and map bounds are caller preconditions checked outside timed execution.
    No input, bias table or map is mutated.

    INT8/BF16 optionally fuse a gated residual update when residual and gate
    are provided. This requires explicit BF16 output, BF16 residual[M,N],
    and BF16 gate[N] or gate[P,N] with int32 gate_row_map[M]. The branch is
    rounded to BF16 before FP32 gate multiplication/addition, then the updated
    residual is rounded to BF16. Only the updated residual is materialized.
    A bias and a gated residual cannot be combined in this adapter.
    """
    m, k = a.shape
    n, kb = b.shape
    if k != kb:
        raise ValueError("K mismatch")
    if mode == "fp8" and k % 128:
        raise NotImplementedError(
            "FP8 requires K/row stride divisible by 128; padding is not implicit"
        )
    if mode == "int8" and k > 133144:
        raise NotImplementedError("INT32 worst-case overflow bound exceeded")
    gate_kind = "none"
    if residual is not None or gate is not None or gate_row_map is not None:
        if residual is None or gate is None:
            raise ValueError("gated GEMM requires both residual and gate")
        if mode not in {"int8", "bf16"}:
            raise NotImplementedError("gated GEMM supports INT8 and BF16 only")
        if output_dtype is None or output_dtype != mx.bfloat16:
            raise NotImplementedError("gated GEMM requires explicit BF16 output_dtype")
        if bias is not None or bias_row_map is not None:
            raise NotImplementedError("combining GEMM bias and gated residual is unsupported")
        if residual.shape != (m, n) or residual.dtype != mx.bfloat16:
            raise ValueError("residual must have BF16 dtype and shape [M,N]")
        if gate.dtype != mx.bfloat16:
            raise ValueError("gate must have BF16 dtype")
        if gate_row_map is None:
            if gate.shape != (n,):
                raise ValueError("gate must have shape [N] without gate_row_map")
            gate_kind = "vector"
        else:
            if len(gate.shape) != 2 or not all(gate.shape) or gate.shape[1] != n:
                raise ValueError("mapped gate must have nonempty shape [P,N]")
            if gate_row_map.shape != (m,) or gate_row_map.dtype != mx.int32:
                raise ValueError("gate_row_map must be an int32 vector of length M")
            gate_kind = "table"
        expected_dtype = mx.int8 if mode == "int8" else mx.bfloat16
        if a.dtype != expected_dtype or b.dtype != expected_dtype:
            raise ValueError("gated GEMM operand dtypes must match the declared mode")
        if mode == "int8" and (
            a_scales is None
            or b_scales is None
            or a_scales.shape != (m,)
            or b_scales.shape != (n,)
            or a_scales.dtype != mx.float32
            or b_scales.dtype != mx.float32
        ):
            raise ValueError("gated INT8 GEMM requires float32 scales with shapes [M] and [N]")
    bias_kind = "none"
    if bias is None:
        if bias_row_map is not None:
            raise ValueError("bias_row_map requires a bias table")
    else:
        if mode != "int8":
            raise NotImplementedError("optional GEMM bias currently supports INT8 only")
        if bias.dtype != mx.float32:
            raise ValueError("bias must have float32 dtype")
        if bias_row_map is None:
            if bias.shape != (n,):
                raise ValueError("bias must have shape [N] without bias_row_map")
            bias_kind = "vector"
        else:
            if len(bias.shape) != 2 or not all(bias.shape) or bias.shape[1] != n:
                raise ValueError("mapped bias must have nonempty shape [P,N]")
            if bias_row_map.shape != (m,) or bias_row_map.dtype != mx.int32:
                raise ValueError("bias_row_map must be an int32 vector of length M")
            bias_kind = "table"
    tm, tn, groups, swizzle = tile
    inputs = [a, b]
    if mode in {"int8", "fp8", "mxint8"}:
        if a_scales is None or b_scales is None:
            raise ValueError("scales required")
        inputs += [a_scales, b_scales]
    if bias is not None:
        inputs += [bias] + ([bias_row_map] if bias_row_map is not None else [])
    if residual is not None:
        inputs += [residual, gate] + ([gate_row_map] if gate_row_map is not None else [])
    return _gemm_kernel(mode, tm, tn, groups, swizzle, bias_kind, gate_kind)(
        inputs=inputs,
        grid=(((m + tm - 1) // tm) * ((n + tn - 1) // tn) * groups * 32, 1, 1),
        threadgroup=(groups * 32, 1, 1),
        template=[("O", output_dtype or mx.float32)],
        output_shapes=[(m, n)],
        output_dtypes=[output_dtype or mx.float32],
    )[0]


@lru_cache(None)
def _rms_kernel(threads):
    return mx.fast.metal_kernel(
        name=f"rms_modulate_{threads}",
        input_names=["X", "W", "Scale", "Shift"],
        output_names=["Y"],
        source=(KERNELS / "rms_modulate.metal").read_text().replace("$THREADS", str(threads)),
    )


def rms_modulate(x, w, scale, shift, threads=256):
    return _rms_kernel(threads)(
        inputs=[x, w, scale, shift],
        grid=(x.shape[0] * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        template=[("O", x.dtype)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
