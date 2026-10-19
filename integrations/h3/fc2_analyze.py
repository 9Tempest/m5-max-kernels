"""CPU-only independent references for native FC2 readbacks.

Quantization error is compared with original BF16 operands in float64.
Implementation error is compared with the actual decoded INT8 operands.
Projection evidence alone never authorizes replacing an original checkpoint.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


POLICIES = ("native-group1024", "hadamard-row", "native-row")


def bf16(path, shape):
    return (np.fromfile(path, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)


def hadamard_reference(x, dtype=np.float64):
    """Independent natural-order Sylvester butterflies, complete H256 blocks."""
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 2 or x.shape[1] % 256 or not np.isfinite(x).all():
        raise ValueError("finite rows with K divisible by 256 required")
    y = x.copy().reshape(-1, 256)
    stride = 1
    while stride < 256:
        z = y.reshape(-1, 2, stride)
        a, b = z[:, 0].copy(), z[:, 1].copy()
        z[:, 0], z[:, 1] = a + b, a - b
        stride *= 2
    return (y * dtype(1 / 16)).reshape(x.shape)


def metrics(actual, reference):
    a, r = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    if a.shape != r.shape or not a.size:
        raise ValueError("matching nonempty shapes required")
    finite = bool(np.isfinite(a).all() and np.isfinite(r).all())
    if not finite:
        return {"finite": False}
    e, denom = a - r, np.linalg.norm(r)
    return {
        "finite": True,
        "relative_l2": float(np.linalg.norm(e) / max(float(denom), 1e-30)),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "max_absolute": float(np.max(np.abs(e))),
    }


def pack_check(original, codes, scales, *, hadamard=False, group_size=None):
    """Check max-abs scales and nearest quantization cells independently.

    A 1e-4 cell allowance covers FP32 divide/reciprocal rounding, independently
    of the projection-error gate. It cannot hide a wrong H ordering or scale.
    Exact FP32-reference code mismatch counts are reported separately.
    """
    transformed = hadamard_reference(original, np.float32) if hadamard else original
    width = transformed.shape[1]
    group_size = group_size or width
    values = transformed.reshape(transformed.shape[0], -1, group_size)
    q = codes.reshape(values.shape)
    s = scales.reshape(values.shape[:2])
    if not np.isfinite(s).all() or not np.all(s > 0):
        return {"finite": False, "pass": False, "error": "scales must be positive and finite"}
    peak = np.max(np.abs(values), axis=-1)
    expected_scales = np.where(peak > 0, peak / np.float32(127), 1 if hadamard else 1 / 127)
    expected_scales = expected_scales.astype(np.float32)
    # Runtime's native path uses multiplication by reciprocal; H path divides.
    ideal = values / s[..., None]
    inverse = np.where(peak > 0, np.float32(127) / np.where(peak > 0, peak, np.float32(1)), 127)
    expected = np.clip(np.rint(ideal if hadamard else values * inverse[..., None]), -127, 127)
    scale_error = metrics(s, expected_scales)
    max_cell_error = float(np.max(np.abs(q.astype(np.float64) - ideal.astype(np.float64))))
    finite = bool(np.isfinite(s).all() and np.all(s > 0) and np.isfinite(ideal).all())
    return {
        "finite": finite,
        "scale_error": scale_error,
        "exact_reference_code_mismatches": int(np.count_nonzero(q != expected)),
        "maximum_quantization_cell_error": max_cell_error,
        "pass": bool(
            finite
            and scale_error["relative_l2"] <= 1e-6
            and max_cell_error <= 0.5001
            and np.max(np.abs(q.astype(np.int16))) <= 127
        ),
    }


def analyze(directory):
    d = Path(directory)
    meta = json.loads((d / "driver.json").read_text())
    m, padded, k, n = (meta[x] for x in ("rows", "padded_rows", "K", "N"))
    c, real_rows = meta["sample_columns"], meta["sample_rows"]
    columns = np.fromfile(d / "columns.u32", dtype="<u4")
    if columns.shape != (c,) or np.any(columns >= n):
        raise ValueError("invalid selected output columns")
    x = bf16(d / "input.bf16", (m, k))
    w = bf16(d / "weight-samples.bf16", (c, k))
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("nonfinite original operands")
    reference = x.astype(np.float64) @ w.astype(np.float64).T
    rotated_reference = hadamard_reference(x) @ hadamard_reference(w).T
    dot_preservation = metrics(rotated_reference, reference)
    records = {}
    errors = {}
    for policy in POLICIES:
        hadamard = policy == "hadamard-row"
        groups = k // 1024 if policy == "native-group1024" else 1
        output = bf16(d / f"{policy}.output.bf16", (m, n))
        a_codes = np.fromfile(d / f"{policy}.activation.i8", dtype=np.int8).reshape(padded, k)
        a_scales = np.fromfile(d / f"{policy}.activation-scales.f32", dtype="<f4").reshape(
            padded, groups
        )
        w_codes = np.fromfile(d / f"{policy}.weight-samples.i8", dtype=np.int8).reshape(c, k)
        w_scales = np.fromfile(d / f"{policy}.weight-scales.f32", dtype="<f4")
        a_decoded = a_codes[:m].astype(np.float64).reshape(m, groups, k // groups)
        a_decoded = (a_decoded * a_scales[:m, :, None]).reshape(m, k)
        w_decoded = w_codes.astype(np.float64) * w_scales[:, None]
        decoded = a_decoded @ w_decoded.T
        actual = output[:, columns]
        implementation = metrics(actual, decoded)
        quantization = metrics(decoded[:real_rows], reference[:real_rows])
        total = metrics(actual[:real_rows], reference[:real_rows])
        activation_pack = pack_check(
            x,
            a_codes[:m],
            a_scales[:m],
            hadamard=hadamard,
            group_size=1024 if groups > 1 else None,
        )
        weight_pack = pack_check(w, w_codes, w_scales[:, None], hadamard=hadamard)
        finite = bool(
            np.isfinite(output).all()
            and np.isfinite(a_scales).all()
            and np.isfinite(w_scales).all()
        )
        zeros_pass = bool(np.all(output[real_rows:] == 0))
        padding_pass = bool(np.all(a_codes[m:] == 0) and np.all(a_scales[m:] == 1))
        passed = (
            finite
            and zeros_pass
            and padding_pass
            and activation_pack["pass"]
            and weight_pack["pass"]
            and implementation["finite"]
            and implementation["relative_l2"] <= 8e-3
        )
        errors[policy] = np.abs(actual[:real_rows].astype(np.float64) - reference[:real_rows])
        records[policy] = {
            "status": "passed" if passed else "incorrect",
            "all_output_finite": finite,
            "zero_rows_exact": zeros_pass,
            "padded_rows_codes_zero_scales_one": padding_pass,
            "activation_pack": activation_pack,
            "weight_pack": weight_pack,
            "implementation_error_vs_actual_decoded_operands": implementation,
            "quantization_error_vs_original_bf16_operands": quantization,
            "total_error_vs_original_bf16_operands": total,
            "per_input_row_total_error": [
                metrics(a, b) for a, b in zip(actual[:real_rows], reference[:real_rows])
            ],
        }
    h, base = errors["hadamard-row"], errors["native-group1024"]
    row_h, row_base = np.linalg.norm(h, axis=1), np.linalg.norm(base, axis=1)
    h_total = records["hadamard-row"]["total_error_vs_original_bf16_operands"]
    base_total = records["native-group1024"]["total_error_vs_original_bf16_operands"]
    aggregate_strictly_better = bool(
        h_total["finite"]
        and base_total["finite"]
        and all(h_total[key] < base_total[key] for key in ("relative_l2", "rmse", "max_absolute"))
    )
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(d.iterdir())
        if p.is_file()
    }
    return {
        "schema": 1,
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "frozen FC2 activation rows and selected output channels; full K",
        "metadata": meta,
        "input_sha256": hashes,
        "reference": "independent NumPy float64 full-K products from original BF16 and actual decoded INT8 operands",
        "float64_rotation_dot_preservation": dot_preservation,
        "policies": records,
        "hadamard_vs_native_group1024": {
            "sampled_cells_better": int(np.count_nonzero(h < base)),
            "sampled_cells_equal": int(np.count_nonzero(h == base)),
            "sampled_cells_worse": int(np.count_nonzero(h > base)),
            "per_row_l2_better": int(np.count_nonzero(row_h < row_base)),
            "per_row_l2_worse": int(np.count_nonzero(row_h > row_base)),
            "every_sampled_cell_no_worse": bool(np.all(h <= base)),
            "every_sampled_row_l2_strictly_better": bool(np.all(row_h < row_base)),
            "aggregate_l2_rmse_maxabs_all_strictly_better": aggregate_strictly_better,
            "every_sampled_row_l2_no_worse": bool(np.all(row_h <= row_base)),
            "single_sample_numerics_gate_pass": bool(
                aggregate_strictly_better
                and np.all(row_h <= row_base)
                and all(record["status"] == "passed" for record in records.values())
                and dot_preservation["relative_l2"] <= 1e-12
            ),
        },
        "checkpoint_replacement_eligible": False,
        "replacement_reason": "Projection samples do not establish whole-model denoiser trajectory or final visual/audio quality.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {p: r["total_error_vs_original_bf16_operands"] for p, r in result["policies"].items()}
        )
    )


if __name__ == "__main__":
    main()
