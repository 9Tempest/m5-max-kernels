"""Independent NumPy references. No MLX dependency (also runs in CPU CI)."""

import numpy as np


def fp8_decode(q):
    q = np.asarray(q, dtype=np.uint8)
    sign = np.where(q & 128, -1.0, 1.0)
    e = (q >> 3) & 15
    m = q & 7
    out = sign * np.where(
        e == 0, m.astype(float) * 2**-9, (1 + m / 8.0) * np.exp2(e.astype(int) - 7)
    )
    return np.where((e == 15) & (m == 7), np.nan, out).astype(np.float32)


def fp8_encode(x):
    # Exhaustive positive finite codebook: search nearest representable value, ties to even.
    table = fp8_decode(np.arange(127, dtype=np.uint8))
    a = np.minimum(np.abs(x), 448.0)
    hi = np.minimum(np.searchsorted(table, a), 126)
    lo = np.maximum(hi - 1, 0)
    dl = a - table[lo]
    dh = table[hi] - a
    chosen = np.where((dl < dh) | ((dl == dh) & ((lo % 2) == 0)), lo, hi)
    return (chosen | np.where(np.signbit(x), 128, 0)).astype(np.uint8)


def quantize_reference(x, mode):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("quantization input must be finite")
    if mode == "mxint8":
        if x.shape[-1] % 32:
            raise ValueError("K must be divisible by 32")
        z = x.reshape(*x.shape[:-1], -1, 32)
        peak = np.max(np.abs(z), axis=-1, keepdims=True)
        e = np.clip(np.floor(np.log2(np.maximum(peak, np.finfo(np.float32).tiny))) - 6, -127, 127)
        e = np.where(peak == 0, 0, e)
        q = np.clip(np.rint(z / np.exp2(e)), -127, 127).astype(np.int8)
        return q.reshape(x.shape), (e.squeeze(-1) + 127).astype(np.uint8)
    peak = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = np.where(peak == 0, 1.0, peak / (448.0 if mode == "fp8" else 127.0))
    z = x / scale
    q = fp8_encode(z) if mode == "fp8" else np.clip(np.rint(z), -127, 127).astype(np.int8)
    return q, scale.squeeze(-1)


def dequantize(q, s, mode):
    if mode == "mxint8":
        z = q.reshape(*q.shape[:-1], -1, 32).astype(np.float32)
        return (z * np.exp2(s.astype(np.int32) - 127)[..., None]).reshape(q.shape)
    z = fp8_decode(q) if mode == "fp8" else q.astype(np.float32)
    return z * s[..., None]


def metrics(actual, reference):
    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if a.shape != r.shape:
        raise ValueError("reference shape mismatch")
    if not np.isfinite(a).all() or not np.isfinite(r).all():
        return {"finite": False, "relative_l2": None, "max_abs": None, "rmse": None, "cosine": None}
    diff = a - r
    rn = np.linalg.norm(r)
    an = np.linalg.norm(a)
    return {
        "finite": True,
        "relative_l2": float(np.linalg.norm(diff) / max(rn, 1e-30)),
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "cosine": float(np.dot(a.ravel(), r.ravel()) / max(rn * an, 1e-30))
        if rn
        else float(an == 0),
    }


def sample_indices(length, count=32):
    return np.unique(np.linspace(0, length - 1, min(length, count), dtype=int))
