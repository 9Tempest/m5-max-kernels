"""Versioned, explicit workload shapes; user JSON uses the same schema."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def validate(c):
    if not isinstance(c.get("id"), str) or not c["id"]:
        raise ValueError("nonempty id required")
    fields = {
        "gemm": ("m", "n", "k"),
        "norm_hadamard_gemm": ("m", "n", "k"),
        "attention": ("batch", "heads", "seq", "head_dim"),
        "rms_modulate": ("m", "n"),
        "swiglu": ("m", "n"),
    }[c["op"]]
    for f in fields:
        if type(c.get(f)) is not int or c[f] < 1:
            raise ValueError(f"{f} must be a positive integer")
    return c


def load(name):
    p = Path(name)
    if not p.exists():
        p = ROOT / "workloads" / f"{name}.json"
    data = json.loads(p.read_text())
    cases = [validate(c) for c in data["cases"]]
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("duplicate case id")
    return data


def h3_cases(tokens=(128, 1024, 4096)):
    cfg = json.loads((ROOT / "sources/transformer-config.json").read_text())
    h = cfg["hidden_size"]
    d = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    f = cfg["ffn_dim"]
    projections = {
        "attn_q": (h, d),
        "attn_k": (h, d),
        "attn_v": (h, d),
        "attn_qkv_fused": (h, 3 * d),
        "attn_out": (d, h),
        "ffn_gate": (h, f),
        "ffn_up": (h, f),
        "ffn_gate_up_fused": (h, 2 * f),
        "ffn_down": (f, h),
        "text_in": (cfg["text_dim"], h),
        "video_in": (96, h),
        "audio_in": (32, h),
        "video_out": (h, 96),
        "audio_out": (h, 32),
    }
    cases = []
    for m in tokens:
        for name, (k, n) in projections.items():
            cases.append(
                dict(
                    id=f"h3.{name}.m{m}",
                    op="gemm",
                    m=m,
                    n=n,
                    k=k,
                    component=name,
                    shape_origin="official config; synthetic token count",
                )
            )
        cases += [
            dict(
                id=f"h3.attention.s{m}",
                op="attention",
                batch=1,
                heads=56,
                seq=m,
                head_dim=128,
                causal=False,
                component="dit_attention",
            ),
            dict(id=f"h3.rms_modulate.m{m}", op="rms_modulate", m=m, n=h),
            dict(id=f"h3.swiglu.m{m}", op="swiglu", m=m, n=f),
        ]
    # AdaLN is precomputable: benchmark separately, never multiply by packed sequence length.
    for name, k, n in [
        ("adaln", 2688, 96768),
        ("final_adaln", 2688, 10752),
        ("time_in", 256, 5376),
        ("time_out", 5376, 2688),
    ]:
        cases.append(dict(id=f"h3.{name}", op="gemm", m=1, k=k, n=n, component=name))
    return cases
