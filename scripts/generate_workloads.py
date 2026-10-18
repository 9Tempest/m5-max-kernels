"""Regenerate source-derived microbenchmark matrices. Run from the repository root."""

import json
from pathlib import Path
from m5kernels.workloads import h3_cases


def save(name, cases, **meta):
    Path(f"workloads/{name}.json").write_text(
        json.dumps(dict(schema_version=1, **meta, cases=cases), indent=2) + "\n"
    )


save(
    "h3",
    h3_cases(),
    description="H3 transformer operator inventory; token counts are synthetic, not a runtime trace.",
)
save(
    "smoke",
    [
        dict(id="smoke.gemm", op="gemm", m=127, n=96, k=256),
        dict(
            id="smoke.attention",
            op="attention",
            batch=1,
            heads=2,
            seq=128,
            head_dim=128,
            causal=False,
        ),
        dict(id="smoke.rms", op="rms_modulate", m=127, n=5376),
    ],
)
save(
    "h3-long",
    [
        dict(
            id=f"h3.attention.s{s}",
            op="attention",
            batch=1,
            heads=56,
            seq=s,
            head_dim=128,
            causal=False,
        )
        for s in [8192, 16384, 32768]
    ],
    description="Long-context synthetic attention. Run explicitly; not the default smoke suite.",
)
t = json.loads(Path("sources/text_encoder-config.json").read_text())["text_config"]
c = []
for m in [1, 128, 1024]:
    for name, k, n in [
        ("q", 5120, 8192),
        ("kv", 5120, 1024),
        ("out", 8192, 5120),
        ("gate_up", 5120, 51200),
        ("down", 25600, 5120),
    ]:
        c.append(dict(id=f"qwen3vl32b.{name}.m{m}", op="gemm", m=m, k=k, n=n))
save(
    "llm",
    c,
    description="H3 text encoder / Qwen3-VL-32B projection shapes; supports decode and prefill studies.",
)
save(
    "h3-norm-hadamard",
    [
        dict(
            id=f"h3.norm_{name}.m{m}",
            op="norm_hadamard_gemm",
            m=m,
            n=n,
            k=5376,
            fusion=dict(norm="rms", block_size=256, round_output="input"),
            shape_origin="official H3 widths; scenario token count recorded explicitly",
        )
        for m in [128, 4096, 9914]
        for name, n in [("q", 7168), ("ffn_up", 14336)]
    ],
    description=(
        "RMSNorm/AdaLN + H256 + INT8 + projection; shared modulation vector, native BF16 "
        "rounding boundary. 128/4096 synthetic; 9914 is the integration task's 512x512 "
        "124-frame request (9472 video + 414 audio + 28 text), still synthetic tensor values."
    ),
)
save(
    "h3-gated-norm-hadamard",
    [
        dict(
            id=f"h3.gated_{name}.m{m}",
            op="gated_norm_hadamard_gemm",
            m=m,
            n=n,
            k=5376,
            fusion=dict(norm="rms", block_size=256, round_output="input"),
            shape_origin="official H3 widths; scenario token count recorded explicitly",
        )
        for m in [128, 4096, 9914]
        for name, n in [("q", 7168), ("ffn_up", 14336)]
    ],
    description=(
        "Gated residual add + RMSNorm/AdaLN + H256 + INT8 + projection, including the "
        "updated BF16 residual output. Shared gate and upcoming modulation are independent; "
        "synthetic values. 9914 matches the integration request's recorded token count."
    ),
)
save(
    "rms-quant",
    [
        dict(
            id=f"rms.{name}.m{m}",
            op="rms_quant_gemm",
            m=m,
            n=7168,
            k=5376,
            rms_quant=dict(block_size=block, gamma=gamma, threads=256, epsilon=1e-5),
            shape_origin="H3 Q widths; synthetic zero-shift RMS inputs, alternate FP32 contract",
        )
        for m in [4096, 9914]
        for name, gamma, block in [
            ("plain", "identity", 1),
            ("weighted", "random", 1),
            ("weighted_h256", "random", 256),
        ]
    ],
    description=(
        "Direct versus scale-folded RMS dynamic INT8 preparation + Q-sized GEMM. "
        "No additive shift or BF16 normalized intermediate. This is not the H3 AdaLN contract. "
        "Identical BF16 inputs/weights and FP32 gamma within each paired seed."
    ),
)
save(
    "h3-shift-split",
    [
        dict(
            id=f"h3.shift_q.m{m}",
            op="norm_hadamard_gemm",
            m=m,
            n=7168,
            k=5376,
            fusion=dict(norm="rms", block_size=256, round_output="fp32", threads=256),
            shape_origin="H3 projection widths; synthetic shared conditioning row, alternate FP32 contract",
        )
        for m in [4096, 9914]
    ],
    description=(
        "Move AdaLN shift through GEMM to a projected bias and fold RMS into activation scales. "
        "Shared P=1 condition microbenchmark; original BF16 weights for bias, INT8 main GEMM. "
        "Condition preparation is measured separately and in an inclusive stage."
    ),
)
