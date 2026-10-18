"""Paired plain/rotated GEMM experiment with fixed offline transforms and fresh-seed reruns."""

import argparse
import hashlib
import json
from pathlib import Path

from m5kernels.engine import environment, run_case, write_result
from m5kernels.locking import gpu_lock
from m5kernels.workloads import load


def online_ms(result):
    if result["status"] != "ok":
        return None
    return result.get("stages", {}).get("online_activation_plus_gemm", result["timing"])[
        "median_ms"
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", default="h3.attn_q.m4096,h3.ffn_up.m4096,h3.ffn_down.m4096")
    p.add_argument("--blocks", default="64,128,256")
    p.add_argument("--sign-seeds", default="none,0")
    p.add_argument("--implementations", default="metal")
    p.add_argument("--precisions", default="int8")
    p.add_argument("--distributions", default="normal,outliers")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.repeat < 3:
        p.error("repeat >= 3 required")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    cases = {c["id"]: c for c in load("h3")["cases"]}
    # Fixed pre-existing GEMM winners: this campaign changes the rotation, not the tile.
    tiles = {
        "h3.attn_q.m4096": (128, 64, 4, 8),
        "h3.ffn_up.m4096": (128, 64, 4, 8),
        "h3.ffn_down.m4096": (128, 128, 8, 8),
    }
    with gpu_lock():
        doc = dict(
            schema_version=1,
            environment=environment(),
            results=[],
            experiment="same-input normal/outlier ablation; offline W rotation+quantization",
            campaign_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        for name in args.cases.split(","):
            case = cases[name]
            tile = tiles.get(name, (64, 64, 4, 8))
            for distribution in args.distributions.split(","):
                configurations = [("mlx", "bf16", case)]
                for precision in args.precisions.split(","):
                    configurations.append(("mpp", precision, case))
                    if precision != "int8":
                        continue
                    for implementation in args.implementations.split(","):
                        for block in map(int, args.blocks.split(",")):
                            for sign in args.sign_seeds.split(","):
                                config = dict(
                                    block_size=block,
                                    sign_seed=None if sign == "none" else int(sign),
                                    implementation=implementation,
                                )
                                configurations.append(
                                    ("mpp-hadamard", precision, dict(case, hadamard=config))
                                )
                for backend, precision, configured in configurations:
                    r = run_case(
                        configured,
                        backend,
                        precision,
                        args.seed,
                        distribution,
                        5,
                        args.repeat,
                        tile,
                    )
                    doc["results"].append(r)
                    print(
                        json.dumps(
                            dict(
                                case=name,
                                distribution=distribution,
                                backend=backend,
                                precision=precision,
                                rotation=configured.get("hadamard"),
                                status=r["status"],
                                online_ms=online_ms(r),
                                quality=r.get("quality_error", {}).get("relative_l2"),
                                quality_pass=r.get("quality_pass"),
                                reason=r.get("reason"),
                            )
                        ),
                        flush=True,
                    )
        write_result(args.output, doc)
        return int(any(r["status"] in {"error", "incorrect"} for r in doc["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
