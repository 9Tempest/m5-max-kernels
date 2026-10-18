"""Rerun selected tiles against a fresh MLX BF16 baseline, interleaving each pair."""

import argparse
import json
from pathlib import Path

from m5kernels.engine import environment, run_case, write_result
from m5kernels.locking import gpu_lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--distribution", choices=["normal", "outliers", "zeros"], default="normal")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if args.repeat < 3:
        p.error("at least 3 repeats required")
    with gpu_lock():
        doc = dict(schema_version=1, environment=environment(), results=[], comparisons=[])
        for path in args.inputs:
            for selection in json.loads(Path(path).read_text())["selections"]:
                case = selection["holdout"]["case"]
                baseline = run_case(
                    case, "mlx", "bf16", args.seed, args.distribution, 5, args.repeat
                )
                candidate = run_case(
                    case,
                    "mpp",
                    selection["precision"],
                    args.seed,
                    args.distribution,
                    5,
                    args.repeat,
                    tuple(selection["tile"]),
                )
                doc["results"] += [baseline, candidate]
                comparison = dict(
                    case_id=case["id"],
                    precision=selection["precision"],
                    tile=selection["tile"],
                    selection_record=Path(path).name,
                )
                if all(r["status"] == "ok" and r["quality_pass"] for r in [baseline, candidate]):
                    base = baseline["timing"]["median_ms"]
                    online = candidate["stages"].get(
                        "online_activation_plus_gemm", candidate["timing"]
                    )["median_ms"]
                    comparison.update(baseline_ms=base, candidate_ms=online, speedup=base / online)
                doc["comparisons"].append(comparison)
                print(comparison, flush=True)
        write_result(args.output, doc)
        return int(any(r["status"] in {"error", "incorrect"} for r in doc["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
