import argparse
import json
import sys
from pathlib import Path

from .workloads import load


def _run(args):
    from .engine import environment, run_case, write_result

    cases = load(args.suite)["cases"]
    cases = [c for c in cases if args.filter in c["id"]]
    if not cases:
        raise ValueError("no cases matched")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    doc = dict(schema_version=1, environment=environment(), suite=args.suite, results=[])
    for c in cases:
        for precision in args.precisions.split(","):
            for backend in args.backends.split(","):
                tile = (256, 1, 1, 1) if c["op"] == "rms_modulate" else tuple(args.tile)
                r = run_case(
                    c,
                    backend,
                    precision,
                    args.seed,
                    args.distribution,
                    args.warmup,
                    args.repeat,
                    tile,
                )
                doc["results"].append(r)
                ms = r.get("timing", {}).get("median_ms")
                print(
                    c["id"],
                    backend,
                    precision,
                    r["status"],
                    f"{ms:.3f} ms" if ms is not None else r.get("reason", ""),
                    flush=True,
                )
    write_result(args.output, doc)
    if any(r["status"] in {"error", "incorrect"} for r in doc["results"]):
        return 1
    return 0


def _tune(args):
    from .engine import environment, run_case, write_result

    cases = [c for c in load(args.suite)["cases"] if args.filter in c["id"] and c["op"] == "gemm"]
    if not cases:
        raise ValueError("no GEMM cases matched")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    doc = dict(
        schema_version=1, environment=environment(), suite=args.suite, results=[], selections=[]
    )
    tiles = [
        (32, 64, 4, 1),
        (64, 64, 4, 1),
        (64, 64, 4, 8),
        (64, 128, 4, 8),
        (128, 64, 4, 8),
        (128, 128, 4, 8),
        (64, 128, 8, 8),
        (128, 128, 8, 8),
    ]
    for c in cases:
        for precision in args.precisions.split(","):
            runs = []
            for tile in tiles:
                r = run_case(
                    c,
                    "mpp",
                    precision,
                    args.seed,
                    args.distribution,
                    args.warmup,
                    args.repeat,
                    tile,
                )
                doc["results"].append(r)
                runs.append(r)
                print(
                    c["id"],
                    precision,
                    tile,
                    r["status"],
                    r.get("timing", {}).get("median_ms"),
                    flush=True,
                )
            valid = [r for r in runs if r["status"] == "ok" and r["quality_pass"]]
            if valid:
                best = min(
                    valid,
                    key=lambda r: r.get("stages", {}).get(
                        "online_activation_plus_gemm", r["timing"]
                    )["median_ms"],
                )
                # Fresh holdout seed, fresh timing; selected tile remains fixed.
                holdout = run_case(
                    c,
                    "mpp",
                    precision,
                    args.seed + 1,
                    args.distribution,
                    args.warmup,
                    args.repeat,
                    tuple(best["tile"]),
                )
                doc["selections"].append(
                    dict(case_id=c["id"], precision=precision, tile=best["tile"], holdout=holdout)
                )
    write_result(args.output, doc)
    return (
        0
        if doc["selections"] and all(s["holdout"]["status"] == "ok" for s in doc["selections"])
        else 1
    )


def _report(args):
    docs = [json.loads(Path(p).read_text()) for p in args.inputs]
    print("| Case | Backend | Precision | Median ms | Effective TOPS | Relative L2 | Status |")
    print("|---|---|---|---:|---:|---:|---|")
    for doc in docs:
        for r in doc["results"]:
            ms = r.get("timing", {}).get("median_ms")
            tops = r.get("effective_tops")
            err = r.get("quality_error", {}).get("relative_l2")
            cells = [
                r["case"]["id"],
                r["backend"],
                r["precision"],
                f"{ms:.4f}" if ms is not None else "—",
                f"{tops:.2f}" if tops is not None else "—",
                f"{err:.5f}" if err is not None else "—",
                r["status"],
            ]
            print("| " + " | ".join(cells) + " |")
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Apple Silicon numerical accuracy and kernel benchmark harness"
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    shapes = sub.add_parser("shapes")
    shapes.add_argument("--suite", default="h3")
    for name in ["bench", "tune"]:
        q = sub.add_parser(name)
        q.add_argument("--suite", default="smoke" if name == "bench" else "h3")
        q.add_argument("--filter", default="")
        q.add_argument("--precisions", default="bf16")
        q.add_argument("--backends", default="mlx,mpp")
        q.add_argument(
            "--tile",
            type=int,
            nargs=4,
            default=[64, 64, 4, 1],
            metavar=("M", "N", "SIMDGROUPS", "M_GROUP"),
        )
        q.add_argument("--seed", type=int, default=17)
        q.add_argument("--distribution", choices=["normal", "outliers", "zeros"], default="normal")
        q.add_argument("--warmup", type=int, default=3)
        q.add_argument("--repeat", type=int, default=10)
        q.add_argument("--output", required=True)
    report = sub.add_parser("report")
    report.add_argument("inputs", nargs="+")
    args = p.parse_args()
    try:
        if args.command == "doctor":
            from .engine import environment

            print(json.dumps(environment(), indent=2))
            return 0
        if args.command == "shapes":
            print(json.dumps(load(args.suite), indent=2))
            return 0
        if args.command == "report":
            return _report(args)
        if args.warmup < 1 or args.repeat < 3:
            raise ValueError("warmup >=1 and repeat >=3 required")
        if any(v <= 0 for v in args.tile) or args.tile[2] > 32:
            raise ValueError("invalid tile")
        from .locking import gpu_lock

        with gpu_lock():
            return _run(args) if args.command == "bench" else _tune(args)
    except (ValueError, KeyError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
