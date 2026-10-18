"""Download the pinned original FL2VA checkpoint without copying weights into Git.

Uses curl for resumable HTTPS transfers and verifies every LFS SHA-256 against
the pinned Hugging Face tree API. No authentication is requested or logged.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import fcntl
import json
from pathlib import Path
import subprocess
import time


REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
MANIFEST_SHA256 = "7597d33ca71af128705e802150b4b39395cf20f96bddc45d2041d668145c27be"
BASE = f"https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/{REVISION}"
ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(entry, target):
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe manifest path")
    path = target / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.get("lfs", {}).get("oid")
    size = entry.get("size")
    started = time.monotonic()
    if not path.exists():
        temporary = path.with_name(path.name + ".partial")
        print(f"download {relative} ({(size or 0) / 1e9:.3f} GB)", flush=True)
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "5",
                "--retry-delay",
                "3",
                "--connect-timeout",
                "30",
                "--continue-at",
                "-",
                f"{BASE}/{relative.as_posix()}",
                "--output",
                str(temporary),
            ],
            check=True,
        )
        candidate = temporary
    else:
        candidate = path
    if size is not None and candidate.stat().st_size != size:
        raise ValueError(f"size mismatch: {relative}")
    actual = sha256(candidate)
    if expected and actual != expected:
        raise ValueError(f"SHA-256 mismatch: {relative}")
    if candidate != path:
        candidate.rename(path)
    elapsed = time.monotonic() - started
    print(f"verified {relative} ({elapsed:.1f}s)", flush=True)
    return {
        "path": relative.as_posix(),
        "size": path.stat().st_size,
        "sha256": actual,
        "lfs_sha256": expected,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=Path(__file__).with_name("model-files.json"))
    parser.add_argument("--target", type=Path, default=ROOT / ".cache/models/MiniMax-H3")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be in [1,32]")
    if sha256(args.tree) != MANIFEST_SHA256:
        parser.error("manifest does not match the pinned FL2VA snapshot")
    tree = json.loads(args.tree.read_text())
    # Original runtime needs no upstream Python implementation files.
    entries = [
        entry
        for entry in tree
        if entry["type"] == "file"
        and Path(entry["path"]).suffix in {".safetensors", ".json", ".txt"}
    ]
    entries += [{"path": "LICENSE"}, {"path": "README.md"}, {"path": "model_index.json"}]
    args.target.mkdir(parents=True, exist_ok=True)
    lock = (args.target / ".download.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another downloader owns this model directory")
    output = args.target / f"download-manifest-{time.time_ns()}.json"
    started = time.monotonic()
    records, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, entry, args.target): entry for entry in entries}
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as exc:
                error = {"path": futures[future]["path"], "error": str(exc)}
                errors.append(error)
                print(json.dumps(error), flush=True)
    result = {
        "model": "MiniMaxAI/MiniMax-H3",
        "revision": REVISION,
        "tree_sha256": sha256(args.tree),
        "files": sorted(records, key=lambda x: x["path"]),
        "errors": errors,
        "elapsed_seconds": time.monotonic() - started,
    }
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"manifest: {output}", flush=True)
    lock.close()
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
