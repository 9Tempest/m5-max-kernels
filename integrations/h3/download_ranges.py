"""Optional parallel HTTP Range completion of ONE pinned large H3 weight file.

Use only after the original downloader and its curl children have stopped. The
existing .partial prefix is never changed. Tail chunks remain resumable in a
.ranges directory; a complete, SHA256-verified assembly is atomically renamed.
This trades extra disk space and local I/O for possible single-connection relief.
No signed redirect URLs or raw transport exceptions are printed or persisted.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
MANIFEST_SHA256 = "7597d33ca71af128705e802150b4b39395cf20f96bddc45d2041d668145c27be"
BASE = f"https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/{REVISION}"
BUFFER_BYTES = 1024 * 1024


class DownloadError(ValueError):
    """A deliberately sanitized error safe to put in a download record."""


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def safe_error(error):
    if isinstance(error, DownloadError):
        return str(error)
    if isinstance(error, HTTPError):
        return f"HTTP request failed with status {error.code}"
    return (
        f"operation failed ({type(error).__name__}); raw details omitted to protect redirect URLs"
    )


def selected_entry(manifest_path, relative):
    data = manifest_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
        raise DownloadError("manifest does not match the pinned H3 snapshot")
    entries = [item for item in json.loads(data) if item.get("path") == relative]
    if len(entries) != 1:
        raise DownloadError("file must match exactly one pinned manifest entry")
    return entries[0]


def validate_entry(entry):
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
        raise DownloadError("only a safe relative .safetensors manifest path is accepted")
    size, digest = entry.get("size"), entry.get("lfs", {}).get("oid", "")
    if type(size) is not int or size <= 0 or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise DownloadError("a positive size and pinned LFS SHA256 are required")
    return relative, size, digest


@contextmanager
def model_lock(target):
    target.mkdir(parents=True, exist_ok=True)
    with (target / ".download.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DownloadError(
                "another downloader owns the model directory; stop it first"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def assert_no_legacy_writer(prefix):
    # Older download.py processes may predate .download.lock. Never print ps/lsof
    # output: curl arguments may include signed URLs from unrelated processes.
    ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10)
    if ps.returncode:
        raise DownloadError("cannot inspect legacy downloader processes; refusing to proceed")
    for line in ps.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or fields[0] == str(os.getpid()):
            continue
        try:
            argv = shlex.split(fields[1])
        except ValueError:
            argv = fields[1].split()
        if any(arg == "download.py" or arg.endswith("/h3/download.py") for arg in argv):
            raise DownloadError(
                "legacy H3 downloader is running; stop it and its curl children first"
            )
    if not prefix.exists():
        return
    lsof = shutil.which("lsof")
    if not lsof:
        raise DownloadError("lsof is required to check for orphaned prefix writers")
    owners = subprocess.run(
        [lsof, "-t", "--", str(prefix)], capture_output=True, text=True, timeout=10
    )
    if owners.returncode not in (0, 1) or owners.stderr.strip():
        raise DownloadError("cannot inspect prefix ownership; refusing to proceed")
    if owners.stdout.strip():
        raise DownloadError(
            "the existing prefix is open in another process; stop its writers first"
        )


def fingerprint(path):
    if not path.exists():
        return None
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def fetch_part(url, path, start, end, total, opener=urlopen, timeout=60):
    wanted = end - start + 1
    present = path.stat().st_size if path.exists() else 0
    receipt_path = path.with_name(path.name + ".verified.json")
    if present > wanted:
        raise DownloadError("saved range chunk is larger than its declared interval")
    if present == wanted:
        if receipt_path.exists():
            actual = sha256(path)
            expected_receipt = dict(start=start, end=end, total=total, bytes=wanted, sha256=actual)
            if json.loads(receipt_path.read_text()) != expected_receipt:
                raise DownloadError("saved verified chunk changed; chunk and receipt retained")
            return dict(start=start, end=end, bytes=wanted, sha256=actual, resumed_bytes=present)
        # A previous body may have had excess bytes or failed after its last read.
        # Preserve that unvalidated chunk and retry this interval, never the prefix.
        path.rename(path.with_name(path.name + f".unverified-{time.time_ns()}"))
        present = 0
    next_byte = start + present
    request = Request(
        url, headers={"Range": f"bytes={next_byte}-{end}", "Accept-Encoding": "identity"}
    )
    with opener(request, timeout=timeout) as response:
        if response.status != 206:
            raise DownloadError("range request did not return HTTP 206; prefix and chunks retained")
        value = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
        if not match or tuple(map(int, match.groups())) != (next_byte, end, total):
            raise DownloadError("HTTP Content-Range does not exactly match the requested interval")
        length = response.headers.get("Content-Length")
        if length is not None and (not length.isdigit() or int(length) != end - next_byte + 1):
            raise DownloadError("HTTP Content-Length does not match the requested interval")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise DownloadError("compressed range responses are not accepted")
        remaining = wanted - present
        with path.open("ab") as handle:
            while remaining:
                block = response.read(min(BUFFER_BYTES, remaining))
                if not block:
                    raise DownloadError(
                        "range body was truncated; partial chunk retained for resume"
                    )
                if len(block) > remaining:
                    raise DownloadError("range body exceeded its declared interval")
                handle.write(block)
                remaining -= len(block)
            if response.read(1):
                raise DownloadError("range body exceeded its declared interval")
            handle.flush()
            os.fsync(handle.fileno())
    if path.stat().st_size != wanted:
        raise DownloadError("completed range chunk size mismatch")
    actual = sha256(path)
    write_json(receipt_path, dict(start=start, end=end, total=total, bytes=wanted, sha256=actual))
    return dict(start=start, end=end, bytes=wanted, sha256=actual, resumed_bytes=present)


def download_one(entry, target, workers=4, chunk_bytes=64 * 1024**2, opener=urlopen, guard=None):
    relative, total, expected = validate_entry(entry)
    if not 1 <= workers <= 16 or chunk_bytes <= 0:
        raise DownloadError("workers must be in [1,16] and chunk size must be positive")
    target = target.resolve()
    path = target / relative
    prefix = path.with_name(path.name + ".partial")
    state_dir = path.with_name(path.name + ".ranges")
    guard = assert_no_legacy_writer if guard is None else guard
    with model_lock(target):
        guard(prefix)
        if path.exists():
            if path.stat().st_size != total or sha256(path) != expected:
                raise DownloadError(
                    "existing final file fails pinned size/SHA256; it was not changed"
                )
            return dict(status="already_verified", path=relative.as_posix(), sha256=expected)
        initial = fingerprint(prefix)
        prefix_size = initial[2] if initial else 0
        if prefix_size > total:
            raise DownloadError("existing prefix is larger than the pinned file size")
        prefix_hash = sha256(prefix) if initial else hashlib.sha256(b"").hexdigest()
        if fingerprint(prefix) != initial:
            raise DownloadError("existing prefix changed during inspection; stop all writers")
        ranges = [
            [start, min(total, start + chunk_bytes) - 1]
            for start in range(prefix_size, total, chunk_bytes)
        ]
        plan = dict(
            revision=REVISION,
            path=relative.as_posix(),
            size=total,
            lfs_sha256=expected,
            prefix_bytes=prefix_size,
            prefix_sha256=prefix_hash,
            chunk_bytes=chunk_bytes,
            ranges=ranges,
        )
        state_dir.mkdir(parents=True, exist_ok=True)
        plan_path = state_dir / "plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise DownloadError(
                    "saved range plan differs from prefix or chunk size; no files changed"
                )
        else:
            write_json(plan_path, plan)
        result_path = state_dir / f"attempt-{time.time_ns()}.json"
        started = time.monotonic()
        result = dict(status="failed", plan=plan, chunks=[], errors=[])
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        fetch_part,
                        f"{BASE}/{relative.as_posix()}",
                        state_dir / f"{start}-{end}.part",
                        start,
                        end,
                        total,
                        opener,
                    ): (start, end)
                    for start, end in ranges
                }
                for future in as_completed(futures):
                    start, end = futures[future]
                    try:
                        result["chunks"].append(future.result())
                        print(
                            json.dumps(dict(event="range_complete", start=start, end=end)),
                            flush=True,
                        )
                    except Exception as error:
                        result["errors"].append(dict(start=start, end=end, error=safe_error(error)))
            if result["errors"]:
                raise DownloadError(
                    "one or more ranges failed; all prefix/chunk data retained for resume"
                )
            guard(prefix)
            if fingerprint(prefix) != initial:
                raise DownloadError("existing prefix changed during download; refusing assembly")
            assembly = state_dir / f"assembly-{time.time_ns()}.partial"
            with assembly.open("xb") as output:
                sources = ([prefix] if initial else []) + [
                    state_dir / f"{start}-{end}.part" for start, end in ranges
                ]
                for source in sources:
                    with source.open("rb") as handle:
                        shutil.copyfileobj(handle, output, 8 * BUFFER_BYTES)
                output.flush()
                os.fsync(output.fileno())
            if assembly.stat().st_size != total or sha256(assembly) != expected:
                raise DownloadError(
                    "assembled file failed pinned size/SHA256; original prefix retained"
                )
            guard(prefix)
            if fingerprint(prefix) != initial or path.exists():
                raise DownloadError(
                    "destination or original prefix changed; verified assembly retained"
                )
            assembly.rename(path)
            result.update(
                status="verified",
                sha256=expected,
                bytes=total,
                retained_prefix_bytes=prefix_size,
                retained_range_bytes=total - prefix_size,
            )
        except KeyboardInterrupt:
            result.update(
                status="interrupted", error="interrupted; prefix and range chunks retained"
            )
        except Exception as error:
            result["error"] = safe_error(error)
        finally:
            result.update(elapsed_seconds=time.monotonic() - started)
            result["chunks"].sort(key=lambda item: item["start"])
            write_json(result_path, result)
        return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--file", required=True, help="exact relative path in the pinned model manifest"
    )
    cli.add_argument("--target", type=Path, default=ROOT / ".cache/models/MiniMax-H3")
    cli.add_argument("--tree", type=Path, default=Path(__file__).with_name("model-files.json"))
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--chunk-mib", type=int, default=64)
    args = cli.parse_args()
    try:
        result = download_one(
            selected_entry(args.tree, args.file),
            args.target,
            args.workers,
            args.chunk_mib * 1024**2,
        )
        print(
            json.dumps(dict(status=result["status"], error=result.get("error"), path=args.file)),
            flush=True,
        )
        return 0 if result["status"] in ("verified", "already_verified") else 1
    except Exception as error:
        print(json.dumps(dict(status="refused", error=safe_error(error))), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
