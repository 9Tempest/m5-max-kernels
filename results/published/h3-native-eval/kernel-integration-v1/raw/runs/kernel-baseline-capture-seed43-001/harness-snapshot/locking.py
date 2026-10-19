"""Coordinate local agents without running simultaneous GPU benchmarks."""

from contextlib import contextmanager
import fcntl
import os
import tempfile
from pathlib import Path


@contextmanager
def gpu_lock():
    path = Path(tempfile.gettempdir()) / f"m5-max-kernels-{os.getuid()}.lock"
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another m5bench process owns the GPU experiment lock") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
