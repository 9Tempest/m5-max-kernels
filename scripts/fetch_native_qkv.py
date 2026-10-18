"""Fetch only pinned MIT h3.c source for the native grouped-QKV comparison.

Existing destinations are verified read-only. No pull, reset, fetch, checkout,
build, model download, or GPU work is performed in an existing checkout.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from m5kernels.native_qkv import (
    REPOSITORY_ROOT,
    UPSTREAM_COMMIT,
    UPSTREAM_URL,
    verify_upstream,
)


def fetch(destination=None):
    """Create a new pinned checkout, or verify an existing one without changes.

    Atomic mkdir reserves the destination before cloning, so an existing empty
    directory or a concurrently created path is never repurposed by this helper.
    Failed clones/checkouts are cleaned up only when this invocation created the
    destination. Git LFS smudging and submodule recursion are disabled; there are
    no model downloads or build steps.
    """
    target = Path(destination) if destination is not None else REPOSITORY_ROOT / ".cache/h3.c"
    target = target.expanduser().absolute()

    def verified(created):
        return {
            "destination": str(target),
            "created": created,
            "provenance": verify_upstream(target),
            "content": "pinned source only; no model weights or builds",
        }

    # lexists includes broken symlinks: they count as existing and are untouched.
    if os.path.lexists(target):
        return verified(False)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.mkdir()
    except FileExistsError:
        return verified(False)

    environment = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
    git = ["git", "-c", f"core.hooksPath={os.devnull}"]
    try:
        subprocess.run(
            [*git, "clone", "--no-checkout", "--no-recurse-submodules", UPSTREAM_URL, str(target)],
            check=True,
            env=environment,
        )
        subprocess.run(
            [*git, "-C", str(target), "checkout", "--detach", UPSTREAM_COMMIT],
            check=True,
            env=environment,
        )
        return verified(True)
    except BaseException:
        # This directory was created above by this invocation, never borrowed.
        shutil.rmtree(target)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=REPOSITORY_ROOT / ".cache/h3.c",
        help="source checkout path (existing paths are only verified)",
    )
    args = parser.parse_args(argv)
    try:
        result = fetch(args.destination)
    except (ValueError, NotImplementedError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"fetch_native_qkv: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
