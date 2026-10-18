"""Compile and execute the experimental native scaled-MXINT8 API path in isolation."""

import json
from pathlib import Path
import subprocess
import sys

from m5kernels.engine import environment, write_result
from m5kernels.locking import gpu_lock


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: uv run python scripts/probe_capabilities.py output.json")
    output = Path(sys.argv[1])
    if output.exists():
        raise FileExistsError(output)
    code = """import mlx.core as mx
from m5kernels.metal import quantize, gemm
x=mx.ones((32,128),dtype=mx.bfloat16)
q,s=quantize(x,"mxint8");mx.eval(q,s)
y=gemm(q,q,"mxint8",a_scales=s,b_scales=s)
mx.eval(y)
print(y[0,0].item())
"""
    with gpu_lock():
        doc = dict(
            schema_version=1,
            environment=environment(),
            path="MPP INT8 tensor with E8M0 scale plane, block32, FP32 accumulator",
            caution="One API/implementation probe; failure does not prove hardware impossibility",
        )
        try:
            run = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
            )
            doc.update(
                status="executed" if run.returncode == 0 else "failed",
                returncode=run.returncode,
                diagnostic=(run.stdout + run.stderr).replace(str(Path.home()), "$HOME")[-16000:],
            )
        except subprocess.TimeoutExpired:
            doc.update(status="timeout", diagnostic="Compilation/execution exceeded 60 seconds")
        write_result(output, doc)
        print(json.dumps(dict(status=doc["status"], output=str(output))))


if __name__ == "__main__":
    main()
