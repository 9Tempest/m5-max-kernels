"""Compare decoded H3 media; similarity metrics are not a perceptual quality gate."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run import path_label, run_logged, sha256, write_json


def decode(ffmpeg, path, kind):
    options = (
        ["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24"]
        if kind == "video"
        else ["-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-ar", "32000", "-ac", "2"]
    )
    data = subprocess.check_output([ffmpeg, "-v", "error", "-i", str(path), *options, "-"])
    return np.frombuffer(data, dtype=np.uint8 if kind == "video" else "<f4")


def metrics(reference, candidate, peak):
    if reference.shape != candidate.shape or not reference.size:
        raise ValueError("decoded sizes differ or are empty")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("nonfinite decoded media")
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    return dict(
        elements=a.size,
        mse=mse,
        rmse=math.sqrt(mse),
        maximum_absolute_error=float(np.max(np.abs(a - b))),
        psnr_db=10 * math.log10(peak * peak / mse) if mse else None,
        exactly_equal=bool(np.array_equal(reference, candidate)),
        reference_min=float(a.min()),
        reference_max=float(a.max()),
        candidate_min=float(b.min()),
        candidate_max=float(b.max()),
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("candidate", type=Path)
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError("FFmpeg/FFprobe required")
    record = dict(
        status="failed",
        metric_scope="decoded RGB pixels and stereo PCM; no motion alignment; "
        "similarity is not a semantic/perceptual quality acceptance decision",
        recipe_note="caller must match prompt, seed, sampler and dimensions before attribution",
        files={},
    )
    try:
        probes = []
        for name, path in (("reference", args.reference), ("candidate", args.candidate)):
            probe = json.loads(
                subprocess.check_output(
                    [ffprobe, "-v", "error", "-show_streams", "-of", "json", str(path)], text=True
                )
            )
            probes.append(probe)
            record["files"][name] = dict(path=path_label(path), sha256=sha256(path), probe=probe)
            result = run_logged(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-n",
                    "-i",
                    str(path.resolve()),
                    "-vf",
                    "fps=1,scale=256:-1,tile=5x1",
                    "-frames:v",
                    "1",
                    str(out / f"{name}.png"),
                ],
                Path.cwd(),
                os.environ.copy(),
                out,
                name + "-contact-sheet",
            )
            if result["returncode"]:
                raise ValueError("contact sheet generation failed")
        geometries = []
        for probe in probes:
            video = next(s for s in probe["streams"] if s["codec_type"] == "video")
            geometries.append(tuple(video.get(k) for k in ("width", "height", "avg_frame_rate")))
        if geometries[0] != geometries[1]:
            raise ValueError("video geometry/fps differ")
        record["video_rgb"] = metrics(
            decode(ffmpeg, args.reference, "video"), decode(ffmpeg, args.candidate, "video"), 255
        )
        record["audio_pcm32k_stereo"] = metrics(
            decode(ffmpeg, args.reference, "audio"), decode(ffmpeg, args.candidate, "audio"), 1
        )
        record["ssim_process"] = run_logged(
            [
                ffmpeg,
                "-v",
                "info",
                "-i",
                str(args.reference.resolve()),
                "-i",
                str(args.candidate.resolve()),
                "-lavfi",
                "[0:v][1:v]ssim",
                "-an",
                "-f",
                "null",
                "-",
            ],
            Path.cwd(),
            os.environ.copy(),
            out,
            "ssim",
        )
        if record["ssim_process"]["returncode"]:
            raise ValueError("SSIM calculation failed")
        record["status"] = "completed"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    write_json(out / "comparison.json", record)
    print(json.dumps({"status": record["status"], "output": path_label(out)}))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
