"""Strict paired H3 output-drift evaluation; no model-quality pass/fail claim.

Portable: Python 3.11+, NumPy, and (for media only) FFmpeg/FFprobe. No GPU or
model downloads. Outputs are exclusive and preserve failures and raw evidence.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_provenance():
    try:
        base = ["git", "-C", str(Path(__file__).resolve().parent)]
        revision = subprocess.check_output(base + ["rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        dirty = subprocess.check_output(base + ["status", "--porcelain"], stderr=subprocess.DEVNULL)
        return {"status": "available", "revision": revision.decode().strip(), "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"status": "unavailable", "reason": "script is outside an accessible Git checkout"}


def score(value=None, status="finite", reason=None):
    result = {"status": status, "value": value}
    if reason:
        result["reason"] = reason
    return result


def from_log(log_value):
    if log_value > math.log(sys.float_info.max):
        return score(status="positive_infinity", reason="exceeds float64 reporting range")
    value = math.exp(log_value)
    if value == 0:
        return score(status="underflow", reason="below float64 reporting range")
    return score(value)


def log_norm(values):
    peak = float(np.max(np.abs(values)))
    if peak == 0:
        return None
    # Scaling avoids overflow from squaring finite, large values.
    return math.log(peak) + 0.5 * math.log(float(np.sum((values / peak) ** 2)))


def numeric_metrics(reference, candidate, data_range):
    """Global metrics in float64; PSNR aggregates MSE before taking the log."""
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or not a.size:
        raise ValueError("paired arrays must have identical, nonempty shapes")
    if a.dtype.kind not in "biuf" or b.dtype.kind not in "biuf":
        raise ValueError("only real numeric arrays are supported; no object/complex dtype")
    for array in (a, b):
        if array.dtype.kind in "iu" and (array.min() < -(2**53) or array.max() > 2**53):
            raise ValueError("integer inputs must be exactly representable in float64")
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be explicit, finite and positive")
    result = {
        "status": "completed",
        "shape": list(a.shape),
        "elements": int(a.size),
        "reference_dtype": str(a.dtype),
        "candidate_dtype": str(b.dtype),
        "data_range": data_range,
        "accumulation_dtype": "float64",
        "reference_nonfinite": int(np.count_nonzero(~np.isfinite(a))),
        "candidate_nonfinite": int(np.count_nonzero(~np.isfinite(b))),
    }
    names = ("snr_db", "relative_l2", "rmse", "maximum_absolute_error", "psnr_db")
    if result["reference_nonfinite"] or result["candidate_nonfinite"]:
        result.update(status="failed", reason="nonfinite input")
        result["metrics"] = {name: score(status="invalid_nonfinite") for name in names}
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        a, b = a.astype(np.float64), b.astype(np.float64)
        error = b - a
    if not np.isfinite(error).all():
        result.update(status="failed", reason="float64 conversion or subtraction overflow")
        result["metrics"] = {name: score(status="numerical_overflow") for name in names}
        return result
    ref_log, err_log = log_norm(a), log_norm(error)
    result["exactly_equal"] = bool(np.array_equal(reference, candidate))
    result["reference_min_max"] = [float(a.min()), float(a.max())]
    result["candidate_min_max"] = [float(b.min()), float(b.max())]
    metrics = {"maximum_absolute_error": score(float(np.max(np.abs(error))))}
    if err_log is None:
        metrics["rmse"] = score(0.0)
        metrics["psnr_db"] = score(status="positive_infinity", reason="zero error")
        metrics["snr_db"] = score(
            status="undefined" if ref_log is None else "positive_infinity",
            reason="zero reference and zero error" if ref_log is None else "zero error",
        )
        metrics["relative_l2"] = (
            score(status="undefined", reason="zero reference and zero error")
            if ref_log is None
            else score(0.0)
        )
    else:
        rmse_log = err_log - 0.5 * math.log(a.size)
        metrics["rmse"] = from_log(rmse_log)
        metrics["psnr_db"] = score(20 / math.log(10) * (math.log(data_range) - rmse_log))
        metrics["snr_db"] = (
            score(status="negative_infinity", reason="zero reference, nonzero error")
            if ref_log is None
            else score(20 / math.log(10) * (ref_log - err_log))
        )
        metrics["relative_l2"] = (
            score(status="positive_infinity", reason="zero reference, nonzero error")
            if ref_log is None
            else from_log(err_log - ref_log)
        )
    result["metrics"] = metrics
    return result


def array_pairs(reference, candidate):
    """No pickle, dtype reinterpretation, broadcasting, or automatic reshaping."""
    if reference.suffix != candidate.suffix or reference.suffix not in (".npy", ".npz"):
        raise ValueError("arrays require matching .npy or .npz formats")
    if reference.suffix == ".npy":
        return {
            "array": (
                np.load(reference, allow_pickle=False),
                np.load(candidate, allow_pickle=False),
            )
        }
    with np.load(reference, allow_pickle=False) as a, np.load(candidate, allow_pickle=False) as b:
        if set(a.files) != set(b.files) or not a.files:
            raise ValueError("NPZ keys must match exactly and be nonempty")
        return {key: (a[key], b[key]) for key in sorted(a.files)}


def command(argv, out, name, processes, stdout_file=None):
    started = time.perf_counter()
    log = out / (stdout_file or f"{name}.stdout.log")
    with log.open("xb") as stdout, (out / f"{name}.stderr.log").open("xb") as stderr:
        process = subprocess.run(argv, stdout=stdout, stderr=stderr, check=False)
    processes.append(
        {
            "name": name,
            "argv": [Path(arg).name if Path(arg).is_absolute() else arg for arg in argv],
            "returncode": process.returncode,
            "wall_seconds": time.perf_counter() - started,
        }
    )
    if process.returncode:
        raise ValueError(f"{name} failed; see preserved stderr log")
    return log


def fraction(value, label):
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"missing/invalid {label}") from error


def media_contract(probe):
    streams = probe.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise ValueError("exactly one video and one audio stream are required")
    video, audio = videos[0], audios[0]
    fps = fraction(video.get("avg_frame_rate"), "average FPS")
    if fps <= 0 or fraction(video.get("r_frame_rate"), "nominal FPS") != fps:
        raise ValueError("constant, consistent positive FPS is required")
    if (audio.get("sample_rate"), audio.get("channels"), audio.get("channel_layout")) != (
        "32000",
        2,
        "stereo",
    ):
        raise ValueError("audio must already be 32000 Hz stereo; resampling is forbidden")
    for side in video.get("side_data_list", []):
        if side.get("rotation", 0) != 0:
            raise ValueError("rotated video is unsupported; no automatic rotation")
    if str(video.get("tags", {}).get("rotate", "0")) != "0":
        raise ValueError("rotated video is unsupported; no automatic rotation")
    frames = probe.get("frames", [])
    vf = [f for f in frames if f.get("media_type") == "video"]
    af = [f for f in frames if f.get("media_type") == "audio"]
    if not vf or not af:
        raise ValueError("decoded video/audio frame timelines are required")
    vbase = fraction(video.get("time_base"), "video time base")
    abase = fraction(audio.get("time_base"), "audio time base")
    vtimes = [fraction(f.get("best_effort_timestamp"), "video PTS") * vbase for f in vf]
    atimes = [fraction(f.get("best_effort_timestamp"), "audio PTS") * abase for f in af]
    samples = [int(f["nb_samples"]) for f in af]
    if any(n <= 0 for n in samples):
        raise ValueError("audio frame sample counts must be positive")
    if any(t != vtimes[0] + i / fps for i, t in enumerate(vtimes)):
        raise ValueError("video timestamps are not a contiguous constant-rate timeline")
    for frame in vf:
        duration = fraction(
            frame.get("duration", frame.get("pkt_duration")), "video frame duration"
        )
        if duration * vbase != 1 / fps:
            raise ValueError("video frame durations do not match FPS")
        if any(
            frame.get(key, 0)
            for key in (
                "crop_top",
                "crop_bottom",
                "crop_left",
                "crop_right",
                "interlaced_frame",
                "repeat_pict",
            )
        ):
            raise ValueError("cropped/interlaced/repeated-field video is unsupported")
    for frame, count in zip(af, samples, strict=True):
        duration = fraction(
            frame.get("duration", frame.get("pkt_duration")), "audio frame duration"
        )
        if duration * abase != Fraction(count, 32000):
            raise ValueError("audio frame duration differs from sample count")
        if (frame.get("channels"), frame.get("channel_layout")) != (2, "stereo"):
            raise ValueError("audio frame channel layout changed")
    offset = 0
    for timestamp, count in zip(atimes, samples, strict=True):
        if timestamp != atimes[0] + Fraction(offset, 32000):
            raise ValueError("audio timestamps contain a gap/overlap")
        offset += count
    if vtimes[0] != atimes[0]:
        raise ValueError("video/audio start times are not aligned")
    # Integer duration_ts avoids ffprobe's decimal timestamp rounding.
    vduration = fraction(video.get("duration_ts"), "video duration ticks") * vbase
    aduration = fraction(audio.get("duration_ts"), "audio duration ticks") * abase
    if vduration != len(vf) / fps or vduration != aduration:
        raise ValueError("declared video/audio durations or video frame count differ")
    width, height = int(video["width"]), int(video["height"])
    if width <= 0 or height <= 0:
        raise ValueError("invalid video dimensions")
    if any(f.get("width") != width or f.get("height") != height for f in vf):
        raise ValueError("video frame dimensions change")
    color_keys = (
        "pix_fmt",
        "color_range",
        "color_space",
        "color_transfer",
        "color_primaries",
        "chroma_location",
        "sample_aspect_ratio",
    )
    return {
        "width": width,
        "height": height,
        "fps": str(fps),
        "frames": len(vf),
        "video_pts": [str(t) for t in vtimes],
        "audio_pts": [str(t) for t in atimes],
        "audio_frame_samples": samples,
        "decoded_audio_samples": sum(samples),
        "duration_seconds": str(vduration),
        "audio_rate": 32000,
        "audio_channels": 2,
        "color_metadata": {key: video.get(key) for key in color_keys},
        "video_frame_color_metadata": [{key: frame.get(key) for key in color_keys} for frame in vf],
    }


def decode_media(ffmpeg, path, contract, out, name, processes):
    base = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-hwaccel",
        "none",
        "-noautorotate",
        "-i",
        str(path.resolve()),
    ]
    rgb = command(
        base
        + [
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        out,
        name + "-video",
        processes,
        name + ".rgb",
    )
    pcm = command(
        base + ["-map", "0:a:0", "-vn", "-sn", "-dn", "-c:a", "pcm_f32le", "-f", "f32le", "-"],
        out,
        name + "-audio",
        processes,
        name + ".pcm",
    )
    shape = (contract["frames"], contract["height"], contract["width"], 3)
    if rgb.stat().st_size != math.prod(shape):
        raise ValueError("decoded RGB byte count differs from probed frame count/geometry")
    if pcm.stat().st_size != contract["decoded_audio_samples"] * 2 * 4:
        raise ValueError("decoded PCM sample count differs from probed timeline")
    return np.memmap(rgb, dtype=np.uint8, mode="r", shape=shape), np.memmap(
        pcm, dtype="<f4", mode="r", shape=(contract["decoded_audio_samples"], 2)
    )


def ffmpeg_ssim(ffmpeg, contract, out, processes):
    argv = [ffmpeg, "-nostdin", "-v", "info"]
    for name in ("reference", "candidate"):
        argv += [
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{contract['width']}x{contract['height']}",
            "-framerate",
            contract["fps"],
            "-i",
            str(out / f"{name}.rgb"),
        ]
    # Fixed local relative name: no user-controlled filtergraph/path interpolation.
    argv += [
        "-filter_complex",
        "[0:v]format=gbrp[a];[1:v]format=gbrp[b];[a][b]ssim=stats_file=-",
        "-f",
        "null",
        "-",
    ]
    log = command(argv, out, "ssim", processes)
    values = [float(value) for value in re.findall(r"\bAll:([-+\d.eE]+)", log.read_text())]
    if len(values) != contract["frames"] or not all(math.isfinite(x) for x in values):
        raise ValueError("SSIM did not report one finite value per aligned frame")
    return {
        "status": "completed",
        "backend": "ffmpeg ssim, planar RGB gbrp",
        "mean": score(float(np.mean(values))),
        "per_frame": values,
        "note": "FFmpeg SSIM implementation; not interchangeable with other SSIM variants",
    }


def optional_backend(requested, name):
    return {
        "status": "unavailable" if requested else "not_measured",
        "reason": f"{name} adapter/checkpoint is not configured; no download or substitute metric",
    }


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("candidate", type=Path)
    cli.add_argument("--kind", choices=("arrays", "media"), required=True)
    cli.add_argument(
        "--data-range", type=float, help="required for arrays; never inferred from extrema"
    )
    cli.add_argument("--audio-data-range", type=float, default=2.0)
    cli.add_argument("--ssim", action="store_true")
    cli.add_argument("--lpips", action="store_true")
    cli.add_argument("--vbench", action="store_true")
    cli.add_argument("--output-dir", type=Path, required=True)
    return cli


def execute(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {
        "schema_version": 1,
        "status": "failed",
        "kind": args.kind,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "paired output drift; BF16 reference is not perceptual ground truth",
        "quality_decision": "not_evaluated",
        "recipe_contract": "caller must match prompt/seed/sampler and array capture location",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "harness_sha256": sha256(Path(__file__)),
        "source": source_provenance(),
        "files": {},
        "processes": [],
        "optional": {
            "lpips": optional_backend(args.lpips, "LPIPS"),
            "vbench": optional_backend(args.vbench, "VBench"),
            "fvd": {
                "status": "not_measured",
                "reason": "requires video collections; one paired clip is insufficient",
            },
            "ssim": {"status": "not_measured"},
        },
    }
    try:
        for name, path in (("reference", args.reference), ("candidate", args.candidate)):
            record["files"][name] = {
                "name": path.name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        if args.kind == "arrays":
            if args.data_range is None:
                raise ValueError("arrays require --data-range")
            record["arrays"] = {
                name: numeric_metrics(a, b, args.data_range)
                for name, (a, b) in array_pairs(args.reference, args.candidate).items()
            }
            metrics = list(record["arrays"].values())
            if args.ssim:
                record["optional"]["ssim"] = {
                    "status": "unavailable",
                    "reason": "SSIM requires media inputs",
                }
        else:
            if args.data_range is not None:
                raise ValueError(
                    "media RGB data_range is fixed at 255; use --audio-data-range for PCM"
                )
            ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
            if not ffmpeg or not ffprobe:
                raise ValueError("FFmpeg and FFprobe are required for media")
            for label, binary in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
                log = command([binary, "-version"], out, label + "-version", record["processes"])
                record["environment"][label] = {
                    "version": log.read_text().splitlines()[0],
                    "binary_sha256": sha256(binary),
                }
            contracts = []
            for name, path in (("reference", args.reference), ("candidate", args.candidate)):
                log = command(
                    [
                        ffprobe,
                        "-v",
                        "error",
                        "-show_streams",
                        "-show_frames",
                        "-of",
                        "json",
                        str(path.resolve()),
                    ],
                    out,
                    name + "-probe",
                    record["processes"],
                )
                contract = media_contract(json.loads(log.read_text()))
                contracts.append(contract)
                record["files"][name]["contract"] = contract
            if contracts[0] != contracts[1]:
                raise ValueError(
                    "paired media contracts differ: geometry/fps/timestamps/audio/color"
                )
            a, b = [
                decode_media(ffmpeg, path, contracts[0], out, name, record["processes"])
                for name, path in (("reference", args.reference), ("candidate", args.candidate))
            ]
            record["video_rgb"] = numeric_metrics(a[0], b[0], 255.0)
            record["audio_pcm32k_stereo"] = numeric_metrics(a[1], b[1], args.audio_data_range)
            record["audio_channels"] = {
                label: numeric_metrics(a[1][:, i], b[1][:, i], args.audio_data_range)
                for i, label in enumerate(("left", "right"))
            }
            metrics = [record["video_rgb"], record["audio_pcm32k_stereo"]]
            if args.ssim:
                try:
                    record["optional"]["ssim"] = ffmpeg_ssim(
                        ffmpeg, contracts[0], out, record["processes"]
                    )
                except Exception as error:
                    record["optional"]["ssim"] = {"status": "failed", "reason": str(error)}
                    raise
        if any(m["status"] != "completed" for m in metrics):
            raise ValueError("numerical evaluation rejected nonfinite/overflowing inputs")
        record["status"] = (
            "partial"
            if any(m["status"] == "unavailable" for m in record["optional"].values())
            else "completed"
        )
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    for name, path in (("reference", args.reference), ("candidate", args.candidate)):
        if name in record["files"]:
            try:
                if sha256(path) != record["files"][name]["sha256"]:
                    record.update(status="failed", error=f"{name} input changed during evaluation")
            except OSError:
                record.update(status="failed", error=f"{name} input became inaccessible")
    record["artifacts"] = {
        p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(out.iterdir())
        if p.is_file()
    }
    record["evaluation_wall_seconds"] = time.perf_counter() - started
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out / "evaluation.json", record)
    return record


def main():
    record = execute(parser().parse_args())
    print(json.dumps({"status": record["status"], "quality_decision": "not_evaluated"}))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
