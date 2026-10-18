"""Package completed H3 sample videos and paired eval JSON for repository publication.

This exports selected recipe/provenance fields and copies evaluation JSON and
MP4 bytes unchanged. It never exports weights, latent arrays, private traces,
or original run logs. Every package directory is new and immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import run as base


RECIPE_KEYS = (
    "benchmark",
    "prompt",
    "seed",
    "width",
    "height",
    "seconds",
    "steps",
    "active_dit_blocks",
    "reuse",
    "core_reuse",
    "reference_rope",
)
LABELS = {"bf16": "BF16", "upstream-int8": "Native INT8", "hadamard-fc2": "INT8 + H256 FC2"}


def load_json(path):
    text = path.read_text()
    value = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    json.dumps(value, allow_nan=False)
    return value


def public_text_check(value):
    encoded = json.dumps(value, ensure_ascii=False)
    if any(token in encoded for token in ("/Users/", "/home/", "codex://threads/")):
        raise ValueError("publication data contains a private absolute path or thread reference")


def model_identity(record, directory):
    model = record.get("model", {})
    if record.get("prepared_checkpoint") is not True:
        return {key: model[key] for key in ("repository", "revision", "manifest_sha256")}
    if record.get("mode") != "hadamard-fc2":
        raise ValueError("prepared checkpoint policy requires the H256 precision mode")
    # Bind only a run-local snapshot. Never follow the original checkpoint's
    # private filesystem location or silently omit its preparation provenance.
    declaration = record.get("derived_checkpoint", {})
    path = directory / "derived-checkpoint.json"
    if declaration.get("manifest_file") != path.name or not path.is_file():
        raise ValueError("prepared sample requires a run-local derived manifest snapshot")
    digest = base.sha256(path)
    if digest != model.get("manifest_sha256") or digest != declaration.get("manifest_sha256"):
        raise ValueError("derived manifest snapshot differs from completed run")
    manifest = load_json(path)
    if (
        manifest.get("schema") != "h3_prepared_model_v1"
        or manifest.get("policy_id") != "h3_dit_int8_native_fc2_h256_v1"
        or manifest.get("original_checkpoint_retained") is not True
    ):
        raise ValueError("prepared sample schema, policy or preservation contract differs")
    for key in ("source_model", "producer", "policy_id", "prepared_manifest_sha256"):
        if manifest.get(key) != declaration.get(key):
            raise ValueError(f"derived manifest declaration differs: {key}")
    for key in ("source_model", "prepared_manifest_sha256"):
        if manifest.get(key) != model.get(key):
            raise ValueError(f"derived manifest model binding differs: {key}")
    return {
        "manifest_sha256": digest,
        **{
            key: manifest[key]
            for key in (
                "source_model",
                "producer",
                "policy_id",
                "prepared_manifest_sha256",
                "original_checkpoint_retained",
            )
        },
    }


def sample_record(directory):
    path = directory / "run.json"
    record = load_json(path)
    if record.get("status") != "completed" or record.get("mode") not in LABELS:
        raise ValueError("only completed explicit precision runs can be published")
    recipe = {key: record[key] for key in RECIPE_KEYS}
    if any(
        type(recipe[key]) is not int or recipe[key] <= 0 for key in ("width", "height", "seconds")
    ):
        raise ValueError("sample geometry and duration must be positive integers")
    if record.get("fps", 24) != 24:
        raise ValueError("sample collection requires the native 24 fps recipe")
    identifier = f"{recipe['benchmark']}-seed{recipe['seed']}-{record['mode']}"
    if not re.fullmatch(r"[a-z0-9-]+", identifier):
        raise ValueError("unsafe or unsupported sample identifier")
    expected = record["exact_video"]
    filename = f"video-{recipe['seconds']}s.mp4"
    if expected.get("file") != filename:
        raise ValueError("exact video filename differs from the requested duration")
    geometry = dict(
        width=recipe["width"],
        height=recipe["height"],
        frames=recipe["seconds"] * 24,
        fps=24,
        duration_seconds=recipe["seconds"],
    )
    if any(expected.get(key) != value for key, value in geometry.items()):
        raise ValueError("verified exact video geometry differs from its recipe")
    video = directory / filename
    digest = base.sha256(video)
    if expected["sha256"] != digest or expected.get("file") != video.name:
        raise ValueError("sample video differs from its completed generation record")
    source = record["upstream"]
    entry = {
        "id": identifier,
        "label": LABELS[record["mode"]],
        "mode": record["mode"],
        "recipe": recipe,
        "source_run_sha256": base.sha256(path),
        "source_run_status": "completed",
        "diagnostic_capture": record.get("diagnostic", False),
        "runtime": {
            key: source[key]
            for key in ("revision", "dirty", "source_sha256", "binary_sha256", "library_sha256")
            if key in source
        },
        "runtime_patch": record.get("patch"),
        "environment": {
            key: value
            for key, value in record.get("environment", {}).items()
            if key
            in (
                "system",
                "release",
                "machine",
                "python",
                "numpy",
                "chip",
                "memory_bytes",
                "xcode",
                "ffmpeg",
                "ffprobe",
            )
        },
        "harness_revision": {
            key: value
            for key, value in record.get("harness", {}).items()
            if key in ("revision", "dirty")
        },
        "harness_files": record.get("harness_files", {}),
        "prepared_checkpoint": record.get("prepared_checkpoint", False),
        "model": model_identity(record, directory),
        "video": {
            "file": f"videos/{identifier}.mp4",
            "sha256": digest,
            "bytes": video.stat().st_size,
            "frames": recipe["seconds"] * 24,
            "fps": 24,
            "duration_seconds": recipe["seconds"],
        },
        "poster": f"posters/{identifier}.jpg",
        "quality_decision": "not_evaluated",
        "timing_note": (
            "Sample capture timing is diagnostic; see the separate performance report."
            if record.get("diagnostic", False)
            else "This sample is not a resident latency experiment; see the separate performance report."
        ),
    }
    if "conditioning" in record:
        # The reference image stays local; only its reproducibility identity is exported.
        entry["conditioning"] = {
            key: value
            for key, value in record["conditioning"].items()
            if key
            in (
                "type",
                "mode",
                "sha256",
                "image_sha256",
                "width",
                "height",
                "original_width",
                "original_height",
                "preprocessing",
                "target_canvas",
            )
        }
    public_text_check(entry)
    return entry, video


def validate_matched_samples(samples):
    first = samples[0]
    if any(
        sample["recipe"] != first["recipe"]
        or sample.get("conditioning") != first.get("conditioning")
        for sample in samples[1:]
    ):
        raise ValueError("comparison samples have different recipes or conditioning")


def evaluation_association(kind, record, samples):
    def unique(key, digest):
        matches = [
            sample
            for sample in samples
            if (sample["video"]["sha256"] if key == "media" else sample["source_run_sha256"])
            == digest
        ]
        if len(matches) != 1:
            raise ValueError(f"{key} evaluation source is missing or ambiguous")
        return matches[0]

    if kind == "media":
        if record.get("kind") != "media":
            raise ValueError("media association requires a media evaluation")
        associated = {
            role: unique("media", record["files"][role]["sha256"])
            for role in ("reference", "candidate")
        }
    else:
        sources = record.get("source", record.get("sources", {}))
        if set(sources) not in ({"bf16", "native", "hadamard"}, {"bf16", "hadamard"}):
            raise ValueError("latent evaluation must bind all expected comparison roles")
        associated = {
            role: unique("latents", value["run_sha256"]) for role, value in sources.items()
        }
        for role, sample in associated.items():
            if (
                sample["mode"]
                != {"bf16": "bf16", "native": "upstream-int8", "hadamard": "hadamard-fc2"}[role]
            ):
                raise ValueError("latent evaluation role differs from sample precision")
    validate_matched_samples(list(associated.values()))
    return {role: sample["id"] for role, sample in associated.items()}


def copy_exclusive(source, destination):
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        shutil.copyfileobj(input_file, output_file)
    if base.sha256(source) != base.sha256(destination):
        raise ValueError("publication copy checksum mismatch")


def render_readme(samples, evaluations):
    lines = [
        "# H3 generated samples and paired evaluation",
        "",
        "[Open local video gallery](gallery.html) · [Machine-readable manifest](manifest.json)",
        "",
        "Each comparison uses the same prompt, seed and conditioning. Videos are separate exact-duration exports at 24 fps; original native frame alignment is retained in the generation records. Resolution, duration, sampling recipe and source identities are recorded per sample in the manifest.",
        "",
        "BF16 is the pinned h3.c close-reference implementation, not official-pipeline or perceptual ground truth. H256 rotates only post-SwiGLU FC2 before INT8 packing; QKV, attention output and FC1 retain native INT8. Other tensors retain their original precision.",
        "",
        "| Benchmark / seed | Mode | Preview / video |",
        "|---|---|---|",
    ]
    for sample in samples:
        recipe = sample["recipe"]
        lines.append(
            f"| {recipe['benchmark']} / {recipe['seed']} | {sample['label']} | [![Preview]({sample['poster']})]({sample['video']['file']}) |"
        )
    lines += [
        "",
        "## Evaluation records",
        "",
        "All linked evaluation JSON files are copied byte-for-byte. Their checksums, sample associations, original run digests and video hashes are in the manifest. Raw latent arrays and weights are not included.",
        "",
    ]
    for entry in evaluations:
        lines.append(f"- [{entry['kind']}: {entry['file']}]({entry['file']})")
    lines += [
        "",
        "SNR/relative L2 quantify drift from BF16; RGB SSIM is FFmpeg's planar-RGB variant. These metrics do not establish human preference, prompt adherence or motion quality. FVD/LPIPS/VBench remain explicitly unmeasured. Diagnostic capture runs are not latency samples; use the separately reported warm performance experiment.",
        "",
    ]
    return "\n".join(lines)


def render_gallery(samples):
    cards = []
    for sample in samples:
        recipe = sample["recipe"]
        cards.append(
            f"<article><h2>{html.escape(recipe['benchmark'])} · seed {recipe['seed']}</h2>"
            f"<h3>{html.escape(sample['label'])}</h3>"
            f'<video controls playsinline preload="metadata" poster="{sample["poster"]}" src="{sample["video"]["file"]}"></video>'
            f"<p>{html.escape(recipe['prompt'])}</p><small>{recipe['width']} × {recipe['height']} · {recipe['seconds']} s · 24 fps · {recipe['steps']} steps</small></article>"
        )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3 kernel evaluation samples</title>
<style>body{font:16px system-ui,sans-serif;background:#10151b;color:#eaf0f5;margin:0;padding:32px;max-width:1600px}h1{margin-top:0}header p{max-width:900px;color:#bdcbd8}main{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:20px}article{background:#1b242e;padding:16px;border-radius:12px}h2{font-size:17px}h3{font-size:15px;color:#93d7e9}video{width:100%;height:auto;background:#000}article p,small{font-size:13px;color:#b8c7d7}a{color:#93d7e9}@media(max-width:900px){main{grid-template-columns:1fr}}</style>
<header><h1>H3 kernel evaluation samples</h1><p>Paired BF16 / native INT8 / INT8 + H256 FC2 outputs. Same prompt and seed within each group. Numerical similarity is not a perceptual quality score. Use each video's controls to inspect motion and audio.</p><p><a href="README.md">Evaluation notes</a> · <a href="manifest.json">Recipes and checksums</a></p></header><main>"""
        + "\n".join(cards)
        + "</main></html>\n"
    )


def execute(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    sources = []
    for directory in args.run_dir:
        entry, video = sample_record(directory)
        samples.append(entry)
        sources.append(video)
    if len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("duplicate sample IDs")
    groups = {}
    for sample in samples:
        key = sample["recipe"]["benchmark"], sample["recipe"]["seed"]
        groups.setdefault(key, []).append(sample)
    for group in groups.values():
        validate_matched_samples(group)
    evaluations = []
    eval_sources = []
    for kind, paths in (("media", args.media_evaluation), ("latents", args.latent_evaluation)):
        for path in paths:
            record = load_json(path)
            public_text_check(record)
            if record.get("status") != "completed":
                raise ValueError("only completed eval records can be published")
            association = evaluation_association(kind, record, samples)
            name = f"evaluations/{path.parent.name}-{path.name}"
            evaluations.append(
                {"kind": kind, "file": name, "sha256": base.sha256(path), "samples": association}
            )
            eval_sources.append(path)
    if len({entry["file"] for entry in evaluations}) != len(evaluations):
        raise ValueError("duplicate eval filenames")
    for folder in ("videos", "posters", "evaluations"):
        (output / folder).mkdir()
    for sample, source in zip(samples, sources, strict=True):
        copy_exclusive(source, output / sample["video"]["file"])
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-ss",
                str(min(2, sample["recipe"]["seconds"] / 2)),
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                "scale=256:-2",
                str(output / sample["poster"]),
            ],
            check=True,
        )
        sample["poster_sha256"] = base.sha256(output / sample["poster"])
    for entry, source in zip(evaluations, eval_sources, strict=True):
        copy_exclusive(source, output / entry["file"])
    (output / "README.md").write_text(render_readme(samples, evaluations))
    (output / "gallery.html").write_text(render_gallery(samples))
    manifest = {
        "schema_version": 1,
        "created_utc": base.utc_now(),
        "publisher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "generated sample videos, selected generation provenance and immutable paired evaluations",
        "samples": samples,
        "evaluations": evaluations,
    }
    base.write_json(output / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, action="append", required=True)
    p.add_argument("--media-evaluation", type=Path, action="append", default=[])
    p.add_argument("--latent-evaluation", type=Path, action="append", default=[])
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    try:
        result = execute(args)
    except Exception as error:
        print(
            f"publication failed; partial directory retained: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "samples": len(result["samples"]),
                "evaluations": len(result["evaluations"]),
                "manifest": str(args.output_dir / "manifest.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
