"""Publish explicitly selected kernel-integration JSON and exact-five-second MP4s.

This is an integrity/identity exporter, not another numerical evaluator. It
never reads weights, latent arrays or executables, and never decodes media or
creates a GPU context. Raw JSON is copied byte-for-byte after privacy checks.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re

import publish_samples as public
import run as base


SCHEMA = "h3_kernel_publication_v1"
RUN_SCHEMA = "h3_kernel_integration_run_v1"
EVAL_SCHEMA = "h3_kernel_integration_evaluation_v1"
RESIDENT_SCHEMA = "h3_kernel_resident_comparison_v1"
DECISION_SCHEMA = "h3_kernel_default_decision_v1"
TERMINAL = {
    "completed",
    "completed_with_explicit_unsupported",
    "failed",
    "incorrect",
    "unsupported",
    "interrupted",
}
POLICIES = {"baseline", "qkv-only", "adaln-h-only", "combined"}


def equal(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError("publication binding mismatch: " + label)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def required_digest(value, label):
    if not digest(value):
        raise ValueError("required SHA256 binding is missing or invalid: " + label)
    return value


def identifier(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value)
        or value in (".", "..")
    ):
        raise ValueError("publication identifiers must be plain basenames")
    return value


def private_check(value):
    public.public_text_check(value)
    text = json.dumps(value, ensure_ascii=False)
    if re.search(r"(?i)(registry_id|serial_number|thread_id|api_key|access_token)", text):
        raise ValueError("raw JSON contains a private identifier or credential field")
    if re.search(r"(?i)[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", text):
        raise ValueError("raw JSON contains an unreviewed unique identifier")


def read_public_bytes(path, require_object=True):
    data = path.read_bytes()
    value = json.loads(data, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    if require_object and not isinstance(value, dict):
        raise ValueError("publication JSON must contain an object")
    private_check(value)
    return value, hashlib.sha256(data).hexdigest(), len(data)


class Package:
    def __init__(self, root, include_source_snapshots=False):
        self.root = Path(root).resolve()
        self.files = {}
        self.include_source_snapshots = include_source_snapshots

    def source(self, path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("publication source must be an existing regular file")
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("publication source must stay within the repository")
        relative = resolved.relative_to(self.root).as_posix()
        private_check(relative)
        return resolved, relative

    def add(self, path, target, role, expected_sha256=None, media=False, source_text=False):
        source, relative = self.source(path)
        destination = Path(target)
        if destination.is_absolute() or ".." in destination.parts or "." in destination.parts:
            raise ValueError("unsafe publication destination")
        if media:
            required_digest(expected_sha256, "selected video")
            if destination.suffix != ".mp4" or source.name != "video-5s.mp4":
                raise ValueError("only the declared exact-five-second MP4 may be copied")
            value, sha, size = None, base.sha256(source), source.stat().st_size
        elif source_text:
            if source.name != "UPSTREAM-LICENSE" and source.suffix not in (
                ".py",
                ".c",
                ".h",
                ".patch",
                ".json",
            ):
                raise ValueError("source snapshot type is not on the publication whitelist")
            if not digest(expected_sha256):
                raise ValueError("source snapshots require a recorded expected SHA256")
            if source.suffix == ".json":
                value, sha, size = read_public_bytes(source, require_object=False)
            else:
                data = source.read_bytes()
                public.public_text_check(data.decode("utf-8"))
                value, sha, size = None, hashlib.sha256(data).hexdigest(), len(data)
        else:
            if destination.suffix != ".json" or source.suffix != ".json":
                raise ValueError("only explicitly listed JSON reports may be copied")
            value, sha, size = read_public_bytes(source)
        if expected_sha256 is not None and (not digest(expected_sha256) or sha != expected_sha256):
            raise ValueError("source checksum differs from its evidence binding")
        name = destination.as_posix()
        entry = dict(
            source=relative,
            destination=name,
            sha256=sha,
            bytes=size,
            role=role,
            copy="byte_identical",
        )
        if name in self.files and self.files[name] != entry:
            raise ValueError("two different sources collide at a publication destination")
        self.files[name] = entry
        return value, entry

    def copy(self, output):
        output = Path(output)
        if output.exists() or output.is_symlink():
            raise FileExistsError("publication destination already exists")
        if not output.resolve().is_relative_to(self.root):
            raise ValueError("publication destination must stay within the repository")
        output.mkdir(parents=True, exist_ok=False)
        for entry in self.files.values():
            source = self.root / entry["source"]
            target = output / entry["destination"]
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation plus the pre-read expected SHA detects both
            # replacement before copy and mutation during streaming copy.
            with source.open("rb") as src, target.open("xb") as dst:
                while block := src.read(1024 * 1024):
                    dst.write(block)
            if target.stat().st_size != entry["bytes"] or base.sha256(target) != entry["sha256"]:
                raise ValueError("copy differs from verified source; partial destination retained")
            if base.sha256(source) != entry["sha256"]:
                raise ValueError("source changed during publication; partial destination retained")


def terminal(record, schema):
    equal(record.get("schema"), schema, "record schema")
    if record.get("status") not in TERMINAL:
        raise ValueError("only explicit terminal evidence may be published")


def collect_campaign(package, directory, kind, pin, pin_sha, label):
    directory = Path(directory)
    target = "raw/campaigns/" + label
    campaign, entry = package.add(
        directory / "campaign.json", target + "/campaign.json", kind + "_campaign"
    )
    terminal(campaign, f"h3_{kind}_campaign_v1")
    equal(campaign.get("source_manifest_sha256"), pin_sha, "campaign source pin")
    equal(campaign.get("source_manifest"), pin, "campaign source manifest")
    equal(
        campaign.get("upstream", {}).get("source_sha256"),
        pin["source_sha256"],
        "campaign native source",
    )
    request, _ = package.add(
        directory / "request.json", target + "/request.json", kind + "_request"
    )
    for key in ("source_manifest_sha256", "source_manifest", "harness_files", "environment"):
        equal(request.get(key), campaign.get(key), "campaign request " + key)
    package.add(
        directory / "sources/source-manifest.json",
        target + "/source-manifest.json",
        "source_manifest",
        pin_sha,
    )
    if package.include_source_snapshots:
        for name, sha in campaign.get("harness_files", {}).items():
            identifier(name)
            package.add(
                directory / "sources" / name,
                target + "/sources/" + name,
                "producer_source_snapshot",
                sha,
                source_text=True,
            )
        for patch in pin.get("patches", []):
            name = identifier(patch.get("file"))
            if not name.endswith(".patch"):
                raise ValueError("source pin patch must have .patch suffix")
            package.add(
                directory / "sources" / name,
                target + "/sources/" + name,
                "pinned_native_patch",
                patch.get("sha256"),
                source_text=True,
            )
    seen, names = [], set()
    for embedded in campaign.get("cases", []):
        name = identifier(embedded.get("directory"))
        if name in names:
            raise ValueError("duplicate campaign case")
        names.add(name)
        case, _ = package.add(
            directory / name / "case.json", target + "/" + name + "/case.json", kind + "_case"
        )
        equal(case, {k: v for k, v in embedded.items() if k != "directory"}, "embedded raw case")
        if case.get("status") not in TERMINAL:
            raise ValueError("campaign contains a nonterminal case")
        driver_sha = case.get("driver_sha256")
        if driver_sha is not None:
            report, _ = package.add(
                directory / name / "driver.json",
                target + "/" + name + "/driver.json",
                kind + "_driver",
                driver_sha,
            )
            equal(report.get("seed"), case.get("seed"), "driver seed")
            if kind == "prep":
                equal(report.get("schema"), "h3_prep_validation_v1", "prep driver schema")
                for key in ("rows", "gate", "distinct_maps"):
                    equal(report.get(key), case.get(key), "driver " + key)
            else:
                equal(report.get("schema"), "h3_native_qkv_v1", "QKV driver schema")
                if report.get("status") == "unsupported":
                    equal(report.get("rows"), case.get("rows"), "unsupported QKV rows")
                    if not isinstance(report.get("reason"), str) or not report["reason"]:
                        raise ValueError("unsupported QKV report requires its original reason")
                    # This early native report has no operand-policy field.
                    # Requested policy is only established by case/request JSON;
                    # do not invent GPU policy evidence for an unsupported call.
                else:
                    equal(report.get("shape", {}).get("rows"), case.get("rows"), "QKV shape rows")
                    equal(
                        report.get("operand_policy"),
                        case.get("policy", "") + "-int8",
                        "QKV operand policy",
                    )
        elif case.get("status") in ("completed", "unsupported"):
            raise ValueError("successful/unsupported case lacks immutable driver evidence")
        seen.append(
            [case["seed"], case["rows"], [case["gate"], case["distinct_maps"]]]
            if kind == "prep"
            else [case["seed"], case["rows"], case["policy"]]
        )
    equal(seen, request.get("cases_requested"), "complete campaign case list")
    if not seen:
        raise ValueError("empty operator campaign")
    return dict(
        sha256=entry["sha256"],
        kind=kind,
        status=campaign["status"],
        cases=len(seen),
        file=entry["destination"],
    )


def collect_run(package, directory, pin, pin_sha):
    directory = Path(directory)
    label = identifier(directory.name)
    target = "raw/runs/" + label
    run, entry = package.add(directory / "run.json", target + "/run.json", "run_metadata")
    if run.get("status") != "completed":
        raise ValueError("referenced model runs must be completed")
    if type(run.get("seed")) is not int or run["seed"] not in (42, 43):
        raise ValueError("kernel integration publication expects seeds42/43")
    for key, value in dict(
        width=512, height=512, seconds=5, steps=20, active_dit_blocks=50
    ).items():
        equal(run.get(key), value, "run recipe " + key)
    if run.get("schema") == RUN_SCHEMA:
        if run.get("policy") not in POLICIES or run.get("kind") not in ("capture", "resident"):
            raise ValueError("unknown native integration policy/run kind")
        equal(run.get("source_pin", {}).get("sha256"), pin_sha, "run source pin")
        equal(
            run.get("upstream", {}).get("source_sha256"), pin["source_sha256"], "run native source"
        )
        package.add(
            directory / "source-pin/manifest.json",
            target + "/source-manifest.json",
            "source_manifest",
            pin_sha,
        )
        if package.include_source_snapshots:
            snapshots = run.get("harness_snapshot", {})
            snapshot_manifest, _ = package.add(
                directory / "harness-snapshot/manifest.json",
                target + "/harness-snapshot/manifest.json",
                "source_snapshot_inventory",
            )
            equal(snapshot_manifest, snapshots, "run source snapshot manifest")
            for name, item in snapshots.items():
                identifier(name)
                equal(item.get("file"), "harness-snapshot/" + name, "source snapshot filename")
                package.add(
                    directory / item["file"],
                    target + "/" + item["file"],
                    "producer_source_snapshot",
                    item.get("sha256"),
                    source_text=True,
                )
            for patch in pin.get("patches", []):
                name = identifier(patch.get("file"))
                if not name.endswith(".patch"):
                    raise ValueError("source pin patch must have .patch suffix")
                package.add(
                    directory / "source-pin" / name,
                    target + "/source-pin/" + name,
                    "pinned_native_patch",
                    patch.get("sha256"),
                    source_text=True,
                )
    elif run.get("mode") not in ("bf16", "hadamard-fc2"):
        raise ValueError("unknown historical reference run")
    origin = {k: run.get("model", {}).get(k) for k in ("repository", "revision", "manifest_sha256")}
    equal(
        origin,
        dict(
            repository="MiniMaxAI/MiniMax-H3",
            revision=base.MODEL_REVISION,
            manifest_sha256=base.MODEL_MANIFEST_SHA256,
        ),
        "original model identity",
    )
    validation = run.get("latent_validation")
    if validation:
        meta, _ = package.add(
            directory / "latents/latent-metadata.json",
            target + "/latent-metadata.json",
            "latent_metadata_only",
            required_digest(validation.get("metadata_sha256"), "latent metadata"),
        )
        equal(meta, validation.get("metadata"), "latent metadata binding")
    adaln = run.get("adaln_validation", {})
    for name, declared in adaln.get("files", {}).items():
        identifier(name)
        if name.endswith(".json"):
            package.add(
                directory / "adaln" / name,
                target + "/adaln-metadata/" + name,
                "activation_metadata_only",
                required_digest(declared.get("sha256"), "activation metadata"),
            )
    return entry["sha256"], dict(record=run, directory=directory, entry=entry)


def bound_run(runs, digest_value, seed=None):
    if not digest(digest_value) or digest_value not in runs:
        raise ValueError("evidence refers to a run not supplied to the publication")
    value = runs[digest_value]
    if seed is not None:
        equal(value["record"]["seed"], seed, "evidence seed")
    return value


def collect_evaluation(package, path, pin_sha, runs):
    path = Path(path)
    value, entry = package.add(
        path,
        "raw/evaluations/" + identifier(path.parent.name) + "/kernel-evaluation.json",
        "two_seed_evaluation",
    )
    terminal(value, EVAL_SCHEMA)
    equal(value.get("source_pin_sha256"), pin_sha, "evaluation source pin")
    seed = value.get("seed")
    if type(seed) is not int or seed not in (42, 43):
        raise ValueError("evaluation seed must be42/43")
    sources = value.get("sources", {})
    for source in sources.values():
        bound_run(runs, source.get("run_sha256"), seed)
    for label, comparison in value.get("comparisons", {}).items():
        identifier(label)
        media = comparison.get("media")
        if not media:
            continue
        raw, _ = package.add(
            path.parent / (label + "-media") / "evaluation.json",
            str(Path(entry["destination"]).parent / (label + "-media.json")),
            "decoded_media_evaluation",
            required_digest(media.get("evaluation_sha256"), "media evaluation"),
        )
        for key in ("video_rgb", "audio_pcm32k_stereo"):
            equal(raw.get(key), media.get(key), "joined media metrics")
        equal(raw.get("optional", {}).get("ssim"), media.get("ssim"), "joined SSIM")
        candidate_label = "old_hadamard" if label == "old-hadamard" else label
        for role, source_label in (("reference", "bf16"), ("candidate", candidate_label)):
            run = bound_run(runs, sources[source_label]["run_sha256"], seed)["record"]
            equal(
                raw.get("files", {}).get(role, {}).get("sha256"),
                run.get("exact_video", {}).get("sha256"),
                "media to run video SHA",
            )
    return entry["sha256"], value


def collect_resident(package, path, pin_sha, runs):
    path = Path(path)
    value, entry = package.add(
        path, "raw/resident/" + identifier(path.parent.name) + ".json", "resident_comparison"
    )
    terminal(value, RESIDENT_SCHEMA)
    equal(value.get("source_pin_sha256"), pin_sha, "resident source pin")
    for arm in value.get("arms", []):
        run = bound_run(runs, arm.get("run_sha256"), arm.get("seed"))["record"]
        equal(run.get("kind"), "resident", "resident arm kind")
        equal(run.get("policy"), arm.get("policy"), "resident arm policy")
    return entry["sha256"], value


def collect_analysis(package, path, runs):
    path = Path(path)
    value, entry = package.add(
        path,
        "raw/analyses/" + identifier(path.parent.name) + "-" + identifier(path.name),
        "activation_analysis",
    )
    equal(value.get("schema"), 1, "activation analysis schema")
    run = bound_run(runs, value.get("source_run_sha256"))["record"]
    recorded = run.get("adaln_validation", {})
    equal(
        value.get("capture", {}).get("files"),
        recorded.get("files"),
        "analysis activation file inventory",
    )
    equal(value.get("capture", {}).get("cases"), recorded.get("cases"), "analysis activation cases")
    if not recorded:
        raise ValueError("analysis run lacks actual activation capture")
    return entry["sha256"], value


def collect_decision(package, path, pin_sha, runs, evaluations, analyses, campaigns, residents):
    path = Path(path)
    value, entry = package.add(
        path, "raw/decisions/" + identifier(path.parent.name) + ".json", "default_decision"
    )
    terminal(value, DECISION_SCHEMA)
    if value.get("decision") not in ("eligible_for_default", "keep_experimental"):
        raise ValueError("unknown explicit default-decision state")
    for item in value.get("quality_evidence", []):
        if item.get("evaluation_sha256") not in evaluations:
            raise ValueError("default decision refers to an unpublished evaluation")
        bound_run(runs, item.get("candidate_run_sha256"), item.get("seed"))
        equal(item.get("source_pin_sha256"), pin_sha, "decision quality pin")
    for item in value.get("actual_activation_evidence", []):
        if item.get("analysis_sha256") not in analyses:
            raise ValueError("default decision refers to an unpublished activation analysis")
        bound_run(runs, item.get("source_run_sha256"), item.get("seed"))
    for items in value.get("implementation_evidence", {}).values():
        for item in items:
            if item.get("campaign_sha256") not in campaigns:
                raise ValueError("default decision refers to an unpublished operator campaign")
    performance = value.get("performance", {})
    for arm in performance.get("arms", []):
        bound_run(runs, arm.get("run_sha256"), arm.get("seed"))
        if not any(arm in resident.get("arms", []) for resident in residents.values()):
            raise ValueError("decision timing arm differs from published resident evidence")
    for label, result in performance.get("comparisons", {}).items():
        if not any(
            resident.get("comparisons", {}).get(label) == result for resident in residents.values()
        ):
            raise ValueError("decision timing differs from published resident comparison")
    return dict(
        file=entry["destination"],
        sha256=entry["sha256"],
        status=value["status"],
        decision=value.get("decision"),
        eligible_for_default=value.get("eligible_for_default"),
    )


def execute(args):
    output = args.output_dir
    if output.exists() or output.is_symlink():
        raise FileExistsError("publication destination already exists")
    if any(
        not values
        for values in (
            args.resident_comparison,
            args.decision,
            args.analysis,
            args.run_dir,
            args.video_run_dir,
        )
    ):
        raise ValueError(
            "resident, decision, activation, run and selected video evidence are required"
        )
    if len(args.prep_campaign) != 2 or len(args.qkv_campaign) != 2 or len(args.evaluation) != 2:
        raise ValueError("two prep campaigns, two QKV campaigns and two seed evaluations required")
    for values in (
        args.prep_campaign,
        args.qkv_campaign,
        args.evaluation,
        args.run_dir,
        args.analysis,
        args.resident_comparison,
        args.decision,
        args.video_run_dir,
    ):
        if len({Path(p).resolve() for p in values}) != len(values):
            raise ValueError("duplicate input is not independent evidence")
    package = Package(base.ROOT, args.include_source_snapshots)
    pin, _ = package.add(
        args.source_manifest,
        "raw/source-manifest.json",
        "source_manifest",
        required_digest(args.source_manifest_sha256, "native source manifest"),
    )
    equal(pin.get("schema"), "h3_kernel_integration_source_v1", "source manifest schema")
    equal(pin.get("status"), "frozen", "frozen native source")
    if not digest(pin.get("source_sha256")):
        raise ValueError("source manifest has no native source digest")
    campaigns = {}
    for kind, directories in (("prep", args.prep_campaign), ("qkv", args.qkv_campaign)):
        for number, directory in enumerate(directories, 1):
            item = collect_campaign(
                package, directory, kind, pin, args.source_manifest_sha256, f"{kind}-r{number}"
            )
            if item["sha256"] in campaigns:
                raise ValueError("identical campaign bytes cannot represent a rerun")
            campaigns[item["sha256"]] = item
    runs = {}
    for directory in args.run_dir:
        sha, item = collect_run(package, directory, pin, args.source_manifest_sha256)
        if sha in runs:
            raise ValueError("duplicate raw run identity")
        runs[sha] = item
    evaluations = dict(
        collect_evaluation(package, path, args.source_manifest_sha256, runs)
        for path in args.evaluation
    )
    if len(evaluations) != 2 or {v["seed"] for v in evaluations.values()} != {42, 43}:
        raise ValueError("both distinct evaluation seeds42/43 required")
    residents = dict(
        collect_resident(package, path, args.source_manifest_sha256, runs)
        for path in args.resident_comparison
    )
    analyses = dict(collect_analysis(package, path, runs) for path in args.analysis)
    decisions = [
        collect_decision(
            package,
            path,
            args.source_manifest_sha256,
            runs,
            evaluations,
            analyses,
            campaigns,
            residents,
        )
        for path in args.decision
    ]
    videos = []
    for directory in args.video_run_dir:
        path = Path(directory) / "run.json"
        sha = base.sha256(path)
        run = bound_run(runs, sha)["record"]
        if run.get("schema") != RUN_SCHEMA or run.get("kind") != "capture":
            raise ValueError("selected videos must come from native integration capture runs")
        artifact = run.get("exact_video", {})
        for key, expected in dict(
            file="video-5s.mp4", width=512, height=512, frames=120, fps=24, duration_seconds=5.0
        ).items():
            if artifact.get(key) != expected:
                raise ValueError("selected video is not the verified exact-five-second artifact")
        _, entry = package.add(
            Path(directory) / artifact["file"],
            "videos/" + identifier(Path(directory).name) + ".mp4",
            "exact_5s_video",
            artifact.get("sha256"),
            media=True,
        )
        videos.append(
            dict(
                file=entry["destination"],
                sha256=entry["sha256"],
                bytes=entry["bytes"],
                source_run_sha256=sha,
                seed=run["seed"],
                policy=run.get("policy", run.get("mode")),
                frames=120,
                fps=24,
                seconds=5,
            )
        )
    expected_videos = {
        (seed, policy) for seed in (42, 43) for policy in ("qkv-only", "adaln-h-only", "combined")
    }
    if len(videos) != 6 or {(v["seed"], v["policy"]) for v in videos} != expected_videos:
        raise ValueError("all six candidate policy/seed videos are required exactly once")
    manifest = dict(
        schema=SCHEMA,
        status="completed",
        publisher_sha256=base.sha256(__file__),
        publisher_dependencies={
            "publish_samples.py": base.sha256(Path(public.__file__)),
            "run.py": base.sha256(Path(base.__file__)),
        },
        source_pin_sha256=args.source_manifest_sha256,
        source_sha256=pin["source_sha256"],
        files=sorted(package.files.values(), key=lambda item: item["destination"]),
        campaigns=list(campaigns.values()),
        decisions=decisions,
        videos=videos,
        publication_scope="Byte-identical explicitly selected JSON/MP4 with checksum and identity joins. Publication success is not a new numerical/perceptual/default-eligibility judgment.",
        excluded=[
            "model weights",
            "latent/activation arrays",
            "executables/libraries",
            "private logs/traces",
            "reference images",
        ],
        producer_source_snapshots=(
            "Recorded-hash .py/.c/.h/.patch/.json/license whitelist included for new runs and campaigns."
            if args.include_source_snapshots
            else "Not copied. Original report harness/source/patch hashes remain intact; corresponding implementation is in integrations/h3. Executables and arrays remain excluded."
        ),
        verification_limits="No GPU, media decode, weight/latent re-read or scientific re-evaluation. Existing report statuses and limits are preserved verbatim; video identity is checked against the completed run's exact-video artifact.",
        unsupported_qkv_policy_scope="Early unsupported QKV driver reports contain rows/seed/reason, not operand policy. Requested policy comes only from the raw case/request; no successful policy execution is inferred.",
    )
    private_check(manifest)
    package.copy(output)
    documents = package_documents(manifest)
    manifest["generated_files"] = []
    for name, text in documents.items():
        path = output / name
        with path.open("x") as file:
            file.write(text)
        manifest["generated_files"].append(
            dict(
                file=name,
                sha256=base.sha256(path),
                bytes=path.stat().st_size,
                copy="generated_publication_index",
            )
        )
    with (output / "manifest.json").open("x") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    return manifest


def package_documents(manifest):
    """A small playable index; no derived posters, transcoding or claims."""
    lines = [
        "# Native QKV / AdaLN-H256 integration evidence",
        "",
        "[Measured results](results.md) · [Playable gallery](gallery.html) · [File manifest and checksums](manifest.json)",
        "",
        "This package preserves the selected operator campaigns, model evaluations, resident comparisons, default decisions, activation analyses and run metadata as byte-identical files. Each manifest entry records its repository-relative source, public destination, byte count and SHA256. Source report statuses, unsupported cases and numerical limits are unchanged.",
        "",
        "Publication verifies file identity and report-to-run joins. It does not rerun numerical evaluation, decode media, judge perceptual quality, or change defaults. Exact-five-second videos are hash-bound to the completed runs' validated artifacts. Weights, activation/latent arrays, executables and private logs remain excluded.",
        "",
        "## Selected videos",
        "",
        "| Policy | Seed | Exact 5 s video | Source run SHA256 |",
        "|---|---:|---|---|",
    ]
    cards = []
    for video in manifest["videos"]:
        label = f"{video['policy']} · seed {video['seed']}"
        lines.append(
            f"| {video['policy']} | {video['seed']} | [Play]({video['file']}) | `{video['source_run_sha256']}` |"
        )
        cards.append(
            "<article><h2>"
            + html.escape(label)
            + '</h2><video controls preload="metadata" playsinline width="512" height="512">'
            + '<source src="'
            + html.escape(video["file"], quote=True)
            + '" type="video/mp4"></video>'
            + '<p><a href="'
            + html.escape(video["file"], quote=True)
            + '">Original published MP4 bytes</a> · 120 frames / 24 fps</p></article>'
        )
    lines += ["", "## Default-decision reports", ""]
    for decision in manifest["decisions"]:
        lines.append(
            f"- [{decision['status']}: {decision['decision']}]({decision['file']}) — the original report controls its scope and eligibility."
        )
    lines += ["", manifest["producer_source_snapshots"], ""]
    gallery = (
        """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Native H3 kernel integration samples</title>
<style>body{font:16px system-ui,sans-serif;max-width:1120px;margin:2rem auto;padding:0 1rem;background:#10151c;color:#edf1f7}a{color:#9ecbff}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:1.5rem}article{background:#1c2530;padding:1rem;border-radius:12px}video{width:100%;height:auto}h2{font-size:1.1rem}p{line-height:1.5}</style>
<h1>Native H3 kernel integration samples</h1>
<p>Exact 5-second, 512 × 512 samples at 24 fps. These videos are linked to their completed run records by SHA256. Quantitative drift and default decisions are documented in the original reports; publication itself is not a perceptual quality verdict.</p>
<p><a href="results.md">Measured results</a> · <a href="README.md">Evidence index</a> · <a href="manifest.json">File manifest and checksums</a></p>
<main>"""
        + "\n".join(cards)
        + "</main></html>\n"
    )
    return {"README.md": "\n".join(lines), "gallery.html": gallery}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--include-source-snapshots", action="store_true")
    for name in (
        "prep-campaign",
        "qkv-campaign",
        "evaluation",
        "resident-comparison",
        "decision",
        "analysis",
        "run-dir",
        "video-run-dir",
    ):
        p.add_argument("--" + name, type=Path, action="append", required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=base.ROOT / "results/published/h3-native-eval/kernel-integration-v1",
    )
    return p


if __name__ == "__main__":
    result = execute(parser().parse_args())
    print(
        json.dumps(
            dict(status=result["status"], files=len(result["files"]), videos=len(result["videos"]))
        )
    )
