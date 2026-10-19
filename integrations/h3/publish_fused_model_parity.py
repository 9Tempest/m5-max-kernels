"""Verify and publish on-load H256 fusion diagnostics; never launches GPU work."""

import argparse
import json
from pathlib import Path

import publish_fused_campaign as public
import run as base
import run_fc2_experiment as capture
import run_fused_campaign as fused
import run_fused_experiment as diagnostic


RECIPE = dict(
    benchmark="fox-winter-t2va-v1",
    prompt=base.PROMPT,
    width=512,
    height=512,
    seconds=5,
    steps=20,
    active_dit_blocks=50,
    reuse=1,
    core_reuse=1,
    reference_rope=True,
)
MARKERS = {
    "H3_HADAMARD_FC2": "h3 precision: FC2 plain normalized H256",
    "H3_FUSE_FC2_GATE": "h3 fusion: FC2 GEMM gate/residual epilogue",
    "H3_FUSE_ATTN_OUT_GATE": "h3 fusion: attention output GEMM gate/residual epilogue",
}


def runtime_identity(record):
    return {
        key: record["upstream"][key]
        for key in ("revision", "source_sha256", "binary_sha256", "library_sha256", "dirty")
        if key in record["upstream"]
    }


def validate_record(record, mode, seed):
    expected = RECIPE | dict(
        status="completed", mode="hadamard-fc2", seed=seed, capture_latents=True, diagnostic=True
    )
    if any(
        type(record.get(key)) is not type(value) or record[key] != value
        for key, value in expected.items()
    ):
        raise ValueError("completed same-seed fixed diagnostic recipe required")
    if record.get("prepared_checkpoint") or any(
        key.startswith("H3_PREPARED_") for key in record.get("runtime_overrides", {})
    ):
        raise ValueError(
            "this parity contract requires on-load H256, not prepared-checkpoint loading"
        )
    if record.get("fusion", "none") != mode:
        raise ValueError("fusion flag differs from requested parity run")
    source = capture if mode == "none" else fused
    if record.get("patch") != dict(
        source_sha256=source.SOURCE_SHA256, patch_sha256=source.PATCH_SHA256
    ):
        raise ValueError("fusion/reference source patch identity differs")
    if (
        record.get("upstream", {}).get("revision") != base.UPSTREAM_REVISION
        or record["upstream"].get("source_sha256") != source.SOURCE_SHA256
    ):
        raise ValueError("fusion/reference upstream identity differs")
    model = record.get("model", {})
    if {key: model.get(key) for key in ("repository", "revision", "manifest_sha256")} != dict(
        repository="MiniMaxAI/MiniMax-H3",
        revision=base.MODEL_REVISION,
        manifest_sha256=base.MODEL_MANIFEST_SHA256,
    ):
        raise ValueError("fusion parity requires the original pinned model identity")
    process = record.get("measurements", {}).get("generation", {})
    if process.get("returncode") != 0 or process.get("fallback_detected") is not False:
        raise ValueError("fusion generation failed or used fallback")
    if mode != "none" and record.get("video_bytes_equal_reference") is not True:
        raise base.NumericalError(
            "completed runner record does not assert exact exported video parity"
        )


def verify_run(directory, mode, seed):
    path = directory / "run.json"
    record = public.public.load_json(path)
    validate_record(record, mode, seed)
    log_path = directory / "generation.log"
    if base.sha256(log_path) != record["measurements"]["generation"]["log_sha256"]:
        raise ValueError("generation log differs from immutable run record")
    log = log_path.read_text(errors="replace")
    expected_flags = dict(
        H3_HADAMARD_FC2="1",
        H3_FUSE_FC2_GATE=str(int(mode in ("fc2", "both"))),
        H3_FUSE_ATTN_OUT_GATE=str(int(mode in ("attention", "both"))),
    )
    for key, value in expected_flags.items():
        if log.count(MARKERS[key]) != int(value):
            raise ValueError("actual runtime marker differs from requested fusion policy")
        if key in record["runtime_overrides"] or mode != "none" or key == "H3_HADAMARD_FC2":
            if record["runtime_overrides"].get(key) != value:
                raise ValueError("recorded runtime override differs from fusion policy")
    latents = capture.validate_latent_capture(directory / "latents", "hadamard-fc2", seed)
    if latents != record.get("latent_validation"):
        raise ValueError("latent tensor bytes/finite checks differ from completed record")
    metadata = latents["metadata"]
    for key, value in dict(
        dtype="float32_little_endian",
        scope="diagnostic_CPU_readback_after_completed_GPU_Euler",
        final_video_sigma=0,
        final_audio_sigma=0,
    ).items():
        if metadata.get(key) != value:
            raise ValueError("latent boundary/dtype differs from parity contract")
    media = {}
    for key, filename, frames in (
        ("original_video", "original.mp4", 124),
        ("exact_video", "video-5s.mp4", 120),
    ):
        entry = record[key]
        if any(
            entry.get(k) != v
            for k, v in dict(file=filename, width=512, height=512, frames=frames, fps=24).items()
        ):
            raise ValueError("media geometry differs from fixed parity recipe")
        digest = base.sha256(directory / filename)
        if digest != entry.get("sha256"):
            raise ValueError("video bytes differ from completed run record")
        media[key] = dict(file=filename, sha256=digest, bytes=(directory / filename).stat().st_size)
    return record, latents, media


def verify_pair(
    reference,
    reference_latents,
    reference_media,
    candidate,
    candidate_latents,
    candidate_media,
    reference_digest,
):
    if candidate.get("reference") != dict(
        run_sha256=reference_digest, source_sha256=capture.SOURCE_SHA256, seed=reference["seed"]
    ):
        raise ValueError("candidate reference digest/source/seed binding differs")
    parity = diagnostic.latent_parity(reference_latents, candidate_latents)
    if parity != candidate.get("latent_parity") or parity["passed"] is not True:
        raise base.NumericalError("all 42 latent tensors must have exact finite parity")
    media = [
        dict(
            kind=key,
            reference_sha256=reference_media[key]["sha256"],
            candidate_sha256=candidate_media[key]["sha256"],
            byte_equal=reference_media[key]["sha256"] == candidate_media[key]["sha256"],
        )
        for key in reference_media
    ]
    if candidate.get("video_bytes_equal_reference") is not True or not all(
        item["byte_equal"] for item in media
    ):
        raise base.NumericalError("both original and exact-export video bytes must match reference")
    return dict(latents=parity, media=media, passed=True)


def execute(reference_dir, attention_dir, both_dir, output):
    source_digests = {
        directory: base.sha256(directory / "run.json")
        for directory in (reference_dir, attention_dir, both_dir)
    }
    reference_path = reference_dir / "run.json"
    seed = public.public.load_json(reference_path).get("seed")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("reference requires a nonnegative signed-64 seed")
    reference, ref_latents, ref_media = verify_run(reference_dir, "none", seed)
    reference_digest = source_digests[reference_dir]
    records = [("reference", reference_dir, reference)]
    comparisons = []
    for mode, directory in (("attention", attention_dir), ("both", both_dir)):
        record, latents, media = verify_run(directory, mode, seed)
        parity = verify_pair(
            reference, ref_latents, ref_media, record, latents, media, reference_digest
        )
        comparisons.append(
            dict(
                fusion=mode,
                source_run_sha256=source_digests[directory],
                runtime=runtime_identity(record),
                patch=record["patch"],
                observed_flags={key: record["runtime_overrides"][key] for key in MARKERS},
                metadata_sha256=latents["metadata_sha256"],
                **parity,
            )
        )
        records.append((mode, directory, record))
    result = dict(
        schema_version=1,
        status="completed",
        publisher_sha256=base.sha256(__file__),
        seed=seed,
        recipe=RECIPE,
        scope="on-load H256 attention/both fusion versus unfused on-load H256; exact diagnostic trajectory/media parity, not performance or quality",
        original_model={
            key: reference["model"][key] for key in ("repository", "revision", "manifest_sha256")
        },
        reference=dict(
            source_run_sha256=reference_digest,
            runtime=runtime_identity(reference),
            patch=reference["patch"],
            metadata_sha256=ref_latents["metadata_sha256"],
        ),
        comparisons=comparisons,
        source_records=[],
        prepared_checkpoint_scope="not this executable path; prepared import replay has separate evidence and does not establish prepared-plus-fusion parity",
        performance="not measured; captures add readback/synchronization",
        default_enablement=False,
    )
    output.mkdir(parents=True, exist_ok=False)
    for label, directory, record in records:
        path = directory / "run.json"
        if base.sha256(path) != source_digests[directory]:
            raise ValueError("run record changed during verification")
        artifact = dict(role=label, source_sha256=source_digests[directory])
        try:
            public.read_public(path)
        except ValueError:
            artifact.update(
                export="sanitized_summary_only",
                reason="original run JSON failed publication privacy check; no raw copy claimed",
            )
        else:
            target = output / "raw" / f"{label}-run.json"
            target.parent.mkdir(exist_ok=True)
            public.public.copy_exclusive(path, target)
            if base.sha256(target) != artifact["source_sha256"]:
                raise ValueError("run record changed during publication")
            artifact.update(export="byte_identical_raw_json", file=f"raw/{target.name}")
        result["source_records"].append(artifact)
    public.public.public_text_check(result)
    base.write_json(output / "summary.json", result)
    lines = [
        "# Seed " + str(seed) + " whole-model fusion parity",
        "",
        "Attention-only and both-flags runs match the original on-load H256 reference exactly for every packed Euler video/audio tensor (20 steps × two modalities), both final unpacked VAE-input tensors, and both original/exact-duration MP4 files. All 42 tensors in each run were re-read, size/hash checked and scanned for finite values. The completed runner's video-parity boolean is explicitly required to be true and media hashes are independently recomputed.",
        "",
        "[Detailed checksums and comparisons](summary.json). Source/patch/model identities, recipe, reference-run binding and generation-log policy markers are verified. Raw JSON files are included only after privacy review; the summary explicitly lists whether each source was copied byte-for-byte or only represented by a sanitized summary. Latent arrays, weights and original logs are excluded.",
        "",
        "These are diagnostic five-second fox runs on on-load H256 runtime sources, with readbacks/synchronization. They are not latency samples or a perceptual quality decision. Prepared-checkpoint import uses a different runtime path and has separate byte-parity evidence; this does not validate prepared loading combined with fusion. One seed does not establish every prompt/shape/timestep configuration. Neither flag is enabled by default. Matched same-source resident performance remains a separate experiment.",
        "",
    ]
    with (output / "README.md").open("x") as handle:
        handle.write("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--both-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.reference_dir, args.attention_dir, args.both_dir, args.output_dir)
    print(json.dumps(dict(status=result["status"], seed=result["seed"])))
