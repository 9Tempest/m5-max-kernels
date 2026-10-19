import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def pub(monkeypatch, tmp_path):
    directory = Path(__file__).parents[1] / "integrations/h3"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "kernel_publisher_test", directory / "publish_kernel_integration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.base, "ROOT", tmp_path)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deliberately noncanonical JSON: publication must retain these bytes.
    path.write_text(json.dumps(value, indent=3, ensure_ascii=False) + "\n\n")
    return path


def setup(tmp_path, pub):
    source = tmp_path / "integration/source.json"
    patch = b"fixture patch\n"
    import hashlib

    patch_sha = hashlib.sha256(patch).hexdigest()
    pin = dict(
        schema="h3_kernel_integration_source_v1",
        status="frozen",
        source_sha256="a" * 64,
        patches=[dict(file="native.patch", sha256=patch_sha)],
    )
    write(source, pin)
    pin_sha = pub.base.sha256(source)
    campaign_dirs = {"prep": [], "qkv": []}
    campaign_hashes = {}
    for kind in ("prep", "qkv"):
        for number in (1, 2):
            directory = tmp_path / f"local/{kind}-r{number}"
            name = "case-a"
            driver = dict(
                schema="h3_prep_validation_v1" if kind == "prep" else "h3_native_qkv_v1",
                rows=17,
                seed=17,
                **(
                    dict(gate=0, distinct_maps=0)
                    if kind == "prep"
                    else dict(status="unsupported", reason="native API requires M>=128")
                ),
            )
            write(directory / name / "driver.json", driver)
            case = dict(
                rows=17,
                seed=17,
                **(dict(gate=0, distinct_maps=0) if kind == "prep" else dict(policy="native")),
            ) | dict(
                status="completed" if kind == "prep" else "unsupported",
                driver_sha256=pub.base.sha256(directory / name / "driver.json"),
            )
            write(directory / name / "case.json", case)
            cases = [case | dict(directory=name)]
            if kind == "qkv":
                completed = dict(
                    schema="h3_native_qkv_v1",
                    status="completed",
                    seed=17,
                    operand_policy="h256-int8",
                    shape=dict(
                        rows=129,
                        padded_rows=256,
                        input_dim=5376,
                        heads=56,
                        head_dim=128,
                        rope_half=48,
                    ),
                )
                write(directory / "case-b/driver.json", completed)
                completed_case = dict(
                    rows=129,
                    seed=17,
                    policy="h256",
                    status="completed",
                    driver_sha256=pub.base.sha256(directory / "case-b/driver.json"),
                )
                write(directory / "case-b/case.json", completed_case)
                cases.append(completed_case | dict(directory="case-b"))
            header = dict(
                source_manifest_sha256=pin_sha,
                source_manifest=pin,
                harness_files={},
                environment=dict(chip="Apple M5 Max"),
            )
            requested = (
                [[17, 17, [0, 0]]] if kind == "prep" else [[17, 17, "native"], [17, 129, "h256"]]
            )
            write(
                directory / "request.json",
                header | dict(cases_requested=requested, reverse=number == 2),
            )
            write(directory / "sources/source-manifest.json", pin)
            (directory / "sources/native.patch").write_bytes(patch)
            campaign = header | dict(
                schema=f"h3_{kind}_campaign_v1",
                status="completed" if kind == "prep" else "completed_with_explicit_unsupported",
                upstream=dict(source_sha256=pin["source_sha256"]),
                cases=cases,
                repetition=number,
            )
            path = write(directory / "campaign.json", campaign)
            campaign_hashes[kind, number] = pub.base.sha256(path)
            campaign_dirs[kind].append(directory)
    runs, dirs = {}, []
    for seed in (42, 43):
        for label in (
            "bf16",
            "old-hadamard",
            "baseline",
            "qkv-only",
            "adaln-h-only",
            "combined",
            "resident-baseline",
            "resident-qkv-only",
        ):
            directory = tmp_path / f"local/{label}-{seed}"
            directory.mkdir(parents=True)
            historical = label in ("bf16", "old-hadamard")
            resident = label.startswith("resident-")
            policy = label.removeprefix("resident-")
            run = dict(
                status="completed",
                seed=seed,
                width=512,
                height=512,
                seconds=5,
                steps=20,
                active_dit_blocks=50,
                mode="bf16" if label == "bf16" else "hadamard-fc2",
                model=dict(
                    repository="MiniMaxAI/MiniMax-H3",
                    revision=pub.base.MODEL_REVISION,
                    manifest_sha256=pub.base.MODEL_MANIFEST_SHA256,
                ),
            )
            if not historical:
                run.update(
                    schema=pub.RUN_SCHEMA,
                    kind="resident" if resident else "capture",
                    policy=policy,
                    source_pin=dict(sha256=pin_sha),
                    upstream=dict(source_sha256=pin["source_sha256"]),
                )
                write(directory / "source-pin/manifest.json", pin)
                (directory / "source-pin/native.patch").write_bytes(patch)
                code = directory / "harness-snapshot/run.py"
                code.parent.mkdir()
                code.write_text("# Public fixture source\n")
                run["harness_snapshot"] = {
                    "run.py": dict(file="harness-snapshot/run.py", sha256=pub.base.sha256(code))
                }
                inventory = write(
                    directory / "harness-snapshot/model-files.json",
                    [dict(path="FL2VA/transformer/config.json", sha256="7" * 64)],
                )
                run["harness_snapshot"][inventory.name] = dict(
                    file="harness-snapshot/" + inventory.name,
                    sha256=pub.base.sha256(inventory),
                )
                write(directory / "harness-snapshot/manifest.json", run["harness_snapshot"])
            if not resident:
                meta = dict(dtype="float32_little_endian", seed=seed)
                write(directory / "latents/latent-metadata.json", meta)
                (directory / "latents/do-not-copy.f32").write_bytes(b"private latent fixture")
                run["latent_validation"] = dict(
                    metadata=meta,
                    metadata_sha256=pub.base.sha256(directory / "latents/latent-metadata.json"),
                )
                video = directory / "video-5s.mp4"
                video.write_bytes(f"MP4-fixture-{label}-{seed}".encode())
                run["exact_video"] = dict(
                    file=video.name,
                    width=512,
                    height=512,
                    frames=120,
                    fps=24,
                    duration_seconds=5.0,
                    sha256=pub.base.sha256(video),
                )
            if label == "combined":
                meta = dict(family="qkv", step=0, block=0)
                metadata = write(directory / "adaln/qkv-step00-block00.json", meta)
                run["adaln_validation"] = dict(
                    files={
                        metadata.name: dict(
                            sha256=pub.base.sha256(metadata), bytes=metadata.stat().st_size
                        )
                    },
                    cases=[meta],
                )
                (directory / "adaln/do-not-copy.bf16").write_bytes(b"private activation fixture")
            write(directory / "run.json", run)
            (directory / "do-not-copy.safetensors").write_bytes(b"private weights fixture")
            (directory / "native-cli").write_bytes(b"private executable fixture")
            runs[label, seed] = dict(
                record=run, directory=directory, sha=pub.base.sha256(directory / "run.json")
            )
            dirs.append(directory)
    evaluations = []
    for seed in (42, 43):
        directory = tmp_path / f"local/eval-{seed}"
        sources = {
            (
                "old_hadamard"
                if label == "old-hadamard"
                else label + "-fused"
                if label not in ("bf16",)
                else label
            ): dict(run_sha256=runs[label, seed]["sha"])
            for label in (
                "bf16",
                "old-hadamard",
                "baseline",
                "qkv-only",
                "adaln-h-only",
                "combined",
            )
        }
        comparisons = {}
        for label in (
            "old-hadamard",
            "baseline-fused",
            "qkv-only-fused",
            "adaln-h-only-fused",
            "combined-fused",
        ):
            run_label = label.removesuffix("-fused")
            media = dict(
                status="completed",
                video_rgb=dict(rmse=0.01),
                audio_pcm32k_stereo=dict(rmse=0.02),
                optional=dict(ssim=0.9),
                files=dict(
                    reference=dict(sha256=runs["bf16", seed]["record"]["exact_video"]["sha256"]),
                    candidate=dict(sha256=runs[run_label, seed]["record"]["exact_video"]["sha256"]),
                ),
            )
            media_path = write(directory / (label + "-media") / "evaluation.json", media)
            comparisons[label] = dict(
                media=dict(
                    evaluation_sha256=pub.base.sha256(media_path),
                    video_rgb=media["video_rgb"],
                    audio_pcm32k_stereo=media["audio_pcm32k_stereo"],
                    ssim=0.9,
                )
            )
        value = dict(
            schema=pub.EVAL_SCHEMA,
            status="completed",
            source_pin_sha256=pin_sha,
            seed=seed,
            sources=sources,
            comparisons=comparisons,
        )
        evaluations.append(write(directory / "kernel-evaluation.json", value))
    arms = [
        dict(run_sha256=runs[f"resident-{policy}", seed]["sha"], seed=seed, policy=policy)
        for seed in (42, 43)
        for policy in ("baseline", "qkv-only")
    ]
    resident = dict(
        schema=pub.RESIDENT_SCHEMA,
        status="completed",
        source_pin_sha256=pin_sha,
        arms=arms,
        comparisons={"baseline__qkv-only": dict(both_seeds_complete_request_benefit=False)},
    )
    resident_path = write(tmp_path / "local/resident/resident-comparison.json", resident)
    analyses = []
    for seed in (42, 43):
        run = runs["combined", seed]
        report = dict(
            schema=1,
            source_run_sha256=run["sha"],
            capture=run["record"]["adaln_validation"],
            all_preparation_checks_pass=True,
        )
        analyses.append(write(tmp_path / f"local/analysis-{seed}/analysis.json", report))
    decision = dict(
        schema=pub.DECISION_SCHEMA,
        status="completed",
        decision="keep_experimental",
        eligible_for_default=False,
        quality_evidence=[
            dict(
                evaluation_sha256=pub.base.sha256(path),
                candidate_run_sha256=runs["qkv-only", seed]["sha"],
                seed=seed,
                source_pin_sha256=pin_sha,
            )
            for seed, path in zip((42, 43), evaluations)
        ],
        actual_activation_evidence=[
            dict(
                analysis_sha256=pub.base.sha256(path),
                source_run_sha256=runs["combined", seed]["sha"],
                seed=seed,
            )
            for seed, path in zip((42, 43), analyses)
        ],
        implementation_evidence={
            kind: [dict(campaign_sha256=campaign_hashes[kind, number]) for number in (1, 2)]
            for kind in ("prep", "qkv")
        },
        performance=dict(arms=arms, comparisons=resident["comparisons"]),
    )
    decision_path = write(tmp_path / "local/decision/default-decision.json", decision)
    args = SimpleNamespace(
        source_manifest=source,
        source_manifest_sha256=pin_sha,
        prep_campaign=campaign_dirs["prep"],
        qkv_campaign=campaign_dirs["qkv"],
        evaluation=evaluations,
        resident_comparison=[resident_path],
        decision=[decision_path],
        analysis=analyses,
        run_dir=dirs,
        video_run_dir=[
            runs[label, seed]["directory"]
            for seed in (42, 43)
            for label in ("qkv-only", "adaln-h-only", "combined")
        ],
        include_source_snapshots=False,
        output_dir=tmp_path / "published/kernel-integration-v1",
    )
    return args, runs


def test_complete_package_preserves_raw_bytes_and_excludes_arrays_weights_binaries(tmp_path, pub):
    args, _ = setup(tmp_path, pub)
    result = pub.execute(args)
    assert result["status"] == "completed" and len(result["videos"]) == 6
    assert result["decisions"][0]["decision"] == "keep_experimental"
    assert result["decisions"][0]["eligible_for_default"] is False
    assert any(c["status"] == "completed_with_explicit_unsupported" for c in result["campaigns"])
    for entry in result["files"]:
        source = tmp_path / entry["source"]
        target = args.output_dir / entry["destination"]
        assert target.read_bytes() == source.read_bytes()
        assert pub.base.sha256(target) == entry["sha256"]
        assert target.stat().st_size == entry["bytes"]
        assert not Path(entry["source"]).is_absolute()
        assert target.suffix in (".json", ".mp4")
    assert not list(args.output_dir.rglob("*.f32"))
    assert not list(args.output_dir.rglob("*.bf16"))
    assert not list(args.output_dir.rglob("*.safetensors"))
    assert not list(args.output_dir.rglob("native-cli"))
    gallery = (args.output_dir / "gallery.html").read_text()
    assert gallery.count("<video ") == 6
    assert '<a href="results.md">Measured results</a>' in gallery
    assert "[Measured results](results.md)" in (args.output_dir / "README.md").read_text()
    for video in result["videos"]:
        assert video["file"] in gallery
    assert "not a new numerical" in result["publication_scope"]
    for generated in result["generated_files"]:
        assert pub.base.sha256(args.output_dir / generated["file"]) == generated["sha256"]


def test_recorded_source_snapshots_are_optional_and_hash_checked(tmp_path, pub):
    args, _ = setup(tmp_path, pub)
    args.include_source_snapshots = True
    result = pub.execute(args)
    assert any(e["role"] == "producer_source_snapshot" for e in result["files"])
    assert any(e["role"] == "pinned_native_patch" for e in result["files"])
    inventories = [e for e in result["files"] if e["source"].endswith("/model-files.json")]
    assert inventories
    assert isinstance(
        json.loads((args.output_dir / inventories[0]["destination"]).read_text()), list
    )
    assert len(result["videos"]) == 6


def test_existing_destination_is_never_changed(tmp_path, pub):
    args, _ = setup(tmp_path, pub)
    args.output_dir.mkdir(parents=True)
    marker = args.output_dir / "keep"
    marker.write_bytes(b"untouched")
    with pytest.raises(FileExistsError):
        pub.execute(args)
    assert marker.read_bytes() == b"untouched"
    assert list(args.output_dir.iterdir()) == [marker]


@pytest.mark.parametrize(
    "damage",
    [
        "driver",
        "case",
        "video",
        "media",
        "source",
        "seed",
        "decision",
        "analysis",
        "missing_run",
        "snapshot",
    ],
)
def test_wrong_sha_or_evidence_join_fails_before_publication(tmp_path, pub, damage):
    args, runs = setup(tmp_path, pub)
    if damage == "driver":
        path = args.prep_campaign[0] / "case-a/driver.json"
        write(path, json.loads(path.read_text()) | dict(seed=29))
    elif damage == "case":
        path = args.prep_campaign[0] / "case-a/case.json"
        write(path, json.loads(path.read_text()) | dict(status="incorrect"))
    elif damage == "video":
        (args.video_run_dir[0] / "video-5s.mp4").write_bytes(b"wrong video")
    elif damage == "media":
        path = args.evaluation[0].parent / "combined-fused-media/evaluation.json"
        write(path, dict(status="failed"))
    elif damage == "source":
        args.source_manifest_sha256 = "0" * 64
    elif damage == "seed":
        path = args.evaluation[1]
        write(path, json.loads(path.read_text()) | dict(seed=42))
    elif damage == "decision":
        path = args.decision[0]
        value = json.loads(path.read_text())
        value["performance"]["comparisons"]["baseline__qkv-only"][
            "both_seeds_complete_request_benefit"
        ] = True
        write(path, value)
    elif damage == "analysis":
        path = args.analysis[0]
        value = json.loads(path.read_text())
        value["capture"]["cases"] = []
        write(path, value)
    elif damage == "missing_run":
        args.run_dir.remove(runs["combined", 42]["directory"])
    else:
        args.include_source_snapshots = True
        (runs["combined", 42]["directory"] / "harness-snapshot/run.py").write_text("changed\n")
    with pytest.raises(ValueError):
        pub.execute(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize(
    "text",
    [
        "/Users/person/private",
        "/private/tmp/image.jpg",
        "/tmp/trace",
        "codex://threads/private",
        "access_token",
        "01a150de-2b4e-7f12-95e3-789b1f51e404",
    ],
)
def test_private_raw_json_is_rejected_without_sanitizing(tmp_path, pub, text):
    args, _ = setup(tmp_path, pub)
    path = args.decision[0]
    value = json.loads(path.read_text()) | dict(note=text)
    write(path, value)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        pub.execute(args)
    assert path.read_bytes() == before
    assert not args.output_dir.exists()


def test_parent_traversal_and_symlink_sources_are_rejected(tmp_path, pub):
    source = write(tmp_path / "local/x.json", {"value": 1})
    package = pub.Package(tmp_path)
    with pytest.raises(ValueError, match="destination"):
        package.add(source, "../x.json", "test")
    link = tmp_path / "local/link.json"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="regular file"):
        package.add(link, "raw/x.json", "test")
    with pytest.raises(ValueError, match="identifiers"):
        pub.identifier("../../escape")


def test_copy_rejects_mutation_after_validation_and_preserves_partial_evidence(tmp_path, pub):
    source = write(tmp_path / "source.json", {"value": 1})
    package = pub.Package(tmp_path)
    package.add(source, "raw/source.json", "test")
    write(source, {"value": 2})
    output = tmp_path / "public"
    with pytest.raises(ValueError, match="partial destination"):
        package.copy(output)
    assert (output / "raw/source.json").is_file()
    assert not (output / "manifest.json").exists()


def test_source_whitelist_never_accepts_executable_even_with_matching_hash(tmp_path, pub):
    source = tmp_path / "binary"
    source.write_bytes(b"binary")
    package = pub.Package(tmp_path)
    with pytest.raises(ValueError, match="whitelist"):
        package.add(source, "raw/binary", "source", pub.base.sha256(source), source_text=True)


def test_list_is_allowed_only_for_hash_bound_source_json_not_report(tmp_path, pub):
    source = write(tmp_path / "model-files.json", [{"path": "model/config.json"}])
    package = pub.Package(tmp_path)
    with pytest.raises(ValueError, match="object"):
        package.add(source, "raw/report.json", "report")
    with pytest.raises(ValueError, match="expected SHA256"):
        package.add(source, "raw/source.json", "source", source_text=True)
    package.add(source, "raw/source.json", "source", pub.base.sha256(source), source_text=True)


@pytest.mark.parametrize("expected", [None, "", "invalid", "0" * 63])
def test_selected_media_cannot_be_published_without_recorded_artifact_digest(
    tmp_path, pub, expected
):
    video = tmp_path / "video-5s.mp4"
    video.write_bytes(b"fixture media")
    package = pub.Package(tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        package.add(video, "videos/sample.mp4", "exact_5s_video", expected, media=True)
    assert not package.files


@pytest.mark.parametrize("field", ["latent", "activation"])
@pytest.mark.parametrize("missing", [False, True])
def test_capture_metadata_requires_recorded_sha_including_missing_and_null(
    tmp_path, pub, field, missing
):
    args, runs = setup(tmp_path, pub)
    directory = runs["combined", 42]["directory"]
    path = directory / "run.json"
    value = json.loads(path.read_text())
    record = (
        value["latent_validation"]
        if field == "latent"
        else next(iter(value["adaln_validation"]["files"].values()))
    )
    key = "metadata_sha256" if field == "latent" else "sha256"
    if missing:
        del record[key]
    else:
        record[key] = None
    write(path, value)
    with pytest.raises(ValueError, match="SHA256"):
        pub.collect_run(
            pub.Package(tmp_path),
            directory,
            json.loads(args.source_manifest.read_text()),
            args.source_manifest_sha256,
        )


@pytest.mark.parametrize("damage", ["missing", "wrong_policy", "historical"])
def test_selected_video_set_is_exactly_three_new_policies_for_both_seeds(tmp_path, pub, damage):
    args, runs = setup(tmp_path, pub)
    if damage == "missing":
        args.video_run_dir.pop()
    else:
        args.video_run_dir[-1] = runs[
            "baseline" if damage == "wrong_policy" else "old-hadamard", 43
        ]["directory"]
    with pytest.raises(ValueError, match="videos"):
        pub.execute(args)
    assert not args.output_dir.exists()


def test_duplicate_campaign_is_not_claimed_as_two_reruns(tmp_path, pub):
    args, _ = setup(tmp_path, pub)
    args.prep_campaign[1] = args.prep_campaign[0]
    with pytest.raises(ValueError, match="duplicate"):
        pub.execute(args)


@pytest.mark.parametrize(
    "field", ["resident_comparison", "decision", "analysis", "run_dir", "video_run_dir"]
)
def test_required_evidence_cannot_be_silently_omitted(tmp_path, pub, field):
    args, _ = setup(tmp_path, pub)
    setattr(args, field, [])
    with pytest.raises(ValueError, match="required"):
        pub.execute(args)
