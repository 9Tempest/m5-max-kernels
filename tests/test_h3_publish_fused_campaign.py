"""Raw fusion publication preserves provenance, timing and immutable bytes."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from test_h3_fused_campaign import valid_report


@pytest.fixture
def publisher(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "integrations/h3"
    monkeypatch.syspath_prepend(str(directory))
    for name in ("run", "publish_samples", "run_fused_campaign"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "publish_fused_test", directory / "publish_fused_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path, publisher, *, name="input", warmups=3, repeats=3, fused_multiplier=1):
    root = tmp_path / name
    root.mkdir()
    fused = publisher.fused
    record = dict(
        status="completed",
        patch=dict(source_sha256=fused.SOURCE_SHA256, patch_sha256=fused.PATCH_SHA256),
        upstream=dict(revision=fused.base.UPSTREAM_REVISION, source_sha256=fused.SOURCE_SHA256),
        driver=dict(source_sha256=fused.DRIVER_SHA256, reference_sha256=fused.REFERENCE_SHA256),
        model=dict(
            repository="MiniMaxAI/MiniMax-H3",
            revision=fused.base.MODEL_REVISION,
            manifest_sha256=fused.base.MODEL_MANIFEST_SHA256,
        ),
        wrapper_sha256="a" * 64,
        environment=dict(chip="Apple M5 Max"),
        harness=dict(revision="pinned", dirty=False),
        cases=[],
    )
    for seed, rows, policy in fused.CASES:
        name = f"{policy}-m{rows}-seed{seed}"
        (root / name).mkdir()
        report = valid_report(fused, seed, rows, policy, warmups, repeats)
        for sample in report["timing"]["fused"]:
            for key in (
                "wall_seconds",
                "cpu_encode_seconds",
                "submit_wait_seconds",
                "supplemental_gpu_seconds",
            ):
                sample[key] *= fused_multiplier
        (root / name / "driver.json").write_text(json.dumps(report))
        case = dict(
            seed=seed,
            rows=rows,
            policy=policy,
            status="completed",
            process=dict(returncode=0, fallback_detected=False),
            driver_report_sha256=fused.base.sha256(root / name / "driver.json"),
            summary=fused.validate_case(report, seed, rows, policy, warmups, repeats),
        )
        (root / name / "case.json").write_text(json.dumps(case))
        record["cases"].append(case | dict(directory=name))
    (root / "campaign.json").write_text(json.dumps(record))
    (root / "request.json").write_text(
        json.dumps(record | dict(cases_requested=fused.CASES, warmups=warmups, repeats=repeats))
    )
    return root, record


def test_raw_json_and_samples_copied_exactly(tmp_path, publisher):
    source, _ = fixture(tmp_path, publisher)
    output = tmp_path / "public"
    result = publisher.execute(source, output)
    assert len(result["cases"]) == 12
    assert len(result["raw_files"]) == 26
    for entry in result["raw_files"]:
        assert (output / entry["file"]).read_bytes() == (
            source / Path(entry["file"]).relative_to("raw")
        ).read_bytes()
    assert (
        result["conclusions"]["end_to_end_acceleration"]
        == "not measured by this operator-only export"
    )
    assert "CPU residual restoration is outside timing" in (output / "README.md").read_text()
    with pytest.raises(FileExistsError):
        publisher.execute(source, output)


def test_private_paths_rejected_before_publication(tmp_path, publisher):
    source, record = fixture(tmp_path, publisher)
    record["private"] = "/Users/owner/local"
    (source / "campaign.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="private"):
        publisher.collect(source)


def test_unreviewed_device_identity_rejected(tmp_path, publisher):
    source, record = fixture(tmp_path, publisher)
    record["registry_id"] = "unique-device"
    (source / "campaign.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="private identifier"):
        publisher.collect(source)


def test_driver_digest_change_rejected(tmp_path, publisher):
    source, _ = fixture(tmp_path, publisher)
    (source / "h256-m2049-seed17/driver.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        publisher.collect(source)


def test_runtime_pin_change_rejected(tmp_path, publisher):
    source, record = fixture(tmp_path, publisher)
    record["patch"]["source_sha256"] = "0" * 64
    (source / "campaign.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="pinned"):
        publisher.collect(source)


def test_missing_case_rejected(tmp_path, publisher):
    source, record = fixture(tmp_path, publisher)
    record["cases"].pop()
    (source / "campaign.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="missing"):
        publisher.collect(source)


@pytest.mark.parametrize(
    "multiplier,ratio,lower,higher",
    [(2, "0.50000", 0, 4), (0.5, "2.00000", 4, 0), (1, "1.00000", 0, 0)],
)
def test_readme_uses_actual_campaign_values(tmp_path, publisher, multiplier, ratio, lower, higher):
    source, _ = fixture(
        tmp_path,
        publisher,
        name="fused-campaign-002",
        warmups=4,
        repeats=5,
        fused_multiplier=multiplier,
    )
    result, _ = publisher.collect(source)
    page = publisher.readme(result)
    assert page.startswith("# Native gate epilogue fused-campaign-002\n")
    assert "campaign-001" not in page
    assert "4 warmups and 5 alternating-order measured repetitions" in page
    assert (
        f"`h256`: {ratio}–{ratio}; fused median lower in {lower}/4 cases, higher in {higher}/4"
        in page
    )
    assert "seed 17: native first; seed 29: fused first" in page
    assert "remain pending" not in page
    assert (
        result["conclusions"]["independent_rerun"] == "not assessed by this single-campaign export"
    )


def test_campaign_identifier_cannot_inject_markdown(tmp_path, publisher):
    source, _ = fixture(tmp_path, publisher, name="bad\n# title")
    with pytest.raises(ValueError, match="plain publication identifier"):
        publisher.collect(source)
