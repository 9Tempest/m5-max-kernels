"""Cross-campaign publication requires distinct, matching serial experiments."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from test_h3_publish_fused_campaign import fixture as campaign_fixture


@pytest.fixture
def comparison(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "integrations/h3"
    monkeypatch.syspath_prepend(str(directory))
    for name in ("run", "publish_samples", "run_fused_campaign", "publish_fused_campaign"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "fusion_comparison_test", directory / "compare_fused_campaigns.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def update(directory, change):
    for name in ("request.json", "campaign.json"):
        path = directory / name
        record = json.loads(path.read_text())
        change(record)
        path.write_text(json.dumps(record))


def pair(tmp_path, comparison):
    directories = []
    for index, name in enumerate(("first", "second")):
        directory, _ = campaign_fixture(
            tmp_path, comparison.publisher, name=name, fused_multiplier=0.5 if index == 0 else 2
        )

        def metadata(record, index=index):
            record.update(
                started_utc=f"2026-10-18T{index * 2:02d}:00:00+00:00",
                finished_utc=f"2026-10-18T{index * 2 + 1:02d}:00:00+00:00",
            )
            record["driver"].update(binary_sha256="c" * 64, library_sha256=str(index + 1) * 64)

        update(directory, metadata)
        directories.append(directory)
    return directories


def test_direction_changes_are_not_repeat_gains(tmp_path, comparison):
    first, second = pair(tmp_path, comparison)
    result = comparison.execute(first, second, tmp_path / "published")
    assert result["numerical_cases_passed"] == 24
    assert all(not case["observed_lower_median_in_both_campaigns"] for case in result["cases"])
    assert all(not group["lower_median_all_seeds_both_campaigns"] for group in result["groups"])
    assert result["library_archive_sha_equal"] is False
    assert "Library archive hashes differ" in comparison.readme(result)
    assert result["cases"][0]["observations"][0]["native_over_fused_ratio"] == 2
    assert result["cases"][0]["observations"][1]["native_over_fused_ratio"] == 0.5
    with pytest.raises(FileExistsError):
        comparison.execute(first, second, tmp_path / "published")


def test_identical_campaign_is_not_an_independent_rerun(tmp_path, comparison):
    first, _ = pair(tmp_path, comparison)
    with pytest.raises(ValueError, match="same campaign"):
        comparison.compare(first, first)


def test_different_linked_binary_rejected(tmp_path, comparison):
    first, second = pair(tmp_path, comparison)
    update(second, lambda record: record["driver"].update(binary_sha256="d" * 64))
    with pytest.raises(ValueError, match="binary_sha256"):
        comparison.compare(first, second)


def test_different_device_environment_rejected(tmp_path, comparison):
    first, second = pair(tmp_path, comparison)
    update(second, lambda record: record["environment"].update(chip="different device"))
    with pytest.raises(ValueError, match="environment"):
        comparison.compare(first, second)


def test_overlapping_campaigns_rejected(tmp_path, comparison):
    first, second = pair(tmp_path, comparison)
    update(second, lambda record: record.update(started_utc="2026-10-18T00:30:00+00:00"))
    with pytest.raises(ValueError, match="overlap"):
        comparison.compare(first, second)


def test_matching_archives_are_not_mislabeled_different(tmp_path, comparison):
    first, second = pair(tmp_path, comparison)
    update(second, lambda record: record["driver"].update(library_sha256="1" * 64))
    result = comparison.compare(first, second)
    assert result["library_archive_sha_equal"] is True
    assert "Library archive hashes match" in comparison.readme(result)
