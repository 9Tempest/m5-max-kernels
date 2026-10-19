"""Counterbalanced resident comparison gates, using small JSON-only fixtures."""

import copy
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/h3"))
import compare_kernel_resident as comparison
import summarize_kernel_resident as descriptive


def test_partial_matched_comparison_cannot_promote():
    value = descriptive.summarize(records()[:2])
    assert value["promotion_eligible"] is False
    assert len(value["pairs"]) == 1


def test_partial_comparison_still_rejects_source_mismatch():
    values = records()[:2]
    values[1]["producer"]["source_sha256"] = "changed"
    with pytest.raises(ValueError):
        descriptive.summarize(values)


def test_unpaired_seeds_cannot_form_partial_speedup():
    values = [records()[0], records()[4]]
    with pytest.raises(ValueError, match="same-seed"):
        descriptive.summarize(values)


def timestamp(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def records():
    values = []
    for seed, policies in comparison.ORDER.items():
        for policy in policies:
            start = 10000 + len(values) * 1000
            wall = 150.0 if policy == "baseline" else 140.0
            values.append(
                dict(
                    seed=seed,
                    policy=policy,
                    run_sha256=str(len(values)) * 64,
                    producer=dict(
                        source_pin_sha256="a" * 64,
                        source_sha256="b" * 64,
                        model={"revision": "fixed"},
                        harness={"driver": "same"},
                    ),
                    environment={"chip": "M5"},
                    compilation={"flags": "same"},
                    timing_contract={"warmups": 1, "repetitions": 3},
                    shape={"packed": 9914},
                    process=dict(started_utc=timestamp(start), finished_utc=timestamp(start + 700)),
                    video_sha256=[policy] * 4,
                    timing={
                        key: {"raw_seconds": [wall, wall + 1, wall - 1]}
                        for key in ("cached_generation", "cached_denoise_callback")
                    },
                )
            )
    return values


def test_two_seed_independent_reversed_processes_and_drift_allowed():
    result = comparison.compare_records(records())
    pair = result["comparisons"]["baseline__qkv-only"]
    assert pair["both_seeds_complete_request_benefit"]
    assert pair["neither_seed_denoise_regresses"]
    assert pair["42"]["same_seed_outputs_byte_identical"] is False
    assert len(result["comparisons"]) == 5


def test_selected_candidate_requires_its_own_two_seeds_and_reverse_order():
    selected = [r for r in records() if r["policy"] in ("baseline", "combined")]
    result = comparison.compare_records(selected, ("baseline", "combined"))
    assert set(result["comparisons"]) == {"baseline__combined"}
    with pytest.raises(ValueError, match="exactly one"):
        comparison.compare_records(selected[:-1], ("baseline", "combined"))
    with pytest.raises(ValueError, match="baseline"):
        comparison.compare_records(selected, ("combined",))


def test_second_seed_regression_not_hidden_by_average():
    values = records()
    target = next(r for r in values if (r["seed"], r["policy"]) == (43, "qkv-only"))
    target["timing"]["cached_generation"]["raw_seconds"] = [160, 161, 159]
    pair = comparison.compare_records(values)["comparisons"]["baseline__qkv-only"]
    assert pair["both_seeds_complete_request_benefit"] is False


@pytest.mark.parametrize("field", ["environment", "compilation", "timing_contract", "shape"])
def test_unmatched_experiment_rejected(field):
    values = records()
    values[-1][field] = {"different": True}
    with pytest.raises(ValueError, match="matched"):
        comparison.compare_records(values)


@pytest.mark.parametrize("field", ["source_pin_sha256", "source_sha256", "model", "harness"])
def test_producer_difference_rejected(field):
    values = records()
    values[-1]["producer"][field] = "changed"
    with pytest.raises(ValueError, match="matched producer"):
        comparison.compare_records(values)


def test_missing_or_duplicate_policy_cannot_be_replication():
    for values in (records()[:-1], records()[:-1] + [copy.deepcopy(records()[0])]):
        with pytest.raises(ValueError, match="exactly one"):
            comparison.compare_records(values)


def test_overlapping_processes_rejected():
    values = records()
    values[1]["process"]["started_utc"] = values[0]["process"]["started_utc"]
    with pytest.raises(ValueError, match="overlap"):
        comparison.compare_records(values)


def test_second_seed_must_reverse_order():
    values = records()
    last = values[4:]
    for record, original in zip(last, reversed(copy.deepcopy(last)), strict=True):
        record["process"] = original["process"]
    with pytest.raises(ValueError, match="counterbalanced"):
        comparison.compare_records(values)
