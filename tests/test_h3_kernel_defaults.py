"""Default promotion must require numerical, quality and repeated speed evidence."""

import itertools
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/h3"))
import adaln_capture
import decide_kernel_defaults as defaults


@pytest.mark.parametrize(
    "implementation,quality,e2e,denoise",
    (
        (False, True, True, True),
        (True, False, True, True),
        (True, True, False, True),
        (True, True, True, False),
        (None, True, True, True),
        (1, True, True, True),
    ),
)
def test_any_missing_or_failed_gate_keeps_opt_in(implementation, quality, e2e, denoise):
    performance = {
        "comparisons": {
            "baseline__combined": {
                "both_seeds_complete_request_benefit": e2e,
                "neither_seed_denoise_regresses": denoise,
            }
        }
    }
    result = defaults.decide("combined", performance, quality, implementation)
    assert result["decision"] == "keep_experimental"
    assert result["eligible_for_default"] is False
    assert result["runtime_defaults_modified"] is False


def test_success_is_recommendation_not_automatic_mutation():
    performance = {
        "comparisons": {
            "baseline__qkv-only": {
                "both_seeds_complete_request_benefit": True,
                "neither_seed_denoise_regresses": True,
            }
        }
    }
    result = defaults.decide("qkv-only", performance, True, True)
    assert result["eligible_for_default"] is True
    assert result["runtime_defaults_modified"] is False


def activation_report(run_sha):
    cases = []
    for family, step, block in itertools.product(("qkv", "fc1"), (0, 10, 19), (0, 24, 49)):
        metric = dict(finite=True, relative_l2=0.001, rmse=0.002, max_absolute=0.003)
        cases.append(
            dict(
                metadata=dict(family=family, step=step, block=block),
                preparation={
                    "status": "passed",
                    "relative_l2_limit": 0.008,
                    "gate_error_vs_independent_fp64_then_bf16": metric,
                    "adaln_error_vs_independent_fp64_then_bf16": metric,
                    "activation_pack_implementation": {"pass": True},
                },
                projection_precision={
                    "rotation_check_pass": True,
                    "plain_row_int8_quantization_error": metric,
                    "h256_row_int8_quantization_error": metric,
                },
            )
        )
    return dict(
        analyzer_sha256=defaults.base.sha256(adaln_capture.__file__),
        reference_helper_sha256=defaults.base.sha256(
            Path(adaln_capture.__file__).with_name("fc2_analyze.py")
        ),
        source_run_sha256=run_sha,
        cases=cases,
    )


def test_activation_evidence_uses_actual_analyzer_metric_names(tmp_path):
    paths, evaluations = [], []
    for seed in (42, 43):
        run_sha = str(seed) * 32
        path = tmp_path / f"{seed}.json"
        path.write_text(json.dumps(activation_report(run_sha)))
        paths.append(path)
        evaluations.append(dict(candidate_run_sha256=run_sha, seed=seed))
    assert len(defaults.activation_evidence(paths, evaluations)) == 2
    report = json.loads(paths[1].read_text())
    report["cases"][-1]["projection_precision"]["h256_row_int8_quantization_error"].pop(
        "max_absolute"
    )
    paths[1].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        defaults.activation_evidence(paths, evaluations)
