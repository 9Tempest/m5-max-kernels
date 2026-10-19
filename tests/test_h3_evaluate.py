"""CPU-only numerical and mocked-media coverage; never starts H3 or a GPU."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "h3_evaluate", Path(__file__).resolve().parents[1] / "integrations/h3/evaluate.py"
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_known_metrics_and_global_psnr():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = a + np.array([0.0, 1.0, -1.0, 0.0])
    result = evaluation.numeric_metrics(a, b, 4)
    m = result["metrics"]
    assert m["snr_db"]["value"] == pytest.approx(10 * math.log10(30 / 2))
    assert m["relative_l2"]["value"] == pytest.approx(math.sqrt(2 / 30))
    assert m["rmse"]["value"] == pytest.approx(math.sqrt(0.5))
    assert m["maximum_absolute_error"]["value"] == 1
    assert m["psnr_db"]["value"] == pytest.approx(10 * math.log10(16 / 0.5))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "a,b,snr,l2",
    [
        ([0, 0], [0, 0], "undefined", "undefined"),
        ([0, 0], [1, 0], "negative_infinity", "positive_infinity"),
        ([1, 2], [1, 2], "positive_infinity", "finite"),
    ],
)
def test_zero_cases_are_explicit_json_states(a, b, snr, l2):
    result = evaluation.numeric_metrics(np.array(a), np.array(b), 255)
    assert result["metrics"]["snr_db"]["status"] == snr
    assert result["metrics"]["relative_l2"]["status"] == l2
    if a == b:
        assert result["metrics"]["psnr_db"] == {
            "status": "positive_infinity",
            "value": None,
            "reason": "zero error",
        }
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_reject_every_metric(bad):
    result = evaluation.numeric_metrics(np.array([1.0, bad]), np.ones(2), 1)
    assert result["status"] == "failed"
    assert result["reference_nonfinite"] == 1
    assert all(v["status"] == "invalid_nonfinite" for v in result["metrics"].values())
    assert all(v["value"] is None for v in result["metrics"].values())
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("data_range", [0, -1, np.inf, np.nan])
def test_invalid_data_range(data_range):
    with pytest.raises(ValueError, match="data_range"):
        evaluation.numeric_metrics(np.ones(1), np.ones(1), data_range)


def test_shape_empty_complex_and_object_rejected():
    for a, b in [
        (np.ones((2, 1)), np.ones(2)),
        (np.array([]), np.array([])),
        (np.array([1j]), np.array([1j])),
        (np.array(["1"], dtype=object), np.array(["1"], dtype=object)),
    ]:
        with pytest.raises(ValueError):
            evaluation.numeric_metrics(a, b, 1)


def test_outliers_and_tail_do_not_get_sampled_away():
    a = np.ones(1025)
    b = a.copy()
    b[-1] = 1e6
    result = evaluation.numeric_metrics(a, b, 1)
    assert result["metrics"]["maximum_absolute_error"]["value"] == 999999
    assert result["metrics"]["rmse"]["value"] == pytest.approx(999999 / math.sqrt(1025))


def test_large_finite_energy_avoids_square_overflow():
    result = evaluation.numeric_metrics(np.array([1e300, -1e300]), np.array([0.0, 0.0]), 1)
    assert result["status"] == "completed"
    assert result["metrics"]["snr_db"]["value"] == 0
    assert result["metrics"]["relative_l2"]["value"] == 1
    assert result["metrics"]["rmse"]["value"] == pytest.approx(1e300)
    json.dumps(result, allow_nan=False)


def test_subtraction_overflow_is_explicit_failure():
    result = evaluation.numeric_metrics(np.array([1e308]), np.array([-1e308]), 1)
    assert result["status"] == "failed"
    assert result["metrics"]["rmse"]["status"] == "numerical_overflow"
    json.dumps(result, allow_nan=False)


def test_range_changes_psnr_but_not_snr_and_large_integer_is_rejected():
    a, b = np.array([0.1, 0.2]), np.array([0.2, 0.3])
    one = evaluation.numeric_metrics(a, b, 1)["metrics"]
    two = evaluation.numeric_metrics(a, b, 2)["metrics"]
    assert two["psnr_db"]["value"] - one["psnr_db"]["value"] == pytest.approx(20 * math.log10(2))
    assert two["snr_db"] == one["snr_db"]
    with pytest.raises(ValueError, match="exactly representable"):
        evaluation.numeric_metrics(np.array([2**53 + 1]), np.array([2**53]), 1)


def valid_probe():
    # Three frames at 24fps = 4000 stereo PCM samples at 32kHz.
    return {
        "streams": [
            dict(
                codec_type="video",
                width=8,
                height=8,
                avg_frame_rate="24/1",
                r_frame_rate="24/1",
                time_base="1/24000",
                duration_ts=3000,
                pix_fmt="yuv420p",
                sample_aspect_ratio="1:1",
            ),
            dict(
                codec_type="audio",
                sample_rate="32000",
                channels=2,
                channel_layout="stereo",
                time_base="1/32000",
                duration_ts=4000,
            ),
        ],
        "frames": [
            dict(
                media_type="video", best_effort_timestamp=i * 1000, duration=1000, width=8, height=8
            )
            for i in range(3)
        ]
        + [
            dict(
                media_type="audio",
                best_effort_timestamp=0,
                nb_samples=2000,
                duration=2000,
                channels=2,
                channel_layout="stereo",
            ),
            dict(
                media_type="audio",
                best_effort_timestamp=2000,
                nb_samples=2000,
                duration=2000,
                channels=2,
                channel_layout="stereo",
            ),
        ],
    }


def test_exact_media_contract_and_equivalent_rational_timebases():
    a = valid_probe()
    b = copy.deepcopy(a)
    b["streams"][0]["time_base"] = "1/12000"
    b["streams"][0]["duration_ts"] = 1500
    for frame in b["frames"][:3]:
        frame["best_effort_timestamp"] //= 2
        frame["duration"] //= 2
    assert evaluation.media_contract(a) == evaluation.media_contract(b)
    assert evaluation.media_contract(a)["decoded_audio_samples"] == 4000


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["streams"][1].update(sample_rate="44100"),
        lambda p: p["streams"][1].update(channels=1),
        lambda p: p["streams"][1].update(channel_layout="unknown"),
        lambda p: p["streams"][0].update(r_frame_rate="30/1"),
        lambda p: p["streams"][0].update(avg_frame_rate="0/0"),
        lambda p: p["streams"][0].update(side_data_list=[{"rotation": 90}]),
        lambda p: p["streams"][1].update(duration_ts=3999),
        lambda p: p["frames"][1].update(best_effort_timestamp=1001),
        lambda p: p["frames"][2].update(width=9),
        lambda p: p["frames"][3].update(best_effort_timestamp=1),
        lambda p: p["frames"][4].update(best_effort_timestamp=2001),
        lambda p: p["frames"][4].update(nb_samples=0),
        lambda p: p["frames"][1].update(duration=999),
        lambda p: p["frames"][1].update(crop_left=1),
        lambda p: p["frames"][4].update(duration=1999),
        lambda p: p["frames"][4].update(channels=1),
        lambda p: p.update(frames=[]),
    ],
)
def test_media_rejects_geometry_timewarp_and_audio_misalignment(mutation):
    probe = valid_probe()
    mutation(probe)
    with pytest.raises(ValueError):
        evaluation.media_contract(probe)


def args(tmp_path, *extra):
    return evaluation.parser().parse_args(
        [
            str(tmp_path / "a.npy"),
            str(tmp_path / "b.npy"),
            "--kind",
            "arrays",
            "--data-range",
            "1",
            "--output-dir",
            str(tmp_path / "evaluation"),
            *extra,
        ]
    )


def test_array_cli_preserves_evidence_and_refuses_overwrite(tmp_path):
    np.save(tmp_path / "a.npy", np.array([1.0, 2.0]))
    np.save(tmp_path / "b.npy", np.array([1.0, 2.1]))
    result = evaluation.execute(args(tmp_path))
    assert result["status"] == "completed"
    assert result["quality_decision"] == "not_evaluated"
    assert result["optional"]["lpips"]["status"] == "not_measured"
    assert result["files"]["reference"]["sha256"] == evaluation.sha256(tmp_path / "a.npy")
    assert json.loads((tmp_path / "evaluation/evaluation.json").read_text()) == result
    with pytest.raises(FileExistsError):
        evaluation.execute(args(tmp_path))


def test_array_failure_is_saved(tmp_path):
    np.save(tmp_path / "a.npy", np.array([np.nan]))
    np.save(tmp_path / "b.npy", np.array([1.0]))
    result = evaluation.execute(args(tmp_path))
    assert result["status"] == "failed"
    assert "nonfinite" in result["error"]
    assert result["arrays"]["array"]["metrics"]["snr_db"]["value"] is None
    json.loads((tmp_path / "evaluation/evaluation.json").read_text())


def test_unavailable_requested_backend_is_partial_never_substituted(tmp_path):
    np.save(tmp_path / "a.npy", np.ones(2))
    np.save(tmp_path / "b.npy", np.ones(2))
    result = evaluation.execute(args(tmp_path, "--lpips", "--vbench", "--ssim"))
    assert result["status"] == "partial"
    assert all(
        result["optional"][k]["status"] == "unavailable" for k in ("lpips", "vbench", "ssim")
    )


def test_npz_keys_must_match_and_pickle_is_forbidden(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, latent=np.ones(2))
    np.savez(b, other=np.ones(2))
    with pytest.raises(ValueError, match="keys"):
        evaluation.array_pairs(a, b)
    np.savez(b, latent=np.zeros(2))
    assert set(evaluation.array_pairs(a, b)) == {"latent"}
    np.save(tmp_path / "object.npy", np.array([{}], dtype=object))
    with pytest.raises(ValueError):
        evaluation.array_pairs(tmp_path / "object.npy", tmp_path / "object.npy")


def test_mock_decoder_never_rescales_or_resamples(tmp_path, monkeypatch):
    contract = evaluation.media_contract(valid_probe())
    commands = []

    def fake_command(argv, out, name, processes, stdout_file=None):
        commands.append(argv)
        path = out / stdout_file
        if stdout_file.endswith(".rgb"):
            path.write_bytes(bytes(3 * 8 * 8 * 3))
        else:
            path.write_bytes(bytes(4000 * 2 * 4))
        return path

    monkeypatch.setattr(evaluation, "command", fake_command)
    rgb, pcm = evaluation.decode_media(
        "ffmpeg", tmp_path / "input.mp4", contract, tmp_path, "reference", []
    )
    assert rgb.shape == (3, 8, 8, 3)
    assert pcm.shape == (4000, 2)
    assert all("-hwaccel" in argv and "none" in argv for argv in commands)
    assert all(not ({"-ar", "-ac", "-r", "-s", "-vf"} & set(argv)) for argv in commands)
    assert "passthrough" in commands[0]


def test_mock_decoder_rejects_truncated_pcm(tmp_path, monkeypatch):
    contract = evaluation.media_contract(valid_probe())

    def fake_command(argv, out, name, processes, stdout_file=None):
        path = out / stdout_file
        path.write_bytes(bytes(3 * 8 * 8 * 3 if stdout_file.endswith(".rgb") else 8))
        return path

    monkeypatch.setattr(evaluation, "command", fake_command)
    with pytest.raises(ValueError, match="PCM"):
        evaluation.decode_media("ffmpeg", tmp_path / "input.mp4", contract, tmp_path, "ref", [])


def test_ssim_is_explicit_rgb_and_requires_all_frames(tmp_path, monkeypatch):
    contract = evaluation.media_contract(valid_probe())
    commands = []

    def fake_command(argv, out, name, processes):
        commands.append(argv)
        path = out / "ssim.stdout.log"
        path.write_text("n:1 All:0.8\nn:2 All:0.9\nn:3 All:1.0\n")
        return path

    monkeypatch.setattr(evaluation, "command", fake_command)
    result = evaluation.ffmpeg_ssim("ffmpeg", contract, tmp_path, [])
    assert result["mean"]["value"] == pytest.approx(0.9)
    assert "format=gbrp" in " ".join(commands[0])
    contract["frames"] = 4
    with pytest.raises(ValueError, match="one finite value per"):
        evaluation.ffmpeg_ssim("ffmpeg", contract, tmp_path, [])
