"""CPU-only frozen-catalog and runner preflight tests; never launches Metal."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys

import pytest


DIRECTORY = Path(__file__).resolve().parents[1] / "integrations/h3"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("h3_benchmark", DIRECTORY / "run_benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def arguments(tmp_path, mode="bf16", *extra):
    return benchmark.parser().parse_args(
        [
            "--benchmark-id",
            "person-turn-v1",
            "--mode",
            mode,
            "--output-dir",
            str(tmp_path / "run"),
            *extra,
        ]
    )


@pytest.mark.parametrize(
    "name,seed", [("person-turn-v1", 71), ("rainy-street-v1", 73), ("water-pour-v1", 79)]
)
def test_catalog_is_exact_fixed_recipe(name, seed):
    item = benchmark.catalog.load_benchmark(name)
    assert item["seed"] == seed
    assert (item["width"], item["height"], item["seconds"], item["fps"], item["steps"]) == (
        512,
        512,
        5,
        24,
        20,
    )
    assert item["active_dit_blocks"] == 50
    assert set(item["checks"]) == {"temporal", "texture", "audio"}


def test_changed_catalog_and_unknown_id_rejected(tmp_path):
    changed = tmp_path / "catalog.json"
    changed.write_bytes(benchmark.catalog.DEFAULT.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        benchmark.catalog.load_benchmark("person-turn-v1", changed)
    with pytest.raises(ValueError, match="exactly one"):
        benchmark.catalog.load_benchmark("not-a-benchmark")


def test_command_uses_catalog_prompt_without_mutating_original(tmp_path):
    item = benchmark.catalog.load_benchmark("rainy-street-v1")
    original = benchmark.base.PROMPT
    for mode in ("bf16", "hadamard-fc2"):
        command = benchmark.build_command(item, mode, tmp_path, tmp_path, tmp_path / "video.mp4")
        assert command[command.index("-p") + 1] == item["prompt"]
        assert command[command.index("--seed") + 1] == "73"
        assert "--use-reference-rope" in command
        assert all((flag in command) == (mode == "bf16") for flag in benchmark.base.BF16_FLAGS)
    assert benchmark.base.PROMPT == original


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--checkpoint-dir", "absent"],
        ["--checkpoint-dir", "absent", "--checkpoint-manifest-sha256", "bad"],
        [
            "--checkpoint-dir",
            "absent",
            "--checkpoint-manifest-sha256",
            "a" * 64,
            "--model-dir",
            "fallback",
        ],
    ],
)
def test_hadamard_requires_explicit_derived_checkpoint_without_fallback(tmp_path, extra):
    with pytest.raises(ValueError):
        benchmark.configuration(arguments(tmp_path, "hadamard-fc2", *extra))


def test_source_choices_and_bf16_rejects_prepared_weights(tmp_path):
    _, source, _ = benchmark.configuration(arguments(tmp_path))
    assert source.name == "h3.c-hadamard"
    _, source, model = benchmark.configuration(
        arguments(
            tmp_path,
            "hadamard-fc2",
            "--checkpoint-dir",
            str(tmp_path / "derived"),
            "--checkpoint-manifest-sha256",
            "a" * 64,
        )
    )
    assert source.name == "h3.c-hadamard-cache"
    assert model == tmp_path / "derived/model"
    with pytest.raises(ValueError):
        benchmark.configuration(arguments(tmp_path, "bf16", "--checkpoint-dir", "wrong"))


def test_source_rejection_preserves_run_and_releases_lock(tmp_path, monkeypatch):
    states = []

    @contextmanager
    def lock(*_):
        states.append("acquired")
        try:
            yield
        finally:
            states.append("released")

    monkeypatch.setattr(benchmark.base, "wait_for_gpu_lock", lock)
    monkeypatch.setattr(benchmark.base, "git_provenance", lambda _: {})

    def reject(*_):
        raise ValueError("source guard rejected fixture")

    monkeypatch.setattr(benchmark.experiment, "validate_source", reject)
    result = benchmark.execute(arguments(tmp_path))
    assert result["status"] == "failed"
    assert "source guard" in result["error"]
    assert states == ["acquired", "released"]
    assert json.loads((tmp_path / "run/run.json").read_text()) == result
    with pytest.raises(FileExistsError):
        benchmark.execute(arguments(tmp_path))


def test_catalog_rejection_occurs_before_any_gpu_lock(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.benchmark_id = "unknown"

    def forbidden(*_):
        raise AssertionError("GPU lock must not be reached")

    monkeypatch.setattr(benchmark.base, "wait_for_gpu_lock", forbidden)
    result = benchmark.execute(args)
    assert result["status"] == "failed"
    assert "exactly one" in result["error"]
    assert "GPU lock" not in result["error"]
