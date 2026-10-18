"""Fetch tests use local temporary Git repositories; no network or GPU calls."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from m5kernels import native_qkv


@pytest.fixture
def helper():
    path = Path(__file__).resolve().parents[1] / "scripts/fetch_native_qkv.py"
    spec = importlib.util.spec_from_file_location("fetch_native_qkv_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*arguments):
    return (
        subprocess.check_output(["git", *map(str, arguments)], stderr=subprocess.PIPE)
        .decode()
        .strip()
    )


@pytest.fixture
def local_source(helper, monkeypatch, tmp_path):
    origin = tmp_path / "upstream"
    origin.mkdir()
    git("init", "-q", origin)
    for name in ("h3_shaders.metal", "h3_gpu.m", "LICENSE"):
        (origin / name).write_text(f"local CPU fixture: {name}\n")
    git("-C", origin, "add", ".")
    git(
        "-C",
        origin,
        "-c",
        "user.name=CPU fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-qm",
        "fixture",
    )
    revision = git("-C", origin, "rev-parse", "HEAD")
    for module in (helper, native_qkv):
        monkeypatch.setattr(module, "UPSTREAM_COMMIT", revision)
        monkeypatch.setattr(module, "UPSTREAM_URL", str(origin))
    return origin, revision


def test_new_checkout_is_detached_pinned_and_verified(helper, local_source, tmp_path):
    origin, revision = local_source
    destination = tmp_path / "cache" / "h3.c"
    result = helper.fetch(destination)
    assert result["created"] is True
    assert result["provenance"]["commit"] == revision
    assert set(result["provenance"]["source_sha256"]) == {"h3_shaders.metal", "h3_gpu.m", "LICENSE"}
    assert git("-C", destination, "rev-parse", "HEAD") == revision
    assert git("-C", destination, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    assert git("-C", destination, "remote", "get-url", "origin") == str(origin)
    assert (destination / "h3_shaders.metal").read_bytes() == (
        origin / "h3_shaders.metal"
    ).read_bytes()


def snapshot(path):
    return {
        str(file.relative_to(path)): (file.read_bytes(), file.stat().st_mtime_ns)
        for file in path.rglob("*")
        if file.is_file()
    }


def test_existing_checkout_is_read_only_and_keeps_untracked_files(
    helper, local_source, monkeypatch, tmp_path
):
    destination = tmp_path / "existing"
    helper.fetch(destination)
    (destination / "local-notes.txt").write_text("keep my work")
    before = snapshot(destination)

    def forbidden(*args, **kwargs):
        raise AssertionError("existing checkout must not run mutating subprocess commands")

    # verify_upstream uses check_output (which internally calls run), so leave
    # that read-only verifier real while intercepting only clone/checkout calls.
    original_run = subprocess.run

    def read_only(command, *args, **kwargs):
        if "clone" in command or "checkout" in command or "fetch" in command:
            forbidden()
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(helper.subprocess, "run", read_only)
    result = helper.fetch(destination)
    assert result["created"] is False
    assert snapshot(destination) == before


def test_existing_modified_required_source_fails_without_changes(helper, local_source, tmp_path):
    destination = tmp_path / "modified"
    helper.fetch(destination)
    (destination / "h3_shaders.metal").write_text("local kernel edits")
    before = snapshot(destination)
    with pytest.raises(ValueError, match="local modifications"):
        helper.fetch(destination)
    assert snapshot(destination) == before


def test_existing_wrong_revision_fails_without_checkout(
    helper, local_source, monkeypatch, tmp_path
):
    destination = tmp_path / "wrong-revision"
    helper.fetch(destination)
    before = snapshot(destination)
    monkeypatch.setattr(native_qkv, "UPSTREAM_COMMIT", "0" * 40)
    with pytest.raises(ValueError, match="HEAD must be"):
        helper.fetch(destination)
    assert snapshot(destination) == before


@pytest.mark.parametrize("kind", ["empty", "file", "broken_symlink"])
def test_existing_non_checkout_path_is_preserved(helper, tmp_path, kind):
    target = tmp_path / "existing"
    if kind == "empty":
        target.mkdir()
    elif kind == "file":
        target.write_text("keep me")
    else:
        target.symlink_to(tmp_path / "absent")
    with pytest.raises(NotImplementedError):
        helper.fetch(target)
    if kind == "empty":
        assert target.is_dir() and list(target.iterdir()) == []
    elif kind == "file":
        assert target.read_text() == "keep me"
    else:
        assert target.is_symlink()


@pytest.mark.parametrize("fail_stage", ["clone", "checkout", "verify"])
def test_failed_new_checkout_cleans_only_its_new_destination(
    helper, monkeypatch, tmp_path, fail_stage
):
    target = tmp_path / "new"
    sibling = tmp_path / "keep.txt"
    sibling.write_text("untouched")
    calls = []

    def run(command, *, check, env):
        calls.append(command)
        assert check is True
        assert env["GIT_LFS_SKIP_SMUDGE"] == "1"
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        if "clone" in command:
            assert "--no-checkout" in command and "--no-recurse-submodules" in command
            assert command[-2] == helper.UPSTREAM_URL
        else:
            assert command[-2:] == ["--detach", helper.UPSTREAM_COMMIT]
        (target / "partial").write_text("new clone only")
        if fail_stage in command:
            raise subprocess.CalledProcessError(1, command)

    def verify(_):
        raise ValueError("fixture verification failure")

    monkeypatch.setattr(helper.subprocess, "run", run)
    monkeypatch.setattr(helper, "verify_upstream", verify)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        helper.fetch(target)
    assert not target.exists()
    assert sibling.read_text() == "untouched"
    assert len(calls) == (1 if fail_stage == "clone" else 2)


def test_cli_emits_source_provenance(helper, local_source, tmp_path, capsys):
    target = tmp_path / "cli-checkout"
    assert helper.main(["--destination", str(target)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["destination"] == str(target)
    assert result["provenance"]["license"].startswith("MIT")


def test_default_destination_is_repository_cache(helper, monkeypatch, tmp_path):
    target = tmp_path / ".cache" / "h3.c"
    target.mkdir(parents=True)
    monkeypatch.setattr(helper, "REPOSITORY_ROOT", tmp_path)
    visited = []
    monkeypatch.setattr(helper, "verify_upstream", lambda path: visited.append(path) or {})
    assert helper.fetch()["created"] is False
    assert visited == [target]
