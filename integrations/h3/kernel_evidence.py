"""Read-only producer provenance checks shared by the new integration reports."""

from pathlib import Path

import kernel_integration as contract
import run as base
import run_fc2_experiment as capture


SNAPSHOT_FILES = {
    "run_kernel_integration.py",
    "kernel_integration.py",
    "kernel_session.c",
    "run.py",
    "run_fc2_experiment.py",
    "run_session.py",
    "model-files.json",
    "UPSTREAM-LICENSE",
    "locking.py",
    "adaln_capture.py",
    "fc2_analyze.py",
}
MODEL = dict(
    repository="MiniMaxAI/MiniMax-H3",
    revision=base.MODEL_REVISION,
    manifest_sha256=base.MODEL_MANIFEST_SHA256,
)


def local_file(directory, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("evidence must use a run-relative filename")
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("evidence leaves its immutable run directory")
    return path


def producer(directory, record, expected_pin):
    declaration = record.get("source_pin", {})
    contract.exact(declaration.get("file"), "source-pin/manifest.json", "source pin path")
    contract.exact(declaration.get("sha256"), expected_pin, "source pin digest")
    pin = contract.load_pin(local_file(directory, declaration["file"]), expected_pin)
    contract.exact(declaration.get("manifest"), pin, "source pin snapshot")
    for key, expected in (
        ("source_sha256", pin["source_sha256"]),
        ("revision", pin["upstream_revision"]),
    ):
        contract.exact(record.get("upstream", {}).get(key), expected, "native " + key)
    model = {key: record.get("model", {}).get(key) for key in MODEL}
    contract.exact(model, MODEL, "original model")
    entries = record.get("harness_snapshot", {})
    # AdaLN verifier is present only when diagnostic capture was requested.
    expected = (
        SNAPSHOT_FILES
        if record.get("capture_adaln")
        else SNAPSHOT_FILES - {"adaln_capture.py", "fc2_analyze.py"}
    )
    contract.exact(set(entries), expected, "complete harness snapshot")
    contract.exact(
        capture.load_json(local_file(directory, "harness-snapshot/manifest.json")),
        entries,
        "snapshot manifest",
    )
    for name, entry in entries.items():
        contract.exact(entry.get("file"), "harness-snapshot/" + name, "snapshot file")
        contract.exact(
            base.sha256(local_file(directory, entry["file"])),
            entry.get("sha256"),
            "snapshot digest",
        )
    contract.exact(
        entries["model-files.json"]["sha256"], MODEL["manifest_sha256"], "model inventory snapshot"
    )
    driver = record.get("driver", {})
    name = "native-cli" if record.get("kind") == "capture" else "kernel-session"
    contract.exact(driver.get("file"), name, "retained executable")
    contract.exact(
        base.sha256(local_file(directory, name)),
        driver.get("binary_sha256"),
        "retained binary digest",
    )
    if not contract.digest(driver.get("library_sha256")):
        raise ValueError("missing native library digest")
    return dict(
        source_pin_sha256=expected_pin,
        source_sha256=pin["source_sha256"],
        model=model,
        harness={name: entry["sha256"] for name, entry in entries.items()},
        driver=driver,
    )
