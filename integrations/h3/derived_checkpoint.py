"""Build a separate loadable H3 model tree around verified prepared projections.

The original BF16 checkpoint is retained. Unchanged files are hardlinked when
possible; changed safetensors shards are streamed into new files. No source file
is modified. The final manifest is written only after complete verification.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import time

import prepared_checkpoint as prepared
import run as base
import run_fc2_experiment as experiment


DERIVED_SCHEMA = "h3_prepared_model_v1"
DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
}
CHUNK_BYTES = 8 * 1024 * 1024


def contained_file(root, name):
    if (
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
    ):
        raise ValueError("checkpoint paths must be nonempty and relative")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("checkpoint path escapes its model directory")
    return path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate safetensors JSON key")
        result[key] = value
    return result


def read_header(file):
    prefix = file.read(8)
    if len(prefix) != 8:
        raise ValueError("truncated safetensors length")
    size = struct.unpack("<Q", prefix)[0]
    if not 2 <= size <= 100 * 1024 * 1024:
        raise ValueError("invalid safetensors header length")
    encoded = file.read(size)
    if len(encoded) != size:
        raise ValueError("truncated safetensors header")
    header = json.loads(encoded, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    tensors = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict) or tensor.get("dtype") not in DTYPE_BYTES:
            raise ValueError("unsupported tensor dtype/header")
        shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
        if (
            not isinstance(shape, list)
            or any(type(v) is not int or v < 0 for v in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(v) is not int or v < 0 for v in offsets)
            or offsets[1] - offsets[0] != math.prod(shape) * DTYPE_BYTES[tensor["dtype"]]
        ):
            raise ValueError("invalid tensor shape/offsets")
        tensors.append((name, tensor))
    tensors.sort(key=lambda item: (item[1]["data_offsets"][0], item[1]["data_offsets"][1]))
    position = 0
    for _, tensor in tensors:
        begin, end = tensor["data_offsets"]
        if begin != position:
            raise ValueError("safetensors payload must be contiguous and nonoverlapping")
        position = end
    if os.fstat(file.fileno()).st_size != 8 + size + position:
        raise ValueError("safetensors payload size mismatch")
    return header, tensors, prefix + encoded


def rewrite_shard(source, target, removed, expected_source_sha256):
    """Copy kept tensor bytes verbatim, hash removed tensors against the export."""
    with source.open("rb") as input_file:
        header, tensors, original_prefix = read_header(input_file)
        present = {name for name, _ in tensors}
        if not removed.keys() <= present:
            raise ValueError("prepared tensor missing from original shard")
        rewritten = {"__metadata__": header["__metadata__"]} if "__metadata__" in header else {}
        position = 0
        for name, tensor in tensors:
            if name not in removed:
                length = tensor["data_offsets"][1] - tensor["data_offsets"][0]
                rewritten[name] = dict(tensor, data_offsets=[position, position + length])
                position += length
        encoded = json.dumps(rewritten, separators=(",", ":"), ensure_ascii=False).encode()
        encoded += b" " * (-len(encoded) % 8)
        new_prefix = struct.pack("<Q", len(encoded)) + encoded
        old_hash, new_hash = hashlib.sha256(original_prefix), hashlib.sha256(new_prefix)
        retained_records, removed_records = [], []
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output_file:
            output_file.write(new_prefix)
            for name, tensor in tensors:
                begin, end = tensor["data_offsets"]
                remaining, tensor_hash = end - begin, hashlib.sha256()
                while remaining:
                    chunk = input_file.read(min(remaining, CHUNK_BYTES))
                    if not chunk:
                        raise ValueError("truncated source tensor")
                    remaining -= len(chunk)
                    old_hash.update(chunk)
                    tensor_hash.update(chunk)
                    if name not in removed:
                        output_file.write(chunk)
                        new_hash.update(chunk)
                tensor_sha = tensor_hash.hexdigest()
                item = dict(
                    tensor=name,
                    dtype=tensor["dtype"],
                    shape=tensor["shape"],
                    original_data_offsets=[begin, end],
                    tensor_sha256=tensor_sha,
                )
                if name in removed:
                    if tensor_sha != removed[name]:
                        raise ValueError(f"export/original tensor SHA256 mismatch: {name}")
                    removed_records.append(item)
                else:
                    item["derived_data_offsets"] = rewritten[name]["data_offsets"]
                    retained_records.append(item)
        if old_hash.hexdigest() != expected_source_sha256:
            raise ValueError("original shard SHA256 changed during rewrite")
        if base.sha256(target) != new_hash.hexdigest():
            raise ValueError("rewritten shard readback SHA256 mismatch")
    return dict(
        storage="streamed_rewrite",
        sha256=new_hash.hexdigest(),
        bytes=target.stat().st_size,
        original_sha256=old_hash.hexdigest(),
        original_header_bytes=len(original_prefix),
        derived_header_bytes=len(new_prefix),
        retained_tensors=retained_records,
        removed_tensors=removed_records,
    )


def preserve_file(source, target, expected_sha256):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        storage = "hardlink"
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
            raise
        with source.open("rb") as input_file, target.open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file, CHUNK_BYTES)
        storage = "copy"
    digest = base.sha256(target)
    if digest != expected_sha256:
        raise ValueError("unchanged source/file SHA256 mismatch")
    return dict(storage=storage, sha256=digest, original_sha256=digest, bytes=target.stat().st_size)


def verify_original_file(path, specification):
    size = path.stat().st_size
    if size != specification["size"]:
        raise ValueError("original pinned file size mismatch")
    digest = base.sha256(path)
    expected = specification.get("lfs", {}).get("oid")
    if expected:
        if digest != expected:
            raise ValueError("original pinned file SHA256 mismatch")
    else:
        git_hash = hashlib.sha1(f"blob {size}\0".encode() + path.read_bytes()).hexdigest()
        if git_hash != specification["oid"]:
            raise ValueError("original pinned file Git blob mismatch")
    return digest


def build_model(original, checkpoint, prepared_manifest):
    prepared.validate_manifest_schema(prepared_manifest)
    output = checkpoint / "model"
    output.mkdir(exist_ok=False)
    lookup_path = Path(__file__).with_name("model-files.json")
    if base.sha256(lookup_path) != base.MODEL_MANIFEST_SHA256:
        raise ValueError("original model inventory pin mismatch")
    lookup = json.loads(lookup_path.read_text())
    missing = set(prepared_manifest["projections"][i]["original_source_file"] for i in range(200))
    removed_by_file = {}
    for entry in prepared_manifest["projections"]:
        removed_by_file.setdefault(entry["original_source_file"], {})[entry["tensor"]] = entry[
            "original_tensor_sha256"
        ]
    records = []
    for specification in sorted(lookup, key=lambda x: x["path"]):
        if specification.get("type") != "file":
            continue
        name = specification["path"]
        source = contained_file(original, name)
        if not source.is_file():
            continue  # Preserve the existing downloaded inference subset.
        target = contained_file(output, name)
        expected = specification.get("lfs", {}).get("oid")
        if name in removed_by_file:
            if not expected:
                raise ValueError("projection shard must have pinned LFS SHA256")
            for entry in prepared_manifest["projections"]:
                if (
                    entry["original_source_file"] == name
                    and entry["original_source_file_sha256"] != expected
                ):
                    raise ValueError("prepared/source shard provenance differs")
            record = rewrite_shard(source, target, removed_by_file[name], expected)
            missing.remove(name)
        elif name == "FL2VA/transformer/model.safetensors.index.json":
            original_hash = verify_original_file(source, specification)
            index = experiment.load_json(source)
            weights = index["weight_map"]
            removed_names = {entry["tensor"] for entry in prepared_manifest["projections"]}
            if not removed_names <= weights.keys():
                raise ValueError("original transformer index lacks prepared tensors")
            index["weight_map"] = {k: v for k, v in weights.items() if k not in removed_names}
            if "total_size" in index.get("metadata", {}):
                index["metadata"]["total_size"] -= sum(
                    e["rows"] * e["columns"] * 2 for e in prepared_manifest["projections"]
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            base.write_json(target, index)
            record = dict(
                storage="rewritten_index",
                bytes=target.stat().st_size,
                sha256=base.sha256(target),
                original_sha256=original_hash,
                removed_keys=sorted(removed_names),
            )
        else:
            # Checking source plus target binds unchanged hardlinks/copies to the
            # pinned original, even for small Git-blob metadata files.
            digest = verify_original_file(source, specification)
            record = preserve_file(source, target, digest)
        records.append(dict(path=name, **record))
    if missing or sum(len(r.get("removed_tensors", [])) for r in records) != 200:
        raise ValueError("derived model did not remove exactly all prepared projection tensors")
    return records


def verify_derived(checkpoint, expected_manifest_sha256):
    manifest_path = checkpoint / "derived-checkpoint.json"
    if (
        not prepared.is_sha(expected_manifest_sha256)
        or base.sha256(manifest_path) != expected_manifest_sha256
    ):
        raise ValueError("derived checkpoint manifest SHA256 mismatch")
    manifest = experiment.load_json(manifest_path)
    return verify_contents(checkpoint, manifest) | {"manifest_sha256": expected_manifest_sha256}


def verify_contents(checkpoint, manifest):
    if manifest.get("schema") != DERIVED_SCHEMA or manifest.get("policy_id") != prepared.POLICY_ID:
        raise ValueError("unknown derived checkpoint schema/policy")
    if (
        manifest.get("original_checkpoint_retained") is not True
        or manifest.get("model_directory") != "model"
    ):
        raise ValueError("derived checkpoint preservation/layout contract mismatch")
    weights = checkpoint / "weights"
    if base.sha256(weights / "manifest.json") != manifest.get("prepared_manifest_sha256"):
        raise ValueError("prepared manifest differs from derived manifest")
    prepared_manifest = experiment.load_json(weights / "manifest.json")
    if any(manifest.get(key) != prepared_manifest.get(key) for key in ("source_model", "producer")):
        raise ValueError("derived and prepared source provenance differs")
    payloads = prepared.validate_payloads(prepared_manifest, weights)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("derived file inventory missing")
    model = checkpoint / "model"
    expected_paths = {record["path"] for record in files}
    if len(expected_paths) != len(files) or expected_paths != {
        p.relative_to(model).as_posix() for p in model.rglob("*") if p.is_file()
    }:
        raise ValueError("derived model file inventory differs")
    for record in files:
        path = contained_file(model, record["path"])
        if path.stat().st_size != record["bytes"] or base.sha256(path) != record["sha256"]:
            raise ValueError(f"derived model payload SHA256 mismatch: {record['path']}")
    return dict(
        prepared_manifest_sha256=manifest["prepared_manifest_sha256"],
        files=len(files),
        payloads=payloads,
        verification="full files and prepared payload hashes",
        model_directory=str(model.resolve()),
    )


def execute(args):
    checkpoint, original = args.checkpoint_dir.resolve(), args.model_dir.resolve()
    if checkpoint == original or checkpoint.is_relative_to(original):
        raise ValueError("derived checkpoint must be separate from original")
    record = dict(
        schema=1,
        status="failed",
        started_utc=base.utc_now(),
        harness_sha256=base.sha256(__file__),
        original_checkpoint_retained=True,
    )
    started = time.perf_counter()
    try:
        with base.wait_for_gpu_lock(args.gpu_wait_seconds, record):
            export_record = experiment.load_json(checkpoint / "prepare.json")
            prepared_path = checkpoint / "weights/manifest.json"
            if export_record.get("status") != "completed" or base.sha256(
                prepared_path
            ) != export_record.get("prepared_manifest_sha256"):
                raise ValueError("successful immutable prepared export is required")
            manifest = experiment.load_json(prepared_path)
            prepared.validate_payloads(manifest, prepared_path.parent)
            # Enforce the native model's required components before copying the
            # complete locally downloaded subset into a separate tree.
            record["original_model"] = base.model_inventory(
                original, Path(__file__).with_name("model-files.json")
            )
            files = build_model(original, checkpoint, manifest)
            derived = dict(
                schema=DERIVED_SCHEMA,
                policy_id=prepared.POLICY_ID,
                model_directory="model",
                original_checkpoint_retained=True,
                source_model=manifest["source_model"],
                producer=manifest["producer"],
                prepared_manifest="weights/manifest.json",
                prepared_manifest_sha256=base.sha256(prepared_path),
                builder_sha256=base.sha256(__file__),
                files=files,
                precision="QKV/out/FC1 native INT8, FC2 H256 INT8; other tensors unchanged",
            )
            record["verification"] = verify_contents(checkpoint, derived)
            # Publish only after all payloads and rewritten files pass readback.
            base.write_json(checkpoint / "derived-checkpoint.json", derived)
            record["derived_manifest_sha256"] = base.sha256(checkpoint / "derived-checkpoint.json")
            record["status"] = "completed"
    except Exception as error:
        record.update(error=f"{type(error).__name__}: {error}")
    finally:
        record.update(finished_utc=base.utc_now(), total_wall_seconds=time.perf_counter() - started)
        base.write_json(checkpoint / "derive.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=base.ROOT / ".cache/models/MiniMax-H3")
    parser.add_argument("--gpu-wait-seconds", type=base.gpu_wait_seconds, default=0)
    args = parser.parse_args()
    try:
        record = execute(args)
    except FileExistsError:
        print(
            "choose an export checkpoint without an existing model/derive result", file=sys.stderr
        )
        return 2
    print(
        json.dumps({key: record.get(key) for key in ("status", "error", "derived_manifest_sha256")})
    )
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
