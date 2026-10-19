"""Export and verify local prepared DiT INT8 weights; FC2 alone uses H256.

Original weights are read-only. The complete BF16 checkpoint remains available.
The new payload manifest is published only after all 200 projections are present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re
import struct
import sys
import time

import numpy as np

import run as base
import run_fc2_experiment as experiment


POLICY_ID = "h3_dit_int8_native_fc2_h256_v1"
FAMILIES = {
    "qkv": ("attn.qkv_proj.weight", 21504, 5376),
    "out": ("attn.out_proj.weight", 5376, 7168),
    "fc1": ("mlp.fc1.weight", 28672, 5376),
    "fc2": ("mlp.fc2.weight", 5376, 14336),
}


def is_sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def projection_spec(block, family):
    if type(block) is not int or not 0 <= block < 50 or family not in FAMILIES:
        raise ValueError("unknown block/projection family")
    suffix, rows, columns = FAMILIES[family]
    return {
        "block": block,
        "family": family,
        "tensor": f"blocks.{block}.{suffix}",
        "rows": rows,
        "columns": columns,
        "weight_transform": "normalized_plain_H256_FP32" if family == "fc2" else "identity",
        "quantization": "symmetric_per_output_row_RNE",
    }


def validate_manifest_schema(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("prepared manifest must be an object")
    if (
        manifest.get("schema") != 1
        or type(manifest.get("schema")) is not int
        or manifest.get("policy_id") != POLICY_ID
    ):
        raise ValueError("unknown prepared checkpoint schema/policy")
    model = manifest.get("source_model", {})
    producer = manifest.get("producer", {})
    if not isinstance(producer, dict):
        raise ValueError("prepared checkpoint producer must be an object")
    if model != {
        "repository": "MiniMaxAI/MiniMax-H3",
        "revision": base.MODEL_REVISION,
        "model_manifest_sha256": base.MODEL_MANIFEST_SHA256,
    }:
        raise ValueError("prepared checkpoint model provenance mismatch")
    if any(
        producer.get(k) != v
        for k, v in {
            "upstream_revision": base.UPSTREAM_REVISION,
            "source_sha256": experiment.SOURCE_SHA256,
            "patch_sha256": experiment.PATCH_SHA256,
        }.items()
    ):
        raise ValueError("prepared checkpoint producer provenance mismatch")
    entries = manifest.get("projections")
    if not isinstance(entries, list) or len(entries) != 200:
        raise ValueError("prepared checkpoint requires all 200 projections")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("prepared projection must be an object")
        family = tuple(FAMILIES)[index % 4]
        spec = projection_spec(index // 4, family)
        if any(type(entry.get(k)) is not type(v) or entry[k] != v for k, v in spec.items()):
            raise ValueError("projection identity/dimensions/transform mismatch")
        if not is_sha(entry.get("original_tensor_sha256")) or not is_sha(
            entry.get("original_source_file_sha256")
        ):
            raise ValueError("original tensor/shard digest missing")
        original_file = entry.get("original_source_file")
        if (
            type(original_file) is not str
            or Path(original_file).is_absolute()
            or ".." in Path(original_file).parts
            or Path(original_file).parent.as_posix() != "FL2VA/transformer"
            or not original_file.endswith(".safetensors")
        ):
            raise ValueError("original source path must be relative and contained")
        offsets = entry.get("original_data_offsets")
        file_offset = entry.get("original_file_offset")
        if (
            entry.get("original_dtype") != "BF16"
            or entry.get("original_shape") != [spec["rows"], spec["columns"]]
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(x) is not int for x in offsets)
            or offsets[0] < 0
            or offsets[1] - offsets[0] != spec["rows"] * spec["columns"] * 2
            or type(file_offset) is not int
            or file_offset < offsets[0] + 10
        ):
            raise ValueError("original BF16 tensor provenance mismatch")
        for key, suffix, size, dtype, layout in (
            ("codes", "codes.i8", spec["rows"] * spec["columns"], "int8", "N,K_row_major"),
            ("scales", "scales.f32", spec["rows"] * 4, "float32-le", "N"),
        ):
            payload = entry.get(key, {})
            if not isinstance(payload, dict):
                raise ValueError("prepared payload must be an object")
            expected = dict(
                file=f"block{spec['block']:02d}-{family}.{suffix}",
                bytes=size,
                dtype=dtype,
                layout=layout,
            )
            if any(
                type(payload.get(k)) is not type(v) or payload[k] != v for k, v in expected.items()
            ) or not is_sha(payload.get("sha256")):
                raise ValueError("prepared payload metadata mismatch")
    return manifest


def validate_payload(payload, directory, kind):
    if kind not in ("codes", "scales"):
        raise ValueError("unknown prepared payload kind")
    path = directory / payload["file"]
    if path.stat().st_size != payload["bytes"] or base.sha256(path) != payload["sha256"]:
        raise ValueError(f"prepared payload size/SHA256 mismatch: {path.name}")
    values = np.memmap(path, mode="r", dtype=np.int8 if kind == "codes" else "<f4")
    for start in range(0, len(values), 8 * 1024 * 1024):
        chunk = values[start : start + 8 * 1024 * 1024]
        if (kind == "codes" and np.any(chunk == -128)) or (
            kind == "scales" and (not np.isfinite(chunk).all() or np.any(chunk <= 0))
        ):
            raise base.NumericalError(f"invalid prepared numerical values: {path.name}")


def validate_payloads(manifest, directory):
    validate_manifest_schema(manifest)
    for entry in manifest["projections"]:
        for kind in ("codes", "scales"):
            validate_payload(entry[kind], directory, kind)
    return {
        "projections": 200,
        "payloads": 400,
        "verified_bytes": sum(
            e[k]["bytes"] for e in manifest["projections"] for k in ("codes", "scales")
        ),
    }


def original_tensor_inventory(model, model_record):
    file_hashes = {entry["path"]: entry["sha256"] for entry in model_record["files"]}
    wanted = {
        projection_spec(b, f)["tensor"]: projection_spec(b, f) for b in range(50) for f in FAMILIES
    }
    result = {}
    for path in sorted((model / "FL2VA/transformer").glob("*.safetensors")):
        with path.open("rb") as file:
            prefix = file.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated safetensors header")
            header_bytes = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_bytes <= 100 * 1024 * 1024:
                raise ValueError("invalid safetensors header size")
            header = json.loads(file.read(header_bytes))
        for name in wanted.keys() & header.keys():
            tensor, spec = header[name], wanted[name]
            begin, end = tensor["data_offsets"]
            if (
                name in result
                or tensor["dtype"] != "BF16"
                or tensor["shape"] != [spec["rows"], spec["columns"]]
                or end - begin != spec["rows"] * spec["columns"] * 2
                or not 0 <= begin < end <= path.stat().st_size - header_bytes - 8
            ):
                raise ValueError(f"bad original projection tensor: {name}")
            relative = path.relative_to(model).as_posix()
            result[name] = {
                "original_source_file": relative,
                "original_source_file_sha256": file_hashes[relative],
                "original_dtype": "BF16",
                "original_shape": tensor["shape"],
                "original_data_offsets": [begin, end],
                "original_file_offset": 8 + header_bytes + begin,
            }
    if result.keys() != wanted.keys():
        raise ValueError("original model does not contain exactly the required projection keys")
    return result


def assemble_manifest(weights, original, producer):
    manifest = {
        "schema": 1,
        "policy_id": POLICY_ID,
        "source_model": {
            "repository": "MiniMaxAI/MiniMax-H3",
            "revision": base.MODEL_REVISION,
            "model_manifest_sha256": base.MODEL_MANIFEST_SHA256,
        },
        "producer": producer,
        "projection_scope": "50 DiT blocks; QKV/out/FC1 native INT8, FC2 normalized plain H256 INT8; encoders/VAE and other tensors retain original precision",
        "projections": [],
    }
    timings = []
    for block in range(50):
        for family in FAMILIES:
            spec = projection_spec(block, family)
            stem = f"block{block:02d}-{family}"
            exported = experiment.load_json(weights / f"{stem}.export.json")
            if (
                any(
                    exported.get(k) != spec[k]
                    for k in ("block", "family", "tensor", "rows", "columns")
                )
                or exported.get("all_values_finite") is not True
            ):
                raise ValueError("exported projection metadata mismatch")
            entry = (
                spec
                | original[spec["tensor"]]
                | {"original_tensor_sha256": exported["original_tensor_sha256"]}
            )
            for kind, suffix, size, dtype, layout in (
                ("codes", "codes.i8", spec["rows"] * spec["columns"], "int8", "N,K_row_major"),
                ("scales", "scales.f32", spec["rows"] * 4, "float32-le", "N"),
            ):
                entry[kind] = {
                    "file": f"{stem}.{suffix}",
                    "bytes": size,
                    "sha256": exported[f"{kind}_sha256"],
                    "dtype": dtype,
                    "layout": layout,
                }
            manifest["projections"].append(entry)
            timings.append(
                {
                    "block": block,
                    "family": family,
                    **{
                        k: exported[k]
                        for k in (
                            "weight_load_seconds",
                            "original_hash_seconds",
                            "offline_preparation_seconds",
                            "readback_hash_write_seconds",
                        )
                    },
                }
            )
    validate_manifest_schema(manifest)
    return manifest, timings


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=base.ROOT / ".cache/models/MiniMax-H3")
    p.add_argument("--source-dir", type=Path, default=base.ROOT / ".cache/h3.c-hadamard")
    p.add_argument("--gpu-wait-seconds", type=base.gpu_wait_seconds, default=0)
    return p


def execute(args):
    output, source, model = (
        args.output_dir.resolve(),
        args.source_dir.resolve(),
        args.model_dir.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    weights = output / "weights"
    weights.mkdir()
    started = time.perf_counter()
    record = {
        "schema": 1,
        "status": "failed",
        "started_utc": base.utc_now(),
        "harness_sha256": base.sha256(__file__),
        "original_checkpoint_retained": True,
        "phase_timings": {},
        "measurements": {},
    }
    try:
        env, record["runtime_overrides"] = base.runtime_environment()
        with base.wait_for_gpu_lock(args.gpu_wait_seconds, record):
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise base.UnsupportedError(
                    "native prepared export requires Metal on Apple Silicon"
                )
            record["environment"] = dict(
                system=platform.system(),
                release=platform.release(),
                machine=platform.machine(),
                python=platform.python_version(),
                numpy=np.__version__,
                chip=base.read_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                memory_bytes=int(base.read_command(["sysctl", "-n", "hw.memsize"])),
                compiler=base.read_command(["clang", "--version"]),
                xcode=base.read_command(["xcodebuild", "-version"]),
            )
            if "M5" not in record["environment"]["chip"]:
                raise base.UnsupportedError("M5 TensorOps required")
            record["source"] = base.git_provenance(source)
            record["patch"] = experiment.validate_source(source, record["source"])
            preflight = time.perf_counter()
            record["model"] = base.model_inventory(
                model, Path(__file__).with_name("model-files.json")
            )
            original = original_tensor_inventory(model, record["model"])
            record["phase_timings"]["model_fullhash_preflight_seconds"] = (
                time.perf_counter() - preflight
            )
            build = base.run_logged(["make", "-B", "-j8"], source, env, output, "build")
            record["measurements"]["build"] = build
            if build["returncode"]:
                raise base.UnsupportedError("native producer compilation failed")
            driver = output / "prepared-export"
            c_source = Path(__file__).with_name("prepared_export.c")
            command = [
                "clang",
                "-O2",
                "-std=c11",
                "-D_DARWIN_C_SOURCE",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(source),
                str(c_source),
                str(source / "libh3.a"),
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                "-framework",
                "MetalPerformanceShaders",
                "-framework",
                "MetalPerformanceShadersGraph",
                "-framework",
                "Accelerate",
                "-licucore",
                "-lm",
                "-o",
                str(driver),
            ]
            built = base.run_logged(command, base.ROOT, env, output, "build-exporter")
            record["measurements"]["build_exporter"] = built
            if built["returncode"]:
                raise base.UnsupportedError("native exporter compilation failed")
            producer = {
                "upstream_revision": base.UPSTREAM_REVISION,
                "source_sha256": experiment.SOURCE_SHA256,
                "patch_sha256": experiment.PATCH_SHA256,
                "library_sha256": base.sha256(source / "libh3.a"),
                "exporter_source_sha256": base.sha256(c_source),
                "exporter_binary_sha256": base.sha256(driver),
            }
            record["producer"] = producer
            result = base.run_logged(
                [
                    str(driver),
                    str(model / "FL2VA/transformer"),
                    str(weights),
                    str(source / "h3_shaders.metal"),
                ],
                source,
                env,
                output,
                "export",
                True,
            )
            record["measurements"]["export"] = result
            if result["fallback_detected"]:
                raise base.UnsupportedError("TensorOps compilation fallback rejected")
            if result["returncode"]:
                raise ValueError("native prepared export failed; partial files retained")
            complete = experiment.load_json(weights / "export-complete.json")
            if (
                complete.get("first_block"),
                complete.get("blocks"),
                complete.get("projections"),
            ) != (0, 50, 200):
                raise ValueError("prepared export is incomplete")
            manifest, record["projection_timings"] = assemble_manifest(weights, original, producer)
            verification = time.perf_counter()
            record["payload_validation"] = validate_payloads(manifest, weights)
            record["phase_timings"]["payload_verification_seconds"] = (
                time.perf_counter() - verification
            )
            # The manifest is the commit point: partial data never masquerades
            # as a loadable derived checkpoint.
            base.write_json(weights / "manifest.json", manifest)
            record["prepared_manifest_sha256"] = base.sha256(weights / "manifest.json")
            record["prepared_manifest"] = base.path_label(weights / "manifest.json")
            record["phase_timings"].update(
                {
                    key: sum(x[key] for x in record["projection_timings"])
                    for key in (
                        "weight_load_seconds",
                        "original_hash_seconds",
                        "offline_preparation_seconds",
                        "readback_hash_write_seconds",
                    )
                }
            )
            record["phase_timings"]["reload_seconds"] = None
            record["reload_scope"] = (
                "not measured by exporter; measure native prepared-load run separately"
            )
        record["status"] = "completed"
    except KeyboardInterrupt:
        record.update(status="interrupted", error="interrupted by user; partial files retained")
    except Exception as error:
        record.update(
            status="incorrect"
            if isinstance(error, base.NumericalError)
            else "unsupported"
            if isinstance(error, base.UnsupportedError)
            else "failed",
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        record.update(finished_utc=base.utc_now(), total_wall_seconds=time.perf_counter() - started)
        base.write_json(output / "prepare.json", record)
    return record


def main():
    args = parser().parse_args()
    try:
        result = execute(args)
    except FileExistsError:
        print("choose a new immutable output directory", file=sys.stderr)
        return 2
    print(json.dumps({k: result.get(k) for k in ("status", "error", "prepared_manifest_sha256")}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
