"""Frozen prompt/seed catalog for the additional H3 paired five-second clips."""

import hashlib
import json
from pathlib import Path


DEFAULT = Path(__file__).with_name("benchmarks.json")
CATALOG_SHA256 = "a82e6544fc15c59beeca87a81e8e36f0d75ab2913a7eb3e4924ee62a5815d762"


def load_benchmark(benchmark_id, path=DEFAULT):
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CATALOG_SHA256:
        raise ValueError("benchmark catalog differs from the frozen version/hash")
    catalog = json.loads(payload)
    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported benchmark catalog schema")
    matches = [item for item in catalog["benchmarks"] if item["id"] == benchmark_id]
    if len(matches) != 1:
        raise ValueError("benchmark ID must identify exactly one frozen catalog entry")
    return (
        catalog["recipe"]
        | matches[0]
        | {"catalog_sha256": digest, "catalog_suite": catalog["suite"]}
    )
