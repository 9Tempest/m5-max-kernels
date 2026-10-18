"""CPU-only checks for trace export parsing and counter interpretation."""

import importlib.util
from pathlib import Path

import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gpudebug_mixed_json():
    parse = script("profile_report").objects
    assert parse('Session 1 created.\n{"a":1}\n{\n"b":2\n}') == [{"a": 1}, {"b": 2}]


def test_union_handles_nested_duplicate_and_disjoint_intervals():
    union = script("summarize_system_trace").union_ns
    assert union([(10, 20), (10, 20), (12, 15), (19, 25), (30, 40)]) == 25


def test_system_trace_filters_pid_and_resolves_references(tmp_path):
    summary = script("summarize_system_trace").summarize
    path = tmp_path / "trace.xml"
    columns = "start duration channel-name process".split()
    schema = "".join(f"<col><mnemonic>{c}</mnemonic></col>" for c in columns)
    path.write_text(
        f'<root><node><schema name="metal-gpu-intervals">{schema}</schema>'
        '<row><t>0</t><d>1000000</d><c id="1" fmt="Compute"/>'
        '<p id="2"><pid>123</pid></p></row>'
        '<row><t>500000</t><d>1000000</d><c ref="1"/><p ref="2"/></row>'
        '<row><t>0</t><d>999999999</d><c ref="1"/><p><pid>456</pid></p></row>'
        "</node></root>"
    )
    result = summary(path, 123)
    assert result["interval_count"] == 2
    assert result["gpu_active_union_ms"] == 1.5
    assert result["observed_busy_fraction"] == 1
    assert "123" not in str(result)
    with pytest.raises(ValueError, match="No GPU intervals"):
        summary(path, 789)
