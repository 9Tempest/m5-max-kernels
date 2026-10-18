"""CPU-only profiling adapters: native handles never enter MLX evaluation/capture."""

from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from m5kernels import profile


class NativeOutput:
    def __init__(self, finite=True):
        self.evaluations = 0
        self.finite = finite

    def __m5_evaluate__(self):
        self.evaluations += 1

    def __m5_all_finite__(self):
        return self.finite


class NativeCall:
    def __init__(self):
        self.output = NativeOutput()
        self.calls = 0
        self.syncs = 0
        self.captures = []

    def __call__(self):
        self.calls += 1
        return self.output

    def synchronize(self):
        self.syncs += 1

    def capture(self, path, iterations=1):
        if iterations != 1:
            raise NotImplementedError("native fixture supports one invocation")
        self.captures.append((path, iterations))


@pytest.fixture
def no_mlx(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)


def args(tmp_path, **updates):
    result = dict(
        suite="fixture",
        case="grouped.fixture",
        backend="native-qkv",
        precision="int8",
        seed=17,
        tile=[64, 128, 4, 8],
        online=False,
        kind="gpu",
        worker=False,
        output=str(tmp_path / "native.gputrace"),
        iterations=1,
        seconds=1,
    )
    result.update(updates)
    return SimpleNamespace(**result)


@pytest.mark.parametrize("online", [False, True])
def test_workload_routes_grouped_qkv_and_evaluates_native_protocol(
    no_mlx, monkeypatch, tmp_path, online
):
    core, full = NativeCall(), NativeCall()
    case = {"id": "grouped.fixture", "op": "grouped_qkv"}
    prepared = []

    def prepare(*values):
        prepared.append(values)
        return (
            core,
            lambda output: np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            {"online_activation_plus_gemm": full},
            {"transform_pass": True},
        )

    monkeypatch.setattr(profile, "load", lambda _: {"cases": [case]})
    monkeypatch.setattr(profile, "prepare_grouped_qkv", prepare)
    function, metadata = profile.workload(args(tmp_path, online=online))
    selected, unused = (full, core) if online else (core, full)
    assert function is selected
    assert selected.calls == selected.output.evaluations == 6
    assert unused.calls == 0
    expected_case = dict(case, k_specialization="dynamic", scratch="threadgroup", rms_order="simd")
    assert prepared == [(expected_case, "native-qkv", "int8", 17, "normal", (64, 128, 4, 8))]
    assert "scratch" not in case  # Preparing a capture does not mutate the loaded suite.
    assert metadata["implementation_error"]["relative_l2"] == 0
    assert metadata["online"] is online


@pytest.mark.parametrize("operation", ["grouped_qkv", "gemm"])
def test_qkv_profile_overrides_apply_only_to_grouped_cases(
    no_mlx, monkeypatch, tmp_path, operation
):
    function = NativeCall()
    case = {"id": "grouped.fixture", "op": operation}
    received = []

    def prepare(prepared_case, *_):
        received.append(prepared_case)
        return function, lambda _: np.ones(1), np.ones(1), np.ones(1), {}, {}

    monkeypatch.setattr(profile, "load", lambda _: {"cases": [case]})
    monkeypatch.setattr(profile, "prepare_grouped_qkv", prepare)
    monkeypatch.setattr(profile, "prepare_gemm", prepare)
    _, metadata = profile.workload(
        args(tmp_path, qkv_k_specialization="native", qkv_scratch="device", qkv_rms_order="native")
    )
    expected = dict(case)
    if operation == "grouped_qkv":
        expected.update(k_specialization="native", scratch="device", rms_order="native")
    assert received == [expected]
    assert metadata["case"] == expected
    assert case == {"id": "grouped.fixture", "op": operation}


def test_unset_qkv_flags_preserve_workload_configuration(no_mlx, monkeypatch, tmp_path):
    function = NativeCall()
    case = dict(
        id="grouped.fixture",
        op="grouped_qkv",
        k_specialization="native",
        scratch="device",
        rms_order="native",
    )
    monkeypatch.setattr(profile, "load", lambda _: {"cases": [case]})
    monkeypatch.setattr(
        profile,
        "prepare_grouped_qkv",
        lambda *_: (function, lambda _: np.ones(1), np.ones(1), np.ones(1), {}, {}),
    )
    _, metadata = profile.workload(args(tmp_path))
    assert metadata["case"] == case


def test_qkv_cli_flags_parse_without_gpu(monkeypatch, tmp_path):
    received = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "m5kernels.profile",
            "--case",
            "grouped.fixture",
            "--output",
            str(tmp_path / "trace"),
            "--qkv-k-specialization",
            "native",
            "--qkv-scratch",
            "device",
            "--qkv-rms-order",
            "native",
        ],
    )
    monkeypatch.setattr(profile, "run", lambda options: received.append(options) or 0)
    assert profile.main() == 0
    assert received[0].qkv_k_specialization == "native"
    assert received[0].qkv_scratch == "device"
    assert received[0].qkv_rms_order == "native"


@pytest.mark.parametrize("failure", ["nonfinite", "transform", "reference"])
def test_workload_preserves_correctness_gate(no_mlx, monkeypatch, tmp_path, failure):
    function = NativeCall()
    function.output.finite = failure != "nonfinite"
    monkeypatch.setattr(
        profile, "load", lambda _: {"cases": [{"id": "grouped.fixture", "op": "grouped_qkv"}]}
    )
    monkeypatch.setattr(
        profile,
        "prepare_grouped_qkv",
        lambda *_: (
            function,
            lambda _: np.array([10.0 if failure == "reference" else 1.0]),
            np.array([1.0]),
            np.array([1.0]),
            {},
            {"transform_pass": failure != "transform"},
        ),
    )
    with pytest.raises(ValueError, match="incorrect kernel"):
        profile.workload(args(tmp_path))


def mock_run_environment(monkeypatch, function):
    records = []
    monkeypatch.setattr(profile, "gpu_lock", nullcontext)
    monkeypatch.setattr(profile, "workload", lambda _: (function, {"fixture": True}))
    monkeypatch.setattr(profile, "environment", lambda: {"device": "CPU fixture"})
    monkeypatch.setattr(profile, "write_result", lambda path, doc: records.append((path, doc)))
    return records


def test_native_gpu_capture_uses_common_runner_without_mlx(no_mlx, monkeypatch, tmp_path):
    function = NativeCall()
    records = mock_run_environment(monkeypatch, function)
    options = args(tmp_path)
    assert profile.run(options) == 0
    assert function.captures == [(tmp_path / "native.gputrace", 1)]
    assert function.calls == 0  # capture method dispatches its own complete invocation
    assert function.syncs == 2
    assert records[0][0] == options.output + ".json"
    assert records[0][1]["iterations"] == 1
    assert records[0][1]["not_a_benchmark_result"] is True


def test_native_unsupported_capture_does_not_fall_back(no_mlx, monkeypatch, tmp_path):
    function = NativeCall()
    records = mock_run_environment(monkeypatch, function)
    with pytest.raises(NotImplementedError, match="one invocation"):
        profile.run(args(tmp_path, iterations=2))
    assert function.calls == 0 and function.captures == []
    assert records == []


def test_native_capture_refuses_existing_path(no_mlx, monkeypatch, tmp_path):
    function = NativeCall()
    records = mock_run_environment(monkeypatch, function)
    (tmp_path / "native.gputrace").mkdir()
    with pytest.raises(FileExistsError):
        profile.run(args(tmp_path))
    assert function.captures == [] and records == []


def test_native_system_worker_loop_uses_native_evaluate_and_sync(no_mlx, monkeypatch, tmp_path):
    function = NativeCall()
    records = mock_run_environment(monkeypatch, function)
    times = iter([0, 0.1, 0.9, 1.1, 1.2])
    monkeypatch.setattr(profile.time, "monotonic", lambda: next(times))
    assert profile.run(args(tmp_path, kind="system", worker=True)) == 0
    assert function.calls == function.output.evaluations == 2
    assert function.syncs == 1
    assert function.captures == []
    assert records[0][1]["iterations"] == 2
    assert records[0][1]["hot_loop_seconds"] == 1.2


@pytest.fixture
def mlx_stub(monkeypatch):
    calls = []
    core = ModuleType("mlx.core")
    marker_output = object()

    def metal_kernel(**kwargs):
        assert kwargs["name"] == "capture_boundary"
        calls.append("marker_create")

        def marker(**launch):
            assert launch["grid"] == (1, 1, 1)
            calls.append("marker_run")
            return marker_output

        return marker

    core.eval = lambda output: calls.append(("eval", output))
    core.synchronize = lambda: calls.append("sync")
    core.int32 = "int32"
    core.fast = SimpleNamespace(metal_kernel=metal_kernel)
    core.metal = SimpleNamespace(
        start_capture=lambda path: calls.append(("start", path)),
        stop_capture=lambda: calls.append("stop"),
    )
    package = ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return calls, marker_output


def test_mlx_capture_keeps_boundary_marker_and_all_iterations(mlx_stub, tmp_path):
    calls, marker = mlx_stub
    output = object()
    profile._capture(lambda: output, tmp_path / "mlx.gputrace", 2)
    assert calls == [
        "sync",
        ("start", str(tmp_path / "mlx.gputrace")),
        "marker_create",
        "marker_run",
        ("eval", marker),
        "sync",
        ("eval", output),
        ("eval", output),
        "sync",
        "stop",
    ]


def test_mlx_capture_stops_after_workload_failure(mlx_stub, tmp_path):
    calls, _ = mlx_stub

    def broken():
        raise RuntimeError("fixture failure")

    with pytest.raises(RuntimeError, match="fixture failure"):
        profile._capture(broken, tmp_path / "mlx.gputrace", 1)
    assert calls[-1] == "stop"
