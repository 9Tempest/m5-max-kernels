from argparse import Namespace
from m5kernels.cli import _with_hadamard, _sign_seed


def test_workload_transform_not_overwritten_by_unspecified_cli_options():
    case = dict(id="test", hadamard=dict(block_size=128, sign_seed=41, implementation="mlx"))
    assert _with_hadamard(case, Namespace()) == case
    overridden = _with_hadamard(case, Namespace(hadamard_block=256, hadamard_sign_seed=None))
    assert overridden["hadamard"] == dict(block_size=256, sign_seed=None, implementation="mlx")
    assert case["hadamard"]["block_size"] == 128
    assert _sign_seed("none") is None
    assert _sign_seed("0") == 0
