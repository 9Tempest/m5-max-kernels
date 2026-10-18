"""Independent float64 reference for gated residual updates.

The ideal expression is residual + gate * branch. There is no implicit BF16
or FP32 rounding; a backend must explicitly apply its residual dtype boundary
before passing the updated residual to a normalization reference.
"""

import numpy as np


def _finite_real(value, name):
    value = np.asarray(value)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def gated_residual_reference(residual, branch, gate, *, row_map=None):
    """Return residual + gate * branch in float64, without mutating inputs.

    Residual and branch have identical nonempty shape [M,K]. A shared gate
    has shape [K] and does not accept a row map. A gate table has shape [P,K]
    and requires an int32 row_map[M] selecting one table entry for each row.
    All table values must be finite, including entries that are not selected.
    """
    residual = _finite_real(residual, "residual")
    branch = _finite_real(branch, "branch")
    gate = _finite_real(gate, "gate")
    if residual.ndim != 2 or not all(residual.shape):
        raise ValueError("residual must have nonempty shape [M,K]")
    if branch.shape != residual.shape:
        raise ValueError("branch must have the same shape as residual")
    rows, width = residual.shape
    if gate.ndim == 1:
        if gate.shape != (width,):
            raise ValueError("shared gate must have shape [K]")
        if row_map is not None:
            raise ValueError("row_map is only supported with a gate table")
        selected_gate = gate
    elif gate.ndim == 2:
        if gate.shape[0] == 0 or gate.shape[1] != width:
            raise ValueError("gate table must have nonempty shape [P,K]")
        if row_map is None:
            raise ValueError("gate table requires row_map")
        row_map = np.asarray(row_map)
        if row_map.shape != (rows,) or row_map.dtype != np.dtype(np.int32):
            raise ValueError("row_map must be an int32 vector of length M")
        if np.any(row_map < 0) or np.any(row_map >= gate.shape[0]):
            raise ValueError("row_map indices must be in [0,P)")
        selected_gate = gate[row_map]
    else:
        raise ValueError("gate must have shape [K] or [P,K]")
    with np.errstate(over="ignore", invalid="ignore"):
        result = residual + selected_gate * branch
    if not np.isfinite(result).all():
        raise ValueError("gated residual exceeds finite float64 range")
    return result
