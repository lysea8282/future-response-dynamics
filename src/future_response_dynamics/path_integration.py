"""Frozen intervention-dose path and integrated-Jacobian accounting."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .operator import operator_distance


GAMMA_R = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float64)
GAMMA_I = np.arange(33, dtype=np.float64) / 32.0


def composite_simpson(values: np.ndarray, subintervals: int) -> np.ndarray:
    if subintervals not in (8, 16, 32) or subintervals % 2:
        raise ValueError("SIMPSON_REQUIRES_FROZEN_EVEN_GRID")
    values = np.asarray(values, dtype=np.float64)
    stride = 32 // subintervals
    y = values[::stride]
    if len(y) != subintervals + 1:
        raise ValueError("INCOMPLETE_SIMPSON_GRID")
    return (y[0] + y[-1] + 4.0 * y[1:-1:2].sum(axis=0) + 2.0 * y[2:-1:2].sum(axis=0)) / (3.0 * subintervals)


def path_accounting(future_numpy: Callable[[np.ndarray], np.ndarray], operator_numpy: Callable[[np.ndarray], np.ndarray], z_f: np.ndarray, delta: np.ndarray) -> dict[str, object]:
    y_f = future_numpy(z_f).reshape(-1)
    y_p = future_numpy(z_f + delta).reshape(-1)
    actual = y_p - y_f
    operators = [operator_numpy(z_f + g * delta) for g in GAMMA_I]
    q = np.stack([r @ delta for r in operators], axis=0)
    i8, i16, i32 = (composite_simpson(q, n) for n in (8, 16, 32))
    denominator = np.linalg.norm(actual) + 1e-12
    r0, rmid, r1 = operators[0], operators[16], operators[32]
    chord = 0.5 * (r0 + r1)
    chord_deviation = float(np.linalg.norm(rmid - chord) / (np.linalg.norm(r1 - r0) + 1e-12))
    return {
        "E_local": float(np.linalg.norm(actual - r0 @ delta) / denominator),
        "E_int32": float(np.linalg.norm(actual - i32) / denominator),
        "I32_I16_relative": float(np.linalg.norm(i32 - i16) / denominator),
        "I16_I8_relative": float(np.linalg.norm(i16 - i8) / denominator),
        "C_R_midpoint": chord_deviation,
        "operator_grid": [operators[int(round(g * 32))] for g in GAMMA_R],
    }


def numerical_validity(e_int32_medians: list[float], convergence_medians: list[float]) -> bool:
    return bool(all(x < 0.01 for x in e_int32_medians) and all(x < 0.005 for x in convergence_medians))
