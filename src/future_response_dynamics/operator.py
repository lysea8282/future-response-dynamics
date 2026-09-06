"""Frozen finite-horizon response-operator mathematics.

This module has no path discovery, tuning, or checkpoint-selection behavior.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch


def response_operator(future_map: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor) -> np.ndarray:
    """Return d vec(future[t+1:t+9]) / dz with shape (64, 192)."""
    matrix = torch.func.jacrev(lambda q: future_map(q).reshape(-1))(z)
    out = matrix.detach().cpu().numpy().astype(np.float64, copy=False)
    if out.shape != (64, 192) or not np.isfinite(out).all():
        raise RuntimeError(f"INVALID_RESPONSE_OPERATOR:{out.shape}")
    return out


def r90_from_native_cf(r_cf: np.ndarray, energy: float = 0.90) -> int:
    s = np.linalg.svd(np.asarray(r_cf, dtype=np.float64), compute_uv=False)
    total = float(s @ s)
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError("NONPOSITIVE_NATIVE_CF_OPERATOR_ENERGY")
    return int(np.searchsorted(np.cumsum(s * s) / total, energy) + 1)


def right_subspace(r: np.ndarray, rank: int) -> np.ndarray:
    return np.linalg.svd(np.asarray(r, dtype=np.float64), full_matrices=False)[2][:rank]


def subspace_similarity(a: np.ndarray, cf: np.ndarray, rank: int) -> float:
    va, vc = right_subspace(a, rank), right_subspace(cf, rank)
    cosines = np.linalg.svd(va @ vc.T, compute_uv=False)
    return float(np.mean(np.clip(cosines, 0.0, 1.0) ** 2))


def operator_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (math.sqrt(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def compare_routes(r_f: np.ndarray, r_cf: np.ndarray, r_p: np.ndarray) -> dict[str, float | int]:
    rank = r90_from_native_cf(r_cf)
    s_f = subspace_similarity(r_f, r_cf, rank)
    s_p = subspace_similarity(r_p, r_cf, rank)
    d_f = operator_distance(r_f, r_cf)
    d_p = operator_distance(r_p, r_cf)
    return {
        "r_CF": rank,
        "S_sub_F_CF": s_f,
        "S_sub_P_CF": s_p,
        "Delta_align": s_p - s_f,
        "d_R_F_CF": d_f,
        "d_R_P_CF": d_p,
        "Delta_d": d_f - d_p,
    }


def contrastability(median_f_cf: float, q99_same_route: float, q99_reconstruction: float) -> dict[str, float | bool]:
    floor = max(20.0 * q99_same_route, 20.0 * q99_reconstruction, 1e-6)
    return {"numerical_floor": floor, "contrastable": bool(median_f_cf > floor)}
