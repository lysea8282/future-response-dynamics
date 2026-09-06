"""Frozen local causal-validation direction bank and statistics."""
from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-14:
        raise RuntimeError("DEGENERATE_CAUSAL_DIRECTION")
    return np.asarray(v, dtype=np.float64) / n


def direction_bank(r: np.ndarray, phi: np.ndarray, anchor_decoder: np.ndarray, u4: np.ndarray, checkpoint: int, unit_id: str, route: str, r_cf: int) -> list[tuple[str, int, np.ndarray]]:
    v_r = np.linalg.svd(r, full_matrices=True)[2]
    v_phi = np.linalg.svd(phi, full_matrices=False)[2]
    v_dec = np.linalg.svd(anchor_decoder, full_matrices=False)[2]
    bank: list[tuple[str, int, np.ndarray]] = []
    for family, values in (("top_weighted_R", v_r[:4]), ("bottom_null_weighted_R", v_r[-4:]), ("top_Phi", v_phi[:4]), ("top_anchor_decoder", v_dec[:4]), ("Paper1_U4", u4.T)):
        bank.extend((family, index, _unit(value)) for index, value in enumerate(values))
    primary = v_r[:r_cf].T
    for index in range(32):
        literal = f"P2_FFD_FRESH_CAUSAL_V1|{checkpoint}|{unit_id}|{route}|{index}"
        seed = int.from_bytes(hashlib.sha256(literal.encode("utf-8")).digest()[:8], "little")
        random = np.random.default_rng(seed).normal(size=192)
        random = random - primary @ (primary.T @ random)
        bank.append(("random_orthogonal", index, _unit(random)))
    if len(bank) != 52:
        raise RuntimeError("CAUSAL_DIRECTION_BANK_CARDINALITY_FAILURE")
    return bank


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate_bank(future_map: Callable[[np.ndarray], np.ndarray], z: np.ndarray, r: np.ndarray, bank: list[tuple[str, int, np.ndarray]], epsilon: float) -> dict[str, object]:
    base = future_map(z).reshape(-1)
    rows = []
    for family, index, direction in bank:
        predicted_vector = r @ direction
        plus = future_map(z + epsilon * direction).reshape(-1)
        minus = future_map(z - epsilon * direction).reshape(-1)
        actual_vector = (plus - minus) / (2.0 * abs(epsilon))
        rows.append({
            "direction_family": family,
            "direction_index": index,
            "predicted_gain": float(np.linalg.norm(predicted_vector)),
            "actual_gain": float((np.linalg.norm(plus - base) + np.linalg.norm(minus - base)) / (2.0 * abs(epsilon))),
            "directional_cosine": float(actual_vector @ predicted_vector / (np.linalg.norm(actual_vector) * np.linalg.norm(predicted_vector) + 1e-30)),
        })
    rho = spearman(np.asarray([x["predicted_gain"] for x in rows]), np.asarray([x["actual_gain"] for x in rows]))
    return {"spearman": rho, "rows": rows}
