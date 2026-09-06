"""Exact frozen scientific capture function; persistence orchestration omitted."""
from __future__ import annotations
from typing import Any, Callable
import numpy as np
from .causal_validation import spearman

def evaluate_bank_with_raw(
    future_map: Callable[[np.ndarray], np.ndarray],
    z: np.ndarray,
    r: np.ndarray,
    bank: list[tuple[str, int, np.ndarray]],
    epsilon: float,
) -> dict[str, Any]:
    """RC_V2-equivalent loop with observational capture of already-computed arrays."""
    base = future_map(z).reshape(-1)
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    pluses: list[np.ndarray] = []
    minuses: list[np.ndarray] = []
    predicted_vectors: list[np.ndarray] = []
    actual_vectors: list[np.ndarray] = []
    for family, family_index, direction in bank:
        predicted_vector = r @ direction
        plus = future_map(z + epsilon * direction).reshape(-1)
        minus = future_map(z - epsilon * direction).reshape(-1)
        actual_vector = (plus - minus) / (2.0 * abs(epsilon))
        rows.append({
            "direction_family": family,
            "direction_family_index": family_index,
            "predicted_gain": float(np.linalg.norm(predicted_vector)),
            "actual_gain": float((np.linalg.norm(plus - base) + np.linalg.norm(minus - base)) / (2.0 * abs(epsilon))),
            "directional_cosine": float(actual_vector @ predicted_vector / (np.linalg.norm(actual_vector) * np.linalg.norm(predicted_vector) + 1e-30)),
        })
        vectors.append(np.asarray(direction))
        bases.append(np.asarray(base))
        pluses.append(np.asarray(plus))
        minuses.append(np.asarray(minus))
        predicted_vectors.append(np.asarray(predicted_vector))
        actual_vectors.append(np.asarray(actual_vector))
    rho = spearman(
        np.asarray([row["predicted_gain"] for row in rows]),
        np.asarray([row["actual_gain"] for row in rows]),
    )
    return {
        "spearman": rho,
        "rows": rows,
        "raw": {
            "direction_vectors": np.stack(vectors),
            "base_future_vectors": np.stack(bases),
            "plus_future_vectors": np.stack(pluses),
            "minus_future_vectors": np.stack(minuses),
            "predicted_vectors": np.stack(predicted_vectors),
            "actual_vectors": np.stack(actual_vectors),
        },
    }
