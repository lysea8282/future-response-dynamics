"""Checkpoint-specific uncentered local-edit delta carrier utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


CARRIER_DIM = 384


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest().upper()


def _canonicalize_column_signs(basis: np.ndarray) -> np.ndarray:
    result = np.asarray(basis, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


@dataclass(frozen=True)
class CarrierFit:
    basis: np.ndarray
    singular_values: np.ndarray
    explained_second_moment_ratio: np.ndarray
    unit_count: int
    carrier_dim: int
    max_rank: int
    basis_sha256: str
    singular_values_sha256: str
    orthonormality_max_abs_error: float

    def summary(self) -> dict[str, object]:
        return {
            "family": "LOCAL_EDIT_DELTA_PCA",
            "basis_preprocessing": "UNCENTERED_RAW_DELTA_H_SECOND_MOMENT_SVD",
            "centering_performed": False,
            "unit_count": self.unit_count,
            "carrier_dim": self.carrier_dim,
            "max_rank": self.max_rank,
            "basis_sha256": self.basis_sha256,
            "singular_values_sha256": self.singular_values_sha256,
            "orthonormality_max_abs_error": self.orthonormality_max_abs_error,
            "cumulative_second_moment_ratio": {
                str(rank): float(np.sum(self.explained_second_moment_ratio[:rank]))
                for rank in (1, 2, 4, 8, 12, 16, 20, 32, 64)
                if rank <= self.max_rank
            },
        }


def fit_local_edit_delta_pca(deltas: np.ndarray, *, max_rank: int = 64, carrier_dim: int = 384) -> CarrierFit:
    if carrier_dim not in (192,384): raise ValueError("BOUND_MODEL_DIMENSION_REQUIRED")
    CARRIER_DIM = carrier_dim
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != CARRIER_DIM:
        raise ValueError(f"DELTA_MATRIX_SHAPE_REQUIRED_N_BY_{CARRIER_DIM}:{values.shape}")
    if len(values) < max_rank:
        raise ValueError("INSUFFICIENT_CARRIER_FIT_UNITS")
    if not np.isfinite(values).all():
        raise ValueError("NONFINITE_CARRIER_FIT_DELTA")
    _left, singular_values, right_transpose = np.linalg.svd(values, full_matrices=False)
    basis64 = _canonicalize_column_signs(right_transpose[:max_rank].T)
    gram = basis64.T @ basis64
    orthonormality_error = float(np.max(np.abs(gram - np.eye(max_rank, dtype=np.float64))))
    if orthonormality_error > 1e-10:
        raise RuntimeError(f"CARRIER_BASIS_ORTHONORMALITY_FAILURE:{orthonormality_error}")
    second_moment = singular_values ** 2
    denominator = float(np.sum(second_moment))
    ratios = second_moment / denominator if denominator > 0.0 else np.zeros_like(second_moment)
    basis = basis64.astype(np.float32)
    retained_singular = singular_values[:max_rank].astype(np.float64)
    retained_ratios = ratios[:max_rank].astype(np.float64)
    return CarrierFit(
        basis=basis,
        singular_values=retained_singular,
        explained_second_moment_ratio=retained_ratios,
        unit_count=int(len(values)),
        carrier_dim=CARRIER_DIM,
        max_rank=int(max_rank),
        basis_sha256=array_sha256(basis),
        singular_values_sha256=array_sha256(retained_singular),
        orthonormality_max_abs_error=orthonormality_error,
    )


def oracle_coefficients(deltas: np.ndarray, basis: np.ndarray, rank: int) -> np.ndarray:
    values = np.asarray(deltas, dtype=np.float32)
    directions = np.asarray(basis, dtype=np.float32)[:, :rank]
    return values @ directions


def reconstruct_delta(coefficients: np.ndarray, basis: np.ndarray, rank: int, *, beta: float = 1.0) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=np.float32)[..., :rank]
    directions = np.asarray(basis, dtype=np.float32)[:, :rank]
    return (float(beta) * (coeff @ directions.T)).astype(np.float32, copy=False)


def random_orthogonal_equal_norm(
    reference_delta: np.ndarray,
    basis: np.ndarray,
    rank: int,
    *,
    seed: int,
) -> np.ndarray:
    reference = np.asarray(reference_delta, dtype=np.float64)
    directions = np.asarray(basis, dtype=np.float64)[:, :rank]
    # The stored runtime basis is float32.  QR restores a numerically exact
    # orthonormal projector for this diagnostic without changing its span.
    directions, _ = np.linalg.qr(directions, mode="reduced")
    generator = np.random.default_rng(int(seed))
    random = generator.standard_normal(reference.shape)
    random -= (random @ directions) @ directions.T
    random_norm = np.linalg.norm(random, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    if np.any(random_norm <= 1e-12):
        raise RuntimeError("RANDOM_ORTHOGONAL_DEGENERACY")
    scaled = random * (reference_norm / random_norm)[:, None]
    residual = np.max(np.abs(scaled @ directions)) if rank else 0.0
    if residual > 1e-8:
        raise RuntimeError(f"RANDOM_ORTHOGONALITY_FAILURE:{residual}")
    return scaled.astype(np.float32)
