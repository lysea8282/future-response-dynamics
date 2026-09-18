"""Frozen train-only affine ridge operator for W30 carrier coefficients."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


FEATURE_NAMES = (
    "intercept",
    "object0_vx_edit", "object0_vy_edit", "object1_vx_edit", "object1_vy_edit",
    "object0_x", "object0_y", "object0_vx", "object0_vy",
    "object1_x", "object1_y", "object1_vx", "object1_vy",
)


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest().upper()


def _signed_edit(record: dict[str, Any]) -> float:
    return float(record["counterfactual_values"][0]) - float(record["factual_values"][0])


def build_features(records: Sequence[dict[str, Any]], *, wrong_object: bool = False) -> np.ndarray:
    features = np.zeros((len(records), len(FEATURE_NAMES)), dtype=np.float64)
    features[:, 0] = 1.0
    for row_index, record in enumerate(records):
        obj = int(record["edit_object_id"])
        if wrong_object:
            obj = 1 - obj
        axis = int(record["edit_axis"])
        features[row_index, 1 + obj * 2 + axis] = _signed_edit(record)
        state = np.asarray(record["factual_state_t"], dtype=np.float64).reshape(8)
        features[row_index, 5:] = state
    if not np.isfinite(features).all():
        raise ValueError("NONFINITE_OPERATOR_FEATURES")
    return features


def swapped_identity_features(records: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    original = build_features(records)
    transformed = np.zeros_like(original)
    transformed[:, 0] = 1.0
    transformed[:, 1:3] = original[:, 3:5]
    transformed[:, 3:5] = original[:, 1:3]
    transformed[:, 5:9] = original[:, 9:13]
    transformed[:, 9:13] = original[:, 5:9]
    remapped_back = np.zeros_like(transformed)
    remapped_back[:, 0] = transformed[:, 0]
    remapped_back[:, 1:3] = transformed[:, 3:5]
    remapped_back[:, 3:5] = transformed[:, 1:3]
    remapped_back[:, 5:9] = transformed[:, 9:13]
    remapped_back[:, 9:13] = transformed[:, 5:9]
    return original, remapped_back


@dataclass(frozen=True)
class RidgeOperator:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    ridge_lambda: float
    std_floor: float
    fit_unit_count: int

    @property
    def parameter_sha256(self) -> str:
        payload = b"".join(
            np.ascontiguousarray(value).tobytes(order="C")
            for value in (self.feature_mean, self.feature_scale, self.weights)
        )
        return hashlib.sha256(payload).hexdigest().upper()

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        transformed = values.copy()
        transformed[:, 1:] = (values[:, 1:] - self.feature_mean[1:]) / self.feature_scale[1:]
        transformed[:, 0] = 1.0
        return transformed

    def predict(self, features: np.ndarray, rank: int) -> np.ndarray:
        return (self.transform(features) @ self.weights[:, :rank]).astype(np.float32)

    def summary(self) -> dict[str, object]:
        return {
            "operator": "TRAIN_ONLY_EDIT_TO_COEFFICIENT_OPERATOR",
            "model": "RIDGE_AFFINE",
            "ridge_lambda": self.ridge_lambda,
            "std_floor": self.std_floor,
            "fit_unit_count": self.fit_unit_count,
            "feature_names": list(FEATURE_NAMES),
            "feature_mean_sha256": array_sha256(self.feature_mean),
            "feature_scale_sha256": array_sha256(self.feature_scale),
            "weights_sha256": array_sha256(self.weights),
            "parameter_bundle_sha256": self.parameter_sha256,
        }


def fit_ridge_operator(
    features: np.ndarray,
    coefficient_targets: np.ndarray,
    *,
    ridge_lambda: float = 1e-4,
    std_floor: float = 1e-6,
) -> RidgeOperator:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(coefficient_targets, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or y.ndim != 2 or len(x) != len(y):
        raise ValueError("RIDGE_FIT_SHAPE_MISMATCH")
    mean = np.zeros(x.shape[1], dtype=np.float64)
    scale = np.ones(x.shape[1], dtype=np.float64)
    mean[1:] = np.mean(x[:, 1:], axis=0)
    raw_std = np.std(x[:, 1:], axis=0, ddof=0)
    scale[1:] = np.maximum(raw_std, float(std_floor))
    standardized = x.copy()
    standardized[:, 1:] = (x[:, 1:] - mean[1:]) / scale[1:]
    standardized[:, 0] = 1.0
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(ridge_lambda)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(standardized.T @ standardized + penalty, standardized.T @ y)
    if not np.isfinite(weights).all():
        raise RuntimeError("NONFINITE_RIDGE_WEIGHTS")
    return RidgeOperator(mean, scale, weights, float(ridge_lambda), float(std_floor), int(len(x)))
