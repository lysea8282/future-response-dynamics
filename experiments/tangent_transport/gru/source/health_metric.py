from __future__ import annotations

import math
from typing import Any

import numpy as np


METRICS = ("M1", "M2", "M3", "M4", "M5")
DT = 0.25
RADIUS = 0.25
ARENA = 4.75
ENGINEERING_LIMIT = 1e6


def _time_of_impact(positions: np.ndarray, velocities: np.ndarray) -> float | None:
    relative_position = positions[1] - positions[0]
    relative_velocity = velocities[1] - velocities[0]
    if float(relative_position @ relative_velocity) >= 0:
        return None
    a = float(relative_velocity @ relative_velocity)
    b = 2.0 * float(relative_position @ relative_velocity)
    c = float(relative_position @ relative_position) - (2.0 * RADIUS) ** 2
    discriminant = b * b - 4.0 * a * c
    if a <= 1e-12 or discriminant < -1e-10:
        return None
    hit = (-b - math.sqrt(max(discriminant, 0.0))) / (2.0 * a)
    return float(np.clip(hit, 0.0, DT)) if -1e-10 <= hit <= DT + 1e-10 else None


def simulate_from(initial: np.ndarray, actions: np.ndarray) -> np.ndarray:
    states = np.zeros((len(actions) + 1, 2, 4), dtype=np.float64)
    states[0] = np.asarray(initial, dtype=np.float64)
    for step, action in enumerate(np.asarray(actions, dtype=np.float64)):
        positions = states[step, :, :2].copy()
        velocities = states[step, :, 2:].copy()
        if np.any(action[:2]):
            velocities[int(np.argmax(action[:2]))] += action[2:]
        hit = _time_of_impact(positions, velocities)
        if hit is None:
            positions += velocities * DT
        else:
            positions += velocities * hit
            displacement = positions[1] - positions[0]
            normal = displacement / max(float(np.linalg.norm(displacement)), 1e-12)
            impulse = -float((velocities[1] - velocities[0]) @ normal)
            velocities[0] -= impulse * normal
            velocities[1] += impulse * normal
            positions += velocities * (DT - hit)
        for slot in range(2):
            for axis in range(2):
                while positions[slot, axis] > ARENA or positions[slot, axis] < -ARENA:
                    if positions[slot, axis] > ARENA:
                        positions[slot, axis] = 2.0 * ARENA - positions[slot, axis]
                        velocities[slot, axis] *= -1.0
                    if positions[slot, axis] < -ARENA:
                        positions[slot, axis] = -2.0 * ARENA - positions[slot, axis]
                        velocities[slot, axis] *= -1.0
        states[step + 1, :, :2] = positions
        states[step + 1, :, 2:] = velocities
    return states


def derived_vector(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    positions, velocities = state[:, :2], state[:, 2:]
    relative_position = positions[1] - positions[0]
    relative_velocity = velocities[1] - velocities[0]
    center_position = positions.mean(axis=0)
    center_velocity = velocities.mean(axis=0)
    momentum = velocities.sum(axis=0)
    kinetic_energy = 0.5 * float(np.sum(velocities * velocities))
    distance = float(np.linalg.norm(relative_position))
    contact = float(distance <= 2.0 * RADIUS + 1e-8)
    return np.concatenate((relative_position, relative_velocity, center_position, center_velocity, momentum, [kinetic_energy, distance, contact])).astype(np.float64)


def metric_m1(prediction: np.ndarray, truth: np.ndarray, edit_object: int, edit_axis: int, primitive_scale: np.ndarray) -> float:
    component = 2 + int(edit_axis)
    return float(abs(prediction[0, int(edit_object), component] - truth[0, int(edit_object), component]) / primitive_scale[int(edit_object), component])


def metric_m2(prediction: np.ndarray, truth: np.ndarray, derived_scale: np.ndarray) -> float:
    normalized = (derived_vector(prediction[0]) - derived_vector(truth[0])) / np.asarray(derived_scale, dtype=np.float64)
    return float(np.sqrt(np.mean(normalized ** 2)))


def metric_m3(prediction: np.ndarray, truth: np.ndarray, primitive_scale: np.ndarray) -> float:
    normalized = (prediction[1:] - truth[1:]) / np.asarray(primitive_scale, dtype=np.float64)
    return float(np.sqrt(np.mean(normalized ** 2)))


def metric_m4(prediction: np.ndarray, truth: np.ndarray, unaffected_mask: np.ndarray, primitive_scale: np.ndarray) -> float:
    mask = np.asarray(unaffected_mask, dtype=bool).reshape(2, 4)
    return float(np.max(np.abs(prediction[0] - truth[0])[mask] / np.asarray(primitive_scale, dtype=np.float64)[mask]))


def post_edit_law_residual(outputs: np.ndarray, actions: np.ndarray, primitive_scale: np.ndarray) -> float:
    outputs = np.asarray(outputs, dtype=np.float64)
    if not np.isfinite(outputs).all():
        return math.inf
    residuals = []
    for step in range(len(outputs) - 1):
        expected = simulate_from(outputs[step], np.asarray(actions)[step:step + 1])
        residuals.append(float(np.sqrt(np.mean(((outputs[step + 1] - expected[1]) / primitive_scale) ** 2))))
    positions, velocities = outputs[..., :2], outputs[..., 2:]
    distance = np.linalg.norm(positions[:, 1] - positions[:, 0], axis=-1)
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    adjacent_speed = np.maximum(np.linalg.norm(velocities[:-1], axis=-1), np.linalg.norm(velocities[1:], axis=-1))
    return max(
        max(residuals, default=0.0),
        float(np.max(np.maximum(0.0, 0.5 - distance))) / 0.01,
        float(np.max(np.maximum(0.0, np.abs(positions) - ARENA))) / 0.01,
        float(np.max(np.maximum(0.0, displacement - adjacent_speed * DT - 1e-8))) / 0.01,
    )


def metric_m5(post_edit_law: float, edit_boundary_error: float) -> float:
    return float(max(post_edit_law, edit_boundary_error / 1e-8))


def score_m1_m5(prediction: np.ndarray, truth: np.ndarray, actions_after_anchor: np.ndarray, record: dict[str, Any], scales: dict[str, Any]) -> dict[str, float]:
    primitive_scale = np.asarray(scales["primitive_scale"], dtype=np.float64)
    derived_scale = np.asarray(scales["derived_scale"], dtype=np.float64)
    obj, axis = int(record["edit_object_id"]), int(record["edit_axis"])
    component = 2 + axis
    edit_boundary = np.asarray(record["counterfactual_state_t"], dtype=np.float64) - np.asarray(record["factual_state_t"], dtype=np.float64)
    declared = np.zeros((2, 4), dtype=np.float64)
    declared[obj, component] = float(record["counterfactual_values"][0]) - float(record["factual_values"][0])
    boundary_error = float(np.max(np.abs(edit_boundary - declared)))
    return {
        "M1": metric_m1(prediction, truth, obj, axis, primitive_scale),
        "M2": metric_m2(prediction, truth, derived_scale),
        "M3": metric_m3(prediction, truth, primitive_scale),
        "M4": metric_m4(prediction, truth, np.asarray(record["unaffected_primitive_mask"]), primitive_scale),
        "M5": metric_m5(post_edit_law_residual(prediction, actions_after_anchor, primitive_scale), boundary_error),
    }


def apply_thresholds(raw: dict[str, float], thresholds: dict[str, float]) -> dict[str, dict[str, object]]:
    if set(raw) != set(METRICS) or set(thresholds) != set(METRICS):
        raise ValueError("EXACT_M1_M5_KEYS_REQUIRED_LEGACY_M6_FORBIDDEN")
    return {
        metric: {
            "raw": float(raw[metric]),
            "threshold": float(thresholds[metric]),
            "direction": "LESS_THAN_OR_EQUAL",
            "pass": bool(np.isfinite(raw[metric]) and raw[metric] <= thresholds[metric]),
        }
        for metric in METRICS
    }


def evaluate_m6e(hidden: np.ndarray, decoded_state: np.ndarray, autonomous_trajectory: np.ndarray,
                 *, normalized_decoded_state: np.ndarray | None = None, runtime_ok: bool = True,
                 serialization_ok: bool = True) -> dict[str, object]:
    hidden = np.asarray(hidden)
    decoded_state = np.asarray(decoded_state)
    trajectory = np.asarray(autonomous_trajectory)
    normalized = np.asarray(normalized_decoded_state if normalized_decoded_state is not None else decoded_state)
    subchecks = {
        "finite_hidden": bool(np.isfinite(hidden).all()),
        "finite_decoded_state": bool(np.isfinite(decoded_state).all()),
        "finite_autonomous_trajectory": bool(np.isfinite(trajectory).all()),
        "no_nan_or_inf": bool(np.isfinite(hidden).all() and np.isfinite(decoded_state).all() and np.isfinite(trajectory).all()),
        "no_serialization_or_runtime_failure": bool(runtime_ok and serialization_ok),
        "max_hidden_l2_le_1e6": bool(np.isfinite(hidden).all() and np.max(np.linalg.norm(hidden.reshape(-1, hidden.shape[-1]), axis=-1)) <= ENGINEERING_LIMIT),
        "max_normalized_decoded_l2_le_1e6": bool(np.isfinite(normalized).all() and np.max(np.linalg.norm(normalized.reshape(-1, normalized.shape[-2] * normalized.shape[-1]), axis=-1)) <= ENGINEERING_LIMIT),
        "max_abs_physical_component_le_1e6": bool(np.isfinite(trajectory).all() and np.max(np.abs(trajectory)) <= ENGINEERING_LIMIT),
    }
    return {"role": "MANDATORY_ENGINEERING_STATE_SANITY_ELIGIBILITY", "scientific_evidence": False, "subchecks": subchecks, "pass": all(subchecks.values())}


def unit_decision(metric_results: dict[str, dict[str, object]], m6e: dict[str, object], m7: dict[str, object]) -> dict[str, object]:
    if set(metric_results) != set(METRICS):
        raise ValueError("EXACT_M1_M5_REQUIRED")
    eligible = bool(m6e["pass"] and m7["pass"])
    scientifically_accepted = bool(eligible and all(metric_results[metric]["pass"] for metric in METRICS))
    return {
        "eligible": eligible,
        "scientifically_accepted": scientifically_accepted,
        "failure_classification": None if eligible else "NOT_SCIENTIFICALLY_EVALUABLE_ENGINEERING_OR_AUTONOMY_FAILURE",
    }


def aggregate_cell(unit_records: list[dict[str, Any]], thresholds: dict[str, float], *, expected_denominator: int,
                   mandatory_controls_pass: bool, fixture_only: bool = False) -> dict[str, object]:
    if set(thresholds) != set(METRICS):
        raise ValueError("EXACT_M1_M5_THRESHOLDS_REQUIRED")
    denominator_complete = len(unit_records) == expected_denominator
    jointly_accepted = sum(bool(row["eligible"] and all(row["M1_M5"][m]["pass"] for m in METRICS)) for row in unit_records)
    coverage = jointly_accepted / expected_denominator if expected_denominator else 0.0
    medians = {}
    median_pass = {}
    for metric in METRICS:
        values = np.asarray([row["M1_M5"][metric]["raw"] for row in unit_records], dtype=np.float64)
        medians[metric] = float(np.median(values)) if len(values) else math.nan
        median_pass[metric] = bool(np.isfinite(medians[metric]) and medians[metric] <= thresholds[metric])
    passed = bool(denominator_complete and coverage >= 0.8 and all(median_pass.values()) and mandatory_controls_pass and all(row["eligible"] for row in unit_records))
    return {
        "denominator": expected_denominator,
        "materialized_count": len(unit_records),
        "denominator_complete": denominator_complete,
        "joint_pass_count": jointly_accepted,
        "joint_coverage": coverage,
        "coverage_target": 0.8,
        "medians": medians,
        "thresholds": thresholds,
        "median_pass": median_pass,
        "mandatory_controls_pass": mandatory_controls_pass,
        "pass": passed,
        "artifact_class": "ENGINEERING_FIXTURE_ONLY" if fixture_only else "SCIENTIFIC_EVALUATION",
        "interpretation": "NOT_A_MODEL_ADEQUACY_RESULT" if fixture_only else "SEPARATE_AUTHORITY_REQUIRED",
    }


def g2_decision(hidden_max_abs: float, rollout_max_abs: float, tolerance: float = 1e-6) -> dict[str, object]:
    return {
        "hidden_max_abs_discrepancy": float(hidden_max_abs),
        "rollout_max_abs_discrepancy": float(rollout_max_abs),
        "tolerance": float(tolerance),
        "physics_evidence": False,
        "pass": bool(np.isfinite(hidden_max_abs) and np.isfinite(rollout_max_abs) and hidden_max_abs <= tolerance and rollout_max_abs <= tolerance),
    }


def seed_decision(cells: dict[str, dict[str, object]]) -> dict[str, object]:
    required = {f"{stratum}/{route}" for stratum in ("S1", "S2") for route in ("G0", "G1", "G2")}
    complete = set(cells) == required
    return {"required_cells": sorted(required), "complete": complete, "pass": bool(complete and all(cells[key]["pass"] for key in required))}


def panel_decision(seed_results: dict[str, dict[str, object]]) -> dict[str, object]:
    passing = sum(bool(row["pass"]) for row in seed_results.values())
    passed = passing >= 2 and len(seed_results) == 3
    return {
        "passing_seed_count": passing,
        "total_seed_count": len(seed_results),
        "minimum_passing_seeds": 2,
        "pass": passed,
        "decision": "PASS_DEVELOPMENT_MODEL_ADEQUACY_PANEL_READY_FOR_FRESH_REPLICATION_PREREGISTRATION" if passed else "MODEL_ADEQUACY_NOT_ESTABLISHED_DESIGN_REOPEN_REQUIRED",
    }
