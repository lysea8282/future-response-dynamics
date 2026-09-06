"""Frozen component and overall decision rules; supporting descriptors never rescue gates."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

ALIGN_PASS = "ALIGNMENT_REPLICATED"
ALIGN_FAIL = "ALIGNMENT_NOT_REPLICATED"
ALIGN_NE = "ALIGNMENT_NOT_EVALUABLE"
CAUSAL_PASS = "LOCAL_OPERATOR_CAUSAL_PREDICTIVENESS_REPLICATED"
CAUSAL_FAIL = "LOCAL_OPERATOR_CAUSAL_PREDICTIVENESS_NOT_REPLICATED"
CAUSAL_NE = "LOCAL_OPERATOR_CAUSAL_PREDICTIVENESS_NOT_EVALUABLE"
DOSE_PASS = "FULL_DOSE_FUNCTIONAL_RECONFIGURATION_REPLICATED"
DOSE_FAIL = "FULL_DOSE_FUNCTIONAL_RECONFIGURATION_NOT_REPLICATED"
DOSE_NE = "FULL_DOSE_FUNCTIONAL_RECONFIGURATION_NOT_EVALUABLE"


def classify_alignment(cells: list[dict[str, Any]]) -> str:
    evaluable = [c for c in cells if c.get("contrastable", False)]
    by_stratum: dict[str, set[int]] = defaultdict(set)
    supportive: set[int] = set()
    by_checkpoint: dict[int, dict[str, float]] = defaultdict(dict)
    for cell in evaluable:
        seed, stratum = int(cell["checkpoint"]), str(cell["stratum"])
        by_stratum[stratum].add(seed)
        by_checkpoint[seed][stratum] = float(cell["median_Delta_align"])
    if any(len(by_stratum.get(s, set())) < 2 for s in ("S1", "S2")):
        return ALIGN_NE
    for seed, values in by_checkpoint.items():
        if values.get("S1", 0.0) > 0 and values.get("S2", 0.0) > 0:
            supportive.add(seed)
    family = {s: [float(c["checkpoint_unit_median_Delta_align"]) for c in evaluable if c["stratum"] == s] for s in ("S1", "S2")}
    family_positive = all(values and sorted(values)[len(values) // 2] > 0 for values in family.values())
    return ALIGN_PASS if len(supportive) >= 2 and family_positive else ALIGN_FAIL


def classify_causal(cells: list[dict[str, Any]]) -> str:
    if any(c.get("evaluable") is False for c in cells):
        return CAUSAL_NE
    by_checkpoint: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_checkpoint[int(cell["checkpoint"])].append(cell)
    if len(by_checkpoint) < 3:
        return CAUSAL_NE
    supportive = 0
    for group in by_checkpoint.values():
        keys = {(c["stratum"], c["route"]) for c in group}
        if keys == {("S1", "CF"), ("S1", "P"), ("S2", "CF"), ("S2", "P")} and all(float(c["median_spearman"]) > 0.70 for c in group):
            supportive += 1
    return CAUSAL_PASS if supportive >= 2 else CAUSAL_FAIL


def classify_full_dose(cells: list[dict[str, Any]]) -> str:
    if len(cells) != 6 or any(not c.get("numerical_valid", False) for c in cells):
        return DOSE_NE
    by_checkpoint: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_checkpoint[int(cell["checkpoint"])].append(cell)
    supportive = sum(
        {(c["stratum"]) for c in group} == {"S1", "S2"}
        and all(float(c["median_E_local"]) > 0.10 and float(c["median_C_R_midpoint"]) > 0.10 for c in group)
        for group in by_checkpoint.values()
    )
    return DOSE_PASS if supportive >= 2 else DOSE_FAIL


def classify_overall(alignment: str, causal: str, full_dose: str, **supporting: Any) -> str:
    del supporting  # r90, Delta_d, contrasts, and monotonicity are deliberately non-gating.
    required = (alignment, causal, full_dose)
    if any(x.endswith("NOT_EVALUABLE") for x in required):
        return "FRESH_FUTURE_FUNCTIONAL_DYNAMICS_NOT_EVALUABLE"
    if alignment == ALIGN_FAIL:
        return "FRESH_FUTURE_FUNCTIONAL_DYNAMICS_FAIL"
    passes = (alignment == ALIGN_PASS, causal == CAUSAL_PASS, full_dose == DOSE_PASS)
    if all(passes):
        return "FRESH_FUTURE_FUNCTIONAL_DYNAMICS_REPLICATION_PASS"
    if any(passes):
        return "FRESH_FUTURE_FUNCTIONAL_DYNAMICS_PARTIAL"
    return "FRESH_FUTURE_FUNCTIONAL_DYNAMICS_FAIL"
