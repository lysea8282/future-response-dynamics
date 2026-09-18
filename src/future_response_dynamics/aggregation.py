"""Frozen checkpoint-first aggregation extracted without run/claim/write behavior."""
from __future__ import annotations
import numpy as np
from .classifier import classify_alignment, classify_causal, classify_full_dose, classify_overall

def aggregate_certified_rows(unit_rows, causal_rows, dose_rows, checkpoint_seeds, contrastability_floor):
    config = {"checkpoint_seeds": checkpoint_seeds, "contrastability_floor": contrastability_floor}
    def med(rows: list[dict[str, object]], key: str) -> float:
        return float(np.median([float(row[key]) for row in rows]))

    alignment_cells = []
    causal_cells = []
    dose_cells = []
    contrast_floor = float(config["contrastability_floor"])
    for seed in config["checkpoint_seeds"]:
        for stratum in ("S1", "S2"):
            ucell = [r for r in unit_rows if r["checkpoint"] == seed and r["stratum"] == stratum]
            alignment_cells.append({"checkpoint": seed, "stratum": stratum, "median_Delta_align": med(ucell, "Delta_align"), "checkpoint_unit_median_Delta_align": med(ucell, "Delta_align"), "median_Delta_d": med(ucell, "Delta_d"), "median_d_R_F_CF": med(ucell, "d_R_F_CF"), "contrastable": med(ucell, "d_R_F_CF") > contrast_floor})
            dcell = [r for r in dose_rows if r["checkpoint"] == seed and r["stratum"] == stratum]
            dose_cells.append({"checkpoint": seed, "stratum": stratum, "median_E_local": med(dcell, "E_local"), "median_C_R_midpoint": med(dcell, "C_R_midpoint"), "median_E_int32": med(dcell, "E_int32"), "median_I32_I16_relative": med(dcell, "I32_I16_relative"), "numerical_valid": med(dcell, "E_int32") < 0.01 and med(dcell, "I32_I16_relative") < 0.005})
            for route in ("CF", "P"):
                ccell = [r for r in causal_rows if r["checkpoint"] == seed and r["stratum"] == stratum and r["route"] == route]
                causal_cells.append({"checkpoint": seed, "stratum": stratum, "route": route, "median_spearman": med(ccell, "spearman"), "evaluable": bool(ccell) and np.isfinite([float(x["spearman"]) for x in ccell]).all()})
    components = {
        "alignment": classify_alignment(alignment_cells),
        "causal": classify_causal(causal_cells),
        "full_dose": classify_full_dose(dose_cells),
    }
    supporting_delta_d = all(sum(1 for c in alignment_cells if c["stratum"] == st and float(c["median_Delta_d"]) > 0) >= 2 for st in ("S1", "S2"))
    components["supporting_operator_distance"] = "FULL_OPERATOR_CONVERGENCE_SUPPORT_REPLICATED" if supporting_delta_d else "FULL_OPERATOR_CONVERGENCE_SUPPORT_NOT_REPLICATED"
    overall = classify_overall(components["alignment"], components["causal"], components["full_dose"], supporting_operator_distance=components["supporting_operator_distance"])
    summary = {"components": components, "overall": overall, "alignment_cells": alignment_cells, "causal_cells": causal_cells, "dose_cells": dose_cells, "contrastability_floor": contrast_floor}
    support_counts = {
        "alignment": sum(all(any(c["checkpoint"] == seed and c["stratum"] == stratum and c["contrastable"] and c["median_Delta_align"] > 0 for c in alignment_cells) for stratum in ("S1", "S2")) for seed in config["checkpoint_seeds"]),
        "causal": sum(all(any(c["checkpoint"] == seed and c["stratum"] == stratum and c["route"] == route and c["evaluable"] and c["median_spearman"] > 0.70 for c in causal_cells) for stratum in ("S1", "S2") for route in ("CF", "P")) for seed in config["checkpoint_seeds"]),
        "full_dose": sum(all(any(c["checkpoint"] == seed and c["stratum"] == stratum and c["numerical_valid"] and c["median_E_local"] > 0.10 and c["median_C_R_midpoint"] > 0.10 for c in dose_cells) for stratum in ("S1", "S2")) for seed in config["checkpoint_seeds"]),
    }
    return summary, support_counts
