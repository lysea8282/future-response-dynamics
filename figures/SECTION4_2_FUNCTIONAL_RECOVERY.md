# Section 4.2: structured-GRU functional recovery

One publication dot plot replaces a numeric results table in the manuscript. The derived CSV is solely the machine-readable figure input; no duplicate display table is generated.

## Reproduce

From this repository root, using the pinned plotting environment in `environment/plotting-requirements.txt`:

```sh
python -B scripts/figures/plot_section4_2_functional_recovery.py
python -B scripts/verify_artifacts.py
```

Validated with Python 3.12.10 and Matplotlib 3.10.9. Output is a 7.2 by 3.5 inch vector PDF and 600 dpi PNG (4320 by 2100 pixels), suitable for a two-column-width conference figure. The caption is stored separately beside the artwork.

## Scientific inputs and aggregation

The script reads existing `results/accepted_summaries/original.json` and `extension.json`, bound by SHA256. These are the accepted original confirmation and six-eligible-checkpoint extension independent-audit results. Full canonical locators, source hashes, exact fields and population provenance are in `provenance/SECTION4_2_FUNCTIONAL_RECOVERY_SOURCE.json`.

The shared fixed registry has 64 units: 32 S1 and 32 S2. Each checkpoint-stratum recovery scalar is the accepted median of within-unit temporal medians, with normalized recovery `(eF-eT)/(eF-eP)` evaluated at contrastable times (`eF-eP > 1e-5`). Accepted scalars are extracted directly; unit membership, missingness, aggregation and scientific thresholds are unchanged. Complete-assay status comes directly from `supports`, cross-checked against the accepted gates. Capture is used for parity verification only and is not plotted.

Frozen baseline-eligible order: 291405, 291406, 291407, 291410, 291411, 291412, 291413, 291415, 291416. Checkpoint 291414 is baseline-ineligible and mechanism not evaluated. The 1,024-unit Section 4.1 diagnostic is a distinct population.

S1 uses blue circles; S2 uses vermilion squares. Filled points pass the complete assay. Open points at 291405/S1 and 291413/S2 retain transport/function support but fail the complete assay. The dashed recovery=1 line is a reference, not a decision threshold. No uncertainty intervals or additional scientific claims are inferred.

All seven requested parity checks pass: 9 checkpoints/18 cells, range 0.744-1.034, 17/18 above 0.85, 18/18 transport/function support, 16/18 complete-assay support, exact two failed cells and their rounded recovery/capture values, and exclusion of 291414. No model, refitting or experimental execution is involved.
