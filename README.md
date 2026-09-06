# Future-response dynamics

Local publication-preparation repository for the tested deterministic recurrent world-model family's matched factual, patched and native-counterfactual future responses. Proposed paper title: **Future-Response Dynamics after Compact Hidden Interventions in Recurrent World Models**. Title and publication metadata remain provisional.

## Scope and relationship to Paper 1

Paper 1 supplies checkpoint-specific compact hidden intervention entry and bounded one-step non-closure/coupling diagnostics. The present evidence concerns full-carrier eight-transition future-output response after that entry. Rank4 is an inherited intervention interface, not an identified state dimension.

The certified `_002` panel has alignment support3/3, local perturbation rank agreement3/3, and finite-dose reconfiguration2/3. Local causal evidence concerns Spearman rank agreement, not calibrated effect magnitudes. `Delta_PF-Align` is post-hoc descriptive and non-gating; `Delta_d` is supporting-only. Checkpoint291404's two strata differ and remain visible in the tables.

## Included material

- `src/future_response_dynamics/`: frozen scientific modules with AST-preserving whitespace normalization and science-only AST-preserved extracts; no internal formal runner or claim-creation machinery.
- `configs/`: scientific contracts and small registries with public path metadata. Historical manifest hashes identify originals; `data/PUBLIC_INTERFACE_BINDINGS.json` identifies actual local public interface copies.
- `data/interfaces/`: small frozen affine interface arrays; `data/EXTERNAL_ARTIFACTS.json`: bindings for excluded checkpoint/population/raw operands.
- `results/`: full-precision certified and explicitly descriptive tables.
- `scripts/`: read-only byte verification and stored-result display.
- `tests/`: actual positive/negative integrity tests, no target-model execution.
- `docs/`: environment, source roles, frozen scientific-loop excerpt, publication TODOs.
- `provenance/`: sanitized certified readback, original/current copy hashes and repository payload manifest.
- `figures/`: data pointers only; final manuscript figures are not prepared here.

## Reproduce the included result readback

From the repository root, using Python3.12:

```console
python -B scripts/verify_artifacts.py
python -B scripts/show_certified_results.py
python -B tests/test_integrity.py --scratch .integrity-scratch
```

These commands verify the packaged bytes and display previously certified summaries. They do not recompute scientific metrics or run a model. Running full scientific reproduction is a separate activity requiring the external artifacts and a prepared execution environment. The scientific modules expose the frozen route/operator/causal/path/classifier functions; the internal one-shot formal launcher is intentionally outside the public reproduction interface. No complete end-to-end public model-run CLI is advertised.

`DOSE_PROGRESSION_SUMMARY.csv` is an index to the frozen unit-level progression table: no precomputed cell-level aggregate existed, and none was calculated during repository preparation.

## Environment and availability

The summary checks use only the Python standard library. Scientific modules depend on NumPy and PyTorch; tested local metadata and minimal dependencies are in `docs/ENVIRONMENT.md` and `requirements-scientific.txt`.

Checkpoint binaries and raw NPZ operands are not included. `data/EXTERNAL_ARTIFACTS.json` lists logical IDs, sizes, hashes and requirements. No public download host or URL has been selected; full raw reproduction is not yet self-contained. Small affine interface arrays are included. The Writer source ZIP, supplied separately, contains all material needed for Methods/Results drafting without these raw binaries.

## Publication status

This is a local preparation repository. No remote is configured and nothing has been pushed. License, authors/citation, final title, account/organization and data host are pending human decisions. No license or citation identity is implied. See `docs/PUBLICATION_METADATA_TODO.md`.
