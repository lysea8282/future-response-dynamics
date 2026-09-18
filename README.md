# Compact but Moving: Intervention-Relevant Geometry in Recurrent World Models

Public reproducibility material for Paper 2: released derived data, figure/table generators, scientific reference modules, and executable verification tests for a controlled two-object recurrent world-model study.

## Findings and scope

Paper 1 established the checkpoint-specific rank-4 **operational intervention-entry interface**. Paper 2 studies its downstream transport and future-response geometry. Rank 4 is a tested intervention-interface property, not an intrinsic hidden-state dimension.

- The correction leaves the fixed entry plane, while its trajectory-specific, factual-Jacobian-transported image captures most of the realized correction (Figure 1).
- Tangent restart retains future counterfactual function in all 18 cells across nine baseline-eligible structured-GRU checkpoints (Figure 2). Complete specificity is stricter: 291405/S1 and 291413/S2 remain failures. Checkpoint 291414 is baseline-ineligible and mechanism-not-evaluated. Complete-assay support is 7/10 end-to-end, or 7/9 conditional on baseline eligibility; the original and extension panels contribute 2/3 and 5/6. The ten checkpoints were not all preregistered at the original start.
- The monolithic LSTM comparator has 494,664 parameters, hidden size 192, a complete 384-D hidden-plus-cell carrier, and a hidden-only decoder. Privileged rank 6 was frozen on the development grid before fresh evaluation. It initializes the transport assay, supported in all six cells of checkpoints 392001-392003 (Appendix Table H1). It does not transfer the GRU address map, establish operational semantic addressability, or test LSTM semantic specificity.
- The full-carrier, eight-transition future-response assay supports CF-directed leading-subspace alignment in 3/3 checkpoints (Appendix Table H2), local prediction of finite-perturbation rankings in 3/3 (Appendix Table H3), and finite-dose response reconfiguration in 2/3 (Appendix Table H4). Its 291402-291404 panel is distinct from the transport confirmation panel. P-F separation is descriptive and non-gating; Simpson integration is numerical path accounting.

The perturbation assay uses the same frozen six-family construction and index ordering at P and native CF, not 52 identical coordinate vectors: 4 leading future-response, 4 null future-response, 4 leading carrier-propagation, 4 leading anchor-decoder, 4 shared Paper 1 interface, and 32 projected-random directions. The 16 operator-derived and 32 random directions are route-specific; the route token enters the random seed. Ranking is evaluated separately on each route's registered perturbation panel. It does not validate arbitrary unseen directions, paired P-vs-CF effects on identical vectors, or calibrated effect magnitudes.

The finite-dose path joins factual and once-patched hidden carriers; it is not a physical-edit axis. Midpoint operator-chord deviation is the registered reconfiguration statistic. Numerical Simpson integration is accounting for the endpoint change, not an online model mechanism.

Compactness can persist as a moving, trajectory-dependent local geometry embedded in high-dimensional recurrent dynamics. **No claim of a fixed low-dimensional recurrent state or closed effective state is made.** Neither rank 4 nor privileged rank 6 is a universal cross-architecture state dimension.

## Reproduce and verify

Run from this repository with Python 3.12 (3.11+ supports the standard-library tools):

```sh
python -B scripts/verify_artifacts.py
python -B scripts/generate_manuscript_tables.py --output ../paper2-manuscript-tables
python -B scripts/generate_paper_tables.py --sources results/accepted_summaries --output ../paper2-transport-tables
python -B tests/test_integrity.py --scratch ../paper2-integrity-scratch
```

Table generators require new output directories. Appendix Tables H1-H4 (stored as `table1`-`table4` for stable filenames) and path-accounting readbacks are in `results/manuscript_tables/`; the broader transport-panel tables remain in `results/paper_tables/`. Values are read from released result files, not manually inserted into plotting or analysis code. These commands use the standard library and normally finish in seconds.

For the public plotting and CPU fixture checks, install the dependencies used by CI:

```sh
python -m pip install numpy==2.4.4 matplotlib==3.10.9
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

The standard-library verification and table commands above need none of these packages. Reference-only GRU signed-rank/bootstrap modules additionally need SciPy; see `requirements-scientific.txt` and `environment/README.md`.

With the plotting dependencies in `environment/plotting-requirements.txt`:

```sh
python -B scripts/figures/plot_section4_1_transport_geometry.py --output-root ../paper2-figure1
python -B scripts/figures/plot_section4_2_functional_recovery.py --output-root ../paper2-figure2
```

Both figures regenerate from public inputs without model execution. Figure 1 is the accepted 1,024-unit exploratory population; Figure 2 uses the fixed 64-unit confirmation registry. Plotting normally takes under a minute per figure. See [figure sources](figures/README.md).

With NumPy and PyTorch installed, run the CPU semantic checks:

```sh
python -B tests/test_manuscript_sync.py
```

These execute the public response, perturbation, path-integration, transport and LSTM code on controlled fixtures, and reaggregate the released unit-level results. They normally take under a minute after imports. Fixtures test implementation semantics; they are not fresh scientific replications. [.github/workflows/verify.yml](.github/workflows/verify.yml) provides the same checks for CI.

## Available assets and limitations

| Location | Contents |
| --- | --- |
| `model/`, `experiments/`, `src/` | Architecture and frozen scientific reference code; no Formal launcher |
| `configs/` | Frozen scientific thresholds, registries and contracts |
| `results/` | Released unit-level response data, summaries and manuscript table readbacks |
| `figures/` | Figures 1-2 and existing Appendix B Figure B1 (editable PPTX and two 600 dpi PNGs) |
| `data/interfaces/` | Public Paper 1 checkpoint-specific affine interfaces |
| `provenance/` | Source identity, historical logical locators and local file hashes |

Full neural reproduction requires separately held canonical checkpoints, training/evaluation populations, and the frozen scientific environment. These assets are **not publicly reproducible by design in this package**; see [data scope](data/README.md) and [environment](environment/README.md). The public verification route checks released derived results and executes asset-free mathematical fixtures. It does not rerun training, load private checkpoints, retune thresholds or certify scientific truth.

Historical logical locators in provenance are retained for traceability; public commands do not open them. Transport evidence originates in Stage2_V3.2; the response assay retains its Stage2_V3.3 origin. A fixed Git revision or review-package hash is the external integrity anchor. The public repository is [future-response-dynamics](https://github.com/lysea8282/future-response-dynamics).

## Citation and publication metadata

Paper 1: Liu, Y., and Chen, Y. (2026). *Low-Rank Dynamics-Effective Latent Carriers for Counterfactual Rollout in Learned World Models*. [arXiv:2608.15156](https://arxiv.org/abs/2608.15156).

Paper 2 citation metadata will be updated after arXiv posting. No Paper 2 arXiv identifier or DOI is assigned in this repository. This repository is released under the [MIT License](LICENSE). See [publication choices](docs/PUBLICATION_METADATA_TODO.md).
