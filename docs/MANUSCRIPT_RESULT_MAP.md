# Current manuscript result map

| Object | Public source | Generator / verification |
| --- | --- | --- |
| Figure 1, Section 4.1 | results/paper_figures/section4_1_source/POOLING_EFFECT_SUMMARY.csv | scripts/figures/plot_section4_1_transport_geometry.py |
| Figure 2, Section 4.2 | results/accepted_summaries/original.json; extension.json | scripts/figures/plot_section4_2_functional_recovery.py |
| Table H1, Appendix H.2; Section 4.3 | results/accepted_summaries/lstm.json | scripts/generate_manuscript_tables.py |
| Table H2, Appendix H.3; Section 4.4 | results/ALIGNMENT_CHECKPOINT_SUMMARY.csv; DELTA_TRIAD_SUMMARY.csv | scripts/generate_manuscript_tables.py |
| Table H3, Appendix H.4; Section 4.5 | results/CAUSAL_CHECKPOINT_STRATUM_ROUTE_SUMMARY.csv | scripts/generate_manuscript_tables.py |
| Table H4, Appendix H.5; path accounting in Section 4.6 | results/FULL_DOSE_SUMMARY.csv | scripts/generate_manuscript_tables.py |
| Figure B1, Appendix B | figures/appendix_b1/ | Existing PPTX and PNG source; model/monolithic_lstm/lstm_model.py and semantic tests |

Tables H2-H4 additionally reaggregate from the released unit-level files in `tests/test_manuscript_sync.py`. Table H2 P-F separation remains post-hoc descriptive and non-gating. The all-strata checkpoint-first finite-dose decision remains 2/3, with 291404/S1 failing the midpoint gate. Simpson path accounting is numerical validation, not a separate mechanism.

Public results verify reported observations without regenerating the original neural trajectories. Full checkpoint/data-based reproduction remains NOT_PUBLICLY_REPRODUCIBLE_BY_DESIGN. Historical logical source locators are provenance only and are never runtime dependencies of the public verification commands.

The `table1`-`table4` public filenames are retained for compatibility; they correspond to final Appendix Tables H1-H4. Historical draft destinations in `provenance/CLAIM_EVIDENCE_MATRIX.md` are not current section numbers.
