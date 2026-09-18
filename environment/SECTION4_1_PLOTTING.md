# Section 4.1 plotting environment

Validated with Python 3.12.10 and environment/plotting-requirements.txt. No torch, checkpoint or private data access is required. The repository-local frozen summary is checked against its recorded SHA256 before plotting.

Run from the repository root:

```sh
python -B scripts/figures/plot_section4_1_transport_geometry.py
python -B scripts/verify_artifacts.py
```

For an independent regeneration, pass `--output-root ../section4-1-rebuild`. The generated CSV/PDF/PNG use deterministic metadata and fixed fonts. The native canvas is 3.5 inches wide for a manuscript column; PNG resolution is 600 dpi. Endpoint parity checks run before any figure is generated. Exact data are copied from the accepted summary, keyed by checkpoint and release, rather than interpolated from manuscript endpoints. No IQR is inferred from other quantile columns. The task-specific source report records the full private canonical paths; public provenance uses logical workspace-relative locators.
