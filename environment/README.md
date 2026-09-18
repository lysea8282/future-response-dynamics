# Environment and reproduction scope

Portable verification and table regeneration require Python 3.12 (standard library only). They do not import torch or run a model. Frozen scientific reference modules require their canonical NumPy/PyTorch environment and separately held data/checkpoints; those neural runs are not claimed reproduced by this package. The GRU transport scientific config uses float32 model execution, float64 analysis and cuda:0; privileged LSTM uses CPU with the same precision division. Frozen configs remain the authority. Existing environment metadata for the older H8 assay is retained separately and must not be assumed to specify every newer run.

## Current manuscript synchronization checks

Standard-library verification/table commands also ran on Python 3.11.9; scientific fixture tests and plotting ran on Python 3.12.10, NumPy 2.4.4, PyTorch 2.11.0+cu128 (CPU execution), and Matplotlib 3.10.9. The public fixture tests require NumPy and PyTorch but no CUDA device, checkpoint or private dataset. The hosted CI job is configured separately; local execution is not proof that a hosted CI run has occurred.

The frozen GRU `paired_statistics.py` reference module also imports SciPy. The local final-readiness boundary/statistics checks used SciPy 1.17.1; this is an observed public-reference test dependency, not a newly asserted historical neural-run environment. `requirements-scientific.txt` records this dependency. The standard public fixture suite and CI do not import that reference module. CPU installation commands are in README; the CUDA wheel in the scientific requirements records the observed scientific environment and is not needed for public CPU checks.
