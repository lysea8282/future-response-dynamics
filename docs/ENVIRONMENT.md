# Environment facts

- Inspected environment: Python3.12.10; Windows11 AMD64; NumPy2.4.4; PyTorch2.11.0+cu128.
- Inspection used interpreter/distribution metadata only. No torch/NumPy module or checkpoint was loaded during packaging.
- Byte-verification, sanitization and certified-summary readback: Python standard library only.
- Scientific modules: NumPy and PyTorch; minimal exact observed versions are in requirements-scientific.txt. PyTorch CUDA build is the observed cu128 build; installing the matching wheel requires the appropriate package source for the eventual target platform.
- No full environment dump is used as a requirements specification. No claim of testing other OS/GPU/driver combinations.
- Original model/state/action/future evaluation float32; exported Jacobian float64; preserve mixed operand dtypes and operation order specified by NUMERIC_READBACK_CONTRACT.json.
- Reproduction is not yet self-contained without the externally listed raw data and checkpoints. No model-run test was performed in this preparation.
