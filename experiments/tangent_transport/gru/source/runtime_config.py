"""Explicit process-local immutable scientific configuration."""
C={}
def configure(config):
    if C and C != config: raise RuntimeError('CONFIG_ALREADY_BOUND')
    C.update(config)
