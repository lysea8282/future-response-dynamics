"""Existing post-hoc P-F subspace separation; NON_GATING. No entrypoint."""
from .operator import r90_from_native_cf, subspace_similarity

STATUS = "POST_HOC_DESCRIPTIVE"
NON_GATING = True

def separation(r_f, r_p, r_cf):
    """Use the same native-CF energy rank as the frozen matched assay."""
    rank = r90_from_native_cf(r_cf)
    return 1.0 - subspace_similarity(r_p, r_f, rank)
