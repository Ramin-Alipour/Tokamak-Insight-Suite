"""Product-owned Poisson-count optical Thomson EVDF solver entry point.

The physical inverse is implemented in the shared-domain numerical kernel;
this module establishes the product's explicit executable ownership path.
"""
from tis.physics.shared.thomson_evdf_poisson import reconstruct_thomson_evdf_poisson

__all__ = ["reconstruct_thomson_evdf_poisson"]
