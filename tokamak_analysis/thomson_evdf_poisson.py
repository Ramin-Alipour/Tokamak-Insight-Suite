"""Compatibility import for the registered Poisson Thomson EVDF subanalysis.

The product-owned solver remains canonical; this legacy module path allows
older analysis-registry consumers to resolve entry point 89 without duplicating
the physical implementation.
"""
from tis.products.thomson.physics.thomson_evdf_poisson import (
    reconstruct_thomson_evdf_poisson,
)

__all__ = ["reconstruct_thomson_evdf_poisson"]
