"""Product-owned Poisson-count optical Thomson EVDF workflow.

The shared numerical solver owns the physical likelihood, constrained EVDF,
Poisson sandwich covariance, and exact q-parallel physical-moment closure.
This adapter deliberately preserves that complete result without recomputing
or overwriting the physical moments.
"""
from __future__ import annotations

from typing import Any

from tis.physics.shared.thomson_evdf_poisson import (
    reconstruct_thomson_evdf_poisson as _reconstruct,
)


def reconstruct_thomson_evdf_poisson(*args: Any, **kwargs: Any) -> dict:
    """Run the registered product-owned count-space EVDF inverse."""
    result = _reconstruct(*args, **kwargs)
    result["product_execution_path"] = "thomson-scattering/analysis-89"
    return result


__all__ = ["reconstruct_thomson_evdf_poisson"]
