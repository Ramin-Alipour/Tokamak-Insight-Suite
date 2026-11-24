"""Product-owned Poisson-count optical Thomson EVDF workflow.

The numerical inverse is shared-domain code. This product adapter completes
its physical output contract by deriving normalized q-parallel EVDF moments
from the fitted distribution and propagating the conditional EVDF covariance.
"""
from __future__ import annotations

from typing import Any

from tis.physics.shared.thomson_evdf_poisson import (
    reconstruct_thomson_evdf_poisson as _reconstruct,
)
from .thomson_distribution import _projected_evdf_physical_moments


def reconstruct_thomson_evdf_poisson(*args: Any, **kwargs: Any) -> dict:
    """Run count-space inverse and derive physical q-parallel moments.

    The density-normalized projected flux contribution is reported, while
    absolute heat flux and density are withheld because a normalized single-
    view EVDF does not identify absolute electron density.
    """
    result = _reconstruct(*args, **kwargs)
    covariance = result.get("evdf_covariance")
    moments = _projected_evdf_physical_moments(
        result["velocity_parallel_m_s"],
        result["evdf_parallel_s_per_m"],
        covariance,
    )
    result["projected_evdf_physical_moments"] = moments
    result["physical_moment_scope"] = (
        "one_dimensional_q_parallel_marginal_only; no absolute density or "
        "three-dimensional heat flux inferred"
    )
    result["physical_moment_covariance_status"] = (
        "conditional_expected_poisson_fisher_delta_method"
        if covariance is not None else "unavailable_singular_active_set_fisher"
    )
    return result


__all__ = ["reconstruct_thomson_evdf_poisson"]
