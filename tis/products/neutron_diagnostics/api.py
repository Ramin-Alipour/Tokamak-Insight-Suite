"""Public API for the Neutron Diagnostics specialist product.

The product owns these workflow IDs and delegates to canonical domain solvers.
Solver implementation stays behind this stable product boundary during migration.
"""
from __future__ import annotations

from typing import Any

from ..api import execute_product_analysis

PRODUCT_ID = "neutron-diagnostics"


def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute a workflow only if it is owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)


def analyze_neutron_flux(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 40: Neutron yield monitor."""
    return run_analysis(40, *args, **kwargs)

def analyze_neutron_spectrum(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 41: Neutron spectrometry."""
    return run_analysis(41, *args, **kwargs)

def fusion_reaction_rate(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 42: Neutron imaging / tomography."""
    return run_analysis(42, *args, **kwargs)
