"""Public API for the Doppler & Charge-Exchange Spectroscopy specialist product.

The product owns these workflow IDs and delegates to canonical domain solvers.
Solver implementation stays behind this stable product boundary during migration.
"""
from __future__ import annotations

from typing import Any

from ..api import execute_product_analysis

PRODUCT_ID = "doppler-charge-exchange-spectroscopy"


def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute a workflow only if it is owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)


def analyze_cxrs_line(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 4: Charge-exchange recombination spectroscopy (CXRS)."""
    return run_analysis(4, *args, **kwargs)

def analyze_magnetic_field(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 5: Ion temperature from Doppler spectroscopy."""
    return run_analysis(5, *args, **kwargs)

def estimate_impurity_fraction(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 24: Charge-exchange spectroscopy."""
    return run_analysis(24, *args, **kwargs)
