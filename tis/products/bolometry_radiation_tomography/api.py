"""Public API for the Bolometry & Radiation Tomography specialist product.

The product owns these workflow IDs and delegates to canonical domain solvers.
Solver implementation stays behind this stable product boundary during migration.
"""
from __future__ import annotations

from typing import Any

from ..api import execute_product_analysis

PRODUCT_ID = "bolometry-radiation-tomography"


def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute a workflow only if it is owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)


def analyze_hard_xray(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 32: Bolometry."""
    return run_analysis(32, *args, **kwargs)

def analyze_soft_xray(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 33: Fast bolometry."""
    return run_analysis(33, *args, **kwargs)

def reconstruct_bolometric_tomography(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 39: Bolometric tomography."""
    return run_analysis(39, *args, **kwargs)
