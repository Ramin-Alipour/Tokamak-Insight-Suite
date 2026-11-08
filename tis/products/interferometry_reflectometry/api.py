"""Public API for the Interferometry & Reflectometry specialist product.

The product owns these workflow IDs and delegates to canonical domain solvers.
Solver implementation stays behind this stable product boundary during migration.
"""
from __future__ import annotations

from typing import Any

from ..api import execute_product_analysis

PRODUCT_ID = "interferometry-reflectometry"


def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute a workflow only if it is owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)


def interferometry_density(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 28: Laser / microwave interferometry."""
    return run_analysis(28, *args, **kwargs)

def cutoff_density(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 29: Microwave reflectometry."""
    return run_analysis(29, *args, **kwargs)

def analyze_wave_signal(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 31: Refractive-index / ray-tracing diagnostics."""
    return run_analysis(31, *args, **kwargs)
