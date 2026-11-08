"""Public API for the Imaging & IR Thermography specialist product.

The product owns these workflow IDs and delegates to canonical domain solvers.
Solver implementation stays behind this stable product boundary during migration.
"""
from __future__ import annotations

from typing import Any

from ..api import execute_product_analysis

PRODUCT_ID = "imaging-ir-thermography"


def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute a workflow only if it is owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)


def reconstruct_emissivity(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 36: Visible imaging."""
    return run_analysis(36, *args, **kwargs)

def analyze_visible_image(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 37: Fast visible imaging."""
    return run_analysis(37, *args, **kwargs)

def analyze_line_radiation(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 38: IR thermography."""
    return run_analysis(38, *args, **kwargs)

def analyze_edge_field(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 75: Strike-point / hot-spot tracking."""
    return run_analysis(75, *args, **kwargs)
