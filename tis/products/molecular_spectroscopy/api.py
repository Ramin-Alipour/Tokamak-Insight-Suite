"""Public API for the Molecular Spectroscopy specialist product.

The product owns these workflow IDs and delegates to canonical domain solvers.
Solver implementation stays behind this stable product boundary during migration.
"""
from __future__ import annotations

from typing import Any

from ..api import execute_product_analysis

PRODUCT_ID = "molecular-spectroscopy"


def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute a workflow only if it is owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)


def analyze_molecular_spectrum(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 6: Molecular Spectroscopy."""
    return run_analysis(6, *args, **kwargs)
