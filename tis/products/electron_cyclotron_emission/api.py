"""Public API for the Electron Cyclotron Emission (ECE) specialist product.

The product owns these workflow IDs and delegates to canonical domain solvers.
Solver implementation stays behind this stable product boundary during migration.
"""
from __future__ import annotations

from typing import Any

from ..api import execute_product_analysis

PRODUCT_ID = "electron-cyclotron-emission"


def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute a workflow only if it is owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)


def analyze_ece(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 3: Electron Cyclotron Emission (ECE)."""
    return run_analysis(3, *args, **kwargs)
