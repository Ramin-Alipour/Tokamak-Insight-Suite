"""Public API for the Gamma-Ray Spectroscopy specialist product."""
from __future__ import annotations
from typing import Any
from ..api import execute_product_analysis

PRODUCT_ID = "gamma-ray-spectroscopy"

def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute only a workflow owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)

def analyze_gamma_spectrum(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 43: Gamma spectroscopy."""
    return run_analysis(43, *args, **kwargs)
