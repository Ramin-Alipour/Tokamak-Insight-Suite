"""Public API for the X-Ray Diagnostics specialist product."""
from __future__ import annotations
from typing import Any
from ..api import execute_product_analysis

PRODUCT_ID = "x-ray-diagnostics"

def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute only a workflow owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)

def doppler_flow_velocity(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 23: X-ray spectroscopy."""
    return run_analysis(23, *args, **kwargs)

def analyze_xray_spectrum(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 34: Soft X-ray arrays."""
    return run_analysis(34, *args, **kwargs)

def analyze_bolometry(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 35: Hard X-ray detector."""
    return run_analysis(35, *args, **kwargs)
