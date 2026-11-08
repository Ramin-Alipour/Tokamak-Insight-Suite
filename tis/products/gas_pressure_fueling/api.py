"""Public API for the Gas, Pressure & Fueling Diagnostics specialist product."""
from __future__ import annotations
from typing import Any
from ..api import execute_product_analysis

PRODUCT_ID = "gas-pressure-fueling"

def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute only a workflow owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)

def analyze_vacuum_pressure(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 49: Pressure gauges."""
    return run_analysis(49, *args, **kwargs)

def analyze_gas_puffing(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 50: Residual gas analyzer (RGA)."""
    return run_analysis(50, *args, **kwargs)

def analyze_fueling_rate(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 51: Gas flow meters."""
    return run_analysis(51, *args, **kwargs)

def neutral_density(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 52: Neutral pressure / recycling estimates."""
    return run_analysis(52, *args, **kwargs)

def recycling_coefficient(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 53: Effective pumping speed."""
    return run_analysis(53, *args, **kwargs)
