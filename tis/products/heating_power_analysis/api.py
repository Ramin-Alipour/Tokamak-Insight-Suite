"""Public API for the Heating & Power Analysis specialist product."""
from __future__ import annotations
from typing import Any
from ..api import execute_product_analysis

PRODUCT_ID = "heating-power-analysis"

def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute only a workflow owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)

def analyze_electron_heating(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 66: Neutral beam injection (NBI) power."""
    return run_analysis(66, *args, **kwargs)

def analyze_ion_heating(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 67: Electron cyclotron heating (ECRH)."""
    return run_analysis(67, *args, **kwargs)

def ecrh_resonance_field(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 68: Ion cyclotron heating (ICRH)."""
    return run_analysis(68, *args, **kwargs)

def icrh_resonance_field(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 69: Lower hybrid heating/current drive."""
    return run_analysis(69, *args, **kwargs)

def analyze_rf_interaction(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 70: Electron/ion heating power."""
    return run_analysis(70, *args, **kwargs)
