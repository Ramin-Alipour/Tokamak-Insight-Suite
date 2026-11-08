"""Public API for the Plasma-Facing Components & Thermal Analysis specialist product."""
from __future__ import annotations
from typing import Any
from ..api import execute_product_analysis

PRODUCT_ID = "plasma-facing-components-thermal-analysis"

def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute only a workflow owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)

def analyze_divertor_flux(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 71: Thermocouples / embedded sensors."""
    return run_analysis(71, *args, **kwargs)

def plasma_facing_component_heat_load(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 72: Target/divertor heat flux."""
    return run_analysis(72, *args, **kwargs)

def sputtering_yield(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 73: Surface erosion / deposition."""
    return run_analysis(73, *args, **kwargs)

def analyze_divertor_particle_flux(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 74: Wall impurity / dust monitoring."""
    return run_analysis(74, *args, **kwargs)

def detect_lh_transition(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 76: Thermal response / heat diffusion."""
    return run_analysis(76, *args, **kwargs)
