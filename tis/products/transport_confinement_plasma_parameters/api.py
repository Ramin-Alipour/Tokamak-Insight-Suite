"""Public API for the Transport, Confinement & Plasma Parameters specialist product."""
from __future__ import annotations
from typing import Any
from ..api import execute_product_analysis

PRODUCT_ID = "transport-confinement-plasma-parameters"

def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute only a workflow owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)

def radial_transport_analysis(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 61: Particle flux and effective diffusion."""
    return run_analysis(61, *args, **kwargs)

def poloidal_transport_analysis(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 62: Energy confinement."""
    return run_analysis(62, *args, **kwargs)

def reynolds_stress(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 63: Energy balance."""
    return run_analysis(63, *args, **kwargs)

def energy_confinement_time(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 64: Greenwald density reference."""
    return run_analysis(64, *args, **kwargs)

def analyze_heat_flux(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 65: Pressure and beta."""
    return run_analysis(65, *args, **kwargs)
