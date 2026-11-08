"""Public API for the Fast-Particle Diagnostics specialist product."""
from __future__ import annotations
from typing import Any
from ..api import execute_product_analysis

PRODUCT_ID = "fast-particle-diagnostics"

def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute only a workflow owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)

def analyze_runaway_events(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 44: Neutral particle analyzer (NPA)."""
    return run_analysis(44, *args, **kwargs)

def analyze_energy_distribution(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 45: Fast-ion D-alpha (FIDA)."""
    return run_analysis(45, *args, **kwargs)

def energetic_particle_flux(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 47: Fast-ion loss detector (FILD)."""
    return run_analysis(47, *args, **kwargs)

def estimate_runaway_energy(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 48: Runaway-electron diagnostics."""
    return run_analysis(48, *args, **kwargs)
