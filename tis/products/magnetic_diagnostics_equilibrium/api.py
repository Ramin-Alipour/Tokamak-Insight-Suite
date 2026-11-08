"""Public API for the Magnetic Diagnostics & Equilibrium specialist product.

The product owns these workflow IDs and delegates to canonical domain solvers.
Solver implementation stays behind this stable product boundary during migration.
"""
from __future__ import annotations

from typing import Any

from ..api import execute_product_analysis

PRODUCT_ID = "magnetic-diagnostics-equilibrium"


def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute a workflow only if it is owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)


def analyze_toroidal_field_coil(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 8: Toroidal-field coils / magnetic probes."""
    return run_analysis(8, *args, **kwargs)

def find_x_point_candidates(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 9: Poloidal-field coils / magnetic probes."""
    return run_analysis(9, *args, **kwargs)

def shafranov_shift(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 10: Mirnov / pickup coils."""
    return run_analysis(10, *args, **kwargs)

def axisymmetric_volume_from_boundary(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 11: Flux loops."""
    return run_analysis(11, *args, **kwargs)

def cross_section_area_from_boundary(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 12: Rogowski coil."""
    return run_analysis(12, *args, **kwargs)

def reconstruct_equilibrium(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 13: Equilibrium reconstruction (Grad–Shafranov / EFIT-like)."""
    return run_analysis(13, *args, **kwargs)

def analyze_polarimetry(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 14: Polarimetry / Faraday rotation."""
    return run_analysis(14, *args, **kwargs)

def reconstruct_poloidal_flux(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 15: Magnetic flux / poloidal flux."""
    return run_analysis(15, *args, **kwargs)

def estimate_mode_number(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 55: MHD mode / Mirnov analysis."""
    return run_analysis(55, *args, **kwargs)
