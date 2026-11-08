"""Public API for the Emission Spectroscopy specialist product.

The product owns these workflow IDs and delegates to canonical domain solvers.
Solver implementation stays behind this stable product boundary during migration.
"""
from __future__ import annotations

from typing import Any

from ..api import execute_product_analysis

PRODUCT_ID = "emission-spectroscopy"


def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute a workflow only if it is owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)


def analyze_doppler_spectrum(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 21: Visible / optical emission spectroscopy."""
    return run_analysis(21, *args, **kwargs)

def ion_temperature_from_doppler_fwhm(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 22: VUV spectroscopy."""
    return run_analysis(22, *args, **kwargs)

def decompose_gaussian_linewidth(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 25: Hydrogen Balmer / Balmer-alpha spectroscopy."""
    return run_analysis(25, *args, **kwargs)
