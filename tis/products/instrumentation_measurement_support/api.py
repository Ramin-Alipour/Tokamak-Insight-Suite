"""Public API for the Instrumentation & Measurement Support specialist product."""
from __future__ import annotations
from typing import Any
from ..api import execute_product_analysis

PRODUCT_ID = "instrumentation-measurement-support"

def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute only a workflow owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)

def identify_spectral_lines(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 26: Spectral wavelength calibration."""
    return run_analysis(26, *args, **kwargs)

def analyze_line_ratio(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 27: Instrumental line-shape characterization."""
    return run_analysis(27, *args, **kwargs)

def calibrate_signal(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 77: Calibration curve and uncertainty."""
    return run_analysis(77, *args, **kwargs)

def denoise_signal(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 78: Noise floor and detection limit."""
    return run_analysis(78, *args, **kwargs)

def uncertainty_summary(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 79: Sampling / aliasing audit."""
    return run_analysis(79, *args, **kwargs)

def synchronize_channels(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 80: Channel synchronization."""
    return run_analysis(80, *args, **kwargs)
