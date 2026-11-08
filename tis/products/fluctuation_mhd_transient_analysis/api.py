"""Public API for the Fluctuation, MHD & Transient Analysis specialist product."""
from __future__ import annotations
from typing import Any
from ..api import execute_product_analysis

PRODUCT_ID = "fluctuation-mhd-transient-analysis"

def run_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute only a workflow owned by this specialist product."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)

def analyze_mirnov_signal(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 54: Multi-channel fluctuation analysis."""
    return run_analysis(54, *args, **kwargs)

def magnetic_mode_frequency(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 56: ELM / transient event detection."""
    return run_analysis(56, *args, **kwargs)

def analyze_fft(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 57: Sawtooth analysis."""
    return run_analysis(57, *args, **kwargs)

def analyze_svd(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 58: Disruption analysis."""
    return run_analysis(58, *args, **kwargs)

def analyze_wavelet(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 59: Correlation / cross-correlation."""
    return run_analysis(59, *args, **kwargs)

def analyze_turbulence(*args: Any, **kwargs: Any) -> Any:
    """Execute analysis 60: Spectral transfer / coherence."""
    return run_analysis(60, *args, **kwargs)
