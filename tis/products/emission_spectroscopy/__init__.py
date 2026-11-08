"""Emission Spectroscopy specialist product public interface."""
from .api import (
    PRODUCT_ID,
    run_analysis,
    analyze_doppler_spectrum,
    ion_temperature_from_doppler_fwhm,
    decompose_gaussian_linewidth,
)

__all__ = [
    "PRODUCT_ID",
    "run_analysis",
    "analyze_doppler_spectrum",
    "ion_temperature_from_doppler_fwhm",
    "decompose_gaussian_linewidth",
]
