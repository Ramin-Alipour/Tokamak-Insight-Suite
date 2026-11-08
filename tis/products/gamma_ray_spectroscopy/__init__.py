"""Gamma-Ray Spectroscopy specialist product public interface."""
from .api import (
    PRODUCT_ID,
    run_analysis,
    "analyze_gamma_spectrum",
)

__all__ = ["PRODUCT_ID", "run_analysis",
    "analyze_gamma_spectrum",
]
