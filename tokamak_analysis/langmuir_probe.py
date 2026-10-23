"""Langmuir probe analysis entry point for the canonical diagnostic registry."""
from __future__ import annotations
from tis.core import langmuir_iv_analysis


def analyze_langmuir_probe(voltage_V, current_A, *, electron_fit_range=None, ion_fit_range=None,
                            area_m2=None, ion_mass_amu=None):
    """Analyze a calibrated Langmuir I-V sweep using explicit fit ranges and geometry.

    Returns the numerical result from the shared, validated core implementation.
    Calibration uncertainty and experiment-specific sheath interpretation remain external.
    """
    return langmuir_iv_analysis(
        voltage_V, current_A,
        electron_fit_range=electron_fit_range,
        ion_fit_range=ion_fit_range,
        area_m2=area_m2,
        ion_mass_amu=ion_mass_amu,
    )
