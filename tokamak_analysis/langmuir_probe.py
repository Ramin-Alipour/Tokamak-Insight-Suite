"""Langmuir probe analysis entry point for the canonical diagnostic registry."""
from __future__ import annotations

from tis.core import langmuir_iv_analysis


def analyze_langmuir_probe(
    voltage_V,
    current_A,
    *,
    electron_fit_range=None,
    ion_fit_range=None,
    area_m2=None,
    ion_mass_amu=None,
    ti_eV=0.0,
    sheath_factor=0.61,
    gamma_i=1.0,
    current_convention="electron_positive",
):
    """Analyze one calibrated Langmuir I–V sweep with explicit physical assumptions.

    Fit windows are supplied by the analyst rather than guessed. The optional
    density estimate uses the core's stated sheath/sound-speed model. Calibration,
    fit-window sensitivity, and experiment-specific interpretation remain external.
    """
    return langmuir_iv_analysis(
        voltage_V,
        current_A,
        electron_fit_range=electron_fit_range,
        ion_fit_range=ion_fit_range,
        area_m2=area_m2,
        ion_mass_amu=ion_mass_amu,
        ti_eV=ti_eV,
        sheath_factor=sheath_factor,
        gamma_i=gamma_i,
        current_convention=current_convention,
    )
