"""Langmuir probe analysis entry point for the canonical diagnostic registry."""
from __future__ import annotations

from tis.core import langmuir_iv_analysis


def analyze_langmuir_probe(
    voltage_V,
    current_A,
    *,
    electron_fit_range=None,
    ion_fit_range=None,
    current_convention="electron_positive",
    probe_area_m2=None,
    ion_mass_kg=None,
    ion_saturation_coefficient=None,
    sheath_model=None,
):
    """Analyze one calibrated Langmuir I–V sweep with explicit physical assumptions.

    Fit windows and, for optional density inversion, collection area, ion mass,
    sheath-model coefficient and model label are supplied explicitly by the analyst.
    Density is not inferred from descriptive tail fallbacks.
    """
    return langmuir_iv_analysis(
        voltage_V,
        current_A,
        electron_fit_range=electron_fit_range,
        ion_fit_range=ion_fit_range,
        current_convention=current_convention,
        probe_area_m2=probe_area_m2,
        ion_mass_kg=ion_mass_kg,
        ion_saturation_coefficient=ion_saturation_coefficient,
        sheath_model=sheath_model,
    )
