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
):
    """Analyze one calibrated Langmuir I–V sweep with explicit physical assumptions.

    Fit windows are supplied by the analyst rather than guessed. This workflow
    intentionally excludes geometry-dependent density/flux inversion. Calibration,
    fit-window sensitivity, and experiment-specific interpretation remain external.
    """
    return langmuir_iv_analysis(
        voltage_V,
        current_A,
        electron_fit_range=electron_fit_range,
        ion_fit_range=ion_fit_range,
        current_convention=current_convention,
    )
