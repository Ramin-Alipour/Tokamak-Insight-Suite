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
    ion_current_uncertainty_A=None,
    probe_area_relative_uncertainty=None,
    ion_mass_relative_uncertainty=None,
    ion_saturation_coefficient_relative_uncertainty=None,
    electron_temperature_relative_uncertainty=None,
    electron_temperature_calibration_relative_uncertainty=None,
    electron_temperature_fit_window_relative_uncertainty=None,
    density_model_relative_uncertainty=None,
):
    """Analyze one calibrated Langmuir I–V sweep with explicit physical assumptions.

    Fit windows and, for optional density inversion, collection area, ion mass,
    sheath-model coefficient and model label are supplied explicitly by the analyst.
    Density is not inferred from descriptive tail fallbacks. Optional density
    uncertainty inputs are forwarded to the core for explicit-component propagation.
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
        ion_current_uncertainty_A=ion_current_uncertainty_A,
        probe_area_relative_uncertainty=probe_area_relative_uncertainty,
        ion_mass_relative_uncertainty=ion_mass_relative_uncertainty,
        ion_saturation_coefficient_relative_uncertainty=ion_saturation_coefficient_relative_uncertainty,
        electron_temperature_relative_uncertainty=electron_temperature_relative_uncertainty,
        electron_temperature_calibration_relative_uncertainty=electron_temperature_calibration_relative_uncertainty,
        electron_temperature_fit_window_relative_uncertainty=electron_temperature_fit_window_relative_uncertainty,
        density_model_relative_uncertainty=density_model_relative_uncertainty,
    )



def analyze_langmuir_probe_array(
    voltage_V,
    current_A=None,
    *,
    channel_names=None,
    **analysis_options,
):
    """Analyze calibrated multi-channel Langmuir I–V sweeps independently.

    Accepts either (channels, samples) arrays sharing a voltage grid, or a
    mapping of channel name to {'voltage_V': ..., 'current_A': ...}. Each
    channel is passed through the canonical single-sweep implementation;
    no spatial interpolation or cross-channel physical inference is performed.
    """
    import numpy as np

    if isinstance(voltage_V, dict) and current_A is None:
        channels = voltage_V
        results = {}
        for name, pair in channels.items():
            if not isinstance(pair, dict) or "voltage_V" not in pair or "current_A" not in pair:
                raise ValueError("Each channel must provide voltage_V and current_A arrays.")
            results[str(name)] = analyze_langmuir_probe(
                pair["voltage_V"], pair["current_A"], **analysis_options
            )
        return {
            "analysis_scope": "calibrated_multi_channel_independent_IV_sweeps",
            "channel_count": len(results),
            "channels": results,
            "cross_channel_inference_performed": False,
        }

    v = np.asarray(voltage_V)
    i = np.asarray(current_A)
    if v.ndim == 1 and i.ndim == 2:
        if i.shape[1] != v.size:
            raise ValueError("For a shared voltage grid, current_A shape must be (channels, samples).")
        v = np.broadcast_to(v, i.shape)
    elif v.ndim == 2 and i.ndim == 2:
        if v.shape != i.shape:
            raise ValueError("Two-dimensional voltage_V and current_A arrays must have identical shapes.")
    else:
        raise ValueError("Array inputs must be (samples,) with (channels, samples), or matching (channels, samples) arrays.")
    if channel_names is None:
        names = [f"channel_{index + 1}" for index in range(i.shape[0])]
    else:
        names = list(channel_names)
        if len(names) != i.shape[0] or len(set(names)) != len(names):
            raise ValueError("channel_names must contain one unique name per channel.")
    results = {
        str(name): analyze_langmuir_probe(v[index], i[index], **analysis_options)
        for index, name in enumerate(names)
    }
    return {
        "analysis_scope": "calibrated_multi_channel_independent_IV_sweeps",
        "channel_count": len(results),
        "channels": results,
        "cross_channel_inference_performed": False,
    }
