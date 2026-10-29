"""Basic Langmuir, ball-pen and multi-pin probe data analysis."""
import numpy as np


def analyze_probe_voltage_current(
    voltage_V, current_A, *, electron_fit_range=None, ion_fit_range=None,
    current_convention="electron_positive",
):
    """Compatibility adapter to the canonical calibrated Langmuir I-V pipeline.

    Uses the same eight-distinct-voltage minimum, polarity handling, fit models,
    and output contract as tokamak_analysis.langmuir_probe.analyze_langmuir_probe.
    Legacy keys are retained as aliases; this function does not implement a
    second independent Langmuir fitting algorithm.
    """
    from tokamak_analysis.langmuir_probe import analyze_langmuir_probe

    result = analyze_langmuir_probe(
        voltage_V, current_A, electron_fit_range=electron_fit_range,
        ion_fit_range=ion_fit_range, current_convention=current_convention,
    )
    result.update({
        "voltage_V": np.asarray(result["processed_voltage_V"], dtype=float),
        "current_A": np.asarray(result["processed_current_A"], dtype=float),
        "dI_dV_S": np.asarray(result["dIdV_A_per_V"], dtype=float),
        "floating_potential_status_legacy": {
            "single_zero_crossing": "unique_crossing",
            "no_zero_crossing_in_measured_sweep": "no_crossing",
            "ambiguous_multiple_zero_crossings": "ambiguous_multiple_crossings",
        }[result["floating_potential_status"]],
        "sample_count": result["n_points"],
        "voltage_span_V": result["voltage_max_V"] - result["voltage_min_V"],
        "current_span_A": result["current_max_A"] - result["current_min_A"],
        "electron_temperature_estimate_eV": result["electron_temperature_eV"],
        "electron_fit_status": (
            "not_requested" if electron_fit_range is None
            else result["electron_retarding_fit_status"]
        ),
        "electron_fit_r_squared": result.get("electron_retarding_fit_R2"),
        "electron_fit_residual_rms_A": result.get("electron_retarding_fit_residual_rms_A"),
        "ion_fit_linear_slope_A_V": result.get("ion_saturation_linear_slope_A_per_V"),
        "ion_fit_linear_intercept_A": result.get("ion_saturation_intercept_A"),
        "ion_fit_points": result.get("ion_saturation_fit_points"),
        "ion_fit_r_squared": result.get("ion_saturation_fit_R2"),
    })
    return result

def electron_retarding_current(V_probe_V, plasma_potential_V, electron_saturation_current_A, electron_temperature_eV):
    """Maxwellian electron-retarding branch: Ie=Ies*exp((V-Vp)/Te[eV])."""
    v = np.asarray(V_probe_V, dtype=float)
    vp, ies, te = float(plasma_potential_V), float(electron_saturation_current_A), float(electron_temperature_eV)
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite([vp, ies, te])) or ies <= 0 or te <= 0:
        raise ValueError("Inputs must be finite; Ies and Te must be positive.")
    if np.any(v > vp):
        raise ValueError("This relation applies only to the electron-retarding region V <= Vp.")
    current = ies * np.exp((v - vp) / te)
    return {"electron_current_A": current, "equation": "Ie=Ies*exp((Vprobe-Vplasma)/Te_eV)",
            "assumption": "Ideal Maxwellian electron-retarding branch; sheath geometry, collisions, and ion current are not included."}
