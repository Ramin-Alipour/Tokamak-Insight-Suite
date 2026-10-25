"""Molecular-band spectrum analysis with explicit, model-independent limits.

This module reports calibrated-axis spectral features. It does not infer rotational or
vibrational temperature without a user-supplied, physically justified molecular model.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, peak_widths


def analyze_molecular_spectrum(
    wavelength_nm,
    intensity,
    *,
    reference_bandheads=None,
    prominence_fraction=0.03,
    baseline_quantile=0.10,
    band_ranges_nm=None,
):
    """Summarize a molecular emission spectrum and optionally match supplied bandheads.

    Parameters
    ----------
    wavelength_nm, intensity : 1-D array-like
        Finite samples on a strictly increasing wavelength axis. Wavelengths must
        already be calibrated; intensity units are preserved.
    reference_bandheads : iterable of (wavelength_nm, label), optional
        User-supplied reference bandheads. No built-in database is assumed.
    prominence_fraction : float
        Minimum detected peak prominence as a fraction of the baseline-subtracted
        spectral contrast; must be in (0, 1].
    baseline_quantile : float
        Quantile used for a robust constant baseline estimate; must be in [0, 0.5).
    band_ranges_nm : iterable of (lower_nm, upper_nm), optional
        Optional integration windows, each fully inside the measured wavelength range.

    Returns
    -------
    dict
        Baseline, contrast, integrated signal, peaks and optional band integrals.
        Integrated signal is in intensity_unit * nm, not radiated power unless the
        supplied intensity has the corresponding calibrated spectral-power unit.
    """
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 5:
        raise ValueError("wavelength_nm and intensity must be matching 1-D arrays with at least 5 samples")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Spectrum values must all be finite")
    if np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("Wavelengths must be positive and strictly increasing")
    prominence_fraction = float(prominence_fraction)
    baseline_quantile = float(baseline_quantile)
    if not np.isfinite(prominence_fraction) or not 0 < prominence_fraction <= 1:
        raise ValueError("prominence_fraction must be in (0, 1]")
    if not np.isfinite(baseline_quantile) or not 0 <= baseline_quantile < 0.5:
        raise ValueError("baseline_quantile must be in [0, 0.5)")
    baseline = float(np.quantile(y, baseline_quantile))
    corrected = y - baseline
    contrast = float(np.max(corrected))
    if contrast <= 0:
        raise ValueError("Spectrum has no positive contrast above the estimated baseline")
    peaks, props = find_peaks(corrected, prominence=prominence_fraction * contrast)
    width_data = peak_widths(corrected, peaks, rel_height=0.5) if peaks.size else None
    widths = width_data[0] if width_data is not None else np.array([])
    sample_indices = np.arange(x.size, dtype=float)
    peak_results = []
    for i, peak in enumerate(peaks):
        left_i = int(np.clip(np.floor(width_data[2][i]), 0, x.size - 1))
        right_i = int(np.clip(np.ceil(width_data[3][i]), 0, x.size - 1))
        left_nm = float(np.interp(width_data[2][i], sample_indices, x))
        right_nm = float(np.interp(width_data[3][i], sample_indices, x))
        peak_results.append({
            "wavelength_nm": float(x[peak]),
            "intensity": float(y[peak]),
            "prominence": float(props["prominences"][i]),
            "approx_fwhm_nm": float(right_nm - left_nm),
            "left_half_max_nm_approx": left_nm,
            "right_half_max_nm_approx": right_nm,
        })
    result = {
        "baseline": baseline,
        "contrast": contrast,
        "integrated_signal_intensity_nm": float(trapezoid(np.maximum(corrected, 0.0), x)),
        "peak_count": int(len(peak_results)),
        "peaks": peak_results,
        "wavelength_range_nm": [float(x[0]), float(x[-1])],
        "prominence_fraction": prominence_fraction,
        "baseline_quantile": baseline_quantile,
        "caveats": [
            "Peak detection and constant-quantile baseline are exploratory, not a validated molecular-band inversion.",
            "Overlapping bands, blends, wavelength-dependent response, noise, and continuum emission can bias features and integrated signal.",
            "Rotational/vibrational temperatures require an appropriate species-specific model, transition data, instrument response, and uncertainty analysis.",
            "Molecular excitation temperatures must not automatically be interpreted as electron or ion temperature.",
            "Band identification only compares against reference bandheads explicitly supplied by the caller.",
        ],
    }
    if reference_bandheads is not None:
        refs = []
        for item in reference_bandheads:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("Each reference bandhead must be (wavelength_nm, label)")
            wl = float(item[0])
            if not np.isfinite(wl) or wl <= 0 or not str(item[1]).strip():
                raise ValueError("Reference bandheads need positive finite wavelengths and nonempty labels")
            refs.append((wl, str(item[1])))
        if not refs:
            raise ValueError("reference_bandheads cannot be empty when supplied")
        result["reference_matches"] = [
            {
                **peak,
                "nearest_reference_nm": float(min(refs, key=lambda r: abs(r[0] - peak["wavelength_nm"]))[0]),
                "reference_label": min(refs, key=lambda r: abs(r[0] - peak["wavelength_nm"]))[1],
                "offset_nm": float(peak["wavelength_nm"] - min(refs, key=lambda r: abs(r[0] - peak["wavelength_nm"]))[0]),
            }
            for peak in peak_results
        ]
        result["reference_match_caveat"] = "Nearest-reference matches are candidates only; caller must supply authoritative molecular data and choose an appropriate tolerance."
    if band_ranges_nm is not None:
        bands = []
        for item in band_ranges_nm:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("Each band range must contain lower and upper wavelengths")
            lo, hi = map(float, item)
            if not np.isfinite(lo + hi) or lo >= hi or lo < x[0] or hi > x[-1]:
                raise ValueError("Band ranges must be finite, increasing, and inside the measured wavelength range")
            mask = (x >= lo) & (x <= hi)
            xx = np.concatenate(([lo], x[mask], [hi]))
            yy = np.interp(xx, x, np.maximum(corrected, 0.0))
            bands.append({"lower_nm": lo, "upper_nm": hi,
                          "integrated_signal_intensity_nm": float(trapezoid(yy, xx))})
        result["band_integrals"] = bands
    return result


def fit_custom_molecular_model(wavelength_nm, intensity, model, *, parameter_names,
                               initial_values, bounds=(-np.inf, np.inf),
                               intensity_uncertainty=None):
    """Fit a caller-supplied molecular spectrum model without assigning its physics.

    model must be callable as model(wavelength_nm, *parameters). Model meaning,
    units, validity, and parameter interpretation remain the caller's responsibility.
    """
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(intensity, dtype=float)
    names = tuple(str(n) for n in parameter_names)
    p0 = np.asarray(initial_values, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 5:
        raise ValueError("Wavelength and intensity must be matching 1-D arrays with at least 5 samples")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("Spectrum must be finite with a positive, strictly increasing wavelength axis")
    if not callable(model):
        raise ValueError("model must be callable")
    if not names or len(set(names)) != len(names) or len(names) != p0.size:
        raise ValueError("Parameter names must be nonempty, unique, and match initial_values")
    if p0.size >= x.size:
        raise ValueError("The spectrum must contain more samples than fitted parameters")
    if not np.all(np.isfinite(p0)):
        raise ValueError("Initial parameter values must be finite")
    sigma = None
    if intensity_uncertainty is not None:
        sigma = np.asarray(intensity_uncertainty, dtype=float)
        if sigma.shape != y.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("intensity_uncertainty must be positive, finite, and match intensity")
    try:
        lower, upper = bounds
        lower = np.broadcast_to(np.asarray(lower, dtype=float), p0.shape)
        upper = np.broadcast_to(np.asarray(upper, dtype=float), p0.shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("bounds must contain broadcastable lower and upper parameter bounds") from exc
    if (not np.all(np.isfinite(lower) | np.isneginf(lower))
            or not np.all(np.isfinite(upper) | np.isposinf(upper))
            or np.any(lower >= upper)):
        raise ValueError("Each lower bound must be less than its upper bound; use -inf/+inf for unbounded sides")
    if np.any(p0 < lower) or np.any(p0 > upper):
        raise ValueError("initial_values must lie within the supplied parameter bounds")

    def checked_model(wavelength, *parameters):
        values = np.asarray(model(wavelength, *parameters), dtype=float)
        if values.shape != np.asarray(wavelength).shape or not np.all(np.isfinite(values)):
            raise ValueError("Model must return a finite array matching the wavelength input")
        return values

    try:
        popt, pcov = curve_fit(checked_model, x, y, p0=p0, bounds=(lower, upper),
                               sigma=sigma, absolute_sigma=sigma is not None, maxfev=50000)
        fitted = checked_model(x, *popt)
    except (TypeError, ValueError, FloatingPointError, RuntimeError, OverflowError) as exc:
        raise ValueError(f"Custom molecular model fitting failed: {exc}") from exc
    residual = y - fitted
    dof = int(y.size - len(popt))
    out = {
        "parameters": {name: float(value) for name, value in zip(names, popt)},
        "parameter_uncertainties": {name: float(np.sqrt(max(pcov[i, i], 0.0))) if np.isfinite(pcov[i, i]) else float("nan") for i, name in enumerate(names)},
        "covariance": pcov.tolist(),
        "fitted_spectrum": fitted.tolist(),
        "residual": residual.tolist(),
        "residual_sum_squares": float(np.sum(residual**2)),
        "degrees_of_freedom": dof,
        "model_name": getattr(model, "__name__", type(model).__name__),
        "caveat": "Fit statistics do not validate the molecular physics; parameter meanings and model suitability are the caller's responsibility.",
    }
    if sigma is not None:
        chi2 = float(np.sum((residual / sigma)**2))
        out["chi_square"] = chi2
        out["reduced_chi_square"] = chi2 / dof if dof > 0 else float("nan")
    return out

def fit_molecular_boltzmann_temperature(
    upper_energy_eV,
    integrated_intensity,
    einstein_A_s,
    degeneracy,
    wavelength_nm,
    *,
    intensity_uncertainty=None,
):
    """Estimate an excitation temperature from a supplied molecular transition table.

    Inputs are line-integrated *energy* intensities after background subtraction and
    instrument-response correction, upper-state energies in eV, Einstein A values in
    s^-1, upper-state statistical weights, and line wavelengths in nm. The fitted
    Boltzmann ordinate is proportional to ln(N_u/g_u), using photon rate proportional
    to energy intensity times wavelength. This is a generic caller-supplied line-list
    workflow; it does not identify species or establish LTE/optical thinness.

    At least four distinct upper-state energies are required. The returned temperature
    is an excitation temperature in eV/K, not automatically electron or ion temperature.
    """
    energy = np.asarray(upper_energy_eV, dtype=float)
    intensity = np.asarray(integrated_intensity, dtype=float)
    einstein_a = np.asarray(einstein_A_s, dtype=float)
    weight = np.asarray(degeneracy, dtype=float)
    wavelength = np.asarray(wavelength_nm, dtype=float)
    arrays = (energy, intensity, einstein_a, weight, wavelength)
    if any(a.ndim != 1 for a in arrays) or any(a.shape != energy.shape for a in arrays):
        raise ValueError("All transition columns must be matching one-dimensional arrays")
    if energy.size < 4:
        raise ValueError("At least four molecular transitions are required")
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("Transition columns must contain only finite values")
    if (np.any(energy < 0) or np.any(intensity <= 0) or np.any(einstein_a <= 0)
            or np.any(weight <= 0) or np.any(wavelength <= 0)):
        raise ValueError("Energies must be non-negative; intensity, Einstein A, degeneracy and wavelength must be positive")
    if np.unique(energy).size < 3:
        raise ValueError("At least three distinct upper-state energies are needed to constrain the fit")

    # For energy-integrated line radiance, photon rate is proportional to I*lambda.
    ordinate = np.log(intensity * wavelength / (einstein_a * weight))
    sigma_log = None
    if intensity_uncertainty is not None:
        sigma_i = np.asarray(intensity_uncertainty, dtype=float)
        if (sigma_i.shape != intensity.shape or not np.all(np.isfinite(sigma_i))
                or np.any(sigma_i <= 0)):
            raise ValueError("intensity_uncertainty must be positive, finite, and match integrated_intensity")
        sigma_log = sigma_i / intensity

    design = np.column_stack((energy, np.ones_like(energy)))
    if sigma_log is None:
        weights = np.ones_like(energy)
    else:
        weights = 1.0 / sigma_log**2
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_y = ordinate * np.sqrt(weights)
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
        if rank != 2:
            raise ValueError("Transition energies do not constrain a unique Boltzmann slope")
        slope, intercept = map(float, coefficients)
        covariance_base = np.linalg.inv(weighted_design.T @ weighted_design)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Boltzmann fit is numerically singular") from exc
    if not np.isfinite(slope) or slope >= 0:
        raise ValueError("Boltzmann slope must be negative; no positive finite excitation temperature can be inferred")

    fitted = design @ coefficients
    residual = ordinate - fitted
    dof = int(energy.size - 2)
    chi_square = float(np.sum((residual / sigma_log)**2)) if sigma_log is not None else float(np.sum(residual**2))
    if sigma_log is None:
        covariance = covariance_base * (chi_square / dof)
    else:
        covariance = covariance_base
    slope_uncertainty = float(np.sqrt(max(float(covariance[0, 0]), 0.0)))
    temperature_eV = -1.0 / slope
    temperature_uncertainty_eV = slope_uncertainty / slope**2
    kelvin_per_eV = 11604.518121550082
    total = float(np.sum((ordinate - np.mean(ordinate))**2))
    r_squared = float(1.0 - np.sum(residual**2) / total) if total > 0 else float("nan")
    return {
        "excitation_temperature_eV": float(temperature_eV),
        "excitation_temperature_uncertainty_eV": float(temperature_uncertainty_eV),
        "excitation_temperature_K": float(temperature_eV * kelvin_per_eV),
        "excitation_temperature_uncertainty_K": float(temperature_uncertainty_eV * kelvin_per_eV),
        "boltzmann_slope_per_eV": slope,
        "boltzmann_intercept": intercept,
        "slope_uncertainty_per_eV": slope_uncertainty,
        "r_squared": r_squared,
        "chi_square": chi_square,
        "degrees_of_freedom": dof,
        "fitted_log_population_over_weight": fitted.tolist(),
        "residual": residual.tolist(),
        "line_count": int(energy.size),
        "uncertainty_model": "First-order weighted log-intensity fit; excludes wavelength/A-value/degeneracy uncertainties, correlated errors, calibration systematics and model discrepancy.",
        "assumptions": [
            "All lines belong to the same correctly identified molecular species and compatible population distribution.",
            "Line-integrated energy intensities are background-subtracted, absolutely or consistently response-corrected, and positive.",
            "Optically thin emission and a Boltzmann population distribution are adequate for the selected transitions.",
            "Einstein A values, upper-state energies, degeneracies and wavelengths come from an authoritative transition database.",
        ],
        "caveats": [
            "This is an excitation temperature, not automatically electron or ion temperature.",
            "A high R-squared does not prove LTE, optical thinness, correct species assignment or validity of the supplied molecular data.",
            "Systematic calibration, blending, self-absorption and non-equilibrium populations can bias the result.",
        ],
    }
