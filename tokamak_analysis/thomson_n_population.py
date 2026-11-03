"""N-population relativistic optical Thomson forward model.

Each population is an isotropic Maxwell-Juttner component with its own
temperature and number-density fraction. Components share the supplied
scattering-vector flow projection, polarization and geometry. Mixture power
is weighted by component differential cross-sections, not temperature averages.
This is a parametric mixture, not arbitrary-EVDF inversion.
"""
from __future__ import annotations
import numpy as np
from .thomson_kinetic_models import C, relativistic_maxwell_juttner_spectrum

def _real_vector(value, name, minimum=1):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size < minimum:
        raise ValueError(f"{name} must be a real one-dimensional vector with at least {minimum} entries")
    out = raw.astype(float, copy=False)
    if not np.all(np.isfinite(out)): raise ValueError(f"{name} must contain finite values")
    return out

def n_population_relativistic_thomson_spectrum(wavelength_nm, laser_wavelength_nm, scattering_angle_rad, population_temperatures_eV, population_fractions, *, flow_projection_m_s, incident_polarization, radial_order=36, angular_order=18, maximum_recoil_correction_fraction=0.01, include_compton_recoil=False):
    """Return a cross-section-weighted N-component relativistic MJ spectrum.

    Fractions are number-density fractions and must be non-negative and sum
    to one. All components share the scattering-vector flow projection.
    Zero-fraction components are skipped. Absolute density is not inferred.
    """
    wavelength = _real_vector(wavelength_nm, "wavelength_nm", 7)
    temperatures = _real_vector(population_temperatures_eV, "population_temperatures_eV", 2)
    fractions = _real_vector(population_fractions, "population_fractions", 2)
    if temperatures.shape != fractions.shape: raise ValueError("temperature and fraction vectors must have identical lengths")
    if np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0): raise ValueError("wavelength_nm must be positive and strictly increasing")
    if np.any(temperatures <= 0): raise ValueError("population temperatures must be positive")
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0, rtol=1e-9, atol=1e-12): raise ValueError("population fractions must be non-negative and sum to one")
    laser, angle, flow = float(laser_wavelength_nm), float(scattering_angle_rad), float(flow_projection_m_s)
    if not np.isfinite(laser) or laser <= 0: raise ValueError("laser wavelength must be positive and finite")
    if not np.isfinite(angle) or not 0 < angle < np.pi: raise ValueError("scattering angle must lie in (0, pi)")
    if not np.isfinite(flow) or abs(flow) >= C: raise ValueError("flow projection must be finite and subluminal")
    if not isinstance(include_compton_recoil, (bool, np.bool_)): raise ValueError("include_compton_recoil must be boolean")
    components, weighted, total = [], [], 0.0
    for i, (temperature, fraction) in enumerate(zip(temperatures, fractions)):
        if fraction == 0:
            components.append({"index": int(i), "temperature_eV": float(temperature), "number_fraction": 0.0, "evaluated": False})
            continue
        result = relativistic_maxwell_juttner_spectrum(wavelength_nm=wavelength, laser_wavelength_nm=laser, scattering_angle_rad=angle, electron_temperature_eV=float(temperature), flow_projection_m_s=flow, incident_polarization=incident_polarization, radial_order=radial_order, angular_order=angular_order, maximum_recoil_correction_fraction=maximum_recoil_correction_fraction, include_compton_recoil=include_compton_recoil)
        cross = float(result["effective_differential_cross_section_m2_sr"])
        if not np.isfinite(cross) or cross <= 0: raise ValueError(f"population {i} has non-positive differential cross-section")
        weight = float(fraction) * cross
        total += weight
        weighted.append(weight * np.asarray(result["spectral_density_per_nm"], dtype=float))
        components.append({"index": int(i), "temperature_eV": float(temperature), "number_fraction": float(fraction), "power_weight_m2_sr": weight, "effective_differential_cross_section_m2_sr": cross, "evaluated": True})
    if not np.isfinite(total) or total <= 0: raise ValueError("mixture has no positive scattered-power contribution")
    spectrum = np.sum(weighted, axis=0) / total
    if spectrum.shape != wavelength.shape or not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0): raise ValueError("mixture produced an invalid spectrum")
    return {"spectral_density_per_nm": spectrum, "effective_differential_cross_section_m2_sr": float(total), "population_temperatures_eV": temperatures.copy(), "population_number_fractions": fractions.copy(), "population_components": components, "population_count": int(temperatures.size), "flow_projection_m_s": flow, "absolute_density_inferred": False, "model": "N-population isotropic relativistic Maxwell-Juttner Thomson mixture", "assumptions": ["Isotropic Maxwell-Juttner components.", "Shared bulk-flow projection along the scattering vector.", "Number fractions weighted by component differential scattered power.", "Not an arbitrary or anisotropic EVDF reconstruction.", "Absolute density is not inferred."]}


def fit_n_population_relativistic_thomson_spectrum(
    wavelength_nm, measured_spectrum, laser_wavelength_nm, scattering_angle_rad,
    instrument_response, *, population_count, initial_temperatures_eV,
    initial_fractions, initial_flow_projection_m_s, incident_polarization,
    signal_uncertainty=None, spectral_covariance=None,
    background_model="linear", background_degree=2,
    background_interior_knots=3, background_templates=None,
    model_discrepancy_covariance=None, radial_order=24, angular_order=12,
    maximum_recoil_correction_fraction=0.01, include_compton_recoil=False,
    max_nfev=1200,
):
    """Fit an explicitly selected N-population relativistic MJ mixture.

    Population temperatures are strictly ordered through positive log-gaps;
    number fractions are constrained to the simplex through a softmax
    parameterization. All populations share one scattering-vector flow
    projection and one measured response operator. Amplitude and the selected
    detector-space background basis are fitted jointly. The model is
    intentionally finite-dimensional and is not an unrestricted EVDF inverse.
    """
    from scipy.optimize import least_squares
    from scipy.special import softmax
    from .thomson_background import _background_design
    from .thomson_kinetic_models import _apply_response

    x = _real_vector(wavelength_nm, "wavelength_nm", 12)
    y = _real_vector(measured_spectrum, "measured_spectrum", 12)
    if x.shape != y.shape or np.any(x <= 0.0) or np.any(np.diff(x) <= 0.0) or np.any(y < 0.0):
        raise ValueError("wavelength_nm and measured_spectrum must match; wavelengths must be positive/increasing and measured intensity non-negative")
    if isinstance(population_count, (bool, np.bool_)) or not isinstance(population_count, (int, np.integer)) or not 2 <= int(population_count) <= 8:
        raise ValueError("population_count must be an integer from 2 to 8")
    count = int(population_count)
    temperatures = _real_vector(initial_temperatures_eV, "initial_temperatures_eV", count)
    fractions = _real_vector(initial_fractions, "initial_fractions", count)
    if temperatures.shape != (count,) or fractions.shape != (count,):
        raise ValueError("initial temperature and fraction vectors must each match population_count")
    if np.any(temperatures <= 0.0) or np.any(np.diff(temperatures) <= 0.0):
        raise ValueError("initial temperatures must be positive and strictly increasing to remove label switching")
    if np.any(fractions <= 0.0) or not np.isclose(fractions.sum(), 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError("initial_fractions must be strictly positive and sum to one")
    laser = float(laser_wavelength_nm)
    angle = float(scattering_angle_rad)
    flow0 = float(initial_flow_projection_m_s)
    if not np.isfinite(laser) or laser <= 0.0 or not np.isfinite(angle) or not 0.0 < angle < np.pi:
        raise ValueError("laser wavelength must be positive and scattering angle must lie strictly inside (0, pi)")
    if not np.isfinite(flow0) or abs(flow0) >= C:
        raise ValueError("initial flow projection must be finite and subluminal")
    if not isinstance(include_compton_recoil, (bool, np.bool_)):
        raise ValueError("include_compton_recoil must be boolean")
    if isinstance(max_nfev, (bool, np.bool_)) or not isinstance(max_nfev, (int, np.integer)) or max_nfev < 100:
        raise ValueError("max_nfev must be an integer >= 100")

    response = np.asarray(instrument_response, dtype=float)
    if (response.shape != (x.size, x.size) or np.any(~np.isfinite(response))
            or np.any(response < 0.0)
            or not np.allclose(response.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10)):
        raise ValueError("instrument_response must be a measured non-negative square matrix with unit-sum columns")
    polarization = _real_vector(incident_polarization, "incident_polarization", 3)
    if polarization.shape != (3,) or np.linalg.norm(polarization) <= 0.0:
        raise ValueError("incident_polarization must be a non-zero finite 3-vector")
    polarization = polarization / np.linalg.norm(polarization)
    basis = _background_design(
        x, background_model, degree=background_degree,
        interior_knots=background_interior_knots, templates=background_templates,
    )

    if signal_uncertainty is not None and spectral_covariance is not None:
        raise ValueError("supply signal_uncertainty or spectral_covariance, not both")
    covariance = None
    if signal_uncertainty is not None:
        sigma = _real_vector(signal_uncertainty, "signal_uncertainty", x.size)
        if sigma.shape != x.shape or np.any(sigma <= 0.0):
            raise ValueError("signal_uncertainty must be positive and match measured_spectrum")
        covariance = np.diag(sigma**2)
    elif spectral_covariance is not None:
        covariance = np.asarray(spectral_covariance, dtype=float)
        if (covariance.shape != (x.size, x.size) or np.any(~np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-13)):
            raise ValueError("spectral_covariance must be finite, symmetric and wavelength-sized")
    if model_discrepancy_covariance is not None:
        discrepancy = np.asarray(model_discrepancy_covariance, dtype=float)
        if (discrepancy.shape != (x.size, x.size) or np.any(~np.isfinite(discrepancy))
                or not np.allclose(discrepancy, discrepancy.T, rtol=1e-10, atol=1e-13)):
            raise ValueError("model_discrepancy_covariance must be finite, symmetric and wavelength-sized")
        eigenvalues = np.linalg.eigvalsh(discrepancy)
        scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        if float(np.min(eigenvalues)) < -1e-10 * scale:
            raise ValueError("model_discrepancy_covariance must be positive semidefinite")
        covariance = discrepancy.copy() if covariance is None else covariance + discrepancy
    if covariance is None:
        whiten = lambda values: np.asarray(values, dtype=float)
        noise_model = "unweighted_no_measurement_uncertainty_supplied"
    else:
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("combined measurement and discrepancy covariance must be positive definite") from exc
        whiten = lambda values: np.linalg.solve(chol, np.asarray(values, dtype=float))
        noise_model = "full_GLS_covariance" if model_discrepancy_covariance is None else "measurement_plus_model_discrepancy_GLS_covariance"

    # Coordinates: log(T0), log-positive temperature gaps, N-1 logits,
    # atanh(v/c), log(amplitude), and unconstrained background coefficients.
    log_temperatures = np.log(temperatures)
    log_gaps = np.log(np.diff(temperatures))
    fraction_logits = np.log(fractions[:-1] / fractions[-1])
    baseline = float(np.quantile(y, 0.05))
    amplitude0 = max(float(np.max(y) - baseline), np.finfo(float).tiny)
    p0 = np.r_[
        log_temperatures[0], log_gaps, fraction_logits,
        np.arctanh(flow0 / C), np.log(amplitude0),
        np.zeros(basis.shape[1]),
    ]

    def unpack(parameters):
        first_temperature = float(np.exp(np.clip(parameters[0], -700.0, 700.0)))
        gaps = np.exp(np.clip(parameters[1:count], -700.0, 700.0))
        population_temperatures = np.r_[first_temperature, first_temperature + np.cumsum(gaps)]
        logits = np.r_[parameters[count:count + count - 1], 0.0]
        population_fractions = softmax(logits)
        flow = float(np.tanh(parameters[2 * count - 1]) * C)
        amplitude = float(np.exp(np.clip(parameters[2 * count], -700.0, 700.0)))
        background_coefficients = np.asarray(parameters[2 * count + 1:], dtype=float)
        return population_temperatures, population_fractions, flow, amplitude, background_coefficients

    def predict(parameters, return_parts=False):
        temp, frac, flow, amplitude, coefficients = unpack(parameters)
        physical = n_population_relativistic_thomson_spectrum(
            x, laser, angle, temp, frac,
            flow_projection_m_s=flow, incident_polarization=polarization,
            radial_order=radial_order, angular_order=angular_order,
            maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
            include_compton_recoil=include_compton_recoil,
        )
        shape = _apply_response(response, physical["spectral_density_per_nm"], x)
        plasma = amplitude * shape
        background = basis @ coefficients
        fitted = plasma + background
        return (fitted, physical, plasma, background) if return_parts else fitted

    def residual(parameters):
        return whiten(predict(parameters) - y)

    fit = least_squares(
        residual, p0, method="trf", x_scale="jac", max_nfev=int(max_nfev)
    )
    if not fit.success or np.any(~np.isfinite(fit.x)):
        raise ValueError("N-population relativistic Thomson optimizer failed: " + str(fit.message))
    jacobian = np.asarray(fit.jac, dtype=float)
    column_norms = np.linalg.norm(jacobian, axis=0)
    scaled_jacobian = np.divide(
        jacobian, column_norms[None, :],
        out=np.zeros_like(jacobian),
        where=column_norms[None, :] > np.finfo(float).tiny,
    )
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    tolerance = np.finfo(float).eps * max(scaled_jacobian.shape) * (
        singular_values[0] if singular_values.size else 0.0
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0 else float("inf")
    )
    if rank < fit.x.size or not np.isfinite(condition) or condition > 1e12:
        raise ValueError(
            f"N-population fit is not identifiable (rank {rank}/{fit.x.size}, condition={condition:.6g}); "
            "reduce population_count or improve spectral information"
        )

    covariance_internal = np.linalg.pinv(jacobian.T @ jacobian, rcond=1e-12)
    dof = max(1, y.size - fit.x.size)
    if covariance is None:
        covariance_internal *= float(fit.fun @ fit.fun / dof)

    # Transform the complete covariance to physical parameters and nuisance
    # coordinates with a bound-aware central finite-difference Jacobian.
    def reported_vector(parameters):
        temp, frac, flow, amplitude, coefficients = unpack(parameters)
        return np.r_[temp, frac, flow, amplitude, coefficients]
    physical_jacobian = np.empty((2 * count + 2 + basis.shape[1], fit.x.size))
    for index, value in enumerate(fit.x):
        step = np.cbrt(np.finfo(float).eps) * max(1.0, abs(float(value)))
        plus, minus = fit.x.copy(), fit.x.copy()
        plus[index] += step
        minus[index] -= step
        physical_jacobian[:, index] = (
            reported_vector(plus) - reported_vector(minus)
        ) / (2.0 * step)
    joint_covariance = physical_jacobian @ covariance_internal @ physical_jacobian.T
    fitted, physical, plasma, background = predict(fit.x, return_parts=True)
    temp, frac, flow, amplitude, coefficients = unpack(fit.x)
    parameter_names = (
        [f"population_{i}_temperature_eV" for i in range(count)]
        + [f"population_{i}_number_fraction" for i in range(count)]
        + ["electron_flow_projection_m_s", "spectral_amplitude"]
        + [f"background_coefficient_{i}" for i in range(basis.shape[1])]
    )
    return {
        "model": "N_population_relativistic_Maxwell_Juttner_optical_Thomson",
        "population_count": count,
        "population_temperatures_eV": temp,
        "population_number_fractions": frac,
        "electron_flow_projection_m_s": flow,
        "spectral_amplitude": amplitude,
        "background_model": background_model,
        "background_coefficients": coefficients,
        "fitted_thomson_spectrum": plasma,
        "fitted_background": background,
        "fitted_spectrum": fitted,
        "residual": y - fitted,
        "parameter_names": parameter_names,
        "joint_parameter_covariance": joint_covariance,
        "parameter_standard_uncertainties": np.sqrt(
            np.maximum(np.diag(joint_covariance), 0.0)
        ),
        "temperature_covariance_eV2": joint_covariance[:count, :count],
        "fraction_covariance": joint_covariance[count:2*count, count:2*count],
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "jacobian_rank": rank,
        "jacobian_parameter_count": int(fit.x.size),
        "jacobian_condition_number": condition,
        "degrees_of_freedom": dof,
        "noise_model": noise_model,
        "model_discrepancy_covariance_included": model_discrepancy_covariance is not None,
        "covariance_status": (
            "residual_scaled_unweighted_joint_covariance_not_measurement_uncertainty"
            if covariance is None else
            "local_joint_GLS_covariance_conditional_on_supplied_noise_and_fixed_response"
        ),
        "population_assumptions": [
            "Finite N-component isotropic Maxwell-Juttner mixture.",
            "Strictly ordered component temperatures remove label switching.",
            "Number fractions are constrained to the unit simplex.",
            "All components share one flow projection along the scattering vector.",
            "Amplitude and detector-space background are nuisance parameters.",
            "A finite-dimensional mixture is not an unrestricted electron velocity distribution.",
        ],
        "absolute_density_inferred": False,
        "experimental_validation": "not_performed",
    }
