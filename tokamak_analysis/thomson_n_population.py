"""N-population relativistic optical Thomson forward model.

Each population is an isotropic Maxwell-Juttner component with its own
temperature, number-density fraction and (optionally) scattering-vector flow
projection. Mixture power is weighted by component differential cross-sections,
not temperature averages. This is a parametric mixture, not arbitrary-EVDF inversion.
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

def n_population_relativistic_thomson_spectrum(wavelength_nm, laser_wavelength_nm, scattering_angle_rad, population_temperatures_eV, population_fractions, *, incident_polarization, flow_projection_m_s=None, population_flow_projections_m_s=None, radial_order=36, angular_order=18, maximum_recoil_correction_fraction=0.01, include_compton_recoil=False):
    """Return a cross-section-weighted N-component relativistic MJ spectrum.

    Fractions are number-density fractions and must be non-negative and sum
    to one. Supply either one shared flow_projection_m_s or one
    population_flow_projections_m_s value per component. Zero-fraction
    components are skipped. Absolute density is not inferred.
    """
    wavelength = _real_vector(wavelength_nm, "wavelength_nm", 7)
    temperatures = _real_vector(population_temperatures_eV, "population_temperatures_eV", 2)
    fractions = _real_vector(population_fractions, "population_fractions", 2)
    if temperatures.shape != fractions.shape: raise ValueError("temperature and fraction vectors must have identical lengths")
    if np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0): raise ValueError("wavelength_nm must be positive and strictly increasing")
    if np.any(temperatures <= 0): raise ValueError("population temperatures must be positive")
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0, rtol=1e-9, atol=1e-12): raise ValueError("population fractions must be non-negative and sum to one")
    laser, angle = float(laser_wavelength_nm), float(scattering_angle_rad)
    if not np.isfinite(laser) or laser <= 0: raise ValueError("laser wavelength must be positive and finite")
    if not np.isfinite(angle) or not 0 < angle < np.pi: raise ValueError("scattering angle must lie in (0, pi)")
    if (flow_projection_m_s is None) == (population_flow_projections_m_s is None):
        raise ValueError("supply exactly one of flow_projection_m_s or population_flow_projections_m_s")
    if population_flow_projections_m_s is None:
        flow = float(flow_projection_m_s)
        flows = np.full(temperatures.size, flow, dtype=float)
    else:
        flows = _real_vector(population_flow_projections_m_s, "population_flow_projections_m_s", temperatures.size)
        if flows.shape != temperatures.shape:
            raise ValueError("population_flow_projections_m_s must match the population count")
        flow = None
    if np.any(~np.isfinite(flows)) or np.any(np.abs(flows) >= C):
        raise ValueError("every population flow projection must be finite and subluminal")
    if not isinstance(include_compton_recoil, (bool, np.bool_)): raise ValueError("include_compton_recoil must be boolean")
    components, weighted, total = [], [], 0.0
    for i, (temperature, fraction, population_flow) in enumerate(zip(temperatures, fractions, flows)):
        if fraction == 0:
            components.append({"index": int(i), "temperature_eV": float(temperature), "number_fraction": 0.0, "evaluated": False})
            continue
        result = relativistic_maxwell_juttner_spectrum(wavelength_nm=wavelength, laser_wavelength_nm=laser, scattering_angle_rad=angle, electron_temperature_eV=float(temperature), flow_projection_m_s=float(population_flow), incident_polarization=incident_polarization, radial_order=radial_order, angular_order=angular_order, maximum_recoil_correction_fraction=maximum_recoil_correction_fraction, include_compton_recoil=include_compton_recoil)
        cross = float(result["effective_differential_cross_section_m2_sr"])
        if not np.isfinite(cross) or cross <= 0: raise ValueError(f"population {i} has non-positive differential cross-section")
        weight = float(fraction) * cross
        total += weight
        weighted.append(weight * np.asarray(result["spectral_density_per_nm"], dtype=float))
        components.append({"index": int(i), "temperature_eV": float(temperature), "number_fraction": float(fraction), "power_weight_m2_sr": weight, "effective_differential_cross_section_m2_sr": cross, "flow_projection_m_s": float(population_flow), "evaluated": True})
    if not np.isfinite(total) or total <= 0: raise ValueError("mixture has no positive scattered-power contribution")
    spectrum = np.sum(weighted, axis=0) / total
    if spectrum.shape != wavelength.shape or not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0): raise ValueError("mixture produced an invalid spectrum")
    return {"spectral_density_per_nm": spectrum, "effective_differential_cross_section_m2_sr": float(total), "population_temperatures_eV": temperatures.copy(), "population_number_fractions": fractions.copy(), "population_components": components, "population_count": int(temperatures.size), "flow_projection_m_s": (float(flows[0]) if np.allclose(flows, flows[0]) else None), "population_flow_projections_m_s": flows.copy(), "absolute_density_inferred": False, "model": "N-population isotropic relativistic Maxwell-Juttner Thomson mixture", "assumptions": ["Isotropic Maxwell-Juttner components.", "Each population has its own scattering-vector flow projection when supplied; a shared-flow closure remains available.", "Number fractions weighted by component differential scattered power.", "Not an arbitrary or anisotropic EVDF reconstruction.", "Absolute density is not inferred."]}


def fit_n_population_relativistic_thomson_spectrum(
    wavelength_nm, measured_spectrum, laser_wavelength_nm, scattering_angle_rad,
    instrument_response, *, population_count, initial_temperatures_eV,
    initial_fractions, initial_flow_projection_m_s, incident_polarization,
    initial_population_flow_projections_m_s=None,
    signal_uncertainty=None, spectral_covariance=None,
    background_model="linear", background_degree=2,
    background_interior_knots=3, background_templates=None,
    model_discrepancy_covariance=None, systematic_covariance=None,
    geometry_covariance_nm2_rad2=None,
    radial_order=24, angular_order=12,
    maximum_recoil_correction_fraction=0.01, include_compton_recoil=False,
    max_nfev=1200,
):
    """Fit an explicitly selected N-population relativistic MJ mixture.

    Population temperatures are strictly ordered through positive log-gaps;
    number fractions are constrained to the simplex through a softmax
    parameterization. By default all populations share one flow projection;
    supplying initial_population_flow_projections_m_s fits an independent
    subluminal projection for every population. The measured response operator,
    amplitude and detector-space background are fitted jointly.

    The physical mixture is cross-section weighted:
      S_mix(lambda) = sum_j [f_j sigma_j S_j(lambda)] / sum_k [f_k sigma_k].
    The GLS objective uses the additive covariance closure:
      Sigma_total = Sigma_measurement + Sigma_model_discrepancy + Sigma_systematic,
      chi2 = (S_model-y)^T Sigma_total^-1 (S_model-y).
    Each additive covariance is validated as positive semidefinite; the total
    must be positive definite to whiten the residuals. This remains a
    finite-dimensional parametric mixture, not an unrestricted EVDF inverse.
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
    independent_population_flows = initial_population_flow_projections_m_s is not None
    if independent_population_flows:
        flow_initial = _real_vector(initial_population_flow_projections_m_s, "initial_population_flow_projections_m_s", count)
        if flow_initial.shape != (count,) or np.any(np.abs(flow_initial) >= C):
            raise ValueError("initial_population_flow_projections_m_s must contain one finite subluminal value per population")
    else:
        flow_initial = np.full(count, flow0, dtype=float)
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
    covariance_components = {
        "measurement": None if covariance is None else covariance.copy(),
        "model_discrepancy": None,
        "systematic": None,
    }

    def _validated_additive_covariance(value, name):
        matrix = np.asarray(value, dtype=float)
        if (matrix.shape != (x.size, x.size) or np.any(~np.isfinite(matrix))
                or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-13)):
            raise ValueError(f"{name} must be finite, symmetric and wavelength-sized")
        eigenvalues = np.linalg.eigvalsh(matrix)
        scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        if float(np.min(eigenvalues)) < -1e-10 * scale:
            raise ValueError(f"{name} must be positive semidefinite")
        return 0.5 * (matrix + matrix.T)

    if model_discrepancy_covariance is not None:
        discrepancy = _validated_additive_covariance(
            model_discrepancy_covariance, "model_discrepancy_covariance"
        )
        covariance_components["model_discrepancy"] = discrepancy.copy()
        covariance = discrepancy.copy() if covariance is None else covariance + discrepancy
    if systematic_covariance is not None:
        systematic = _validated_additive_covariance(
            systematic_covariance, "systematic_covariance"
        )
        covariance_components["systematic"] = systematic.copy()
        covariance = systematic.copy() if covariance is None else covariance + systematic

    geometry_covariance = None
    if geometry_covariance_nm2_rad2 is not None:
        geometry_covariance = np.asarray(geometry_covariance_nm2_rad2, dtype=float)
        if (geometry_covariance.shape != (2, 2)
                or np.any(~np.isfinite(geometry_covariance))
                or not np.allclose(geometry_covariance, geometry_covariance.T,
                                   rtol=1e-10, atol=1e-13)):
            raise ValueError(
                "geometry_covariance_nm2_rad2 must be a finite symmetric 2x2 "
                "covariance for [laser_wavelength_nm, scattering_angle_rad]"
            )
        geometry_eigenvalues = np.linalg.eigvalsh(geometry_covariance)
        geometry_scale = max(float(np.max(np.abs(geometry_eigenvalues))), np.finfo(float).tiny)
        if float(np.min(geometry_eigenvalues)) < -1e-10 * geometry_scale:
            raise ValueError("geometry_covariance_nm2_rad2 must be positive semidefinite")
        geometry_covariance = 0.5 * (geometry_covariance + geometry_covariance.T)

    if covariance is None:
        whiten = lambda values: np.asarray(values, dtype=float)
        noise_model = "unweighted_no_measurement_uncertainty_supplied"
    else:
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "combined measurement, model-discrepancy and systematic covariance "
                "must be positive definite"
            ) from exc
        whiten = lambda values: np.linalg.solve(chol, np.asarray(values, dtype=float))
        active_sources = [
            name for name, matrix in covariance_components.items()
            if matrix is not None
        ]
        noise_model = "GLS_" + "_plus_".join(active_sources) + "_covariance"

    # Coordinates: log(T0), log-positive temperature gaps, N-1 logits,
    # one or N atanh(v_j/c) flow coordinates, log(amplitude), background.
    log_temperatures = np.log(temperatures)
    log_gaps = np.log(np.diff(temperatures))
    fraction_logits = np.log(fractions[:-1] / fractions[-1])
    baseline = float(np.quantile(y, 0.05))
    amplitude0 = max(float(np.max(y) - baseline), np.finfo(float).tiny)
    p0 = np.r_[
        log_temperatures[0], log_gaps, fraction_logits,
        np.arctanh(flow_initial / C) if independent_population_flows else np.asarray([np.arctanh(flow0 / C)]),
        np.log(amplitude0), np.zeros(basis.shape[1]),
    ]

    def unpack(parameters):
        first_temperature = float(np.exp(np.clip(parameters[0], -700.0, 700.0)))
        gaps = np.exp(np.clip(parameters[1:count], -700.0, 700.0))
        population_temperatures = np.r_[first_temperature, first_temperature + np.cumsum(gaps)]
        logits = np.r_[parameters[count:count + count - 1], 0.0]
        population_fractions = softmax(logits)
        flow_start = 2 * count - 1
        n_flow = count if independent_population_flows else 1
        flows = np.tanh(parameters[flow_start:flow_start + n_flow]) * C
        if not independent_population_flows:
            flows = np.full(count, float(flows[0]))
        amplitude_index = flow_start + n_flow
        amplitude = float(np.exp(np.clip(parameters[amplitude_index], -700.0, 700.0)))
        background_coefficients = np.asarray(parameters[amplitude_index + 1:], dtype=float)
        return population_temperatures, population_fractions, flows, amplitude, background_coefficients

    def predict(parameters, return_parts=False, *, laser_value=None, angle_value=None):
        temp, frac, flows, amplitude, coefficients = unpack(parameters)
        evaluated_laser = laser if laser_value is None else float(laser_value)
        evaluated_angle = angle if angle_value is None else float(angle_value)
        physical = n_population_relativistic_thomson_spectrum(
            x, evaluated_laser, evaluated_angle, temp, frac,
            population_flow_projections_m_s=flows, incident_polarization=polarization,
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

    normal_inverse = np.linalg.pinv(jacobian.T @ jacobian, rcond=1e-12)
    covariance_internal = normal_inverse.copy()
    dof = max(1, y.size - fit.x.size)
    if covariance is None:
        covariance_internal *= float(fit.fun @ fit.fun / dof)

    geometry_parameter_sensitivity = None
    geometry_induced_internal_covariance = None
    if geometry_covariance is not None and np.any(geometry_covariance):
        # Implicit differentiation of the GLS normal equations:
        # dp/dg = -(J_p.T J_p)^-1 J_p.T J_g.
        # This propagates correlated laser/angle calibration errors into the
        # fitted population temperatures, fractions, flows and nuisance terms.
        geometry_jacobian = np.empty((y.size, 2), dtype=float)
        geometry_values = (laser, angle)
        for geometry_index, geometry_value in enumerate(geometry_values):
            step = np.cbrt(np.finfo(float).eps) * max(1.0, abs(geometry_value))
            plus = list(geometry_values)
            minus = list(geometry_values)
            plus[geometry_index] += step
            minus[geometry_index] -= step
            if geometry_index == 0 and minus[geometry_index] <= 0.0:
                minus[geometry_index] = geometry_value
                denominator = plus[geometry_index] - minus[geometry_index]
                derivative = (
                    whiten(predict(fit.x, laser_value=plus[0], angle_value=plus[1]) - y)
                    - whiten(predict(fit.x, laser_value=minus[0], angle_value=minus[1]) - y)
                ) / denominator
            elif geometry_index == 1 and not 0.0 < minus[geometry_index] < np.pi:
                raise ValueError("scattering-angle uncertainty step leaves the physical interval (0, pi)")
            else:
                derivative = (
                    whiten(predict(fit.x, laser_value=plus[0], angle_value=plus[1]) - y)
                    - whiten(predict(fit.x, laser_value=minus[0], angle_value=minus[1]) - y)
                ) / (2.0 * step)
            geometry_jacobian[:, geometry_index] = derivative
        geometry_parameter_sensitivity = (
            -normal_inverse @ jacobian.T @ geometry_jacobian
        )
        geometry_induced_internal_covariance = (
            geometry_parameter_sensitivity @ geometry_covariance
            @ geometry_parameter_sensitivity.T
        )
        covariance_internal += geometry_induced_internal_covariance

    # Transform the complete covariance to physical parameters and nuisance
    # coordinates with a bound-aware central finite-difference Jacobian.
    def reported_vector(parameters):
        temp, frac, flows, amplitude, coefficients = unpack(parameters)
        flow_report = flows if independent_population_flows else np.asarray([flows[0]])
        return np.r_[temp, frac, flow_report, amplitude, coefficients]
    physical_jacobian = np.empty((2 * count + (count if independent_population_flows else 1) + 1 + basis.shape[1], fit.x.size))
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
    temp, frac, flows, amplitude, coefficients = unpack(fit.x)
    flow_report = flows if independent_population_flows else np.asarray([flows[0]])
    parameter_names = (
        [f"population_{i}_temperature_eV" for i in range(count)]
        + [f"population_{i}_number_fraction" for i in range(count)]
        + ([f"population_{i}_flow_projection_m_s" for i in range(count)] if independent_population_flows else ["electron_flow_projection_m_s"])
        + ["spectral_amplitude"]
        + [f"background_coefficient_{i}" for i in range(basis.shape[1])]
    )
    return {
        "model": "N_population_relativistic_Maxwell_Juttner_optical_Thomson",
        "population_count": count,
        "population_temperatures_eV": temp,
        "population_number_fractions": frac,
        "electron_flow_projection_m_s": (float(flows[0]) if not independent_population_flows else None),
        "population_flow_projections_m_s": flows.copy(),
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
        "systematic_covariance_included": systematic_covariance is not None,
        "geometry_covariance_included": geometry_covariance is not None,
        "geometry_covariance_nm2_rad2": (
            None if geometry_covariance is None else geometry_covariance.copy()
        ),
        "geometry_parameter_sensitivity_internal_per_nm_rad": geometry_parameter_sensitivity,
        "geometry_induced_internal_parameter_covariance": geometry_induced_internal_covariance,
        "covariance_component_trace": {
            name: (None if matrix is None else float(np.trace(matrix)))
            for name, matrix in covariance_components.items()
        },
        "total_spectral_covariance": covariance.copy() if covariance is not None else None,
        "covariance_status": (
            "residual_scaled_unweighted_joint_covariance_not_measurement_uncertainty"
            if covariance is None else
            "local_joint_GLS_covariance_with_additive_measurement_model_discrepancy_systematic_and_delta_method_geometry_uncertainty_conditional_on_fixed_response"
        ),
        "population_assumptions": [
            "Finite N-component isotropic Maxwell-Juttner mixture.",
            "Strictly ordered component temperatures remove label switching.",
            "Number fractions are constrained to the unit simplex.",
            "A shared flow projection is the default; independent per-population projections are fitted only when explicitly initialized.",
            "Amplitude and detector-space background are nuisance parameters.",
            "A finite-dimensional mixture is not an unrestricted electron velocity distribution.",
        ],
        "absolute_density_inferred": False,
        "experimental_validation": "not_performed",
    }
