"""Two-population relativistic optical Thomson spectral model.

The model represents an isotropic electron distribution as an explicit
density-weighted mixture of two Maxwell-Juttner populations. It is a compact
parametric approximation to the arbitrary-VDF forward-modelling direction
demonstrated by Sakai et al. (JINST, 2026); it is not their Monte Carlo
algorithm and does not claim unrestricted VDF reconstruction.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit

from .thomson_kinetic_models import (
    C,
    relativistic_maxwell_juttner_spectrum,
    _apply_response,
)


def _vector(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    out = raw.astype(float, copy=False)
    if out.ndim != 1 or out.size < 7 or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite one-dimensional vector with at least seven samples")
    return out


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be positive and finite")
    out = float(value)
    if not np.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def two_population_relativistic_thomson_spectrum(
    wavelength_nm,
    laser_wavelength_nm,
    scattering_angle_rad,
    cold_temperature_eV,
    hot_temperature_eV,
    hot_population_fraction,
    *,
    flow_projection_m_s,
    incident_polarization,
    radial_order=36,
    angular_order=18,
    maximum_recoil_correction_fraction=0.01,
    include_compton_recoil=False,
):
    """Return the cross-section-weighted spectrum of two MJ electron populations.

    The hot fraction is a number-density fraction in [0, 1]. Both populations
    share one bulk-flow projection and incident polarization. Each population
    is evaluated with the same relativistic photon kinematics and its own
    Maxwell-Juttner momentum distribution; their scattered powers are summed
    before normalization.
    """
    wavelength = _vector(wavelength_nm, "wavelength_nm")
    if np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0):
        raise ValueError("wavelength_nm must be positive and strictly increasing")
    laser = _positive(laser_wavelength_nm, "laser_wavelength_nm")
    angle = float(scattering_angle_rad)
    if not np.isfinite(angle) or not 0 < angle < np.pi:
        raise ValueError("scattering_angle_rad must lie strictly inside (0, pi)")
    tc = _positive(cold_temperature_eV, "cold_temperature_eV")
    th = _positive(hot_temperature_eV, "hot_temperature_eV")
    if th <= tc:
        raise ValueError("hot_temperature_eV must be greater than cold_temperature_eV")
    fraction = float(hot_population_fraction)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("hot_population_fraction must lie in [0, 1]")
    flow = float(flow_projection_m_s)
    if not np.isfinite(flow) or abs(flow) >= C:
        raise ValueError("flow_projection_m_s must be finite and subluminal")

    common = dict(
        wavelength_nm=wavelength,
        laser_wavelength_nm=laser,
        scattering_angle_rad=angle,
        flow_projection_m_s=flow,
        incident_polarization=incident_polarization,
        radial_order=radial_order,
        angular_order=angular_order,
        maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
        include_compton_recoil=include_compton_recoil,
    )
    cold = relativistic_maxwell_juttner_spectrum(
        electron_temperature_eV=tc, **common
    )
    hot = relativistic_maxwell_juttner_spectrum(
        electron_temperature_eV=th, **common
    )
    cold_weight = (1.0 - fraction) * cold["effective_differential_cross_section_m2_sr"]
    hot_weight = fraction * hot["effective_differential_cross_section_m2_sr"]
    total_cross_section = cold_weight + hot_weight
    if not np.isfinite(total_cross_section) or total_cross_section <= 0:
        raise ValueError("two-population effective differential cross-section is non-positive")
    mixed = (
        cold_weight * cold["spectral_density_per_nm"]
        + hot_weight * hot["spectral_density_per_nm"]
    ) / total_cross_section
    return {
        "spectral_density_per_nm": mixed,
        "effective_differential_cross_section_m2_sr": float(total_cross_section),
        "cold_population_fraction": float(1.0 - fraction),
        "hot_population_fraction": float(fraction),
        "cold_temperature_eV": float(tc),
        "hot_temperature_eV": float(th),
        "flow_projection_m_s": float(flow),
        "component_spectra_per_nm": {
            "cold": cold["spectral_density_per_nm"],
            "hot": hot["spectral_density_per_nm"],
        },
        "model": "two-population isotropic relativistic Maxwell-Juttner Thomson mixture",
        "population_assumption": "two isotropic Maxwell-Juttner populations sharing one scattering-vector flow projection",
        "absolute_density_inferred": False,
    }


def fit_two_population_relativistic_thomson_spectrum(
    wavelength_nm,
    measured_spectrum,
    laser_wavelength_nm,
    scattering_angle_rad,
    instrument_response,
    *,
    incident_polarization,
    cold_temperature_initial_eV=None,
    hot_temperature_initial_eV=None,
    hot_population_fraction_initial=None,
    flow_projection_initial_m_s=None,
    signal_uncertainty=None,
    spectral_covariance=None,
    radial_order=24,
    angular_order=12,
    maximum_recoil_correction_fraction=0.01,
    include_compton_recoil=False,
    background_degree=0,
    background_model=None,
    background_interior_knots=3,
    background_templates=None,
    max_nfev=500,
):
    """Fit the two-population model directly to measured calibrated spectra.

    The measured response is applied as a fixed forward operator. Signal amplitude
    and a structured detector-space polynomial background are jointly profiled
    with the nonlinear population parameters. The polynomial degree is an explicit
    modelling choice (0=constant, 1=linear, 2=quadratic); it is not a substitute
    for a measured background template. Absolute density is deliberately not
    inferred by this normalized shape-only subanalysis.
    """
    x = _vector(wavelength_nm, "wavelength_nm")
    y = _vector(measured_spectrum, "measured_spectrum")
    if x.shape != y.shape or np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm and measured_spectrum must match; wavelength must increase strictly and remain positive")
    response = np.asarray(instrument_response, dtype=float)
    if response.shape != (x.size, x.size) or not np.all(np.isfinite(response)) or np.any(response < 0):
        raise ValueError("instrument_response must be a finite non-negative square matrix on the measured wavelength grid")
    if not np.allclose(response.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("measured instrument_response columns must already sum to one; the supplied calibrated operator is not renormalized")
    if np.any(y < 0):
        raise ValueError("measured calibrated spectral intensity must be non-negative")
    if signal_uncertainty is not None and spectral_covariance is not None:
        raise ValueError("Supply signal_uncertainty or spectral_covariance, not both")
    if signal_uncertainty is None and spectral_covariance is None:
        whitening = lambda z: z
        uncertainty_status = "unweighted; residual-scaled covariance is not measurement-noise-calibrated"
    elif signal_uncertainty is not None:
        sigma = _vector(signal_uncertainty, "signal_uncertainty")
        if sigma.shape != y.shape or np.any(sigma <= 0):
            raise ValueError("signal_uncertainty must match measured_spectrum and be strictly positive")
        whitening = lambda z: (np.asarray(z) / sigma[:, None]) if np.asarray(z).ndim == 2 else (np.asarray(z) / sigma)
        uncertainty_status = "supplied pointwise 1-sigma uncertainty"
    else:
        covariance = np.asarray(spectral_covariance, dtype=float)
        if covariance.shape != (x.size, x.size) or not np.all(np.isfinite(covariance)) or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("spectral_covariance must be finite, symmetric and match the measured wavelength grid")
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("spectral_covariance must be positive definite") from exc
        whitening = lambda z: np.linalg.solve(cholesky, z)
        uncertainty_status = "supplied full spectral covariance"

    if isinstance(background_degree, (bool, np.bool_)) or not isinstance(background_degree, (int, np.integer)) or not 0 <= int(background_degree) <= 8:
        raise ValueError("background_degree must be an integer in {0, ..., 8}")
    background_degree = int(background_degree)
    if background_model is None:
        background_model = "polynomial" if background_degree else "constant"
    from .thomson_background import _background_design
    background_basis = _background_design(
        x, background_model, degree=background_degree,
        interior_knots=background_interior_knots, templates=background_templates,
    )
    if not isinstance(include_compton_recoil, (bool, np.bool_)):
        raise ValueError("include_compton_recoil must be boolean")

    laser = _positive(laser_wavelength_nm, "laser_wavelength_nm")
    angle = float(scattering_angle_rad)
    if not np.isfinite(angle) or not 0 < angle < np.pi:
        raise ValueError("scattering_angle_rad must lie strictly inside (0, pi)")

    supplied_initial = (cold_temperature_initial_eV, hot_temperature_initial_eV,
                        hot_population_fraction_initial, flow_projection_initial_m_s)
    if all(value is None for value in supplied_initial):
        # Deterministic moment-based seeds are optimizer aids only; they are
        # never reported as physical measurements or substituted for the fit.
        baseline = float(np.quantile(y, 0.05))
        weights = np.maximum(y - baseline, 0.0)
        area = float(np.sum(weights))
        if not np.isfinite(area) or area <= np.finfo(float).tiny:
            raise ValueError("automatic two-population initialization failed: spectrum has no positive contrast above its 5th-percentile baseline")
        centroid = float(np.sum(weights * x) / area)
        variance = float(np.sum(weights * (x - centroid) ** 2) / area)
        if not np.isfinite(variance) or variance <= 0.0:
            raise ValueError("automatic two-population initialization failed: measured spectrum has no positive wavelength variance")
        # Width-to-Te conversion is used only for seeds. The relativistic
        # Maxwell-Juttner forward model remains the fitted physical model.
        width_temperature = 510_998.95069 * variance / (
            2.0 * laser**2 * np.sin(angle / 2.0)**2
        )
        width_temperature = float(np.clip(width_temperature, 1e-3, 2e5))
        flow_seed = -C * (centroid / laser - 1.0) / (2.0 * np.sin(angle / 2.0))
        flow_seed = float(np.clip(flow_seed, -0.2*C, 0.2*C))
        seed_specs = [
            (max(0.35*width_temperature, 1e-3), max(4.0*width_temperature, 2e-3), 0.08, flow_seed),
            (max(0.15*width_temperature, 1e-3), max(2.5*width_temperature, 2e-3), 0.18, flow_seed),
            (max(0.60*width_temperature, 1e-3), max(8.0*width_temperature, 2e-3), 0.04, flow_seed),
        ]
        initialization_status = "automatic_deterministic_spectral_moment_multistart"
    elif any(value is None for value in supplied_initial):
        raise ValueError("Supply either all four optimizer initial values or none; partial physical-parameter seeds are not accepted")
    else:
        tc0 = _positive(cold_temperature_initial_eV, "cold_temperature_initial_eV")
        th0 = _positive(hot_temperature_initial_eV, "hot_temperature_initial_eV")
        if th0 <= tc0:
            raise ValueError("hot_temperature_initial_eV must exceed cold_temperature_initial_eV")
        fraction0 = float(hot_population_fraction_initial)
        if not np.isfinite(fraction0) or not 0 < fraction0 < 1:
            raise ValueError("hot_population_fraction_initial must lie strictly inside (0, 1)")
        flow0 = float(flow_projection_initial_m_s)
        if not np.isfinite(flow0) or abs(flow0) >= C:
            raise ValueError("flow_projection_initial_m_s must be finite and subluminal")
        seed_specs = [(tc0, th0, fraction0, flow0)]
        initialization_status = "researcher_supplied_optimizer_seed"

    scale_flow = 1.0e6
    initial_vectors = [
        np.array([np.log(tc0), np.log(th0 - tc0), logit(fraction0), flow0 / scale_flow])
        for tc0, th0, fraction0, flow0 in seed_specs
    ]
    lower = np.array([np.log(np.finfo(float).tiny) + 5, np.log(np.finfo(float).tiny) + 5, -30.0, -C / scale_flow * (1.0 - 1e-12)])
    upper = np.array([np.log(np.finfo(float).max) - 5, np.log(np.finfo(float).max) - 5, 30.0, C / scale_flow * (1.0 - 1e-12)])

    def decode(p):
        tc = float(np.exp(p[0]))
        th = float(tc + np.exp(p[1]))
        fraction = float(expit(p[2]))
        flow = float(p[3] * scale_flow)
        return tc, th, fraction, flow

    def forward(p):
        tc, th, fraction, flow = decode(p)
        model = two_population_relativistic_thomson_spectrum(
            x, laser, angle, tc, th, fraction,
            flow_projection_m_s=flow,
            incident_polarization=incident_polarization,
            radial_order=radial_order,
            angular_order=angular_order,
            maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
            include_compton_recoil=include_compton_recoil,
        )["spectral_density_per_nm"]
        return _apply_response(response, model, x)

    # The shared detector-space background basis is fitted simultaneously
    # with the relativistic Thomson signal.
    x_center = float(np.mean(x))
    x_scale = float(np.ptp(x))

    def profile(p):
        z = forward(p)
        design_unwhitened = np.column_stack((z, background_basis))
        design = whitening(design_unwhitened)
        coefficients = np.linalg.lstsq(design, whitening(y), rcond=None)[0]
        amplitude = float(coefficients[0])
        if amplitude < 0:
            amplitude = 0.0
            background_coefficients = np.linalg.lstsq(
                whitening(background_basis), whitening(y), rcond=None
            )[0]
        else:
            background_coefficients = coefficients[1:]
        background = background_basis @ background_coefficients
        fitted = amplitude * z + background
        return amplitude, background_coefficients, background, fitted

    def residual(p):
        _, _, _, fitted = profile(p)
        return whitening(fitted-y)

    fit_candidates = []
    fit_failures = []
    for initial in initial_vectors:
        try:
            candidate = least_squares(
                residual, initial, bounds=(lower, upper), x_scale="jac",
                max_nfev=int(max_nfev),
            )
            if candidate.success and np.all(np.isfinite(candidate.x)):
                fit_candidates.append((float(np.dot(candidate.fun, candidate.fun)), candidate))
            else:
                fit_failures.append(str(candidate.message))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            fit_failures.append(str(exc))
    if not fit_candidates:
        raise ValueError("two-population Thomson fit failed for every deterministic start: " + "; ".join(fit_failures[:3]))
    fit_candidates.sort(key=lambda item: item[0])
    _, fit = fit_candidates[0]
    amplitude, background_coefficients, fitted_background, fitted = profile(fit.x)
    tc, th, fraction, flow = decode(fit.x)
    jac = np.asarray(fit.jac, dtype=float)
    column_norms = np.linalg.norm(jac, axis=0)
    scaled_jac = np.divide(
        jac, column_norms[None, :],
        out=np.zeros_like(jac),
        where=column_norms[None, :] > np.finfo(float).tiny,
    )
    singular = np.linalg.svd(scaled_jac, compute_uv=False)
    tolerance = np.finfo(float).eps * max(scaled_jac.shape) * (singular[0] if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0.0 else float("inf")
    if rank < fit.x.size or not np.isfinite(condition) or condition > 1e12:
        raise ValueError(
            "two-population Thomson fit rejected: population parameters are not jointly identifiable "
            f"(scaled Jacobian rank {rank}/{fit.x.size}, condition number {condition:.6g})"
        )
    dof = max(1, y.size-fit.x.size-(background_degree + 2))
    covariance_optimizer = np.linalg.pinv(jac.T@jac)
    if signal_uncertainty is None and spectral_covariance is None:
        covariance_optimizer *= float(residual(fit.x)@residual(fit.x)/dof)
    transform = np.diag([tc, th-tc, fraction*(1-fraction), scale_flow])
    transform[1, 0] = tc
    covariance_physical = transform @ covariance_optimizer @ transform.T
    standard = np.sqrt(np.maximum(np.diag(covariance_physical), 0.0))
    final_model = two_population_relativistic_thomson_spectrum(
        x, laser, angle, tc, th, fraction,
        flow_projection_m_s=flow, incident_polarization=incident_polarization,
        radial_order=radial_order, angular_order=angular_order,
        maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
        include_compton_recoil=include_compton_recoil,
    )
    # Conditional covariance of linear background coefficients at the fitted
    # nonlinear population parameters. This block is not cross-covariance with
    # the nonlinear parameters; the latter is represented only through the
    # profiled nonlinear Jacobian covariance below.
    final_shape = forward(fit.x)
    background_design = whitening(np.column_stack((final_shape, background_basis)))
    background_covariance_whitened = np.linalg.pinv(background_design.T @ background_design)
    if signal_uncertainty is None and spectral_covariance is None:
        background_covariance_whitened *= float(residual(fit.x) @ residual(fit.x) / dof)
    background_covariance = background_covariance_whitened[1:, 1:]
    return {
        "cold_electron_temperature_eV": tc,
        "hot_electron_temperature_eV": th,
        "hot_population_fraction": fraction,
        "electron_velocity_projection_m_s": flow,
        "cold_electron_temperature_uncertainty_eV": float(standard[0]),
        "hot_electron_temperature_uncertainty_eV": float(standard[1]),
        "hot_population_fraction_uncertainty": float(standard[2]),
        "electron_velocity_projection_uncertainty_m_s": float(standard[3]),
        "population_fraction_definition": "hot-electron number-density fraction",
        "fitted_spectrum": fitted,
        "residual": y-fitted,
        "fitted_amplitude": amplitude,
        "fitted_constant_background": float(background_coefficients[0]) if background_degree == 0 else None,
        "fitted_background_coefficients_scaled_wavelength": np.asarray(background_coefficients, dtype=float),
        "fitted_background_degree": background_degree,
        "fitted_background_model": background_model,
        "fitted_background_interior_knots": int(background_interior_knots) if background_model == "cubic_spline" else None,
        "background_wavelength_center_nm": x_center,
        "background_wavelength_scale_nm": x_scale,
        "background_coefficient_covariance_conditional": background_covariance,
        "background_covariance_scope": "conditional linear-background covariance at fitted nonlinear population parameters; nonlinear-background cross-covariance is not reported",
        "compton_recoil_included": bool(include_compton_recoil),
        "cross_section_model": "polarized Klein-Nishina with single-scattering recoil" if include_compton_recoil else "polarized Thomson; recoil screened only",
        "fit_covariance": covariance_physical,
        "fit_jacobian_rank": rank,
        "fit_jacobian_condition_number": condition,
        "fit_degrees_of_freedom": int(dof),
        "optimizer_success": bool(fit.success),
        "optimizer_initialization": initialization_status,
        "optimizer_multistart_count": int(len(initial_vectors)),
        "optimizer_seed_parameters_are_not_inferred_outputs": True,
        "spectral_uncertainty_status": uncertainty_status,
        "effective_differential_cross_section_m2_sr": final_model["effective_differential_cross_section_m2_sr"],
        "absolute_density_inferred": False,
        "model": final_model["model"],
        "model_reference": "Sakai et al., JINST 21 (2026) C01002, DOI 10.1088/1748-0221/21/01/C01002; two-component Maxwell-Juttner mixture is a parametric approximation, not the paper's Monte Carlo algorithm.",
        "limitations": [
            "Two isotropic Maxwell-Juttner populations are assumed; a general arbitrary or anisotropic 3-D electron distribution is not reconstructed.",
            "Both populations share one bulk-flow projection along the scattering vector.",
            "The fitted population fraction and temperatures are conditional on the selected two-population model and measured response.",
            "Absolute electron density is not inferred by this normalized shape-only subanalysis.",
            "Structured background is a detector-space nuisance model fitted jointly with the Thomson signal; inspect fitted_background_model and coefficients, and do not interpret an analytic basis as a measured stray-light template.",
            "Background coefficient covariance is conditional and does not include cross-covariance with nonlinear population parameters.",
            "Compton recoil is included only when include_compton_recoil=True; otherwise the Thomson approximation is used subject to the recoil screen.",
            "No experimental validation is performed or claimed.",
        ],
    }



def multi_population_relativistic_thomson_spectrum(
    wavelength_nm, laser_wavelength_nm, scattering_angle_rad,
    population_temperatures_eV, population_fractions, *,
    flow_projection_m_s, incident_polarization, radial_order=36,
    angular_order=18, maximum_recoil_correction_fraction=0.01,
    include_compton_recoil=False,
):
    """Forward spectrum for an arbitrary finite mixture of isotropic MJ populations.

    The component emissivities are added as number-density-weighted differential
    cross sections, not by averaging temperatures or spectral widths:
      S_mix(lambda) = sum_j eta_j sigma_j S_j(lambda) / sum_j eta_j sigma_j.
    All components share the supplied q-parallel bulk flow and polarization.
    """
    wavelength = _vector(wavelength_nm, "wavelength_nm")
    if np.any(wavelength <= 0.0) or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_nm must be positive and strictly increasing")
    temperatures = np.asarray(population_temperatures_eV, dtype=float)
    fractions = np.asarray(population_fractions, dtype=float)
    if (temperatures.ndim != 1 or fractions.ndim != 1
            or temperatures.size != fractions.size or temperatures.size < 2
            or np.any(~np.isfinite(temperatures)) or np.any(~np.isfinite(fractions))):
        raise ValueError("population temperatures and fractions must be finite matching 1-D vectors of length >=2")
    if np.any(temperatures <= 0.0) or np.any(fractions < 0.0):
        raise ValueError("population temperatures must be positive and fractions non-negative")
    fraction_sum = float(np.sum(fractions))
    if not np.isclose(fraction_sum, 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("population_fractions must sum to one")
    flow = float(flow_projection_m_s)
    if not np.isfinite(flow) or abs(flow) >= C:
        raise ValueError("flow_projection_m_s must be finite and subluminal")
    common = dict(
        wavelength_nm=wavelength,
        laser_wavelength_nm=_positive(laser_wavelength_nm, "laser_wavelength_nm"),
        scattering_angle_rad=float(scattering_angle_rad),
        flow_projection_m_s=flow,
        incident_polarization=incident_polarization,
        radial_order=radial_order,
        angular_order=angular_order,
        maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
        include_compton_recoil=include_compton_recoil,
    )
    components = [
        relativistic_maxwell_juttner_spectrum(
            electron_temperature_eV=float(temperature), **common
        )
        for temperature in temperatures
    ]
    cross_sections = np.asarray([
        item["effective_differential_cross_section_m2_sr"] for item in components
    ], dtype=float)
    weighted_cross_sections = fractions * cross_sections
    total_cross_section = float(np.sum(weighted_cross_sections))
    if not np.isfinite(total_cross_section) or total_cross_section <= 0.0:
        raise ValueError("mixture effective differential cross-section must be positive and finite")
    component_shapes = np.asarray([
        item["spectral_density_per_nm"] for item in components
    ], dtype=float)
    spectrum = np.sum(
        weighted_cross_sections[:, None] * component_shapes, axis=0
    ) / total_cross_section
    return {
        "spectral_density_per_nm": spectrum,
        "population_temperatures_eV": temperatures.copy(),
        "population_number_density_fractions": fractions.copy(),
        "population_effective_cross_sections_m2_sr": cross_sections,
        "mixture_effective_differential_cross_section_m2_sr": total_cross_section,
        "component_spectra_per_nm": component_shapes,
        "flow_projection_m_s": flow,
        "population_count": int(temperatures.size),
        "mixture_relation": "S_mix=sum_j(eta_j*sigma_j*S_j)/sum_j(eta_j*sigma_j); eta_j=n_j/sum_k(n_k)",
        "model": "arbitrary-finite-component isotropic relativistic Maxwell-Juttner optical Thomson mixture",
        "population_assumption": "finite mixture of isotropic Maxwell-Juttner components with a shared q-parallel bulk flow",
        "absolute_density_inferred": False,
        "limitations": [
            "This is a parametric finite mixture, not an unrestricted 3-D EVDF reconstruction.",
            "Component fractions and temperatures are supplied inputs to this forward model.",
            "All components share one bulk-flow projection and incident polarization.",
        ],
    }


def fit_multi_population_relativistic_thomson_spectrum(
    wavelength_nm, measured_spectrum, laser_wavelength_nm,
    scattering_angle_rad, instrument_response, *,
    population_count, incident_polarization,
    flow_projection_initial_m_s=0.0, signal_uncertainty=None,
    spectral_covariance=None, radial_order=20, angular_order=10,
    maximum_recoil_correction_fraction=0.01,
    include_compton_recoil=False, background_degree=0, max_nfev=600,
):
    """Infer ordered component temperatures, simplex fractions and shared flow.

    Linear signal amplitude/background coefficients are profiled at each
    nonlinear step. A scaled-Jacobian rank/condition gate prevents reporting
    an underdetermined population decomposition as measured parameters.
    """
    x = _vector(wavelength_nm, "wavelength_nm")
    y = _vector(measured_spectrum, "measured_spectrum")
    if x.shape != y.shape or np.any(x <= 0.0) or np.any(np.diff(x) <= 0.0) or np.any(y < 0.0):
        raise ValueError("measured wavelength and non-negative spectrum must match; wavelength must increase")
    if isinstance(population_count, (bool, np.bool_)) or not isinstance(population_count, (int, np.integer)) or not 2 <= int(population_count) <= 6:
        raise ValueError("population_count must be an integer from 2 through 6")
    count = int(population_count)
    response = np.asarray(instrument_response, dtype=float)
    if response.shape != (x.size, x.size) or np.any(~np.isfinite(response)) or np.any(response < 0.0):
        raise ValueError("instrument_response must be a finite non-negative square matrix")
    if not np.allclose(response.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("instrument_response columns must sum to one")
    if signal_uncertainty is not None and spectral_covariance is not None:
        raise ValueError("supply signal_uncertainty or spectral_covariance, not both")
    if signal_uncertainty is not None:
        sigma = _vector(signal_uncertainty, "signal_uncertainty")
        if sigma.shape != y.shape or np.any(sigma <= 0.0):
            raise ValueError("signal_uncertainty must match the spectrum and be positive")
        whiten = lambda value: np.asarray(value) / sigma
        noise_status = "supplied_pointwise_uncertainty"
    elif spectral_covariance is not None:
        covariance = np.asarray(spectral_covariance, dtype=float)
        if covariance.shape != (y.size, y.size) or np.any(~np.isfinite(covariance)) or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("spectral_covariance must be finite, symmetric and match the measured spectrum")
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("spectral_covariance must be positive definite") from exc
        whiten = lambda value: np.linalg.solve(chol, value)
        noise_status = "supplied_full_spectral_covariance"
    else:
        whiten = lambda value: np.asarray(value)
        noise_status = "unweighted_residual_scaled_only"
    if isinstance(background_degree, (bool, np.bool_)) or not isinstance(background_degree, (int, np.integer)) or not 0 <= int(background_degree) <= 5:
        raise ValueError("background_degree must be an integer from 0 through 5")
    from .thomson_background import _background_design
    background = _background_design(
        x, "polynomial" if int(background_degree) else "constant",
        degree=int(background_degree),
    )
    flow0 = float(flow_projection_initial_m_s)
    if not np.isfinite(flow0) or abs(flow0) >= C:
        raise ValueError("flow_projection_initial_m_s must be finite and subluminal")
    # Ordered temperatures remove permutation-equivalent labels. Softmax
    # fractions enforce positivity and exact number-density normalization.
    initial_temperatures = np.geomspace(2.0, 200.0, count)
    p0 = np.r_[np.log(initial_temperatures[0]),
               np.log(np.diff(initial_temperatures)),
               np.zeros(count - 1), flow0 / 1.0e6]
    lower = np.r_[np.full(count, np.log(1e-4)), np.full(count - 1, -20.0), -C/1e6*(1-1e-12)]
    upper = np.r_[np.full(count, np.log(5e6)), np.full(count - 1, 20.0), C/1e6*(1-1e-12)]
    def decode(parameters):
        logs = parameters[:count]
        temperatures = np.empty(count)
        temperatures[0] = np.exp(logs[0])
        for j in range(1, count):
            temperatures[j] = temperatures[j-1] + np.exp(logs[j])
        logits = np.r_[parameters[count:2*count-1], 0.0]
        logits -= np.max(logits)
        fractions = np.exp(logits)
        fractions /= np.sum(fractions)
        return temperatures, fractions, float(parameters[-1] * 1.0e6)
    def forward(parameters):
        temperatures, fractions, flow = decode(parameters)
        shape = multi_population_relativistic_thomson_spectrum(
            x, laser_wavelength_nm, scattering_angle_rad, temperatures,
            fractions, flow_projection_m_s=flow,
            incident_polarization=incident_polarization,
            radial_order=radial_order, angular_order=angular_order,
            maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
            include_compton_recoil=include_compton_recoil,
        )["spectral_density_per_nm"]
        return _apply_response(response, shape, x)
    def profile(parameters):
        signal = forward(parameters)
        design = whiten(np.column_stack((signal, background)))
        coefficients = np.linalg.lstsq(design, whiten(y), rcond=None)[0]
        if coefficients[0] < 0.0:
            coefficients[0] = 0.0
            coefficients[1:] = np.linalg.lstsq(whiten(background), whiten(y), rcond=None)[0]
        fitted = coefficients[0]*signal + background@coefficients[1:]
        return coefficients, fitted
    def residual(parameters):
        return whiten(profile(parameters)[1] - y)
    fit = least_squares(residual, p0, bounds=(lower, upper), x_scale="jac", max_nfev=int(max_nfev))
    if not fit.success or np.any(~np.isfinite(fit.x)):
        raise ValueError(f"multi-population Thomson optimizer failed: {fit.message}")
    jac = np.asarray(fit.jac, dtype=float)
    norms = np.linalg.norm(jac, axis=0)
    scaled = np.divide(jac, norms[None, :], out=np.zeros_like(jac), where=norms[None, :] > np.finfo(float).tiny)
    singular = np.linalg.svd(scaled, compute_uv=False)
    tol = np.finfo(float).eps * max(scaled.shape) * (singular[0] if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > tol))
    condition = float(singular[0]/singular[-1]) if singular.size and singular[-1] > 0.0 else float("inf")
    if rank < fit.x.size or not np.isfinite(condition) or condition > 1e12:
        raise ValueError(f"multi-population Thomson fit rejected as non-identifiable (Jacobian rank {rank}/{fit.x.size}, condition={condition:.6g}); reduce population_count or improve spectral coverage")
    dof = max(1, y.size-fit.x.size-background.shape[1]-1)
    covariance_p = np.linalg.pinv(jac.T@jac)
    if signal_uncertainty is None and spectral_covariance is None:
        covariance_p *= float(residual(fit.x)@residual(fit.x)/dof)
    temperatures, fractions, flow = decode(fit.x)
    standard_p = np.sqrt(np.maximum(np.diag(covariance_p), 0.0))
    # Delta-method Jacobian for ordered temperatures, softmax fractions and flow.
    transform = np.zeros((2*count + 1, fit.x.size))
    transform[0, 0] = temperatures[0]
    for j in range(1, count):
        transform[j, 0] = temperatures[0]
        for k in range(1, j + 1):
            transform[j, k] = np.exp(fit.x[k])
    for j in range(count):
        for k in range(count-1):
            transform[count+j, count+k] = fractions[j] * ((1.0 if j == k else 0.0) - fractions[k])
    transform[-1, -1] = 1.0e6
    physical_covariance = transform @ covariance_p @ transform.T
    coefficients, fitted = profile(fit.x)
    return {
        "population_temperatures_eV": temperatures,
        "population_number_density_fractions": fractions,
        "shared_flow_projection_m_s": flow,
        "population_parameter_covariance": physical_covariance,
        "population_temperature_uncertainties_eV": np.sqrt(np.maximum(np.diag(physical_covariance)[:count], 0.0)),
        "population_fraction_uncertainties": np.sqrt(np.maximum(np.diag(physical_covariance)[count:2*count], 0.0)),
        "fitted_spectrum": fitted,
        "residual": y-fitted,
        "fitted_signal_amplitude": float(coefficients[0]),
        "fitted_background_coefficients": coefficients[1:],
        "population_count": count,
        "jacobian_rank": rank,
        "jacobian_condition_number": condition,
        "degrees_of_freedom": int(dof),
        "spectral_noise_model": noise_status,
        "optimizer_success": bool(fit.success),
        "model": "arbitrary-finite-component isotropic relativistic Maxwell-Juttner optical Thomson mixture",
        "population_fraction_definition": "component electron number-density fractions, constrained to the probability simplex",
        "identifiability_status": "identified_by_scaled_spectral_jacobian",
        "limitations": [
            "The inversion is conditional on the finite isotropic Maxwell-Juttner mixture family and shared q-parallel flow.",
            "A full-rank Jacobian is necessary but does not establish model adequacy or experimental validation.",
            "Absolute electron density and unrestricted 3-D EVDF are not inferred.",
            "Component labels are ordered by increasing temperature to remove permutation degeneracy.",
        ],
    }
