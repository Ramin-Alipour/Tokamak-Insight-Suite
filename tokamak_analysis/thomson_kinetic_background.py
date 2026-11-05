"""Native structured-background fitting for non-collective kinetic Thomson models.

The physical forward model, calibrated response, and detector-space nuisance
basis are fitted simultaneously. This helper is deliberately shape-only:
absolute density is not inferred from normalized or freely-amplitude-scaled
spectra. It complements, rather than silently changes, the legacy channel-wise
fitters until their UI contracts are migrated.
"""
from __future__ import annotations

import numpy as np
from scipy.special import stdtr

_C = 299_792_458.0
_E_REST_EV = 510_998.95069


def _real_vector(value, name, minimum=7):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size < minimum:
        raise ValueError(f"{name} must be a real one-dimensional vector with at least {minimum} entries")
    out = raw.astype(float, copy=False)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain finite values")
    return out


def _wavelength_edges(axis):
    """Midpoint-defined finite-volume wavelength edges for measured bins."""
    x = np.asarray(axis, dtype=float)
    if x.ndim != 1 or x.size < 7 or np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("wavelength axis must be finite, one-dimensional and strictly increasing")
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    if np.any(np.diff(edges) <= 0):
        raise ValueError("wavelength bins must have positive widths")
    return edges


def _kappa_bin_average_spectral_density(wavelength_nm, laser_wavelength_nm,
                                        scattering_angle_rad, temperature_eV,
                                        flow_projection_m_s, kappa):
    """Exact finite-bin Student-t probability for the isotropic Kappa q-marginal.

    The projected marginal has nu=2*kappa-1 degrees of freedom and scale
    sigma_lambda*sqrt((nu-2)/nu), where sigma_lambda is fixed by the
    second-moment temperature. Integrating the CDF over each wavelength bin
    avoids point-sampling bias and preserves probability before detector
    response convolution.
    """
    x = np.asarray(wavelength_nm, dtype=float)
    laser = float(laser_wavelength_nm)
    angle = float(scattering_angle_rad)
    temperature = float(temperature_eV)
    flow = float(flow_projection_m_s)
    index = float(kappa)
    if (x.ndim != 1 or x.size < 7 or np.any(~np.isfinite(x))
            or np.any(np.diff(x) <= 0) or laser <= 0 or temperature <= 0
            or not 0 < angle < np.pi or abs(flow) >= _C or index <= 1.5):
        raise ValueError("invalid wavelength or isotropic finite-variance Kappa parameters")
    edges = _wavelength_edges(x)
    q = 2.0 * np.sin(angle / 2.0)
    center = laser * (1.0 - q * flow / _C)
    sigma_nm = laser * q * np.sqrt(temperature / _E_REST_EV)
    degrees = 2.0 * index - 1.0
    scale_nm = sigma_nm * np.sqrt((degrees - 2.0) / degrees)
    probability = (
        stdtr(degrees, (edges[1:] - center) / scale_nm)
        - stdtr(degrees, (edges[:-1] - center) / scale_nm)
    )
    if np.any(probability < -1e-12) or np.any(~np.isfinite(probability)):
        raise ValueError("Kappa finite-bin integration produced invalid probability")
    return np.maximum(probability, 0.0) / np.diff(edges)


def _apply_response(response, spectrum, axis):
    x = np.asarray(axis, dtype=float)
    y = np.asarray(spectrum, dtype=float)
    r = np.asarray(response, dtype=float)
    if r.shape != (x.size, x.size) or y.shape != x.shape or np.any(~np.isfinite(r)) or np.any(r < 0):
        raise ValueError("instrument_response must be a finite non-negative square matrix on the measured wavelength grid")
    if not np.allclose(r.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("measured instrument-response columns must already sum to one")
    edges = np.empty(x.size + 1)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0):
        raise ValueError("wavelength grid must define positive bin widths")
    return (r @ (y * widths)) / widths


def fit_noncollective_kinetic_thomson_with_background(
    wavelength_nm, measured_spectrum, *, physical_model,
    laser_wavelength_nm, scattering_angle_rad, instrument_response,
    physical_initial, background_model="linear", background_degree=2,
    background_interior_knots=3, background_templates=None,
    measurement_uncertainty=None, measurement_covariance=None,
    model_discrepancy_covariance=None, systematic_covariance=None,
    incident_polarization=None, radial_order=36, angular_order=18,
    maximum_recoil_correction_fraction=0.01, include_compton_recoil=False,
    laser_spectrum_wavelength_nm=None, laser_spectrum_density_per_nm=None,
    laser_spectrum_quadrature_order=9,
    physical_bounds=None, max_nfev=5000,
):
    """Fit a Kappa or relativistic Maxwell-Juttner optical spectrum plus structured background.

    physical_initial order is [Te_eV, flow_projection_m_s, kappa] for Kappa,
    and [Te_eV, flow_projection_m_s] for Maxwell-Juttner. Amplitude is profiled
    linearly with the background coefficients. Kappa is the finite-variance
    isotropic Student-t projection; it is not the generalized Skolar plasma
    dispersion-function method. MJ uses the polarized single-particle forward
    model and its explicit Thomson/Compton recoil contract.
    """
    from scipy.optimize import least_squares
    from .thomson_background import _background_design
    from .thomson_kinetic_models import relativistic_maxwell_juttner_spectrum

    x = _real_vector(wavelength_nm, "wavelength_nm")
    y = _real_vector(measured_spectrum, "measured_spectrum")
    if y.shape != x.shape or np.any(np.diff(x) <= 0) or np.any(x <= 0) or np.any(y < 0):
        raise ValueError("measured spectrum must be non-negative and match a positive increasing wavelength grid")
    laser = float(laser_wavelength_nm)
    angle = float(scattering_angle_rad)
    if not np.isfinite(laser) or laser <= 0 or not np.isfinite(angle) or not 0 < angle < np.pi:
        raise ValueError("laser_wavelength_nm must be positive and scattering_angle_rad must lie in (0, pi)")
    if physical_model not in {"kappa", "relativistic_maxwell_juttner"}:
        raise ValueError("physical_model must be 'kappa' or 'relativistic_maxwell_juttner'")
    if not isinstance(include_compton_recoil, (bool, np.bool_)):
        raise ValueError("include_compton_recoil must be boolean")
    p0 = _real_vector(physical_initial, "physical_initial", minimum=2)
    expected = 3 if physical_model == "kappa" else 2
    if p0.size != expected:
        raise ValueError(f"physical_initial must contain {expected} values for {physical_model}")
    if p0[0] <= 0 or abs(p0[1]) >= _C:
        raise ValueError("initial Te must be positive and flow projection subluminal")
    if physical_model == "kappa" and p0[2] <= 1.5:
        raise ValueError("finite-variance Kappa index must exceed 1.5")
    if physical_model == "relativistic_maxwell_juttner" and incident_polarization is None:
        raise ValueError("incident_polarization is required for relativistic Maxwell-Juttner inference")
    if (laser_spectrum_wavelength_nm is None) != (laser_spectrum_density_per_nm is None):
        raise ValueError("supply both laser_spectrum_wavelength_nm and laser_spectrum_density_per_nm, or neither")
    source_nodes = source_weights = None
    if laser_spectrum_wavelength_nm is not None:
        if physical_model != "relativistic_maxwell_juttner":
            raise ValueError("measured source-profile convolution is currently implemented for the relativistic Maxwell-Juttner forward model")
        source_x = _real_vector(laser_spectrum_wavelength_nm, "laser_spectrum_wavelength_nm", minimum=3)
        source_y = _real_vector(laser_spectrum_density_per_nm, "laser_spectrum_density_per_nm", minimum=3)
        if source_y.shape != source_x.shape or np.any(np.diff(source_x) <= 0) or np.any(source_x <= 0) or np.any(source_y < 0):
            raise ValueError("laser source profile must be paired, positive-wavelength, increasing and non-negative")
        source_area = float(np.trapz(source_y, source_x))
        if not np.isfinite(source_area) or source_area <= 0:
            raise ValueError("laser source profile must have positive finite integrated spectral power")
        order = int(laser_spectrum_quadrature_order)
        if isinstance(laser_spectrum_quadrature_order, (bool, np.bool_)) or order != laser_spectrum_quadrature_order or order < 3 or order > 31:
            raise ValueError("laser_spectrum_quadrature_order must be an integer from 3 through 31")
        # Integrate the measured piecewise-linear spectral-power density as
        # photon flux: dN/dlambda = P_lambda * lambda / (h*c). Construct its
        # exact piecewise-polynomial CDF, then apply a fixed-order Gauss rule in
        # probability space. Runtime therefore scales with the selected order,
        # not with the number of laser-profile samples.
        segment_width = np.diff(source_x)
        segment_slope = np.diff(source_y) / segment_width
        segment_photon_integral = (
            source_x[:-1]*source_y[:-1]*segment_width
            + 0.5*(source_y[:-1] + source_x[:-1]*segment_slope)*segment_width**2
            + (segment_slope*segment_width**3)/3.0
        )
        photon_cdf = np.r_[0.0, np.cumsum(segment_photon_integral)]
        photon_norm = float(photon_cdf[-1])
        if not np.isfinite(photon_norm) or photon_norm <= 0:
            raise ValueError("laser source profile has zero photon-flux integral")
        photon_cdf /= photon_norm
        unique_cdf, unique_indices = np.unique(photon_cdf, return_index=True)
        if unique_cdf.size < 2:
            raise ValueError("laser source profile has insufficient non-zero photon-flux support")
        probability_nodes, probability_weights = np.polynomial.legendre.leggauss(order)
        probability_nodes = 0.5*(probability_nodes+1.0)
        source_nodes = np.interp(
            probability_nodes, unique_cdf, source_x[unique_indices]
        )
        source_weights = 0.5*probability_weights
    response = np.asarray(instrument_response, dtype=float)
    # Validate response independently before optimization.
    _apply_response(response, np.ones_like(x), x)
    basis = _background_design(
        x, background_model, degree=background_degree,
        interior_knots=background_interior_knots, templates=background_templates,
    )

    if physical_bounds is None:
        lower = np.r_[1e-4, -0.95 * _C, 1.500001] if physical_model == "kappa" else np.array([1e-4, -0.95 * _C])
        upper = np.r_[2e6, 0.95 * _C, 100.0] if physical_model == "kappa" else np.array([2e6, 0.95 * _C])
    else:
        lower, upper = (np.asarray(v, dtype=float) for v in physical_bounds)
        if lower.shape != p0.shape or upper.shape != p0.shape or np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("physical_bounds must be two finite vectors matching physical_initial")
    if np.any(lower >= upper) or np.any(p0 < lower) or np.any(p0 > upper):
        raise ValueError("physical initial values must lie within strictly ordered physical_bounds")

    if measurement_uncertainty is not None and measurement_covariance is not None:
        raise ValueError("supply measurement_uncertainty or measurement_covariance, not both")
    if measurement_uncertainty is not None:
        sigma = _real_vector(measurement_uncertainty, "measurement_uncertainty")
        if sigma.shape != y.shape or np.any(sigma <= 0):
            raise ValueError("measurement_uncertainty must be positive and match measured_spectrum")
        total_covariance = np.diag(sigma**2)
        measurement_supplied = True
    elif measurement_covariance is not None:
        total_covariance = np.asarray(measurement_covariance, dtype=float)
        if (total_covariance.shape != (x.size, x.size)
                or np.any(~np.isfinite(total_covariance))
                or not np.allclose(total_covariance, total_covariance.T, rtol=1e-10, atol=1e-13)):
            raise ValueError("measurement_covariance must be finite, symmetric and wavelength-sized")
        total_covariance = 0.5 * (total_covariance + total_covariance.T)
        measurement_supplied = True
    else:
        total_covariance = np.zeros((x.size, x.size), dtype=float)
        measurement_supplied = False

    discrepancy_supplied = model_discrepancy_covariance is not None
    systematic_supplied = systematic_covariance is not None
    for name, component in (
        ("model_discrepancy_covariance", model_discrepancy_covariance),
        ("systematic_covariance", systematic_covariance),
    ):
        if component is None:
            continue
        component = np.asarray(component, dtype=float)
        if (component.shape != (x.size, x.size) or np.any(~np.isfinite(component))
                or not np.allclose(component, component.T, rtol=1e-10, atol=1e-13)):
            raise ValueError(f"{name} must be finite, symmetric and wavelength-sized")
        component = 0.5 * (component + component.T)
        eigenvalues = np.linalg.eigvalsh(component)
        scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        if eigenvalues.min() < -1e-10 * scale:
            raise ValueError(f"{name} must be positive semidefinite")
        total_covariance += component

    covariance_supplied = measurement_supplied or discrepancy_supplied or systematic_supplied
    if covariance_supplied:
        try:
            chol = np.linalg.cholesky(total_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "the sum of supplied measurement, model-discrepancy and systematic "
                "covariances must be positive definite"
            ) from exc
        whiten = lambda a: np.linalg.solve(chol, np.asarray(a))
        if discrepancy_supplied and systematic_supplied:
            noise_model = "measurement_plus_model_discrepancy_plus_systematic_covariance" if measurement_supplied else "model_discrepancy_plus_systematic_covariance_only"
        elif discrepancy_supplied:
            noise_model = "measurement_plus_model_discrepancy_covariance" if measurement_supplied else "model_discrepancy_covariance_only"
        elif systematic_supplied:
            noise_model = "measurement_plus_systematic_covariance" if measurement_supplied else "systematic_covariance_only"
        else:
            noise_model = "supplied_measurement_covariance"
    else:
        chol = None
        whiten = lambda a: np.asarray(a)
        noise_model = "unweighted_no_measurement_uncertainty_supplied"

    q = 2.0 * np.sin(angle / 2.0)

    def intrinsic(parameters):
        te, flow = float(parameters[0]), float(parameters[1])
        if physical_model == "kappa":
            return _kappa_bin_average_spectral_density(
                x, laser, angle, te, flow, float(parameters[2])
            )
        # Evaluate the relativistic kernel inside each measured finite bin and
        # Gauss-Legendre average before the response acts on bin-integrated power.
        edges = _wavelength_edges(x)
        nodes, weights = np.polynomial.legendre.leggauss(8)
        half_width = 0.5 * np.diff(edges)
        midpoint = 0.5 * (edges[1:] + edges[:-1])
        quadrature_wavelength = (midpoint[:, None] + half_width[:, None] * nodes[None, :]).ravel()
        source_lasers = [(laser, 1.0)] if source_nodes is None else zip(source_nodes, source_weights)
        combined = np.zeros(quadrature_wavelength.size, dtype=float)
        total_scattered_weight = 0.0
        for incident_laser_nm, photon_weight in source_lasers:
            result = relativistic_maxwell_juttner_spectrum(
                quadrature_wavelength, float(incident_laser_nm), angle, te, flow_projection_m_s=flow,
                incident_polarization=np.asarray(incident_polarization, dtype=float),
                radial_order=radial_order, angular_order=angular_order,
                maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
                include_compton_recoil=include_compton_recoil,
                require_full_spectrum=False,
            )
            # Restore finite-window capture and wavelength-dependent differential
            # cross-section before mixing incident photons. This prevents the
            # per-wavelength normalized kernels from receiving equal artificial
            # weights when their physical scattering probabilities differ.
            cross_section = float(result["effective_differential_cross_section_m2_sr"])
            captured = float(result["captured_spectral_probability"])
            component_weight = float(photon_weight)*cross_section*captured
            # The forward kernel is a scattered-photon probability density.
            # Convert it to detected spectral power with E_s = h*c/lambda_s
            # before integrating over the measured finite wavelength bins.
            photon_density = np.asarray(result["spectral_density_per_nm"], dtype=float)
            power_density = photon_density / quadrature_wavelength
            combined += component_weight*power_density
            bin_energy_weight = float(np.sum(
                (power_density.reshape(x.size, nodes.size) * weights[None, :]
                 * half_width[:, None])
            ))
            total_scattered_weight += component_weight*bin_energy_weight
        if not np.isfinite(total_scattered_weight) or total_scattered_weight <= 0:
            raise ValueError("source-convolved relativistic Thomson spectrum has zero captured scattering power")
        density = (combined/total_scattered_weight).reshape(x.size, nodes.size)
        return 0.5 * (density @ weights)

    def profiled(parameters):
        shape = _apply_response(response, intrinsic(parameters), x)
        design = np.column_stack((shape, basis))
        a = whiten(design)
        b = whiten(y)
        coefficients = np.linalg.lstsq(a, b, rcond=None)[0]
        if coefficients[0] < 0:
            coefficients[0] = 0.0
            coefficients[1:] = np.linalg.lstsq(whiten(basis), b, rcond=None)[0]
        fitted = design @ coefficients
        return coefficients, fitted, shape

    def residual(parameters):
        _, fitted, _ = profiled(parameters)
        return whiten(fitted - y)

    fit = least_squares(residual, p0, bounds=(lower, upper), method="trf", x_scale="jac", max_nfev=int(max_nfev))
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"kinetic Thomson/background fit failed: {fit.message}")
    profiled_jacobian = np.asarray(fit.jac, dtype=float)
    profiled_singular = np.linalg.svd(profiled_jacobian, compute_uv=False)
    profiled_tolerance = np.finfo(float).eps * max(profiled_jacobian.shape) * (
        profiled_singular[0] if profiled_singular.size else 0.0
    )
    profiled_rank = int(np.count_nonzero(profiled_singular > profiled_tolerance))
    profiled_condition = (
        float(profiled_singular[0] / profiled_singular[-1])
        if profiled_singular.size and profiled_singular[-1] > 0 else float("inf")
    )
    if profiled_rank < p0.size or not np.isfinite(profiled_condition) or profiled_condition > 1e12:
        raise ValueError(
            f"{physical_model} physical parameters are not jointly identifiable "
            f"after profiling background (rank {profiled_rank}/{p0.size}, "
            f"condition={profiled_condition:.6g})"
        )
    coefficients, fitted, shape = profiled(fit.x)
    amplitude = float(coefficients[0])
    background_coefficients = np.asarray(coefficients[1:], dtype=float)
    background = basis @ background_coefficients
    fitted_signal = amplitude * shape

    # Build the full local Jacobian for [physical parameters, amplitude,
    # background coefficients]. This restores the nuisance cross-covariance
    # that a profile-only covariance cannot provide.
    n_physical = p0.size
    n_nuisance = coefficients.size
    joint_jacobian = np.empty((x.size, n_physical + n_nuisance), dtype=float)
    step_scale = np.cbrt(np.finfo(float).eps)
    for index in range(n_physical):
        step = step_scale * max(1.0, abs(float(fit.x[index])))
        plus, minus = fit.x.copy(), fit.x.copy()
        plus[index] += step
        minus[index] -= step
        plus_shape = _apply_response(response, intrinsic(plus), x)
        minus_shape = _apply_response(response, intrinsic(minus), x)
        derivative = amplitude * (plus_shape - minus_shape) / (2.0 * step)
        joint_jacobian[:, index] = whiten(derivative)
    design_at_solution = np.column_stack((shape, basis))
    joint_jacobian[:, n_physical:] = whiten(design_at_solution)
    joint_singular = np.linalg.svd(joint_jacobian, compute_uv=False)
    joint_tolerance = np.finfo(float).eps * max(joint_jacobian.shape) * (
        joint_singular[0] if joint_singular.size else 0.0
    )
    rank = int(np.count_nonzero(joint_singular > joint_tolerance))
    condition = (
        float(joint_singular[0] / joint_singular[-1])
        if joint_singular.size and joint_singular[-1] > 0 else float("inf")
    )
    parameter_count = n_physical + n_nuisance
    if rank < parameter_count or not np.isfinite(condition) or condition > 1e12:
        raise ValueError(
            f"{physical_model} plus selected structured background is not jointly "
            f"identifiable (rank {rank}/{parameter_count}, condition={condition:.6g}); "
            "change the measured background basis or acquire additional spectral information"
        )
    joint_covariance = np.linalg.pinv(joint_jacobian.T @ joint_jacobian, rcond=1e-12)
    dof = max(1, x.size - parameter_count)
    if not covariance_supplied:
        joint_covariance *= float(fit.fun @ fit.fun / dof)
        covariance_status = (
            "residual_scaled_unweighted_joint_covariance_not_measurement_uncertainty"
        )
    else:
        covariance_status = (
            "local_joint_GLS_covariance_conditional_on_supplied_measurement_"
            "model_discrepancy_and_systematic_covariance_components"
        )
    physical_covariance = joint_covariance[:n_physical, :n_physical]
    nuisance_covariance = joint_covariance[n_physical:, n_physical:]
    physical_background_cross_covariance = joint_covariance[
        :n_physical, n_physical + 1:
    ]
    return {
        "model": f"non_collective_{physical_model}_with_structured_detector_background",
        "physical_model": physical_model,
        "physical_parameter_names": ["electron_temperature_eV", "electron_velocity_projection_m_s"] + (["kappa_index"] if physical_model == "kappa" else []),
        "physical_parameters": fit.x.copy(),
        "electron_temperature_eV": float(fit.x[0]),
        "electron_velocity_projection_m_s": float(fit.x[1]),
        "kappa_index": float(fit.x[2]) if physical_model == "kappa" else None,
        "amplitude": amplitude,
        "background_model": background_model,
        "background_coefficients": background_coefficients,
        "fitted_thomson_spectrum": fitted_signal,
        "fitted_background": background,
        "fitted_spectrum": fitted,
        "residual": y - fitted,
        "physical_parameter_covariance": physical_covariance,
        "physical_parameter_standard_uncertainty": np.sqrt(np.maximum(np.diag(physical_covariance), 0.0)),
        "joint_parameter_names": (
            ["electron_temperature_eV", "electron_velocity_projection_m_s"]
            + (["kappa_index"] if physical_model == "kappa" else [])
            + ["spectral_amplitude"]
            + [f"background_coefficient_{i}" for i in range(background_coefficients.size)]
        ),
        "joint_parameter_covariance": joint_covariance,
        "background_parameter_covariance": nuisance_covariance[1:, 1:],
        "physical_background_cross_covariance": physical_background_cross_covariance,
        "spectral_amplitude_standard_uncertainty": float(np.sqrt(max(nuisance_covariance[0, 0], 0.0))),
        "jacobian_rank": rank,
        "jacobian_parameter_count": parameter_count,
        "jacobian_condition_number": condition,
        "profiled_physical_jacobian_rank": profiled_rank,
        "profiled_physical_jacobian_condition_number": profiled_condition,
        "parameter_identifiability": {
            "status": "locally_identifiable_after_joint_structured_background_fit",
            "physical_parameter_count": n_physical,
            "joint_parameter_count": parameter_count,
            "joint_jacobian_rank": rank,
            "joint_jacobian_condition_number": condition,
            "background_basis": background_model,
            "regularization_used": False,
        },
        "model_discrepancy_covariance_included": discrepancy_supplied,
        "systematic_covariance_included": systematic_supplied,
        "covariance_components": {
            "measurement_covariance_supplied": measurement_supplied,
            "model_discrepancy_covariance_supplied": discrepancy_supplied,
            "systematic_covariance_supplied": systematic_supplied,
            "combined_covariance_used_for_whitening": covariance_supplied,
        },
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "noise_model": noise_model,
        "covariance_status": covariance_status,
        "uncertainty_semantics": "Joint local covariance for physical parameters, spectral amplitude and detector-background coefficients; physical/background cross-covariance is returned. Conditional on the selected background basis, fixed instrument response and explicitly supplied covariance components. This is a local covariance, not a posterior distribution.",
        "absolute_density_inferred": False,
        "recoil_included": bool(include_compton_recoil) if physical_model == "relativistic_maxwell_juttner" else False,
        "cross_section_model": ("polarized_single_scattering_Klein_Nishina" if include_compton_recoil else "polarized_elastic_Thomson") if physical_model == "relativistic_maxwell_juttner" else "not_applicable_nonrelativistic_kappa_projection",
        "laser_source_profile_used": source_nodes is not None,
        "laser_source_profile_convolution": (
            "piecewise_linear_spectral_power_to_photon_flux_times_scattered_photon_energy_with_wavelength_resolved_recoil_and_Klein_Nishina_cross_section"
            if source_nodes is not None else "monochromatic_incident_laser"
        ),
        "laser_source_quadrature_nodes": 0 if source_nodes is None else int(source_nodes.size),
        "experimental_validation": "not_performed",
    }


def fit_noncollective_kinetic_thomson_with_joint_background(*args, **kwargs):
    """Joint physical, amplitude and structured-background fit with full covariance.

    This entry point uses the same physical forward models and input contract
    as fit_noncollective_kinetic_thomson_with_background, but optimizes all
    nuisance coefficients explicitly so the returned covariance contains the
    physical/background cross blocks.
    """
    from scipy.optimize import least_squares
    from .thomson_background import _background_design
    from .thomson_kinetic_models import relativistic_maxwell_juttner_spectrum

    (wavelength_nm, measured_spectrum) = args[:2] if len(args) >= 2 else (kwargs.pop("wavelength_nm"), kwargs.pop("measured_spectrum"))
    kw = dict(kwargs)
    model = kw.pop("physical_model")
    laser = float(kw.pop("laser_wavelength_nm"))
    angle = float(kw.pop("scattering_angle_rad"))
    response = np.asarray(kw.pop("instrument_response"), dtype=float)
    p0 = np.asarray(kw.pop("physical_initial"), dtype=float)
    bg_model = kw.pop("background_model", "linear")
    bg_degree = kw.pop("background_degree", 2)
    bg_knots = kw.pop("background_interior_knots", 3)
    bg_templates = kw.pop("background_templates", None)
    uncertainty = kw.pop("measurement_uncertainty", None)
    covariance = kw.pop("measurement_covariance", None)
    model_discrepancy_covariance = kw.pop("model_discrepancy_covariance", None)
    polarization = kw.pop("incident_polarization", None)
    radial_order = kw.pop("radial_order", 36)
    angular_order = kw.pop("angular_order", 18)
    recoil_limit = kw.pop("maximum_recoil_correction_fraction", 0.01)
    recoil = kw.pop("include_compton_recoil", False)
    bounds = kw.pop("physical_bounds", None)
    max_nfev = kw.pop("max_nfev", 5000)
    if kw:
        raise TypeError("unexpected keyword arguments: " + ", ".join(sorted(kw)))
    x = _real_vector(wavelength_nm, "wavelength_nm")
    y = _real_vector(measured_spectrum, "measured_spectrum")
    if y.shape != x.shape or np.any(np.diff(x) <= 0) or np.any(x <= 0) or np.any(y < 0):
        raise ValueError("measured spectrum must be non-negative and match a positive increasing wavelength grid")
    if model not in {"kappa", "relativistic_maxwell_juttner"}:
        raise ValueError("physical_model must be kappa or relativistic_maxwell_juttner")
    nphys = 3 if model == "kappa" else 2
    if p0.shape != (nphys,) or np.any(~np.isfinite(p0)) or p0[0] <= 0 or abs(p0[1]) >= _C:
        raise ValueError("physical_initial must contain finite positive Te and subluminal flow, plus Kappa index when selected")
    if model == "kappa" and p0[2] <= 1.5:
        raise ValueError("Kappa index must exceed 1.5")
    if model == "relativistic_maxwell_juttner" and polarization is None:
        raise ValueError("incident_polarization is required for relativistic Maxwell-Juttner")
    if not isinstance(recoil, (bool, np.bool_)):
        raise ValueError("include_compton_recoil must be boolean")
    if model == "kappa" and recoil:
        raise ValueError("Compton recoil selection is available only for the relativistic Maxwell-Juttner model")
    if uncertainty is not None and covariance is not None:
        raise ValueError("supply measurement_uncertainty or measurement_covariance, not both")
    if uncertainty is not None:
        sigma = _real_vector(uncertainty, "measurement_uncertainty")
        if sigma.shape != y.shape or np.any(sigma <= 0):
            raise ValueError("measurement_uncertainty must be positive and match the spectrum")
        total_covariance = np.diag(sigma**2)
        noise = "supplied_pointwise_measurement_uncertainty"
    elif covariance is not None:
        total_covariance = np.asarray(covariance, dtype=float)
        noise = "supplied_full_spectral_covariance"
    else:
        total_covariance = None
        noise = "unweighted_no_measurement_uncertainty_supplied"

    if model_discrepancy_covariance is not None:
        discrepancy = np.asarray(model_discrepancy_covariance, dtype=float)
        if (discrepancy.shape != (x.size, x.size)
                or np.any(~np.isfinite(discrepancy))
                or not np.allclose(discrepancy, discrepancy.T, rtol=1e-10, atol=1e-13)):
            raise ValueError("model_discrepancy_covariance must be finite, symmetric and wavelength-sized")
        try:
            np.linalg.cholesky(discrepancy)
        except np.linalg.LinAlgError as exc:
            raise ValueError("model_discrepancy_covariance must be positive definite") from exc
        total_covariance = discrepancy.copy() if total_covariance is None else total_covariance + discrepancy
        noise = "measurement_plus_model_discrepancy_covariance" if uncertainty is not None or covariance is not None else "model_discrepancy_covariance_only"

    if total_covariance is not None:
        if (total_covariance.shape != (x.size, x.size)
                or np.any(~np.isfinite(total_covariance))
                or not np.allclose(total_covariance, total_covariance.T, rtol=1e-10, atol=1e-13)):
            raise ValueError("combined spectral covariance must be finite, symmetric and wavelength-sized")
        try:
            chol = np.linalg.cholesky(total_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("combined spectral covariance must be positive definite") from exc
        whiten = lambda z: np.linalg.solve(chol, np.asarray(z))
    else:
        whiten = lambda z: np.asarray(z)
    basis = _background_design(x, bg_model, degree=bg_degree, interior_knots=bg_knots, templates=bg_templates)
    if response.shape != (x.size, x.size) or np.any(~np.isfinite(response)) or np.any(response < 0) or not np.allclose(response.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("instrument_response must be a measured non-negative column-normalized square matrix")
    if bounds is None:
        lo = np.r_[1e-4, -0.95*_C, 1.500001] if model == "kappa" else np.array([1e-4, -0.95*_C])
        hi = np.r_[2e6, 0.95*_C, 100.0] if model == "kappa" else np.array([2e6, 0.95*_C])
    else:
        lo, hi = (np.asarray(v, dtype=float) for v in bounds)
    if lo.shape != p0.shape or hi.shape != p0.shape or np.any(lo >= hi) or np.any(p0 < lo) or np.any(p0 > hi):
        raise ValueError("physical_bounds must match physical_initial and contain the initial values")
    q = 2*np.sin(angle/2)
    def shape(p):
        if model == "kappa":
            te, flow, kap = p
            center = laser*(1-q*flow/_C)
            sigma_nm = laser*2.0*np.sin(angle/2)*np.sqrt(te/_E_REST_EV)
            df = 2*kap-1
            scale = sigma_nm*np.sqrt((df-2)/df)
            intrinsic = (1+((x-center)/scale)**2/df)**(-kap)
        else:
            physical = relativistic_maxwell_juttner_spectrum(
                x, laser, angle, float(p[0]), flow_projection_m_s=float(p[1]),
                incident_polarization=np.asarray(polarization, dtype=float),
                radial_order=radial_order, angular_order=angular_order,
                maximum_recoil_correction_fraction=recoil_limit,
                include_compton_recoil=recoil, require_full_spectrum=False)
            intrinsic = np.asarray(physical["spectral_density_per_nm"], dtype=float)
        edges = np.empty(x.size+1)
        edges[1:-1] = 0.5*(x[:-1]+x[1:])
        edges[0] = x[0]-0.5*(x[1]-x[0])
        edges[-1] = x[-1]+0.5*(x[-1]-x[-2])
        widths = np.diff(edges)
        return (response@(intrinsic*widths))/widths
    initial_shape = shape(p0)
    linear = np.linalg.lstsq(whiten(np.column_stack((initial_shape,basis))),whiten(y),rcond=None)[0]
    linear[0] = max(float(linear[0]), np.finfo(float).eps)
    pinit = np.r_[p0, linear]
    lower = np.r_[lo, 0.0, np.full(basis.shape[1], -np.inf)]
    upper = np.r_[hi, np.inf, np.full(basis.shape[1], np.inf)]
    def predict(p):
        return p[nphys]*shape(p[:nphys])+basis@p[nphys+1:]
    def residual(p):
        return whiten(predict(p)-y)
    fit = least_squares(residual,pinit,bounds=(lower,upper),x_scale="jac",method="trf",max_nfev=int(max_nfev))
    if not fit.success or np.any(~np.isfinite(fit.x)):
        raise ValueError("joint kinetic Thomson/background optimizer failed: "+str(fit.message))
    jac = np.asarray(fit.jac,dtype=float)
    column_norms = np.linalg.norm(jac, axis=0)
    scaled_jac = np.divide(jac, column_norms[None, :],
                           out=np.zeros_like(jac),
                           where=column_norms[None, :] > np.finfo(float).tiny)
    singular = np.linalg.svd(scaled_jac,compute_uv=False)
    tol = np.finfo(float).eps*max(scaled_jac.shape)*(singular[0] if singular.size else 0.0)
    rank = int(np.count_nonzero(singular>tol))
    cond = float(singular[0]/singular[-1]) if singular.size and singular[-1]>0 else float("inf")
    if rank < fit.x.size or not np.isfinite(cond) or cond > 1e12:
        raise ValueError(f"joint physical/background fit is non-identifiable (rank {rank}/{fit.x.size}, condition={cond})")
    dof = max(1,y.size-fit.x.size)
    joint_cov = np.linalg.pinv(jac.T@jac,rcond=1e-12)
    if noise == "unweighted_no_measurement_uncertainty_supplied":
        joint_cov *= float(fit.fun@fit.fun/dof)
    background = basis@fit.x[nphys+1:]
    fitted = predict(fit.x)
    if model == "kappa":
        theta_e = float(fit.x[0] / _E_REST_EV)
        beta_flow = float(abs(fit.x[1]) / _C)
        if theta_e >= 0.01 or beta_flow >= 0.1:
            raise ValueError("non-relativistic Kappa model rejected by its declared theta_e < 0.01 and |v|/c < 0.1 screening policy")
    else:
        theta_e = float(fit.x[0] / _E_REST_EV)
        beta_flow = float(abs(fit.x[1]) / _C)
    return {
        "physical_model": model, "physical_parameters": fit.x[:nphys].copy(),
        "physical_parameter_names": ["electron_temperature_eV","electron_velocity_projection_m_s"]+(["kappa_index"] if model=="kappa" else []),
        "amplitude": float(fit.x[nphys]), "background_model": bg_model,
        "background_coefficients": fit.x[nphys+1:].copy(),
        "fitted_thomson_spectrum": fitted-background, "fitted_background": background,
        "fitted_spectrum": fitted, "residual": y-fitted,
        "joint_parameter_covariance": joint_cov,
        "physical_parameter_covariance": joint_cov[:nphys,:nphys],
        "physical_background_cross_covariance": joint_cov[:nphys,nphys+1:],
        "joint_parameter_order": ["electron_temperature_eV","electron_velocity_projection_m_s"]+(["kappa_index"] if model=="kappa" else [])+["spectral_amplitude"]+[f"background_coefficient_{i}" for i in range(basis.shape[1])],
        "jacobian_rank": rank, "jacobian_parameter_count": int(fit.x.size),
        "jacobian_condition_number": cond, "optimizer_success": bool(fit.success),
        "noise_model": noise,
        "model_discrepancy_covariance_included": model_discrepancy_covariance is not None,
        "model_discrepancy_covariance_semantics": "researcher_supplied_independent_additive_spectral_covariance; cross-covariance with measurement noise is assumed zero" if model_discrepancy_covariance is not None else "not_supplied",
        "covariance_status": "residual_scaled_unweighted_joint_covariance_not_measurement_uncertainty" if total_covariance is None else "local_joint_GLS_covariance_conditional_on_combined_measurement_and_model_discrepancy_covariance",
        "absolute_density_inferred": False, "theta_e": theta_e, "bulk_flow_beta": beta_flow,
        "recoil_included": bool(recoil) if model=="relativistic_maxwell_juttner" else False,
        "cross_section_model": ("polarized_single_scattering_Klein_Nishina" if recoil else "polarized_elastic_Thomson") if model=="relativistic_maxwell_juttner" else "isotropic_finite_variance_Kappa_projection",
        "experimental_validation": "not_performed"
    }


# Public canonical name resolves to the full joint-covariance implementation.
fit_noncollective_kinetic_thomson_with_background = fit_noncollective_kinetic_thomson_with_joint_background
