"""Additional inverse operators for relativistic projected EVDF and PSF-aware profiles.

These routines operate on already calibrated measured spectra/profiles. They do
not calibrate instruments, infer an unrestricted 3-D EVDF, or create information
outside the measured spectral/spatial bandwidth.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear


_C = 299792458.0


def _real_vector(value, name, minimum=2):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size < minimum:
        raise ValueError(f"{name} must be a real 1-D array with at least {minimum} samples")
    out = raw.astype(float, copy=False)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain finite values")
    return out


def _bin_widths(axis):
    edges = np.empty(axis.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
    edges[0] = axis[0] - 0.5 * (axis[1] - axis[0])
    edges[-1] = axis[-1] + 0.5 * (axis[-1] - axis[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0):
        raise ValueError("axis must define positive finite-volume bin widths")
    return widths

def _trapezoid_quadrature_weights(axis, name):
    """Return integration weights for a strictly increasing, uniform 1-D grid."""
    values = _real_vector(axis, name, minimum=2)
    steps = np.diff(values)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-12):
        raise ValueError(f"{name} must be strictly increasing and uniform")
    weights = np.full(values.size, float(steps[0]), dtype=float)
    weights[[0, -1]] *= 0.5
    return weights


def _whitener(n, uncertainty=None, covariance=None):
    if uncertainty is not None and covariance is not None:
        raise ValueError("supply pointwise uncertainty or covariance, not both")
    if uncertainty is not None:
        sigma = _real_vector(uncertainty, "uncertainty", minimum=n)
        if sigma.shape != (n,) or np.any(sigma <= 0):
            raise ValueError("uncertainty must be positive and match the data")
        return lambda z: np.asarray(z, dtype=float) / sigma, "supplied_pointwise_measurement_uncertainty"
    if covariance is not None:
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (n, n) or not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-13):
            raise ValueError("covariance must be finite, symmetric, and match the data")
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance must be positive definite") from exc
        return lambda z: np.linalg.solve(chol, np.asarray(z, dtype=float)), "supplied_full_measurement_covariance"
    return lambda z: np.asarray(z, dtype=float), "unweighted_no_measurement_uncertainty_supplied"



def relativistic_projected_compton_kinematics(
    velocity_grid_m_s, laser_wavelength_nm, scattering_angle_rad, *,
    incident_polarization=None,
):
    """Exact single-electron Compton kinematics for the symmetric 1-D q projection.

    The electron velocity is parallel to q=k_s-k_i. Photon energy is transformed
    to the electron rest frame, the Compton energy relation is applied there,
    and the scattered photon is transformed back to the laboratory frame.
    When an incident polarization is supplied, the polarized Klein-Nishina
    differential weight (including incident-flux and outgoing-solid-angle
    Jacobians) is returned. The weight is relative to r_e^2 and is suitable
    for a normalized projected-velocity mixture; it is not an absolute density.
    """
    velocity = _real_vector(velocity_grid_m_s, "velocity_grid_m_s", minimum=2)
    laser = float(laser_wavelength_nm)
    angle = float(scattering_angle_rad)
    if not np.isfinite(laser) or laser <= 0.0:
        raise ValueError("laser_wavelength_nm must be positive and finite")
    if not np.isfinite(angle) or not 0.0 < angle < np.pi:
        raise ValueError("scattering_angle_rad must lie strictly inside (0, pi)")
    if np.any(np.diff(velocity) <= 0.0):
        raise ValueError("velocity_grid_m_s must be strictly increasing")
    if np.any(np.abs(velocity) >= _C):
        raise ValueError("every represented electron velocity must be strictly subluminal")

    beta = velocity / _C
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    sine = float(np.sin(angle / 2.0))
    cosine = float(np.cos(angle / 2.0))
    mu_i = -sine
    mu_s = sine
    denominator_i = 1.0 - beta * mu_i
    denominator_s = 1.0 - beta * mu_s

    # Aberration into the electron rest frame. Both photons share the same
    # transverse direction in the chosen scattering plane.
    mu_i_rest = (mu_i - beta) / denominator_i
    mu_s_rest = (mu_s - beta) / denominator_s
    transverse_i_rest = cosine / (gamma * denominator_i)
    transverse_s_rest = cosine / (gamma * denominator_s)
    cosine_rest = np.clip(
        mu_i_rest * mu_s_rest
        + transverse_i_rest * transverse_s_rest,
        -1.0, 1.0,
    )

    # epsilon_i' = h nu_i'/(m_e c^2), and
    # nu_s'/nu_i' = [1 + epsilon_i'(1-cos(theta'))]^-1.
    electron_rest_energy_eV = 510_998.95069
    photon_energy_eV = 1239.8419843320026 / laser
    epsilon_i_rest = (
        photon_energy_eV / electron_rest_energy_eV
        * gamma * denominator_i
    )
    energy_ratio_rest = 1.0 / (
        1.0 + epsilon_i_rest * (1.0 - cosine_rest)
    )
    thomson_wavelength = laser * denominator_s / denominator_i
    scattered_wavelength = thomson_wavelength / energy_ratio_rest
    recoil_fraction = 1.0 - energy_ratio_rest

    kn_weight = None
    thomson_weight = None
    if incident_polarization is not None:
        raw_polarization = np.asarray(incident_polarization)
        if raw_polarization.dtype.kind not in "iuf":
            raise ValueError("incident_polarization must contain real numeric values")
        polarization = raw_polarization.astype(float, copy=False)
        if (polarization.shape != (3,) or not np.all(np.isfinite(polarization))
                or np.linalg.norm(polarization) <= 0.0):
            raise ValueError("incident_polarization must be a finite non-zero 3-vector")
        polarization = polarization / np.linalg.norm(polarization)
        incident_direction = np.array([-sine, cosine, 0.0])
        scattered_direction = np.array([sine, cosine, 0.0])
        if abs(float(polarization @ incident_direction)) > 1e-6:
            raise ValueError("incident_polarization must be transverse to the incident direction")

        beta_vector = np.zeros((velocity.size, 3), dtype=float)
        beta_vector[:, 0] = beta
        incident_rest = np.column_stack((
            mu_i_rest,
            transverse_i_rest,
            np.zeros(velocity.size),
        ))
        scattered_rest = np.column_stack((
            mu_s_rest,
            transverse_s_rest,
            np.zeros(velocity.size),
        ))
        magnetic_direction_cross_e = np.cross(incident_direction, polarization)
        e_rest = (
            gamma[:, None] * (
                polarization[None, :]
                + np.cross(beta_vector, magnetic_direction_cross_e[None, :])
            )
            - (gamma * gamma / (gamma + 1.0))[:, None]
            * beta_vector
            * np.einsum("ij,j->i", beta_vector, polarization)[:, None]
        )
        e_rest -= (
            np.einsum("ij,ij->i", e_rest, incident_rest)[:, None]
            * incident_rest
        )
        e_norm = np.linalg.norm(e_rest, axis=1)
        if np.any(e_norm <= np.finfo(float).tiny):
            raise ValueError("Lorentz-transformed incident polarization is degenerate")
        e_rest /= e_norm[:, None]
        polarization_projection_squared = np.square(
            np.einsum("ij,ij->i", scattered_rest, e_rest)
        )
        sin_squared_rest = np.maximum(0.0, 1.0 - cosine_rest * cosine_rest)
        # d sigma'/d Omega' divided by r_e^2. This is the polarized
        # Klein-Nishina expression and tends to 1-(e'.n_s')^2 in the
        # zero-recoil Thomson limit.
        differential_rest = 0.5 * energy_ratio_rest**2 * (
            1.0 / energy_ratio_rest + energy_ratio_rest
            - 2.0 * polarization_projection_squared
        )
        kn_weight = (
            denominator_i * np.maximum(differential_rest, 0.0)
            / (gamma**2 * denominator_s**2)
        )
        thomson_differential_rest = 1.0 - polarization_projection_squared
        thomson_weight = (
            denominator_i * np.maximum(thomson_differential_rest, 0.0)
            / (gamma**2 * denominator_s**2)
        )

    return {
        "thomson_scattered_wavelength_nm": thomson_wavelength,
        "compton_scattered_wavelength_nm": scattered_wavelength,
        "compton_energy_ratio_rest_frame": energy_ratio_rest,
        "rest_frame_scattering_angle_cosine": cosine_rest,
        "maximum_recoil_correction_fraction": float(np.max(recoil_fraction)),
        "relative_lab_differential_scattering_weight": kn_weight,
        "relative_lab_thomson_differential_scattering_weight": thomson_weight,
        "cross_section_weight_reference": "classical_electron_radius_squared",
        "kinematics": "exact_single_electron_Compton_four_momentum_conservation",
    }

def reconstruct_relativistic_projected_thomson_evdf(
    wavelength_nm, signal, *, laser_wavelength_nm, scattering_angle_rad,
    instrument_response_matrix, background_signal, velocity_grid_m_s,
    signal_uncertainty=None, signal_covariance=None,
    background_uncertainty=None, background_covariance=None,
    include_compton_recoil=False, incident_polarization=None,
    maximum_recoil_correction_fraction=0.01,
    regularization_strength=None, max_iter=3000,
):
    """Invert a relativistic 1-D q-marginal with Thomson or Compton scattering.

    The Compton route uses exact single-electron four-momentum kinematics and
    the polarized Klein-Nishina differential cross-section for the supplied
    incident polarization. Electrons remain restricted to velocities parallel
    to q in symmetric scattering geometry; this is not a general 3-D EVDF.
    In the default Thomson route, the maximum recoil correction is screened
    against maximum_recoil_correction_fraction. If polarization is supplied,
    the elastic Thomson differential weight is also applied.
    """
    x = _real_vector(wavelength_nm, "wavelength_nm", minimum=9)
    y = _real_vector(signal, "signal", minimum=9)
    b = _real_vector(background_signal, "background_signal", minimum=9)
    v = _real_vector(velocity_grid_m_s, "velocity_grid_m_s", minimum=41)
    if x.shape != y.shape or b.shape != y.shape or np.any(np.diff(x) <= 0) or np.any(np.diff(v) <= 0):
        raise ValueError("spectral arrays must match and both wavelength/velocity grids must increase")
    if np.any(x <= 0) or np.any(v <= -_C) or np.any(v >= _C):
        raise ValueError("wavelengths must be positive and every represented velocity must be subluminal")
    laser = float(laser_wavelength_nm)
    angle = float(scattering_angle_rad)
    if not np.isfinite(laser) or laser <= 0 or not np.isfinite(angle) or not 0 < angle < np.pi:
        raise ValueError("laser wavelength must be positive and scattering angle must lie in (0, pi)")
    response = np.asarray(instrument_response_matrix, dtype=float)
    if response.shape != (x.size, x.size) or not np.all(np.isfinite(response)) or np.any(response < 0):
        raise ValueError("instrument response must be a finite non-negative square matrix")
    if not np.allclose(response.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("measured instrument-response columns must already sum to one")
    if signal_uncertainty is not None and signal_covariance is not None:
        raise ValueError("supply signal_uncertainty or signal_covariance, not both")
    if background_uncertainty is not None and background_covariance is not None:
        raise ValueError("supply background_uncertainty or background_covariance, not both")
    n = x.size
    if signal_uncertainty is not None:
        sc = np.diag(_real_vector(signal_uncertainty, "signal_uncertainty", n)**2)
    elif signal_covariance is not None:
        sc = np.asarray(signal_covariance, dtype=float)
    else:
        sc = None
    if background_uncertainty is not None:
        bc = np.diag(_real_vector(background_uncertainty, "background_uncertainty", n)**2)
    elif background_covariance is not None:
        bc = np.asarray(background_covariance, dtype=float)
    else:
        bc = None
    if sc is not None and (sc.shape != (n,n) or not np.all(np.isfinite(sc)) or not np.allclose(sc,sc.T,rtol=1e-10,atol=1e-13)):
        raise ValueError("signal covariance must be finite, symmetric, and wavelength-sized")
    if bc is not None and (bc.shape != (n,n) or not np.all(np.isfinite(bc)) or not np.allclose(bc,bc.T,rtol=1e-10,atol=1e-13)):
        raise ValueError("background covariance must be finite, symmetric, and wavelength-sized")
    residual_cov = None if sc is None and bc is None else (0 if sc is None else sc) + (0 if bc is None else bc)
    whiten, noise_status = _whitener(n, covariance=residual_cov) if residual_cov is not None else _whitener(n)
    if not isinstance(include_compton_recoil, (bool, np.bool_)):
        raise ValueError("include_compton_recoil must be boolean")
    if include_compton_recoil and incident_polarization is None:
        raise ValueError("incident_polarization is required for polarized Klein-Nishina Compton inversion")
    recoil_limit = float(maximum_recoil_correction_fraction)
    if (not np.isfinite(recoil_limit) or recoil_limit <= 0.0
            or recoil_limit >= 1.0):
        raise ValueError("maximum_recoil_correction_fraction must be finite and lie in (0, 1)")
    kinematics = relativistic_projected_compton_kinematics(
        v, laser, angle,
        incident_polarization=incident_polarization if include_compton_recoil else None,
    )
    maximum_recoil = kinematics["maximum_recoil_correction_fraction"]
    if not include_compton_recoil and maximum_recoil > recoil_limit:
        raise ValueError(
            "The Thomson/no-recoil approximation exceeds the selected correction "
            f"limit ({maximum_recoil:.6g} > {recoil_limit:.6g}); set "
            "include_compton_recoil=True and supply incident_polarization to use "
            "the Compton/Klein-Nishina inverse."
        )
    wavelength_centres = (
        kinematics["compton_scattered_wavelength_nm"] if include_compton_recoil
        else kinematics["thomson_scattered_wavelength_nm"]
    )
    if include_compton_recoil:
        scattering_weights = kinematics["relative_lab_differential_scattering_weight"]
    elif incident_polarization is not None:
        scattering_weights = kinematics["relative_lab_thomson_differential_scattering_weight"]
    else:
        scattering_weights = np.ones(v.size, dtype=float)
    if np.any(~np.isfinite(scattering_weights)) or np.any(scattering_weights < 0.0):
        raise ValueError("Compton differential-scattering weights must be finite and non-negative")
    if np.any(wavelength_centres < x[0]) or np.any(wavelength_centres > x[-1]):
        raise ValueError("velocity grid maps outside the measured wavelength interval")
    latent = np.zeros((n, v.size), dtype=float)
    for j, centre in enumerate(wavelength_centres):
        right = int(np.searchsorted(x, centre, side="right"))
        if right <= 0:
            latent[0,j] = 1.0 / widths[0]
        elif right >= n:
            latent[-1,j] = 1.0 / widths[-1]
        else:
            left = right - 1
            frac = (centre-x[left])/(x[right]-x[left])
            latent[left,j] = (1-frac)/widths[left]
            latent[right,j] = frac/widths[right]
    dv = np.diff(v)
    velocity_weights = _trapezoid_quadrature_weights(v, "velocity_grid_m_s")
    # The latent columns are wavelength-bin probability densities for a
    # mono-velocity population. A continuous projected EVDF f(v) [s/m]
    # contributes through integral K(lambda,v) f(v) dv; omitting these
    # quadrature weights makes the predicted spectrum depend on grid density.
    # Each velocity column is weighted by its lab-frame differential
    # scattering rate before integrating the unknown projected EVDF.
    latent *= scattering_weights[None, :]
    forward = (
        response @ (latent * widths[:,None])
    ) * velocity_weights[None,:] / widths[:,None]
    design = whiten(forward)
    target = whiten(y-b)
    d2 = np.diff(np.eye(v.size), n=2, axis=0) / float(dv[0])**2
    scale = np.linalg.norm(design, ord="fro") / max(np.linalg.norm(d2, ord="fro"), np.finfo(float).tiny)
    candidates = np.logspace(-8, 4, 13) if regularization_strength is None else np.asarray([float(regularization_strength)])
    if np.any(~np.isfinite(candidates)) or np.any(candidates <= 0):
        raise ValueError("regularization_strength must be positive and finite")
    fits, scores = [], []
    for lam in candidates:
        a = np.vstack((design, np.sqrt(lam)*scale*d2))
        z = np.r_[target, np.zeros(d2.shape[0])]
        fit = lsq_linear(a, z, bounds=(0,np.inf), method="trf", max_iter=int(max_iter), tol=1e-10)
        fits.append(fit if fit.success else None)
        if not fit.success:
            scores.append(np.inf); continue
        active = fit.x > max(float(np.max(fit.x))*1e-9, np.finfo(float).tiny)
        if not np.any(active):
            scores.append(np.inf); continue
        da = design[:,active]
        pa = scale*d2[:,active]
        hess = da.T@da + lam*(pa.T@pa)
        try:
            df = float(np.trace(np.linalg.solve(hess, da.T@da)))
        except np.linalg.LinAlgError:
            scores.append(np.inf); continue
        rss = float(np.sum((target-design@fit.x)**2))
        scores.append(float(target.size*rss/max(target.size-df,1e-9)**2))
    if not np.any(np.isfinite(scores)):
        raise ValueError("relativistic projected EVDF inversion failed for all regularization candidates")
    idx = int(np.argmin(scores))
    coeff = np.maximum(fits[idx].x,0.0)
    integral_weights = np.ones(v.size); integral_weights[[0,-1]] = 0.5
    norm = float(np.sum(coeff*integral_weights)*float(dv[0]))
    if norm <= np.finfo(float).tiny:
        raise ValueError("reconstructed projected EVDF has zero integral")
    evdf = coeff/norm
    active = coeff > max(float(np.max(coeff))*1e-9,np.finfo(float).tiny)
    da = design[:,active]; pa = scale*d2[:,active]
    hess = da.T@da + float(candidates[idx])*(pa.T@pa)
    cov_coeff = np.zeros((v.size,v.size))
    cov_active = np.linalg.pinv(hess, hermitian=True)
    if noise_status == "unweighted_no_measurement_uncertainty_supplied":
        rr = target-design@coeff
        cov_active *= float(rr@rr/max(1,target.size-np.count_nonzero(active)))
    cov_coeff[np.ix_(np.flatnonzero(active),np.flatnonzero(active))] = cov_active
    jac_norm = (np.eye(v.size)*norm - np.outer(coeff, integral_weights*float(dv[0])))/norm**2
    cov_evdf = jac_norm@cov_coeff@jac_norm.T
    singular = np.linalg.svd(design, compute_uv=False)
    tol = np.finfo(float).eps*max(design.shape)*(singular[0] if singular.size else 0)
    rank = int(np.count_nonzero(singular > tol))
    condition = (
        float(singular[0] / singular[-1])
        if singular.size and singular[-1] > tol else float("inf")
    )
    data_identifiable = bool(rank == v.size and np.isfinite(condition) and condition < 1e12)
    return {
        "status": "completed" if data_identifiable else "conditional_regularized_candidate",
        "primary_result_status": "identified_from_measurement" if data_identifiable else "not_fully_identifiable_from_measured_spectrum",
        "model": (
            "relativistic_compton_klein_nishina_projected_1d_inverse"
            if include_compton_recoil else
            "exact_relativistic_thomson_limit_projected_1d_kinematic_inversion"
        ),
        "quantity": "Projected 1D electron velocity distribution along the scattering vector",
        "velocity_parallel_m_s": v.copy(),
        "mono_velocity_scattered_wavelength_nm": wavelength_centres.copy(),
        "relative_lab_differential_scattering_weight": scattering_weights.copy(),
        "recoil_included": bool(include_compton_recoil),
        "maximum_recoil_correction_fraction": float(maximum_recoil),
        "maximum_recoil_correction_fraction_limit": float(recoil_limit),
        "scattering_cross_section_model": (
            "polarized_single_scattering_Klein_Nishina_in_electron_rest_frame"
            if include_compton_recoil else
            "polarized_single_scattering_Thomson_in_electron_rest_frame"
            if incident_polarization is not None else
            "kinematic_delta_kernel_without_cross_section_weighting"
        ),
        "projected_evdf_s_m": evdf if data_identifiable else None,
        "regularized_candidate_evdf_s_m": evdf,
        "projected_evdf_covariance": cov_evdf if data_identifiable else None,
        "projected_evdf_standard_uncertainty_s_m": np.sqrt(np.maximum(np.diag(cov_evdf),0)),
        "fitted_spectrum": forward@coeff,
        "residual": (y-b)-forward@coeff,
        "regularization_strength": float(candidates[idx]),
        "regularization_selection": "GCV" if regularization_strength is None else "researcher_supplied",
        "gcv_scores": np.asarray(scores),
        "data_only_jacobian_rank": rank,
        "data_only_jacobian_condition_number": condition,
        "data_only_parameter_count": int(v.size),
        "velocity_grid_points": int(v.size),
        "velocity_quadrature": "trapezoidal_integral_over_velocity_in_forward_operator",
        "velocity_quadrature_weights_m_s": velocity_weights.copy(),
        "noise_model": noise_status,
        "absolute_density_inferred": False,
        "assumptions": [
            "symmetric incident/scattered geometry about the scattering-vector bisector",
            "one-dimensional velocity marginal represented by velocities parallel to the scattering vector",
            (
                "exact Compton four-momentum conservation and polarized Klein-Nishina "
                "differential scattering weight"
                if include_compton_recoil else
                "exact relativistic Doppler kinematics in the Thomson zero-recoil limit"
            ),
            "incident polarization is supplied and transverse to the incident ray for the Compton/Klein-Nishina route",
            "transverse velocity reconstruction and unrestricted 3-D EVDF remain non-identifiable in this single-view inverse",
        ],
        "uncertainty_interpretation": "local regularized linear covariance conditional on the selected regularization, fixed response, fixed background, and stated 1-D kinematic model; not a posterior distribution",
        "experimental_validation": "not_performed",
    }


def reconstruct_psf_aware_thomson_profile(
    measured_channel_values, spatial_mixing_matrix, *, regularization_strength,
    measurement_uncertainty=None, measurement_covariance=None,
    positive=True, max_iter=3000,
):
    """Reconstruct latent spatial nodes from channel values and a fixed PSF/mixing matrix.

    The operator maps latent node values to measured channels. This is a
    regularized inverse problem, not interpolation. Only data-supported modes
    are reported as primary; null-space modes remain non-identifiable.
    """
    y = _real_vector(measured_channel_values, "measured_channel_values", minimum=2)
    h = np.asarray(spatial_mixing_matrix, dtype=float)
    if h.ndim != 2 or h.shape[0] != y.size or not np.all(np.isfinite(h)) or np.any(h < 0):
        raise ValueError("spatial_mixing_matrix must be finite, non-negative, and have one row per measured channel")
    lam = float(regularization_strength)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("regularization_strength must be positive and finite")
    whiten, noise_status = _whitener(y.size, measurement_uncertainty, measurement_covariance)
    a = whiten(h); z = whiten(y)
    if h.shape[1] < 2:
        raise ValueError("at least two latent spatial nodes are required")
    d2 = np.diff(np.eye(h.shape[1]), n=2, axis=0) if h.shape[1] >= 3 else np.zeros((0,h.shape[1]))
    aw = np.vstack((a, np.sqrt(lam)*d2))
    zw = np.r_[z, np.zeros(d2.shape[0])]
    bounds = (0,np.inf) if positive else (-np.inf,np.inf)
    if positive:
        fit = lsq_linear(aw, zw, bounds=bounds, method="trf", max_iter=int(max_iter), tol=1e-10)
        if not fit.success:
            raise ValueError(f"PSF-aware profile inversion failed: {fit.message}")
        values = fit.x
    else:
        values = np.linalg.lstsq(aw,zw,rcond=None)[0]
    scaled = a / np.maximum(np.linalg.norm(a,axis=0),np.finfo(float).tiny)[None,:]
    singular = np.linalg.svd(scaled,compute_uv=False)
    tol = np.finfo(float).eps*max(scaled.shape)*(singular[0] if singular.size else 0)
    rank = int(np.count_nonzero(singular>tol))
    condition = float(singular[0]/singular[-1]) if singular.size and singular[-1]>tol else float("inf")
    data_identifiable = rank == h.shape[1] and np.isfinite(condition) and condition < 1e12
    active = np.ones(values.size,dtype=bool) if not positive else values > max(float(np.max(values))*1e-9,np.finfo(float).tiny)
    aa = a[:,active]; pp = d2[:,active]
    normal = aa.T@aa + lam*(pp.T@pp)
    cov = np.zeros((values.size,values.size))
    cov_active = np.linalg.pinv(normal,hermitian=True)
    residual = z-a@values
    if noise_status == "unweighted_no_measurement_uncertainty_supplied":
        cov_active *= float(residual@residual/max(1,y.size-rank))
    cov[np.ix_(np.flatnonzero(active),np.flatnonzero(active))] = cov_active
    return {
        "status": "identified_from_measurement" if data_identifiable else "conditional_regularized_candidate",
        "latent_profile": values if data_identifiable else None,
        "regularized_candidate_profile": values,
        "profile_covariance": cov if data_identifiable else None,
        "fitted_channel_values": h@values,
        "residual": y-h@values,
        "spatial_mixing_matrix": h.copy(),
        "data_only_rank": rank,
        "latent_node_count": int(h.shape[1]),
        "condition_number": condition,
        "null_space_dimension": int(max(0,h.shape[1]-rank)),
        "regularization_strength": lam,
        "positive_constraint": bool(positive),
        "noise_model": noise_status,
        "identifiability": {
            "primary": ["latent_profile"] if data_identifiable else [],
            "conditional": ["regularized_candidate_profile; conditional on supplied PSF, regularization, and noise model"],
            "non_identifiable": ["spatial modes in the data-only null space", "sub-resolution modes not encoded by the supplied mixing operator"],
        },
        "uncertainty_interpretation": "local regularized linear covariance on the active set; it is not a posterior interval and excludes PSF/mixing-operator uncertainty",
        "experimental_validation": "not_performed",
    }
