"""Kinetic Thomson models beyond non-relativistic Maxwellian fitting.

The relativistic single-particle model uses exact photon-frequency kinematics
for a Lorentz-boosted Maxwell-Juttner electron population. The collective
kappa model evaluates the unmagnetized longitudinal dielectric response from
the supplied 1-D projected electron VDF and Maxwellian ion populations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaln
from numpy.polynomial.legendre import leggauss

E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
ME = 9.1093837015e-31
AMU = 1.66053906660e-27
C = 299792458.0
E_REST_EV = 510998.95069
ELECTRON_CLASSICAL_RADIUS_M = 2.8179403262e-15
_LANDAU_NODES, _LANDAU_WEIGHTS = leggauss(192)


def _trapezoid(y, x):
    """NumPy-version-independent trapezoidal integration."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0
        integrate = np.trapz
    return float(integrate(y, x))


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be positive and finite")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _apply_response(response, spectrum, wavelength_nm):
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(spectrum, dtype=float)
    r = np.asarray(response, dtype=float)
    if r.shape != (x.size, x.size) or y.shape != x.shape:
        raise ValueError("response must be square on the supplied wavelength grid")
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5*(x[:-1] + x[1:])
    edges[0] = x[0] - 0.5*(x[1]-x[0])
    edges[-1] = x[-1] + 0.5*(x[-1]-x[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise ValueError("wavelength grid must define positive bin widths")
    return (r @ (y*widths))/widths


def relativistic_maxwell_juttner_spectrum(
    wavelength_nm, laser_wavelength_nm, scattering_angle_rad,
    electron_temperature_eV, *, flow_projection_m_s=0.0,
    incident_polarization, radial_order=36, angular_order=18,
    maximum_recoil_correction_fraction=0.01,
    require_full_spectrum=True,
):
    """Return a unit-area relativistic Thomson spectrum on a wavelength grid.

    Electron momenta follow an isotropic Maxwell-Juttner distribution in the
    plasma rest frame and are Lorentz-boosted along the scattering-vector
    direction by flow_projection_m_s. The scattered wavelength uses the exact
    elastic Thomson frequency ratio in the instantaneous electron rest frame:
    lambda_s/lambda_i=(1-n_s·beta)/(1-n_i·beta).

    The supplied incident polarization is Lorentz-transformed into each
    electron rest frame and the polarized Thomson differential cross-section
    is evaluated there, together with incident-flux and aberration Jacobians.
    Recoil and quantum Compton corrections are not included.
    """
    x = np.asarray(wavelength_nm, dtype=float)
    laser = _positive(laser_wavelength_nm, "laser_wavelength_nm")
    angle = float(scattering_angle_rad)
    temperature = _positive(electron_temperature_eV, "electron_temperature_eV")
    flow = float(flow_projection_m_s)
    if (x.ndim != 1 or x.size < 7 or np.any(~np.isfinite(x))
            or np.any(np.diff(x) <= 0) or np.any(x <= 0)):
        raise ValueError("wavelength_nm must be a finite, positive, strictly increasing vector")
    if not np.isfinite(angle) or not 0 < angle <= np.pi:
        raise ValueError("scattering_angle_rad must lie in (0, pi]")
    if not np.isfinite(flow) or abs(flow) >= C:
        raise ValueError("flow_projection_m_s must be finite and strictly subluminal (|v| < c)")
    if int(radial_order) < 12 or int(angular_order) < 8:
        raise ValueError("radial_order >=12 and angular_order >=8 are required")
    recoil_limit = _positive(maximum_recoil_correction_fraction, "maximum_recoil_correction_fraction")
    if recoil_limit >= 1.0:
        raise ValueError("maximum_recoil_correction_fraction must be below 1")

    theta = temperature/E_REST_EV
    beta_bulk = flow/C
    gamma_bulk = 1.0/np.sqrt(1.0-beta_bulk**2)
    sine, cosine = np.sin(angle/2.0), np.cos(angle/2.0)
    # Choose coordinates with q=k_s-k_i along +x. Then n_s-n_i=2 sin(theta/2) xhat.
    n_incident = np.array([-sine, cosine, 0.0])
    n_scattered = np.array([sine, cosine, 0.0])
    polarization = np.asarray(incident_polarization, dtype=float)
    if polarization.shape != (3,) or np.any(~np.isfinite(polarization)) or np.linalg.norm(polarization) == 0:
        raise ValueError("incident_polarization must be a finite non-zero 3-vector")
    if abs(float(np.dot(polarization/np.linalg.norm(polarization), n_incident))) > 1e-6:
        raise ValueError("incident_polarization must be transverse to incident_direction")
    polarization = polarization/np.linalg.norm(polarization)

    radial_nodes, radial_weights = leggauss(int(radial_order))
    umax = max(12.0*np.sqrt(theta)+12.0*theta, 1e-5)
    u = 0.5*umax*(radial_nodes+1.0)
    wu = 0.5*umax*radial_weights
    mu, wmu = leggauss(int(angular_order))
    nphi = max(16, 2*int(angular_order))
    phi = 2.0*np.pi*np.arange(nphi)/nphi
    U, MU_GRID, PHI = np.meshgrid(u, mu, phi, indexing="ij")
    GAMMA_REST = np.sqrt(1.0+U*U)
    beta_prime = U/GAMMA_REST
    beta_parallel = beta_prime*MU_GRID
    beta_perp = beta_prime*np.sqrt(np.maximum(1.0-MU_GRID*MU_GRID, 0.0))
    denominator = 1.0+beta_bulk*beta_parallel
    bx = (beta_parallel+beta_bulk)/denominator
    by = beta_perp*np.cos(PHI)/(gamma_bulk*denominator)
    bz = beta_perp*np.sin(PHI)/(gamma_bulk*denominator)
    beta = np.stack((bx, by, bz), axis=-1)
    beta_mag = np.sqrt(np.sum(beta*beta, axis=-1))
    gamma_lab = gamma_bulk*GAMMA_REST*denominator
    ni_dot = np.einsum("...j,j->...", beta, n_incident)
    ns_dot = np.einsum("...j,j->...", beta, n_scattered)
    lambda_scattered = laser*(1.0-ns_dot)/(1.0-ni_dot)

    beta_hat = np.divide(
        beta, beta_mag[..., None],
        out=np.zeros_like(beta), where=beta_mag[..., None] > 0,
    )
    def aberrated_direction(direction):
        mu_direction = np.einsum("...j,j->...", beta_hat, direction)
        denominator_direction = 1.0-beta_mag*mu_direction
        parallel = (mu_direction-beta_mag)/denominator_direction
        perpendicular = (
            direction-mu_direction[..., None]*beta_hat
        )/(gamma_lab[..., None]*denominator_direction[..., None])
        return parallel[..., None]*beta_hat+perpendicular

    ni_rest = aberrated_direction(n_incident)
    ns_rest = aberrated_direction(n_scattered)
    # Screen the Thomson/no-recoil approximation in each represented electron
    # rest frame. The Compton fractional correction is r/(1+r), where
    # r = (h nu' / m_e c^2) * (1 - cos(theta_scatter')).
    rest_angle_cosine = np.clip(
        np.einsum("...j,...j->...", ni_rest, ns_rest), -1.0, 1.0
    )
    incident_photon_energy_eV = 1239.8419843320026 / laser
    rest_photon_energy_eV = (
        incident_photon_energy_eV * gamma_lab * (1.0 - ni_dot)
    )
    recoil_parameter = (
        rest_photon_energy_eV / E_REST_EV * (1.0 - rest_angle_cosine)
    )
    recoil_correction_fraction = float(np.max(
        recoil_parameter / (1.0 + recoil_parameter)
    ))
    if not np.isfinite(recoil_correction_fraction) or recoil_correction_fraction > recoil_limit:
        raise ValueError(
            "The Thomson/no-recoil approximation is outside the selected validity "
            f"limit: maximum represented Compton correction={recoil_correction_fraction:.6g}, "
            f"limit={recoil_limit:.6g}. Use a lower photon energy or a Compton-capable model."
        )
    # Lorentz-transform the incident electric field to each electron rest frame.
    magnetic_direction_cross_e = np.cross(n_incident, polarization)
    e_rest = (
        gamma_lab[..., None]*(
            polarization+np.cross(beta, magnetic_direction_cross_e)
        )
        -(gamma_lab*gamma_lab/(gamma_lab+1.0))[..., None]
        *beta*np.einsum("...j,j->...", beta, polarization)[..., None]
    )
    e_rest -= np.einsum("...j,...j->...", e_rest, ni_rest)[..., None]*ni_rest
    e_rest_norm = np.linalg.norm(e_rest, axis=-1)
    e_rest = np.divide(
        e_rest, e_rest_norm[..., None],
        out=np.zeros_like(e_rest), where=e_rest_norm[..., None] > 0,
    )
    polarization_factor = np.maximum(
        1.0-np.square(np.einsum("...j,...j->...", ns_rest, e_rest)), 0.0
    )
    differential_weight = (
        (1.0-ni_dot)*polarization_factor
        /(gamma_lab*gamma_lab*np.square(1.0-ns_dot))
    )
    radial_weight = (
        wu[:, None, None]*u[:, None, None]**2
        * np.exp(-(GAMMA_REST-1.0)/theta)
    )
    distribution_quadrature_weight = (
        radial_weight*wmu[None, :, None]*(2.0*np.pi/nphi)
    )
    distribution_normalization = float(np.sum(distribution_quadrature_weight))
    if not np.isfinite(distribution_normalization) or distribution_normalization <= 0.0:
        raise ValueError("Maxwell-Juttner momentum quadrature has a non-positive normalization")
    quadrature_weight = distribution_quadrature_weight * differential_weight

    edges = np.empty(x.size+1, dtype=float)
    edges[1:-1] = 0.5*(x[:-1]+x[1:])
    edges[0] = x[0]-0.5*(x[1]-x[0])
    edges[-1] = x[-1]+0.5*(x[-1]-x[-2])
    # Deposit quadrature weight with a linear (cloud-in-cell) spectral basis.
    # A hard histogram is piecewise constant in flow/temperature parameters and
    # gives least-squares optimizers a zero or discontinuous numerical Jacobian.
    # Linear deposition is continuous, conserves probability and represents the
    # finite wavelength-bin measurement without introducing a fitted line shape.
    centers = 0.5 * (edges[:-1] + edges[1:])
    sampled_wavelength = lambda_scattered.ravel()
    sampled_weight = quadrature_weight.ravel()
    total_weight = float(np.sum(sampled_weight))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("Relativistic Thomson quadrature has a non-positive integrated scattering weight")
    effective_differential_cross_section = (
        ELECTRON_CLASSICAL_RADIUS_M**2 * total_weight / distribution_normalization
    )
    inside = (
        (sampled_wavelength >= edges[0])
        & (sampled_wavelength <= edges[-1])
        & np.isfinite(sampled_weight)
        & (sampled_weight >= 0.0)
    )
    captured = float(np.sum(sampled_weight[inside])) / total_weight if total_weight > 0 else 0.0
    counts = np.zeros(x.size, dtype=float)
    if np.any(inside):
        lam = sampled_wavelength[inside]
        weight = sampled_weight[inside]
        right = np.searchsorted(centers, lam, side="right")
        below = right == 0
        above = right == centers.size
        right = np.clip(right, 1, centers.size - 1)
        left = right - 1
        fraction = (lam - centers[left]) / (centers[right] - centers[left])
        left[below] = 0
        right[below] = 0
        fraction[below] = 0.0
        left[above] = centers.size - 1
        right[above] = centers.size - 1
        fraction[above] = 0.0
        np.add.at(counts, left, weight * (1.0 - fraction))
        np.add.at(counts, right, weight * fraction)
    if total_weight <= 0 or (require_full_spectrum and captured < 0.995):
        raise ValueError(
            "wavelength_nm must capture at least 99.5% of the relativistic Thomson spectrum; "
            f"captured fraction was {captured:.6g}"
        )
    widths = np.diff(edges)
    density = counts/widths
    area = float(np.sum(counts))
    if area <= 0:
        raise ValueError("relativistic Thomson quadrature produced zero spectral power")
    density /= area
    centers = 0.5*(edges[:-1]+edges[1:])
    centroid = float(np.sum(centers*density*widths))
    variance = float(np.sum((centers-centroid)**2*density*widths))
    return {
        "spectral_density_per_nm": density,
        "wavelength_bin_widths_nm": widths,
        "integrated_probability": 1.0,
        "effective_differential_cross_section_m2_sr": float(effective_differential_cross_section),
        "fitted_centroid_nm": centroid,
        "spectral_rms_width_nm": float(np.sqrt(max(variance, 0.0))),
        "captured_probability": captured,
        "relativistic_kinematics": True,
        "frequency_ratio": "lambda_s/lambda_i=(1-n_s dot beta)/(1-n_i dot beta)",
        "distribution": "Lorentz-boosted isotropic Maxwell-Juttner",
        "recoil_included": False,
        "thomson_recoil_correction_fraction_max_on_quadrature_support": recoil_correction_fraction,
        "thomson_recoil_correction_fraction_limit": float(recoil_limit),
        "thomson_no_recoil_validity_status": "within_user_selected_fractional_correction_limit",
        "polarization_model": "supplied incident polarization Lorentz-transformed to the electron rest frame",
        "quadrature": {"radial_order": int(radial_order), "angular_order": int(angular_order), "gyrophase_points": nphi},
    }


def fit_relativistic_maxwell_juttner(
    wavelength_nm, signal, laser_wavelength_nm, scattering_angle_rad,
    response, uncertainty=None, *, incident_direction, scattered_direction,
    incident_polarization, radial_order=36, angular_order=18,
    maximum_recoil_correction_fraction=0.01,
):
    """Fit relativistic Maxwell-Juttner Te and q-parallel bulk flow to one spectrum."""

    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(signal, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 12:
        raise ValueError("wavelength_nm and signal must be matching vectors with >=12 samples")
    ni = np.array(incident_direction, dtype=float, copy=True)
    ns = np.array(scattered_direction, dtype=float, copy=True)
    if ni.shape != (3,) or ns.shape != (3,) or np.any(~np.isfinite(ni)) or np.any(~np.isfinite(ns)):
        raise ValueError("incident_direction and scattered_direction must be finite 3-vectors")
    if np.linalg.norm(ni) == 0 or np.linalg.norm(ns) == 0:
        raise ValueError("incident_direction and scattered_direction must be non-zero")
    ni /= np.linalg.norm(ni)
    ns /= np.linalg.norm(ns)
    angle_from_vectors = float(np.arccos(np.clip(np.dot(ni, ns), -1.0, 1.0)))
    if not np.isclose(angle_from_vectors, scattering_angle_rad, rtol=0.0, atol=1e-5):
        raise ValueError("scattering_angle_rad is inconsistent with incident_direction and scattered_direction")
    polarization_lab = np.asarray(incident_polarization, dtype=float)
    if (polarization_lab.shape != (3,) or np.any(~np.isfinite(polarization_lab))
            or np.linalg.norm(polarization_lab) == 0):
        raise ValueError("incident_polarization must be a finite non-zero 3-vector")
    polarization_lab = polarization_lab/np.linalg.norm(polarization_lab)
    if abs(float(np.dot(polarization_lab, ni))) > 1e-6:
        raise ValueError("incident_polarization must be transverse to incident_direction")
    qhat = ns-ni
    qhat /= np.linalg.norm(qhat)
    e3 = np.cross(ni, ns)
    if np.linalg.norm(e3) <= 1e-12:
        e3 = polarization_lab-np.dot(polarization_lab,qhat)*qhat
    e3 /= np.linalg.norm(e3)
    e2 = np.cross(qhat,e3)
    e2 /= np.linalg.norm(e2)
    local_to_lab = np.column_stack((qhat,e2,e3))
    polarization_local = local_to_lab.T@polarization_lab
    scale = max(float(np.max(np.abs(y))), np.finfo(float).tiny)
    ys = y/scale
    unc = None
    covariance = None
    covariance_cholesky = None
    if uncertainty is not None:
        supplied_uncertainty = np.asarray(uncertainty, dtype=float)
        if supplied_uncertainty.shape == y.shape:
            unc = supplied_uncertainty/scale
            if np.any(~np.isfinite(unc)) or np.any(unc <= 0):
                raise ValueError("uncertainty must be positive and match signal")
        elif supplied_uncertainty.shape == (y.size, y.size):
            covariance = supplied_uncertainty/(scale*scale)
            if np.any(~np.isfinite(covariance)):
                raise ValueError("spectral_covariance must contain only finite values")
            covariance_scale = float(np.max(np.abs(covariance)))
            if covariance_scale <= 0 or not np.allclose(
                covariance, covariance.T, rtol=1e-9, atol=1e-12*covariance_scale
            ):
                raise ValueError("spectral_covariance must be symmetric and non-zero")
            try:
                covariance_cholesky = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError("spectral_covariance must be positive definite") from exc
        else:
            raise ValueError("uncertainty must be a positive pointwise vector or a full n_wavelength x n_wavelength covariance matrix")
    residual_scale = (
        np.full_like(ys, max(float(np.max(np.abs(ys))), np.finfo(float).tiny))
        if unc is None and covariance_cholesky is None else unc
    )
    def whiten(residual):
        if covariance_cholesky is not None:
            return np.linalg.solve(covariance_cholesky, residual)
        return residual/residual_scale
    edge = max(2, x.size//10)
    baseline = float(np.median(np.r_[ys[:edge], ys[-edge:]]))
    positive = np.maximum(ys-baseline, 0.0)
    if np.sum(positive) <= 0:
        raise ValueError("signal must contain a positive Thomson-scattering feature")
    centroid0 = float(np.sum(x*positive)/np.sum(positive))
    sigma0 = float(np.sqrt(np.sum((x-centroid0)**2*positive)/np.sum(positive)))
    sigma0 = max(sigma0, float(np.median(np.diff(x))))
    sine = np.sin(float(scattering_angle_rad)/2.0)
    te0 = E_REST_EV*(sigma0/(2.0*float(laser_wavelength_nm)*sine))**2
    if not np.isfinite(te0) or te0 <= 0:
        raise ValueError("The measured spectral width does not provide a positive relativistic temperature initial estimate")
    beta0 = -(centroid0-float(laser_wavelength_nm))/(2.0*float(laser_wavelength_nm)*sine)
    if not np.isfinite(beta0) or abs(beta0) >= 1.0:
        raise ValueError("The measured spectral centroid implies a non-physical initial flow projection |v| >= c")
    amp0 = max(float(np.max(positive)), np.finfo(float).tiny)
    p0 = np.array([np.log(te0), np.arctanh(beta0), np.log(amp0), baseline, 0.0])

    def evaluate(parameters, return_parts=False, *, require_full_spectrum=True):
        te = float(np.exp(np.clip(parameters[0], -700.0, 700.0)))
        beta_flow = float(np.tanh(parameters[1]))
        physical = relativistic_maxwell_juttner_spectrum(
            x, laser_wavelength_nm, scattering_angle_rad, te,
            flow_projection_m_s=beta_flow*C,
            incident_polarization=polarization_local,
            radial_order=radial_order, angular_order=angular_order,
            maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
            require_full_spectrum=require_full_spectrum,
        )
        shape = _apply_response(response, physical["spectral_density_per_nm"], x)
        amplitude = float(np.exp(np.clip(parameters[2], -700.0, 700.0)))
        offset, slope = float(parameters[3]), float(parameters[4])
        predicted = amplitude*shape+offset+slope*(x-float(laser_wavelength_nm))
        return (predicted, physical, shape) if return_parts else predicted

    def objective(parameters):
        try:
            return whiten(evaluate(parameters, require_full_spectrum=False)-ys)
        except ValueError as exc:
            message = str(exc)
            if "capture at least 99.5%" in message or "produced zero spectral power" in message:
                # A trial step can move the spectrum outside the measured window.
                # Penalize that trial; retain the strict full-window requirement
                # for the final reported physical fit.
                return np.full_like(ys, 1.0e6)
            raise

    # Keep the atanh(beta) coordinate away from floating-point saturation: tanh(±20)
    # rounds to exactly ±1 and would create a spurious |v|=c trial during optimization.
    fit_bounds = (
        np.array([-np.inf, -18.0, -np.inf, -np.inf, -np.inf]),
        np.array([np.inf, 18.0, np.inf, np.inf, np.inf]),
    )
    fit = least_squares(
        objective, p0, x_scale="jac", max_nfev=1200, bounds=fit_bounds,
    )
    # A broad baseline slope can partially trade against a small Doppler shift.
    # If refinement reverses the centroid-implied flow sign, retry from a
    # smaller centroid-guided shift and from zero flow, retaining the best fit.
    # These are numerical starts only, not priors or physical flow constraints.
    if abs(beta0) > 1e-8 and np.sign(np.tanh(fit.x[1])) != np.sign(beta0):
        for alternate_beta in (0.5 * beta0, 0.0):
            alternate_start = p0.copy()
            alternate_start[1] = np.arctanh(alternate_beta)
            alternate_fit = least_squares(
                objective, alternate_start, x_scale="jac", max_nfev=1200,
                bounds=fit_bounds,
            )
            if alternate_fit.success and (
                not fit.success or alternate_fit.cost < fit.cost
            ):
                fit = alternate_fit
    if not fit.success:
        raise ValueError(f"relativistic Maxwell-Juttner Thomson fit failed: {fit.message}")
    jacobian = np.asarray(fit.jac, dtype=float)
    column_norms = np.linalg.norm(jacobian, axis=0)
    scaled_jacobian = np.divide(
        jacobian, column_norms[None, :],
        out=np.zeros_like(jacobian),
        where=column_norms[None, :] > np.finfo(float).tiny,
    )
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    rank_tolerance = (
        np.finfo(float).eps * max(scaled_jacobian.shape)
        * (singular_values[0] if singular_values.size else 0.0)
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0 else float("inf")
    )
    if rank < 5 or not np.isfinite(condition) or condition > 1e12:
        raise ValueError(
            "relativistic Maxwell-Juttner Thomson fit is not jointly identifiable "
            f"(scaled Jacobian rank {rank}/5, condition number {condition:.6g})"
        )
    predicted_scaled, physical, shape = evaluate(
        fit.x, return_parts=True, require_full_spectrum=True
    )
    dof = max(1, x.size-5)
    covariance_fit = np.linalg.pinv(fit.jac.T@fit.jac)
    if unc is None and covariance_cholesky is None:
        covariance_fit *= float(2.0*fit.cost/dof)
    te = float(np.exp(np.clip(fit.x[0], -700.0, 700.0)))
    beta_flow = float(np.tanh(fit.x[1]))
    amplitude_scaled = float(np.exp(np.clip(fit.x[2], -700.0, 700.0)))
    transform = np.diag([te, C*(1.0-beta_flow*beta_flow), amplitude_scaled*scale, scale, scale])
    covariance_physical = transform@covariance_fit@transform.T
    def derived_spectral_quantities(parameters):
        _, sampled_physical, sampled_shape = evaluate(
            parameters, return_parts=True, require_full_spectrum=False
        )
        sampled_window_area = (
            float(np.exp(np.clip(parameters[2], -700.0, 700.0)))
            * scale * float(np.sum(
                sampled_shape * sampled_physical["wavelength_bin_widths_nm"]
            ))
        )
        captured_probability = float(sampled_physical["captured_probability"])
        if not np.isfinite(captured_probability) or captured_probability <= 0.0:
            raise ValueError("The fitted relativistic spectrum has no finite captured probability")
        total_area = sampled_window_area / captured_probability
        return (
            float(sampled_physical["effective_differential_cross_section_m2_sr"]),
            captured_probability,
            total_area,
            sampled_window_area,
        )
    cross_section_gradient = np.zeros(5, dtype=float)
    area_gradient = np.zeros(5, dtype=float)
    captured_gradient = np.zeros(5, dtype=float)
    for parameter_index in range(3):
        step = np.cbrt(np.finfo(float).eps) * max(abs(float(fit.x[parameter_index])), 1.0)
        plus = np.asarray(fit.x, dtype=float).copy()
        minus = np.asarray(fit.x, dtype=float).copy()
        plus[parameter_index] += step
        minus[parameter_index] -= step
        cs_plus, capture_plus, area_plus, _ = derived_spectral_quantities(plus)
        cs_minus, capture_minus, area_minus, _ = derived_spectral_quantities(minus)
        cross_section_gradient[parameter_index] = (cs_plus-cs_minus)/(2.0*step)
        captured_gradient[parameter_index] = (capture_plus-capture_minus)/(2.0*step)
        area_gradient[parameter_index] = (area_plus-area_minus)/(2.0*step)
    cross_section, captured_probability, area, window_area = derived_spectral_quantities(fit.x)
    cross_section_variance = float(
        cross_section_gradient @ covariance_fit @ cross_section_gradient
    )
    cross_section_uncertainty = float(np.sqrt(max(cross_section_variance, 0.0)))
    area_variance = float(area_gradient @ covariance_fit @ area_gradient)
    area_sigma = float(np.sqrt(max(area_variance, 0.0)))
    captured_variance = float(captured_gradient @ covariance_fit @ captured_gradient)
    captured_uncertainty = float(np.sqrt(max(captured_variance, 0.0)))
    power_cross_section_covariance = float(
        area_gradient @ covariance_fit @ cross_section_gradient
    )
    power_cross_section_relative_covariance = (
        power_cross_section_covariance/(area*cross_section)
        if area > 0.0 and cross_section > 0.0 else 0.0
    )
    amplitude = amplitude_scaled*scale
    offset = float(fit.x[3]*scale)
    slope = float(fit.x[4]*scale)
    fitted = predicted_scaled*scale
    residual = y-fitted
    return {
        "electron_temperature_eV": te,
        "electron_temperature_uncertainty_eV": float(np.sqrt(max(covariance_physical[0, 0], 0.0))),
        "electron_velocity_projection_m_s": beta_flow*C,
        "electron_velocity_projection_uncertainty_m_s": float(np.sqrt(max(covariance_physical[1, 1], 0.0))),
        "fitted_center_nm": physical["fitted_centroid_nm"],
        "intrinsic_sigma_nm": physical["spectral_rms_width_nm"],
        "amplitude": amplitude,
        "background_intercept": offset,
        "background_slope_per_nm": slope,
        "fit_covariance": covariance_physical,
        "spectral_weighting_status": (
            "full_within_channel_generalized_least_squares"
            if covariance_cholesky is not None else
            "diagonal_pointwise_measurement_uncertainty_weighted"
            if unc is not None else
            "unweighted_least_squares_no_measurement_uncertainty_supplied"
        ),
        "fit_covariance_interpretation": (
            "local_parameter_covariance_from_supplied_full_spectral_covariance"
            if covariance_cholesky is not None else
            "local_parameter_covariance_from_supplied_pointwise_measurement_uncertainty"
            if unc is not None else
            "residual_variance_scaled_local_covariance_not_measurement_noise_calibrated"
        ),
        "fit_parameters": np.array([te, beta_flow*C, amplitude, offset, slope], dtype=float),
        "fit_parameter_names": ["electron_temperature_eV", "electron_velocity_projection_m_s", "amplitude", "background_intercept", "background_slope_per_nm"],
        "integrated_scattered_power_W": area,
        "integrated_scattered_power_uncertainty_W": area_sigma,
        "integrated_scattered_power_fit_uncertainty_W": area_sigma,
        "effective_differential_cross_section_m2_sr": cross_section,
        "effective_differential_cross_section_uncertainty_m2_sr": cross_section_uncertainty,
        "power_cross_section_relative_covariance": power_cross_section_relative_covariance,
        "captured_spectral_probability_uncertainty": captured_uncertainty,
        "integrated_scattered_power_window_W": window_area,
        "effective_cross_section_uncertainty_method": "finite-difference propagation of fitted Te/flow covariance through the Lorentz-transformed polarized Thomson cross-section",
        "reduced_chi_square": (
            float(2.0*fit.cost/dof)
            if covariance_cholesky is not None or unc is not None else None
        ),
        "fitted_spectrum": fitted,
        "residual": residual,
        "optimizer_success": bool(fit.success),
        "jacobian_rank": rank,
        "jacobian_condition_number": condition,
        "model": "relativistic non-collective Thomson scattering with Lorentz-boosted Maxwell-Juttner electrons",
        "relativistic_kinematics": True,
        "recoil_included": False,
        "thomson_recoil_correction_fraction_max_on_quadrature_support": physical[
            "thomson_recoil_correction_fraction_max_on_quadrature_support"
        ],
        "thomson_recoil_correction_fraction_limit": physical["thomson_recoil_correction_fraction_limit"],
        "thomson_no_recoil_validity_status": physical["thomson_no_recoil_validity_status"],
        "quadrature_diagnostics": physical["quadrature"],
        "captured_spectral_probability": captured_probability,
        "assumptions": [
            "Single-scattering non-collective Thomson regime.",
            "Isotropic Maxwell-Juttner electron distribution in its rest frame.",
            "Bulk electron flow is parallel to the scattering vector.",
            "The incident polarization is explicitly supplied and Lorentz-transformed into each electron rest frame.",
            "Exact relativistic photon-frequency ratio and Lorentz aberration are used.",
            "Photon recoil/Compton corrections are neglected; the Thomson limit is assumed.",
            "Measured instrument response is applied to the calculated spectral density."
        ],
    }


def _kappa_pdf_and_derivatives(velocity, standard_deviation, kappa):
    nu = 2.0*kappa-1.0
    if nu <= 2.0:
        raise ValueError("kappa must be > 1.5 for finite variance")
    scale = standard_deviation*np.sqrt((nu-2.0)/nu)
    log_norm = (
        gammaln((nu+1.0)/2.0)-gammaln(nu/2.0)
        -0.5*np.log(nu*np.pi)-np.log(scale)
    )
    denominator = nu*scale*scale+velocity*velocity
    pdf = np.exp(log_norm)*(1.0+velocity*velocity/(nu*scale*scale))**(-(nu+1.0)/2.0)
    first = -(nu+1.0)*velocity/denominator*pdf
    second = (-(nu+1.0)*nu*scale*scale+(nu+1.0)*(nu+2.0)*velocity*velocity)/(denominator*denominator)*pdf
    return pdf, first, second


def _maxwellian_pdf_and_derivatives(velocity, standard_deviation):
    pdf = np.exp(-0.5*(velocity/standard_deviation)**2)/(np.sqrt(2.0*np.pi)*standard_deviation)
    first = -velocity/(standard_deviation**2)*pdf
    second = ((velocity**2)/(standard_deviation**4)-1.0/(standard_deviation**2))*pdf
    return pdf, first, second


def _landau_susceptibility(phase_velocity, plasma_frequency_squared, wavenumber,
                           standard_deviation, distribution="maxwellian", kappa=None):
    if distribution == "kappa":
        pdf_fn = lambda v: _kappa_pdf_and_derivatives(v, standard_deviation, kappa)
    else:
        pdf_fn = lambda v: _maxwellian_pdf_and_derivatives(v, standard_deviation)
    vphi = float(phase_velocity)
    span = max(30.0*standard_deviation, 2.0*abs(vphi)+20.0*standard_deviation)
    velocity = span*_LANDAU_NODES
    _, derivative, _ = pdf_fn(velocity)
    _, derivative_at, second_at = pdf_fn(np.asarray(vphi))
    denominator = velocity-vphi
    regular = np.divide(
        derivative-derivative_at, denominator,
        out=np.full_like(velocity, float(second_at)),
        where=np.abs(denominator) > 1e-10*max(span, standard_deviation),
    )
    principal = span*float(np.sum(_LANDAU_WEIGHTS*regular))
    principal += float(derivative_at)*np.log(abs((span-vphi)/(span+vphi)))
    landau_integral = principal+1j*np.pi*float(derivative_at)
    return -(plasma_frequency_squared/(wavenumber*wavenumber))*landau_integral


def _resolve_ion_species(ion_species):
    result = []
    for index, item in enumerate(ion_species):
        if isinstance(item, Mapping):
            name = str(item.get("name", f"ion_{index}"))
            charge_state = float(item["charge_state"])
            if "mass_kg" in item:
                mass = _positive(item["mass_kg"], f"ion_species[{index}].mass_kg")
            else:
                mass = _positive(item["mass_amu"], f"ion_species[{index}].mass_amu")*AMU
        else:
            try:
                from plasmapy.particles import Particle
            except ImportError as exc:
                raise ImportError("String ion species require the optional PlasmaPy particle package; alternatively provide explicit name, charge_state and mass_amu/mass_kg mappings.") from exc
            particle = item if hasattr(item, "mass") and hasattr(item, "charge_number") else Particle(item)
            name = str(particle)
            mass = float(particle.mass.to_value("kg"))
            charge_state = float(particle.charge_number)
        if not np.isfinite(charge_state) or charge_state <= 0:
            raise ValueError(f"ion_species[{index}] must have a positive charge state")
        result.append({"name": name, "mass_kg": mass, "charge_state": charge_state})
    return result


def collective_kappa_spectral_density(
    wavelength_nm, laser_wavelength_nm, scattering_angle_rad, *,
    electron_density_m3, electron_temperature_eV, kappa,
    ion_species, ion_fractions, ion_temperature_eV,
    electron_velocity_m_s, ion_velocity_m_s,
    incident_direction, scattered_direction,
):
    """Unmagnetized collective spectrum with a kappa electron VDF and Maxwellian ions.

    The longitudinal susceptibilities are evaluated from the Landau-contour
    integral of each supplied 1-D projected velocity distribution. This is a
    collective dielectric model; it is not a non-collective Student-t fit.
    """
    x = np.asarray(wavelength_nm, dtype=float)
    ne = _positive(electron_density_m3, "electron_density_m3")
    te = _positive(electron_temperature_eV, "electron_temperature_eV")
    kap = float(kappa)
    if not np.isfinite(kap) or kap <= 1.5:
        raise ValueError("kappa must be finite and > 1.5")
    laser = _positive(laser_wavelength_nm, "laser_wavelength_nm")
    angle = float(scattering_angle_rad)
    if x.ndim != 1 or x.size < 7 or np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm must be finite and strictly increasing")
    if not 0 < angle <= np.pi:
        raise ValueError("scattering_angle_rad must lie in (0, pi]")
    ni = np.asarray(incident_direction, dtype=float)
    ns = np.asarray(scattered_direction, dtype=float)
    if ni.shape != (3,) or ns.shape != (3,) or np.any(~np.isfinite(ni)) or np.any(~np.isfinite(ns)):
        raise ValueError("incident_direction and scattered_direction must be finite 3-vectors")
    if np.linalg.norm(ni) == 0 or np.linalg.norm(ns) == 0:
        raise ValueError("incident_direction and scattered_direction must be non-zero")
    ni = ni/np.linalg.norm(ni)
    ns = ns/np.linalg.norm(ns)
    if not np.isclose(np.arccos(np.clip(np.dot(ni, ns), -1.0, 1.0)), angle, atol=1e-5, rtol=0.0):
        raise ValueError("scattering_angle_rad is inconsistent with incident/scattered directions")
    qhat = ns-ni
    qhat /= np.linalg.norm(qhat)
    ve = np.asarray(electron_velocity_m_s, dtype=float)
    vi = np.asarray(ion_velocity_m_s, dtype=float)
    if ve.shape != (3,) or np.any(~np.isfinite(ve)) or np.linalg.norm(ve) >= C:
        raise ValueError("electron_velocity_m_s must be a finite subluminal 3-vector")
    species = _resolve_ion_species(ion_species)
    fractions = np.asarray(ion_fractions, dtype=float)
    temperatures = np.asarray(np.atleast_1d(ion_temperature_eV), dtype=float)
    if (fractions.shape != (len(species),) or temperatures.shape != (len(species),)
            or np.any(~np.isfinite(fractions)) or np.any(fractions <= 0)
            or not np.isclose(np.sum(fractions), 1.0, rtol=1e-6, atol=1e-8)
            or np.any(~np.isfinite(temperatures)) or np.any(temperatures <= 0)):
        raise ValueError("ion fractions and temperatures must provide one positive value per species; fractions must sum to one")
    if vi.shape == (3,) and len(species) == 1:
        vi = vi.reshape(1, 3)
    if vi.shape != (len(species), 3) or np.any(~np.isfinite(vi)) or np.any(np.linalg.norm(vi, axis=1) >= C):
        raise ValueError("ion_velocity_m_s must contain one finite subluminal 3-vector per species")
    ue = float(np.dot(ve, qhat))
    ui = vi@qhat
    k = 4.0*np.pi*np.sin(angle/2.0)/(laser*1e-9)
    vphase = -C*(x-laser)/(2.0*laser*np.sin(angle/2.0))
    sigma_e = np.sqrt(te*E_CHARGE/ME)
    omega_pe2 = ne*E_CHARGE**2/(EPS0*ME)
    electron_pdf, _, _ = _kappa_pdf_and_derivatives(vphase-ue, sigma_e, kap)
    chi_e = np.empty(x.size, dtype=complex)
    ion_chi = []
    ion_pdf = []
    for index, item in enumerate(species):
        density = fractions[index]*ne/item["charge_state"]
        sigma_i = np.sqrt(temperatures[index]*E_CHARGE/item["mass_kg"])
        omega_pi2 = density*(item["charge_state"]*E_CHARGE)**2/(EPS0*item["mass_kg"])
        ion_pdf_i, _, _ = _maxwellian_pdf_and_derivatives(vphase-ui[index], sigma_i)
        ion_pdf.append(ion_pdf_i)
        chi_i = np.empty(x.size, dtype=complex)
        for j in range(x.size):
            chi_i[j] = _landau_susceptibility(
                vphase[j]-ui[index], omega_pi2, k, sigma_i
            )
        ion_chi.append(chi_i)
    for j in range(x.size):
        chi_e[j] = _landau_susceptibility(
            vphase[j]-ue, omega_pe2, k, sigma_e, distribution="kappa", kappa=kap
        )
    ion_sum = np.sum(np.asarray(ion_chi), axis=0)
    dielectric = 1.0+chi_e+ion_sum
    denominator = np.maximum(np.abs(dielectric)**2, np.finfo(float).tiny)
    electron_term = np.abs(1.0+ion_sum)**2/denominator*electron_pdf
    ion_term = np.zeros(x.size, dtype=float)
    for index, item in enumerate(species):
        ion_term += (
            item["charge_state"]**2*fractions[index]/item["charge_state"]
            *np.abs(chi_e)**2/denominator*ion_pdf[index]
        )
    dynamic_structure = (2.0*np.pi/k)*(electron_term+ion_term)
    omega_per_nm = 2.0*np.pi*C*1e9/(x*x)
    spectral = np.maximum(dynamic_structure, 0.0)*omega_per_nm
    if np.any(~np.isfinite(spectral)) or not np.any(spectral > 0):
        raise ValueError("collective kappa dielectric model produced a non-finite or zero spectrum")
    return {
        "spectral_density_per_nm": spectral,
        "electron_susceptibility": chi_e,
        "ion_susceptibility": np.asarray(ion_chi),
        "dielectric_function": dielectric,
        "dynamic_structure_factor": dynamic_structure,
        "electron_flow_projection_m_s": ue,
        "ion_flow_projections_m_s": ui,
        "ion_species": species,
        "ion_fractions": fractions,
        "ion_temperature_eV": temperatures,
        "kappa_index": kap,
        "wavenumber_m_inverse": k,
        "phase_velocity_m_s": vphase,
        "model": "unmagnetized collective Thomson scattering with kappa electrons and Maxwellian ions",
    }


def fit_collective_kappa(
    wavelength_nm, signal, laser_wavelength_nm, scattering_angle_rad,
    response, uncertainty=None, *, incident_direction, scattered_direction,
    ion_species, ion_fractions, ion_temperature_eV,
    electron_velocity_m_s, ion_velocity_m_s,
    initial_density_m3, initial_temperature_eV, initial_kappa,
    density_bounds_m3=None, temperature_bounds_eV=None, kappa_bounds=None,
):
    """Fit ne, Te and kappa from a collective spectrum; ion inputs remain fixed."""
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(signal, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 12:
        raise ValueError("wavelength_nm and signal must be matching vectors with >=12 samples")
    scale = max(float(np.max(np.abs(y))), np.finfo(float).tiny)
    ys = y/scale
    unc = None if uncertainty is None else np.asarray(uncertainty, dtype=float)/scale
    if unc is not None and (unc.shape != y.shape or np.any(~np.isfinite(unc)) or np.any(unc <= 0)):
        raise ValueError("uncertainty must be positive and match signal")
    residual_scale = np.full_like(ys, max(float(np.max(np.abs(ys))), np.finfo(float).tiny)) if unc is None else unc
    ne0 = _positive(initial_density_m3, "initial_density_m3")
    te0 = _positive(initial_temperature_eV, "initial_temperature_eV")
    kap0 = _positive(initial_kappa-1.5, "initial_kappa_minus_1p5")+1.5
    lower = np.full(3, -np.inf, dtype=float)
    upper = np.full(3, np.inf, dtype=float)
    for bounds, name, index, transform in (
        (density_bounds_m3, "density_bounds_m3", 0, np.log10),
        (temperature_bounds_eV, "temperature_bounds_eV", 1, np.log10),
        (kappa_bounds, "kappa_bounds", 2, lambda value: np.log(value-1.5)),
    ):
        if bounds is None:
            continue
        pair = np.asarray(bounds, dtype=float)
        if pair.shape != (2,) or np.any(~np.isfinite(pair)) or pair[0] >= pair[1]:
            raise ValueError(f"{name} must be a finite increasing pair")
        if index == 2 and pair[0] <= 1.5:
            raise ValueError("kappa_bounds lower bound must be >1.5")
        if index in (0,1) and pair[0] <= 0:
            raise ValueError(f"{name} must be positive")
        if index == 0 and not pair[0] <= ne0 <= pair[1]:
            raise ValueError("initial_density_m3 must lie within density_bounds_m3")
        if index == 1 and not pair[0] <= te0 <= pair[1]:
            raise ValueError("initial_temperature_eV must lie within temperature_bounds_eV")
        if index == 2 and not pair[0] <= kap0 <= pair[1]:
            raise ValueError("initial_kappa must lie within kappa_bounds")
        lower[index], upper[index] = transform(pair[0]), transform(pair[1])
    edge = max(2, x.size//10)
    baseline = float(np.median(np.r_[ys[:edge], ys[-edge:]]))
    amp0 = max(float(np.max(ys)-baseline), np.finfo(float).tiny)
    p0 = np.array([np.log10(ne0), np.log10(te0), np.log(kap0-1.5)])

    def unpack(p):
        return (
            10.0**np.clip(p[0], -300.0, 300.0),
            10.0**np.clip(p[1], -300.0, 300.0),
            1.5+np.exp(np.clip(p[2], -700.0, 700.0)),
        )

    def spectrum(p):
        ne, te, kap = unpack(p)
        physical = collective_kappa_spectral_density(
            x, laser_wavelength_nm, scattering_angle_rad,
            electron_density_m3=ne, electron_temperature_eV=te, kappa=kap,
            ion_species=ion_species, ion_fractions=ion_fractions,
            ion_temperature_eV=ion_temperature_eV,
            electron_velocity_m_s=electron_velocity_m_s,
            ion_velocity_m_s=ion_velocity_m_s,
            incident_direction=incident_direction, scattered_direction=scattered_direction,
        )["spectral_density_per_nm"]
        return _apply_response(response, physical, x)

    def nuisance(shape):
        shape_scale=max(float(np.max(np.abs(shape))),np.finfo(float).tiny)
        design=np.column_stack((shape/shape_scale,np.ones_like(shape),x-float(laser_wavelength_nm)))
        coefficients=np.linalg.lstsq(design/residual_scale[:,None],ys/residual_scale,rcond=None)[0]
        if coefficients[0] < 0:
            bg=np.linalg.lstsq(design[:,1:]/residual_scale[:,None],ys/residual_scale,rcond=None)[0]
            coefficients=np.array([0.0,bg[0],bg[1]])
        coefficients[0]/=shape_scale
        return coefficients

    def residual(p, return_fit=False):
        shape=spectrum(p)
        amp,offset,slope=nuisance(shape)
        predicted=amp*shape+offset+slope*(x-float(laser_wavelength_nm))
        value=(predicted-ys)/residual_scale
        return (value, predicted, shape, amp, offset, slope) if return_fit else value

    # `lower` and `upper` were already assembled above in the optimizer's
    # mixed transformed coordinates: log10(ne), log10(Te), and log(kappa-1.5).
    # Rebuilding them here from undefined aliases both raised NameError and
    # discarded the optional/unbounded parameter contract.
    fit=least_squares(residual,p0,bounds=(lower,upper),x_scale="jac",max_nfev=3000)
    if not fit.success:
        raise ValueError(f"collective kappa Thomson fit failed: {fit.message}")
    rank=int(np.linalg.matrix_rank(fit.jac))
    condition=float(np.linalg.cond(fit.jac))
    if rank<3 or not np.isfinite(condition) or condition>1e12:
        raise ValueError("collective kappa Thomson density, temperature and kappa are not jointly identifiable")
    r,predicted,shape,amp,offset,slope=residual(fit.x,return_fit=True)
    ne,te,kap=unpack(fit.x)
    dof=max(1,x.size-6)
    covariance_log=np.linalg.pinv(fit.jac.T@fit.jac)
    if unc is None:
        covariance_log*=float(r@r/dof)
    transform=np.diag([np.log(10.0)*ne,np.log(10.0)*te,kap-1.5])
    covariance=transform@covariance_log@transform.T
    sigma=np.sqrt(np.maximum(np.diag(covariance),0.0))
    incident_unit=np.asarray(incident_direction,dtype=float)
    scattered_unit=np.asarray(scattered_direction,dtype=float)
    incident_unit=incident_unit/np.linalg.norm(incident_unit)
    scattered_unit=scattered_unit/np.linalg.norm(scattered_unit)
    qhat=scattered_unit-incident_unit
    qhat=qhat/np.linalg.norm(qhat)
    ue=float(np.dot(np.asarray(electron_velocity_m_s,dtype=float),qhat))
    ion_properties=_resolve_ion_species(ion_species)
    ion_v=np.asarray(ion_velocity_m_s,dtype=float)
    if ion_v.shape==(3,) and len(ion_properties)==1:
        ion_v=ion_v.reshape(1,3)
    ui=ion_v@qhat
    k=4.0*np.pi*np.sin(float(scattering_angle_rad)/2.0)/(float(laser_wavelength_nm)*1e-9)
    debye=np.sqrt(EPS0*te/(ne*E_CHARGE))
    alpha=1.0/(k*debye)
    fitted=predicted*scale
    residual_signal=y-fitted
    return {
        "electron_density_m3":float(ne),
        "electron_density_uncertainty_m3":float(sigma[0]),
        "electron_temperature_eV":float(te),
        "electron_temperature_uncertainty_eV":float(sigma[1]),
        "kappa_index":float(kap),
        "kappa_index_uncertainty":float(sigma[2]),
        "electron_velocity_projection_m_s":ue,
        "electron_flow_source":"explicitly supplied electron velocity vector",
        "ion_velocity_projection_m_s":ui.tolist(),
        "ion_species":[item["name"] for item in ion_properties],
        "ion_fractions":np.asarray(ion_fractions,dtype=float).tolist(),
        "ion_temperature_eV":np.asarray(np.atleast_1d(ion_temperature_eV),dtype=float).tolist(),
        "ion_velocity_m_s":ion_v.tolist(),
        "electron_velocity_m_s":np.asarray(electron_velocity_m_s,dtype=float).tolist(),
        "fitted_spectrum":fitted,
        "residual":residual_signal,
        "amplitude":float(amp*scale),
        "background_intercept":float(offset*scale),
        "background_slope_per_nm":float(slope*scale),
        "fit_covariance":covariance,
        "fit_parameters":np.array([ne,te,kap],dtype=float),
        "fit_parameter_names":["electron_density_m3","electron_temperature_eV","kappa_index"],
        "optimizer_success":bool(fit.success),
        "optimizer_message":str(fit.message),
        "collective_parameter_jacobian_rank":rank,
        "collective_parameter_jacobian_condition_number":condition,
        "reduced_chi_square":float(np.sum((residual_signal/np.asarray(uncertainty,dtype=float))**2)/dof) if uncertainty is not None else None,
        "scattering_parameter_alpha":float(alpha),
        "scattering_regime":"collective" if alpha>1.0 else "non_collective_or_weakly_collective",
        "model":"Plasma kinetic dielectric model with kappa electron VDF and Maxwellian ion species",
        "distribution_model":"1-D projected kappa electron distribution; Maxwellian ions",
    }


def fit_relativistic_maxwell_juttner_channels(
    wavelength_nm, measured_spectra, laser_wavelength_nm, scattering_angle_rad,
    response, *, incident_direction, scattered_direction, incident_polarization,
    joint_spectral_covariance, initial_temperature_eV,
    initial_flow_projection_m_s=0.0, radial_order=36, angular_order=18,
    maximum_recoil_correction_fraction=0.01, max_nfev=1600,
):
    """Joint relativistic MJ fit to multiple channels with full cross-channel GLS.

    The data and covariance are flattened in channel-major order:
    vec(Y) = [Y[0,:], Y[1,:], ...]. A single Te and q-parallel bulk flow are
    shared by all channels; each channel has its own positive amplitude,
    constant background and linear background slope. The complete covariance
    (including off-diagonal cross-channel blocks) is Cholesky-whitened.

    This is a common-plasma-parameter multi-channel fit, not a model for
    channel-dependent electron distributions or channel-dependent geometry.
    All channels must share wavelength sampling, scattering geometry and
    calibrated response. Density is not inferred from normalized spectral
    shapes; absolute density requires independently calibrated collection
    efficiency, throughput, solid angle and laser energy.
    """
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(measured_spectra, dtype=float)
    if x.ndim != 1 or x.size < 12 or np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm must be finite, strictly increasing, with >=12 samples")
    if y.ndim != 2 or y.shape[1] != x.size or y.shape[0] < 2 or np.any(~np.isfinite(y)):
        raise ValueError("measured_spectra must be a finite (n_channels,n_wavelengths) array with >=2 channels")
    m, n = y.shape
    cov = np.asarray(joint_spectral_covariance, dtype=float)
    if cov.shape != (m*n, m*n) or np.any(~np.isfinite(cov)):
        raise ValueError("joint_spectral_covariance must be finite and shaped (n_channels*n_wavelengths)^2 in channel-major order")
    scale_cov = max(float(np.max(np.abs(cov))), np.finfo(float).tiny)
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12*scale_cov):
        raise ValueError("joint_spectral_covariance must be symmetric")
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("joint_spectral_covariance must be positive definite") from exc
    ni = np.asarray(incident_direction, dtype=float)
    ns = np.asarray(scattered_direction, dtype=float)
    pol = np.asarray(incident_polarization, dtype=float)
    if ni.shape != (3,) or ns.shape != (3,) or pol.shape != (3,) or any(np.any(~np.isfinite(v)) for v in (ni, ns, pol)):
        raise ValueError("incident_direction, scattered_direction and incident_polarization must be finite 3-vectors")
    if min(np.linalg.norm(ni), np.linalg.norm(ns), np.linalg.norm(pol)) <= 0:
        raise ValueError("direction and polarization vectors must be non-zero")
    ni = ni/np.linalg.norm(ni)
    ns = ns/np.linalg.norm(ns)
    pol = pol/np.linalg.norm(pol)
    if abs(float(np.dot(pol, ni))) > 1e-6:
        raise ValueError("incident_polarization must be transverse to incident_direction")
    angle = float(np.arccos(np.clip(np.dot(ni, ns), -1.0, 1.0)))
    if not np.isclose(angle, scattering_angle_rad, rtol=0.0, atol=1e-5):
        raise ValueError("scattering_angle_rad is inconsistent with direction vectors")
    qhat = ns-ni
    qhat /= np.linalg.norm(qhat)
    e3 = np.cross(ni, ns)
    if np.linalg.norm(e3) <= 1e-12:
        e3 = pol-np.dot(pol, qhat)*qhat
    e3 /= np.linalg.norm(e3)
    e2 = np.cross(qhat, e3)
    e2 /= np.linalg.norm(e2)
    local_to_lab = np.column_stack((qhat, e2, e3))
    pol_local = local_to_lab.T @ pol
    laser = _positive(laser_wavelength_nm, "laser_wavelength_nm")
    te0 = _positive(initial_temperature_eV, "initial_temperature_eV")
    flow0 = float(initial_flow_projection_m_s)
    if not np.isfinite(flow0) or abs(flow0) >= C:
        raise ValueError("initial_flow_projection_m_s must be finite and subluminal")
    if isinstance(max_nfev, (bool, np.bool_)) or not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    if isinstance(response, (list, tuple)) and len(response) == m:
        responses = [np.asarray(item, dtype=float) for item in response]
    else:
        responses = [np.asarray(response, dtype=float)]*m
    for item in responses:
        if item.shape != (n, n) or np.any(~np.isfinite(item)) or np.any(item < 0):
            raise ValueError("response must be a finite non-negative (n_lambda,n_lambda) matrix per channel")
    yscale = max(float(np.max(np.abs(y))), np.finfo(float).tiny)
    ys = y/yscale
    chol_scaled = chol/yscale
    baseline = np.median(np.concatenate((ys[:, :max(2,n//10)], ys[:, -max(2,n//10):]), axis=1), axis=1)
    amplitudes = np.maximum(np.max(ys, axis=1)-baseline, np.finfo(float).tiny)
    p0 = np.r_[np.log(te0), np.arctanh(flow0/C), np.log(amplitudes), baseline, np.zeros(m)]
    def predict(p):
        te = float(np.exp(np.clip(p[0], -700, 700)))
        flow = float(np.tanh(p[1])*C)
        physical = relativistic_maxwell_juttner_spectrum(
            x, laser, angle, te, flow_projection_m_s=flow,
            incident_polarization=pol_local, radial_order=radial_order,
            angular_order=angular_order,
            maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
            require_full_spectrum=False,
        )
        shape = physical["spectral_density_per_nm"]
        out = np.empty_like(ys)
        amps = np.exp(np.clip(p[2:2+m], -700, 700))
        offsets = p[2+m:2+2*m]
        slopes = p[2+2*m:2+3*m]
        for j in range(m):
            convolved = _apply_response(responses[j], shape, x)
            out[j] = amps[j]*convolved + offsets[j] + slopes[j]*(x-laser)
        return out
    def objective(p):
        try:
            residual = (ys-predict(p)).reshape(-1)
            return np.linalg.solve(chol_scaled, residual)
        except ValueError as exc:
            if "capture at least 99.5%" in str(exc) or "zero spectral power" in str(exc):
                return np.full(m*n, 1e6)
            raise
    bounds = (np.r_[-np.inf, -18.0, np.full(3*m, -np.inf)],
              np.r_[np.inf, 18.0, np.full(3*m, np.inf)])
    fit = least_squares(objective, p0, bounds=bounds, x_scale="jac", max_nfev=int(max_nfev))
    if not fit.success:
        raise ValueError(f"joint relativistic Maxwell-Juttner fit failed: {fit.message}")
    jac = np.asarray(fit.jac, dtype=float)
    norms = np.linalg.norm(jac, axis=0)
    scaled_jac = np.divide(jac, norms[None, :], out=np.zeros_like(jac), where=norms[None, :] > np.finfo(float).tiny)
    singular = np.linalg.svd(scaled_jac, compute_uv=False)
    rank_tol = np.finfo(float).eps*max(scaled_jac.shape)*(singular[0] if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > rank_tol))
    condition = float(singular[0]/singular[-1]) if singular.size and singular[-1] > 0 else float("inf")
    if rank < p0.size or not np.isfinite(condition) or condition > 1e12:
        raise ValueError(f"joint relativistic Maxwell-Juttner fit is not identifiable (rank {rank}/{p0.size}, condition={condition:.6g})")
    covariance_p = np.linalg.pinv(jac.T@jac, rcond=1e-12)
    transform = np.eye(p0.size)
    transform[0, 0] = np.exp(fit.x[0])
    transform[1, 1] = C*(1.0-np.tanh(fit.x[1])**2)
    covariance_physical = transform@covariance_p@transform.T
    fitted = predict(fit.x)*yscale
    residual = y-fitted
    return {
        "electron_temperature_eV": float(np.exp(fit.x[0])),
        "electron_temperature_uncertainty_eV": float(np.sqrt(max(covariance_physical[0,0],0.0))),
        "electron_velocity_projection_m_s": float(np.tanh(fit.x[1])*C),
        "electron_velocity_projection_uncertainty_m_s": float(np.sqrt(max(covariance_physical[1,1],0.0))),
        "channel_amplitudes": np.exp(fit.x[2:2+m])*yscale,
        "channel_background_intercepts": fit.x[2+m:2+2*m]*yscale,
        "channel_background_slopes_per_nm": fit.x[2+2*m:2+3*m]*yscale,
        "fitted_spectra": fitted,
        "residuals": residual,
        "weighted_residual_sum_squares": float(2.0*fit.cost),
        "fit_covariance": covariance_physical,
        "fit_parameter_names": ["electron_temperature_eV","electron_velocity_projection_m_s"]+[f"channel_{j}_amplitude" for j in range(m)]+[f"channel_{j}_background_intercept" for j in range(m)]+[f"channel_{j}_background_slope_per_nm" for j in range(m)],
        "jacobian_rank": rank,
        "jacobian_condition_number": condition,
        "spectral_weighting_status": "full_channel_major_generalized_least_squares_including_cross_channel_covariance",
        "covariance_ordering": "channel_major",
        "model": "relativistic non-collective Thomson scattering; shared Maxwell-Juttner Te and q-parallel flow across channels",
        "density_inference": "not_inferred_from_normalized_spectral_shape",
        "assumptions": ["All channels share Te, q-parallel bulk flow, wavelength grid and scattering geometry.", "Channel amplitudes and affine backgrounds are independent nuisance parameters.", "The supplied full covariance is known, positive definite and includes cross-channel correlations.", "A common measured response is used unless a per-channel response list is supplied.", "Photon recoil is neglected only within the configured Thomson-limit correction threshold."]
    }
