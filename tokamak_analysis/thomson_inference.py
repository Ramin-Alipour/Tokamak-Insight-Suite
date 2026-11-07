"""End-to-end, spatially resolved Thomson spectral inference.

Input spectra are measured, wavelength-calibrated and intensity-calibrated data.
This module does not calibrate detectors or repair/prepare measurements.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq, curve_fit, least_squares
from scipy.special import gammaln, ndtr, stdtr

from .thomson_scattering import density_from_thomson_scattered_power, _covariance_is_positive_semidefinite
from .formula_library import (
    thomson_electron_temperature_from_spectral_width,
    thomson_velocity_projection,
)
from .thomson_kinetic_models import (
    fit_relativistic_maxwell_juttner,
    fit_collective_kappa,
)

ELECTRON_REST_ENERGY_EV = 510_998.95069
C_M_S = 299_792_458.0
# Conservative asymptotic screening policy: theta_e bounds leading thermal
# relativistic corrections (~theta_e), while beta_flow bounds leading bulk
# kinematic corrections (~beta_flow^2). These are software screening limits,
# not universal physical validity boundaries.
NONRELATIVE_THETA_E_SCREEN = 0.01
NONRELATIVE_FLOW_BETA_SCREEN = 0.1
R_E_M = 2.8179403262e-15


def _finite_array(value, name, ndim=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values, not booleans, strings or complex values")
    if ndim is not None and raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    a = raw.astype(float, copy=False)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain finite numeric values")
    return a


def _finite_scalar(value, name, *, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be less than or equal to {maximum}")
    return result


def _collective_spectral_whitener(signal, uncertainty=None, covariance=None):
    """Build a statistically correct residual whitener for collective Thomson fits.

    For covariance C=L L^T, residuals are transformed as L^-1 r and the
    profiled linear nuisance parameters are solved in the same GLS metric.
    Pointwise standard deviations remain the diagonal-noise special case.
    """
    values = _finite_array(signal, "collective measured spectrum", 1)
    if uncertainty is not None and covariance is not None:
        raise ValueError("collective fit accepts uncertainty or covariance, not both")
    if covariance is not None:
        matrix = _finite_array(covariance, "collective spectral covariance", 2)
        if matrix.shape != (values.size, values.size):
            raise ValueError("collective spectral covariance must match the wavelength grid")
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-13):
            raise ValueError("collective spectral covariance must be symmetric")
        try:
            factor = np.linalg.cholesky(0.5 * (matrix + matrix.T))
        except np.linalg.LinAlgError as exc:
            raise ValueError("collective spectral covariance must be positive definite") from exc
        return lambda residual: np.linalg.solve(factor, np.asarray(residual, dtype=float)), "full_covariance"
    if uncertainty is not None:
        sigma = _finite_array(uncertainty, "collective signal uncertainty", 1)
        if sigma.shape != values.shape or np.any(sigma <= 0.0):
            raise ValueError("collective signal uncertainty must be positive and match the wavelength grid")
        return lambda residual: np.asarray(residual, dtype=float) / sigma, "pointwise_uncertainty"
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    return lambda residual: np.asarray(residual, dtype=float) / scale, "relative_unweighted"


def _response_for_channel(response, channel, nchan, nlam, wavelength=None):
    if callable(response):
        if wavelength is None:
            raise ValueError("wavelength_nm is required when instrument_response is callable")
        response = response(np.asarray(wavelength,dtype=float))
    r = _finite_array(response, "instrument_response")
    kernel_input = r.ndim == 1
    if kernel_input:
        if r.size < 1 or r.size % 2 == 0 or np.any(r < 0) or np.sum(r) <= 0:
            raise ValueError("a 1-D instrument-response kernel must be non-negative, odd-length and have positive area")
        if not np.isclose(float(np.sum(r)), 1.0, rtol=1e-8, atol=1e-10):
            raise ValueError(
                "a measured 1-D instrument-response kernel must already have unit total area; "
                "the calibrated kernel is not renormalized"
            )
        if wavelength is not None:
            wavelength_steps = np.diff(np.asarray(wavelength, dtype=float))
            if not np.allclose(wavelength_steps, wavelength_steps[0], rtol=1e-6, atol=1e-12):
                raise ValueError("a 1-D index-space response kernel requires a uniform wavelength grid; use a measured wavelength-to-wavelength response matrix for nonuniform grids")
        # Preserve the supplied calibrated kernel. At the finite wavelength
        # window, off-window response tails are not folded back into measured
        # bins: edge columns are therefore sub-stochastic by design.
        kernel = r.copy()
        out = np.zeros((nlam,nlam),dtype=float)
        center = kernel.size//2
        for col in range(nlam):
            for k,value in enumerate(kernel):
                row=col+(k-center)
                if 0 <= row < nlam:
                    out[row,col]+=value
    elif r.ndim == 2:
        out = r
    elif r.ndim == 3 and r.shape[0] == nchan:
        out = r[channel]
    else:
        raise ValueError("instrument_response must be a callable, a 1-D kernel, a 2-D response matrix, or a channel-specific 3-D response matrix")
    if out.shape != (nlam,nlam) or np.any(out < 0):
        raise ValueError("instrument response must be a non-negative square matrix on the measured wavelength grid")
    sums = out.sum(axis=0)
    if np.any(sums <= 0):
        raise ValueError("each instrument-response column must have positive normalization")
    if kernel_input:
        if np.any(sums > 1.0 + 1e-10):
            raise ValueError("finite-window 1-D response columns cannot exceed unit integrated response")
    elif not np.allclose(sums, 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError(
            "measured instrument-response matrix columns must already sum to one; "
            "the calibrated response operator is not renormalized"
        )
    return out


def _apply_response(response, spectrum, wavelength):
    """Apply a bin-probability response to spectral-density samples.

    The response columns describe probabilities from true wavelength bins to
    measured bins. Convert spectral density to bin-integrated power before the
    matrix operation, then divide by measured-bin widths. This reduces to the
    usual matrix product for a uniform wavelength grid.
    """
    x = np.asarray(wavelength, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0) or values.shape != x.shape:
        raise ValueError("Wavelength-bin widths and spectrum shape must be valid")
    return (response @ (values * widths)) / widths

def _wavelength_bin_edges(wavelength):
    """Midpoint finite-volume edges for strictly increasing wavelength centres."""
    x = _finite_array(wavelength, "wavelength_nm", 1)
    if x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("wavelength_nm must contain at least two strictly increasing bin centres")
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges


def _gaussian_bin_averaged_shape(wavelength, amplitude, center, sigma):
    """Exact bin-average of amplitude*exp[-(lambda-center)^2/(2 sigma^2)]."""
    edges = _wavelength_bin_edges(wavelength)
    sigma = _finite_scalar(sigma, "sigma", minimum=np.finfo(float).tiny)
    amplitude = _finite_scalar(amplitude, "amplitude", minimum=0.0)
    center = _finite_scalar(center, "center")
    z = (edges - center) / sigma
    probability = ndtr(z[1:]) - ndtr(z[:-1])
    positive_tail = z[:-1] > 0.0
    probability[positive_tail] = ndtr(-z[:-1][positive_tail]) - ndtr(-z[1:][positive_tail])
    return amplitude * np.sqrt(2.0*np.pi) * sigma * probability / np.diff(edges), probability


def _kappa_bin_averaged_shape(wavelength, amplitude, center, sigma, kappa):
    """Exact bin-average of the peak-amplitude Student-t/Kappa spectral shape."""
    edges = _wavelength_bin_edges(wavelength)
    amplitude = _finite_scalar(amplitude, "amplitude", minimum=0.0)
    center = _finite_scalar(center, "center")
    sigma = _finite_scalar(sigma, "sigma", minimum=np.finfo(float).tiny)
    kappa = _finite_scalar(kappa, "kappa", minimum=1.500000000001)
    degrees = 2.0*kappa - 1.0
    scale = sigma*np.sqrt((degrees-2.0)/degrees)
    z = (edges-center)/scale
    probability = stdtr(degrees, z[1:]) - stdtr(degrees, z[:-1])
    positive_tail = z[:-1] > 0.0
    probability[positive_tail] = (
        stdtr(degrees, -z[:-1][positive_tail])
        - stdtr(degrees, -z[1:][positive_tail])
    )
    # Integral of (1 + z²/nu)^(-kappa) over wavelength.
    shape_integral = (
        scale*np.sqrt(degrees*np.pi)
        * np.exp(gammaln(kappa-0.5)-gammaln(kappa))
    )
    return amplitude*shape_integral*probability/np.diff(edges), probability




def _laser_source_response(wavelength_nm, laser_wavelength_nm,
                           laser_spectrum_wavelength_nm,
                           laser_spectrum_density_per_nm):
    """Build a photon-conserving wavelength-shift operator for a measured laser line.

    The supplied laser profile is a measured, calibrated spectral density. Its
    absolute power is deliberately irrelevant here: it is normalized to unit
    area and used only as the source spectral shape. For elastic Thomson
    scattering at fixed electron velocity, the scattered wavelength scales
    with the incident wavelength; the operator maps nominal scattered
    wavelengths by the incident-wavelength ratio before the measured response.
    """
    x = _finite_array(wavelength_nm, "wavelength_nm", 1)
    source_x = _finite_array(laser_spectrum_wavelength_nm,
                             "laser_spectrum_wavelength_nm", 1)
    source_y = _finite_array(laser_spectrum_density_per_nm,
                             "laser_spectrum_density_per_nm", 1)
    if source_x.size < 3 or source_y.shape != source_x.shape:
        raise ValueError("The measured laser spectral profile must contain at least three paired wavelength-density samples")
    if np.any(np.diff(source_x) <= 0) or np.any(source_x <= 0):
        raise ValueError("laser_spectrum_wavelength_nm must be positive and strictly increasing")
    if np.any(source_y < 0):
        raise ValueError("laser_spectrum_density_per_nm must be non-negative")
    area = float(trapezoid(source_y, source_x))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("The measured laser spectral profile must have positive finite integrated area")
    normalized = source_y / area
    # Trapezoidal quadrature weights preserve the supplied continuous spectral
    # density on its native grid without assuming uniform laser sampling.
    dx = np.diff(source_x)
    quadrature = np.empty(source_x.size, dtype=float)
    quadrature[0] = dx[0] / 2.0
    quadrature[-1] = dx[-1] / 2.0
    if source_x.size > 2:
        quadrature[1:-1] = (dx[:-1] + dx[1:]) / 2.0
    weights = normalized * quadrature
    weights /= np.sum(weights)
    operator = np.zeros((x.size, x.size), dtype=float)
    source_wavelength_ratio = source_x / float(laser_wavelength_nm)
    for column, true_wavelength in enumerate(x):
        shifted = true_wavelength * source_wavelength_ratio
        for position, weight in zip(shifted, weights):
            if position < x[0] or position > x[-1]:
                continue
            hi = int(np.searchsorted(x, position, side="right"))
            if hi == 0:
                operator[0, column] += weight
            elif hi >= x.size:
                operator[-1, column] += weight
            else:
                lo = hi - 1
                fraction = (position - x[lo]) / (x[hi] - x[lo])
                operator[lo, column] += weight * (1.0 - fraction)
                operator[hi, column] += weight * fraction
    # Do not renormalize edge columns: missing probability represents genuine
    # source-broadened spectral power outside the finite measured wavelength
    # window, not an instrumental efficiency correction.
    return operator, {
        "normalization_area": area,
        "centroid_nm": float(trapezoid(source_x * normalized, source_x)),
        "rms_width_nm": float(np.sqrt(max(trapezoid(
            (source_x - trapezoid(source_x * normalized, source_x))**2
            * normalized, source_x), 0.0))),
        "source_profile_samples": int(source_x.size),
        "source_operator_column_sums_min": float(np.min(operator.sum(axis=0))),
        "source_operator_column_sums_max": float(np.max(operator.sum(axis=0))),
        "convolution_approximation": "incident_wavelength_ratio_scaling_for_elastic_Thomson_kinematics",
    }

def _gaussian(x, amp, center, sigma, offset, slope):
    return amp * np.exp(-0.5 * ((x-center)/sigma)**2) + offset + slope*(x-center)



def _fit_identifiability(
    covariance, residual, uncertainty, parameter_count, *,
    optimizer_success=True, jacobian_rank=None, jacobian_condition_number=None,
):
    """Reject failed optimizers and non-identifiable fits, not merely malformed arrays."""
    cov = np.asarray(covariance, dtype=float)
    residual = np.asarray(residual, dtype=float)
    reasons = []
    if not optimizer_success:
        reasons.append("nonlinear optimizer did not report successful convergence")
    if cov.shape != (parameter_count, parameter_count) or not np.all(np.isfinite(cov)):
        reasons.append("covariance is missing, non-finite or has an invalid shape")
        condition = None
    else:
        diagonal = np.diag(cov)
        if np.any(diagonal <= 0):
            reasons.append("one or more fitted parameter variances are non-positive")
        condition = None
        if np.all(diagonal > 0):
            scale = np.sqrt(diagonal)
            correlation = cov / np.outer(scale, scale)
            condition = float(np.linalg.cond(correlation))
            if np.linalg.matrix_rank(correlation) < parameter_count:
                reasons.append("parameter covariance correlation matrix is rank deficient")
            if not np.isfinite(condition) or condition > 1e14:
                reasons.append("parameter covariance correlation matrix is ill-conditioned")
    if not np.all(np.isfinite(residual)):
        reasons.append("residual contains non-finite values")
    if jacobian_rank is not None and int(jacobian_rank) < parameter_count:
        reasons.append("fit Jacobian is rank deficient; fitted parameters are not jointly identifiable")
    if jacobian_condition_number is not None and (
        not np.isfinite(jacobian_condition_number) or float(jacobian_condition_number) > 1e12
    ):
        reasons.append("fit Jacobian is ill-conditioned")
    dof = max(1, residual.size - parameter_count)
    chi2 = None
    if uncertainty is not None and np.all(np.isfinite(residual)):
        u = np.asarray(uncertainty, dtype=float)
        if u.shape != residual.shape or np.any(~np.isfinite(u)) or np.any(u <= 0):
            reasons.append("fit uncertainty is invalid or does not match the residual")
        else:
            chi2 = float(np.sum((residual / u) ** 2) / dof)
            if not np.isfinite(chi2):
                reasons.append("reduced chi-square is non-finite")
            # Reduced chi-square is retained as a model-adequacy diagnostic and
            # contributes to the separately reported discrepancy covariance.
            # It is not used as a hard acceptance threshold without a user- or
            # reference-defined noise/acceptance policy.
    return {
        "fit_accepted": not reasons,
        "fit_quality_status": "accepted_with_identifiability_diagnostics" if not reasons else "rejected_unidentifiable_or_invalid",
        "fit_quality_reasons": reasons,
        "covariance_correlation_condition_number": condition,
        "fit_residual_rms": float(np.sqrt(np.mean(residual ** 2))) if residual.size else None,
        "fit_reduced_chi_square": chi2,
        "fit_reduced_chi_square_policy": "diagnostic_only_no_hard_acceptance_cutoff_without_an_explicit_noise_and_significance_policy",
        "fit_degrees_of_freedom": int(dof),
    }


def _channel_covariance(value, channel, n_channels, name):
    if value is None:
        return None
    raw = np.asarray(value, dtype=float)
    if raw.shape == (6, 6):
        return raw
    if raw.shape == (n_channels, 6, 6):
        return raw[channel]
    raise ValueError(f"{name} must be a shared 6x6 covariance matrix or one 6x6 matrix per channel")


def _channel_value(value, channel, n_channels, name, *, vector_length=None):
    """Accept a scalar shared by channels or a channel-specific value."""
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.ndim == 0:
        return value
    if vector_length is not None:
        if raw.shape == (vector_length,):
            return value
        if raw.shape == (n_channels, vector_length):
            return raw[channel]
    if raw.shape == (n_channels,):
        return raw[channel]
    raise ValueError(f"{name} must be a scalar shared across channels or have one value per channel")


def _collective_uncertainty_inputs_missing(
    ion_temperature_uncertainty,
    ion_velocity_uncertainty,
    electron_velocity_uncertainty,
    scattering_angle_uncertainty,
    incident_direction_uncertainty,
    scattered_direction_uncertainty,
):
    """Return genuinely missing external-uncertainty inputs for collective fits."""
    missing = []
    if ion_temperature_uncertainty is None:
        missing.append("ion_temperature_uncertainty_eV")
    if ion_velocity_uncertainty is None:
        missing.append("ion_velocity_uncertainty_m_s")
    if electron_velocity_uncertainty is None:
        missing.append("electron_velocity_uncertainty_m_s")
    if (
        scattering_angle_uncertainty is None
        and incident_direction_uncertainty is None
        and scattered_direction_uncertainty is None
    ):
        missing.append("scattering_geometry_uncertainty")
    return missing



def _fit_noncollective(
    wavelength, signal, laser_nm, angle, response, uncertainty, *,
    background_model="linear", background_degree=2,
    background_interior_knots=3, background_templates=None,
    spectral_covariance=None,
):
    """Fit Maxwellian Thomson spectra with exact finite-bin Gaussian probabilities.

    The normal-CDF bin averages are response-convolved in bin-integrated
    spectral-power space; the background remains a detector-space nuisance.
    """
    from .thomson_background import _background_design

    signal_scale = float(np.max(np.abs(signal)))
    if not np.isfinite(signal_scale) or signal_scale <= 0:
        raise ValueError("signal must have a positive finite scale")
    y = np.asarray(signal, dtype=float) / signal_scale
    sigma_y = None if uncertainty is None else np.asarray(uncertainty, dtype=float) / signal_scale
    covariance_cholesky = None
    if spectral_covariance is not None:
        covariance = _finite_array(spectral_covariance, "spectral_covariance", 2) / (signal_scale**2)
        if covariance.shape != (wavelength.size, wavelength.size) or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-13):
            raise ValueError("spectral_covariance must be symmetric and match the wavelength grid")
        if uncertainty is not None:
            raise ValueError("Supply uncertainty or spectral_covariance, not both")
        try:
            covariance_cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("spectral_covariance must be positive definite") from exc
    spacing = float(np.median(np.diff(wavelength)))
    span = float(wavelength[-1] - wavelength[0])
    basis = _background_design(
        wavelength, background_model, degree=background_degree,
        interior_knots=background_interior_knots, templates=background_templates,
    )
    n_background = basis.shape[1]
    peak = int(np.argmax(y))
    background_seed = np.linalg.lstsq(
        basis, y - max(float(y[peak]), np.finfo(float).eps)
        * np.exp(-0.5*((wavelength-wavelength[peak])/max(spacing,span/10))**2),
        rcond=None,
    )[0]
    amp0 = max(float(y[peak] - np.median(y)), np.finfo(float).eps)
    p0 = np.r_[amp0, float(wavelength[peak]), max(spacing, span/10), background_seed]
    lower = np.r_[0.0, wavelength[0], max(spacing*0.2, 1e-12), np.full(n_background,-np.inf)]
    upper = np.r_[np.inf, wavelength[-1], span, np.full(n_background,np.inf)]

    def predict(parameters):
        amp, center, width = parameters[:3]
        coefficients = parameters[3:]
        intrinsic, _ = _gaussian_bin_averaged_shape(wavelength, amp, center, width)
        return _apply_response(response, intrinsic, wavelength) + basis@coefficients

    def residual(parameters):
        difference = predict(parameters)-y
        if covariance_cholesky is not None:
            return np.linalg.solve(covariance_cholesky, difference)
        return difference if sigma_y is None else difference/sigma_y

    fit = least_squares(
        residual, p0, bounds=(lower,upper), x_scale="jac",
        max_nfev=50000, method="trf",
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"Maxwellian Thomson/background fit failed: {fit.message}")
    amp_s, center, width = map(float, fit.x[:3])
    coefficients_scaled = np.asarray(fit.x[3:], dtype=float)
    if width <= 0:
        raise ValueError("Fitted intrinsic spectral width is not positive")
    if center-3*width < wavelength[0] or center+3*width > wavelength[-1]:
        raise ValueError("Wavelength coverage must include at least ±3 fitted intrinsic widths")
    te = thomson_electron_temperature_from_spectral_width(width, laser_nm, angle)["electron_temperature_eV"]
    velocity = thomson_velocity_projection(center, laser_nm, angle)["electron_velocity_projection_m_s"]
    bin_edges = _wavelength_bin_edges(wavelength)
    captured_spectral_probability = float(
        ndtr((bin_edges[-1]-center)/width)-ndtr((bin_edges[0]-center)/width)
    )
    q_projection = 2.0*np.sin(angle/2.0)
    if te/ELECTRON_REST_ENERGY_EV >= NONRELATIVE_THETA_E_SCREEN:
        raise ValueError("Non-relativistic Maxwellian model rejected: fitted electron temperature is outside its declared validity range.")
    if abs(velocity) >= NONRELATIVE_FLOW_BETA_SCREEN*C_M_S:
        raise ValueError("Non-relativistic Maxwellian model rejected: fitted flow projection is outside its declared |v|/c < 0.1 regime.")
    fitted_scaled = predict(fit.x)
    fitted = signal_scale*fitted_scaled
    residual_physical = np.asarray(signal,dtype=float)-fitted
    dof = max(1, wavelength.size-fit.x.size)
    covariance_scaled = np.linalg.pinv(fit.jac.T@fit.jac)
    if sigma_y is None and covariance_cholesky is None:
        covariance_scaled *= float(fit.fun@fit.fun/dof)
    scales = np.r_[signal_scale,1.0,1.0,np.full(n_background,signal_scale)]
    raw_covariance = covariance_scaled*np.outer(scales,scales)
    amp = amp_s*signal_scale
    coefficients = coefficients_scaled*signal_scale
    if background_model == "linear":
        z_center = 2.0*(center-float(wavelength[0]))/span-1.0
        slope = float(coefficients[1]*2.0/span)
        background_at_center = float(coefficients[0]+coefficients[1]*z_center)
        # Convert [amp, center, sigma, c0, c1] in the midpoint-scaled
        # background basis to the established [amp, center, sigma,
        # background_at_center, slope_per_nm] public parameter convention.
        transform = np.zeros((5,5),dtype=float)
        transform[0,0] = signal_scale
        transform[1,1] = 1.0
        transform[2,2] = 1.0
        transform[3,1] = slope
        transform[3,3] = signal_scale
        transform[3,4] = signal_scale*z_center
        transform[4,4] = signal_scale*2.0/span
        covariance = transform@covariance_scaled@transform.T
        fit_parameters = np.asarray([amp,center,width,background_at_center,slope])
        fit_parameter_names = ["amplitude","center_nm","intrinsic_sigma_nm","background_intercept","background_slope_per_nm"]
        coefficient_transform = np.zeros((2,5),dtype=float)
        coefficient_transform[0,3] = signal_scale
        coefficient_transform[1,4] = signal_scale
        background_coefficient_covariance = coefficient_transform@covariance_scaled@coefficient_transform.T
    else:
        slope = None
        background_at_center = float(
            np.asarray([np.interp(center, wavelength, basis[:, j])
                        for j in range(n_background)]) @ coefficients
        )
        covariance = raw_covariance
        fit_parameters = np.r_[amp,center,width,coefficients]
        fit_parameter_names = ["amplitude","center_nm","intrinsic_sigma_nm"]+[f"background_coefficient_{i}" for i in range(n_background)]
        background_coefficient_covariance = covariance[3:,3:]
    te_unc = float(te*2*np.sqrt(max(covariance[2,2],0.0))/width)
    area = float(amp*width*np.sqrt(2.0*np.pi))
    area_gradient = np.zeros(fit_parameters.size)
    area_gradient[:3] = [width*np.sqrt(2.0*np.pi),0.0,amp*np.sqrt(2.0*np.pi)]
    area_unc = float(np.sqrt(max(float(area_gradient@covariance@area_gradient),0.0)))
    return {
        "electron_temperature_eV": float(te),
        "electron_temperature_uncertainty_eV": te_unc,
        "electron_velocity_projection_m_s": float(velocity),
        "electron_velocity_projection_uncertainty_m_s": float(C_M_S*np.sqrt(max(covariance[1,1],0.0))/(laser_nm*q_projection)),
        "fitted_center_nm": center, "intrinsic_sigma_nm": width,
        "amplitude": amp,
        "background_intercept": background_at_center,
        "background_slope_per_nm": slope,
        "background_model": background_model,
        "background_coefficients": coefficients,
        "background_basis": basis,
        "background_coefficient_covariance": background_coefficient_covariance,
        "fit_covariance": covariance, "residual": residual_physical,
        "integrated_scattered_power_W": area,
        "integrated_scattered_power_uncertainty_W": area_unc,
        "reduced_chi_square": float(fit.fun@fit.fun/dof) if (uncertainty is not None or covariance_cholesky is not None) else None,
        "likelihood": "Gaussian_or_GLS",
        "fitted_spectrum": fitted,
        "captured_spectral_probability": captured_spectral_probability,
        "spectral_bin_integration": "exact_Gaussian_CDF_finite_volume_bin_averages",
        "fit_parameters": fit_parameters,
        "fit_parameter_names": fit_parameter_names,
        "optimizer": {"success": bool(fit.success), "message": str(fit.message),
                      "nfev": int(fit.nfev), "jacobian_rank": int(np.linalg.matrix_rank(fit.jac)),
                      "parameter_count": int(fit.x.size)},
        "uncertainty_interpretation": (
            "local covariance from supplied full spectral covariance"
            if covariance_cholesky is not None else
            "local covariance from supplied pointwise measurement uncertainty"
            if uncertainty is not None else
            "residual-variance-scaled local covariance; not measurement-noise-calibrated"
        ),
        "model": "non-collective, non-relativistic Maxwellian electron spectrum",
        "validity_screening": {
            "theta_e": float(te/ELECTRON_REST_ENERGY_EV),
            "theta_e_limit": NONRELATIVE_THETA_E_SCREEN,
            "bulk_flow_beta": float(abs(velocity)/C_M_S),
            "bulk_flow_beta_limit": NONRELATIVE_FLOW_BETA_SCREEN,
            "rationale": "Conservative asymptotic error-budget screens; not universal validity boundaries.",
        },
        "assumptions": ["single-scattering non-collective regime","isotropic Maxwellian electrons",
                        "non-relativistic electron velocities","known scattering geometry",
                        "selected detector-space background basis"],
    }

def _kappa_profile_likelihood_diagnostic(
    x, y, response, parameters, sigma, spectral_covariance=None, *,
    background_model="linear", background_degree=2,
    background_interior_knots=3, background_templates=None,
):
    """Profile Kappa while re-optimizing every selected detector-background coefficient.

    The physical Kappa spectrum uses exact finite-bin Student-t probabilities;
    the selected background basis is held fixed while all its coefficients,
    amplitude, centroid and width are re-fitted at each fixed Kappa.
    """
    from .thomson_background import _background_design

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(parameters, dtype=float)
    if p.ndim != 1 or p.size < 5 or x.shape != y.shape or p.size < 4:
        raise ValueError("Kappa profile parameters must contain four physical parameters and background coefficients")
    basis = _background_design(
        x, background_model, degree=background_degree,
        interior_knots=background_interior_knots, templates=background_templates,
    )
    if p.size != 4 + basis.shape[1]:
        raise ValueError("Kappa profile parameter vector does not match the selected background basis")
    spacing = float(np.median(np.diff(x)))
    nuisance_start = np.r_[p[:3], p[4:]]
    lower = np.r_[0.0, x[0], max(spacing * 0.2, 1e-12),
                  np.full(basis.shape[1], -np.inf)]
    upper = np.r_[np.inf, x[-1], x[-1] - x[0],
                  np.full(basis.shape[1], np.inf)]
    kmin, kmax = 1.500001, max(20.0, float(p[3]) * 8.0)
    grid = np.unique(np.r_[np.geomspace(kmin, kmax, 41), p[3]])

    def forward(nuisance, kappa):
        amp, center, width = nuisance[:3]
        coefficients = nuisance[3:]
        component = _kappa_bin_averaged_shape(x, amp, center, width, kappa)[0]
        return _apply_response(response, component, x) + basis @ coefficients

    if sigma is not None and spectral_covariance is not None:
        raise ValueError("Profile likelihood accepts pointwise uncertainty or spectral_covariance, not both")
    if spectral_covariance is not None:
        covariance = np.asarray(spectral_covariance, dtype=float)
        if (covariance.shape != (y.size, y.size) or np.any(~np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-13)):
            raise ValueError("spectral_covariance must be finite, symmetric and match the wavelength grid")
        try:
            covariance_cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("spectral_covariance must be positive definite") from exc
        whiten = lambda residual: np.linalg.solve(covariance_cholesky, residual)
        noise_semantics = "researcher_supplied_full_within_channel_spectral_covariance_GLS"
    elif sigma is None:
        residual = forward(nuisance_start, p[3]) - y
        scale = max(float(np.sqrt(residual @ residual / max(1, y.size-p.size))),
                    np.finfo(float).tiny)
        whiten = lambda residual: residual / scale
        noise_semantics = "unweighted_fit_residual_variance_estimated_from_best_fit; not_measurement_noise_calibrated"
    else:
        scale = np.asarray(sigma, dtype=float)
        if scale.shape != y.shape or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("pointwise sigma must be positive, finite and match the wavelength grid")
        whiten = lambda residual: residual / scale
        noise_semantics = "researcher_supplied_pointwise_measurement_uncertainty"

    profile = np.full(grid.shape, np.inf)
    for i, kappa in enumerate(grid):
        objective = lambda z: whiten(forward(z, kappa) - y)
        try:
            fit = least_squares(
                objective, nuisance_start, bounds=(lower, upper),
                x_scale="jac", max_nfev=3000,
            )
            if fit.success and np.all(np.isfinite(fit.fun)):
                profile[i] = float(fit.fun @ fit.fun)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass
    finite = np.isfinite(profile)
    if not np.any(finite):
        return {"status": "unavailable", "reason": "All nuisance re-optimizations failed."}
    delta = profile - np.min(profile[finite])
    best = int(np.nanargmin(np.where(finite, profile, np.nan)))

    def find_crossing(indices):
        for a, b in zip(indices[:-1], indices[1:]):
            if (delta[a] - 1.0) * (delta[b] - 1.0) <= 0.0:
                if delta[b] == delta[a]:
                    return float((grid[a] + grid[b]) / 2.0)
                return float(grid[a] + (1.0-delta[a])*(grid[b]-grid[a])/(delta[b]-delta[a]))
        return None

    lo = find_crossing(list(range(best, -1, -1)))
    hi = find_crossing(list(range(best, len(grid))))
    return {
        "status": "finite_profile_interval" if lo is not None and hi is not None else "grid_censored",
        "profile_interval_kappa": [lo, hi],
        "lower_limit_grid_censored": lo is None,
        "upper_limit_grid_censored": hi is None,
        "delta_chi_square_threshold": 1.0,
        "nominal_confidence_level": 0.6827,
        "numerical_profile_domain": [kmin, kmax],
        "profile_grid_kappa": grid,
        "profile_delta_chi_square": delta,
        "profiled_background_model": background_model,
        "profiled_background_parameter_count": int(basis.shape[1]),
        "noise_scale": noise_semantics,
        "interpretation": "Conditional one-parameter profile likelihood; all selected background coefficients are re-optimized at fixed Kappa. A grid-censored limit is not a physical bound."
    }

def _fit_kappa(
    wavelength, signal, laser_nm, angle, response, uncertainty,
    spectral_covariance=None, *, background_model="linear",
    background_degree=2, background_interior_knots=3,
    background_templates=None,
):
    """Fit isotropic finite-variance Kappa spectra with exact Student-t bin integrals.

    The response acts on bin-integrated spectral power. The returned full-model
    area uses the normalized Student-t shape integral, including Gamma(kappa)
    in the denominator, and measured-window coverage is not tail-renormalized.
    """
    from .thomson_background import _background_design

    signal_scale = float(np.max(np.abs(signal)))
    if not np.isfinite(signal_scale) or signal_scale <= 0:
        raise ValueError("signal must have a positive finite scale")
    signal_scaled = signal / signal_scale
    if uncertainty is not None and spectral_covariance is not None:
        raise ValueError("Kappa fitting accepts pointwise uncertainty or spectral_covariance, not both")
    uncertainty_scaled = None if uncertainty is None else uncertainty / signal_scale
    covariance_cholesky = None
    covariance_scaled_input = None
    if spectral_covariance is not None:
        covariance_scaled_input = _finite_array(
            spectral_covariance, "spectral_covariance", 2
        ) / signal_scale**2
        if (covariance_scaled_input.shape != (wavelength.size, wavelength.size)
                or not np.allclose(covariance_scaled_input, covariance_scaled_input.T,
                                   rtol=1e-10, atol=1e-13)):
            raise ValueError("spectral_covariance must be symmetric and match the wavelength grid")
        try:
            covariance_cholesky = np.linalg.cholesky(covariance_scaled_input)
        except np.linalg.LinAlgError as exc:
            raise ValueError("spectral_covariance must be positive definite") from exc

    basis = _background_design(
        wavelength, background_model, degree=background_degree,
        interior_knots=background_interior_knots, templates=background_templates,
    )
    n_background = basis.shape[1]
    spacing = float(np.median(np.diff(wavelength)))
    span = float(wavelength[-1] - wavelength[0])
    edge = max(2, wavelength.size // 10)
    edge_mask = np.zeros(wavelength.size, dtype=bool)
    edge_mask[:edge] = True
    edge_mask[-edge:] = True
    try:
        background_seed = np.linalg.lstsq(
            basis[edge_mask], signal_scaled[edge_mask], rcond=None
        )[0]
    except np.linalg.LinAlgError:
        background_seed = np.zeros(n_background, dtype=float)
    peak = int(np.argmax(signal_scaled))
    baseline = float(np.median(signal_scaled[edge_mask]))
    amp0 = max(float(signal_scaled[peak] - baseline), np.finfo(float).eps)
    p0 = np.r_[amp0, float(wavelength[peak]), max(spacing, span/10), 3.0,
               background_seed]
    lower = np.r_[0.0, wavelength[0], max(spacing*0.2, 1e-12), 1.51,
                  np.full(n_background, -np.inf)]
    upper = np.r_[np.inf, wavelength[-1], span, np.inf,
                  np.full(n_background, np.inf)]

    def intrinsic(x, amp, center, sigma, kappa):
        return _kappa_bin_averaged_shape(x, amp, center, sigma, kappa)[0]

    def background_at(x, coefficients):
        # Evaluate the selected basis on the measured grid and interpolate only
        # within its domain; all fitted samples lie inside this interval.
        return np.column_stack([
            np.interp(np.asarray(x, dtype=float), wavelength, basis[:, j])
            for j in range(n_background)
        ]) @ np.asarray(coefficients, dtype=float)

    def observed(x, *parameters):
        amp, center, sigma, kappa = parameters[:4]
        coefficients = np.asarray(parameters[4:], dtype=float)
        component = intrinsic(x, amp, center, sigma, kappa)
        return _apply_response(response, component, x) + background_at(x, coefficients)

    def weighted_residual(parameters):
        difference = observed(wavelength, *parameters) - signal_scaled
        if covariance_cholesky is not None:
            return np.linalg.solve(covariance_cholesky, difference)
        return difference if uncertainty_scaled is None else difference / uncertainty_scaled

    kappa_seeds = (1.6, 2.0, 3.0, 5.0, 10.0, 30.0)
    width_seeds = (max(spacing, span/10), max(2.0*spacing, span/5),
                   max(4.0*spacing, span/3))
    candidate_fits = []
    attempted_starts = 0
    for kappa_seed in kappa_seeds:
        for width_seed in width_seeds:
            start = p0.copy()
            start[2] = min(max(width_seed, lower[2]*(1.0+1e-8)),
                           upper[2]*(1.0-1e-8))
            start[3] = kappa_seed
            attempted_starts += 1
            try:
                candidate = least_squares(
                    weighted_residual, start, bounds=(lower, upper),
                    method="trf", x_scale="jac", max_nfev=30000,
                    ftol=1e-10, xtol=1e-10, gtol=1e-10,
                )
                if candidate.success and np.all(np.isfinite(candidate.x)) and np.all(np.isfinite(candidate.fun)):
                    candidate_fits.append(candidate)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
    if not candidate_fits:
        raise ValueError("All deterministic multi-start Kappa spectral fits failed to converge.")
    fit = min(candidate_fits, key=lambda candidate: float(candidate.fun @ candidate.fun))
    p_scaled = np.asarray(fit.x, dtype=float)
    dof_fit = max(1, wavelength.size - p_scaled.size)
    covariance_scaled = np.linalg.pinv(fit.jac.T @ fit.jac)
    if uncertainty_scaled is None and covariance_cholesky is None:
        covariance_scaled *= float(fit.fun @ fit.fun / dof_fit)
    parameter_scales = np.r_[signal_scale, 1.0, 1.0, 1.0,
                             np.full(n_background, signal_scale)]
    p = p_scaled * parameter_scales
    cov = np.asarray(covariance_scaled, dtype=float) * np.outer(parameter_scales, parameter_scales)
    amp, center, sigma, kappa = map(float, p[:4])
    background_coefficients = p[4:].copy()
    background_curve = basis @ p_scaled[4:] * signal_scale
    basis_derivative = np.gradient(basis, wavelength, axis=0)
    center_basis = np.array([
        np.interp(center, wavelength, basis[:, j]) for j in range(n_background)
    ])
    center_basis_derivative = np.array([
        np.interp(center, wavelength, basis_derivative[:, j]) for j in range(n_background)
    ])
    background_intercept = float(center_basis @ background_coefficients)
    background_slope = float(center_basis_derivative @ background_coefficients)

    kappa_df = 2.0*kappa - 1.0
    kappa_scale = sigma * np.sqrt((kappa_df - 2.0) / kappa_df)
    wavelength_edges = _wavelength_bin_edges(wavelength)
    captured_probability = float(
        stdtr(kappa_df, (wavelength_edges[-1]-center)/kappa_scale)
        - stdtr(kappa_df, (wavelength_edges[0]-center)/kappa_scale)
    )
    te = thomson_electron_temperature_from_spectral_width(
        sigma, laser_nm, angle
    )["electron_temperature_eV"]
    velocity = thomson_velocity_projection(center, laser_nm, angle)[
        "electron_velocity_projection_m_s"
    ]
    q_projection = 2.0*np.sin(angle/2.0)
    if te / ELECTRON_REST_ENERGY_EV >= NONRELATIVE_THETA_E_SCREEN:
        raise ValueError("Non-relativistic kappa model rejected: fitted electron temperature is not safely small relative to the electron rest energy; the selected non-relativistic model is outside its declared validity range.")
    if abs(velocity) >= NONRELATIVE_FLOW_BETA_SCREEN*C_M_S:
        raise ValueError("Non-relativistic kappa model rejected: fitted flow projection is outside the declared |v|/c < 0.1 regime.")
    fitted_spectrum = signal_scale * observed(wavelength, *p_scaled)
    residual = signal - fitted_spectrum

    jacobian = np.empty((wavelength.size, p_scaled.size), dtype=float)
    for parameter_index in range(p_scaled.size):
        step = np.cbrt(np.finfo(float).eps) * max(abs(float(p_scaled[parameter_index])), 1.0)
        plus, minus = p_scaled.copy(), p_scaled.copy()
        plus[parameter_index] += step
        minus[parameter_index] -= step
        jacobian[:, parameter_index] = (
            observed(wavelength, *plus) - observed(wavelength, *minus)
        ) / (2.0*step)
    if covariance_cholesky is not None:
        weighted_jacobian = np.linalg.solve(covariance_cholesky, jacobian)
    elif uncertainty_scaled is not None:
        weighted_jacobian = jacobian / uncertainty_scaled[:, None]
    else:
        weighted_jacobian = jacobian
    column_norms = np.linalg.norm(weighted_jacobian, axis=0)
    nonzero_columns = column_norms > np.finfo(float).tiny
    normalized_jacobian = np.zeros_like(weighted_jacobian)
    normalized_jacobian[:, nonzero_columns] = (
        weighted_jacobian[:, nonzero_columns] / column_norms[nonzero_columns]
    )
    singular_values = np.linalg.svd(normalized_jacobian, compute_uv=False)
    rank_tolerance = np.finfo(float).eps * max(normalized_jacobian.shape) * (
        float(singular_values[0]) if singular_values.size else 0.0
    )
    jacobian_rank = int(np.sum(singular_values > rank_tolerance))
    jacobian_condition = (
        float(singular_values[0]/singular_values[-1])
        if singular_values.size and singular_values[-1] > rank_tolerance else float("inf")
    )
    kappa_column = weighted_jacobian[:, 3]
    nuisance_indices = [j for j in range(p_scaled.size) if j != 3]
    nuisance_columns = weighted_jacobian[:, nuisance_indices]
    nuisance_coefficients = np.linalg.lstsq(
        nuisance_columns, kappa_column, rcond=None
    )[0]
    conditional_kappa_sensitivity = float(
        np.linalg.norm(kappa_column - nuisance_columns@nuisance_coefficients)
        / max(np.linalg.norm(kappa_column), np.finfo(float).tiny)
    )
    kappa_standard_uncertainty = float(np.sqrt(max(cov[3, 3], 0.0)))
    kappa_identifiable = bool(
        jacobian_rank == p_scaled.size and np.isfinite(jacobian_condition)
        and jacobian_condition <= 1e12 and conditional_kappa_sensitivity > 1e-8
        and np.isfinite(kappa_standard_uncertainty)
    )
    kappa_identifiability = {
        "status": "locally_identifiable" if kappa_identifiable else "weak_or_non_identifiable",
        "reportable_as_inferred_parameter": kappa_identifiable,
        "jacobian_rank": jacobian_rank,
        "parameter_count": int(p_scaled.size),
        "normalized_jacobian_condition_number": jacobian_condition,
        "conditional_kappa_sensitivity": conditional_kappa_sensitivity,
        "kappa_index_standard_uncertainty": kappa_standard_uncertainty,
        "diagnostic_basis": "finite-difference Jacobian of the response-convolved Kappa spectrum and selected detector-background basis; Kappa sensitivity is projected orthogonally to all nuisance-parameter sensitivities",
        "threshold_semantics": "rank and floating-point conditioning diagnostics; not an empirical or universal physical threshold",
        "limitation": None if kappa_identifiable else "The measured spectral window and supplied response do not locally distinguish the Kappa shape parameter from the remaining fitted parameters; do not interpret the optimizer candidate as an identified physical Kappa index.",
    }

    kappa_profile_likelihood = _kappa_profile_likelihood_diagnostic(
        wavelength, signal_scaled, response, p_scaled,
        uncertainty_scaled, spectral_covariance=covariance_scaled_input,
        background_model=background_model, background_degree=background_degree,
        background_interior_knots=background_interior_knots,
        background_templates=background_templates,
    )

    def area_from_params(values):
        a, cen, sig, kap = values[:4]
        df_local = 2.0*kap - 1.0
        scale_local = sig*np.sqrt((df_local-2.0)/df_local)
        return float(a*scale_local*np.sqrt(df_local*np.pi)
                     * np.exp(gammaln(kap-0.5)-gammaln(kap)))
    area = area_from_params(p)
    area_gradient = np.zeros(len(p))
    for j in range(4):
        step = np.cbrt(np.finfo(float).eps)*max(abs(float(p[j])), 1.0)
        plus, minus = p.copy(), p.copy()
        plus[j] += step
        minus[j] -= step
        area_gradient[j] = (area_from_params(plus)-area_from_params(minus))/(2.0*step)
    area_unc = float(np.sqrt(max(float(area_gradient@cov@area_gradient), 0.0)))
    names = ["amplitude", "center_nm", "intrinsic_sigma_nm", "kappa_index"] + [
        f"background_coefficient_{j}" for j in range(n_background)
    ]
    return {
        "electron_temperature_eV": float(te),
        "electron_temperature_uncertainty_eV": float(te*2*np.sqrt(max(cov[2,2],0.0))/sigma),
        "electron_velocity_projection_m_s": float(velocity),
        "electron_velocity_projection_uncertainty_m_s": float(C_M_S*np.sqrt(max(cov[1,1],0.0))/(laser_nm*q_projection)),
        "fitted_center_nm": center, "intrinsic_sigma_nm": sigma,
        "kappa_index": kappa, "kappa_index_identifiability": kappa_identifiability,
        "kappa_profile_likelihood": kappa_profile_likelihood,
        "amplitude": amp, "background_intercept": background_intercept,
        "background_slope_per_nm": background_slope,
        "background_model": background_model,
        "background_coefficients": background_coefficients,
        "background_basis": basis,
        "background_fitted_spectrum": background_curve,
        "fit_covariance": np.asarray(cov, dtype=float), "residual": residual,
        "fit_parameters": np.asarray(p, dtype=float), "fit_parameter_names": names,
        "integrated_scattered_power_W": area,
        "integrated_scattered_power_uncertainty_W": area_unc,
        "captured_spectral_probability": captured_probability,
        "spectral_bin_integration": "exact_Student_t_CDF_finite_volume_bin_averages",
        "spectral_window_truncation_warning": (
            "The measured wavelength window captures only a fraction of the fitted heavy-tailed kappa spectrum; integrated model area is extrapolated beyond the measured window."
            if captured_probability < 0.999 else None
        ),
        "reduced_chi_square": (
            float(np.sum((residual/uncertainty)**2)/max(1, wavelength.size-len(p)))
            if uncertainty is not None else
            float(fit.fun@fit.fun/max(1, wavelength.size-len(p)))
            if covariance_cholesky is not None else None
        ),
        "spectral_covariance_used": covariance_cholesky is not None,
        "spectral_covariance_interpretation": (
            "researcher_supplied_within_channel_covariance_including_model_discrepancy_when_provided"
            if covariance_cholesky is not None else "not_supplied"
        ),
        "fitted_spectrum": fitted_spectrum,
        "optimizer": {
            "method": "deterministic_multistart_bounded_least_squares",
            "attempted_starts": int(attempted_starts),
            "successful_starts": int(len(candidate_fits)),
            "selected_weighted_residual_sum_squares": float(fit.fun@fit.fun),
            "selected_initialization_kappa_grid": list(kappa_seeds),
            "selected_initialization_width_grid_nm": list(width_seeds),
            "success": bool(fit.success), "message": str(fit.message),
            "nfev": int(fit.nfev),
        },
        "model": "non-collective isotropic kappa electron VDF (Student-t projected spectrum)",
        "validity_screening": {
            "theta_e": float(te/ELECTRON_REST_ENERGY_EV),
            "theta_e_limit": NONRELATIVE_THETA_E_SCREEN,
            "bulk_flow_beta": float(abs(velocity)/C_M_S),
            "bulk_flow_beta_limit": NONRELATIVE_FLOW_BETA_SCREEN,
            "rationale": "Conservative asymptotic error-budget screens: thermal relativistic corrections scale with theta_e and leading bulk-flow corrections scale with beta_flow squared; not universal validity boundaries.",
        },
        "assumptions": [
            "non-collective single-particle scattering", "isotropic kappa electron velocity distribution",
            "non-relativistic velocities", "finite-variance kappa index > 1.5",
            "known scattering geometry",
            "selected detector-space background is a nuisance model, not plasma emission",
        ],
    }

def _default_collective_bounds(initial_density_m3, initial_temperature_eV):
    """Return adaptive two-decade bounds when no experiment-specific bounds are supplied."""
    density = _finite_scalar(initial_density_m3, "initial density", minimum=np.finfo(float).tiny)
    temperature = _finite_scalar(initial_temperature_eV, "initial temperature", minimum=np.finfo(float).tiny)
    return (np.array([density / 100.0, density * 100.0]),
            np.array([temperature / 100.0, temperature * 100.0]))

def _collective_external_parameter_indices(channel_index, ion_species_count):
    """Return covariance-vector indices for one channel in channel-major order."""
    channel_index = int(channel_index)
    ion_species_count = int(ion_species_count)
    if channel_index < 0 or ion_species_count < 1:
        raise ValueError("channel_index must be non-negative and ion_species_count must be positive")
    base = channel_index * (3 + ion_species_count)
    return base, base + 1, base + 2, tuple(range(base + 3, base + 3 + ion_species_count))


def analyze_thomson_spectra(
    wavelength_nm, spectra, radius_m, *, laser_wavelength_nm, scattering_angle_rad,
    laser_wavelength_uncertainty_nm=None, scattering_angle_uncertainty_rad=None,
    incident_direction_uncertainty=None, scattered_direction_uncertainty=None,
    instrument_response=None, signal_uncertainty=None,
    laser_power_w=None, interaction_length_m=None, collection_solid_angle_sr=None,
    optical_efficiency=None, incident_polarization=None, incident_direction=None, scattered_direction=None,
    spectral_power_density_calibrated=False, laser_power_uncertainty_W=None,
    interaction_length_uncertainty_m=None, collection_solid_angle_uncertainty_sr=None,
    optical_efficiency_uncertainty=None, polarization_factor_uncertainty=None,
    density_relative_covariance=None, joint_density_relative_covariance=None,
    instrument_response_uncertainty=None, model=None,
    laser_spectrum_wavelength_nm=None, laser_spectrum_density_per_nm=None,
    ion_species=None, ion_fractions=None, ion_temperature_eV=None, ion_velocity_m_s=None, electron_velocity_m_s=None,
    collective_initial_density_m3=None, collective_initial_temperature_eV=None,
    collective_density_bounds_m3=None, collective_temperature_bounds_eV=None,
    instrument_response_mc_samples=200, instrument_response_mc_min_success_fraction=0.8,
    joint_spectral_covariance=None, model_discrepancy_covariance=None,
    systematic_covariance=None,
    ion_temperature_uncertainty_eV=None, ion_velocity_uncertainty_m_s=None,
    electron_velocity_uncertainty_m_s=None, collective_external_parameter_mc_samples=None,
    collective_kappa_initial=None, collective_kappa_bounds=None,
    incident_polarization_uncertainty=None,
    background_model="linear", background_degree=2,
    background_interior_knots=3, background_templates=None,
    maximum_recoil_correction_fraction=0.01,
    include_compton_recoil=False,
    enable_evdf_reconstruction=True,
    evdf_velocity_points=121,
):
    """Infer per-channel Thomson parameters from calibrated multi-channel spectra.

    Spectra have shape (channel, wavelength); wavelength may be shared (1-D) or
    channel-specific (2-D). A measured response callable, kernel or matrix maps true
    spectral bins to detector bins. Supplied response matrices must already have unit-sum columns and are used without renormalization. Absolute density is emitted only
    when all optical-power/geometry/polarization factors are supplied.
    """
    if not isinstance(include_compton_recoil, (bool, np.bool_)):
        raise ValueError("include_compton_recoil must be boolean")
    if not isinstance(enable_evdf_reconstruction, (bool, np.bool_)):
        raise ValueError("enable_evdf_reconstruction must be boolean")
    if (
        isinstance(evdf_velocity_points, (bool, np.bool_))
        or not isinstance(evdf_velocity_points, (int, np.integer))
        or evdf_velocity_points < 41
        or evdf_velocity_points > 2001
    ):
        raise ValueError("evdf_velocity_points must be an integer from 41 to 2001")

    y = _finite_array(spectra, "spectra", 2)
    if y.shape[1] < 7:
        raise ValueError("Each Thomson spectrum must contain at least seven wavelength samples")
    radii = _finite_array(radius_m, "radius_m", 1)
    if y.shape[0] != radii.size:
        raise ValueError("radius_m length must equal the number of spectral channels")
    x = _finite_array(wavelength_nm, "wavelength_nm")
    if x.ndim == 1:
        if x.size != y.shape[1]:
            raise ValueError("wavelength_nm length must equal the spectral width")
        xs = np.broadcast_to(x, y.shape)
    elif x.ndim == 2 and x.shape == y.shape:
        xs = x
    else:
        raise ValueError("wavelength_nm must be shared 1-D or channel-shaped 2-D")
    if np.any(np.diff(xs, axis=1) <= 0):
        raise ValueError("Each channel wavelength axis must be strictly increasing")
    if np.any(xs <= 0):
        raise ValueError("Wavelength samples must be positive")
    if instrument_response is None:
        raise ValueError("A measured instrument-response matrix is required; supply an identity matrix only when a delta-like response is explicitly justified.")
    if (laser_spectrum_wavelength_nm is None) != (laser_spectrum_density_per_nm is None):
        raise ValueError("Supply both laser_spectrum_wavelength_nm and laser_spectrum_density_per_nm, or neither.")
    if laser_spectrum_wavelength_nm is not None and model not in {
        "non_collective_maxwellian", "non_collective_kappa",
        "non_collective_relativistic_maxwell_juttner",
    }:
        raise ValueError("A measured incident-laser spectral profile is currently supported for non-collective optical Thomson models; collective dielectric models require a source-frequency-resolved dielectric forward model.")
    # This function is the channel-wise inference path. The separate joint
    # spatial-spectral inverse owns its own Compton/source-profile capability
    # gate; referencing that mode flag here was an undefined-name failure.
    laser_wavelength_nm = _finite_scalar(laser_wavelength_nm, "laser_wavelength_nm", minimum=np.finfo(float).tiny)
    scattering_angle_rad = _finite_scalar(scattering_angle_rad, "scattering_angle_rad", minimum=np.finfo(float).tiny, maximum=np.pi)
    if scattering_angle_rad >= np.pi:
        raise ValueError("scattering_angle_rad must lie strictly between 0 and pi")
    for uncertainty_value, uncertainty_name in (
        (laser_wavelength_uncertainty_nm,"laser_wavelength_uncertainty_nm"),
        (scattering_angle_uncertainty_rad,"scattering_angle_uncertainty_rad"),
    ):
        if uncertainty_value is not None:
            _finite_scalar(uncertainty_value, uncertainty_name, minimum=0.0)
    laser_wavelength_unc = 0.0 if laser_wavelength_uncertainty_nm is None else float(laser_wavelength_uncertainty_nm)
    scattering_angle_unc = 0.0 if scattering_angle_uncertainty_rad is None else float(scattering_angle_uncertainty_rad)
    incident_direction_unc = None if incident_direction_uncertainty is None else _finite_array(
        incident_direction_uncertainty, "incident_direction_uncertainty", 1)
    scattered_direction_unc = None if scattered_direction_uncertainty is None else _finite_array(
        scattered_direction_uncertainty, "scattered_direction_uncertainty", 1)
    for name, value in (("incident_direction_uncertainty", incident_direction_unc),
                        ("scattered_direction_uncertainty", scattered_direction_unc)):
        if value is not None and (value.shape != (3,) or np.any(value < 0)):
            raise ValueError(f"{name} must be a non-negative 3-vector of component-wise 1-sigma uncertainties")
    if (incident_direction_unc is not None or scattered_direction_unc is not None) and scattering_angle_uncertainty_rad is not None:
        raise ValueError("Use scattering_angle_uncertainty_rad or beam-direction component uncertainties, not both; they describe the same geometry uncertainty.")
    if model is None:
        raise ValueError("Select a Thomson physical model explicitly; no default Maxwellian assumption is applied.")
    supported_models = {
        "non_collective_maxwellian",
        "non_collective_kappa",
        "non_collective_relativistic_maxwell_juttner",
        "collective_maxwellian",
        "collective_kappa",
    }
    maximum_recoil_correction_fraction = _finite_scalar(
        maximum_recoil_correction_fraction, "maximum_recoil_correction_fraction",
        minimum=np.finfo(float).tiny, maximum=1.0 - np.finfo(float).eps,
    )
    if model not in supported_models:
        raise ValueError(
            "Unsupported optical Thomson model. Analysis 1 supports non_collective_maxwellian, "
            "non_collective_kappa and non_collective_relativistic_maxwell_juttner. "
            "Optical collective dielectric models are outside this shortlist; microwave CTS "
            "is implemented as a separate child subanalysis of primary Analysis 1."
        )
    if (incident_direction_unc is not None or scattered_direction_unc is not None) and model != "non_collective_relativistic_maxwell_juttner":
        raise ValueError(
            "Beam-direction component uncertainty is supported only by the relativistic "
            "Maxwell-Juttner optical Thomson model."
        )
    polarization_unc = None
    if incident_polarization_uncertainty is not None:
        if model != "non_collective_relativistic_maxwell_juttner":
            raise ValueError("incident_polarization_uncertainty is supported only by the relativistic Maxwell-Juttner model")
        polarization_unc = _finite_array(
            incident_polarization_uncertainty, "incident_polarization_uncertainty", 1
        )
        if polarization_unc.shape != (3,) or np.any(polarization_unc < 0):
            raise ValueError("incident_polarization_uncertainty must be a non-negative 3-vector of component-wise 1-sigma uncertainties")
    # Collective dielectric models are valid optical Thomson forward models too.
    # They share the measured-spectrum likelihood and detector response with the
    # non-collective fits, but require explicit ion composition, temperatures,
    # flows and bounded electron seeds below; no plasma parameters are defaulted.
    if not isinstance(spectral_power_density_calibrated, (bool, np.bool_)):
        raise ValueError("spectral_power_density_calibrated must be boolean")
    if isinstance(instrument_response_mc_samples, (bool, np.bool_)) or not isinstance(instrument_response_mc_samples, (int, np.integer)) or instrument_response_mc_samples < 20:
        raise ValueError("instrument_response_mc_samples must be an integer >= 20")
    instrument_response_mc_min_success_fraction = _finite_scalar(instrument_response_mc_min_success_fraction, "instrument_response_mc_min_success_fraction", minimum=0.5, maximum=1.0)
    collective_models = {"collective_maxwellian", "collective_kappa"}
    density_bounds = temperature_bounds = None
    if model in collective_models:
        if collective_initial_density_m3 is None or collective_initial_temperature_eV is None:
            raise ValueError("Collective Thomson models require explicit initial electron density and temperature estimates.")
        collective_initial_density_m3 = _finite_scalar(
            collective_initial_density_m3, "collective_initial_density_m3",
            minimum=np.finfo(float).tiny,
        )
        collective_initial_temperature_eV = _finite_scalar(
            collective_initial_temperature_eV, "collective_initial_temperature_eV",
            minimum=np.finfo(float).tiny,
        )
        if collective_density_bounds_m3 is not None:
            density_bounds = _finite_array(collective_density_bounds_m3, "collective_density_bounds_m3", 1)
            if density_bounds.shape != (2,) or np.any(density_bounds <= 0) or density_bounds[0] >= density_bounds[1]:
                raise ValueError("collective_density_bounds_m3 must be a positive increasing pair")
            if not density_bounds[0] <= collective_initial_density_m3 <= density_bounds[1]:
                raise ValueError("collective_initial_density_m3 must lie within collective_density_bounds_m3")
        if collective_temperature_bounds_eV is not None:
            temperature_bounds = _finite_array(collective_temperature_bounds_eV, "collective_temperature_bounds_eV", 1)
            if temperature_bounds.shape != (2,) or np.any(temperature_bounds <= 0) or temperature_bounds[0] >= temperature_bounds[1]:
                raise ValueError("collective_temperature_bounds_eV must be a positive increasing pair")
            if not temperature_bounds[0] <= collective_initial_temperature_eV <= temperature_bounds[1]:
                raise ValueError("collective_initial_temperature_eV must lie within collective_temperature_bounds_eV")
        if model == "collective_maxwellian" and (density_bounds is None or temperature_bounds is None):
            raise ValueError("Collective Maxwellian inference requires explicit electron-density and electron-temperature bounds; no scientific range is assumed.")
    if model == "collective_kappa":
        if collective_kappa_initial is None:
            raise ValueError("Collective kappa inference requires an explicit initial kappa index >1.5.")
        collective_kappa_initial = _finite_scalar(
            collective_kappa_initial, "collective_kappa_initial", minimum=1.5000001,
        )
        if collective_kappa_bounds is not None:
            collective_kappa_bounds = _finite_array(collective_kappa_bounds, "collective_kappa_bounds", 1)
            if (collective_kappa_bounds.shape != (2,) or collective_kappa_bounds[0] <= 1.5
                    or collective_kappa_bounds[0] >= collective_kappa_bounds[1]):
                raise ValueError("collective_kappa_bounds must be an increasing pair with lower bound >1.5")
            if not collective_kappa_bounds[0] <= collective_kappa_initial <= collective_kappa_bounds[1]:
                raise ValueError("collective_kappa_initial must lie within collective_kappa_bounds")
    probe_vec = scatter_vec = None
    if model in collective_models or model == "non_collective_relativistic_maxwell_juttner":
        if incident_direction is None or scattered_direction is None:
            raise ValueError("The selected Thomson model requires explicit incident_direction and scattered_direction 3-vectors.")
        if model == "non_collective_relativistic_maxwell_juttner" and incident_polarization is None:
            raise ValueError("The relativistic Maxwell-Juttner Thomson model requires an explicit incident_polarization vector.")
        probe_vec = _finite_array(incident_direction, "incident_direction", 1)
        scatter_vec = _finite_array(scattered_direction, "scattered_direction", 1)
        if probe_vec.shape != (3,) or scatter_vec.shape != (3,):
            raise ValueError("incident_direction and scattered_direction must each be 3-vectors")
        probe_norm = float(np.linalg.norm(probe_vec))
        scatter_norm = float(np.linalg.norm(scatter_vec))
        if probe_norm <= 0 or scatter_norm <= 0:
            raise ValueError("incident_direction and scattered_direction must be non-zero")
        probe_vec = probe_vec/probe_norm
        scatter_vec = scatter_vec/scatter_norm
        geometry_angle = float(np.arccos(np.clip(np.dot(probe_vec, scatter_vec), -1.0, 1.0)))
        if not np.isclose(geometry_angle, float(scattering_angle_rad), rtol=0.0, atol=1e-5):
            raise ValueError("scattering_angle_rad is inconsistent with incident_direction and scattered_direction")

    if model in collective_models:
        if model in collective_models and (
            isinstance(collective_external_parameter_mc_samples,bool)
            or not isinstance(collective_external_parameter_mc_samples,(int,np.integer))
            or collective_external_parameter_mc_samples<20
        ):
            raise ValueError("collective_external_parameter_mc_samples must be an integer >= 20")
        if not ion_species or isinstance(ion_species, (str, bytes)):
            raise ValueError("Collective Thomson inference requires an explicit non-empty ion species list")
        if ion_velocity_m_s is None or electron_velocity_m_s is None:
            raise ValueError("Collective inference requires explicit ion/electron flow vectors; use zero vectors only to declare stationary flow.")
        if ion_temperature_eV is None:
            raise ValueError("Collective inference requires independently supplied ion temperature(s); no ion temperature is assumed")
        ion_t_check = _finite_array(np.atleast_1d(ion_temperature_eV), "ion_temperature_eV", 1)
        if ion_t_check.size != len(ion_species) or np.any(ion_t_check <= 0):
            raise ValueError("ion_temperature_eV must contain one positive temperature per ion species")
        if ion_fractions is None:
            if len(ion_species) > 1:
                raise ValueError("Collective multi-species inference requires explicit ion_fractions summing to one")
            ion_fractions = [1.0]
        ion_fract_check = _finite_array(ion_fractions, "ion_fractions", 1)
        if ion_fract_check.size != len(ion_species) or np.any(ion_fract_check <= 0) or not np.isclose(np.sum(ion_fract_check),1.0,rtol=1e-6,atol=1e-8):
            raise ValueError("ion_fractions must contain one positive fraction per species and sum to one")
        ion_v_check = _finite_array(ion_velocity_m_s, "ion_velocity_m_s")
        electron_v_check = _finite_array(electron_velocity_m_s, "electron_velocity_m_s")
        if ion_v_check.shape == (3,) and len(ion_species) == 1:
            ion_v_check = ion_v_check.reshape(1,3)
        if electron_v_check.shape == (3,):
            electron_v_check = electron_v_check.reshape(1,3)
        if electron_v_check.shape != (1, 3) or ion_v_check.shape != (len(ion_species), 3):
            raise ValueError("electron_velocity_m_s must be a 3-vector and ion_velocity_m_s one 3-vector per species")
        if np.any(np.linalg.norm(ion_v_check,axis=1) >= C_M_S) or np.linalg.norm(electron_v_check[0]) >= C_M_S:
            raise ValueError("Collective ion/electron flow speeds must be subluminal")
        ion_t_unc = None
        if ion_temperature_uncertainty_eV is not None:
            ion_t_unc = _finite_array(np.atleast_1d(ion_temperature_uncertainty_eV),"ion_temperature_uncertainty_eV",1)
            if ion_t_unc.shape != ion_t_check.shape or np.any(ion_t_unc < 0):
                raise ValueError("ion_temperature_uncertainty_eV must be non-negative and match ion species")
        ion_v_unc = None
        if ion_velocity_uncertainty_m_s is not None:
            ion_v_unc = _finite_array(ion_velocity_uncertainty_m_s,"ion_velocity_uncertainty_m_s")
            if ion_v_unc.shape == (3,) and len(ion_species) == 1:
                ion_v_unc = ion_v_unc.reshape(1,3)
            if ion_v_unc.shape != ion_v_check.shape or np.any(ion_v_unc < 0):
                raise ValueError("ion_velocity_uncertainty_m_s must be non-negative and match ion velocity vectors")
        electron_v_unc = None
        if electron_velocity_uncertainty_m_s is not None:
            electron_v_unc = _finite_array(electron_velocity_uncertainty_m_s,"electron_velocity_uncertainty_m_s",1)
            if electron_v_unc.shape != (3,) or np.any(electron_v_unc < 0):
                raise ValueError("electron_velocity_uncertainty_m_s must be a non-negative 3-vector")
        ion_t = ion_t_check
        ion_fract = ion_fract_check
        ion_v = ion_v_check
        electron_v = electron_v_check
        if model == "collective_maxwellian":
            try:
                from astropy import units as u
                from plasmapy.diagnostics import thomson
            except ImportError as exc:
                raise ImportError("Collective Maxwellian Thomson inference requires the optional PlasmaPy and Astropy packages") from exc
        else:
            thomson = None
    else:
        thomson = None
        ion_t = ion_fract = ion_v = electron_v = None
        ion_t_unc = ion_v_unc = electron_v_unc = None


    unc = None if signal_uncertainty is None else _finite_array(signal_uncertainty, "signal_uncertainty", 2)
    if unc is not None and (unc.shape != y.shape or np.any(unc <= 0)):
        raise ValueError("signal_uncertainty must match spectra and be strictly positive")
    measurement_covariance_supplied = (
        unc is not None or joint_spectral_covariance is not None
    )
    model_discrepancy_covariance_supplied = model_discrepancy_covariance is not None
    systematic_covariance_supplied = systematic_covariance is not None
    lsf_unc = None if instrument_response_uncertainty is None else _finite_array(instrument_response_uncertainty, "instrument_response_uncertainty")
    kernel_uncertainty = False
    matrix_uncertainty = False
    if lsf_unc is not None:
        response_array = None if callable(instrument_response) else np.asarray(instrument_response)
        kernel_uncertainty = bool(
            lsf_unc.ndim == 1 and response_array is not None
            and response_array.ndim == 1 and response_array.shape == lsf_unc.shape
        )
        matrix_uncertainty = bool(
            lsf_unc.ndim in (2, 3)
            and lsf_unc.shape[-2:] == (y.shape[1], y.shape[1])
            and (lsf_unc.ndim != 3 or lsf_unc.shape[0] == y.shape[0])
        )
        if np.any(lsf_unc < 0) or instrument_response is None or not (kernel_uncertainty or matrix_uncertainty):
            raise ValueError(
                "instrument_response_uncertainty must be a non-negative 1-D uncertainty matching a supplied 1-D response kernel, "
                "a shared 2-D response-matrix uncertainty, or a channel-specific 3-D response-matrix uncertainty"
            )
        if kernel_uncertainty and model in {"collective_maxwellian", "collective_kappa"}:
            raise ValueError("A 1-D response-kernel uncertainty is currently supported only by the non-collective optical Maxwellian, Kappa and relativistic Maxwell-Juttner fits; collective CTS requires matrix uncertainty.")


    def require_zero_scattered_power_covariance(covariance_value, name):
        matrix = _finite_array(covariance_value, name)
        if matrix.shape == (6, 6):
            power_rows = [0]
            row_check = matrix[0, :]
            column_check = matrix[:, 0]
        elif matrix.shape == (y.shape[0], 6, 6):
            power_rows = None
            row_check = matrix[:, 0, :]
            column_check = matrix[:, :, 0]
        elif matrix.shape == (6 * y.shape[0], 6 * y.shape[0]):
            power_rows = np.arange(0, 6 * y.shape[0], 6, dtype=int)
            row_check = matrix[power_rows, :]
            column_check = matrix[:, power_rows]
        else:
            raise ValueError(
                f"{name} must be 6x6, n_channels x 6 x 6, or (6*n_channels) x (6*n_channels)."
            )
        if not np.allclose(row_check, 0.0, rtol=0.0, atol=1e-14) or not np.allclose(
            column_check, 0.0, rtol=0.0, atol=1e-14
        ):
            raise ValueError(
                f"{name} must have zero scattered-power rows and columns when spectral "
                "measurement covariance, model-discrepancy covariance, or measured-response "
                "uncertainty is supplied; those power uncertainties are propagated from the "
                "spectrum/response fit and must not be counted twice."
            )
        return matrix

    if density_relative_covariance is not None and joint_spectral_covariance is not None:
        raise ValueError("Per-channel density_relative_covariance is not compatible with joint_spectral_covariance; do not combine it with joint_spectral_covariance.")
    if density_relative_covariance is not None and (
        joint_spectral_covariance is not None
        or model_discrepancy_covariance is not None
        or systematic_covariance is not None
        or lsf_unc is not None
    ):
        require_zero_scattered_power_covariance(
            density_relative_covariance, "density_relative_covariance"
        )
    if joint_density_relative_covariance is not None:
        if model in {"collective_maxwellian", "collective_kappa"}:
            raise ValueError("joint_density_relative_covariance is for non-collective absolute-density inversion; collective models infer density from the dielectric spectral shape.")
        if density_relative_covariance is not None:
            raise ValueError("Supply density_relative_covariance or joint_density_relative_covariance, not both.")
        if (
            joint_spectral_covariance is not None
            or model_discrepancy_covariance is not None
            or systematic_covariance is not None
            or lsf_unc is not None
        ):
            require_zero_scattered_power_covariance(
                joint_density_relative_covariance, "joint_density_relative_covariance"
            )
        if not spectral_power_density_calibrated or any(value is None for value in (
            laser_power_w, interaction_length_m, collection_solid_angle_sr, optical_efficiency
        )):
            raise ValueError("joint_density_relative_covariance requires calibrated W/nm spectra and complete absolute-density optical geometry for every channel.")
        scalar_density_uncertainties = (
            laser_power_uncertainty_W, interaction_length_uncertainty_m,
            collection_solid_angle_uncertainty_sr, optical_efficiency_uncertainty,
            polarization_factor_uncertainty,
        )
        if any(value is not None for value in scalar_density_uncertainties):
            raise ValueError("joint_density_relative_covariance is the cross-channel optical-factor uncertainty model; do not also supply scalar optical-factor uncertainties.")
        joint_density_covariance = _finite_array(
            joint_density_relative_covariance, "joint_density_relative_covariance", 2
        )
        expected_density_covariance_shape = (6 * y.shape[0], 6 * y.shape[0])
        if joint_density_covariance.shape != expected_density_covariance_shape:
            raise ValueError("joint_density_relative_covariance must have shape (6*n_channels, 6*n_channels), ordered as six density-inversion inputs per channel.")
        if not np.allclose(joint_density_covariance, joint_density_covariance.T, rtol=1e-9, atol=0.0):
            raise ValueError("joint_density_relative_covariance must be symmetric")
        if not _covariance_is_positive_semidefinite(joint_density_covariance):
            raise ValueError("joint_density_relative_covariance must be positive semidefinite")
        if (
            joint_spectral_covariance is not None
            or model_discrepancy_covariance is not None
            or systematic_covariance is not None
            or lsf_unc is not None
        ):
            require_zero_scattered_power_covariance(
                joint_density_covariance, "joint_density_relative_covariance"
            )
    else:
        joint_density_covariance = None

    if joint_spectral_covariance is not None and signal_uncertainty is not None:
        raise ValueError(
            "Supply either signal_uncertainty (independent pointwise 1-sigma errors) "
            "or joint_spectral_covariance (the complete channel-major covariance), not both. "
            "The full covariance is authoritative and must not be silently combined with pointwise errors."
        )
    joint_covariance = None
    if joint_spectral_covariance is not None:
        if model not in {"non_collective_maxwellian","non_collective_kappa","non_collective_relativistic_maxwell_juttner"}:
            raise ValueError("joint_spectral_covariance currently supports non_collective_maxwellian, non_collective_kappa, and block-diagonal non_collective_relativistic_maxwell_juttner; other physical models require model-specific joint forward models")
        joint_covariance = _finite_array(joint_spectral_covariance, "joint_spectral_covariance", 2)
        expected_joint_shape = (y.size, y.size)
        if joint_covariance.shape != expected_joint_shape or not np.allclose(joint_covariance, joint_covariance.T, rtol=1e-9, atol=0.0):
            raise ValueError("joint_spectral_covariance must be a symmetric (n_channels*n_wavelengths) square matrix in channel-major order")
        try:
            joint_cholesky = np.linalg.cholesky(joint_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("joint_spectral_covariance must be positive definite") from exc
        # Full channel-major covariance is retained for the joint relativistic
        # shared-parameter fit below; marginal channel fits use diagonal blocks.
    independent_discrepancy_blocks = None
    discrepancy_added_to_joint_covariance = False
    if model_discrepancy_covariance is not None:
        if model not in {
            "non_collective_maxwellian", "non_collective_kappa",
            "non_collective_relativistic_maxwell_juttner", "collective_maxwellian",
            "collective_kappa",
        }:
            raise ValueError(
                "model_discrepancy_covariance is supported for non-collective optical Thomson "
                "models and both collective Maxwellian and collective Kappa fits"
            )
        discrepancy = _finite_array(model_discrepancy_covariance, "model_discrepancy_covariance")
        nchan_disc, nlam_disc = y.shape
        full_discrepancy_covariance = None
        if discrepancy.shape == (y.size, y.size):
            full_discrepancy_covariance = discrepancy.copy()
            if not np.allclose(
                full_discrepancy_covariance,
                full_discrepancy_covariance.T,
                rtol=1e-10, atol=1e-13,
            ):
                raise ValueError(
                    "full channel-major model_discrepancy_covariance must be symmetric"
                )
            full_discrepancy_covariance = 0.5 * (
                full_discrepancy_covariance + full_discrepancy_covariance.T
            )
            eigenvalues = np.linalg.eigvalsh(full_discrepancy_covariance)
            scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
            if float(np.min(eigenvalues)) < -1e-10 * scale:
                raise ValueError(
                    "full channel-major model_discrepancy_covariance must be positive semidefinite"
                )
            if model in {"collective_maxwellian", "collective_kappa"}:
                for row in range(nchan_disc):
                    for col in range(nchan_disc):
                        if row != col:
                            cross = full_discrepancy_covariance[
                                row*nlam_disc:(row+1)*nlam_disc,
                                col*nlam_disc:(col+1)*nlam_disc,
                            ]
                            if np.any(np.abs(cross) > 1e-13 * scale):
                                raise ValueError(
                                    f"{model} currently fits channels independently; "
                                    "cross-channel model-discrepancy covariance cannot be silently discarded"
                                )
            independent_discrepancy_blocks = np.stack([
                full_discrepancy_covariance[
                    channel_index*nlam_disc:(channel_index+1)*nlam_disc,
                    channel_index*nlam_disc:(channel_index+1)*nlam_disc,
                ]
                for channel_index in range(nchan_disc)
            ])
            if joint_covariance is None:
                # Pointwise errors are independent across channels unless a
                # full measured covariance was supplied. Preserve those
                # marginals while adding the complete discrepancy covariance.
                if unc is not None:
                    joint_covariance = np.diag(np.asarray(unc, dtype=float).reshape(-1)**2)
                    unc = None
                else:
                    joint_covariance = np.zeros((y.size, y.size), dtype=float)
            joint_covariance += full_discrepancy_covariance
            discrepancy_added_to_joint_covariance = True
        elif discrepancy.shape == (nlam_disc, nlam_disc):
            independent_discrepancy_blocks = np.broadcast_to(
                discrepancy, (nchan_disc, nlam_disc, nlam_disc)
            ).copy()
        elif discrepancy.shape == (nchan_disc, nlam_disc, nlam_disc):
            independent_discrepancy_blocks = discrepancy.copy()
        else:
            raise ValueError(
                "model_discrepancy_covariance must be (n_wavelength,n_wavelength), "
                "(n_channels,n_wavelength,n_wavelength), or the full "
                "(n_channels*n_wavelength,n_channels*n_wavelength) channel-major matrix"
            )
        for channel_index, block in enumerate(independent_discrepancy_blocks):
            if not np.allclose(block, block.T, rtol=1e-10, atol=1e-13):
                raise ValueError(f"model_discrepancy_covariance[{channel_index}] must be symmetric")
            eigenvalues = np.linalg.eigvalsh(block)
            scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
            if float(np.min(eigenvalues)) < -1e-10*scale:
                raise ValueError(f"model_discrepancy_covariance[{channel_index}] must be positive semidefinite")
        # Add per-channel discrepancy blocks to any supplied full measurement
        # covariance for every optical model, not only the relativistic fit.
        if (
            joint_covariance is not None
            and not discrepancy_added_to_joint_covariance
        ):
            for channel_index, block in enumerate(independent_discrepancy_blocks):
                channel_slice = slice(channel_index*nlam_disc, (channel_index+1)*nlam_disc)
                joint_covariance[channel_slice, channel_slice] += block
            discrepancy_added_to_joint_covariance = True
        if discrepancy_added_to_joint_covariance:
            try:
                joint_cholesky = np.linalg.cholesky(joint_covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "measurement covariance plus model_discrepancy_covariance "
                    "must be positive definite; a discrepancy-only covariance must "
                    "also be positive definite to define GLS weights"
                ) from exc
    independent_systematic_blocks = None
    systematic_added_to_joint_covariance = False
    if systematic_covariance is not None:
        if model not in {
            "non_collective_maxwellian", "non_collective_kappa",
            "non_collective_relativistic_maxwell_juttner", "collective_maxwellian",
            "collective_kappa",
        }:
            raise ValueError(
                "systematic_covariance is supported for non-collective optical Thomson "
                "models and the collective Maxwellian fit"
            )
        systematic = _finite_array(systematic_covariance, "systematic_covariance")
        nchan_sys, nlam_sys = y.shape
        if systematic.shape == (y.size, y.size):
            full_systematic_covariance = systematic.copy()
            if not np.allclose(
                full_systematic_covariance, full_systematic_covariance.T,
                rtol=1e-10, atol=1e-13,
            ):
                raise ValueError("full channel-major systematic_covariance must be symmetric")
            full_systematic_covariance = 0.5 * (
                full_systematic_covariance + full_systematic_covariance.T
            )
            eigenvalues = np.linalg.eigvalsh(full_systematic_covariance)
            scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
            if float(np.min(eigenvalues)) < -1e-10 * scale:
                raise ValueError("full channel-major systematic_covariance must be positive semidefinite")
            if model in {"collective_maxwellian", "collective_kappa"}:
                for row in range(nchan_sys):
                    for col in range(nchan_sys):
                        if row != col:
                            cross = full_systematic_covariance[
                                row*nlam_sys:(row+1)*nlam_sys,
                                col*nlam_sys:(col+1)*nlam_sys,
                            ]
                            if np.any(np.abs(cross) > 1e-13 * scale):
                                raise ValueError(
                                    f"{model} currently fits channels independently; "
                                    "cross-channel systematic covariance cannot be silently discarded"
                                )
            independent_systematic_blocks = np.stack([
                full_systematic_covariance[
                    channel_index*nlam_sys:(channel_index+1)*nlam_sys,
                    channel_index*nlam_sys:(channel_index+1)*nlam_sys,
                ]
                for channel_index in range(nchan_sys)
            ])
            if joint_covariance is None:
                if unc is not None:
                    joint_covariance = np.diag(np.asarray(unc, dtype=float).reshape(-1)**2)
                    unc = None
                else:
                    joint_covariance = np.zeros((y.size, y.size), dtype=float)
            joint_covariance += full_systematic_covariance
            systematic_added_to_joint_covariance = True
        elif systematic.shape == (nlam_sys, nlam_sys):
            independent_systematic_blocks = np.broadcast_to(
                systematic, (nchan_sys, nlam_sys, nlam_sys)
            ).copy()
        elif systematic.shape == (nchan_sys, nlam_sys, nlam_sys):
            independent_systematic_blocks = systematic.copy()
        else:
            raise ValueError(
                "systematic_covariance must be (n_wavelength,n_wavelength), "
                "(n_channels,n_wavelength,n_wavelength), or the full "
                "(n_channels*n_wavelength,n_channels*n_wavelength) channel-major matrix"
            )
        for channel_index, block in enumerate(independent_systematic_blocks):
            if not np.allclose(block, block.T, rtol=1e-10, atol=1e-13):
                raise ValueError(f"systematic_covariance[{channel_index}] must be symmetric")
            eigenvalues = np.linalg.eigvalsh(0.5 * (block + block.T))
            scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
            if float(np.min(eigenvalues)) < -1e-10 * scale:
                raise ValueError(
                    f"systematic_covariance[{channel_index}] must be positive semidefinite"
                )
        if joint_covariance is not None and not systematic_added_to_joint_covariance:
            for channel_index, block in enumerate(independent_systematic_blocks):
                channel_slice = slice(
                    channel_index*nlam_sys, (channel_index+1)*nlam_sys
                )
                joint_covariance[channel_slice, channel_slice] += block
            systematic_added_to_joint_covariance = True
        if systematic_added_to_joint_covariance:
            try:
                joint_cholesky = np.linalg.cholesky(joint_covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "measurement, model-discrepancy and systematic covariance sum "
                    "must be positive definite"
                ) from exc

    cross_channel_covariance_present = False
    if joint_covariance is not None and model == "non_collective_relativistic_maxwell_juttner":
        nchan_cov, nlam_cov = y.shape
        cross_channel_covariance_present = any(
            np.any(joint_covariance[
                row*nlam_cov:(row+1)*nlam_cov,
                col*nlam_cov:(col+1)*nlam_cov,
            ] != 0.0)
            for row in range(nchan_cov) for col in range(nchan_cov) if row != col
        )
    results = []
    channel_responses = []
    for i in range(y.shape[0]):
        xi = xs[i]
        response = _response_for_channel(instrument_response, i, y.shape[0], y.shape[1], xi)
        source_response = None
        source_profile_metadata = None
        if laser_spectrum_wavelength_nm is not None:
            if model == "non_collective_relativistic_maxwell_juttner":
                # Recoil changes the wavelength shift and Klein-Nishina weight
                # for every incident laser wavelength. The exact source-resolved
                # kernel is evaluated inside the nonlinear physical forward fit;
                # do not substitute the elastic wavelength-ratio operator.
                source_profile_metadata = {
                    "source_profile_samples": int(np.asarray(laser_spectrum_wavelength_nm).size),
                    "convolution_approximation": "wavelength_resolved_Compton_Klein_Nishina_forward_kernel",
                }
            else:
                source_response, source_profile_metadata = _laser_source_response(
                    xi, float(laser_wavelength_nm), laser_spectrum_wavelength_nm,
                    laser_spectrum_density_per_nm,
                )
                # Elastic source broadening precedes the spectrometer response.
                response = response @ source_response
        channel_responses.append(response)
        ui = None if unc is None else unc[i]
        channel_covariance = None
        if joint_covariance is not None:
            channel_slice = slice(i*y.shape[1], (i+1)*y.shape[1])
            channel_covariance = joint_covariance[channel_slice, channel_slice]
            ui = None
        if independent_discrepancy_blocks is not None and not discrepancy_added_to_joint_covariance:
            discrepancy_block = independent_discrepancy_blocks[i]
            if channel_covariance is not None:
                channel_covariance = channel_covariance + discrepancy_block
            elif ui is not None:
                channel_covariance = np.diag(np.asarray(ui, dtype=float)**2) + discrepancy_block
            else:
                channel_covariance = discrepancy_block
            ui = None
        if independent_systematic_blocks is not None and not systematic_added_to_joint_covariance:
            systematic_block = independent_systematic_blocks[i]
            if channel_covariance is not None:
                channel_covariance = channel_covariance + systematic_block
            elif ui is not None:
                channel_covariance = np.diag(np.asarray(ui, dtype=float)**2) + systematic_block
            else:
                channel_covariance = systematic_block
            ui = None
        if model == "non_collective_maxwellian":
            fit = _fit_noncollective(xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad), response, ui,
                                background_model=background_model, background_degree=background_degree,
                                background_interior_knots=background_interior_knots,
                                background_templates=(background_templates[i] if np.asarray(background_templates).ndim == 3 else background_templates) if background_templates is not None else None,
                                spectral_covariance=channel_covariance)
        elif model == "non_collective_kappa":
            fit = _fit_kappa(
                xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad),
                response, ui, spectral_covariance=channel_covariance,
                background_model=background_model, background_degree=background_degree,
                background_interior_knots=background_interior_knots,
                background_templates=(
                    background_templates[i]
                    if background_templates is not None and np.asarray(background_templates).ndim == 3
                    else background_templates
                ),
            )
        elif model == "non_collective_relativistic_maxwell_juttner":
            # First obtain a deterministic physical seed with the established
            # relativistic fitter, then refit the measured channel with the
            # selected detector-space background basis in the same GLS problem.
            seed_fit = fit_relativistic_maxwell_juttner(
                xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad),
                response, channel_covariance if channel_covariance is not None else ui,
                incident_direction=probe_vec, scattered_direction=scatter_vec,
                incident_polarization=incident_polarization,
                maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
                include_compton_recoil=include_compton_recoil,
            )
            from .thomson_kinetic_background import (
                fit_noncollective_kinetic_thomson_with_joint_background,
            )
            channel_background_templates = (
                background_templates[i]
                if background_templates is not None and np.asarray(background_templates).ndim == 3
                else background_templates
            )
            joint_background_fit = fit_noncollective_kinetic_thomson_with_joint_background(
                xi, y[i],
                physical_model="relativistic_maxwell_juttner",
                laser_wavelength_nm=float(laser_wavelength_nm),
                scattering_angle_rad=float(scattering_angle_rad),
                instrument_response=response,
                physical_initial=[
                    float(seed_fit["electron_temperature_eV"]),
                    float(seed_fit["electron_velocity_projection_m_s"]),
                ],
                background_model=background_model,
                background_degree=background_degree,
                background_interior_knots=background_interior_knots,
                background_templates=channel_background_templates,
                measurement_uncertainty=(ui if channel_covariance is None else None),
                measurement_covariance=channel_covariance,
                incident_polarization=incident_polarization,
                maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
                include_compton_recoil=include_compton_recoil,
                laser_spectrum_wavelength_nm=(
                    laser_spectrum_wavelength_nm
                    if source_profile_metadata is not None
                    and model == "non_collective_relativistic_maxwell_juttner"
                    and include_compton_recoil else None
                ),
                laser_spectrum_density_per_nm=(
                    laser_spectrum_density_per_nm
                    if source_profile_metadata is not None
                    and model == "non_collective_relativistic_maxwell_juttner"
                    and include_compton_recoil else None
                ),
            )
            fit = dict(seed_fit)
            fit.update(joint_background_fit)
            physical_covariance = np.asarray(
                joint_background_fit["physical_parameter_covariance"], dtype=float
            )
            joint_parameter_covariance = np.asarray(
                joint_background_fit["joint_parameter_covariance"], dtype=float
            )
            fit.update({
                "electron_temperature_eV": float(joint_background_fit["physical_parameters"][0]),
                "electron_velocity_projection_m_s": float(joint_background_fit["physical_parameters"][1]),
                "electron_temperature_uncertainty_eV": float(
                    np.sqrt(max(physical_covariance[0, 0], 0.0))
                ),
                "electron_velocity_projection_uncertainty_m_s": float(
                    np.sqrt(max(physical_covariance[1, 1], 0.0))
                ),
                "fit_covariance": joint_parameter_covariance,
                "fit_covariance_parameter_order": list(
                    joint_background_fit["joint_parameter_order"]
                ),
                "fit_covariance_includes_background_cross_terms": True,
                "background_fit_method": "joint_nonlinear_physical_and_linear_nuisance_fit",
                "background_covariance_scope": (
                    "full local covariance including physical-amplitude-background "
                    "cross terms, conditional on the supplied measured response and "
                    "spectral noise covariance"
                ),
            })
            # The absolute-power integral must correspond to the refitted
            # Thomson component, not the seed fit. Propagate its uncertainty
            # through Te, projected flow and fitted amplitude using the complete
            # joint covariance; background coefficients do not enter this integral.
            from .thomson_kinetic_models import (
                relativistic_maxwell_juttner_spectrum as _mj_forward,
                _apply_response as _mj_apply_response,
            )
            physical_parameters = np.asarray(
                joint_background_fit["physical_parameters"], dtype=float
            )
            fitted_amplitude = float(joint_background_fit["amplitude"])
            def _integrated_power_at(parameters):
                physical = _mj_forward(
                    xi, float(laser_wavelength_nm), float(scattering_angle_rad),
                    float(parameters[0]),
                    flow_projection_m_s=float(parameters[1]),
                    incident_polarization=np.asarray(incident_polarization, dtype=float),
                    maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
                    include_compton_recoil=include_compton_recoil,
                    require_full_spectrum=False,
                )
                convolved_shape = _mj_apply_response(
                    response, np.asarray(physical["spectral_density_per_nm"], dtype=float), xi
                )
                return (
                    float(parameters[2]) * float(trapezoid(convolved_shape, xi)),
                    float(physical["effective_differential_cross_section_m2_sr"]),
                )
            power_point = np.r_[physical_parameters, fitted_amplitude]
            integrated_power, effective_cross_section = _integrated_power_at(power_point)
            power_gradient = np.zeros(joint_parameter_covariance.shape[0], dtype=float)
            for parameter_index in range(3):
                step = np.cbrt(np.finfo(float).eps) * max(
                    abs(float(power_point[parameter_index])), 1.0
                )
                plus, minus = power_point.copy(), power_point.copy()
                plus[parameter_index] += step
                minus[parameter_index] -= step
                if parameter_index == 0 and minus[0] <= 0.0:
                    minus[0] = power_point[0]
                    derivative = (
                        _integrated_power_at(plus)[0] - integrated_power
                    ) / step
                elif parameter_index == 2 and minus[2] <= 0.0:
                    minus[2] = power_point[2]
                    derivative = (
                        _integrated_power_at(plus)[0] - integrated_power
                    ) / step
                else:
                    derivative = (
                        _integrated_power_at(plus)[0] - _integrated_power_at(minus)[0]
                    ) / (plus[parameter_index] - minus[parameter_index])
                power_gradient[parameter_index] = derivative
            cross_section_gradient = np.zeros(
                joint_parameter_covariance.shape[0], dtype=float
            )
            for parameter_index in (0, 1):
                step = np.cbrt(np.finfo(float).eps) * max(
                    abs(float(power_point[parameter_index])), 1.0
                )
                plus, minus = power_point.copy(), power_point.copy()
                plus[parameter_index] += step
                minus[parameter_index] -= step
                if parameter_index == 0 and minus[0] <= 0.0:
                    minus[0] = power_point[0]
                    derivative = (
                        _integrated_power_at(plus)[1] - effective_cross_section
                    ) / step
                else:
                    derivative = (
                        _integrated_power_at(plus)[1] - _integrated_power_at(minus)[1]
                    ) / (plus[parameter_index] - minus[parameter_index])
                cross_section_gradient[parameter_index] = derivative
            power_variance = float(
                power_gradient @ joint_parameter_covariance @ power_gradient
            )
            cross_section_variance = float(
                cross_section_gradient @ joint_parameter_covariance @ cross_section_gradient
            )
            power_cross_section_covariance = float(
                power_gradient @ joint_parameter_covariance @ cross_section_gradient
            )
            fit["integrated_scattered_power_W"] = integrated_power
            fit["integrated_scattered_power_uncertainty_W"] = float(
                np.sqrt(max(power_variance, 0.0))
            )
            fit["effective_differential_cross_section_m2_sr"] = effective_cross_section
            fit["effective_differential_cross_section_uncertainty_m2_sr"] = float(
                np.sqrt(max(cross_section_variance, 0.0))
            )
            fit["power_cross_section_relative_covariance"] = (
                power_cross_section_covariance / (integrated_power * effective_cross_section)
                if integrated_power > 0.0 and effective_cross_section > 0.0 else None
            )
            fit["integrated_power_uncertainty_method"] = (
                "finite_difference_delta_method_from_full_joint_physical_amplitude_background_covariance"
            )
        elif model == "collective_kappa":
            fit = fit_collective_kappa(
                xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad),
                response, ui, incident_direction=probe_vec, scattered_direction=scatter_vec,
                ion_species=ion_species, ion_fractions=ion_fract,
                ion_temperature_eV=ion_t, electron_velocity_m_s=electron_v[0],
                ion_velocity_m_s=ion_v,
                initial_density_m3=collective_initial_density_m3,
                initial_temperature_eV=collective_initial_temperature_eV,
                density_bounds_m3=density_bounds, temperature_bounds_eV=temperature_bounds,
                initial_kappa=collective_kappa_initial, kappa_bounds=collective_kappa_bounds,
                spectral_covariance=channel_covariance,
            )
        elif model == "collective_maxwellian":
            # PlasmaPy's kinetic Maxwellian spectral-density model evaluates
            # collective and non-collective regimes from the same dielectric
            # response; ion species/temperature are never silently defaulted.
            ion_t = _finite_array(np.atleast_1d(ion_temperature_eV), "ion_temperature_eV", 1)
            if ion_t.size != len(ion_species) or np.any(ion_t <= 0):
                raise ValueError("ion_temperature_eV must contain one positive temperature per ion species")
            ion_fract = _finite_array(ion_fractions, "ion_fractions", 1)
            if ion_fract.size != len(ion_species) or np.any(ion_fract <= 0) or not np.isclose(np.sum(ion_fract),1.0,rtol=1e-6,atol=1e-8):
                raise ValueError("ion_fractions must contain one positive fraction per species and sum to one")
            ion_v = _finite_array(ion_velocity_m_s, "ion_velocity_m_s")
            electron_v = _finite_array(electron_velocity_m_s, "electron_velocity_m_s")
            if ion_v.shape == (3,) and len(ion_species) == 1:
                ion_v = ion_v.reshape(1,3)
            if electron_v.shape == (3,):
                electron_v = electron_v.reshape(1,3)
            if electron_v.shape != (1, 3):
                raise ValueError("electron_velocity_m_s must be a 3-vector")
            if ion_v.shape != (len(ion_species), 3):
                raise ValueError("ion_velocity_m_s must have shape (number_of_ion_species, 3)")
            if np.any(np.linalg.norm(ion_v,axis=1) >= C_M_S) or np.linalg.norm(electron_v[0]) >= C_M_S:
                raise ValueError("Collective ion/electron flow speeds must be subluminal")
            if ion_temperature_uncertainty_eV is not None:
                ion_t_unc=_finite_array(np.atleast_1d(ion_temperature_uncertainty_eV),"ion_temperature_uncertainty_eV",1)
                if ion_t_unc.shape!=ion_t.shape or np.any(ion_t_unc<0):
                    raise ValueError("ion_temperature_uncertainty_eV must be non-negative and match ion species")
            else: ion_t_unc=None
            if ion_velocity_uncertainty_m_s is not None:
                ion_v_unc=_finite_array(ion_velocity_uncertainty_m_s,"ion_velocity_uncertainty_m_s")
                if ion_v_unc.shape==(3,) and len(ion_species)==1: ion_v_unc=ion_v_unc.reshape(1,3)
                if ion_v_unc.shape!=ion_v.shape or np.any(ion_v_unc<0):
                    raise ValueError("ion_velocity_uncertainty_m_s must be non-negative and match ion velocity vectors")
            else: ion_v_unc=None
            if electron_velocity_uncertainty_m_s is not None:
                electron_v_unc=_finite_array(electron_velocity_uncertainty_m_s,"electron_velocity_uncertainty_m_s",1)
                if electron_v_unc.shape!=(3,) or np.any(electron_v_unc<0):
                    raise ValueError("electron_velocity_uncertainty_m_s must be a non-negative 3-vector")
            else: electron_v_unc=None
            # Fit ne and Te against the full PlasmaPy spectrum. Absolute signal
            # scaling is a nuisance amplitude; density is identifiable from
            # the collective spectral shape when the spectrum resolves it.
            from scipy.optimize import least_squares
            def collective_shape(params):
                ne, te = params
                _, spec = thomson.spectral_density(
                    xi*u.nm, float(laser_wavelength_nm)*u.nm, ne/u.m**3,
                    T_e=te*u.eV, T_i=ion_t*u.eV, ions=ion_species, ifract=ion_fract,
                    ion_vel=ion_v*u.m/u.s, electron_vel=electron_v*u.m/u.s,
                    probe_vec=probe_vec, scatter_vec=scatter_vec)
                z = np.asarray(spec.value, dtype=float)
                return _apply_response(response, z, xi)
            whiten, noise_model = _collective_spectral_whitener(
                y[i], uncertainty=ui if channel_covariance is None else None,
                covariance=channel_covariance,
            )
            def fit_nuisance(shape):
                shape_scale=max(float(np.max(np.abs(shape))),np.finfo(float).tiny)
                design=np.column_stack((shape/shape_scale,np.ones_like(shape),xi-float(laser_wavelength_nm)))
                weighted_design=whiten(design)
                weighted_signal=whiten(y[i])
                coefficients=np.linalg.lstsq(weighted_design,weighted_signal,rcond=None)[0]
                if coefficients[0] < 0:
                    background_design=whiten(design[:,1:])
                    background=np.linalg.lstsq(background_design,weighted_signal,rcond=None)[0]
                    coefficients=np.array([0.0,background[0],background[1]])
                coefficients[0] /= shape_scale
                return coefficients
            def objective(params):
                shape = collective_shape(params)
                coefficients=fit_nuisance(shape)
                model_signal=coefficients[0]*shape+coefficients[1]+coefficients[2]*(xi-float(laser_wavelength_nm))
                return whiten(model_signal-y[i])
            fitopt = least_squares(
                lambda log_params: objective(10.0**log_params),
                np.log10([collective_initial_density_m3, collective_initial_temperature_eV]),
                bounds=(np.log10([density_bounds[0], temperature_bounds[0]]),
                        np.log10([density_bounds[1], temperature_bounds[1]])),
                x_scale="jac", max_nfev=5000,
            )
            ne_fit, te_fit = map(float, 10.0**fitopt.x)
            # Two nonlinear plasma parameters plus three profiled linear nuisance parameters
            # (amplitude, background intercept, and background slope) are estimated.
            dof = max(1, y[i].size-5)
            jacobian_rank = int(np.linalg.matrix_rank(fitopt.jac))
            jacobian_condition = float(np.linalg.cond(fitopt.jac))
            try:
                covariance_log = np.linalg.pinv(fitopt.jac.T @ fitopt.jac)
                if noise_model == "relative_unweighted":
                    covariance_log *= (2.0*fitopt.cost/dof)
                transform = np.diag(np.log(10.0)*np.array([ne_fit, te_fit]))
                parameter_covariance = transform @ covariance_log @ transform.T
                parameter_uncertainties = np.sqrt(np.maximum(np.diag(parameter_covariance),0.0))
            except np.linalg.LinAlgError:
                parameter_covariance = None
                parameter_uncertainties = np.array([np.nan,np.nan])
            # fitopt.x is in log10 space; the forward model requires physical ne and Te.
            model_spec = collective_shape(10.0**fitopt.x)
            alpha_fit, _ = thomson.spectral_density(
                xi*u.nm, float(laser_wavelength_nm)*u.nm, ne_fit/u.m**3,
                T_e=te_fit*u.eV, T_i=ion_t*u.eV, ions=ion_species, ifract=ion_fract,
                ion_vel=ion_v*u.m/u.s, electron_vel=electron_v*u.m/u.s,
                probe_vec=probe_vec, scatter_vec=scatter_vec)
            alpha_value=float(np.asarray(getattr(alpha_fit,"value",alpha_fit)))
            # PlasmaPy's scattering parameter follows alpha = 1/(|k_s-k_i| lambda_D),
            # hence alpha scales as sqrt(n_e/T_e) * lambda_L / sin(theta/2).
            # Propagate the complete fitted (n_e,T_e) covariance, including its
            # off-diagonal term, plus independently supplied laser/angle errors.
            alpha_gradient_ne_te = np.array([
                alpha_value / (2.0 * ne_fit),
                -alpha_value / (2.0 * te_fit),
            ], dtype=float)
            alpha_variance = None
            if parameter_covariance is not None or laser_wavelength_unc > 0.0 or scattering_angle_unc > 0.0:
                alpha_variance = 0.0
                if parameter_covariance is not None:
                    alpha_variance += float(
                        alpha_gradient_ne_te @ parameter_covariance @ alpha_gradient_ne_te
                    )
                alpha_variance += (
                    alpha_value * laser_wavelength_unc / float(laser_wavelength_nm)
                ) ** 2
                if scattering_angle_unc > 0.0:
                    alpha_angle_sensitivity = -0.5 * alpha_value / np.tan(
                        float(scattering_angle_rad) / 2.0
                    )
                    alpha_variance += (
                        alpha_angle_sensitivity * scattering_angle_unc
                    ) ** 2
                alpha_variance = max(alpha_variance, 0.0)
            alpha_uncertainty = (
                float(np.sqrt(alpha_variance)) if alpha_variance is not None else None
            )
            amp_fit, background_fit, background_slope_fit = map(float,fit_nuisance(model_spec))
            fitted_collective = amp_fit*model_spec+background_fit+background_slope_fit*(xi-float(laser_wavelength_nm))
            residual_collective = y[i]-fitted_collective
            q_vector = scatter_vec-probe_vec
            q_hat = q_vector/np.linalg.norm(q_vector)
            electron_flow_projection = float(np.dot(electron_v[0],q_hat))
            ion_flow_projections = np.asarray(ion_v @ q_hat,dtype=float)
            fit = {"electron_density_m3": ne_fit, "electron_temperature_eV": te_fit,
                   "electron_velocity_projection_m_s": electron_flow_projection,
                   "ion_velocity_projection_m_s": ion_flow_projections.tolist(),
                   "electron_flow_source": "explicitly supplied electron velocity vector",
                   "fitted_spectrum": fitted_collective, "residual": residual_collective,
                   "amplitude": amp_fit, "background_intercept": background_fit,
                   "background_slope_per_nm": background_slope_fit,
                   "model": "PlasmaPy Maxwellian kinetic dielectric spectral-density model",
                   "electron_density_uncertainty_m3": float(parameter_uncertainties[0]),
                   "electron_temperature_uncertainty_eV": float(parameter_uncertainties[1]),
                   "fit_covariance": parameter_covariance,
                   "optimizer_success": bool(fitopt.success), "optimizer_message": str(fitopt.message),
                   "collective_parameter_jacobian_rank": jacobian_rank,
                   "collective_parameter_jacobian_condition_number": jacobian_condition,
                   "reduced_chi_square": float((2.0 * fitopt.cost) / dof) if noise_model != "relative_unweighted" else None,
                   "spectral_noise_model": noise_model,
                   "spectral_covariance_used_in_fit": bool(channel_covariance is not None),
                   "fit_objective": "generalized_least_squares" if noise_model == "full_covariance" else ("weighted_least_squares" if noise_model == "pointwise_uncertainty" else "relative_unweighted_least_squares"),
                   "ion_species": list(ion_species), "ion_fractions": ion_fract.tolist(),
                   "scattering_parameter_alpha": alpha_value,
                   "scattering_parameter_alpha_uncertainty": alpha_uncertainty,
                   "scattering_parameter_alpha_gradient_ne_te": alpha_gradient_ne_te,
                   "scattering_parameter_alpha_laser_sensitivity_per_nm": float(alpha_value / float(laser_wavelength_nm)),
                   "scattering_parameter_alpha_angle_sensitivity_per_rad": float(
                       -0.5 * alpha_value / np.tan(float(scattering_angle_rad) / 2.0)
                   ),
                   "scattering_parameter_alpha_laser_uncertainty_contribution": float(
                       alpha_value * laser_wavelength_unc / float(laser_wavelength_nm)
                   ),
                   "scattering_parameter_alpha_angle_uncertainty_contribution": float(
                       -0.5 * alpha_value / np.tan(float(scattering_angle_rad) / 2.0)
                       * scattering_angle_unc
                   ),
                   "scattering_parameter_alpha_uncertainty_model": (
                       "first-order propagation of the full fitted ne/Te covariance plus supplied independent laser-wavelength and scalar scattering-angle uncertainties"
                       if alpha_uncertainty is not None else "not_available_without_finite_fit_covariance"
                   ),
                   "scattering_regime": "collective" if alpha_value > 1.0 else "non_collective_or_weakly_collective",
                   "ion_temperature_eV": np.atleast_1d(ion_t).tolist(),
                   "ion_velocity_m_s": ion_v.tolist(), "electron_velocity_m_s": electron_v.tolist()}
            if not fitopt.success:
                fit["electron_density_m3"] = None
                fit["electron_temperature_eV"] = None
                fit["electron_density_uncertainty_m3"] = None
                fit["electron_temperature_uncertainty_eV"] = None
                fit["density_status"] = "collective_spectral_optimizer_failed"
            elif jacobian_rank < 2:
                fit["electron_density_m3"] = None
                fit["electron_temperature_eV"] = None
                fit["electron_density_uncertainty_m3"] = None
                fit["electron_temperature_uncertainty_eV"] = None
                fit["density_status"] = "collective_density_temperature_not_jointly_identifiable"
                fit["temperature_status"] = "collective_density_temperature_not_jointly_identifiable"
            elif alpha_value <= 1.0:
                fit["electron_density_m3"] = None
                fit["electron_density_uncertainty_m3"] = None
                fit["density_status"] = "spectral_shape_does_not_resolve_collective_density"
        expected_parameter_count = (
            int(np.asarray(fit["fit_covariance"]).shape[0])
            if model == "non_collective_relativistic_maxwell_juttner"
            and fit.get("fit_covariance") is not None
            else (
                2 if model == "collective_maxwellian" else
                3 if model == "collective_kappa" else
                6 if model == "non_collective_kappa" else 5
            )
        )
        quality = _fit_identifiability(
            fit.get("fit_covariance"), fit.get("residual"), ui, expected_parameter_count,
            optimizer_success=fit.get("optimizer_success", True),
            jacobian_rank=(fit.get("collective_parameter_jacobian_rank")
                           if model in {"collective_maxwellian", "collective_kappa"} else None),
            jacobian_condition_number=(fit.get("collective_parameter_jacobian_condition_number")
                                       if model in {"collective_maxwellian", "collective_kappa"} else None),
        )
        fit.update(quality)
        fit_covariance_for_discrepancy=fit.get("fit_covariance")
        if independent_discrepancy_blocks is None and fit.get("fit_reduced_chi_square") is not None and fit_covariance_for_discrepancy is not None:
            discrepancy_scale=max(float(fit["fit_reduced_chi_square"])-1.0,0.0)
            discrepancy_covariance=discrepancy_scale*np.asarray(fit_covariance_for_discrepancy,dtype=float)
            fit["model_discrepancy_scale"]=discrepancy_scale
            fit["model_discrepancy_covariance"]=discrepancy_covariance
            fit["total_fit_covariance_including_discrepancy"]=np.asarray(fit_covariance_for_discrepancy,dtype=float)+discrepancy_covariance
            fit["model_discrepancy_diagnostics"]={
                "residual_rms":fit.get("fit_residual_rms"),
                "reduced_chi_square":fit.get("fit_reduced_chi_square"),
                "interpretation":"Excess reduced chi-square above unity is reported as a separate model-discrepancy covariance component conditional on the supplied noise model; it is not experimental validation.",
            }
            total_sigma=np.sqrt(np.maximum(np.diag(fit["total_fit_covariance_including_discrepancy"]),0.0))
            if model=="collective_maxwellian" and total_sigma.size>=2:
                if fit.get("electron_density_m3") is not None:
                    fit["electron_density_uncertainty_m3"]=float(total_sigma[0])
                fit["electron_temperature_uncertainty_eV"]=float(total_sigma[1])
            elif model=="collective_kappa" and total_sigma.size>=3:
                fit["electron_density_uncertainty_m3"]=float(total_sigma[0])
                fit["electron_temperature_uncertainty_eV"]=float(total_sigma[1])
                fit["kappa_index_uncertainty"]=float(total_sigma[2])
            elif model=="non_collective_kappa":
                fit["electron_temperature_uncertainty_eV"]=float(np.hypot(
                    fit["electron_temperature_uncertainty_eV"],
                    2.0*fit["electron_temperature_eV"]*total_sigma[2]/max(float(fit["intrinsic_sigma_nm"]),np.finfo(float).tiny)))
                qproj=2.0*np.sin(float(scattering_angle_rad)/2.0)
                velocity_discrepancy=C_M_S*total_sigma[1]/(float(laser_wavelength_nm)*qproj)
                fit["electron_velocity_projection_uncertainty_m_s"]=float(np.hypot(
                    fit["electron_velocity_projection_uncertainty_m_s"],velocity_discrepancy))
            elif model=="non_collective_maxwellian":
                fit["electron_temperature_uncertainty_eV"]=float(np.hypot(
                    fit["electron_temperature_uncertainty_eV"],
                    2.0*fit["electron_temperature_eV"]*total_sigma[2]/max(float(fit["intrinsic_sigma_nm"]),np.finfo(float).tiny)))
                qproj=2.0*np.sin(float(scattering_angle_rad)/2.0)
                velocity_discrepancy=C_M_S*total_sigma[1]/(float(laser_wavelength_nm)*qproj)
                fit["electron_velocity_projection_uncertainty_m_s"]=float(np.hypot(
                    fit["electron_velocity_projection_uncertainty_m_s"],velocity_discrepancy))
        if not fit["fit_accepted"]:
            raise ValueError(f"Thomson fit rejected for channel {i}: {'; '.join(fit['fit_quality_reasons'])}")
        if model in {"collective_maxwellian", "collective_kappa"}:
            fit["parameter_provenance"] = {
                "electron_density": "inferred from the measured collective spectrum",
                "electron_temperature": "inferred from the measured collective spectrum",
                "electron_distribution": ("kappa index inferred from the measured collective spectrum" if model == "collective_kappa" else "Maxwellian"),
                "ion_temperature": "externally supplied fixed input; not inferred",
                "ion_velocity": "externally supplied fixed input; not inferred",
                "electron_velocity": "externally supplied fixed input; not inferred",
                "ion_species_and_fractions": "externally supplied fixed composition",
            }
        else:
            fit["parameter_provenance"] = {
                "electron_temperature": "inferred from measured spectral shape under the selected non-collective model",
                "electron_velocity_projection": "inferred from measured spectral centroid under the selected Doppler model",
                "electron_density": "not identifiable from normalized spectral shape; requires absolute calibrated scattered power and optical geometry",
                "ion_parameters": "not inferred by this non-collective electron feature model",
            }
        if independent_discrepancy_blocks is not None:
            fit["independent_model_discrepancy_covariance"] = independent_discrepancy_blocks[i].copy()
            fit["independent_model_discrepancy_supplied"] = True
            fit["model_discrepancy_model"] = "researcher_supplied_correlated_covariance_independent_of_reduced_chi_square"
            fit["model_discrepancy_covariance_included"] = True
        else:
            fit["independent_model_discrepancy_covariance"] = None
            fit["independent_model_discrepancy_supplied"] = False
        fit["model_discrepancy_uncertainty_available"] = bool(
            fit.get("model_discrepancy_covariance") is not None
            and float(fit.get("model_discrepancy_scale", 0.0)) > 0.0
        ) or independent_discrepancy_blocks is not None
        if independent_discrepancy_blocks is not None:
            fit["model_discrepancy_diagnostics"] = {
                "residual_rms": fit.get("fit_residual_rms"),
                "covariance_included": True,
                "model": "researcher_supplied_correlated_spectral_covariance",
                "independent_of_reduced_chi_square": True,
                "interpretation": "The supplied correlated discrepancy covariance was added to the channel spectral covariance before GLS fitting; no reduced-chi-square rescaling was applied.",
            }
        elif fit.get("model_discrepancy_covariance") is None:
            fit["model_discrepancy_diagnostics"] = {
                "residual_rms": fit.get("fit_residual_rms"),
                "reduced_chi_square": fit.get("fit_reduced_chi_square"),
                "covariance_included": False,
                "interpretation": "No per-channel model-discrepancy covariance is available because a supported reduced chi-square/noise covariance estimate was not produced.",
            }
        else:
            fit["model_discrepancy_diagnostics"]["covariance_included"] = fit[
                "model_discrepancy_uncertainty_available"
            ]
        density_inputs = None
        if model in {"non_collective_maxwellian", "non_collective_kappa", "non_collective_relativistic_maxwell_juttner"} and all(v is not None for v in (
            laser_power_w, interaction_length_m, collection_solid_angle_sr, optical_efficiency,
            incident_polarization, incident_direction, scattered_direction
        )):
            if not spectral_power_density_calibrated:
                raise ValueError("Absolute density requires spectral_power_density_calibrated=True and signal units W/nm")
            area = float(fit["integrated_scattered_power_W"])
            area_sigma = fit.get("integrated_scattered_power_uncertainty_W")
            relative_input_covariance = (
                joint_density_covariance[6*i:6*(i+1), 6*i:6*(i+1)].copy()
                if joint_density_covariance is not None
                else _channel_covariance(density_relative_covariance, i, y.shape[0], "density_relative_covariance")
            )
            power_uncertainty_in_covariance = (
                relative_input_covariance is not None
                and not np.allclose(relative_input_covariance[0, :], 0.0, rtol=0.0, atol=1e-14)
            )
            density_inputs = {
                "scattered_power_W": area, "incident_power_W": _channel_value(laser_power_w, i, y.shape[0], "laser_power_w"),
                "interaction_length_m": _channel_value(interaction_length_m, i, y.shape[0], "interaction_length_m"),
                "scattering_angle_rad": scattering_angle_rad,
                "collection_solid_angle_sr": _channel_value(collection_solid_angle_sr, i, y.shape[0], "collection_solid_angle_sr"),
                "optical_efficiency": _channel_value(optical_efficiency, i, y.shape[0], "optical_efficiency"),
                # A supplied relative covariance is the complete uncertainty model
                # for the six multiplicative density inputs. Do not also pass the
                # fitted area sigma or independent scalar terms: the density helper
                # correctly rejects mixed covariance/scalar uncertainty inputs.
                "scattered_power_uncertainty_W": (None if power_uncertainty_in_covariance else area_sigma),
                "incident_power_uncertainty_W": (None if density_relative_covariance is not None or joint_density_covariance is not None else _channel_value(laser_power_uncertainty_W, i, y.shape[0], "laser_power_uncertainty_W")),
                "interaction_length_uncertainty_m": (None if density_relative_covariance is not None or joint_density_covariance is not None else _channel_value(interaction_length_uncertainty_m, i, y.shape[0], "interaction_length_uncertainty_m")),
                "scattering_angle_uncertainty_rad": (None if scattering_angle_uncertainty_rad is None else scattering_angle_unc),
                "collection_solid_angle_uncertainty_sr": (None if density_relative_covariance is not None or joint_density_covariance is not None else _channel_value(collection_solid_angle_uncertainty_sr, i, y.shape[0], "collection_solid_angle_uncertainty_sr")),
                "optical_efficiency_uncertainty": (None if density_relative_covariance is not None or joint_density_covariance is not None else _channel_value(optical_efficiency_uncertainty, i, y.shape[0], "optical_efficiency_uncertainty")),
                "incident_polarization_vector": _channel_value(incident_polarization, i, y.shape[0], "incident_polarization", vector_length=3),
                "incident_direction_vector": _channel_value(incident_direction, i, y.shape[0], "incident_direction", vector_length=3),
                "scattered_direction_vector": _channel_value(scattered_direction, i, y.shape[0], "scattered_direction", vector_length=3),
                "polarization_factor_uncertainty": (None if density_relative_covariance is not None or joint_density_covariance is not None else _channel_value(polarization_factor_uncertainty, i, y.shape[0], "polarization_factor_uncertainty")),
                "relative_input_covariance": relative_input_covariance,
            }
            if model == "non_collective_relativistic_maxwell_juttner":
                density_inputs.update({
                    "effective_differential_cross_section_m2_sr": fit["effective_differential_cross_section_m2_sr"],
                    "effective_differential_cross_section_uncertainty_m2_sr": fit.get("effective_differential_cross_section_uncertainty_m2_sr"),
                    "power_cross_section_relative_covariance": fit.get("power_cross_section_relative_covariance"),
                    "scattering_angle_uncertainty_rad": None,
                    "incident_polarization_vector": None,
                    "incident_direction_vector": None,
                    "scattered_direction_vector": None,
                    "polarization_factor_uncertainty": None,
                })
            density_result = density_from_thomson_scattered_power(**density_inputs)
            fit["electron_density_m3"] = density_result["electron_density_m3"]
            fit["electron_density_uncertainty_m3"] = density_result["electron_density_uncertainty_m3"]
            fit["integrated_scattered_power_W"] = area
            fit["integrated_scattered_power_uncertainty_W"] = area_sigma
            fit["differential_cross_section_m2_sr"] = density_result["differential_cross_section_m2_sr"]
            fit["scattering_angle_relative_sensitivity"] = density_result.get(
                "scattering_angle_relative_sensitivity"
            )
            fit["density_method"] = (
                "absolute integrated relativistic Thomson power using distribution-averaged effective cross-section"
                if model == "non_collective_relativistic_maxwell_juttner"
                else "absolute integrated Thomson power using shared polarization-aware density inversion"
            )
            fit["density_uncertainty_model"] = density_result["uncertainty_model"]
            if model == "non_collective_relativistic_maxwell_juttner":
                fit["differential_cross_section_m2_sr"] = density_result["differential_cross_section_m2_sr"]
                fit["differential_cross_section_uncertainty_m2_sr"] = fit.get("effective_differential_cross_section_uncertainty_m2_sr")
            if density_inputs.get("relative_input_covariance") is not None:
                fit["integrated_scattered_power_fit_uncertainty_W"] = float(
                    fit.get("integrated_scattered_power_fit_uncertainty_W", area_sigma)
                )
                covariance_area_sigma = abs(float(area)) * float(np.sqrt(max(
                    np.asarray(density_inputs["relative_input_covariance"], dtype=float)[0, 0], 0.0
                )))
                fit["integrated_scattered_power_covariance_uncertainty_W"] = covariance_area_sigma
                fit["integrated_scattered_power_uncertainty_W"] = covariance_area_sigma
            # Retain the exact, already validated physical inputs so a later
            # joint spectral fit can recompute density from its own final area.
            fit["_density_inputs_for_joint_refit"] = dict(density_inputs)
        else:
            fit["electron_density_m3"] = fit.get("electron_density_m3")
            if "density_status" not in fit:
                fit["density_status"] = "not_identifiable_from_spectral_shape_alone"
        fit["spectral_weighting_status"] = (
            "within_channel_marginal_gls_cross_channel_covariance_used_in_joint_shared_parameter_fit"
            if cross_channel_covariance_present else
            "block_diagonal_channel_major_within_channel_generalized_least_squares"
            if joint_covariance is not None and model == "non_collective_relativistic_maxwell_juttner" else
            "full_channel_major_generalized_least_squares"
            if joint_covariance is not None else
            "within_channel_gls_with_measurement_and_model_discrepancy_covariance"
            if channel_covariance is not None and measurement_covariance_supplied and model_discrepancy_covariance_supplied else
            "within_channel_gls_with_model_discrepancy_covariance_only"
            if channel_covariance is not None and model_discrepancy_covariance_supplied else
            "diagonal_pointwise_measurement_uncertainty_weighted"
            if ui is not None else
            "unweighted_least_squares_no_measurement_uncertainty_supplied"
        )
        fit["fit_covariance_interpretation"] = (
            "local_parameter_covariance_from_supplied_measurement_and_model_discrepancy_covariance"
            if measurement_covariance_supplied and model_discrepancy_covariance_supplied else
            "local_parameter_covariance_from_supplied_measurement_noise_model"
            if measurement_covariance_supplied else
            "local_parameter_covariance_from_supplied_model_discrepancy_covariance_only"
            if model_discrepancy_covariance_supplied else
            "residual_variance_scaled_local_covariance_not_measurement_noise_calibrated"
        )
        fit["systematic_covariance_supplied"] = systematic_covariance_supplied
        fit["systematic_covariance_included"] = independent_systematic_blocks is not None
        fit["systematic_covariance_model"] = (
            "researcher_supplied_additive_correlated_spectral_systematic_covariance"
            if systematic_covariance_supplied else None
        )
        if systematic_covariance_supplied:
            if measurement_covariance_supplied and model_discrepancy_covariance_supplied:
                fit["fit_covariance_interpretation"] = (
                    "local_parameter_covariance_from_measurement_model_discrepancy_"
                    "and_systematic_covariance_components"
                )
            elif measurement_covariance_supplied:
                fit["fit_covariance_interpretation"] = (
                    "local_parameter_covariance_from_measurement_and_systematic_covariance"
                )
            elif model_discrepancy_covariance_supplied:
                fit["fit_covariance_interpretation"] = (
                    "local_parameter_covariance_from_model_discrepancy_and_systematic_covariance"
                )
            else:
                fit["fit_covariance_interpretation"] = (
                    "local_parameter_covariance_from_supplied_systematic_covariance_only"
                )
            fit["spectral_weighting_status"] = (
                "full_channel_major_generalized_least_squares_with_systematic_covariance"
                if cross_channel_covariance_present else
                "within_channel_gls_with_measurement_model_discrepancy_and_systematic_covariance"
                if measurement_covariance_supplied and model_discrepancy_covariance_supplied else
                "within_channel_gls_with_measurement_and_systematic_covariance"
                if measurement_covariance_supplied else
                "within_channel_gls_with_model_discrepancy_and_systematic_covariance"
                if model_discrepancy_covariance_supplied else
                "within_channel_gls_with_systematic_covariance_only"
            )
        fit["background_model_status"] = (
            f"{background_model}_detector_space_basis_jointly_fitted"
        )
        fit["background_model_limitation"] = (
            "The selected researcher-supplied detector-space basis is fitted jointly "
            "with the selected optical Thomson feature. Basis inadequacy and "
            "signal/background degeneracy remain model-dependent; the nuisance "
            "background is not a physical plasma-emission decomposition."
        )
        fit["channel_index"] = int(i)
        fit["radius_m"] = float(radii[i])
        fit["instrument_response_used"] = instrument_response is not None
        fit["laser_spectral_profile_used"] = source_profile_metadata is not None
        fit["laser_spectral_profile_model"] = (
            source_profile_metadata if source_profile_metadata is not None
            else {"assumption": "monochromatic_incident_laser"}
        )
        fit["source_and_instrument_response_order"] = (
            "instrument_response @ laser_source_response"
            if source_profile_metadata is not None else "instrument_response_only"
        )
        fit["instrument_response_uncertainty_supplied"] = lsf_unc is not None
        missing_uncertainties=[]
        if unc is None: missing_uncertainties.append("signal_uncertainty")
        if lsf_unc is None: missing_uncertainties.append("instrument_response_uncertainty")
        if laser_wavelength_uncertainty_nm is None: missing_uncertainties.append("laser_wavelength_uncertainty_nm")
        if scattering_angle_uncertainty_rad is None: missing_uncertainties.append("scattering_angle_uncertainty_rad")
        if model == "non_collective_relativistic_maxwell_juttner":
            if laser_wavelength_uncertainty_nm is None:
                missing_uncertainties.append("laser_wavelength_uncertainty_nm")
            if scattering_angle_uncertainty_rad is None and incident_direction_unc is None and scattered_direction_unc is None:
                missing_uncertainties.append("scattering_geometry_uncertainty")
            if polarization_unc is None:
                missing_uncertainties.append("incident_polarization_uncertainty")
        if model in {"collective_maxwellian", "collective_kappa"}:
            # A beam-direction uncertainty is an alternative representation of
            # scattering-angle uncertainty. Do not report the scalar angle as
            # missing when either measured direction has a supplied covariance.
            missing_uncertainties = [
                name for name in missing_uncertainties
                if name != "scattering_angle_uncertainty_rad"
            ]
            missing_uncertainties.extend(_collective_uncertainty_inputs_missing(
                ion_t_unc, ion_v_unc, electron_v_unc, scattering_angle_uncertainty_rad,
                incident_direction_unc, scattered_direction_unc,
            ))
            if lsf_unc is not None:
                # Collective response uncertainty is propagated in the external-input
                # Monte Carlo below; do not mark it missing before that calculation.
                fit["instrument_response_uncertainty_propagated"] = False
        if all(v is not None for v in (laser_power_w,interaction_length_m,collection_solid_angle_sr,optical_efficiency)) and density_relative_covariance is None:
            for key,value in (("laser_power_uncertainty_W",laser_power_uncertainty_W),
                              ("interaction_length_uncertainty_m",interaction_length_uncertainty_m),
                              ("collection_solid_angle_uncertainty_sr",collection_solid_angle_uncertainty_sr),
                              ("optical_efficiency_uncertainty",optical_efficiency_uncertainty),
                              ("polarization_factor_uncertainty",polarization_factor_uncertainty)):
                if value is None: missing_uncertainties.append(key)
        fit["uncertainty_inputs_missing"] = missing_uncertainties
        fit["uncertainty_completeness"] = "partial" if missing_uncertainties else "all_supported_inputs_supplied"
        if lsf_unc is not None and model in {"non_collective_maxwellian", "non_collective_kappa", "non_collective_relativistic_maxwell_juttner"}:
            lsf_sigma = lsf_unc if lsf_unc.ndim in (1, 2) else lsf_unc[i]
            # A shared 2-D response matrix represents one common instrument
            # realization for every channel. Reuse the same random stream and
            # retain failed draw slots so cross-channel samples stay aligned.
            rng = np.random.default_rng(7319 if lsf_unc.ndim in (1, 2) else 7319 + i)
            te_samples, velocity_samples, area_samples, cross_section_samples = [], [], [], []
            for _ in range(int(instrument_response_mc_samples)):
                if lsf_unc.ndim == 1:
                    sampled_kernel = np.maximum(
                        np.asarray(instrument_response, dtype=float)
                        + rng.normal(0.0, lsf_sigma), 0.0
                    )
                    kernel_area = float(sampled_kernel.sum())
                    if not np.isfinite(kernel_area) or kernel_area <= 0:
                        te_samples.append(np.nan)
                        velocity_samples.append(np.nan)
                        area_samples.append(np.nan)
                        cross_section_samples.append(np.nan)
                        continue
                    # Normalize only the perturbed kernel realization; preserve
                    # the supplied calibrated base kernel and its finite-window
                    # edge losses unchanged.
                    sampled_kernel /= kernel_area
                    sampled = _response_for_channel(
                        sampled_kernel, i, y.shape[0], y.shape[1], xi
                    )
                    if source_response is not None:
                        sampled = sampled @ source_response
                else:
                    sampled = np.maximum(response + rng.normal(0.0, lsf_sigma), 0.0)
                    colsum = sampled.sum(axis=0)
                    if np.any(colsum <= 0):
                        te_samples.append(np.nan)
                        velocity_samples.append(np.nan)
                        area_samples.append(np.nan)
                        cross_section_samples.append(np.nan)
                        continue
                    sampled /= colsum[None, :]
                    if source_response is not None:
                        sampled = sampled @ source_response
                try:
                    if model == "non_collective_kappa":
                        draw = _fit_kappa(xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad), sampled, ui, spectral_covariance=channel_covariance)
                    elif model == "non_collective_relativistic_maxwell_juttner":
                        draw = fit_relativistic_maxwell_juttner(
                            xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad),
                            sampled, ui, incident_direction=probe_vec, scattered_direction=scatter_vec,
                            incident_polarization=incident_polarization,
                            maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
                include_compton_recoil=include_compton_recoil,
                        )
                    else:
                        draw = _fit_noncollective(xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad), sampled, ui,
                            background_model=background_model, background_degree=background_degree,
                            background_interior_knots=background_interior_knots,
                            background_templates=(background_templates[i] if np.asarray(background_templates).ndim == 3 else background_templates) if background_templates is not None else None,
                            spectral_covariance=channel_covariance)
                    te_samples.append(draw["electron_temperature_eV"])
                    velocity_samples.append(draw["electron_velocity_projection_m_s"])
                    area_samples.append(float(draw["integrated_scattered_power_W"])
                                        if "integrated_scattered_power_W" in draw
                                        and np.isfinite(draw["integrated_scattered_power_W"]) else np.nan)
                    cross_section_samples.append(float(draw["effective_differential_cross_section_m2_sr"])
                                                  if "effective_differential_cross_section_m2_sr" in draw else np.nan)
                except (ValueError, RuntimeError):
                    te_samples.append(np.nan)
                    velocity_samples.append(np.nan)
                    area_samples.append(np.nan)
                    cross_section_samples.append(np.nan)
            required_successes = max(8, int(np.ceil(instrument_response_mc_min_success_fraction * instrument_response_mc_samples)))
            te_valid = np.asarray(te_samples, dtype=float)
            velocity_valid = np.asarray(velocity_samples, dtype=float)
            area_valid = np.asarray(area_samples, dtype=float)
            cross_section_valid = np.asarray(cross_section_samples, dtype=float)
            te_valid = te_valid[np.isfinite(te_valid)]
            velocity_valid = velocity_valid[np.isfinite(velocity_valid)]
            area_valid = area_valid[np.isfinite(area_valid)]
            cross_section_valid = cross_section_valid[np.isfinite(cross_section_valid)]
            if te_valid.size >= required_successes:
                fit["electron_temperature_uncertainty_eV"] = float(np.hypot(
                    fit["electron_temperature_uncertainty_eV"], np.std(te_valid, ddof=1)))
                fit["electron_velocity_projection_uncertainty_m_s"] = float(np.hypot(
                    fit["electron_velocity_projection_uncertainty_m_s"], np.std(velocity_valid, ddof=1)))
                fit["instrument_response_uncertainty_propagated"] = True
                fit["instrument_response_mc_successful_samples"] = int(te_valid.size)
                fit["instrument_response_temperature_uncertainty_eV"] = float(np.std(te_valid, ddof=1))
                fit["instrument_response_velocity_uncertainty_m_s"] = float(np.std(velocity_valid, ddof=1))
                if model == "non_collective_relativistic_maxwell_juttner" and cross_section_valid.size >= required_successes:
                    cross_section_lsf_sigma = float(np.std(cross_section_valid, ddof=1))
                    fit["effective_differential_cross_section_uncertainty_m2_sr"] = float(np.hypot(
                        fit.get("effective_differential_cross_section_uncertainty_m2_sr") or 0.0,
                        cross_section_lsf_sigma,
                    ))
                    fit["differential_cross_section_uncertainty_m2_sr"] = fit["effective_differential_cross_section_uncertainty_m2_sr"]
                    fit["instrument_response_cross_section_uncertainty_m2_sr"] = cross_section_lsf_sigma
                    paired_lsf = np.column_stack((np.asarray(area_samples, dtype=float), np.asarray(cross_section_samples, dtype=float)))
                    paired_lsf = paired_lsf[np.all(np.isfinite(paired_lsf), axis=1)]
                    if paired_lsf.shape[0] >= required_successes:
                        lsf_power_cross_covariance = float(
                            np.cov(paired_lsf[:, 0], paired_lsf[:, 1], ddof=1)[0, 1]
                            / (np.mean(paired_lsf[:, 0]) * np.mean(paired_lsf[:, 1]))
                        )
                        fit["power_cross_section_relative_covariance"] = float(
                            fit.get("power_cross_section_relative_covariance") or 0.0
                        ) + lsf_power_cross_covariance
                if lsf_unc.ndim in (1, 2):
                    fit["_shared_lsf_temperature_draws"] = np.asarray(te_samples, dtype=float)
                    fit["_shared_lsf_velocity_draws"] = np.asarray(velocity_samples, dtype=float)
                    fit["_shared_lsf_area_draws"] = np.asarray(area_samples, dtype=float)
                    if model == "non_collective_relativistic_maxwell_juttner":
                        fit["_shared_lsf_cross_section_draws"] = np.asarray(cross_section_samples, dtype=float)
                fit["integrated_scattered_power_fit_uncertainty_W"] = float(
                    fit.get("integrated_scattered_power_uncertainty_W") or 0.0
                )
                fit["integrated_scattered_power_lsf_uncertainty_W"] = (
                    float(np.std(area_valid, ddof=1)) if area_valid.size >= required_successes else None
                )
                if fit["integrated_scattered_power_lsf_uncertainty_W"] is not None:
                    fit["integrated_scattered_power_uncertainty_W"] = float(np.hypot(
                        fit["integrated_scattered_power_fit_uncertainty_W"],
                        fit["integrated_scattered_power_lsf_uncertainty_W"],
                    ))
            else:
                fit["instrument_response_uncertainty_propagated"] = False
                fit["instrument_response_mc_successful_samples"] = int(te_valid.size)
                fit["integrated_scattered_power_lsf_uncertainty_W"] = None
                fit["instrument_response_uncertainty_warning"] = "Too few valid LSF perturbation fits to estimate its contribution."
        if density_inputs is not None and lsf_unc is not None:
            if model == "non_collective_relativistic_maxwell_juttner":
                density_inputs["effective_differential_cross_section_uncertainty_m2_sr"] = fit.get(
                    "effective_differential_cross_section_uncertainty_m2_sr"
                )
                density_inputs["power_cross_section_relative_covariance"] = fit.get(
                    "power_cross_section_relative_covariance"
                )
            if area_valid.size >= required_successes:
                fit_area_uncertainty = float(
                    fit.get("integrated_scattered_power_fit_uncertainty_W",
                            fit.get("integrated_scattered_power_uncertainty_W")) or 0.0
                )
                lsf_area_uncertainty = float(np.std(area_valid, ddof=1))
                if density_inputs.get("relative_input_covariance") is None:
                    combined_area_uncertainty = float(np.hypot(fit_area_uncertainty, lsf_area_uncertainty))
                    density_inputs["scattered_power_uncertainty_W"] = combined_area_uncertainty
                else:
                    covariance = np.asarray(density_inputs["relative_input_covariance"], dtype=float).copy()
                    lsf_relative_sigma = 0.0 if area == 0.0 else lsf_area_uncertainty / abs(area)
                    if np.allclose(covariance[0, :], 0.0, rtol=0.0, atol=1e-14) and np.allclose(
                        covariance[:, 0], 0.0, rtol=0.0, atol=1e-14
                    ):
                        # Spectral/response uncertainty is supplied separately;
                        # preserve the optical-factor covariance and combine the
                        # independent area contributions as a scalar marginal.
                        existing_area_sigma = float(
                            density_inputs.get("scattered_power_uncertainty_W") or 0.0
                        )
                        combined_area_uncertainty = float(np.hypot(
                            existing_area_sigma, lsf_area_uncertainty
                        ))
                        density_inputs["scattered_power_uncertainty_W"] = combined_area_uncertainty
                    else:
                        # A complete covariance already contains area uncertainty;
                        # add the independent LSF contribution to its power block.
                        covariance[0, 0] += lsf_relative_sigma**2
                        density_inputs["relative_input_covariance"] = covariance
                        density_inputs["scattered_power_uncertainty_W"] = None
                        combined_area_uncertainty = float(
                            abs(area) * np.sqrt(max(covariance[0, 0], 0.0))
                        )
                density_result = density_from_thomson_scattered_power(**density_inputs)
                fit["electron_density_m3"] = density_result["electron_density_m3"]
                fit["electron_density_uncertainty_m3"] = density_result["electron_density_uncertainty_m3"]
                fit["integrated_scattered_power_fit_uncertainty_W"] = fit_area_uncertainty
                fit["integrated_scattered_power_lsf_uncertainty_W"] = lsf_area_uncertainty
                fit["integrated_scattered_power_uncertainty_W"] = combined_area_uncertainty
                fit["density_uncertainty_model"] = density_result["uncertainty_model"]
                fit["instrument_response_density_uncertainty_propagated"] = True
            else:
                fit["instrument_response_density_uncertainty_propagated"] = False
                fit["uncertainty_inputs_missing"].append("instrument_response_uncertainty_not_propagated_to_absolute_density")
                fit["uncertainty_completeness"] = "partial"
        elif density_inputs is not None:
            fit["instrument_response_density_uncertainty_propagated"] = False

        if model in {"non_collective_maxwellian", "non_collective_kappa"}:
            te = fit["electron_temperature_eV"]
            external_te_sigma = thomson_electron_temperature_from_spectral_width(
                float(fit["intrinsic_sigma_nm"]), float(laser_wavelength_nm),
                float(scattering_angle_rad), sigma_intrinsic_sigma_nm=0.0,
                sigma_laser_wavelength_nm=laser_wavelength_unc,
                sigma_scattering_angle_rad=scattering_angle_unc,
            )["uncertainty_electron_temperature_eV"]
            fit["electron_temperature_external_uncertainty_eV"] = external_te_sigma
            fit["electron_temperature_uncertainty_eV"] = float(np.hypot(
                fit["electron_temperature_uncertainty_eV"], external_te_sigma))
            velocity = fit["electron_velocity_projection_m_s"]
            velocity_external = thomson_velocity_projection(
                float(fit["fitted_center_nm"]),float(laser_wavelength_nm),
                float(scattering_angle_rad),sigma_fitted_center_nm=0.0,
                sigma_laser_wavelength_nm=laser_wavelength_unc,
                sigma_scattering_angle_rad=scattering_angle_unc,
            )
            velocity_laser_angle_sigma = velocity_external["uncertainty_electron_velocity_projection_m_s"]
            velocity_laser_sigma = 0.0
            velocity_angle_sigma = float(velocity_laser_angle_sigma)
            external_velocity_sigma = float(np.hypot(velocity_laser_sigma, velocity_angle_sigma))
            fit["electron_velocity_projection_external_uncertainty_m_s"] = external_velocity_sigma
            fit["electron_velocity_projection_uncertainty_m_s"] = float(np.hypot(
                fit["electron_velocity_projection_uncertainty_m_s"], external_velocity_sigma))
        if model == "non_collective_relativistic_maxwell_juttner" and any(value is not None for value in (
            laser_wavelength_uncertainty_nm, scattering_angle_uncertainty_rad,
            incident_direction_unc, scattered_direction_unc, polarization_unc,
        )):
            rng_external = np.random.default_rng(84271 + i)
            external_te_samples, external_velocity_samples, external_cross_section_samples, external_area_samples = [], [], [], []
            incident_nominal = np.asarray(probe_vec, dtype=float)
            scattered_nominal = np.asarray(scatter_vec, dtype=float)
            polarization_nominal = np.asarray(
                _channel_value(incident_polarization, i, y.shape[0], "incident_polarization", vector_length=3),
                dtype=float,
            )
            for _ in range(int(instrument_response_mc_samples)):
                sampled_laser = float(laser_wavelength_nm) + (
                    0.0 if laser_wavelength_uncertainty_nm is None
                    else float(rng_external.normal(0.0, laser_wavelength_unc))
                )
                if sampled_laser <= 0.0:
                    continue
                sampled_incident = incident_nominal.copy()
                sampled_scattered = scattered_nominal.copy()
                if incident_direction_unc is not None:
                    sampled_incident += rng_external.normal(0.0, incident_direction_unc)
                if scattered_direction_unc is not None:
                    sampled_scattered += rng_external.normal(0.0, scattered_direction_unc)
                if np.linalg.norm(sampled_incident) <= 0.0 or np.linalg.norm(sampled_scattered) <= 0.0:
                    continue
                sampled_incident /= np.linalg.norm(sampled_incident)
                sampled_scattered /= np.linalg.norm(sampled_scattered)
                if scattering_angle_uncertainty_rad is not None:
                    sampled_angle = float(scattering_angle_rad) + float(
                        rng_external.normal(0.0, scattering_angle_unc)
                    )
                    if not 0.0 < sampled_angle < np.pi:
                        continue
                    tangent = sampled_scattered - np.dot(sampled_scattered, sampled_incident) * sampled_incident
                    tangent_norm = float(np.linalg.norm(tangent))
                    if tangent_norm <= 1e-12:
                        continue
                    tangent /= tangent_norm
                    sampled_scattered = (
                        np.cos(sampled_angle) * sampled_incident
                        + np.sin(sampled_angle) * tangent
                    )
                sampled_angle = float(np.arccos(np.clip(
                    np.dot(sampled_incident, sampled_scattered), -1.0, 1.0
                )))
                if not 0.0 < sampled_angle < np.pi:
                    continue
                sampled_polarization = polarization_nominal.copy()
                if polarization_unc is not None:
                    sampled_polarization += rng_external.normal(0.0, polarization_unc)
                sampled_polarization -= (
                    np.dot(sampled_polarization, sampled_incident) * sampled_incident
                )
                polarization_norm = float(np.linalg.norm(sampled_polarization))
                if polarization_norm <= 1e-12:
                    continue
                sampled_polarization /= polarization_norm
                try:
                    draw = fit_relativistic_maxwell_juttner(
                        xi, y[i], sampled_laser, sampled_angle, response, ui,
                        incident_direction=sampled_incident,
                        scattered_direction=sampled_scattered,
                        incident_polarization=sampled_polarization,
                        maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
                include_compton_recoil=include_compton_recoil,
                    )
                    external_te_samples.append(float(draw["electron_temperature_eV"]))
                    external_velocity_samples.append(float(draw["electron_velocity_projection_m_s"]))
                    external_cross_section_samples.append(float(draw["effective_differential_cross_section_m2_sr"]))
                    external_area_samples.append(float(draw["integrated_scattered_power_W"]))
                except (ValueError, RuntimeError):
                    continue
            required_external = max(
                8, int(np.ceil(instrument_response_mc_min_success_fraction * instrument_response_mc_samples))
            )
            fit["relativistic_external_parameter_mc_requested"] = int(instrument_response_mc_samples)
            fit["relativistic_external_parameter_mc_successful_samples"] = len(external_te_samples)
            if len(external_te_samples) >= required_external:
                te_external_sigma = float(np.std(external_te_samples, ddof=1))
                velocity_external_sigma = float(np.std(external_velocity_samples, ddof=1))
                fit["electron_temperature_uncertainty_eV"] = float(np.hypot(
                    fit["electron_temperature_uncertainty_eV"], te_external_sigma
                ))
                fit["electron_velocity_projection_uncertainty_m_s"] = float(np.hypot(
                    fit["electron_velocity_projection_uncertainty_m_s"], velocity_external_sigma
                ))
                cross_section_external_sigma = float(np.std(external_cross_section_samples, ddof=1))
                fit["effective_differential_cross_section_uncertainty_m2_sr"] = float(np.hypot(
                    fit.get("effective_differential_cross_section_uncertainty_m2_sr") or 0.0,
                    cross_section_external_sigma,
                ))
                external_area_sigma = float(np.std(external_area_samples, ddof=1))
                fit["integrated_scattered_power_uncertainty_W"] = float(np.hypot(
                    fit.get("integrated_scattered_power_uncertainty_W") or 0.0,
                    external_area_sigma,
                ))
                fit["relativistic_external_integrated_power_uncertainty_W"] = external_area_sigma
                fit["relativistic_external_cross_section_uncertainty_m2_sr"] = cross_section_external_sigma
                fit["differential_cross_section_uncertainty_m2_sr"] = fit["effective_differential_cross_section_uncertainty_m2_sr"]
                paired_external = np.column_stack((
                    np.asarray(external_area_samples, dtype=float),
                    np.asarray(external_cross_section_samples, dtype=float),
                ))
                paired_external = paired_external[np.all(np.isfinite(paired_external), axis=1)]
                if paired_external.shape[0] >= required_external:
                    external_power_cross_covariance = float(
                        np.cov(paired_external[:, 0], paired_external[:, 1], ddof=1)[0, 1]
                        / (np.mean(paired_external[:, 0]) * np.mean(paired_external[:, 1]))
                    )
                    fit["power_cross_section_relative_covariance"] = float(
                        fit.get("power_cross_section_relative_covariance") or 0.0
                    ) + external_power_cross_covariance
                fit["relativistic_external_parameter_uncertainty_propagated"] = True
                fit["relativistic_external_temperature_uncertainty_eV"] = te_external_sigma
                fit["relativistic_external_velocity_uncertainty_m_s"] = velocity_external_sigma
                fit["uncertainty_inputs_missing"] = [
                    name for name in fit["uncertainty_inputs_missing"]
                    if name not in {
                        "laser_wavelength_uncertainty_nm",
                        "scattering_geometry_uncertainty",
                        "incident_polarization_uncertainty",
                    }
                ]
                fit["uncertainty_completeness"] = (
                    "partial" if fit["uncertainty_inputs_missing"] else "all_supported_inputs_supplied"
                )
                if density_inputs is not None:
                    density_inputs["effective_differential_cross_section_uncertainty_m2_sr"] = fit[
                        "effective_differential_cross_section_uncertainty_m2_sr"
                    ]
                    density_inputs["power_cross_section_relative_covariance"] = fit.get(
                        "power_cross_section_relative_covariance"
                    )
                    covariance_density = density_inputs.get("relative_input_covariance")
                    if covariance_density is None:
                        density_inputs["scattered_power_uncertainty_W"] = float(np.hypot(
                            density_inputs.get("scattered_power_uncertainty_W") or 0.0,
                            external_area_sigma,
                        ))
                    else:
                        covariance_density = np.asarray(covariance_density, dtype=float).copy()
                        if np.allclose(covariance_density[0, :], 0.0, rtol=0.0, atol=1e-14) and np.allclose(
                            covariance_density[:, 0], 0.0, rtol=0.0, atol=1e-14
                        ):
                            density_inputs["scattered_power_uncertainty_W"] = float(np.hypot(
                                density_inputs.get("scattered_power_uncertainty_W") or 0.0,
                                external_area_sigma,
                            ))
                        else:
                            area_now = float(fit["integrated_scattered_power_W"])
                            covariance_density[0, 0] += (external_area_sigma / area_now) ** 2
                            density_inputs["relative_input_covariance"] = covariance_density
                    density_result = density_from_thomson_scattered_power(**density_inputs)
                    fit["electron_density_m3"] = density_result["electron_density_m3"]
                    fit["electron_density_uncertainty_m3"] = density_result["electron_density_uncertainty_m3"]
                    fit["density_uncertainty_model"] = density_result["uncertainty_model"]
            else:
                fit["relativistic_external_parameter_uncertainty_propagated"] = False
                fit["uncertainty_inputs_missing"].append(
                    "relativistic_external_parameter_monte_carlo_propagation_failed"
                )
                fit["uncertainty_completeness"] = "partial"
        if lsf_unc is not None and model not in {"collective_maxwellian", "collective_kappa"} and not fit.get("instrument_response_uncertainty_propagated",False):
            fit["uncertainty_inputs_missing"].append("instrument_response_uncertainty_propagation_failed")
            fit["uncertainty_completeness"] = "partial"
        results.append(fit)
    shared_lsf_covariance = None
    shared_lsf_covariance_order = []
    if lsf_unc is not None and lsf_unc.ndim in (1, 2) and results and all(
        "_shared_lsf_temperature_draws" in item for item in results
    ):
        # Same draw index is the same shared instrument-response realization.
        draw_columns = []
        for channel_index, item in enumerate(results):
            draw_columns.extend([
                item["_shared_lsf_temperature_draws"],
                item["_shared_lsf_velocity_draws"],
                item["_shared_lsf_area_draws"],
            ])
            shared_lsf_covariance_order.extend([
                f"channel_{channel_index}_electron_temperature_eV",
                f"channel_{channel_index}_electron_velocity_projection_m_s",
                f"channel_{channel_index}_integrated_scattered_power_W",
            ])
        aligned_draws = np.column_stack(draw_columns)
        valid_draws = np.all(np.isfinite(aligned_draws), axis=1)
        if np.count_nonzero(valid_draws) >= 2:
            shared_lsf_covariance = np.cov(aligned_draws[valid_draws], rowvar=False, ddof=1)
    shared_response_density_covariance = None
    if shared_lsf_covariance is not None:
        channel_count = len(results)
        area_indices = np.asarray([3 * channel + 2 for channel in range(channel_count)], dtype=int)
        area_covariance = shared_lsf_covariance[np.ix_(area_indices, area_indices)]
        density_values = np.asarray([
            np.nan if item.get("electron_density_m3") is None else item["electron_density_m3"]
            for item in results
        ], dtype=float)
        area_values = np.asarray([
            item.get("integrated_scattered_power_W", np.nan) for item in results
        ], dtype=float)
        shared_response_density_covariance = np.full(
            (channel_count, channel_count), np.nan, dtype=float
        )
        valid_density = np.isfinite(density_values) & np.isfinite(area_values) & (area_values != 0.0)
        if np.any(valid_density):
            scale = density_values[valid_density] / area_values[valid_density]
            shared_response_density_covariance[np.ix_(valid_density, valid_density)] = (
                np.outer(scale, scale)
                * area_covariance[np.ix_(valid_density, valid_density)]
            )
    for item in results:
        item.pop("_shared_lsf_temperature_draws", None)
        item.pop("_shared_lsf_velocity_draws", None)
        item.pop("_shared_lsf_area_draws", None)
    joint_summary = None
    joint_area_covariance_from_spectral_fit = None
    if joint_covariance is not None and model == "non_collective_relativistic_maxwell_juttner":
        common_grid = all(
            np.array_equal(np.asarray(xs[0]), np.asarray(item))
            for item in xs[1:]
        )
        if not common_grid:
            raise ValueError(
                "Full cross-channel relativistic Maxwell-Juttner GLS requires identical wavelength sampling in every channel."
            )
        if incident_direction is None or scattered_direction is None or incident_polarization is None:
            raise ValueError(
                "The joint relativistic Maxwell-Juttner channel fit requires incident_direction, scattered_direction and incident_polarization."
            )
        from .thomson_kinetic_models import fit_relativistic_maxwell_juttner_channels
        initial_joint_te = float(np.median([
            item["electron_temperature_eV"] for item in results
        ]))
        initial_joint_flow = float(np.median([
            item["electron_velocity_projection_m_s"] for item in results
        ]))
        joint_summary = fit_relativistic_maxwell_juttner_channels(
            xs[0], y, laser_wavelength_nm, scattering_angle_rad,
            channel_responses,
            incident_direction=incident_direction,
            scattered_direction=scattered_direction,
            incident_polarization=incident_polarization,
            joint_spectral_covariance=joint_covariance,
            initial_temperature_eV=initial_joint_te,
            initial_flow_projection_m_s=initial_joint_flow,
            maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
            include_compton_recoil=include_compton_recoil,
            background_model=background_model,
            background_degree=background_degree,
            background_interior_knots=background_interior_knots,
            background_templates=(
                [background_templates[j] for j in range(y.shape[0])]
                if background_templates is not None and np.asarray(background_templates).ndim == 3
                else background_templates
            ),
        )
        joint_summary["integration_status"] = "used_by_canonical_multichannel_thomson_api"
        joint_summary["channel_individual_fits_retained"] = True
        joint_summary["interpretation"] = (
            "The full cross-channel covariance is used by a joint relativistic Maxwell-Juttner fit "
            "with shared electron temperature and scattering-vector flow, and channel-specific "
            "amplitudes/background nuisance terms. The per-channel fits remain available as "
            "independent marginal fits; the joint result is a separate shared-parameter inference."
        )
    if joint_covariance is not None and model != "non_collective_relativistic_maxwell_juttner":
        from scipy.optimize import least_squares
        nchan, nlam = y.shape
        from .thomson_background import _background_design
        physical_parameter_count = 3 if model == "non_collective_maxwellian" else 4
        if background_templates is not None and np.asarray(background_templates).ndim == 3:
            templates_by_channel = [np.asarray(background_templates, dtype=float)[j] for j in range(nchan)]
        elif isinstance(background_templates, (list, tuple)):
            if len(background_templates) != nchan:
                raise ValueError("background_templates list must contain one matrix per channel")
            templates_by_channel = list(background_templates)
        else:
            templates_by_channel = [background_templates] * nchan
        background_designs = [
            _background_design(
                xs[j], background_model, degree=background_degree,
                interior_knots=background_interior_knots,
                templates=templates_by_channel[j],
            )
            for j in range(nchan)
        ]
        background_parameter_count = background_designs[0].shape[1]
        if any(item.shape[1] != background_parameter_count for item in background_designs):
            raise ValueError("All channel background bases must have the same number of columns for joint GLS.")
        initial_parts = []
        for item in results:
            physical_seed = [
                float(item["amplitude"]),
                float(item["fitted_center_nm"]),
                float(item["intrinsic_sigma_nm"]),
            ]
            if model == "non_collective_kappa":
                physical_seed.append(float(item["kappa_index"]))
            nuisance_seed = np.asarray(item["background_coefficients"], dtype=float)
            if nuisance_seed.shape != (background_parameter_count,):
                raise ValueError("Per-channel background coefficient count does not match the selected joint GLS basis.")
            initial_parts.append(np.r_[physical_seed, nuisance_seed])
        initial = np.concatenate(initial_parts)
        joint_parameter_count = physical_parameter_count + background_parameter_count
        lower_parts, upper_parts = [], []
        for i in range(nchan):
            dx = float(np.median(np.diff(xs[i])))
            span_i = float(xs[i, -1] - xs[i, 0])
            if model == "non_collective_maxwellian":
                lower_physical = [0.0, xs[i, 0], max(dx * 0.2, 1e-12)]
                upper_physical = [np.inf, xs[i, -1], span_i]
            else:
                lower_physical = [0.0, xs[i, 0], max(dx * 0.2, 1e-12), 1.51]
                upper_physical = [np.inf, xs[i, -1], span_i, np.inf]
            lower_parts.append(lower_physical + [-np.inf] * background_parameter_count)
            upper_parts.append(upper_physical + [np.inf] * background_parameter_count)
        lower = np.asarray(lower_parts).ravel()
        upper = np.asarray(upper_parts).ravel()
        def joint_forward(parameters):
            predicted = []
            for channel in range(nchan):
                p = parameters[joint_parameter_count*channel:joint_parameter_count*(channel+1)]
                if model == "non_collective_maxwellian":
                    amp, center, sigma = p[:3]
                    signal_component = amp*np.exp(-0.5*((xs[channel]-center)/sigma)**2)
                else:
                    amp, center, sigma, kappa = p[:4]
                    df = 2.0*kappa-1.0
                    scale = sigma*np.sqrt((df-2.0)/df)
                    signal_component = amp*(1.0+((xs[channel]-center)/scale)**2/df)**(-kappa)
                background_coefficients = p[physical_parameter_count:]
                background_component = background_designs[channel] @ background_coefficients
                predicted.append(
                    _apply_response(channel_responses[channel], signal_component, xs[channel])
                    + background_component
                )
            return np.asarray(predicted)
        measured_flat = y.ravel()
        def joint_residual(parameters):
            return np.linalg.solve(joint_cholesky, (joint_forward(parameters)-y).ravel())
        joint_fit = least_squares(joint_residual, initial, bounds=(lower, upper), x_scale="jac", max_nfev=30000)
        joint_column_norms = np.linalg.norm(joint_fit.jac, axis=0)
        joint_scaled_jacobian = joint_fit.jac / np.maximum(joint_column_norms, np.finfo(float).tiny)[None, :]
        joint_rank = int(np.linalg.matrix_rank(joint_scaled_jacobian))
        joint_condition = float(np.linalg.cond(joint_scaled_jacobian))
        if not joint_fit.success or joint_rank < initial.size or not np.isfinite(joint_condition) or joint_condition > 1e12:
            raise ValueError("Joint Thomson GLS fit rejected: optimizer failure or non-identifiable channel parameters")
        parameter_covariance = np.linalg.pinv(joint_fit.jac.T @ joint_fit.jac)
        dof_joint = max(1, measured_flat.size-initial.size)
        standardized = joint_residual(joint_fit.x)
        reduced_chi2_joint = float(standardized @ standardized / dof_joint)
        if not np.isfinite(reduced_chi2_joint):
            raise ValueError("Joint Thomson GLS fit rejected: reduced chi-square is non-finite")
        # With a supplied covariance, reduced chi-square is a model-adequacy
        # diagnostic. A universal hard cutoff is not justified without an
        # explicit noise/significance policy supplied by the researcher.
        # Transform the full parameter covariance to all channel Te and flow values.
        def physical_vector(parameters):
            values=[]
            for channel in range(nchan):
                p=parameters[joint_parameter_count*channel:joint_parameter_count*(channel+1)]
                center=float(p[1]); sigma=float(p[2])
                te=thomson_electron_temperature_from_spectral_width(
                    sigma,float(laser_wavelength_nm),float(scattering_angle_rad)
                )["electron_temperature_eV"]
                velocity=thomson_velocity_projection(
                    center,float(laser_wavelength_nm),float(scattering_angle_rad)
                )["electron_velocity_projection_m_s"]
                values.extend([te,velocity])
            return np.asarray(values,dtype=float)
        base_physical=physical_vector(joint_fit.x)
        for channel in range(nchan):
            te_value=base_physical[2*channel]
            velocity_value=base_physical[2*channel+1]
            if te_value/ELECTRON_REST_ENERGY_EV>=NONRELATIVE_THETA_E_SCREEN:
                raise ValueError("Joint Thomson GLS fit rejected: non-relativistic electron-temperature limit exceeded")
            if abs(velocity_value)>=NONRELATIVE_FLOW_BETA_SCREEN*C_M_S:
                raise ValueError("Joint Thomson GLS fit rejected: non-relativistic flow-projection limit exceeded")
        transform=np.zeros((2*nchan,joint_parameter_count*nchan))
        for j in range(joint_parameter_count*nchan):
            step=np.cbrt(np.finfo(float).eps)*max(abs(float(joint_fit.x[j])),1.0)
            plus=joint_fit.x.copy(); minus=joint_fit.x.copy()
            plus[j]+=step; minus[j]-=step
            transform[:,j]=(physical_vector(plus)-physical_vector(minus))/(2.0*step)
        physical_covariance=transform@parameter_covariance@transform.T
        # Excess lack-of-fit is reported as a separate model-discrepancy component.
        discrepancy_scale=max(reduced_chi2_joint-1.0,0.0)
        discrepancy_covariance=discrepancy_scale*physical_covariance
        total_physical_covariance=physical_covariance+discrepancy_covariance
        kappa_parameter_covariance = None
        total_kappa_parameter_covariance = None
        if model == "non_collective_kappa":
            kappa_transform = np.zeros((nchan, joint_parameter_count*nchan))
            for channel in range(nchan):
                kappa_transform[channel, joint_parameter_count*channel+3] = 1.0
            kappa_parameter_covariance = (
                kappa_transform @ parameter_covariance @ kappa_transform.T
            )
            total_kappa_parameter_covariance = (
                kappa_parameter_covariance
                + discrepancy_scale*kappa_parameter_covariance
            )
        joint_fitted=joint_forward(joint_fit.x)
        joint_area_gradient_matrix = np.zeros(
            (nchan, joint_parameter_count * nchan), dtype=float
        )
        for channel in range(nchan):
            p=joint_fit.x[joint_parameter_count*channel:joint_parameter_count*(channel+1)]
            block=parameter_covariance[joint_parameter_count*channel:joint_parameter_count*(channel+1),
                                        joint_parameter_count*channel:joint_parameter_count*(channel+1)]
            te_index=2*channel; velocity_index=te_index+1
            fit=results[channel]
            if model == "non_collective_maxwellian":
                amp, center, sigma = p[:3]
                extra = {}
            else:
                amp, center, sigma, kappa = p[:4]
                extra = {"kappa_index": float(kappa)}
            background_coefficients = p[physical_parameter_count:]
            background_spectrum = background_designs[channel] @ background_coefficients
            laser_index = int(np.argmin(np.abs(xs[channel]-float(laser_wavelength_nm))))
            background_intercept = float(background_spectrum[laser_index])
            background_slope = float(np.gradient(background_spectrum, xs[channel])[laser_index])
            fit.update({
                "amplitude":float(amp),"fitted_center_nm":float(center),"intrinsic_sigma_nm":float(sigma),
                "background_intercept":background_intercept,"background_slope_per_nm":background_slope,
                "background_coefficients":background_coefficients.copy(),
                "background_spectrum":background_spectrum.copy(),
                "background_model":background_model,
                "background_basis_matrix":background_designs[channel].copy(),
                "background_coefficient_covariance":block[physical_parameter_count:, physical_parameter_count:].copy(),
                "physical_background_cross_covariance":block[:physical_parameter_count, physical_parameter_count:].copy(),
                "fit_parameter_names":(
                    (["amplitude", "center_nm", "intrinsic_sigma_nm"] if model == "non_collective_maxwellian"
                     else ["amplitude", "center_nm", "intrinsic_sigma_nm", "kappa_index"])
                    + [f"background_coefficient_{j}" for j in range(background_parameter_count)]
                ),**extra,
                "fit_parameters":p.copy(),"fit_covariance":block.copy(),
                "electron_temperature_eV":float(base_physical[te_index]),
                "electron_velocity_projection_m_s":float(base_physical[velocity_index]),
                "electron_temperature_uncertainty_eV":float(np.sqrt(max(total_physical_covariance[te_index,te_index],0.0))),
                "electron_velocity_projection_uncertainty_m_s":float(np.sqrt(max(total_physical_covariance[velocity_index,velocity_index],0.0))),
                "fitted_spectrum":joint_fitted[channel].copy(),
                "residual":(y[channel]-joint_fitted[channel]).copy(),
                "joint_fit_used":True,
                "joint_channel_parameter_covariance":parameter_covariance.copy(),
                "model_discrepancy_uncertainty_available": bool(discrepancy_scale > 0.0),
                "model_discrepancy_covariance_included": bool(discrepancy_scale > 0.0),
            })
            if model == "non_collective_kappa":
                fit["kappa_index_uncertainty"] = float(np.sqrt(max(
                    total_kappa_parameter_covariance[channel, channel], 0.0
                )))
                fit["kappa_parameter_covariance"] = kappa_parameter_covariance.copy()
                fit["total_kappa_parameter_covariance"] = total_kappa_parameter_covariance.copy()
            channel_residual = fit["residual"]
            channel_slice = slice(channel * y.shape[1], (channel + 1) * y.shape[1])
            channel_spectral_covariance = joint_covariance[channel_slice, channel_slice]
            channel_cholesky = np.linalg.cholesky(channel_spectral_covariance)
            channel_whitened_residual = np.linalg.solve(channel_cholesky, channel_residual)
            channel_dof = max(1, channel_residual.size - joint_parameter_count)
            channel_reduced_chi_square = float(
                np.dot(channel_whitened_residual, channel_whitened_residual) / channel_dof
            )
            fit["fit_residual_rms"] = float(np.sqrt(np.mean(channel_residual**2)))
            fit["fit_reduced_chi_square"] = channel_reduced_chi_square
            fit["fit_degrees_of_freedom"] = int(channel_dof)
            fit["joint_fit_reduced_chi_square"] = float(reduced_chi2_joint)
            fit["joint_fit_degrees_of_freedom"] = int(dof_joint)
            fit["joint_fit_accepted"] = True
            fit["fit_acceptance_basis"] = "accepted_joint_generalized_least_squares"
            fit["fit_accepted"] = True
            fit["fit_quality_status"] = "accepted_joint_gls_with_channel_marginal_diagnostics"
            fit["fit_quality_reasons"] = []
            fit["channel_marginal_fit_quality_status"] = (
                "marginal_reduced_chi_square_within_ceiling"
                if channel_reduced_chi_square <= 10.0
                else "marginal_reduced_chi_square_exceeds_ceiling_joint_fit_accepted"
            )
            physical_indices = [te_index, velocity_index]
            fit["physical_parameter_covariance"] = physical_covariance[np.ix_(physical_indices, physical_indices)].copy()
            fit["model_discrepancy_covariance"] = (discrepancy_scale * block).copy()
            fit["model_discrepancy_physical_parameter_covariance"] = discrepancy_covariance[np.ix_(physical_indices, physical_indices)].copy()
            fit["total_fit_covariance_including_discrepancy"] = block + discrepancy_scale * block
            fit["total_physical_parameter_covariance"] = total_physical_covariance[np.ix_(physical_indices, physical_indices)].copy()
            fit["model_discrepancy_scale"] = float(discrepancy_scale)
            fit["model_discrepancy_diagnostics"] = {
                "residual_rms": fit["fit_residual_rms"],
                "channel_marginal_reduced_chi_square": channel_reduced_chi_square,
                "joint_reduced_chi_square": float(reduced_chi2_joint),
                "covariance_included": bool(discrepancy_scale > 0.0),
                "interpretation": "Joint-GLS residual-based model-adequacy covariance; it is not experimental validation.",
            }
            if model=="non_collective_maxwellian":
                area=float(amp*sigma*np.sqrt(2.0*np.pi))
                def area_function(v): return float(v[0]*v[2]*np.sqrt(2*np.pi))
            else:
                df=2.0*kappa-1.0
                scale=sigma*np.sqrt((df-2.0)/df)
                area=float(amp*scale*np.sqrt(df*np.pi)*np.exp(gammaln(kappa-0.5)-gammaln(kappa)))
                def area_function(v):
                    dfv=2.0*v[3]-1.0
                    scalev=v[2]*np.sqrt((dfv-2.0)/dfv)
                    return float(v[0]*scalev*np.sqrt(dfv*np.pi)*np.exp(gammaln(v[3]-0.5)-gammaln(v[3])))
            grad=np.zeros(p.size)
            for j in range(p.size):
                step=1e-5*max(abs(float(p[j])),1.0)
                plus=p.copy(); minus=p.copy(); plus[j]+=step; minus[j]-=step
                grad[j]=(area_function(plus)-area_function(minus))/(2.0*step)
            joint_area_sigma=float(np.sqrt(max(grad@block@grad,0.0)))
            parameter_slice = slice(
                joint_parameter_count * channel,
                joint_parameter_count * (channel + 1),
            )
            joint_area_gradient_matrix[channel, parameter_slice] = grad
            lsf_area_sigma=fit.get("integrated_scattered_power_lsf_uncertainty_W")
            if lsf_area_sigma is not None:
                joint_area_sigma=float(np.hypot(joint_area_sigma,float(lsf_area_sigma)))
            fit["integrated_scattered_power_W"]=area
            fit["integrated_scattered_power_fit_uncertainty_W"]=float(np.sqrt(max(grad@block@grad,0.0)))
            fit["integrated_scattered_power_uncertainty_W"]=joint_area_sigma
            te_external = fit.get("electron_temperature_external_uncertainty_eV")
            te_lsf = fit.get("instrument_response_temperature_uncertainty_eV")
            if te_external is not None:
                fit["electron_temperature_uncertainty_eV"] = float(np.hypot(
                    fit["electron_temperature_uncertainty_eV"], float(te_external)))
            if te_lsf is not None:
                fit["electron_temperature_uncertainty_eV"] = float(np.hypot(
                    fit["electron_temperature_uncertainty_eV"], float(te_lsf)))
            velocity_external = fit.get("electron_velocity_projection_external_uncertainty_m_s")
            velocity_lsf = fit.get("instrument_response_velocity_uncertainty_m_s")
            if velocity_external is not None:
                fit["electron_velocity_projection_uncertainty_m_s"] = float(np.hypot(
                    fit["electron_velocity_projection_uncertainty_m_s"], float(velocity_external)))
            if velocity_lsf is not None:
                fit["electron_velocity_projection_uncertainty_m_s"] = float(np.hypot(
                    fit["electron_velocity_projection_uncertainty_m_s"], float(velocity_lsf)))
            # The absolute density belongs to the final joint-GLS spectrum, not
            # to the superseded independent-channel fit. Re-run the same physical
            # inversion with the final joint area and its uncertainty.
            if fit.get("density_method") and fit.get("density_status") != "not_identifiable_from_spectral_shape_alone":
                density_inputs = fit.get("_density_inputs_for_joint_refit")
                if density_inputs is not None:
                    density_inputs = dict(density_inputs)
                    density_inputs["scattered_power_W"] = area
                    if density_inputs.get("relative_input_covariance") is None:
                        density_inputs["scattered_power_uncertainty_W"] = joint_area_sigma
                    else:
                        covariance = np.asarray(density_inputs["relative_input_covariance"], dtype=float).copy()
                        new_relative_sigma = 0.0 if area == 0.0 else joint_area_sigma / abs(area)
                        if np.allclose(covariance[0, :], 0.0, rtol=0.0, atol=1e-14) and np.allclose(
                            covariance[:, 0], 0.0, rtol=0.0, atol=1e-14
                        ):
                            density_inputs["scattered_power_uncertainty_W"] = joint_area_sigma
                        else:
                            old_sigma = float(np.sqrt(max(covariance[0, 0], 0.0))) * float(
                                results[channel]["integrated_scattered_power_W"]
                            )
                            old_relative_sigma = (
                                0.0 if results[channel]["integrated_scattered_power_W"] == 0.0
                                else old_sigma / abs(float(results[channel]["integrated_scattered_power_W"]))
                            )
                            if old_relative_sigma > 0.0:
                                scale_cov = new_relative_sigma / old_relative_sigma
                                covariance[0, :] *= scale_cov
                                covariance[:, 0] *= scale_cov
                            covariance[0, 0] = new_relative_sigma**2
                            density_inputs["scattered_power_uncertainty_W"] = None
                        density_inputs["relative_input_covariance"] = covariance
                    density_result = density_from_thomson_scattered_power(**density_inputs)
                    fit["electron_density_m3"] = density_result["electron_density_m3"]
                    fit["electron_density_uncertainty_m3"] = density_result["electron_density_uncertainty_m3"]
                    fit["density_uncertainty_model"] = density_result["uncertainty_model"]
                    if density_inputs.get("relative_input_covariance") is not None:
                        fit["integrated_scattered_power_covariance_uncertainty_W"] = joint_area_sigma
                        fit["integrated_scattered_power_uncertainty_W"] = joint_area_sigma
        joint_area_covariance_from_spectral_fit = (
            joint_area_gradient_matrix @ parameter_covariance @ joint_area_gradient_matrix.T
        )
        joint_summary={
            "method":"joint generalized least squares across all channels",
            "channel_major_covariance_order":True,
            "joint_parameter_covariance":parameter_covariance,
            "physical_parameter_order":[name for i in range(nchan) for name in (f"channel_{i}_electron_temperature_eV",f"channel_{i}_electron_velocity_projection_m_s")],
            "joint_parameter_count_per_channel":joint_parameter_count,
            "physical_parameter_covariance":physical_covariance,
            "kappa_parameter_covariance":kappa_parameter_covariance,
            "total_kappa_parameter_covariance":total_kappa_parameter_covariance,
            "kappa_parameter_order": (
                [f"channel_{i}_kappa_index" for i in range(nchan)]
                if model == "non_collective_kappa" else None
            ),
            "model_discrepancy_covariance":discrepancy_covariance,
            "model_discrepancy_uncertainty_available": bool(discrepancy_scale > 0.0),
            "total_physical_parameter_covariance":total_physical_covariance,
            "reduced_chi_square":reduced_chi2_joint,
            "reduced_chi_square_policy":"diagnostic_only_no_hard_acceptance_cutoff_without_an_explicit_noise_and_significance_policy",
            "jacobian_rank":joint_rank,"jacobian_condition_number":joint_condition,
            "degrees_of_freedom":int(dof_joint),
            "model_discrepancy_scale":discrepancy_scale,
            "model_discrepancy_interpretation":"Excess whitened residual variance above unity is reported as a separate covariance component; it is a model-adequacy diagnostic, not experimental validation.",
        }
    for channel_result in results:
        channel_result.pop("_density_inputs_for_joint_refit", None)
    response_area_covariance_output = None
    if lsf_unc is not None:
        channel_count = len(results)
        response_area_covariance_output = np.zeros((channel_count, channel_count), dtype=float)
        if shared_lsf_covariance is not None:
            area_indices = np.asarray([3 * channel + 2 for channel in range(channel_count)], dtype=int)
            response_area_covariance_output = shared_lsf_covariance[
                np.ix_(area_indices, area_indices)
            ].copy()
        else:
            for channel_index, channel_result in enumerate(results):
                area_sigma_lsf = channel_result.get("integrated_scattered_power_lsf_uncertainty_W")
                response_area_covariance_output[channel_index, channel_index] = (
                    float(area_sigma_lsf) ** 2 if area_sigma_lsf is not None else np.nan
                )
    joint_area_covariance_output = None
    if joint_area_covariance_from_spectral_fit is not None:
        joint_area_covariance_output = joint_area_covariance_from_spectral_fit.copy()
        if response_area_covariance_output is not None:
            joint_area_covariance_output += np.nan_to_num(
                response_area_covariance_output, nan=0.0
            )
    elif response_area_covariance_output is not None:
        joint_area_covariance_output = response_area_covariance_output.copy()
        local_fit_variances = np.asarray([
            (
                float(item["integrated_scattered_power_fit_uncertainty_W"]) ** 2
                if item.get("integrated_scattered_power_fit_uncertainty_W") is not None
                else (
                    float(item["integrated_scattered_power_uncertainty_W"]) ** 2
                    if item.get("integrated_scattered_power_uncertainty_W") is not None
                    else np.nan
                )
            )
            for item in results
        ], dtype=float)
        for channel_index, variance in enumerate(local_fit_variances):
            if np.isfinite(variance):
                joint_area_covariance_output[channel_index, channel_index] += variance
    elif joint_density_covariance is not None:
        power_indices_for_area = np.arange(0, 6 * len(results), 6, dtype=int)
        power_rows_zero_for_area = (
            np.allclose(joint_density_covariance[power_indices_for_area, :], 0.0, rtol=0.0, atol=1e-14)
            and np.allclose(joint_density_covariance[:, power_indices_for_area], 0.0, rtol=0.0, atol=1e-14)
        )
        if power_rows_zero_for_area:
            local_fit_variances = np.asarray([
                (
                    float(item["integrated_scattered_power_uncertainty_W"]) ** 2
                    if item.get("integrated_scattered_power_uncertainty_W") is not None
                    else np.nan
                )
                for item in results
            ], dtype=float)
            if np.any(np.isfinite(local_fit_variances)):
                joint_area_covariance_output = np.diag(
                    np.nan_to_num(local_fit_variances, nan=0.0)
                )
    shared_response_density_covariance = None
    if response_area_covariance_output is not None:
        channel_count = len(results)
        area_values_for_response = np.asarray([
            item.get("integrated_scattered_power_W", np.nan) for item in results
        ], dtype=float)
        density_values_for_response = np.asarray([
            np.nan if item.get("electron_density_m3") is None else item["electron_density_m3"]
            for item in results
        ], dtype=float)
        shared_response_density_covariance = np.full(
            (channel_count, channel_count), np.nan, dtype=float
        )
        for row_index in range(channel_count):
            for column_index in range(channel_count):
                area_row = area_values_for_response[row_index]
                area_column = area_values_for_response[column_index]
                density_row = density_values_for_response[row_index]
                density_column = density_values_for_response[column_index]
                area_covariance_value = response_area_covariance_output[row_index, column_index]
                if (
                    np.isfinite(area_row) and area_row != 0.0
                    and np.isfinite(area_column) and area_column != 0.0
                    and np.isfinite(density_row) and np.isfinite(density_column)
                    and np.isfinite(area_covariance_value)
                ):
                    shared_response_density_covariance[row_index, column_index] = (
                        density_row * density_column * area_covariance_value
                        / (area_row * area_column)
                    )
    density_joint_covariance_output = None
    if joint_density_covariance is not None:
        channel_count = len(results)
        density_joint_covariance_output = np.full((channel_count, channel_count), np.nan, dtype=float)
        relative_density_gradient = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        for row_index, row_result in enumerate(results):
            density_row = row_result.get("electron_density_m3")
            if density_row is None:
                continue
            row_slice = slice(6*row_index, 6*(row_index+1))
            for column_index, column_result in enumerate(results):
                density_column = column_result.get("electron_density_m3")
                if density_column is None:
                    continue
                column_slice = slice(6*column_index, 6*(column_index+1))
                relative_block = joint_density_covariance[row_slice, column_slice]
                density_joint_covariance_output[row_index, column_index] = (
                    float(density_row) * float(density_column)
                    * float(relative_density_gradient @ relative_block @ relative_density_gradient)
                )
        power_indices = np.arange(0, 6 * len(results), 6, dtype=int)
        power_rows_zero = (
            np.allclose(joint_density_covariance[power_indices, :], 0.0, rtol=0.0, atol=1e-14)
            and np.allclose(joint_density_covariance[:, power_indices], 0.0, rtol=0.0, atol=1e-14)
        )
        if power_rows_zero and joint_area_covariance_output is not None:
            for row_index, row_result in enumerate(results):
                density_row = row_result.get("electron_density_m3")
                area_row = row_result.get("integrated_scattered_power_W")
                if density_row is None or area_row in (None, 0.0):
                    continue
                for column_index, column_result in enumerate(results):
                    density_column = column_result.get("electron_density_m3")
                    area_column = column_result.get("integrated_scattered_power_W")
                    if density_column is None or area_column in (None, 0.0):
                        continue
                    density_joint_covariance_output[row_index, column_index] += (
                        float(density_row) * float(density_column)
                        * float(joint_area_covariance_output[row_index, column_index])
                        / (float(area_row) * float(area_column))
                    )
        for channel_index, channel_result in enumerate(results):
            if channel_result.get("electron_density_m3") is not None:
                channel_result["electron_density_joint_covariance_diagonal_m6"] = float(
                    density_joint_covariance_output[channel_index, channel_index]
                )
    elif joint_area_covariance_output is not None and any(
        item.get("electron_density_m3") is not None for item in results
    ):
        channel_count = len(results)
        density_joint_covariance_output = np.full((channel_count, channel_count), np.nan, dtype=float)
        for row_index, row_result in enumerate(results):
            density_row = row_result.get("electron_density_m3")
            area_row = row_result.get("integrated_scattered_power_W")
            if density_row is None or area_row in (None, 0.0):
                continue
            sigma_row = row_result.get("electron_density_uncertainty_m3")
            density_joint_covariance_output[row_index, row_index] = (
                float(sigma_row) ** 2 if sigma_row is not None else np.nan
            )
            for column_index, column_result in enumerate(results):
                if row_index == column_index:
                    continue
                density_column = column_result.get("electron_density_m3")
                area_column = column_result.get("integrated_scattered_power_W")
                if density_column is None or area_column in (None, 0.0):
                    continue
                density_joint_covariance_output[row_index, column_index] = (
                    float(density_row) * float(density_column)
                    * float(joint_area_covariance_output[row_index, column_index])
                    / (float(area_row) * float(area_column))
                )
    density_angle_covariance_output = None
    if scattering_angle_uncertainty_rad is not None and any(
        item.get("scattering_angle_relative_sensitivity") is not None
        and item.get("electron_density_m3") is not None for item in results
    ):
        angle_density_sensitivity = np.asarray([
            (
                float(item["electron_density_m3"])
                * float(item["scattering_angle_relative_sensitivity"])
                if item.get("electron_density_m3") is not None
                and item.get("scattering_angle_relative_sensitivity") is not None
                else np.nan
            )
            for item in results
        ], dtype=float)
        density_angle_covariance_output = (
            np.outer(angle_density_sensitivity, angle_density_sensitivity)
            * float(scattering_angle_unc) ** 2
        )
        if density_joint_covariance_output is not None:
            angle_covariance_to_add = density_angle_covariance_output.copy()
            if joint_density_covariance is None:
                # The marginal density errors already include the common angle
                # contribution; only its cross-channel terms are missing here.
                np.fill_diagonal(angle_covariance_to_add, 0.0)
            valid_angle = np.isfinite(angle_covariance_to_add)
            density_joint_covariance_output[valid_angle] += angle_covariance_to_add[valid_angle]
    density_cross_channel_covariance_available = bool(
        (
            density_joint_covariance_output is not None
            or density_angle_covariance_output is not None
        )
        and sum(result.get("electron_density_m3") is not None for result in results) >= 2
    )
    output = {
        "model": model, "channel_count": len(results), "channels": results,
        "joint_inference": joint_summary,
        "radius_m": radii.copy(),
        "electron_temperature_profile_eV": np.array([r.get("electron_temperature_eV", np.nan) for r in results]),
        "electron_temperature_uncertainty_profile_eV": np.array([r.get("electron_temperature_uncertainty_eV", np.nan) for r in results]),
        "electron_density_profile_m3": np.array([r.get("electron_density_m3", np.nan) if r.get("electron_density_m3") is not None else np.nan for r in results]),
        "electron_density_uncertainty_profile_m3": np.array([r.get("electron_density_uncertainty_m3", np.nan) if r.get("electron_density_uncertainty_m3") is not None else np.nan for r in results]),
        "electron_density_joint_covariance_m6": density_joint_covariance_output,
        "integrated_scattered_power_joint_covariance": joint_area_covariance_output,
        "electron_density_angle_covariance_m6": density_angle_covariance_output,
        "electron_density_shared_response_covariance_m6": shared_response_density_covariance,
        "electron_density_external_parameter_covariance_m6": None,
        "electron_density_cross_channel_covariance_available": density_cross_channel_covariance_available,
        "shared_instrument_response_parameter_covariance": shared_lsf_covariance,
        "shared_instrument_response_parameter_covariance_order": shared_lsf_covariance_order,
        "shared_instrument_response_cross_channel_covariance_available": bool(
            shared_lsf_covariance is not None and len(results) >= 2
        ),
        "electron_velocity_projection_profile_m_s": np.array([r.get("electron_velocity_projection_m_s", np.nan) for r in results]),
        "electron_velocity_projection_uncertainty_profile_m_s": np.array([r.get("electron_velocity_projection_uncertainty_m_s", np.nan) for r in results]),
        "limitations": [
            *(
                ["Collective Maxwellian inference uses PlasmaPy's Maxwellian kinetic dielectric response; ion temperatures, composition and flow vectors are fixed external inputs."]
                if model == "collective_maxwellian" else
                ["Collective kappa inference uses the implemented unmagnetized longitudinal dielectric response with projected isotropic kappa electrons and Maxwellian ions; supplied external-input and response-matrix 1-sigma uncertainties are propagated by Monte Carlo, but cross-correlations among distinct external inputs and full channel-major GLS are not implemented."]
                if model == "collective_kappa" else
                ["Relativistic Maxwell-Juttner inference assumes an isotropic rest-frame electron distribution and bulk flow parallel to the scattering vector. Supplied laser-wavelength, scalar-angle or beam-direction-component, incident-polarization-component, and response-matrix uncertainties are propagated by separate Monte Carlo refits when provided; correlations among these distinct external inputs are not modelled."]
                if model == "non_collective_relativistic_maxwell_juttner" else
                []
            ),
            "Absolute non-collective density requires calibrated W/nm spectra and all listed optical geometry/polarization factors; first-order uncertainty propagation includes only uncertainty terms supplied by the researcher.",
            "For non-collective Maxwellian/Kappa/relativistic fits, supplied response-matrix uncertainty is propagated by Monte Carlo; absolute-density uncertainty includes only supplied optical-factor uncertainty terms and excludes unprovided correlations and model discrepancy.",
        ],
    }
    if model in collective_models and any(v is not None for v in (
        ion_t_unc,ion_v_unc,electron_v_unc,lsf_unc,laser_wavelength_uncertainty_nm,
        scattering_angle_uncertainty_rad,incident_direction_unc,scattered_direction_unc
    )):
        if lsf_unc is not None and (callable(instrument_response) or np.asarray(instrument_response).ndim==1):
            raise ValueError("Collective response uncertainty requires a measured 2-D or channel-specific 3-D response matrix")
        rng=np.random.default_rng(91427)
        draws=[]
        for _ in range(int(collective_external_parameter_mc_samples)):
            sampled_laser=float(laser_wavelength_nm)+rng.normal(0.0,laser_wavelength_unc)
            if sampled_laser<=0: continue
            sampled_angle=float(scattering_angle_rad)
            sampled_incident_direction=incident_direction
            sampled_scattered_direction=scattered_direction
            if scattering_angle_uncertainty_rad is not None:
                sampled_angle += float(rng.normal(0.0, scattering_angle_unc))
                if not 0.0 < sampled_angle < np.pi: continue
                # Rotate the scattered beam in the plane defined by the nominal
                # incident/scattered directions, preserving a normalized beam pair.
                incident_unit=np.asarray(incident_direction,dtype=float)
                incident_unit=incident_unit/np.linalg.norm(incident_unit)
                scattered_unit=np.asarray(scattered_direction,dtype=float)
                scattered_unit=scattered_unit/np.linalg.norm(scattered_unit)
                tangent=scattered_unit-np.dot(scattered_unit,incident_unit)*incident_unit
                tangent_norm=np.linalg.norm(tangent)
                if tangent_norm<=1e-12: continue
                tangent=tangent/tangent_norm
                sampled_scattered_direction=np.cos(sampled_angle)*incident_unit+np.sin(sampled_angle)*tangent
            elif incident_direction_unc is not None or scattered_direction_unc is not None:
                incident_draw=np.asarray(incident_direction,dtype=float)
                scattered_draw=np.asarray(scattered_direction,dtype=float)
                if incident_direction_unc is not None:
                    incident_draw=incident_draw+rng.normal(0.0,incident_direction_unc)
                if scattered_direction_unc is not None:
                    scattered_draw=scattered_draw+rng.normal(0.0,scattered_direction_unc)
                if np.linalg.norm(incident_draw)<=0 or np.linalg.norm(scattered_draw)<=0: continue
                incident_draw=incident_draw/np.linalg.norm(incident_draw)
                scattered_draw=scattered_draw/np.linalg.norm(scattered_draw)
                sampled_angle=float(np.arccos(np.clip(np.dot(incident_draw,scattered_draw),-1.0,1.0)))
                if not 0.0 < sampled_angle <= np.pi: continue
                sampled_incident_direction=incident_draw
                sampled_scattered_direction=scattered_draw
            sampled_ion_t=ion_t.copy() if ion_t_unc is None else np.maximum(ion_t+rng.normal(0.0,ion_t_unc),np.finfo(float).tiny)
            sampled_ion_v=ion_v.copy() if ion_v_unc is None else ion_v+rng.normal(0.0,ion_v_unc)
            sampled_electron_v=electron_v.copy() if electron_v_unc is None else electron_v+rng.normal(0.0,electron_v_unc.reshape(1,3))
            sampled_response=instrument_response
            if lsf_unc is not None:
                sampled_response=np.asarray(instrument_response,dtype=float).copy()
                if sampled_response.ndim==2:
                    sampled_response=np.maximum(sampled_response+rng.normal(0.0,lsf_unc),0.0)
                    sums=sampled_response.sum(axis=0)
                    if np.any(sums<=0): continue
                    sampled_response/=sums[None,:]
                else:
                    valid_response=True
                    for channel in range(sampled_response.shape[0]):
                        sigma_response=lsf_unc[channel] if lsf_unc.ndim==3 else lsf_unc
                        sampled_response[channel]=np.maximum(sampled_response[channel]+rng.normal(0.0,sigma_response),0.0)
                        sums=sampled_response[channel].sum(axis=0)
                        if np.any(sums<=0):
                            valid_response=False
                            break
                        sampled_response[channel]/=sums[None,:]
                    if not valid_response: continue
            try:
                draw=analyze_thomson_spectra(
                    wavelength_nm,spectra,radius_m,laser_wavelength_nm=sampled_laser,
                    scattering_angle_rad=sampled_angle,instrument_response=sampled_response,
                    signal_uncertainty=signal_uncertainty,model=model,ion_species=ion_species,
                    ion_fractions=ion_fractions,ion_temperature_eV=sampled_ion_t,
                    ion_velocity_m_s=sampled_ion_v,electron_velocity_m_s=sampled_electron_v,
                    incident_direction=sampled_incident_direction,scattered_direction=sampled_scattered_direction,
                    collective_initial_density_m3=collective_initial_density_m3,
                    collective_initial_temperature_eV=collective_initial_temperature_eV,
                    collective_density_bounds_m3=collective_density_bounds_m3,
                    collective_temperature_bounds_eV=collective_temperature_bounds_eV,
                    collective_kappa_initial=collective_kappa_initial,
                    collective_kappa_bounds=collective_kappa_bounds,
                    collective_external_parameter_mc_samples=20,
                )
                vector=[]
                for channel_result in draw["channels"]:
                    density_value=channel_result.get("electron_density_m3")
                    vector.extend([np.nan if density_value is None else float(density_value),
                                   float(channel_result["electron_temperature_eV"])])
                    if model == "collective_kappa":
                        vector.append(float(channel_result["kappa_index"]))
                    vector.append(float(channel_result["electron_velocity_projection_m_s"]))
                    vector.extend(np.asarray(channel_result["ion_velocity_projection_m_s"],dtype=float).tolist())
                draws.append(vector)
            except (ValueError,RuntimeError,ImportError,np.linalg.LinAlgError):
                continue
        output["collective_external_mc_successes"]=len(draws)
        output["collective_external_mc_requested"]=int(collective_external_parameter_mc_samples)
        required=max(12,int(np.ceil(.7*collective_external_parameter_mc_samples)))
        if len(draws)>=required:
            draw_array=np.asarray(draws,dtype=float)
            valid_columns=np.all(np.isfinite(draw_array),axis=0)
            covariance=np.zeros((draw_array.shape[1],draw_array.shape[1]))
            if np.any(valid_columns):
                covariance[np.ix_(valid_columns,valid_columns)]=np.cov(draw_array[:,valid_columns],rowvar=False,ddof=1)
                for channel in range(len(results)):
                    stride = (4 if model == "collective_kappa" else 3) + len(ion_species)
                    base = channel * stride
                    density_col, temperature_col = base, base + 1
                    if model == "collective_kappa":
                        kappa_col, velocity_col = base + 2, base + 3
                        ion_columns = tuple(range(base + 4, base + 4 + len(ion_species)))
                        results[channel]["kappa_index_uncertainty"] = float(np.hypot(
                            results[channel]["kappa_index_uncertainty"],
                            np.sqrt(max(covariance[kappa_col, kappa_col], 0.0))))
                    else:
                        velocity_col = base + 2
                        ion_columns = tuple(range(base + 3, base + 3 + len(ion_species)))
                    results[channel]["electron_temperature_uncertainty_eV"]=float(np.hypot(
                        results[channel]["electron_temperature_uncertainty_eV"],
                        np.sqrt(max(covariance[temperature_col,temperature_col],0.0))))
                    velocity_mc_sigma=float(np.sqrt(max(covariance[velocity_col,velocity_col],0.0)))
                    base_velocity_sigma=results[channel].get("electron_velocity_projection_uncertainty_m_s")
                    if base_velocity_sigma is None:
                        results[channel]["electron_velocity_projection_uncertainty_m_s"]=(
                            velocity_mc_sigma if electron_v_unc is not None or scattering_angle_uncertainty_rad is not None
                            or incident_direction_unc is not None or scattered_direction_unc is not None else None
                        )
                    else:
                        results[channel]["electron_velocity_projection_uncertainty_m_s"]=float(np.hypot(
                            base_velocity_sigma,velocity_mc_sigma))
                    if results[channel].get("electron_density_m3") is not None:
                        results[channel]["electron_density_uncertainty_m3"]=float(np.hypot(
                            results[channel]["electron_density_uncertainty_m3"],
                            np.sqrt(max(covariance[density_col,density_col],0.0))))
                    ion_uncertainties=[]
                    for ion_col in ion_columns:
                        ion_sigma=float(np.sqrt(max(covariance[ion_col,ion_col],0.0)))
                        ion_uncertainties.append(
                            ion_sigma if ion_v_unc is not None or scattering_angle_uncertainty_rad is not None
                            or incident_direction_unc is not None or scattered_direction_unc is not None else None
                        )
                    results[channel]["ion_velocity_projection_uncertainty_m_s"]=ion_uncertainties
            covariance_order=[]
            for i in range(len(results)):
                covariance_order.extend([
                    f"channel_{i}_electron_density_m3",
                    f"channel_{i}_electron_temperature_eV",
                ])
                if model == "collective_kappa":
                    covariance_order.append(f"channel_{i}_kappa_index")
                covariance_order.append(f"channel_{i}_electron_velocity_projection_m_s")
                covariance_order.extend(
                    f"channel_{i}_ion_{species_index}_velocity_projection_m_s"
                    for species_index in range(len(ion_species))
                )
            output["collective_external_parameter_covariance_order"]=covariance_order
            output["collective_external_parameter_covariance"]=covariance
            external_stride = (4 if model == "collective_kappa" else 3) + len(ion_species)
            density_columns = [channel * external_stride for channel in range(len(results))]
            external_density_covariance = covariance[np.ix_(density_columns, density_columns)].copy()
            for channel_index, channel_result in enumerate(results):
                if channel_result.get("electron_density_m3") is None:
                    external_density_covariance[channel_index, :] = np.nan
                    external_density_covariance[:, channel_index] = np.nan
            output["electron_density_external_parameter_covariance_m6"] = external_density_covariance
            output["electron_density_cross_channel_covariance_available"] = bool(
                sum(result.get("electron_density_m3") is not None for result in results) >= 2
            )
            output["collective_external_uncertainty_propagated"]=True
            output["electron_temperature_uncertainty_profile_eV"]=np.array([r.get("electron_temperature_uncertainty_eV",np.nan) for r in results])
            output["electron_density_uncertainty_profile_m3"]=np.array([r.get("electron_density_uncertainty_m3",np.nan) if r.get("electron_density_m3") is not None else np.nan for r in results])
            output["electron_velocity_projection_uncertainty_profile_m_s"]=np.array([
                np.nan if r.get("electron_velocity_projection_uncertainty_m_s") is None
                else r["electron_velocity_projection_uncertainty_m_s"] for r in results])
            for channel_result in results:
                channel_result["collective_external_parameter_uncertainty_propagated"]=True
                if lsf_unc is not None:
                    channel_result["instrument_response_uncertainty_propagated"]=True
                    channel_result["instrument_response_uncertainty_mc_successful_samples"]=len(draws)
                channel_result["collective_external_parameter_covariance_order"]=covariance_order
        else:
            output["collective_external_uncertainty_propagated"]=False
            output["collective_external_uncertainty_warning"]="Too few successful external-parameter/response perturbation refits."
            for channel_result in results:
                channel_result["collective_external_parameter_uncertainty_propagated"]=False
                channel_result.setdefault("uncertainty_inputs_missing", []).append(
                    "collective_external_parameter_monte_carlo_propagation_failed"
                )
                channel_result["uncertainty_completeness"] = "partial"
                if lsf_unc is not None:
                    channel_result["instrument_response_uncertainty_propagated"]=False
                    channel_result["instrument_response_uncertainty_warning"]=output["collective_external_uncertainty_warning"]
                    channel_result.setdefault("uncertainty_inputs_missing", []).append(
                        "instrument_response_uncertainty_propagation_failed"
                    )
                    channel_result["uncertainty_completeness"] = "partial"

    # Attach model-independent observed spectral-distribution diagnostics to every
    # channel, using the background nuisance term already fitted by the selected
    # physical forward model. This avoids a second, unrelated edge-background fit.
    from .thomson_distribution import analyze_thomson_spectral_distribution
    from .thomson_evdf import reconstruct_thomson_evdf
    if not isinstance(enable_evdf_reconstruction, (bool, np.bool_)):
        raise ValueError("enable_evdf_reconstruction must be boolean")
    if isinstance(evdf_velocity_points, (bool, np.bool_)) or not isinstance(evdf_velocity_points, (int, np.integer)) or not 41 <= evdf_velocity_points <= 2001:
        raise ValueError("evdf_velocity_points must be an integer between 41 and 2001")
    spectral_distribution_results = []
    evdf_results = []
    for channel_index, channel_result in enumerate(results):
        channel_uncertainty = None if unc is None else unc[channel_index]
        channel_covariance = None
        if joint_covariance is not None:
            channel_slice = slice(channel_index * y.shape[1], (channel_index + 1) * y.shape[1])
            channel_covariance = joint_covariance[channel_slice, channel_slice]
            channel_uncertainty = None
        if not bool(channel_result.get("fit_accepted", False)):
            distribution = {
                "analysis_status": "unavailable_primary_fit_rejected",
                "reason": (
                    "The selected primary Thomson physical fit was rejected; its fitted background "
                    "nuisance parameters are not used as a basis for same-spectrum subanalysis."
                ),
                "interpretation": (
                    "The calibrated spectrum remains available, but this parent workflow does not "
                    "present dependent spectral-shape diagnostics as completed when the physical "
                    "forward fit supplying their background has been rejected."
                ),
            }
        else:
            background_reference_nm = float(
                channel_result.get("fitted_center_nm", laser_wavelength_nm)
            )
            background_basis = channel_result.get("background_basis")
            background_coefficients = channel_result.get("background_coefficients")
            if background_basis is not None and background_coefficients is not None:
                basis_array = np.asarray(background_basis, dtype=float)
                coefficient_array = np.asarray(background_coefficients, dtype=float)
            else:
                basis_array = coefficient_array = None
            if (basis_array is not None and basis_array.ndim == 2
                    and basis_array.shape[0] == xs.shape[1]
                    and coefficient_array.shape == (basis_array.shape[1],)):
                fitted_background = basis_array @ coefficient_array
            else:
                fitted_background = (
                    float(channel_result["background_intercept"])
                    + float(channel_result["background_slope_per_nm"])
                    * (xs[channel_index] - background_reference_nm)
                )
            channel_result["fitted_background_spectrum"] = np.asarray(
                fitted_background, dtype=float
            ).copy()
            try:
                distribution = analyze_thomson_spectral_distribution(
                    xs[channel_index],
                    y[channel_index],
                    instrument_response_matrix=channel_responses[channel_index],
                    signal_uncertainty=channel_uncertainty,
                    signal_covariance=channel_covariance,
                    background_signal=fitted_background,
                    physical_model=(
                        "non_collective_maxwellian"
                        if model == "non_collective_maxwellian" else None
                    ),
                    laser_wavelength_nm=(
                        laser_wavelength_nm if model == "non_collective_maxwellian" else None
                    ),
                    scattering_angle_rad=(
                        scattering_angle_rad if model == "non_collective_maxwellian" else None
                    ),
                    laser_wavelength_uncertainty_nm=(
                        laser_wavelength_uncertainty_nm if model == "non_collective_maxwellian" else None
                    ),
                    scattering_angle_uncertainty_rad=(
                        scattering_angle_uncertainty_rad if model == "non_collective_maxwellian" else None
                    ),
                )
                distribution["analysis_status"] = "completed"
                distribution["background_source"] = (
                    "background nuisance component from the selected Thomson physical forward fit"
                )
                distribution["interpretation"] = (
                    "Physical electron temperature and identifiable density are exposed from the selected physical forward fit of this same calibrated measured spectrum; "
                    "electron flow is reported only when identifiable in the selected model. "
                    "Wavelength-space moments and the Gaussian shape fit remain complementary spectral-shape descriptors, "
                    "not a unique arbitrary electron velocity-distribution inversion."
                )
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                distribution = {
                    "analysis_status": "unavailable_for_this_channel",
                    "reason": str(exc),
                    "interpretation": (
                        "The primary physical Thomson inference remains available; spectral-distribution "
                        "diagnostics require sufficient wavelength coverage and a positive signed area/variance."
                    ),
                }
        channel_result["spectral_distribution"] = distribution
        spectral_distribution_results.append(distribution)

        # Reconstruct the maximum identifiable 1-D velocity marginal directly
        # from the same measured spectrum. The fitted physical-model background,
        # measured response and supplied noise covariance are reused; regularization
        # is selected by GCV rather than requested as an arbitrary user constant.
        if not bool(channel_result.get("fit_accepted", False)):
            evdf = {
                "analysis_status": "unavailable_primary_fit_rejected",
                "reason": (
                    "The selected primary Thomson physical fit was rejected; its fitted background "
                    "nuisance parameters are not used as a basis for projected-EVDF inversion."
                ),
            }
        elif not enable_evdf_reconstruction:
            evdf = {
                "analysis_status": "not_requested",
                "reason": "EVDF reconstruction was explicitly disabled.",
            }
        elif model == "non_collective_relativistic_maxwell_juttner":
            try:
                from .thomson_gap_closure import (
                    reconstruct_relativistic_projected_thomson_evdf,
                    relativistic_projected_compton_kinematics,
                )
                wavelength_axis = np.asarray(xs[channel_index], dtype=float)
                # Invert the same exact relativistic single-electron wavelength
                # map used by the selected forward model to set the velocity
                # support from the measured spectral-bin interior.
                low_wavelength = float(
                    wavelength_axis[0] + 0.5*(wavelength_axis[1]-wavelength_axis[0])
                )
                high_wavelength = float(
                    wavelength_axis[-1] - 0.5*(wavelength_axis[-1]-wavelength_axis[-2])
                )

                source_reference_wavelength_nm = float(laser_wavelength_nm)
                if laser_spectrum_wavelength_nm is not None:
                    source_axis = np.asarray(laser_spectrum_wavelength_nm, dtype=float)
                    source_density = np.asarray(laser_spectrum_density_per_nm, dtype=float)
                    source_area = float(trapezoid(source_density, source_axis))
                    if not np.isfinite(source_area) or source_area <= 0.0:
                        raise ValueError("measured laser spectrum must have positive integrated area")
                    source_reference_wavelength_nm = float(
                        trapezoid(source_axis * source_density, source_axis) / source_area
                    )

                def mapped_wavelength_for_velocity(velocity_m_s):
                    step = max(1.0, abs(float(velocity_m_s))*1e-10)
                    probe = np.array([float(velocity_m_s), float(velocity_m_s)+step])
                    mapped = relativistic_projected_compton_kinematics(
                        probe, source_reference_wavelength_nm,
                        float(scattering_angle_rad),
                    )
                    key = (
                        "compton_scattered_wavelength_nm"
                        if include_compton_recoil else
                        "thomson_scattered_wavelength_nm"
                    )
                    return float(mapped[key][0])

                velocity_bracket = (-0.99*C_M_S, 0.99*C_M_S)
                velocity_low = brentq(
                    lambda value: mapped_wavelength_for_velocity(value)-high_wavelength,
                    *velocity_bracket, xtol=1e-3,
                )
                velocity_high = brentq(
                    lambda value: mapped_wavelength_for_velocity(value)-low_wavelength,
                    *velocity_bracket, xtol=1e-3,
                )
                if not velocity_high > velocity_low:
                    raise ValueError(
                        "Measured wavelength coverage does not map to an increasing projected-velocity interval."
                    )
                velocity_grid = np.linspace(
                    velocity_low, velocity_high, int(evdf_velocity_points)
                )
                evdf = reconstruct_relativistic_projected_thomson_evdf(
                    wavelength_axis,
                    y[channel_index],
                    laser_wavelength_nm=float(laser_wavelength_nm),
                    scattering_angle_rad=float(scattering_angle_rad),
                    instrument_response_matrix=channel_responses[channel_index],
                    background_signal=fitted_background,
                    velocity_grid_m_s=velocity_grid,
                    signal_uncertainty=channel_uncertainty,
                    signal_covariance=channel_covariance,
                    include_compton_recoil=include_compton_recoil,
                    incident_polarization=incident_polarization,
                    maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
                    regularization_strength=None,
                    laser_spectrum_wavelength_nm=laser_spectrum_wavelength_nm,
                    laser_spectrum_density_per_nm=laser_spectrum_density_per_nm,
                )
                evdf["laser_spectral_profile"]["source_profile_used"] = laser_spectrum_wavelength_nm is not None
                evdf["analysis_status"] = (
                    "completed"
                    if evdf["primary_result_status"] == "identified_from_measurement"
                    else "conditional_regularized_candidate"
                )
                evdf["forward_model_recoil_included"] = bool(include_compton_recoil)
                evdf["background_source"] = (
                    "full detector-space nuisance basis fitted by the selected relativistic Thomson physical forward model"
                )
                evdf["interpretation"] = (
                    "A relativistically mapped, normalized one-dimensional electron-velocity marginal along "
                    "the scattering vector. The Compton route uses the same recoil switch and incident "
                    "polarization as the selected forward fit. A regularized candidate is not promoted to "
                    "a primary measurement when the data-only inverse is rank deficient. A single view "
                    "does not identify transverse velocity structure or an unrestricted 3-D EVDF."
                )
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                evdf = {
                    "analysis_status": "unavailable_for_this_channel",
                    "reason": str(exc),
                    "identifiability": {
                        "primary": [],
                        "conditional": [
                            "A stable relativistic projected EVDF requires adequate wavelength coverage, measured response, polarization for the Compton route, and sufficient spectral information."
                        ],
                        "non_identifiable": [
                            "An unrestricted three-dimensional relativistic EVDF from one scattering geometry."
                        ],
                    },
                }
        else:
            try:
                q_over_ki = 2.0 * np.sin(float(scattering_angle_rad) / 2.0)
                velocity_axis = C_M_S * (
                    1.0 - np.asarray(xs[channel_index], dtype=float) / float(laser_wavelength_nm)
                ) / q_over_ki
                velocity_low = max(float(np.min(velocity_axis)), -0.099 * C_M_S)
                velocity_high = min(float(np.max(velocity_axis)), 0.099 * C_M_S)
                if not velocity_high > velocity_low:
                    raise ValueError("Measured wavelength coverage does not span a finite projected-velocity interval.")
                velocity_grid = np.linspace(velocity_low, velocity_high, int(evdf_velocity_points))
                evdf = reconstruct_thomson_evdf(
                    xs[channel_index],
                    y[channel_index],
                    laser_wavelength_nm=float(laser_wavelength_nm),
                    scattering_angle_rad=float(scattering_angle_rad),
                    instrument_response_matrix=channel_responses[channel_index],
                    background_signal=fitted_background,
                    regularization_strength=None,
                    velocity_grid_m_s=velocity_grid,
                    velocity_points=int(evdf_velocity_points),
                    signal_uncertainty=channel_uncertainty,
                    signal_covariance=channel_covariance,
                )
                evdf["laser_spectral_profile"] = (
                    source_profile_metadata if source_profile_metadata is not None
                    else {"assumption": "monochromatic_incident_laser"}
                )
                evdf["analysis_status"] = "completed"
                evdf["background_source"] = "nuisance background fitted by the selected primary Thomson physical forward model"
                evdf["interpretation"] = (
                    "A normalized one-dimensional electron-velocity marginal along the scattering vector, "
                    "inferred from the measured calibrated spectrum. It is not an unrestricted 3-D EVDF; "
                    "directional equivalent temperature is not automatically the scalar electron temperature."
                )
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                evdf = {
                    "analysis_status": "unavailable_for_this_channel",
                    "reason": str(exc),
                    "identifiability": {
                        "primary": [],
                        "conditional": ["A stable projected EVDF requires sufficient measured wavelength coverage, response information and spectral signal-to-noise."],
                        "non_identifiable": ["unrestricted three-dimensional EVDF from one scattering geometry"],
                    },
                }
        channel_result["electron_velocity_distribution"] = evdf
        evdf_results.append(evdf)
        # Reuse the EVDF already reconstructed by the parent API and close its
        # first two velocity moments against the same-channel measured spectrum.
        # This avoids a duplicate inverse solve and keeps API/Dashboard output aligned.
        if (
            model in {"non_collective_maxwellian", "non_collective_kappa"}
            and distribution.get("analysis_status") == "completed"
            and evdf.get("analysis_status") in {"completed", "conditional_regularized_candidate"}
            and "velocity_parallel_m_s" in evdf
            and "evdf_parallel_s_per_m" in evdf
        ):
            velocity_grid = np.asarray(evdf["velocity_parallel_m_s"], dtype=float)
            projected_pdf = np.asarray(evdf["evdf_parallel_s_per_m"], dtype=float)
            if (velocity_grid.ndim == 1 and projected_pdf.shape == velocity_grid.shape
                    and np.all(np.isfinite(velocity_grid))
                    and np.all(np.isfinite(projected_pdf)) and np.all(projected_pdf >= 0.0)
                    and np.all(np.diff(velocity_grid) > 0.0)):
                evdf_norm = float(trapezoid(projected_pdf, velocity_grid))
                if np.isfinite(evdf_norm) and evdf_norm > 0.0:
                    pdf = projected_pdf / evdf_norm
                    mean_v = float(trapezoid(velocity_grid * pdf, velocity_grid))
                    variance_v = float(trapezoid((velocity_grid - mean_v)**2 * pdf, velocity_grid))
                    q = 2.0 * np.sin(float(scattering_angle_rad) / 2.0)
                    if float(np.max(np.abs(velocity_grid))) < 0.1 * C_M_S:
                        predicted_center = float(laser_wavelength_nm) * (1.0 - q * mean_v / C_M_S)
                        predicted_variance = (float(laser_wavelength_nm) * q / C_M_S)**2 * variance_v
                        distribution["projected_evdf_spectral_moment_closure"] = {
                            "status": "conditional_moment_closure",
                            "formula": "lambda=lambda_0*(1-q*v_parallel/c), q=2*sin(theta/2)",
                            "evdf_mean_velocity_parallel_m_s": mean_v,
                            "evdf_variance_velocity_parallel_m2_s2": variance_v,
                            "evdf_predicted_spectral_centroid_nm": predicted_center,
                            "evdf_predicted_spectral_variance_nm2": predicted_variance,
                            "measured_spectral_centroid_nm": float(distribution["spectral_centroid_nm"]),
                            "measured_spectral_variance_nm2": float(distribution["observed_variance_nm2"]),
                            "centroid_closure_residual_nm": float(distribution["spectral_centroid_nm"] - predicted_center),
                            "variance_closure_residual_nm2": float(distribution["observed_variance_nm2"] - predicted_variance),
                            "evdf_status": evdf["analysis_status"],
                            "interpretation": "Conditional forward-moment consistency check; finite window, background, response and regularization must be considered. It does not convert Gaussian width into a definitive electron temperature.",
                        }
        evdf_results.append(evdf)
    output["spectral_distribution"] = spectral_distribution_results
    output["electron_velocity_distribution"] = evdf_results

    # A fitted spectral integral is not a watt-valued scattered power unless the
    # measured spectral density is explicitly calibrated in W/nm. Preserve the
    # numerical integral while making its actual unit contract unambiguous.
    output["absolute_scattered_power_available"] = bool(spectral_power_density_calibrated)
    if not spectral_power_density_calibrated:
        for channel_result in results:
            area = channel_result.pop("integrated_scattered_power_W", None)
            area_uncertainty = channel_result.pop("integrated_scattered_power_uncertainty_W", None)
            if area is not None:
                channel_result["integrated_spectral_signal_input_units_nm"] = area
            if area_uncertainty is not None:
                channel_result["integrated_spectral_signal_uncertainty_input_units_nm"] = area_uncertainty
            for old_key, new_key in (
                ("integrated_scattered_power_fit_uncertainty_W", "integrated_spectral_signal_fit_uncertainty_input_units_nm"),
                ("integrated_scattered_power_lsf_uncertainty_W", "integrated_spectral_signal_lsf_uncertainty_input_units_nm"),
                ("integrated_scattered_power_covariance_uncertainty_W", "integrated_spectral_signal_covariance_uncertainty_input_units_nm"),
            ):
                if old_key in channel_result:
                    channel_result[new_key] = channel_result.pop(old_key)
            channel_result["integrated_spectral_signal_unit"] = "input spectral-intensity unit × nm"
            channel_result["absolute_scattered_power_available"] = False
    from .thomson_identifiability import optical_thomson_identifiability
    from .thomson_identifiability import optical_model_validity
    for channel_result in results:
        channel_result["model_validity"] = optical_model_validity(
            model,
            electron_temperature_eV=channel_result.get("electron_temperature_eV"),
            electron_velocity_projection_m_s=channel_result.get("electron_velocity_projection_m_s"),
            kappa_index=channel_result.get("kappa_index"),
        )
    output["model_validity"] = [
        channel_result["model_validity"] for channel_result in results
    ]
    output["parameter_identifiability"] = optical_thomson_identifiability(
        model,
        absolute_density_available=any(
            channel_result.get("electron_density_m3") is not None
            for channel_result in results
        ),
    )
    output["parameter_identifiability"]["profile_scope"] = (
        "Per-channel physical inference only; spatial interpolation is a separate child subanalysis."
    )

    # Machine-readable result contract: measured-spectrum inference is separated
    # from conditional quantities and fundamental single-view degeneracies.
    # Values remain in the channel/profile result fields above; this block only
    # classifies their evidential status and states the auxiliary inputs needed.
    channel_classification = []
    for channel_index, channel_result in enumerate(results):
        primary = []
        conditional = []
        non_identifiable = [
            {
                "parameter": "unrestricted_three_dimensional_electron_velocity_distribution",
                "reason": "A single Thomson scattering geometry measures a projection along its scattering vector; perpendicular velocity structure is not uniquely identifiable.",
            }
        ]
        for parameter, unit in (
            ("electron_temperature_eV", "eV"),
            ("electron_velocity_projection_m_s", "m s^-1"),
        ):
            value = channel_result.get(parameter)
            if value is not None and np.isfinite(value):
                primary.append({"parameter": parameter, "unit": unit, "status": "identified_from_measured_spectrum"})
        density_value = channel_result.get("electron_density_m3")
        if density_value is not None and np.isfinite(density_value):
            primary.append({
                "parameter": "electron_density_m3",
                "unit": "m^-3",
                "status": "identified_from_absolute_calibrated_scattered_power",
            })
        else:
            conditional.append({
                "parameter": "electron_density_m3",
                "unit": "m^-3",
                "required_auxiliary_inputs": [
                    "absolute spectral power density in W/nm",
                    "laser power",
                    "interaction length",
                    "collection solid angle",
                    "optical efficiency",
                    "incident/scattered beam geometry and polarization factors required by the selected cross-section",
                ],
                "reason": "Spectral shape alone does not determine absolute electron density.",
            })
        evdf = channel_result.get("electron_velocity_distribution", {})
        if evdf.get("analysis_status") == "completed":
            primary.append({
                "parameter": "projected_one_dimensional_electron_velocity_distribution",
                "unit": "normalized probability density per (m s^-1)",
                "status": "inferred_from_measured_spectrum_conditional_on_selected_nonrelativistic_inverse",
            })
        elif evdf.get("analysis_status") not in {"not_requested"}:
            conditional.append({
                "parameter": "projected_one_dimensional_electron_velocity_distribution",
                "unit": "normalized probability density per (m s^-1)",
                "required_auxiliary_inputs": [
                    "sufficient measured wavelength coverage",
                    "measured spectral response operator",
                    "spectral uncertainty or covariance",
                    "a physical forward model compatible with the selected Thomson model",
                ],
                "reason": evdf.get("reason", "The projected inverse is unavailable for this channel/model."),
            })
        channel_classification.append({
            "channel_index": int(channel_index),
            "primary": primary,
            "conditional": conditional,
            "non_identifiable": non_identifiable,
        })
    output["result_classification"] = {
        "contract_version": "1.0",
        "input_contract": "measured_device_spectra_assumed_prepared_and_calibrated",
        "instrument_calibration_or_correction_performed": False,
        "experimental_validation_claimed": False,
        "channels": channel_classification,
        "classification_semantics": {
            "primary": "Identifiable from the supplied measured spectrum and explicitly supplied fixed model metadata.",
            "conditional": "Not emitted as a measured result until the listed auxiliary inputs or compatible conditions are supplied.",
            "non_identifiable": "Not uniquely recoverable from the supplied measurement geometry/model; no synthetic value is reported.",
        },
    }
    return output
