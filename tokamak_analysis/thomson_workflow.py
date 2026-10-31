"""Thomson spectral-inference entry point and downstream calibrated-profile utilities.

The primary Analysis 1 input is measured, calibrated multi-channel Thomson spectra.
The legacy profile-summary function remains available only for downstream use when
profiles have been obtained independently.
"""
from __future__ import annotations

import numpy as np

from .thomson_scattering import (
    build_profile,
    profile_summary,
    _covariance_is_positive_semidefinite,
)


def _gradient_roundoff_threshold(derivative_matrix, values):
    """Estimate a pointwise floating-point cancellation bound for each derivative.

    The bound follows the absolute terms in the local finite-difference dot
    product. It is therefore local to each coordinate and profile scale, rather
    than being controlled by the largest gradient elsewhere in the profile.
    """
    absolute_terms = np.abs(derivative_matrix) @ np.abs(values)
    eps = np.finfo(float).eps
    sample_count = values.size
    gamma = (sample_count * eps) / (1.0 - sample_count * eps)
    return gamma * absolute_terms


def _scale_length(value, gradient, gradient_threshold):
    """Return -value/gradient where the local derivative exceeds round-off."""
    out = np.full(value.shape, np.nan, dtype=float)
    valid = (
        (value > 0)
        & np.isfinite(gradient)
        & np.isfinite(gradient_threshold)
        & (np.abs(gradient) > gradient_threshold)
    )
    out[valid] = -value[valid] / gradient[valid]
    return out


def _gradient_uncertainty(derivative_matrix, uncertainty, covariance=None):
    if uncertainty is None:
        return None, None
    input_covariance = np.diag(np.square(uncertainty)) if covariance is None else np.asarray(covariance, dtype=float)
    gradient_covariance = derivative_matrix @ input_covariance @ derivative_matrix.T
    variance = np.diag(gradient_covariance)
    standard_uncertainty = np.sqrt(np.maximum(variance, 0.0))
    covariance_value_gradient = np.diag(input_covariance @ derivative_matrix.T)
    return standard_uncertainty, covariance_value_gradient


def _normalized_gradient_uncertainty(value, gradient, value_uncertainty, gradient_uncertainty, covariance):
    out = np.full(value.shape, np.nan, dtype=float)
    valid = value > 0
    if not np.any(valid):
        return out
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        variance = (
            gradient_uncertainty[valid]**2 / value[valid]**2
            + gradient[valid]**2 * value_uncertainty[valid]**2 / value[valid]**4
            - 2.0 * gradient[valid] * covariance[valid] / value[valid]**3
        )
    out[valid] = np.sqrt(np.maximum(variance, 0.0))
    return out


def _scale_length_uncertainty(
    value, gradient, value_uncertainty, gradient_uncertainty, covariance,
    gradient_threshold,
):
    out = np.full(value.shape, np.nan, dtype=float)
    valid = (
        (value > 0)
        & np.isfinite(gradient)
        & np.isfinite(gradient_threshold)
        & (np.abs(gradient) > gradient_threshold)
    )
    if not np.any(valid):
        return out
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        variance = (
            value_uncertainty[valid]**2 / gradient[valid]**2
            + value[valid]**2 * gradient_uncertainty[valid]**2 / gradient[valid]**4
            - 2.0 * value[valid] * covariance[valid] / gradient[valid]**3
        )
    out[valid] = np.sqrt(np.maximum(variance, 0.0))
    return out


def analyze_thomson_profile(
    radius_m,
    density_m3,
    temperature_ev,
    *,
    density_uncertainty_m3=None,
    temperature_uncertainty_ev=None,
    density_covariance_m3_2=None,
    temperature_covariance_eV2=None,
):
    """Calculate profile summaries and gradients with pointwise or full covariance.

    Full covariance matrices are ordered like the supplied profile points and are
    reordered together with the radius/density/temperature arrays.
    """
    radius_raw = np.asarray(radius_m)
    if radius_raw.ndim != 1 or radius_raw.dtype.kind not in "iuf":
        raise ValueError("radius_m must be a one-dimensional real numeric array")
    radius_input = radius_raw.astype(float, copy=False)
    if not np.all(np.isfinite(radius_input)):
        raise ValueError("radius_m must contain only finite values")
    order = np.argsort(radius_input)

    def validate_profile_covariance(value, name, supplied_uncertainty):
        if value is None:
            return None, supplied_uncertainty
        covariance_raw = np.asarray(value)
        if covariance_raw.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numeric values, not booleans or strings")
        covariance = covariance_raw.astype(float, copy=False)
        expected = (radius_input.size, radius_input.size)
        if covariance.shape != expected or not np.all(np.isfinite(covariance)):
            raise ValueError(f"{name} must be a finite square covariance matrix matching the profile length")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0.0):
            raise ValueError(f"{name} must be symmetric")
        if not _covariance_is_positive_semidefinite(covariance):
            raise ValueError(f"{name} must be positive semidefinite")
        diagonal_sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        if supplied_uncertainty is not None:
            supplied = np.asarray(supplied_uncertainty, dtype=float)
            if supplied.shape != diagonal_sigma.shape or not np.allclose(
                supplied**2, np.diag(covariance), rtol=1e-6, atol=1e-24
            ):
                raise ValueError(f"Pointwise uncertainty must match the diagonal of {name}")
        else:
            supplied_uncertainty = diagonal_sigma
        return covariance[np.ix_(order, order)], supplied_uncertainty

    density_covariance, density_uncertainty_m3 = validate_profile_covariance(
        density_covariance_m3_2, "density_covariance_m3_2", density_uncertainty_m3
    )
    temperature_covariance, temperature_uncertainty_ev = validate_profile_covariance(
        temperature_covariance_eV2, "temperature_covariance_eV2", temperature_uncertainty_ev
    )
    p = build_profile(
        radius_m, density_m3, temperature_ev,
        density_uncertainty_m3, temperature_uncertainty_ev,
    )
    r, ne, te = p.radius_m, p.density_m3, p.temperature_ev
    edge_order = 2 if r.size >= 3 else 1
    derivative_matrix = np.gradient(np.eye(r.size), r, axis=0, edge_order=edge_order)
    dne_dr = derivative_matrix @ ne
    dte_dr = derivative_matrix @ te
    density_gradient_threshold = _gradient_roundoff_threshold(derivative_matrix, ne)
    temperature_gradient_threshold = _gradient_roundoff_threshold(derivative_matrix, te)
    if p.density_uncertainty_m3 is not None:
        density_gradient_sigma, _ = _gradient_uncertainty(
            derivative_matrix, p.density_uncertainty_m3, density_covariance
        )
        density_gradient_threshold = np.maximum(density_gradient_threshold, 2.0 * density_gradient_sigma)
    if p.temperature_uncertainty_ev is not None:
        temperature_gradient_sigma, _ = _gradient_uncertainty(
            derivative_matrix, p.temperature_uncertainty_ev, temperature_covariance
        )
        temperature_gradient_threshold = np.maximum(temperature_gradient_threshold, 2.0 * temperature_gradient_sigma)
    density_normalized_gradient = np.divide(
        dne_dr, ne, out=np.full_like(dne_dr, np.nan), where=ne > 0
    )
    temperature_normalized_gradient = np.divide(
        dte_dr, te, out=np.full_like(dte_dr, np.nan), where=te > 0
    )
    result = {
        "radius_m": r.copy(),
        "density_m3": ne.copy(),
        "temperature_eV": te.copy(),
        "density_gradient_m-4": dne_dr,
        "temperature_gradient_eV_m-1": dte_dr,
        "density_normalized_gradient_m-1": density_normalized_gradient,
        "temperature_normalized_gradient_m-1": temperature_normalized_gradient,
        "density_scale_length_m": _scale_length(ne, dne_dr, density_gradient_threshold),
        "temperature_scale_length_m": _scale_length(te, dte_dr, temperature_gradient_threshold),
        "summary": profile_summary(p),
        "scale_length_gradient_significance_threshold_sigma": 2.0,
        "notes": [
            "Input must be calibrated Thomson-scattering profile data.",
            "When pointwise uncertainties are supplied, a scale length is reported only where the local gradient exceeds twice its propagated 1-sigma uncertainty; otherwise the numerical round-off threshold is used.",
            "Full density and temperature covariance matrices are propagated through the gradient operator when supplied; otherwise pointwise uncertainties are treated as independent.",
            "NaN scale lengths mark zero, unresolved or statistically insignificant gradients or profile values.",
            "Peak-location uncertainty is not estimated by this workflow.",
        ],
    }
    if p.density_uncertainty_m3 is not None:
        sne = p.density_uncertainty_m3
        grad_unc, cov = _gradient_uncertainty(derivative_matrix, sne, density_covariance)
        result["density_uncertainty_m3"] = sne.copy()
        result["density_input_covariance_m3_2"] = (
            density_covariance.copy() if density_covariance is not None else np.diag(sne**2)
        )
        result["density_gradient_uncertainty_m-4"] = grad_unc
        density_input_covariance = density_covariance if density_covariance is not None else np.diag(sne**2)
        result["density_gradient_covariance_m-8"] = (
            derivative_matrix @ density_input_covariance @ derivative_matrix.T
        )
        result["density_normalized_gradient_uncertainty_m-1"] = _normalized_gradient_uncertainty(
            ne, dne_dr, sne, grad_unc, cov
        )
        result["density_scale_length_uncertainty_m"] = _scale_length_uncertainty(
            ne, dne_dr, sne, grad_unc, cov, density_gradient_threshold
        )
    if p.temperature_uncertainty_ev is not None:
        ste = p.temperature_uncertainty_ev
        grad_unc, cov = _gradient_uncertainty(derivative_matrix, ste, temperature_covariance)
        result["temperature_uncertainty_eV"] = ste.copy()
        result["temperature_input_covariance_eV2"] = (
            temperature_covariance.copy() if temperature_covariance is not None else np.diag(ste**2)
        )
        result["temperature_gradient_uncertainty_eV_m-1"] = grad_unc
        temperature_input_covariance = temperature_covariance if temperature_covariance is not None else np.diag(ste**2)
        result["temperature_gradient_covariance_eV2_m-2"] = (
            derivative_matrix @ temperature_input_covariance @ derivative_matrix.T
        )
        result["temperature_normalized_gradient_uncertainty_m-1"] = _normalized_gradient_uncertainty(
            te, dte_dr, ste, grad_unc, cov
        )
        result["temperature_scale_length_uncertainty_m"] = _scale_length_uncertainty(
            te, dte_dr, ste, grad_unc, cov, temperature_gradient_threshold
        )
    return result


def analyze_thomson_spectra(*args, **kwargs):
    """Primary Analysis 1 entry point: infer plasma parameters from measured spectra.

    The spectral forward model is maintained in thomson_inference to keep the
    legacy calibrated-profile summary utilities below available as downstream
    operations without confusing their input contract with the primary analysis.
    """
    from .thomson_inference import analyze_thomson_spectra as _infer
    return _infer(*args, **kwargs)



def reconstruct_thomson_profile_from_spectra(*args, **kwargs):
    """Return channel-resolved Thomson profiles inferred from measured spectra.

    This registered subanalysis accepts the same measured, calibrated
    multi-channel spectra as the canonical Analysis 1 engine. It does not accept
    previously inferred density/temperature profile points.
    """
    result = analyze_thomson_spectra(*args, **kwargs)
    result["registered_subanalysis"] = "Thomson spatial profile reconstruction from calibrated spectra"
    result["profile_source"] = "physical inference from measured calibrated multi-channel Thomson spectra"
    return result
