"""Physical Thomson-scattering primitives used by measured-spectrum inference.

The analysis input is measured diagnostic spectra supplied in user-prepared
calibrated wavelength/intensity coordinates. This module contains physical
cross-section and absolute-density relations; it does not accept pre-derived
electron-density or electron-temperature profile samples.
"""
from __future__ import annotations

import numpy as np

def _covariance_is_positive_semidefinite(covariance, *, rtol=1e-10):
    """Check PSD after scaling to a dimensionless correlation matrix.

    Raw covariance eigenvalues are unsafe when a joint matrix mixes quantities
    with very different units, such as density in m^-3 and temperature in eV.
    """
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
        return False
    diagonal = np.diag(matrix)
    if np.any(diagonal < 0.0):
        return False
    standard = np.sqrt(diagonal)
    positive = np.flatnonzero(standard > 0.0)
    zero = np.flatnonzero(standard == 0.0)
    if zero.size and np.any(matrix[zero, :] != 0.0):
        return False
    correlation = np.zeros_like(matrix)
    if positive.size:
        scale = np.outer(standard[positive], standard[positive])
        correlation[np.ix_(positive, positive)] = matrix[np.ix_(positive, positive)] / scale
    eigenvalues = np.linalg.eigvalsh(correlation)
    return bool(float(np.min(eigenvalues)) >= -float(rtol))

# Classical Thomson-scattering differential cross-section and calibrated-power density inversion.
_ELECTRON_CLASSICAL_RADIUS_M = 2.8179403262e-15


def _positive_finite_scalar(value, name: str, *, allow_zero: bool = False) -> float:
    """Validate a finite real scalar, rejecting booleans and numeric strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def thomson_differential_cross_section_m2_sr(scattering_angle_rad: float, *, incident_polarization_vector=None, incident_direction_vector=None, scattered_direction_vector=None) -> float:
    """Thomson differential cross-section in m^2 sr^-1.

    Without vectors, use the unpolarized result. With both vectors, calculate
    the polarization-dependent result summed over unanalysed outgoing states.
    """
    theta = _positive_finite_scalar(scattering_angle_rad, "scattering_angle_rad", allow_zero=True)
    if theta > np.pi:
        raise ValueError("scattering_angle_rad must lie between 0 and pi")
    if incident_polarization_vector is None and incident_direction_vector is None and scattered_direction_vector is None:
        return float(0.5 * _ELECTRON_CLASSICAL_RADIUS_M**2 * (1.0 + np.cos(theta)**2))
    if any(v is None for v in (incident_polarization_vector, incident_direction_vector, scattered_direction_vector)):
        raise ValueError("incident polarization, incident direction and scattered direction vectors must be supplied together")
    eps = np.asarray(incident_polarization_vector, dtype=float)
    incident = np.asarray(incident_direction_vector, dtype=float)
    direction = np.asarray(scattered_direction_vector, dtype=float)
    if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in (eps,incident,direction)):
        raise ValueError("polarization and beam-direction inputs must be finite 3-vectors")
    eps_norm = float(np.linalg.norm(eps)); incident_norm=float(np.linalg.norm(incident)); direction_norm = float(np.linalg.norm(direction))
    if eps_norm <= 0 or incident_norm <= 0 or direction_norm <= 0:
        raise ValueError("polarization and beam-direction vectors must be nonzero")
    eps = eps/eps_norm; incident=incident/incident_norm; direction = direction/direction_norm
    if abs(float(np.dot(eps,incident))) > 1e-6:
        raise ValueError("incident polarization must be perpendicular to the incident beam direction")
    geometric_angle=float(np.arccos(np.clip(np.dot(incident,direction),-1.0,1.0)))
    if abs(geometric_angle-theta)>1e-5:
        raise ValueError("scattering_angle_rad is inconsistent with the supplied beam-direction vectors")
    factor = max(0.0, 1.0-float(np.dot(direction, eps))**2)
    return float(_ELECTRON_CLASSICAL_RADIUS_M**2*factor)


def density_from_thomson_scattered_power(
    scattered_power_W,
    incident_power_W,
    interaction_length_m,
    scattering_angle_rad,
    collection_solid_angle_sr,
    *,
    optical_efficiency=1.0,
    scattered_power_uncertainty_W=None,
    incident_power_uncertainty_W=None,
    interaction_length_uncertainty_m=None,
    scattering_angle_uncertainty_rad=None,
    collection_solid_angle_uncertainty_sr=None,
    optical_efficiency_uncertainty=None,
    incident_polarization_vector=None,
    incident_direction_vector=None,
    scattered_direction_vector=None,
    polarization_factor_uncertainty=None,
    relative_input_covariance=None,
):
    """Infer density from calibrated collected Thomson power.

    P_s = n_e P_i L (d sigma/d Omega) DeltaOmega eta. Optional independent
    1-sigma uncertainties are propagated with first-order derivatives.
    """
    ps = _positive_finite_scalar(scattered_power_W, "scattered_power_W", allow_zero=True)
    pi = _positive_finite_scalar(incident_power_W, "incident_power_W")
    length = _positive_finite_scalar(interaction_length_m, "interaction_length_m")
    theta = _positive_finite_scalar(scattering_angle_rad, "scattering_angle_rad", allow_zero=True)
    if theta > np.pi:
        raise ValueError("scattering_angle_rad must lie between 0 and pi")
    omega = _positive_finite_scalar(collection_solid_angle_sr, "collection_solid_angle_sr")
    eta = _positive_finite_scalar(optical_efficiency, "optical_efficiency")
    if eta > 1.0:
        raise ValueError("optical_efficiency must not exceed 1")
    dsigma = thomson_differential_cross_section_m2_sr(
        theta, incident_polarization_vector=incident_polarization_vector,
        incident_direction_vector=incident_direction_vector,
        scattered_direction_vector=scattered_direction_vector,
    )
    if dsigma <= 0:
        raise ValueError("The supplied polarization/collection geometry has zero Thomson differential cross-section")
    density = ps / (pi * length * dsigma * omega * eta)
    model_power = density * pi * length * dsigma * omega * eta

    rel_variance = 0.0
    uncertainty_supplied = False
    if relative_input_covariance is not None:
        covariance_raw = np.asarray(relative_input_covariance)
        if covariance_raw.dtype.kind not in "iuf":
            raise ValueError("relative_input_covariance must contain real numeric values, not booleans or strings")
        covariance = covariance_raw.astype(float, copy=False)
        if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
            raise ValueError("relative_input_covariance must be a finite 6x6 matrix")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0.0):
            raise ValueError("relative_input_covariance must be symmetric")
        if not _covariance_is_positive_semidefinite(covariance):
            raise ValueError("relative_input_covariance must be positive semidefinite")
        if any(v is not None for v in (scattered_power_uncertainty_W, incident_power_uncertainty_W,
                interaction_length_uncertainty_m, collection_solid_angle_uncertainty_sr,
                optical_efficiency_uncertainty, polarization_factor_uncertainty)):
            raise ValueError("Use relative_input_covariance instead of separate scalar uncertainties; do not double-count uncertainty inputs")
        gradient = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        rel_variance = float(gradient @ covariance @ gradient)
        if rel_variance < -1e-12:
            raise ValueError("relative_input_covariance produced a negative propagated variance")
        rel_variance = max(0.0, rel_variance)
        uncertainty_supplied = True
    inputs = (
        ("scattered_power_uncertainty_W", scattered_power_uncertainty_W, ps),
        ("incident_power_uncertainty_W", incident_power_uncertainty_W, pi),
        ("interaction_length_uncertainty_m", interaction_length_uncertainty_m, length),
        ("collection_solid_angle_uncertainty_sr", collection_solid_angle_uncertainty_sr, omega),
        ("optical_efficiency_uncertainty", optical_efficiency_uncertainty, eta),
    )
    for name, uncertainty, nominal in inputs:
        if uncertainty is None:
            continue
        sigma = _positive_finite_scalar(uncertainty, name, allow_zero=True)
        uncertainty_supplied = True
        if nominal == 0.0:
            if sigma > 0.0:
                raise ValueError(f"{name} cannot be propagated relative to a zero nominal value")
        else:
            rel_variance += (sigma / nominal) ** 2
    if polarization_factor_uncertainty is not None:
        sigma_pol = _positive_finite_scalar(polarization_factor_uncertainty, "polarization_factor_uncertainty", allow_zero=True)
        if incident_polarization_vector is None or dsigma <= 0:
            raise ValueError("polarization-factor uncertainty requires a nonzero polarization-dependent cross-section")
        polarization_factor = dsigma/(_ELECTRON_CLASSICAL_RADIUS_M**2)
        if polarization_factor <= 0:
            raise ValueError("polarization factor must be positive to propagate its uncertainty")
        uncertainty_supplied = True
        rel_variance += (sigma_pol/polarization_factor)**2
    if scattering_angle_uncertainty_rad is not None:
        sigma_theta = _positive_finite_scalar(
            scattering_angle_uncertainty_rad, "scattering_angle_uncertainty_rad", allow_zero=True
        )
        uncertainty_supplied = True
        if incident_polarization_vector is not None and sigma_theta > 0:
            raise ValueError("For polarized geometry, propagate angle uncertainty through the supplied direction vectors or provide polarization_factor_uncertainty; the scalar unpolarized angle derivative is not applicable.")
        if incident_polarization_vector is None:
            relative_sensitivity = abs(
                2.0 * np.sin(theta) * np.cos(theta) / (1.0 + np.cos(theta) ** 2)
            )
            rel_variance += (relative_sensitivity * sigma_theta) ** 2

    return {
        "electron_density_m3": float(density),
        "electron_density_uncertainty_m3": float(density * np.sqrt(rel_variance)) if uncertainty_supplied else None,
        "model_scattered_power_W": float(model_power),
        "differential_cross_section_m2_sr": float(dsigma),
        "assumptions": [
            "Absolutely calibrated scattered and incident laser power.",
            "Optically thin, single-scattering Thomson relation; polarization dependence is included when incident polarization and scattered direction vectors are supplied, otherwise the unpolarized cross-section is used.",
            "The returned density is the interaction-length-averaged effective density; it equals a local density only when density is uniform along the scattering path.",
            "Collection solid angle and optical efficiency describe the actual collection path.",
        ],
        "density_interpretation": "interaction_length_averaged_effective_density",
        "uncertainty_model": ("First-order propagation with a supplied 6x6 relative covariance matrix for scattered power, incident power, interaction length, collection solid angle, optical efficiency and polarization/differential-cross-section factor." if relative_input_covariance is not None else "Optional first-order independent-input propagation; unprovided covariance and model discrepancy are not included."),
    }
