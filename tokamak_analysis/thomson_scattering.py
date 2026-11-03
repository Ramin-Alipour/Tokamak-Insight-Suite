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
    # Covariance matrices must be symmetric. eigvalsh assumes Hermitian input
    # and can otherwise accept an asymmetric matrix by reading one triangle.
    matrix_scale = float(np.max(np.abs(matrix)))
    if matrix_scale > 0.0 and float(np.max(np.abs(matrix - matrix.T))) > float(rtol) * matrix_scale:
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
    theta = _positive_finite_scalar(scattering_angle_rad, "scattering_angle_rad")
    if theta >= np.pi:
        raise ValueError("scattering_angle_rad must lie strictly between 0 and pi")
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
    optical_efficiency=None,
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
    effective_differential_cross_section_m2_sr=None,
    effective_differential_cross_section_uncertainty_m2_sr=None,
    power_cross_section_relative_covariance=None,
):
    """Infer path-averaged density from absolutely calibrated collected power.

    scattered_power_W is the wavelength-integrated Thomson power at the same
    calibrated measurement plane to which optical_efficiency refers.
    incident_power_W is the laser power in the plasma interaction volume;
    interaction_length_m is the effective illuminated plasma path length;
    collection_solid_angle_sr is the solid angle subtended by the collection
    optics at the scattering volume; and optical_efficiency is the end-to-end
    fraction of power scattered into that solid angle that reaches the
    calibrated measurement plane. Do not include the same throughput loss
    both in scattered_power_W and again in optical_efficiency.

    P_s = n_e P_i L (d sigma_eff/d Omega) DeltaOmega eta. The result is the
    interaction-length-averaged effective density, not a local density unless
    the density and collection kernel are uniform along the sightline. The
    cross-section is either the stationary-electron polarized Thomson value
    or a model-supplied velocity-averaged relativistic value.
    """
    ps = _positive_finite_scalar(scattered_power_W, "scattered_power_W", allow_zero=True)
    pi = _positive_finite_scalar(incident_power_W, "incident_power_W")
    length = _positive_finite_scalar(interaction_length_m, "interaction_length_m")
    theta = _positive_finite_scalar(scattering_angle_rad, "scattering_angle_rad")
    if theta >= np.pi:
        raise ValueError("scattering_angle_rad must lie strictly between 0 and pi")
    omega = _positive_finite_scalar(collection_solid_angle_sr, "collection_solid_angle_sr")
    if optical_efficiency is None:
        raise ValueError(
            "optical_efficiency must be supplied explicitly; perfect collection efficiency is not assumed"
        )
    eta = _positive_finite_scalar(optical_efficiency, "optical_efficiency")
    if eta > 1.0:
        raise ValueError("optical_efficiency must not exceed 1")
    if effective_differential_cross_section_m2_sr is None:
        if power_cross_section_relative_covariance is not None:
            raise ValueError("power_cross_section_relative_covariance requires an effective differential cross-section")
        if effective_differential_cross_section_uncertainty_m2_sr is not None:
            raise ValueError(
                "effective_differential_cross_section_uncertainty_m2_sr requires an effective cross-section value"
            )
        dsigma = thomson_differential_cross_section_m2_sr(
            theta, incident_polarization_vector=incident_polarization_vector,
            incident_direction_vector=incident_direction_vector,
            scattered_direction_vector=scattered_direction_vector,
        )
    else:
        dsigma = _positive_finite_scalar(
            effective_differential_cross_section_m2_sr,
            "effective_differential_cross_section_m2_sr",
        )
        if scattering_angle_uncertainty_rad is not None or polarization_factor_uncertainty is not None:
            raise ValueError(
                "For a supplied effective relativistic cross-section, propagate angle/polarization "
                "uncertainty through effective_differential_cross_section_uncertainty_m2_sr instead."
            )
    if dsigma <= 0:
        raise ValueError("The supplied polarization/collection geometry has zero Thomson differential cross-section")
    density = ps / (pi * length * dsigma * omega * eta)
    model_power = density * pi * length * dsigma * omega * eta

    rel_variance = 0.0
    uncertainty_supplied = False
    scattering_angle_relative_sensitivity = None
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
        if any(v is not None for v in (incident_power_uncertainty_W,
                interaction_length_uncertainty_m, collection_solid_angle_uncertainty_sr,
                optical_efficiency_uncertainty, polarization_factor_uncertainty)):
            raise ValueError("Use relative_input_covariance instead of separate scalar optical-factor uncertainties; do not double-count uncertainty inputs")
        gradient = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        covariance = covariance.copy()
        cross_covariance = (
            None if power_cross_section_relative_covariance is None
            else float(power_cross_section_relative_covariance)
        )
        if cross_covariance is not None and not np.isfinite(cross_covariance):
            raise ValueError("power_cross_section_relative_covariance must be finite")
        if cross_covariance is not None and abs(cross_covariance) > 0.0 and ps <= 0.0:
            raise ValueError("power_cross_section_relative_covariance requires positive fitted scattered power")
        if cross_covariance is not None and abs(cross_covariance) > 0.0 and scattered_power_uncertainty_W is not None:
            if not np.allclose(covariance[0, :], 0.0, rtol=0.0, atol=1e-14) or not np.allclose(
                covariance[:, 0], 0.0, rtol=0.0, atol=1e-14
            ):
                raise ValueError(
                    "A separate scattered_power_uncertainty_W can be combined with fitted "
                    "power/cross-section covariance only when the supplied covariance has a zero power row and column."
                )
            sigma_power = _positive_finite_scalar(
                scattered_power_uncertainty_W, "scattered_power_uncertainty_W", allow_zero=True
            )
            if ps <= 0.0 and sigma_power > 0.0:
                raise ValueError("relative scattered-power covariance requires positive scattered_power_W")
            if ps > 0.0:
                covariance[0, 0] += (sigma_power / ps) ** 2
            scattered_power_uncertainty_W = None
        if cross_covariance is not None:
            covariance[0, 5] += cross_covariance
            covariance[5, 0] += cross_covariance
        if effective_differential_cross_section_uncertainty_m2_sr is not None:
            sigma_cross_section = _positive_finite_scalar(
                effective_differential_cross_section_uncertainty_m2_sr,
                "effective_differential_cross_section_uncertainty_m2_sr",
                allow_zero=True,
            )
            covariance[5, 5] += (sigma_cross_section / dsigma) ** 2
            uncertainty_supplied = True
        if cross_covariance is not None and abs(cross_covariance) > 0.0 and covariance[0, 0] <= 0.0:
            raise ValueError(
                "power/cross-section covariance requires a nonzero power variance in the supplied or fitted density covariance."
            )
        if not _covariance_is_positive_semidefinite(covariance):
            raise ValueError("combined density covariance is not positive semidefinite after adding fitted power/cross-section covariance")
        rel_variance = float(gradient @ covariance @ gradient)
        if scattered_power_uncertainty_W is not None:
            # A zero first row/column declares that measured-power uncertainty
            # is supplied separately (e.g. by joint spectral GLS or response MC).
            if not np.allclose(covariance[0, :], 0.0, rtol=0.0, atol=1e-14) or not np.allclose(
                covariance[:, 0], 0.0, rtol=0.0, atol=1e-14
            ):
                raise ValueError(
                    "scattered_power_uncertainty_W may accompany relative_input_covariance "
                    "only when its first row and column are zero; otherwise power uncertainty "
                    "would be double-counted."
                )
            sigma_power = _positive_finite_scalar(
                scattered_power_uncertainty_W, "scattered_power_uncertainty_W", allow_zero=True
            )
            if ps > 0.0:
                rel_variance += (sigma_power / ps) ** 2
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
    if power_cross_section_relative_covariance is not None and relative_input_covariance is None:
        covariance_term = float(power_cross_section_relative_covariance)
        if not np.isfinite(covariance_term):
            raise ValueError("power_cross_section_relative_covariance must be finite")
        rel_variance -= 2.0 * covariance_term
        uncertainty_supplied = True
    if effective_differential_cross_section_uncertainty_m2_sr is not None and relative_input_covariance is None:
        sigma_cross_section = _positive_finite_scalar(
            effective_differential_cross_section_uncertainty_m2_sr,
            "effective_differential_cross_section_uncertainty_m2_sr",
            allow_zero=True,
        )
        uncertainty_supplied = True
        rel_variance += (sigma_cross_section / dsigma) ** 2
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
        if incident_polarization_vector is None:
            scattering_angle_relative_sensitivity = (
                2.0 * np.sin(theta) * np.cos(theta) / (1.0 + np.cos(theta) ** 2)
            )
        else:
            # A scalar angle uncertainty at fixed incident beam and polarization
            # corresponds to rotating the scattered direction in its nominal
            # scattering plane. Differentiate the polarized Thomson factor
            # F = 1 - (k_s . epsilon_i)^2 along that angular coordinate.
            eps = np.asarray(incident_polarization_vector, dtype=float)
            incident = np.asarray(incident_direction_vector, dtype=float)
            scattered = np.asarray(scattered_direction_vector, dtype=float)
            eps = eps / np.linalg.norm(eps)
            incident = incident / np.linalg.norm(incident)
            scattered = scattered / np.linalg.norm(scattered)
            cosine = float(np.clip(np.dot(incident, scattered), -1.0, 1.0))
            sine = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
            factor = dsigma / (_ELECTRON_CLASSICAL_RADIUS_M ** 2)
            if sine <= 1e-12:
                scattering_angle_relative_sensitivity = 0.0
            else:
                tangent = (scattered - cosine * incident) / sine
                direction_derivative = -sine * incident + cosine * tangent
                polarization_projection = float(np.dot(scattered, eps))
                projection_derivative = float(np.dot(direction_derivative, eps))
                factor_derivative = -2.0 * polarization_projection * projection_derivative
                scattering_angle_relative_sensitivity = -factor_derivative / factor
        rel_variance += (scattering_angle_relative_sensitivity * sigma_theta) ** 2

    if rel_variance < -1e-12:
        raise ValueError("density uncertainty propagation produced a negative relative variance")
    rel_variance = max(0.0, rel_variance)

    return {
        "electron_density_m3": float(density),
        "electron_density_uncertainty_m3": float(density * np.sqrt(rel_variance)) if uncertainty_supplied else None,
        "model_scattered_power_W": float(model_power),
        "differential_cross_section_m2_sr": float(dsigma),
        "scattering_angle_relative_sensitivity": (
            None if scattering_angle_relative_sensitivity is None
            else float(scattering_angle_relative_sensitivity)
        ),
        "assumptions": [
            "Absolutely calibrated scattered and incident laser power.",
            "Optically thin, single-scattering Thomson relation; polarization dependence is included through supplied incident-polarization geometry or a model-supplied effective cross-section. The unpolarized stationary-electron cross-section is used only when neither is supplied.",
            "The returned density is the interaction-length-averaged effective density; it equals a local density only when density is uniform along the scattering path.",
            "A model-supplied effective differential cross-section may be used when relativistic electron kinematics and polarization are integrated over the electron distribution.",
            "Collection solid angle and optical efficiency describe the actual collection path.",
        ],
        "density_interpretation": "interaction_length_averaged_effective_density",
        "uncertainty_model": ("First-order propagation with a supplied 6x6 relative covariance matrix for scattered power, incident power, interaction length, collection solid angle, optical efficiency and differential-cross-section factor; independent effective-cross-section model uncertainty and fitted power/cross-section covariance are added internally." if relative_input_covariance is not None else "Optional first-order independent-input propagation, including effective differential-cross-section model uncertainty and fitted power/cross-section covariance when supplied; unprovided covariance and model discrepancy are not included."),
    }


def infer_thomson_path_averaged_density(
    scattered_power_W,
    path_coordinate_m,
    incident_power_profile_W,
    differential_cross_section_profile_m2_sr,
    collection_solid_angle_profile_sr,
    optical_efficiency_profile,
    *,
    scattered_power_uncertainty_W=None,
):
    """Infer the measurement-kernel-weighted electron density along a sightline.

    This is the path-integrated single-scattering relation
        P_s = integral[n_e(s) P_i(s) (d sigma/d Omega)(s)
                       DeltaOmega(s) eta(s) ds].
    A single measured P_s identifies only the weighted path average, not an
    arbitrary spatial density profile. The supplied path-dependent factors
    are fixed, already-characterized inputs; this routine does not calibrate
    an instrument or infer missing geometry/optical factors.

    All profile arrays are sampled at the same strictly increasing path
    coordinate. Cross-section units are m^2 sr^-1, solid angle is sr,
    path is m, and incident/scattered powers are W.
    """
    ps = _positive_finite_scalar(scattered_power_W, "scattered_power_W", allow_zero=True)
    s = np.asarray(path_coordinate_m, dtype=float)
    if s.ndim != 1 or s.size < 2 or not np.all(np.isfinite(s)) or np.any(np.diff(s) <= 0):
        raise ValueError("path_coordinate_m must be a finite strictly increasing vector with at least two samples")
    profiles = {}
    for name, value in (
        ("incident_power_profile_W", incident_power_profile_W),
        ("differential_cross_section_profile_m2_sr", differential_cross_section_profile_m2_sr),
        ("collection_solid_angle_profile_sr", collection_solid_angle_profile_sr),
        ("optical_efficiency_profile", optical_efficiency_profile),
    ):
        arr = np.asarray(value, dtype=float)
        if arr.shape != s.shape or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite and match path_coordinate_m")
        profiles[name] = arr
    pi = profiles["incident_power_profile_W"]
    dsigma = profiles["differential_cross_section_profile_m2_sr"]
    omega = profiles["collection_solid_angle_profile_sr"]
    eta = profiles["optical_efficiency_profile"]
    if np.any(pi <= 0) or np.any(dsigma <= 0) or np.any(omega <= 0) or np.any(eta <= 0) or np.any(eta > 1):
        raise ValueError("incident power, cross-section, solid angle and efficiency must be positive; efficiency must not exceed one")
    kernel = pi * dsigma * omega * eta
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0
        integrate = np.trapz
    denominator = float(integrate(kernel, s))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("path-integrated optical sensitivity must be positive and finite")
    density = ps / denominator
    sigma_ps = None
    if scattered_power_uncertainty_W is not None:
        sigma_ps = _positive_finite_scalar(
            scattered_power_uncertainty_W, "scattered_power_uncertainty_W", allow_zero=True
        )
    return {
        "electron_density_path_averaged_m3": float(density),
        "electron_density_path_averaged_uncertainty_m3": (
            None if sigma_ps is None else float(sigma_ps / denominator)
        ),
        "path_integrated_sensitivity_W_m3": denominator,
        "path_coordinate_m": s.copy(),
        "normalized_path_weight": kernel / denominator,
        "path_weighting_model": "incident_power_times_differential_cross_section_times_collection_solid_angle_times_optical_efficiency",
        "density_interpretation": "measurement_kernel_weighted_path_average",
        "identifiability": {
            "status": "weighted_path_average_identifiable",
            "not_identifiable": ["arbitrary pointwise electron_density_profile_m3"],
            "reason": "One integrated power measurement constrains one weighted path integral; it cannot uniquely recover an unconstrained spatial density function.",
            "profile_reconstruction_requires": [
                "multiple independent spatially resolved scattering measurements or an explicitly parameterized density profile",
                "the corresponding spatial response/kernel and any cross-channel covariance",
            ],
        },
        "assumptions": [
            "Input scattered power and all path-dependent optical/geometry profiles are measured or independently characterized and already calibrated.",
            "Optically thin single scattering; no multiple-scattering contribution.",
            "The supplied path-dependent factors are fixed and are not calibrated or corrected by this function.",
        ],
        "uncertainty_model": "First-order propagation of supplied scattered-power uncertainty only; uncertainty/covariance of path-dependent factors is not included unless separately propagated by the caller.",
    }
