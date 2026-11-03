"""Physical microwave background emission for CTS receiver modelling.

Thermal free-free emission is calculated from the standard non-relativistic
Maxwellian emissivity with an explicit Gaunt factor. Thermal ECE is handled
through mode-resolved absorption supplied by a magnetized kinetic/radiative
transfer calculation and Kirchhoff's law. The latter coefficient is not
fabricated from a scalar magnetic-field magnitude.
"""
from __future__ import annotations

import numpy as np

H = 6.62607015e-34
K_B = 1.380649e-23
C = 299792458.0
E_CHARGE = 1.602176634e-19


def _positive_array(value, name):
    a = np.asarray(value, dtype=float)
    if a.size == 0 or np.any(~np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f"{name} must contain positive finite values")
    return a


def planck_spectral_radiance(frequency_Hz, temperature_K):
    """Planck spectral radiance B_nu in W m^-2 sr^-1 Hz^-1."""
    nu = _positive_array(frequency_Hz, "frequency_Hz")
    temp = _positive_array(temperature_K, "temperature_K")
    x = H * nu / (K_B * temp)
    return (2.0 * H * nu**3 / C**2) / np.expm1(np.minimum(x, 700.0))


def thermal_bremsstrahlung_emissivity_si(
    frequency_Hz, electron_density_m3, ion_densities_m3,
    ion_charge_states, electron_temperature_eV, *, gaunt_factor=1.0,
):
    """Unpolarized Maxwellian free-free emissivity j_nu in W m^-3 sr^-1 Hz^-1.

    Uses j_nu = 5.44e-52 n_e sum_i(Z_i^2 n_i) T_K^-1/2
    exp(-h nu/kT) g_ff, with densities in m^-3 and T in kelvin.
    Relativistic corrections and magnetized-mode anisotropy are not included.
    """
    nu = _positive_array(frequency_Hz, "frequency_Hz")
    ne = float(electron_density_m3)
    te = float(electron_temperature_eV)
    ni = np.asarray(ion_densities_m3, dtype=float)
    z = np.asarray(ion_charge_states, dtype=float)
    g = np.asarray(gaunt_factor, dtype=float)
    if not np.isfinite(ne) or ne <= 0 or not np.isfinite(te) or te <= 0:
        raise ValueError("electron_density_m3 and electron_temperature_eV must be positive finite scalars")
    if ni.ndim != 1 or z.shape != ni.shape or ni.size == 0 or np.any(~np.isfinite(ni)) or np.any(ni < 0) or np.any(~np.isfinite(z)) or np.any(z <= 0):
        raise ValueError("ion densities and charge states must be matching finite vectors with n_i>=0 and Z_i>0")
    if np.any(~np.isfinite(g)) or np.any(g <= 0):
        raise ValueError("gaunt_factor must be positive and finite")
    try:
        g = np.broadcast_to(g, nu.shape)
    except ValueError as exc:
        raise ValueError(
            "gaunt_factor must be scalar or broadcastable to frequency_Hz"
        ) from exc
    temp_K = te * E_CHARGE / K_B
    charge_weighted_ions = float(np.sum(z**2 * ni))
    x = H * nu / (K_B * temp_K)
    return 5.44e-52 * ne * charge_weighted_ions * temp_K**-0.5 * np.exp(-np.minimum(x, 700.0)) * g


def integrate_thermal_microwave_background(
    frequency_Hz, path_length_m, electron_temperature_eV, *,
    electron_density_m3, ion_densities_m3, ion_charge_states,
    ece_absorption_coefficient_m_inverse=None,
    gaunt_factor=1.0, incident_radiance_W_m2_sr_Hz=0.0,
):
    """Integrate thermal free-free + mode-resolved ECE radiative transfer.

    Arrays are sampled along the line of sight. ECE absorption is supplied per
    receiver mode by an external magnetized dielectric/radiative-transfer
    solver. In LTE, j_ECE = alpha_ECE B_nu(T_e). Free-free absorption follows
    Kirchhoff's law using its computed emissivity. The exact constant-slab
    transfer update is applied segment by segment.
    """
    nu = _positive_array(frequency_Hz, "frequency_Hz")
    length = np.asarray(path_length_m, dtype=float)
    te = np.asarray(electron_temperature_eV, dtype=float)
    ne = np.asarray(electron_density_m3, dtype=float)
    ni = np.asarray(ion_densities_m3, dtype=float)
    z = np.asarray(ion_charge_states, dtype=float)
    if length.ndim != 1 or te.shape != length.shape or ne.shape != length.shape:
        raise ValueError("path_length_m, electron_temperature_eV and electron_density_m3 must be matching 1-D path arrays")
    if ni.ndim != 2 or ni.shape[0] != length.size or z.shape != (ni.shape[1],):
        raise ValueError("ion_densities_m3 must be (n_path,n_species) and ion_charge_states one value per species")
    if np.any(~np.isfinite(length)) or np.any(length <= 0) or np.any(~np.isfinite(te)) or np.any(te <= 0) or np.any(~np.isfinite(ne)) or np.any(ne <= 0):
        raise ValueError("path lengths, temperatures and electron densities must be positive finite")
    gaunt = np.asarray(gaunt_factor, dtype=float)
    if np.any(~np.isfinite(gaunt)) or np.any(gaunt <= 0.0):
        raise ValueError("gaunt_factor must contain positive finite values")
    if gaunt.ndim == 0:
        gaunt_path = np.full((length.size, nu.size), float(gaunt))
        gaunt_status = "scalar"
    elif gaunt.shape == (nu.size,):
        gaunt_path = np.broadcast_to(gaunt, (length.size, nu.size))
        gaunt_status = "frequency_resolved"
    elif gaunt.shape == (length.size,):
        gaunt_path = np.broadcast_to(gaunt[:, None], (length.size, nu.size))
        gaunt_status = "path_resolved"
    elif gaunt.shape == (length.size, nu.size):
        gaunt_path = gaunt
        gaunt_status = "path_and_frequency_resolved"
    else:
        raise ValueError(
            "gaunt_factor must be scalar, (n_frequency,), (n_path,), "
            "or (n_path,n_frequency)"
        )
    if ece_absorption_coefficient_m_inverse is None:
        ece_alpha = np.zeros((length.size, nu.size), dtype=float)
        ece_status = "ECE omitted; no mode-resolved absorption supplied"
    else:
        ece_alpha = np.asarray(ece_absorption_coefficient_m_inverse, dtype=float)
        if ece_alpha.shape != (length.size, nu.size) or np.any(~np.isfinite(ece_alpha)) or np.any(ece_alpha < 0):
            raise ValueError("ece_absorption_coefficient_m_inverse must be non-negative with shape (n_path,n_frequency)")
        ece_status = "mode-resolved ECE absorption supplied; Kirchhoff LTE source function used"
    incident = np.asarray(incident_radiance_W_m2_sr_Hz, dtype=float)
    if incident.ndim == 0:
        incident = np.full(nu.size, float(incident))
    if incident.shape != nu.shape or np.any(~np.isfinite(incident)) or np.any(incident < 0):
        raise ValueError("incident_radiance_W_m2_sr_Hz must be non-negative scalar or frequency vector")
    intensity = incident.copy()
    alpha_ff_total = np.zeros(nu.size)
    tau_ece_total = np.zeros(nu.size)
    for index in range(length.size):
        j_ff = thermal_bremsstrahlung_emissivity_si(
            nu, ne[index], ni[index], z, te[index],
            gaunt_factor=gaunt_path[index],
        )
        b_nu = planck_spectral_radiance(nu, te[index] * E_CHARGE / K_B)
        alpha_ff = j_ff / np.maximum(b_nu, np.finfo(float).tiny)
        alpha_total = alpha_ff + ece_alpha[index]
        optical_depth = alpha_total * length[index]
        transmission = np.exp(-np.minimum(optical_depth, 700.0))
        source = b_nu  # LTE free-free and thermal ECE have the same Planck source function.
        intensity = intensity * transmission + source * (1.0 - transmission)
        alpha_ff_total += alpha_ff * length[index]
        tau_ece_total += ece_alpha[index] * length[index]
    return {
        "spectral_radiance_W_m2_sr_Hz": intensity,
        "thermal_bremsstrahlung_optical_depth": alpha_ff_total,
        "ece_optical_depth": tau_ece_total,
        "total_optical_depth": alpha_ff_total + tau_ece_total,
        "ece_model_status": ece_status,
        "gaunt_factor_resolution": gaunt_status,
        "free_free_model": "nonrelativistic_Maxwellian_thermal_bremsstrahlung_with_supplied_Gaunt_factor",
        "radiative_transfer": "slab-by-slab exact formal solution with LTE Kirchhoff source function",
        "receiver_plane_conversion": "multiply radiance by measured receiver etendue and end-to-end efficiency; do not count the same throughput twice",
        "limitations": [
            "Free-free emissivity is non-relativistic and unmagnetized; use a relativistic/magnetized emissivity kernel outside that regime.",
            "The supplied Gaunt factor may be scalar, frequency-resolved, path-resolved, or path-and-frequency-resolved; it is not inferred from incomplete plasma inputs.",
            "ECE absorption must be supplied by a mode-resolved magnetized plasma solver; no scalar approximation is silently substituted.",
            "The result is spectral radiance, not detector power; receiver etendue and efficiency are external measured inputs.",
        ],
    }



def radiance_to_receiver_spectral_power(
    spectral_radiance_W_m2_sr_Hz, *, receiver_etendue_m2_sr,
    end_to_end_efficiency,
):
    """Convert spectral radiance to receiver-plane spectral power in W Hz^-1.

    receiver_etendue_m2_sr is A*Omega at the same plane and geometry used by
    the radiative-transfer calculation. Efficiency is applied once here; do
    not apply it again to the resulting receiver-plane spectrum.
    """
    radiance = np.asarray(spectral_radiance_W_m2_sr_Hz, dtype=float)
    etendue = float(receiver_etendue_m2_sr)
    efficiency = float(end_to_end_efficiency)
    if radiance.size == 0 or np.any(~np.isfinite(radiance)) or np.any(radiance < 0):
        raise ValueError("spectral radiance must be finite and non-negative")
    if not np.isfinite(etendue) or etendue <= 0:
        raise ValueError("receiver_etendue_m2_sr must be positive and finite")
    if not np.isfinite(efficiency) or not 0 < efficiency <= 1:
        raise ValueError("end_to_end_efficiency must lie in (0,1]")
    return radiance * etendue * efficiency
