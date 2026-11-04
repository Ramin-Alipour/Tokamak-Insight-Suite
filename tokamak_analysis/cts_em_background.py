"""Physical microwave background emission for CTS receiver modelling.

Thermal free-free emission is calculated from the standard non-relativistic
Maxwellian emissivity with an explicit Gaunt factor. Thermal ECE requires
mode-resolved absorption plus either an independently supplied emissivity or
an explicit LTE selection for the Planck source function. Absorption alone does
not determine a non-LTE source function. No ECE coefficient is fabricated
from a scalar magnetic-field magnitude.
"""
from __future__ import annotations

import numpy as np

H = 6.62607015e-34
K_B = 1.380649e-23
C = 299792458.0
E_CHARGE = 1.602176634e-19

# Rybicki-Lightman 6.8e-38 cgs emissivity converted for SI densities/output:
# 1e-7 J/erg * 1e6 cm^3/m^3 * (1e-6)^2 for the two densities.
_FREE_FREE_EMISSIVITY_COEFFICIENT_SI = 6.8e-38 * 1.0e-13 / (4.0 * np.pi)


def _real_array(value, name):
    """Convert real numeric inputs without silently accepting booleans or strings."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values (booleans and strings are invalid)")
    result = raw.astype(float, copy=False)
    if result.size == 0 or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain finite, non-empty real values")
    return result


def _real_scalar(value, name, *, positive=False, nonnegative=False):
    """Validate a genuine real scalar before any float coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real numeric scalar; booleans and strings are invalid")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_array(value, name):
    a = _real_array(value, name)
    if np.any(a <= 0):
        raise ValueError(f"{name} must contain positive finite values")
    return a


def planck_spectral_radiance(frequency_Hz, temperature_K):
    """Planck spectral radiance B_nu in W m^-2 sr^-1 Hz^-1."""
    nu = _positive_array(frequency_Hz, "frequency_Hz")
    temp = _positive_array(temperature_K, "temperature_K")
    x = H * nu / (K_B * temp)
    # exp(-x)/[1-exp(-x)] is stable in both the Rayleigh-Jeans
    # limit (use expm1) and the Wien tail (natural underflow to zero).
    occupation = np.exp(-x) / (-np.expm1(-x))
    return (2.0 * H * nu**3 / C**2) * occupation


def plasma_charge_state_moments(electron_density_m3, ion_densities_m3, ion_charge_states):
    """Return charge balance and effective ionic charge without forcing neutrality.

    n_i are ion number densities and Z_i are positive integer charge states:
      n_e,quasi = sum_i Z_i n_i,
      Z_eff = sum_i Z_i^2 n_i / n_e,
      rho_q = e (sum_i Z_i n_i - n_e).
    A non-neutral state is reported, not silently repaired. Z_eff is undefined
    when n_e=0 and is returned as None in that case.
    """
    ne = _real_scalar(electron_density_m3, "electron_density_m3", nonnegative=True)
    ni = _real_array(ion_densities_m3, "ion_densities_m3")
    z = _real_array(ion_charge_states, "ion_charge_states")
    if ni.ndim != 1 or ni.size == 0 or z.shape != ni.shape:
        raise ValueError("ion_densities_m3 and ion_charge_states must be matching non-empty vectors")
    if np.any(ni < 0.0) or np.any(z <= 0.0) or np.any(z != np.floor(z)):
        raise ValueError("ion densities must be non-negative and charge states positive integers")
    ion_charge_density = float(np.sum(z * ni))
    ion_charge_squared_density = float(np.sum(z**2 * ni))
    scale = max(ne, ion_charge_density, np.finfo(float).tiny)
    return {
        "electron_density_m3": ne,
        "ion_charge_density_equivalent_m3": ion_charge_density,
        "ion_charge_squared_density_m3": ion_charge_squared_density,
        "effective_charge_Zeff": (
            float(ion_charge_squared_density / ne) if ne > 0.0 else None
        ),
        "net_charge_density_C_m3": float(E_CHARGE * (ion_charge_density - ne)),
        "quasineutrality_relative_residual": float(
            (ion_charge_density - ne) / scale
        ),
        "quasineutrality_assumed_or_enforced": False,
    }


def thermal_bremsstrahlung_emissivity_si(
    frequency_Hz, electron_density_m3, ion_densities_m3,
    ion_charge_states, electron_temperature_eV, *, gaunt_factor=1.0,
):
    """Unpolarized Maxwellian free-free emissivity j_nu in W m^-3 sr^-1 Hz^-1.

    Uses the angle-resolved (per steradian) SI form
    j_nu = 5.41e-52 n_e sum_i(Z_i^2 n_i) T_K^-1/2
    exp(-h nu/kT) g_ff, with densities in m^-3 and T in kelvin.
    The coefficient is (6.8e-38 * 1e-13)/(4*pi): the angle-integrated cgs
    emissivity is converted to SI with 1e-7 J/erg, 1e6 cm^3/m^3,
    and 1e-12 from the two densities expressed in cm^-3, then divided by 4*pi.
    Relativistic corrections and magnetized-mode anisotropy are not included.
    """
    nu = _positive_array(frequency_Hz, "frequency_Hz")
    ne = _real_scalar(electron_density_m3, "electron_density_m3", nonnegative=True)
    te = _real_scalar(electron_temperature_eV, "electron_temperature_eV", positive=True)
    ni = _real_array(ion_densities_m3, "ion_densities_m3")
    z = _real_array(ion_charge_states, "ion_charge_states")
    g = _real_array(gaunt_factor, "gaunt_factor")
    if not np.isfinite(ne) or ne < 0 or not np.isfinite(te) or te <= 0:
        raise ValueError("electron_density_m3 must be finite and non-negative; electron_temperature_eV must be positive and finite")
    if ni.ndim != 1 or z.shape != ni.shape or ni.size == 0 or np.any(ni < 0) or np.any(z <= 0):
        raise ValueError("ion densities and charge states must be matching finite vectors with n_i>=0 and Z_i>0")
    if np.any(z != np.floor(z)):
        raise ValueError("ion_charge_states must be positive integer ionic charge states")
    if ne == 0.0 and np.any(ni != 0.0):
        raise ValueError("zero electron density requires zero positive-ion densities in this quasineutral plasma model")
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
    coefficient_si = _FREE_FREE_EMISSIVITY_COEFFICIENT_SI
    return coefficient_si * ne * charge_weighted_ions * temp_K**-0.5 * np.exp(-x) * g



def thermal_bremsstrahlung_absorption_si(
    frequency_Hz, electron_density_m3, ion_densities_m3,
    ion_charge_states, electron_temperature_eV, *, gaunt_factor=1.0,
):
    """Thermal free-free absorption coefficient alpha_nu [m^-1].

    Kirchhoff's law gives alpha_nu=j_nu/B_nu(T).  Evaluating that ratio
    algebraically cancels exp(-h nu/kT) before floating-point evaluation:
      alpha_nu = C_ff n_e sum(Z_i^2 n_i) T_K^-1/2 g_ff
                 [1-exp(-x)] / [2 h nu^3/c^2], x=h nu/(kT).
    This avoids a 0/0 underflow in the Wien tail while retaining stimulated
    emission and the exact Planck source function.
    """
    nu = _positive_array(frequency_Hz, "frequency_Hz")
    ne = _real_scalar(electron_density_m3, "electron_density_m3", nonnegative=True)
    te = _real_scalar(electron_temperature_eV, "electron_temperature_eV", positive=True)
    ni = _real_array(ion_densities_m3, "ion_densities_m3")
    z = _real_array(ion_charge_states, "ion_charge_states")
    g = _real_array(gaunt_factor, "gaunt_factor")
    if ni.ndim != 1 or z.shape != ni.shape or ni.size == 0 or np.any(ni < 0.0) or np.any(z <= 0.0):
        raise ValueError("ion densities and charge states must be matching finite vectors with n_i>=0 and Z_i>0")
    if np.any(z != np.floor(z)):
        raise ValueError("ion_charge_states must be positive integer ionic charge states")
    if ne == 0.0 and np.any(ni != 0.0):
        raise ValueError("zero electron density requires zero positive-ion densities in this quasineutral plasma model")
    if np.any(g <= 0.0):
        raise ValueError("gaunt_factor must be positive and finite")
    try:
        g = np.broadcast_to(g, nu.shape)
    except ValueError as exc:
        raise ValueError("gaunt_factor must be scalar or broadcastable to frequency_Hz") from exc
    temperature_K = te * E_CHARGE / K_B
    x = H * nu / (K_B * temperature_K)
    charge_weighted_ions = float(np.sum(z**2 * ni))
    coefficient_si = _FREE_FREE_EMISSIVITY_COEFFICIENT_SI
    emissivity_prefactor = (
        coefficient_si * ne * charge_weighted_ions
        * temperature_K**-0.5 * g
    )
    planck_prefactor = 2.0 * H * nu**3 / C**2
    return emissivity_prefactor * (-np.expm1(-x)) / planck_prefactor


def ece_lte_emissivity_si(
    frequency_Hz, absorption_coefficient_m_inverse, electron_temperature_eV,
):
    """Mode-resolved LTE ECE emissivity j_nu,m = alpha_nu,m B_nu(T).

    The absorption input may have shape (frequency,) or
    (frequency, mode). The returned emissivity has the identical shape.
    This is a Kirchhoff/LTE closure, not a non-LTE ECE model.
    """
    nu = _positive_array(frequency_Hz, "frequency_Hz")
    alpha = _real_array(absorption_coefficient_m_inverse, "absorption_coefficient_m_inverse")
    te = _real_scalar(electron_temperature_eV, "electron_temperature_eV", positive=True)
    if alpha.ndim not in (1, 2) or alpha.shape[0] != nu.size or np.any(alpha < 0.0):
        raise ValueError(
            "absorption_coefficient_m_inverse must be non-negative with shape "
            "(n_frequency,) or (n_frequency,n_mode)"
        )
    source = planck_spectral_radiance(nu, te * E_CHARGE / K_B)
    return alpha * (source if alpha.ndim == 1 else source[:, None])


def integrate_thermal_microwave_background(
    frequency_Hz, path_length_m, electron_temperature_eV, *,
    electron_density_m3, ion_densities_m3, ion_charge_states,
    ece_absorption_coefficient_m_inverse=None,
    ece_mode_weights=None, gaunt_factor=1.0,
    ece_emissivity_W_m3_sr_Hz=None, ece_lte=False,
    incident_radiance_W_m2_sr_Hz=0.0,
):
    """Integrate free-free and mode-resolved ECE radiative transfer.

    ECE absorption may be supplied as (path,frequency) for a single effective
    mode, or (path,frequency,mode) for independently propagated receiver modes.
    In the latter case ece_mode_weights is the measured receiver power fraction
    for each mode and must be non-negative and sum to one. ECE emissivity may be
    supplied with the same shape. If it is omitted, callers must explicitly set
    ece_lte=True to select the Planck source function; LTE is never inferred from
    an absorption coefficient alone. Each mode is propagated independently.
    This does not model mode conversion or coherent cross-mode coupling.
    """
    nu = _positive_array(frequency_Hz, "frequency_Hz")
    length = _real_array(path_length_m, "path_length_m")
    te = _real_array(electron_temperature_eV, "electron_temperature_eV")
    ne = _real_array(electron_density_m3, "electron_density_m3")
    ni = _real_array(ion_densities_m3, "ion_densities_m3")
    z = _real_array(ion_charge_states, "ion_charge_states")
    if length.ndim != 1 or te.shape != length.shape or ne.shape != length.shape:
        raise ValueError("path_length_m, electron_temperature_eV and electron_density_m3 must be matching 1-D path arrays")
    if ni.ndim != 2 or ni.shape[0] != length.size or z.shape != (ni.shape[1],):
        raise ValueError("ion_densities_m3 must be (n_path,n_species) and ion_charge_states one value per species")
    if np.any((ne == 0.0)[:, None] & (ni != 0.0)):
        raise ValueError("zero electron density requires zero positive-ion densities on the same path elements")
    if (np.any(~np.isfinite(length)) or np.any(length <= 0)
            or np.any(~np.isfinite(te)) or np.any(te <= 0)
            or np.any(~np.isfinite(ne)) or np.any(ne < 0)):
        raise ValueError("path lengths and temperatures must be positive finite; electron densities must be finite and non-negative")
    gaunt = _real_array(gaunt_factor, "gaunt_factor")
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
        if ece_mode_weights is not None:
            raise ValueError("ece_mode_weights requires mode-resolved ECE absorption")
        n_modes = 1
        ece_alpha = np.zeros((length.size, nu.size, n_modes), dtype=float)
        ece_status = "ECE omitted; no mode-resolved absorption supplied"
        mode_weights = np.ones(n_modes, dtype=float)
    else:
        supplied_alpha = _real_array(ece_absorption_coefficient_m_inverse, "ece_absorption_coefficient_m_inverse")
        if supplied_alpha.ndim == 2 and supplied_alpha.shape == (length.size, nu.size):
            if ece_mode_weights is not None:
                raise ValueError("ece_mode_weights is only valid with a 3-D mode-resolved ECE absorption array")
            ece_alpha = supplied_alpha[:, :, None]
            n_modes = 1
            mode_weights = np.ones(1, dtype=float)
            ece_status = "mode-resolved single-effective-mode ECE absorption supplied; source function selected separately"
        elif supplied_alpha.ndim == 3 and supplied_alpha.shape[:2] == (length.size, nu.size):
            n_modes = supplied_alpha.shape[2]
            if n_modes < 1:
                raise ValueError("mode-resolved ECE absorption must contain at least one mode")
            if ece_mode_weights is None:
                raise ValueError("ece_mode_weights is required for mode-resolved ECE absorption")
            mode_weights = np.asarray(ece_mode_weights, dtype=float)
            if (mode_weights.shape != (n_modes,) or np.any(~np.isfinite(mode_weights))
                    or np.any(mode_weights < 0.0) or not np.isclose(mode_weights.sum(), 1.0, rtol=1e-8, atol=1e-10)):
                raise ValueError("ece_mode_weights must be a non-negative length-n_modes vector summing to one")
            ece_alpha = supplied_alpha
            ece_status = "independently propagated mode-resolved ECE absorption; source function selected separately"
        else:
            raise ValueError("ece_absorption_coefficient_m_inverse must have shape (n_path,n_frequency) or (n_path,n_frequency,n_mode)")
        if np.any(~np.isfinite(ece_alpha)) or np.any(ece_alpha < 0.0):
            raise ValueError("ece absorption coefficients must be finite and non-negative")
        vacuum_path = ne == 0.0
        if np.any(vacuum_path) and np.any(ece_alpha[vacuum_path] != 0.0):
            raise ValueError("ECE absorption must be zero on path elements with zero electron density")
    if not isinstance(ece_lte, (bool, np.bool_)):
        raise ValueError("ece_lte must be an explicit boolean")
    if ece_absorption_coefficient_m_inverse is None:
        if ece_emissivity_W_m3_sr_Hz is not None or ece_lte:
            raise ValueError("ECE emissivity or LTE source selection requires ECE absorption")
        ece_emissivity = np.zeros_like(ece_alpha)
        ece_source_model = "ECE omitted"
    elif ece_emissivity_W_m3_sr_Hz is None:
        if not ece_lte:
            raise ValueError(
                "ECE absorption alone does not define emissivity; supply "
                "ece_emissivity_W_m3_sr_Hz or explicitly set ece_lte=True"
            )
        ece_emissivity = None
        ece_source_model = "explicit_LTE_Planck_source_function"
    else:
        if ece_lte:
            raise ValueError("supply ECE emissivity or ece_lte=True, not both")
        supplied_j = _real_array(ece_emissivity_W_m3_sr_Hz, "ece_emissivity_W_m3_sr_Hz")
        if supplied_j.ndim == 2 and supplied_j.shape == (length.size, nu.size) and n_modes == 1:
            supplied_j = supplied_j[:, :, None]
        if supplied_j.shape != (length.size, nu.size, n_modes):
            raise ValueError(
                "ece_emissivity_W_m3_sr_Hz must match ECE absorption as "
                "(n_path,n_frequency) or (n_path,n_frequency,n_mode)"
            )
        if np.any(~np.isfinite(supplied_j)) or np.any(supplied_j < 0.0):
            raise ValueError("ECE emissivity must be finite and non-negative")
        ece_emissivity = supplied_j
        ece_source_model = "explicit_mode_resolved_ECE_emissivity"

    incident = _real_array(incident_radiance_W_m2_sr_Hz, "incident_radiance_W_m2_sr_Hz")
    if incident.ndim == 0:
        incident = np.full(nu.size, float(incident))
    if incident.shape != nu.shape or np.any(~np.isfinite(incident)) or np.any(incident < 0):
        raise ValueError("incident_radiance_W_m2_sr_Hz must be non-negative scalar or frequency vector")
    intensity_modes = np.broadcast_to(incident, (n_modes, nu.size)).copy()
    alpha_ff_total = np.zeros(nu.size)
    tau_ece_modes = np.zeros((n_modes, nu.size), dtype=float)
    for index in range(length.size):
        j_ff = thermal_bremsstrahlung_emissivity_si(
            nu, ne[index], ni[index], z, te[index],
            gaunt_factor=gaunt_path[index],
        )
        b_nu = planck_spectral_radiance(nu, te[index] * E_CHARGE / K_B)
        # Kirchhoff absorption is evaluated with the exact Planck source function.
        alpha_ff = thermal_bremsstrahlung_absorption_si(
            nu, ne[index], ni[index], z, te[index], gaunt_factor=gaunt_path[index]
        )
        alpha_total = alpha_ff[None, :] + ece_alpha[index].T
        j_ece = (
            ece_lte_emissivity_si(
                nu, ece_alpha[index], te[index]
            ).T
            if ece_emissivity is None else ece_emissivity[index].T
        )
        j_total = j_ff[None, :] + j_ece
        optical_depth = alpha_total * length[index]
        bounded_depth = np.minimum(optical_depth, 700.0)
        transmission = np.exp(-bounded_depth)
        emission_integral = np.full_like(alpha_total, length[index])
        np.divide(
            -np.expm1(-bounded_depth), alpha_total,
            out=emission_integral, where=alpha_total > np.finfo(float).tiny,
        )
        intensity_modes = intensity_modes * transmission + j_total * emission_integral
        alpha_ff_total += alpha_ff * length[index]
        tau_ece_modes += ece_alpha[index].T * length[index]
    intensity = mode_weights @ intensity_modes
    # An incoherent receiver mixture attenuates as sum_m w_m exp(-tau_m),
    # not exp(-sum_m w_m tau_m).  Report the exact effective optical depth
    # for incident radiation as well as the mode-resolved depths.
    total_tau_by_mode = alpha_ff_total[None, :] + tau_ece_modes
    minimum_tau = np.min(total_tau_by_mode, axis=0)
    scaled_transmission = mode_weights @ np.exp(
        -(total_tau_by_mode - minimum_tau[None, :])
    )
    effective_total_tau = minimum_tau - np.log(scaled_transmission)
    has_free_free = bool(np.any(alpha_ff_total > 0.0))
    has_ece = bool(np.any(tau_ece_modes > 0.0) or (
        ece_emissivity is not None and np.any(ece_emissivity > 0.0)
    ))
    has_incident = bool(np.any(incident > 0.0))
    if has_free_free or has_ece:
        background_status = "plasma_emission_or_absorption_present"
    elif has_incident:
        background_status = "incident_radiance_only_no_plasma_background"
    else:
        background_status = "zero_background_exact_zero_radiance"
    # Dimensionless diagnostics expose when the non-relativistic free-free
    # kernel is being used outside its asymptotically non-relativistic regime.
    # These are screening parameters, not a correction factor or an inferred
    # relativistic emissivity.
    electron_rest_energy_J = 9.1093837015e-31 * C**2
    theta_e = (te * E_CHARGE) / electron_rest_energy_J
    photon_recoil_parameter = (H * nu) / electron_rest_energy_J
    charge_balance = [
        plasma_charge_state_moments(ne[index], ni[index], z)
        for index in range(length.size)
    ]
    return {
        "spectral_radiance_W_m2_sr_Hz": intensity,
        "plasma_charge_state_moments_by_path": charge_balance,
        "background_status": background_status,
        "free_free_relativistic_screening": {
            "electron_temperature_parameter_kT_over_mec2_by_path": theta_e.copy(),
            "maximum_electron_temperature_parameter_kT_over_mec2": float(np.max(theta_e)),
            "photon_recoil_parameter_hnu_over_mec2": photon_recoil_parameter.copy(),
            "relativistic_corrections_applied": False,
            "model_scope": "nonrelativistic_Maxwellian_free_free_only",
            "interpretation": (
                "Dimensionless regime diagnostics only; no threshold-based "
                "model switch or relativistic correction is applied."
            ),
        },
        "zero_background": not (has_free_free or has_ece or has_incident),
        "spectral_radiance_by_mode_W_m2_sr_Hz": intensity_modes,
        "receiver_mode_weights": mode_weights.copy(),
        "thermal_bremsstrahlung_optical_depth": alpha_ff_total,
        "ece_optical_depth": tau_ece_modes.T @ mode_weights,
        "ece_optical_depth_by_mode": tau_ece_modes,
        "total_optical_depth": effective_total_tau,
        "total_optical_depth_by_mode": total_tau_by_mode.T,
        "mean_total_optical_depth": alpha_ff_total + tau_ece_modes.T @ mode_weights,
        "ece_model_status": ece_status,
        "ece_source_model": ece_source_model,
        "gaunt_factor_resolution": gaunt_status,
        "free_free_model": "nonrelativistic_Maxwellian_thermal_bremsstrahlung_with_supplied_Gaunt_factor",
        "free_free_absorption_model": "Kirchhoff_alpha_nu_equals_j_nu_over_exact_Planck_B_nu",
        "ece_lte_emissivity_model": ("Kirchhoff_j_nu_mode_equals_alpha_nu_mode_times_Planck_B_nu" if ece_lte else "explicit_non_LTE_emissivity_or_ECE_omitted"),
        "radiative_transfer": "independent mode-by-mode slab formal solution with LTE Kirchhoff source function; receiver-mode weighted after propagation",
        "mode_coupling_model": "neglected; modes propagated independently",
        "receiver_plane_conversion": "multiply radiance by measured receiver etendue and end-to-end efficiency; do not count the same throughput twice",
        "limitations": [
            "Free-free emissivity is non-relativistic and unmagnetized; use a relativistic/magnetized emissivity kernel outside that regime.",
            "The supplied Gaunt factor may be scalar, frequency-resolved, path-resolved, or path-and-frequency-resolved; it is not inferred from incomplete plasma inputs.",
            "ECE emissivity is independent physical input unless LTE is explicitly selected; absorption alone does not determine a non-LTE source function.",
            "ECE absorption must be supplied by a mode-resolved magnetized plasma solver; no scalar approximation is silently substituted for multiple supplied modes.",
            "Mode conversion and coherent cross-mode coupling are not included; the receiver mode weights are external measured inputs.",
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
    radiance = _real_array(spectral_radiance_W_m2_sr_Hz, "spectral_radiance_W_m2_sr_Hz")
    etendue = _real_scalar(receiver_etendue_m2_sr, "receiver_etendue_m2_sr", positive=True)
    efficiency = _real_scalar(end_to_end_efficiency, "end_to_end_efficiency", positive=True)
    if radiance.size == 0 or np.any(~np.isfinite(radiance)) or np.any(radiance < 0):
        raise ValueError("spectral radiance must be finite and non-negative")
    if not 0 < efficiency <= 1:
        raise ValueError("end_to_end_efficiency must lie in (0,1]")
    return radiance * etendue * efficiency
