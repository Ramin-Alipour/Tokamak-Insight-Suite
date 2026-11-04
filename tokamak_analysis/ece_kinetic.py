"""Magnetized kinetic ECE absorption from the linear Vlasov conductivity.

Absorption is normalized by the actual incident normal Poynting flux of the
supplied plane-wave eigenmode, rather than assuming vacuum impedance. The
kinetic conductivity is non-relativistic Maxwellian; this is not a relativistic
or collisional-Landau closure.
"""
from __future__ import annotations
import numpy as np
from scipy.constants import c, epsilon_0, mu_0
from .cts_full_em import _species_conductivity, ME, E_CHARGE
from .cts_relativistic import relativistic_gyrotropic_species_response


def _strict_real_scalar(value, name, *, positive=False, nonnegative=False):
    """Validate physical scalars before conversion; reject bool/string/complex coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real numeric scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real numeric scalar")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _strict_real_vector(value, name):
    """Return a finite real vector without accepting strings or booleans."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain finite real numeric values")
    result = raw.astype(float, copy=False)
    if result.ndim != 1 or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite real vector")
    return result


def kinetic_ece_absorption_coefficient(
    frequency_Hz, wavevector_rad_m, magnetic_field_T, *,
    electron_density_m3, electron_temperature_eV,
    polarization, harmonic_order=8, quadrature_order=48,
    damping_rate_rad_s=0.0,
):
    """Return kinetic absorption alpha [m^-1] for a supplied EM eigenmode.

    For exp(-i omega t), Q=Re(E* . sigma . E)/2 and
    alpha=Q/<S_n>. The supplied wavevector and polarization must belong to
    the same local mode. Negative alpha (net kinetic gain) is retained.
    """
    f = _strict_real_scalar(frequency_Hz, "frequency_Hz", positive=True)
    ne = _strict_real_scalar(electron_density_m3, "electron_density_m3", positive=True)
    te = _strict_real_scalar(electron_temperature_eV, "electron_temperature_eV", positive=True)
    k = _strict_real_vector(wavevector_rad_m, "wavevector_rad_m")
    b = _strict_real_vector(magnetic_field_T, "magnetic_field_T")
    raw_polarization = np.asarray(polarization)
    if raw_polarization.dtype.kind not in "iufc":
        raise ValueError("polarization must contain finite numeric values")
    e = raw_polarization.astype(complex, copy=False)
    gamma = _strict_real_scalar(damping_rate_rad_s, "damping_rate_rad_s", nonnegative=True)
    if not np.isfinite(f) or f <= 0 or not np.isfinite(ne) or ne <= 0 or not np.isfinite(te) or te <= 0:
        raise ValueError("frequency, density and temperature must be positive and finite")
    if k.shape != (3,) or b.shape != (3,) or e.shape != (3,) or any(np.any(~np.isfinite(x)) for x in (k,b,e)):
        raise ValueError("wavevector, magnetic field and polarization must be finite 3-vectors")
    if np.linalg.norm(k) == 0 or np.linalg.norm(b) == 0 or np.linalg.norm(e) == 0:
        raise ValueError("wavevector, magnetic field and polarization must be non-zero")
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("damping_rate_rad_s must be finite and non-negative")
    if isinstance(harmonic_order, (bool, np.bool_)) or int(harmonic_order) != harmonic_order or harmonic_order < 1:
        raise ValueError("harmonic_order must be a positive integer")
    if isinstance(quadrature_order, (bool, np.bool_)) or int(quadrature_order) != quadrature_order or quadrature_order < 16:
        raise ValueError("quadrature_order must be an integer >= 16")
    omega = 2*np.pi*f
    species = {"name":"e-", "density":ne, "temperature":te,
               "mass":ME, "charge":-E_CHARGE, "flow":np.zeros(3)}
    sigma = _species_conductivity(
        omega, k, b, species, harmonic_order=int(harmonic_order),
        quadrature_order=int(quadrature_order), damping_rate_rad_s=gamma,
    )
    e = e / np.linalg.norm(e)
    dissipated = float(np.real(np.vdot(e, sigma @ e)))
    # H=k x E/(mu0 omega); the signed normal flux follows the wavevector.
    h = np.cross(k, e)/(mu_0*omega)
    khat = k/np.linalg.norm(k)
    flux = 0.5*float(np.real(np.dot(np.cross(e, np.conj(h)), khat)))
    if not np.isfinite(flux) or flux <= 0:
        raise ValueError("supplied polarization/wavevector has non-positive normal Poynting flux")
    alpha = 0.5*dissipated/flux
    return {
        "absorption_coefficient_m_inverse": float(alpha),
        "conductivity_S_m": sigma,
        "polarization_unit": e,
        "dissipative_conductivity_S_m": dissipated,
        "incident_normal_poynting_flux_W_m2_per_V2_m2": flux,
        "frequency_Hz": f,
        "harmonic_order": int(harmonic_order),
        "quadrature_order": int(quadrature_order),
        "damping_rate_rad_s": gamma,
        "model": "magnetized_nonrelativistic_Maxwellian_linear_Vlasov_conductivity",
        "absorption_normalization": "Poynting_theorem_actual_supplied_plane_wave_mode",
        "limitations": [
            "No relativistic/Juttner distribution averaging.",
            "No self-consistent ray tracing or mode conversion; supply the local mode wavevector and polarization.",
            "Damping is user-specified; this routine does not implement a conserving collisional Landau operator.",
            "The plane-wave flux normalization assumes a homogeneous local mode and does not include beam-envelope energy transport.",
        ],
    }



def relativistic_kinetic_ece_absorption_coefficient(
    frequency_Hz, wavevector_rad_m, magnetic_field_T, *,
    electron_density_m3, electron_distribution,
    polarization, quadrature_order=8, gyrophase_points=32,
    resonance_width_rad_s=1e6,
):
    """Relativistic magnetized ECE absorption for an explicit gyrotropic EVDF.

    The relativistic momentum-space Vlasov conductivity is projected onto the
    supplied local EM eigenmode. Signed absorption follows Poynting's theorem;
    negative values indicate kinetic gain and are not clipped. The EVDF is an
    explicit physical input, not inferred from electron temperature.
    """
    f = _strict_real_scalar(frequency_Hz, "frequency_Hz", positive=True)
    ne = _strict_real_scalar(electron_density_m3, "electron_density_m3", positive=True)
    k = _strict_real_vector(wavevector_rad_m, "wavevector_rad_m")
    b = _strict_real_vector(magnetic_field_T, "magnetic_field_T")
    raw_polarization = np.asarray(polarization)
    if raw_polarization.dtype.kind not in "iufc":
        raise ValueError("polarization must contain finite numeric values")
    e = raw_polarization.astype(complex, copy=False)
    width = _strict_real_scalar(resonance_width_rad_s, "resonance_width_rad_s", positive=True)
    if not np.isfinite(f) or f <= 0 or not np.isfinite(ne) or ne <= 0:
        raise ValueError("frequency_Hz and electron_density_m3 must be positive and finite")
    if (k.shape != (3,) or b.shape != (3,) or e.shape != (3,)
            or any(np.any(~np.isfinite(v)) for v in (k, b, e))):
        raise ValueError("wavevector_rad_m, magnetic_field_T and polarization must be finite 3-vectors")
    if np.linalg.norm(k) == 0 or np.linalg.norm(b) == 0 or np.linalg.norm(e) == 0:
        raise ValueError("wavevector, magnetic field and polarization must be non-zero")
    if not np.isfinite(width) or width <= 0:
        raise ValueError("resonance_width_rad_s must be positive and finite")
    if (isinstance(quadrature_order, (bool, np.bool_))
            or int(quadrature_order) != quadrature_order or quadrature_order < 4):
        raise ValueError("quadrature_order must be an integer >= 4")
    if (isinstance(gyrophase_points, (bool, np.bool_))
            or int(gyrophase_points) != gyrophase_points
            or gyrophase_points < 16 or gyrophase_points % 2):
        raise ValueError("gyrophase_points must be an even integer >= 16")

    omega = 2.0 * np.pi * f
    response = relativistic_gyrotropic_species_response(
        omega, k, b, density_m3=ne, mass_kg=ME, charge_C=-E_CHARGE,
        distribution=electron_distribution,
        quadrature_order=int(quadrature_order),
        gyrophase_points=int(gyrophase_points),
        resonance_width_rad_s=width,
    )
    e_unit = e / np.linalg.norm(e)
    sigma = np.asarray(response["conductivity_S_m"], dtype=complex)
    dissipated = float(np.real(np.vdot(e_unit, sigma @ e_unit)))
    h = np.cross(k, e_unit) / (mu_0 * omega)
    khat = k / np.linalg.norm(k)
    flux = 0.5 * float(np.real(np.dot(np.cross(e_unit, np.conj(h)), khat)))
    if not np.isfinite(flux) or flux <= 0:
        raise ValueError("supplied polarization/wavevector has non-positive normal Poynting flux")
    alpha = 0.5 * dissipated / flux
    return {
        **response,
        "absorption_coefficient_m_inverse": float(alpha),
        "dissipative_conductivity_S_m": dissipated,
        "incident_normal_poynting_flux_W_m2_per_V2_m2": flux,
        "polarization_unit": e_unit,
        "frequency_Hz": f,
        "model": "relativistic_gyrotropic_linear_Vlasov_ECE",
        "absorption_normalization": "Poynting_theorem_actual_supplied_plane_wave_mode",
        "distribution_is_external": True,
        "noise_model_used_for_absorption": False,
        "limitations": [
            "Collisionless relativistic gyrotropic response; no conserving collisional Landau operator is included.",
            "The EVDF is supplied explicitly and is not inferred by this forward absorption calculation.",
            "The local wavevector and polarization must be a mutually consistent eigenmode of the local dielectric.",
            "No self-consistent ray tracing, mode conversion, polarized transfer, or experimental validation is performed.",
            "Finite resonance width and momentum/gyrophase quadrature require convergence studies.",
        ],
    }


from scipy.constants import Boltzmann as K_B, elementary_charge as E_CHARGE



def solve_relativistic_kinetic_ece_profile(
    frequency_Hz, segment_length_m, *, wavevector_rad_m, magnetic_field_T,
    polarization, electron_density_m3, electron_distributions,
    quadrature_order=8, gyrophase_points=32, resonance_width_rad_s=1e6,
):
    """Evaluate relativistic gyrotropic ECE absorption along a supplied ray.

    One explicit normalized EVDF is required for each path segment. The
    function returns signed local absorption and does not assume LTE or
    construct an emissivity/source function for a non-Maxwellian population.
    """
    f = np.asarray(frequency_Hz, dtype=float)
    if f.ndim == 0:
        f = f.reshape(1)
    if f.ndim != 1 or f.size == 0 or np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequency_Hz must be a positive finite scalar or vector")
    lengths = np.asarray(segment_length_m, dtype=float)
    if lengths.ndim != 1 or lengths.size == 0 or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("segment_length_m must be a non-empty positive finite vector")
    ns, nf = lengths.size, f.size

    def stack(value, name, complex_values=False):
        array = np.asarray(value, dtype=complex if complex_values else float)
        if array.shape == (ns, 3):
            array = np.broadcast_to(array[None, :, :], (nf, ns, 3))
        if array.shape != (nf, ns, 3) or np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must have shape (n_segment,3) or (n_frequency,n_segment,3)")
        return array

    k_stack = stack(wavevector_rad_m, "wavevector_rad_m")
    b_stack = stack(magnetic_field_T, "magnetic_field_T")
    e_stack = stack(polarization, "polarization", complex_values=True)
    density = np.asarray(electron_density_m3, dtype=float)
    if density.shape != (ns,) or np.any(~np.isfinite(density)) or np.any(density <= 0):
        raise ValueError("electron_density_m3 must contain one positive finite value per segment")
    if not isinstance(electron_distributions, (list, tuple)) or len(electron_distributions) != ns:
        raise ValueError("electron_distributions must contain one explicit EVDF mapping per segment")

    alpha = np.empty((nf, ns), dtype=float)
    local_results = []
    for i in range(nf):
        row = []
        for j in range(ns):
            local = relativistic_kinetic_ece_absorption_coefficient(
                f[i], k_stack[i, j], b_stack[i, j],
                electron_density_m3=density[j],
                electron_distribution=electron_distributions[j],
                polarization=e_stack[i, j],
                quadrature_order=quadrature_order,
                gyrophase_points=gyrophase_points,
                resonance_width_rad_s=resonance_width_rad_s,
            )
            alpha[i, j] = local["absorption_coefficient_m_inverse"]
            row.append(local)
        local_results.append(row)
    return {
        "frequency_Hz": f.copy(),
        "segment_length_m": lengths.copy(),
        "absorption_coefficient_m_inverse": alpha,
        "local_relativistic_responses": local_results,
        "net_gain_mask": alpha < 0.0,
        "profile_solver": "segmentwise_relativistic_gyrotropic_Vlasov_absorption",
        "transfer_performed": False,
        "limitations": [
            "The EVDF is explicit per segment; no distribution or temperature is inferred.",
            "Signed gain is retained. No passive LTE transfer or Kirchhoff source is assumed.",
            "A non-LTE emissivity requires a consistent fluctuation/noise closure and is not inferred from the conductivity alone.",
            "Ray trajectory, mode conversion and eigenmode evolution are external inputs.",
            "Momentum, gyrophase and resonance-width convergence remain required.",
        ],
    }


def integrate_kinetic_ece_radiative_transfer(
    frequency_Hz, segment_length_m, absorption_coefficient_m_inverse,
    electron_temperature_eV, *, incident_brightness_temperature_K=0.0,
):
    """Integrate Rayleigh-Jeans ECE transfer through ordered slab segments.

    The exact piecewise-constant update is Tb_out=Tb_in exp(-alpha L)
    + Te(1-exp(-alpha L)). Absorption must be non-negative; gain is not
    silently clipped. This scalar transfer step does not perform ray tracing.
    """
    f = np.asarray(frequency_Hz, dtype=float)
    lengths = np.asarray(segment_length_m, dtype=float)
    alpha = np.asarray(absorption_coefficient_m_inverse, dtype=float)
    te = np.asarray(electron_temperature_eV, dtype=float)
    tb0 = np.asarray(incident_brightness_temperature_K, dtype=float)
    if f.ndim > 1 or f.size == 0 or np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequency_Hz must be positive finite scalar or 1-D array")
    f = f.reshape(-1)
    if lengths.ndim != 1 or lengths.size == 0 or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("segment_length_m must be a non-empty positive finite vector")
    if te.shape != lengths.shape or np.any(~np.isfinite(te)) or np.any(te <= 0):
        raise ValueError("electron_temperature_eV must be positive with one value per segment")
    if alpha.ndim == 1:
        if alpha.shape != lengths.shape:
            raise ValueError("1-D absorption coefficient must match segment_length_m")
        alpha = np.broadcast_to(alpha[None, :], (f.size, lengths.size)).copy()
    elif alpha.shape != (f.size, lengths.size):
        raise ValueError("2-D absorption coefficient must have shape (n_frequency, n_segment)")
    if np.any(~np.isfinite(alpha)) or np.any(alpha < 0):
        raise ValueError("absorption coefficients must be finite and non-negative")
    if (np.any(~np.isfinite(tb0)) or np.any(tb0 < 0) or tb0.ndim > 1
            or (tb0.ndim == 1 and tb0.shape != f.shape)):
        raise ValueError("incident brightness must be non-negative scalar or match frequency_Hz")
    source_K = te * E_CHARGE / K_B
    tau = alpha * lengths[None, :]
    brightness = np.broadcast_to(tb0, f.shape).astype(float, copy=True)
    for j in range(lengths.size):
        attenuation = np.exp(-tau[:, j])
        brightness = brightness * attenuation + source_K[j] * (-np.expm1(-tau[:, j]))
    total_tau = np.sum(tau, axis=1)
    return {
        "frequency_Hz": f.copy(),
        "brightness_temperature_out_K": brightness,
        "segment_optical_depth": tau,
        "total_optical_depth": total_tau,
        "transmission": np.exp(-total_tau),
        "source_function": "Rayleigh_Jeans_local_thermodynamic_equilibrium",
        "transfer_solver": "ordered_piecewise_constant_exact_formal_solution",
        "limitations": [
            "Uses supplied local kinetic absorption coefficients and segment temperatures.",
            "Assumes LTE and Rayleigh-Jeans source function.",
            "No relativistic emissivity, collisional Landau dielectric, or self-consistent ray tracing.",
            "No polarization conversion or transverse beam evolution in this scalar transfer step.",
        ],
    }



def solve_kinetic_ece_profile(
    frequency_Hz, segment_length_m, *,
    wavevector_rad_m, magnetic_field_T, polarization,
    electron_density_m3, electron_temperature_eV,
    incident_brightness_temperature_K=0.0,
    harmonic_order=8, quadrature_order=48, damping_rate_rad_s=0.0,
):
    """Solve local kinetic absorption and scalar transfer along a supplied ray.

    The caller supplies the local EM eigenmode wavevector and polarization in
    each segment. These must be mutually consistent with the local dielectric
    and propagation direction; this routine does not invent a mode or infer
    missing geometry. The magnetic field may be constant per segment or
    frequency dependent. Gain (negative alpha) is reported as unsupported by
    the LTE scalar transfer step and is never clipped to zero.
    """
    f = np.asarray(frequency_Hz, dtype=float)
    if f.ndim == 0:
        f = f.reshape(1)
    if f.ndim != 1 or f.size == 0 or np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequency_Hz must be a positive finite scalar or 1-D array")
    lengths = np.asarray(segment_length_m, dtype=float)
    if lengths.ndim != 1 or lengths.size == 0 or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("segment_length_m must be a non-empty positive finite vector")
    ns, nf = lengths.size, f.size

    def mode_stack(value, name):
        array = np.asarray(value, dtype=complex if name == "polarization" else float)
        if array.shape == (ns, 3):
            array = np.broadcast_to(array[None, :, :], (nf, ns, 3))
        if array.shape != (nf, ns, 3) or np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must have shape (n_segment,3) or (n_frequency,n_segment,3)")
        return array

    k_stack = mode_stack(wavevector_rad_m, "wavevector_rad_m")
    b_stack = mode_stack(magnetic_field_T, "magnetic_field_T")
    e_stack = mode_stack(polarization, "polarization")
    ne = np.asarray(electron_density_m3, dtype=float)
    te = np.asarray(electron_temperature_eV, dtype=float)
    if ne.shape != (ns,) or np.any(~np.isfinite(ne)) or np.any(ne <= 0):
        raise ValueError("electron_density_m3 must contain one positive finite value per segment")
    if te.shape != (ns,) or np.any(~np.isfinite(te)) or np.any(te <= 0):
        raise ValueError("electron_temperature_eV must contain one positive finite value per segment")

    alpha = np.empty((nf, ns), dtype=float)
    mode_results = []
    for i in range(nf):
        row = []
        for j in range(ns):
            local = kinetic_ece_absorption_coefficient(
                f[i], k_stack[i, j], b_stack[i, j],
                electron_density_m3=ne[j],
                electron_temperature_eV=te[j],
                polarization=e_stack[i, j],
                harmonic_order=harmonic_order,
                quadrature_order=quadrature_order,
                damping_rate_rad_s=damping_rate_rad_s,
            )
            alpha[i, j] = local["absorption_coefficient_m_inverse"]
            row.append(local)
        mode_results.append(row)
    if np.any(alpha < 0):
        indices = np.argwhere(alpha < 0).tolist()
        raise ValueError(
            "kinetic ECE produced net gain at frequency/segment indices "
            f"{indices}; the LTE passive scalar transfer solver cannot represent gain"
        )
    transfer = integrate_kinetic_ece_radiative_transfer(
        f, lengths, alpha, te,
        incident_brightness_temperature_K=incident_brightness_temperature_K,
    )
    return {
        **transfer,
        "absorption_coefficient_m_inverse": alpha,
        "local_mode_absorption": mode_results,
        "mode_geometry_source": "explicit_user_supplied_local_EM_eigenmode",
        "profile_solver": "segmentwise_magnetized_kinetic_Vlasov_absorption_plus_formal_transfer",
        "limitations": [
            "Requires a supplied local wavevector and polarization for every segment and frequency.",
            "Does not solve the eigenmode, ray trajectory or mode conversion; use the full-vector solver for stratified polarization coupling.",
            "Non-relativistic Maxwellian electrons and user-specified damping only.",
            "Scalar LTE Rayleigh-Jeans transfer; not a polarized or non-LTE transfer solver.",
            "Negative absorption is rejected as gain rather than clipped or treated as passive emission.",
        ],
    }
