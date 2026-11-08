"""Cold-plasma magnetoionic dielectric and local wave branches.

Stix R/L/S/D/P dielectric for a homogeneous cold multispecies plasma in a
uniform magnetic field, with optional species-resolved momentum-transfer
collisions. This is a local cold-fluid dispersion kernel, not a warm kinetic
CTS vertex or a ray-tracing/mode-conversion solution.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

EPS0 = 8.8541878128e-12

def _real_scalar(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real finite scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a real finite scalar")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def cold_plasma_dielectric_tensor(
    frequency_Hz, magnetic_field_T, species: Sequence[Mapping],
) -> dict:
    """Return the cold-plasma Stix dielectric tensor for exp(-i omega t).

    Each species mapping contains mass_kg, charge_C (signed), and density_m3.
    The signed gyrofrequency is Omega_s=q_s |B|/m_s. With B along +z
    and species momentum-transfer rate nu_s:
      R=1-sum(omega_ps^2/[omega(omega+i*nu_s+Omega_s)]),
      L=1-sum(omega_ps^2/[omega(omega+i*nu_s-Omega_s)]),
      P=1-sum(omega_ps^2/[omega(omega+i*nu_s)]),
      S=(R+L)/2, D=(R-L)/2.
    Setting every nu_s=0 recovers the collisionless Stix tensor. The tensor
    in a right-handed field-aligned basis is
      [[S,-iD,0],[iD,S,0],[0,0,P]].
    Optional species collision_frequency_rad_s uses the cold-fluid
    momentum-drag closure for exp(-i omega t). It does not represent a
    velocity-dependent kinetic collision operator. Finite-temperature and
    relativistic effects are not included. Collisionless cyclotron poles are
    reported rather than regularized by an invented damping rate.
    """
    f = _real_scalar(frequency_Hz, "frequency_Hz", positive=True)
    b = np.asarray(magnetic_field_T)
    if b.dtype.kind not in "iuf" or b.shape != (3,) or np.any(~np.isfinite(b)):
        raise ValueError("magnetic_field_T must be a finite real 3-vector")
    b = b.astype(float, copy=False)
    bnorm = float(np.linalg.norm(b))
    # The zero-field limit is a valid isotropic plasma; choose +z only as
    # an arbitrary coordinate axis because no magnetic-field direction exists.
    direction = b / bnorm if bnorm > 0.0 else np.array([0.0, 0.0, 1.0])
    if not isinstance(species, Sequence) or isinstance(species, (str, bytes)) or not species:
        raise ValueError("species must be a non-empty sequence of species mappings")
    omega = 2.0 * np.pi * f
    wp2_total = 0.0
    r_sum = 1.0 + 0.0j
    l_sum = 1.0 + 0.0j
    parsed = []
    for index, item in enumerate(species):
        if not isinstance(item, Mapping):
            raise ValueError(f"species[{index}] must be a mapping")
        mass = _real_scalar(item.get("mass_kg"), f"species[{index}].mass_kg", positive=True)
        charge = _real_scalar(item.get("charge_C"), f"species[{index}].charge_C")
        density = _real_scalar(item.get("density_m3"), f"species[{index}].density_m3", nonnegative=True)
        if charge == 0.0:
            raise ValueError(f"species[{index}].charge_C must be non-zero")
        wp2 = density * charge**2 / (EPS0 * mass)
        gyro = charge * bnorm / mass
        collision = _real_scalar(
            item.get("collision_frequency_rad_s", 0.0),
            f"species[{index}].collision_frequency_rad_s",
            nonnegative=True,
        )
        # exp(-i*omega*t): cold-fluid momentum drag gives omega -> omega+i*nu.
        complex_frequency = omega + 1j * collision
        plus = omega * (complex_frequency + gyro)
        minus = omega * (complex_frequency - gyro)
        scale = max(omega**2, abs(omega * gyro), omega * collision, np.finfo(float).tiny)
        if collision == 0.0 and (
            abs(plus) <= 1e-14 * scale or abs(minus) <= 1e-14 * scale
        ):
            raise ValueError("frequency lies on a collisionless cold-plasma cyclotron pole; supply finite collisions or a kinetic dielectric model")
        wp2_total += wp2
        r_sum -= wp2 / plus
        l_sum -= wp2 / minus
        parsed.append({
            "name": str(item.get("name", f"species_{index}")),
            "mass_kg": mass, "charge_C": charge, "density_m3": density,
            "plasma_frequency_squared_s_minus2": float(wp2),
            "signed_gyrofrequency_rad_s": float(gyro),
            "collision_frequency_rad_s": float(collision),
        })
    longitudinal_correction = sum(
        item["plasma_frequency_squared_s_minus2"]
        / (omega * (omega + 1j * item["collision_frequency_rad_s"]))
        for item in parsed
    )
    p_value = 1.0 - longitudinal_correction
    s_value = 0.5 * (r_sum + l_sum)
    d_value = 0.5 * (r_sum - l_sum)
    tensor_b = np.array([
        [s_value, -1j*d_value, 0.0j],
        [1j*d_value, s_value, 0.0j],
        [0.0j, 0.0j, p_value],
    ], dtype=complex)
    # Rotate the field-aligned tensor into the caller's laboratory basis.
    # The first transverse basis vector is chosen deterministically and is
    # only a coordinate convention; physical results are rotation invariant.
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(reference @ direction)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    e1 = reference - (reference @ direction) * direction
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    basis = np.column_stack((e1, e2, direction))
    tensor_lab = basis @ tensor_b @ basis.T
    return {
        "frequency_Hz": f,
        "angular_frequency_rad_s": omega,
        "magnetic_field_magnitude_T": bnorm,
        "magnetic_field_direction": (direction.copy() if bnorm > 0.0 else None),
        "species": parsed,
        "stix_R": complex(r_sum), "stix_L": complex(l_sum),
        "stix_S": complex(s_value), "stix_D": complex(d_value),
        "stix_P": complex(p_value),
        "dielectric_tensor_field_aligned": tensor_b,
        "dielectric_tensor_lab": tensor_lab,
        "dielectric_convention": "exp(-i*omega*t); Stix R/L/S/D/P",
        "model_scope": (
            ("homogeneous_cold_multispecies_unmagnetized_plasma" if bnorm == 0.0
             else "homogeneous_cold_multispecies_magnetized_plasma")
            + ("_with_species_resolved_cold_fluid_collisions"
               if any(item["collision_frequency_rad_s"] > 0.0 for item in parsed)
               else "_collisionless")
        ),
        "limitations": [
            "No finite-temperature/Landau damping; collisional response is cold-fluid Drude only.",
            "No relativistic response, finite-Larmor-radius correction or kinetic CTS vertex.",
            "Local homogeneous tensor only; no ray tracing, interfaces or mode conversion.",
            "At cyclotron/plasma resonances a cold-fluid tensor is singular or inadequate; no artificial damping is inserted.",
        ],
    }



def cold_plasma_absorbed_power_density(
    frequency_Hz, magnetic_field_T, species, electric_field_V_m,
) -> dict:
    """Compute local time-averaged collisional absorption from the cold-fluid dielectric.

    For the exp(-i*omega*t) convention, define the Hermitian dissipative
    dielectric H=(epsilon-epsilon^dagger)/(2i). The cycle-averaged work
    transferred from the wave to the plasma is
        <P> = (epsilon_0*omega/2) E^dagger H E.
    This is collisional cold-fluid absorption only; collisionless Landau,
    cyclotron-resonant kinetic and nonlinear absorption are not represented.
    """
    dielectric = cold_plasma_dielectric_tensor(
        frequency_Hz, magnetic_field_T, species,
    )
    raw = np.asarray(electric_field_V_m)
    if raw.shape != (3,) or raw.dtype.kind not in "iufc" or np.any(~np.isfinite(raw)):
        raise ValueError("electric_field_V_m must be a finite real or complex 3-vector")
    field = raw.astype(complex, copy=False)
    omega = dielectric["angular_frequency_rad_s"]
    epsilon = dielectric["dielectric_tensor_lab"]
    dissipative = (epsilon - epsilon.conj().T) / (2j)
    dissipative = 0.5 * (dissipative + dissipative.conj().T)
    quadratic = np.vdot(field, dissipative @ field)
    power = 0.5 * EPS0 * omega * float(np.real(quadratic))
    scale = 0.5 * EPS0 * omega * max(
        float(np.linalg.norm(field)**2 * np.linalg.norm(dissipative, ord=2)),
        np.finfo(float).tiny,
    )
    if power < -1e-11 * scale:
        raise ArithmeticError(
            "cold-fluid dielectric produced negative dissipative power; check convention or dielectric implementation"
        )
    return {
        **dielectric,
        "electric_field_V_m": field.copy(),
        "hermitian_dissipative_dielectric_tensor": dissipative,
        "absorbed_power_density_W_m3": max(power, 0.0),
        "power_balance_relation": "0.5*epsilon_0*omega*real(E^dagger*Im_H(epsilon_r)*E)",
        "absorption_scope": "local_time_averaged_species_resolved_cold_fluid_collisional_absorption",
        "limitations": [
            "No collisionless Landau damping or finite-temperature cyclotron-resonant absorption.",
            "No relativistic, finite-Larmor-radius, spatial transport or nonlinear absorption.",
            "The collision frequencies are external momentum-transfer closures, not inferred from the spectrum.",
        ],
    }



def cold_plasma_mode_radiative_coefficients(
    frequency_Hz, magnetic_field_T, species, electric_field_V_m, *,
    mode_poynting_flux_W_m2, electron_temperature_eV, assume_lte=False,
):
    """Return mode absorption and, only under explicit LTE, thermal emissivity.

    The supplied complex E field and positive modal Poynting flux must describe
    the same propagating eigenmode and normalization. The local attenuation
    coefficient follows energy conservation, alpha_m = <P_abs>/S_m, where
    <P_abs> = (eps0*omega/2) E^dagger Im_H(epsilon_r) E. Under an explicit
    local-thermodynamic-equilibrium assumption, Kirchhoff's law gives
    j_nu,m = alpha_nu,m B_nu(T_e).

    This supplies a magnetized, species-resolved *cold-fluid collisional*
    free-free/radiative closure. It is not a relativistic or warm-kinetic
    bremsstrahlung kernel; collision frequencies remain externally supplied
    momentum-transfer rates. No emissivity is inferred when assume_lte=False.
    """
    flux = _real_scalar(
        mode_poynting_flux_W_m2, "mode_poynting_flux_W_m2", positive=True
    )
    temperature = _real_scalar(
        electron_temperature_eV, "electron_temperature_eV", positive=True
    )
    if not isinstance(assume_lte, (bool, np.bool_)):
        raise ValueError("assume_lte must be an explicit boolean")
    absorption = cold_plasma_absorbed_power_density(
        frequency_Hz, magnetic_field_T, species, electric_field_V_m
    )
    alpha = absorption["absorbed_power_density_W_m3"] / flux
    emissivity = None
    source = None
    if assume_lte:
        from .cts_em_background import planck_spectral_radiance
        source = float(planck_spectral_radiance(
            np.asarray([_real_scalar(frequency_Hz, "frequency_Hz", positive=True)]),
            temperature * 1.602176634e-19 / 1.380649e-23,
        )[0])
        emissivity = alpha * source
    return {
        **absorption,
        "mode_poynting_flux_W_m2": flux,
        "absorption_coefficient_m_inverse": float(alpha),
        "electron_temperature_eV": temperature,
        "lte_assumed": bool(assume_lte),
        "planck_source_radiance_W_m2_sr_Hz": source,
        "thermal_emissivity_W_m3_sr_Hz": (
            None if emissivity is None else float(emissivity)
        ),
        "absorption_relation": "alpha_mode = time_averaged_absorbed_power_density / modal_Poynting_flux",
        "emissivity_relation": (
            "j_nu_mode = alpha_mode * B_nu(T_e), explicit_LTE_Kirchhoff_closure"
            if assume_lte else "not_evaluated_without_explicit_LTE_assumption"
        ),
        "model_scope": (
            "magnetized_multispecies_cold_fluid_collisional_mode_absorption_"
            + ("and_LTE_thermal_emissivity" if assume_lte else "only")
        ),
        "limitations": [
            "The E-field amplitude and modal Poynting flux must use the same eigenmode and normalization.",
            "Cold-fluid momentum-transfer absorption; no finite-temperature Landau/cyclotron resonance or relativistic correction.",
            "LTE emissivity is returned only when assume_lte=True; non-LTE source functions require independent emissivity data.",
            "This local mode closure does not include ray evolution, mode conversion or coherent cross-mode coupling.",
        ],
    }


def cold_plasma_magnetoionic_modes(
    frequency_Hz, magnetic_field_T, species: Sequence[Mapping], *,
    wave_normal_direction=(0.0, 0.0, 1.0),
) -> dict:
    """Solve the two local cold-plasma refractive-index branches.

    Solves det[epsilon - N(I-nhat nhat^T)]=0 for N=n^2. Since the
    transverse projector has rank two, the determinant is quadratic in N.
    Complex roots are retained (evanescent/absorbing algebraic branches);
    no root is silently discarded. Eigenvectors are returned in the lab basis.
    """
    dielectric = cold_plasma_dielectric_tensor(
        frequency_Hz, magnetic_field_T, species,
    )
    raw_nhat = np.asarray(wave_normal_direction)
    if raw_nhat.dtype.kind not in "iuf" or raw_nhat.shape != (3,):
        raise ValueError("wave_normal_direction must be a finite real non-zero 3-vector")
    nhat = raw_nhat.astype(float, copy=True)
    if np.any(~np.isfinite(nhat)) or np.linalg.norm(nhat) <= 0.0:
        raise ValueError("wave_normal_direction must be a finite real non-zero 3-vector")
    nhat /= np.linalg.norm(nhat)
    epsilon = dielectric["dielectric_tensor_lab"]
    projector = np.eye(3) - np.outer(nhat, nhat)

    def determinant(index_squared):
        return np.linalg.det(epsilon - index_squared * projector)

    c0 = determinant(0.0)
    p_plus = determinant(1.0)
    p_minus = determinant(-1.0)
    c1 = 0.5 * (p_plus - p_minus)
    c2 = 0.5 * (p_plus + p_minus) - c0
    scale = max(abs(c0), abs(c1), abs(c2), np.finfo(float).tiny)
    if abs(c2) <= 1e-12 * scale:
        if abs(c1) <= 1e-12 * scale:
            raise ValueError("cold-plasma dispersion polynomial is degenerate at this frequency and direction")
        roots = np.asarray([-c0 / c1], dtype=complex)
    else:
        roots = np.roots(np.asarray([c2, c1, c0], dtype=complex))
    modes = []
    for root in roots:
        matrix = epsilon - root * projector
        _, singular_values, vh = np.linalg.svd(matrix)
        vector = vh.conj().T[:, -1]
        pivot = int(np.argmax(np.abs(vector)))
        if abs(vector[pivot]) > 0.0:
            vector *= np.exp(-1j * np.angle(vector[pivot]))
        modes.append({
            "refractive_index_squared": complex(root),
            "refractive_index": complex(np.sqrt(root)),
            "electric_polarization_unit": vector,
            "dispersion_residual": float(np.linalg.norm(matrix @ vector)),
            "smallest_singular_value": float(singular_values[-1]),
        })
    return {
        **dielectric,
        "wave_normal_direction": nhat,
        "dispersion_polynomial_coefficients_high_to_low": np.asarray([c2, c1, c0]),
        "modes": modes,
        "dispersion_relation": "det(epsilon_r - n^2*(I - nhat*nhat^T)) = 0",
        "mode_solver": "quadratic_cold_plasma_magnetoionic_dispersion",
        "mode_conversion_performed": False,
    }



def cold_plasma_mode_group_velocity(
    frequency_Hz, magnetic_field_T, species: Sequence[Mapping], *,
    wave_normal_direction=(0.0, 0.0, 1.0), mode_index=0,
    relative_frequency_step=2.0e-5,
) -> dict:
    """Return the 3-D group/ray velocity of a simple lossless cold-plasma mode.

    The local dispersion matrix is
        M(omega,k) = epsilon_r(omega) - (c^2/omega^2)(k^2 I - k k^T).
    For a simple zero eigenvalue with electric eigenvector e, the
    Hellmann--Feynman derivative gives
        v_g,j = - (e^dagger (dM/dk_j) e) /
                  (e^dagger (dM/domega) e).
    The frequency derivative is evaluated by a symmetric finite difference;
    the k derivative is analytic. This returns the energy/ray walk-off
    direction for a transparent, homogeneous, cold-fluid mode. It is not a
    ray integrator and is not valid for collisional/complex roots or a
    degenerate mode crossing.
    """
    from scipy.constants import c, epsilon_0, mu_0

    frequency = _real_scalar(frequency_Hz, "frequency_Hz", positive=True)
    if (isinstance(mode_index, (bool, np.bool_))
            or not isinstance(mode_index, (int, np.integer))
            or int(mode_index) not in (0, 1)):
        raise ValueError("mode_index must be integer 0 or 1")
    step_fraction = _real_scalar(
        relative_frequency_step, "relative_frequency_step", positive=True
    )
    if step_fraction >= 1.0e-2:
        raise ValueError("relative_frequency_step must be less than 1e-2")

    parsed_collisions = []
    for index, item in enumerate(species):
        if not isinstance(item, Mapping):
            raise ValueError(f"species[{index}] must be a mapping")
        collision = _real_scalar(
            item.get("collision_frequency_rad_s", 0.0),
            f"species[{index}].collision_frequency_rad_s",
            nonnegative=True,
        )
        parsed_collisions.append(collision)
    if any(value != 0.0 for value in parsed_collisions):
        raise ValueError(
            "group velocity is restricted to collisionless transparent modes; "
            "complex collisional branches require an energy-transport treatment"
        )

    modes = cold_plasma_magnetoionic_modes(
        frequency, magnetic_field_T, species,
        wave_normal_direction=wave_normal_direction,
    )
    selected = modes["modes"][int(mode_index)]
    n2 = selected["refractive_index_squared"]
    scale_n2 = max(1.0, abs(n2.real))
    if abs(n2.imag) > 1.0e-10 * scale_n2 or n2.real <= 0.0:
        raise ValueError("selected mode must have a positive real refractive-index squared")
    n = float(np.sqrt(n2.real))
    omega = 2.0 * np.pi * frequency
    nhat = np.asarray(modes["wave_normal_direction"], dtype=float)
    wavevector = (n * omega / c) * nhat
    epsilon = np.asarray(modes["dielectric_tensor_lab"], dtype=complex)
    polarization = np.asarray(selected["electric_polarization_unit"], dtype=complex)
    polarization /= np.linalg.norm(polarization)

    identity = np.eye(3, dtype=complex)

    def dispersion_matrix(omega_value, k_value, epsilon_value):
        return epsilon_value - (c**2 / omega_value**2) * (
            np.dot(k_value, k_value) * identity - np.outer(k_value, k_value)
        )

    matrix = dispersion_matrix(omega, wavevector, epsilon)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    spectral_scale = max(float(singular_values[0]), 1.0)
    if singular_values[-1] > 1.0e-8 * spectral_scale:
        raise ArithmeticError(
            "selected refractive root does not satisfy the Maxwell dispersion matrix"
        )
    if singular_values[-2] <= 1.0e-8 * spectral_scale:
        raise ValueError(
            "group velocity is undefined at a degenerate/near-degenerate mode crossing"
        )

    delta_omega = max(omega * step_fraction, 1.0)
    frequency_plus = (omega + delta_omega) / (2.0 * np.pi)
    frequency_minus = (omega - delta_omega) / (2.0 * np.pi)
    eps_plus = cold_plasma_dielectric_tensor(
        frequency_plus, magnetic_field_T, species
    )["dielectric_tensor_lab"]
    eps_minus = cold_plasma_dielectric_tensor(
        frequency_minus, magnetic_field_T, species
    )["dielectric_tensor_lab"]
    matrix_plus = dispersion_matrix(omega + delta_omega, wavevector, eps_plus)
    matrix_minus = dispersion_matrix(omega - delta_omega, wavevector, eps_minus)
    derivative_omega = (matrix_plus - matrix_minus) / (2.0 * delta_omega)
    derivative_epsilon_omega = (eps_plus - eps_minus) / (2.0 * delta_omega)
    d_omega_epsilon = epsilon + omega * derivative_epsilon_omega
    denominator = np.vdot(polarization, derivative_omega @ polarization)
    denominator_scale = max(
        float(np.linalg.norm(derivative_omega, ord=2)), np.finfo(float).tiny
    )
    if abs(denominator) <= 1.0e-12 * denominator_scale:
        raise ValueError("dispersion frequency derivative is singular for the selected mode")

    derivative_k = np.empty(3, dtype=float)
    for axis in range(3):
        unit = np.eye(3)[axis]
        d_matrix = -(c**2 / omega**2) * (
            2.0 * wavevector[axis] * identity
            - np.outer(unit, wavevector)
            - np.outer(wavevector, unit)
        )
        derivative_k[axis] = float(
            np.real(np.vdot(polarization, d_matrix @ polarization))
        )
    group_velocity = -derivative_k / denominator
    if np.max(np.abs(np.imag(group_velocity))) > 1.0e-8 * max(
        1.0, float(np.max(np.abs(np.real(group_velocity))))
    ):
        raise ArithmeticError("lossless group-velocity derivative unexpectedly became complex")
    group_velocity = np.real(group_velocity)
    magnitude = float(np.linalg.norm(group_velocity))
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise ArithmeticError("computed group velocity is not finite and positive")
    ray_direction = group_velocity / magnitude
    wave_normal = nhat.copy()
    walkoff = float(np.arccos(np.clip(np.dot(ray_direction, wave_normal), -1.0, 1.0)))

    # Brillouin energy density for a transparent, temporally dispersive,
    # spatially local dielectric: W=1/4[eps0 E*.d(omega eps)/domega.E
    # + mu0 |H|^2]. The plane-wave Poynting vector is S=1/2 Re(E x H*).
    magnetic_field_amplitude = np.cross(wavevector, polarization) / (mu_0 * omega)
    electric_energy_term = float(np.real(
        np.vdot(polarization, d_omega_epsilon @ polarization)
    ))
    magnetic_energy_term = float(np.real(
        np.vdot(magnetic_field_amplitude, magnetic_field_amplitude)
    ))
    energy_density = 0.25 * (
        epsilon_0 * electric_energy_term + mu_0 * magnetic_energy_term
    )
    poynting_vector = 0.5 * np.real(
        np.cross(polarization, np.conj(magnetic_field_amplitude))
    )
    if not np.isfinite(energy_density) or energy_density <= 0.0:
        raise ArithmeticError(
            "Brillouin wave energy density is non-positive for the selected mode"
        )
    energy_velocity = poynting_vector / energy_density
    energy_velocity_difference = float(
        np.linalg.norm(energy_velocity - group_velocity)
        / max(magnitude, np.finfo(float).tiny)
    )
    return {
        "frequency_Hz": frequency,
        "mode_index": int(mode_index),
        "refractive_index": n,
        "wavevector_rad_m": wavevector,
        "wave_normal_direction": wave_normal,
        "electric_polarization_unit": polarization,
        "group_velocity_m_s": group_velocity,
        "group_speed_m_s": magnitude,
        "ray_direction": ray_direction,
        "brillouin_energy_density_J_m3_per_V2_m2": float(energy_density),
        "poynting_flux_vector_W_m2_per_V2_m2": poynting_vector,
        "energy_velocity_m_s": energy_velocity,
        "relative_group_energy_velocity_difference": energy_velocity_difference,
        "wave_normal_to_ray_walkoff_angle_rad": walkoff,
        "dispersion_relation": "det(epsilon_r - (c^2/omega^2)*(k^2*I-k*k^T)) = 0",
        "group_velocity_relation": (
            "v_g,j = -(e^dagger*dM/dk_j*e)/(e^dagger*dM/domega*e)"
        ),
        "wave_energy_relation": (
            "W=1/4*(epsilon_0*E^dagger*d(omega*epsilon_r)/domega*E"
            "+mu_0*H^dagger*H); S=1/2*real(E cross H*)"
        ),
        "energy_velocity_relation": "v_energy = S/W; equals group velocity for a transparent simple local mode",
        "frequency_derivative_method": "symmetric_finite_difference",
        "wavevector_derivative_method": "analytic_Hellmann_Feynman",
        "mode_is_simple": True,
        "model_scope": "local_homogeneous_lossless_cold_multispecies_magnetoionic_mode",
        "limitations": [
            "No spatial ray integration or inhomogeneous equilibrium gradients.",
            "No collisions, absorption, warm kinetic response or relativistic corrections.",
            "No branch tracking or mode conversion through degeneracies.",
            "The result is a local group velocity, not a full quasi-optical beam solution.",
        ],
    }
