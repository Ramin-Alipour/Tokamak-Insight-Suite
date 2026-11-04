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
    The signed gyrofrequency is Omega_s=q_s |B|/m_s. With B along +z:
      R=1-sum(omega_ps^2/[omega(omega+Omega_s)]),
      L=1-sum(omega_ps^2/[omega(omega-Omega_s)]),
      S=(R+L)/2, D=(R-L)/2, P=1-sum(omega_ps^2/omega^2).
    The tensor in a right-handed field-aligned basis is
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
