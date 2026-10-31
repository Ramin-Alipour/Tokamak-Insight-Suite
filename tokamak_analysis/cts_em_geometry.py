"""Cold-plasma mode propagation and finite Gaussian-beam overlap for CTS.

This module supplies the high-frequency incident/receiver-wave part of a CTS
calculation.  It is deliberately separate from the hot kinetic fluctuation
response: cold-plasma modes are used only for the high-frequency probe and
scattered waves, while the fluctuation dielectric response remains kinetic.

Ray tracing uses a Hamiltonian eigenvalue of the cold-plasma Maxwell wave
operator.  The caller supplies an equilibrium callback returning B and all
species densities at each position.  No equilibrium profile is guessed.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import numpy as np
from scipy.integrate import solve_ivp

EPS0 = 8.8541878128e-12
C = 299792458.0
MU0 = 1.0 / (EPS0 * C * C)


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def cold_plasma_dielectric_tensor(omega_rad_s, magnetic_field_T,
                                  species: Sequence[Mapping]):
    """Stix cold-plasma dielectric tensor in laboratory coordinates.

    Each species mapping must contain density_m3, charge_C and mass_kg.
    Cyclotron frequencies retain the sign of the species charge.  The
    exp(-i omega t) convention is used.  Collisions are intentionally absent.
    """
    omega = _positive(omega_rad_s, "omega_rad_s")
    b = _vector(magnetic_field_T, "magnetic_field_T")
    bmag = float(np.linalg.norm(b))
    if bmag == 0:
        raise ValueError("magnetic_field_T must be non-zero")
    bhat = b / bmag
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(trial, bhat))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    e1 = trial - np.dot(trial, bhat) * bhat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(bhat, e1)
    rotation = np.column_stack((e1, e2, bhat))

    s_value, d_value, p_value = 1.0, 0.0, 1.0
    for index, item in enumerate(species):
        try:
            density = _positive(item["density_m3"], f"species[{index}].density_m3")
            mass = _positive(item["mass_kg"], f"species[{index}].mass_kg")
            charge = float(item["charge_C"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each species requires finite charge_C, positive mass_kg and density_m3") from exc
        if not np.isfinite(charge) or charge == 0:
            raise ValueError(f"species[{index}].charge_C must be finite and non-zero")
        wp2 = density * charge * charge / (EPS0 * mass)
        wc = charge * bmag / mass
        denominator = omega * omega - wc * wc
        if abs(denominator) <= 1e-12 * max(omega * omega, wc * wc, 1.0):
            raise ValueError("cold-plasma dielectric is singular at a cyclotron resonance")
        s_value -= wp2 / denominator
        d_value += (wc / omega) * wp2 / denominator
        p_value -= wp2 / (omega * omega)

    epsilon_b = np.array([
        [s_value, -1j * d_value, 0.0],
        [1j * d_value, s_value, 0.0],
        [0.0, 0.0, p_value],
    ], dtype=complex)
    return rotation @ epsilon_b @ rotation.T


def cold_plasma_wave_operator(omega_rad_s, wavevector_rad_m,
                              magnetic_field_T, species):
    """Return epsilon + N^2 (khat khat - I), the cold Maxwell wave tensor."""
    omega = _positive(omega_rad_s, "omega_rad_s")
    k = _vector(wavevector_rad_m, "wavevector_rad_m")
    kmag = float(np.linalg.norm(k))
    if kmag == 0:
        raise ValueError("wavevector_rad_m must be non-zero")
    epsilon = cold_plasma_dielectric_tensor(omega, magnetic_field_T, species)
    khat = k / kmag
    n2 = (C * kmag / omega) ** 2
    operator = epsilon + n2 * (np.outer(khat, khat) - np.eye(3))
    return operator, epsilon


def solve_cold_plasma_modes(omega_rad_s, wavevector_rad_m,
                            magnetic_field_T, species):
    """Return the three real eigenbranches and transverse electric eigenvectors."""
    operator, epsilon = cold_plasma_wave_operator(
        omega_rad_s, wavevector_rad_m, magnetic_field_T, species
    )
    hermitian = 0.5 * (operator + operator.conj().T)
    values, vectors = np.linalg.eigh(hermitian)
    return {
        "dispersion_eigenvalues": values,
        "electric_polarizations": vectors,
        "wave_operator": operator,
        "dielectric_tensor": epsilon,
        "mode_residuals": np.linalg.norm(operator @ vectors, axis=0),
    }


def gaussian_beam_overlap(center_incident_m, covariance_incident_m2,
                          center_scattered_m, covariance_scattered_m2):
    """Analytic integral of two normalized 3-D Gaussian beam intensities.

    The intensity distributions are normalized to unit volume integral.
    Covariance matrices describe the three-dimensional intensity ellipsoids.
    The returned overlap has units m^-3, as required by integral(I_i I_s dV).
    """
    ci = _vector(center_incident_m, "center_incident_m")
    cs = _vector(center_scattered_m, "center_scattered_m")
    si = np.asarray(covariance_incident_m2, dtype=float)
    ss = np.asarray(covariance_scattered_m2, dtype=float)
    if si.shape != (3, 3) or ss.shape != (3, 3):
        raise ValueError("beam covariance matrices must be 3x3")
    if not np.all(np.isfinite(si)) or not np.all(np.isfinite(ss)):
        raise ValueError("beam covariance matrices must be finite")
    si = 0.5 * (si + si.T)
    ss = 0.5 * (ss + ss.T)
    if np.min(np.linalg.eigvalsh(si)) <= 0 or np.min(np.linalg.eigvalsh(ss)) <= 0:
        raise ValueError("beam covariance matrices must be positive definite")
    total = si + ss
    delta = ci - cs
    sign, logdet = np.linalg.slogdet(total)
    if sign <= 0:
        raise ValueError("combined beam covariance must be positive definite")
    exponent = -0.5 * float(delta @ np.linalg.solve(total, delta))
    log_overlap = -1.5 * np.log(2.0 * np.pi) - 0.5 * logdet + exponent
    return float(np.exp(log_overlap))


def trace_cold_plasma_ray(
    initial_position_m, initial_wavevector_rad_m, omega_rad_s, *,
    equilibrium: Callable[[np.ndarray], Mapping],
    mode_index=None, path_parameter_max=1.0, sample_count=256,
    relative_tolerance=1e-8, absolute_tolerance=1e-10,
):
    """Trace a cold-plasma eigenmode ray through a supplied 3-D equilibrium.

    equilibrium(x) must return {'magnetic_field_T': vector, 'species': [...]}
    with the species format documented by cold_plasma_dielectric_tensor.
    The ray parameter is normalized so dx/ds = -d(lambda)/dk and
    dk/ds = d(lambda)/dx, where lambda is the selected Hermitian wave-operator
    eigenvalue.  This parameterization traces the Hamiltonian characteristic;
    it is not a physical time or path-length coordinate.

    Mode index is the eigenvalue order returned by numpy.linalg.eigh (0..2).
    Near branch crossings, the caller must verify eigenvector continuity.
    """
    x0 = _vector(initial_position_m, "initial_position_m")
    k0 = _vector(initial_wavevector_rad_m, "initial_wavevector_rad_m")
    omega = _positive(omega_rad_s, "omega_rad_s")
    if mode_index is None or isinstance(mode_index, bool) or int(mode_index) not in (0, 1, 2):
        raise ValueError("mode_index must be explicitly selected as 0, 1 or 2; no propagation branch is assumed")
    if int(sample_count) < 2:
        raise ValueError("sample_count must be >= 2")
    smax = _positive(path_parameter_max, "path_parameter_max")
    rtol = _positive(relative_tolerance, "relative_tolerance")
    atol = _positive(absolute_tolerance, "absolute_tolerance")

    def eigenvalue(position, wavevector):
        state = equilibrium(np.asarray(position, dtype=float))
        if not isinstance(state, Mapping) or "magnetic_field_T" not in state or "species" not in state:
            raise ValueError("equilibrium callback must return magnetic_field_T and species")
        result = solve_cold_plasma_modes(
            omega, wavevector, state["magnetic_field_T"], state["species"]
        )
        return float(result["dispersion_eigenvalues"][int(mode_index)])

    def rhs(parameter, state):
        position, wavevector = state[:3], state[3:]
        grad_x = np.empty(3)
        grad_k = np.empty(3)
        for axis in range(3):
            hx = np.sqrt(np.finfo(float).eps) * max(abs(position[axis]), 1.0)
            hk = np.sqrt(np.finfo(float).eps) * max(abs(wavevector[axis]), 1.0)
            dx = np.zeros(3); dx[axis] = hx
            dk = np.zeros(3); dk[axis] = hk
            grad_x[axis] = (eigenvalue(position + dx, wavevector)
                            - eigenvalue(position - dx, wavevector)) / (2.0 * hx)
            grad_k[axis] = (eigenvalue(position, wavevector + dk)
                            - eigenvalue(position, wavevector - dk)) / (2.0 * hk)
        return np.concatenate((-grad_k, grad_x))

    sample = np.linspace(0.0, smax, int(sample_count))
    solution = solve_ivp(
        rhs, (0.0, smax), np.concatenate((x0, k0)), t_eval=sample,
        rtol=rtol, atol=atol, method="DOP853"
    )
    if not solution.success:
        raise RuntimeError(f"cold-plasma ray tracing failed: {solution.message}")
    positions = solution.y[:3].T
    wavevectors = solution.y[3:].T
    eigenvalues = np.empty(solution.t.size)
    polarizations = np.empty((solution.t.size, 3), dtype=complex)
    for index, (position, wavevector) in enumerate(zip(positions, wavevectors)):
        state = equilibrium(position)
        modes = solve_cold_plasma_modes(
            omega, wavevector, state["magnetic_field_T"], state["species"]
        )
        eigenvalues[index] = modes["dispersion_eigenvalues"][int(mode_index)]
        polarizations[index] = modes["electric_polarizations"][:, int(mode_index)]
    return {
        "ray_parameter": solution.t,
        "position_m": positions,
        "wavevector_rad_m": wavevectors,
        "dispersion_eigenvalue": eigenvalues,
        "electric_polarization": polarizations,
        "mode_index": int(mode_index),
        "parameterization": "Hamiltonian eigenvalue parameter; not physical time",
        "limitations": [
            "Cold, collisionless, local dielectric; not valid at unresolved resonant layers.",
            "No mode conversion or branch tracking through degeneracies.",
            "Geometrical-optics approximation; diffraction and finite-beam evolution are not included.",
            "The equilibrium callback and its spatial derivatives are supplied by the caller."
        ],
    }



def trace_cts_frequency_resolved_beams(
    frequency_shift_Hz, probe_frequency_Hz, *,
    incident_ray, receiver_rays, equilibrium,
    incident_beam_covariance_m2, receiver_beam_covariances_m2,
    mode_index=None, path_parameter_max=1.0, sample_count=256,
    scattering_sample_count=16,
    relative_tolerance=1e-8, absolute_tolerance=1e-10,
):
    """Trace probe/receiver rays and integrate Gaussian overlap along the scattering path.

    incident_ray and each receiver_rays item provide initial_position_m and
    initial_wavevector_rad_m. The equilibrium callback supplies magnetic field
    and species densities at every ray point. Beam covariance matrices may be
    constant 3x3 matrices or sampled local covariance arrays along each ray.

    For each scattered frequency, nearest receiver-ray points are paired with
    selected incident-ray samples. The local Gaussian overlap is multiplied by
    the incident-path quadrature length, normalized to a unit-sum scattering
    volume weight, and returned as scattering_samples. The full-EM CTS solver
    evaluates each local sample and sums its spectrum with these weights.

    This is cold-plasma geometrical optics with caller-supplied local Gaussian
    covariance. It does not propagate diffraction or mode conversion, and
    assumes uniform fitted ne/Te/Ti across the sampled volume while allowing
    local ray directions and magnetic field to vary.
    """
    shifts = np.asarray(frequency_shift_Hz, dtype=float)
    if (shifts.ndim != 1 or shifts.size < 1 or np.any(~np.isfinite(shifts))
            or np.any(np.diff(shifts) <= 0)):
        raise ValueError("frequency_shift_Hz must be a finite strictly increasing vector")
    f0 = _positive(probe_frequency_Hz, "probe_frequency_Hz")
    if not isinstance(incident_ray, Mapping):
        raise ValueError("incident_ray must provide initial position and wavevector")
    if not isinstance(receiver_rays, Sequence) or isinstance(receiver_rays, (str, bytes)) or len(receiver_rays) != shifts.size:
        raise ValueError("receiver_rays must contain one ray specification per scattered frequency")
    if isinstance(scattering_sample_count, bool) or int(scattering_sample_count) < 2:
        raise ValueError("scattering_sample_count must be an integer >=2")
    incident = trace_cold_plasma_ray(
        incident_ray["initial_position_m"], incident_ray["initial_wavevector_rad_m"],
        2.0*np.pi*f0, equilibrium=equilibrium, mode_index=mode_index,
        path_parameter_max=incident_ray.get("path_parameter_max", path_parameter_max),
        sample_count=incident_ray.get("sample_count", sample_count),
        relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance,
    )
    incident_positions = np.asarray(incident["position_m"], dtype=float)
    incident_cov = np.asarray(incident_beam_covariance_m2, dtype=float)
    if incident_cov.shape == (3, 3):
        incident_cov = np.broadcast_to(incident_cov, (incident_positions.shape[0], 3, 3))
    if incident_cov.shape != (incident_positions.shape[0], 3, 3):
        raise ValueError("incident_beam_covariance_m2 must be (3,3) or (n_incident_ray,3,3)")
    sample_indices = np.unique(np.linspace(
        0, incident_positions.shape[0]-1,
        min(int(scattering_sample_count), incident_positions.shape[0])
    ).astype(int))
    selected_positions = incident_positions[sample_indices]
    segment_lengths = np.linalg.norm(np.diff(selected_positions, axis=0), axis=1)
    path_weights = np.empty(sample_indices.size, dtype=float)
    path_weights[0] = 0.5*segment_lengths[0]
    path_weights[-1] = 0.5*segment_lengths[-1]
    if sample_indices.size > 2:
        path_weights[1:-1] = 0.5*(segment_lengths[:-1]+segment_lengths[1:])
    if np.any(path_weights <= 0) or np.any(~np.isfinite(path_weights)):
        raise ValueError("incident ray must span a non-zero physical path for scattering-volume integration")

    entries = []
    path_integrals = []
    for index, (shift, ray_spec) in enumerate(zip(shifts, receiver_rays)):
        if not isinstance(ray_spec, Mapping):
            raise ValueError(f"receiver_rays[{index}] must be a mapping")
        receiver = trace_cold_plasma_ray(
            ray_spec["initial_position_m"], ray_spec["initial_wavevector_rad_m"],
            2.0*np.pi*(f0+float(shift)), equilibrium=equilibrium, mode_index=mode_index,
            path_parameter_max=ray_spec.get("path_parameter_max", path_parameter_max),
            sample_count=ray_spec.get("sample_count", sample_count),
            relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance,
        )
        receiver_positions = np.asarray(receiver["position_m"], dtype=float)
        receiver_cov = np.asarray(receiver_beam_covariances_m2, dtype=float)
        if receiver_cov.shape == (3, 3):
            receiver_cov_for_frequency = np.broadcast_to(receiver_cov, (receiver_positions.shape[0], 3, 3))
        elif receiver_cov.shape == (shifts.size, 3, 3):
            receiver_cov_for_frequency = np.broadcast_to(receiver_cov[index], (receiver_positions.shape[0], 3, 3))
        elif receiver_cov.shape == (shifts.size, receiver_positions.shape[0], 3, 3):
            receiver_cov_for_frequency = receiver_cov[index]
        else:
            raise ValueError("receiver_beam_covariances_m2 must be (3,3), (n_frequency,3,3), or (n_frequency,n_receiver_ray,3,3)")
        delta = selected_positions[:, None, :] - receiver_positions[None, :, :]
        distance2 = np.einsum("ijk,ijk->ij", delta, delta)
        nearest_receiver = np.argmin(distance2, axis=1)
        samples = []
        raw_weights = []
        for local_index, incident_index in enumerate(sample_indices):
            receiver_index = int(nearest_receiver[local_index])
            xi = incident_positions[incident_index]
            xs = receiver_positions[receiver_index]
            location = 0.5*(xi+xs)
            local = equilibrium(location)
            if not isinstance(local, Mapping) or "magnetic_field_T" not in local or "species" not in local:
                raise ValueError("equilibrium callback must return magnetic_field_T and species at every scattering location")
            b_local = _vector(local["magnetic_field_T"], "local magnetic_field_T")
            ki = np.asarray(incident["wavevector_rad_m"][incident_index], dtype=float)
            ks = np.asarray(receiver["wavevector_rad_m"][receiver_index], dtype=float)
            overlap = gaussian_beam_overlap(
                xi, incident_cov[incident_index], xs,
                receiver_cov_for_frequency[receiver_index],
            )
            raw_weight = overlap*path_weights[local_index]
            raw_weights.append(raw_weight)
            samples.append({
                "incident_direction": ki/np.linalg.norm(ki),
                "scattered_direction": ks/np.linalg.norm(ks),
                "magnetic_field_T": b_local,
                "scattering_location_m": location,
                "beam_overlap_m_minus3": float(overlap),
                "path_length_weight_m": float(path_weights[local_index]),
                "ray_separation_m": float(np.sqrt(distance2[local_index, receiver_index])),
                "incident_ray_index": int(incident_index),
                "receiver_ray_index": receiver_index,
                "sample_weight": float(raw_weight),
            })
        raw_weights = np.asarray(raw_weights, dtype=float)
        path_integral = float(np.sum(raw_weights))
        if not np.isfinite(path_integral) or path_integral <= 0:
            raise ValueError(f"Gaussian beam overlap has zero scattering-volume integral at frequency bin {index}")
        path_integrals.append(path_integral)
        for sample in samples:
            sample["sample_weight"] /= path_integral
            sample["beam_overlap_relative"] = 1.0
        mean_b = np.sum(
            np.asarray([sample["magnetic_field_T"] for sample in samples])
            *np.asarray([sample["sample_weight"] for sample in samples])[:,None],
            axis=0,
        )
        mean_ki = np.sum(
            np.asarray([sample["incident_direction"] for sample in samples])
            *np.asarray([sample["sample_weight"] for sample in samples])[:,None],
            axis=0,
        )
        mean_ks = np.sum(
            np.asarray([sample["scattered_direction"] for sample in samples])
            *np.asarray([sample["sample_weight"] for sample in samples])[:,None],
            axis=0,
        )
        mean_location = np.sum(
            np.asarray([sample["scattering_location_m"] for sample in samples])
            *np.asarray([sample["sample_weight"] for sample in samples])[:,None],
            axis=0,
        )
        entries.append({
            "incident_direction": mean_ki/np.linalg.norm(mean_ki),
            "scattered_direction": mean_ks/np.linalg.norm(mean_ks),
            "magnetic_field_T": mean_b,
            "scattering_location_m": mean_location,
            "beam_overlap_path_integral_m_minus2": path_integral,
            "scattering_samples": samples,
        })
    scale = max(path_integrals)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Gaussian beam overlap is non-positive for all frequency bins")
    for entry, integral in zip(entries, path_integrals):
        entry["beam_overlap_relative"] = float(integral/scale)
    return {
        "frequency_shift_Hz": shifts.copy(),
        "mode_index": int(mode_index),
        "frequency_resolved_geometry": entries,
        "relative_beam_overlap": np.asarray(path_integrals, dtype=float)/scale,
        "absolute_beam_overlap_path_integral_m_minus2": np.asarray(path_integrals, dtype=float),
        "incident_ray": incident,
        "geometry_model": "cold-plasma Hamiltonian ray tracing with path-integrated local Gaussian intensity overlap",
        "scattering_volume_integration": "normalized Gaussian overlap times incident-ray path-length quadrature",
        "scattering_sample_count": np.asarray([len(entry["scattering_samples"]) for entry in entries], dtype=int),
        "limitations": [
            "Cold-plasma geometrical optics; diffraction, quasi-optical beam evolution and mode conversion are not included.",
            "Beam covariance matrices are supplied by the caller along the rays; covariance propagation is not inferred.",
            "Fitted ne, Te and thermal-ion temperatures are treated as uniform across the sampled scattering volume; local ray directions and magnetic field vary.",
            "Ray branch continuity near eigenvalue crossings must be checked by the caller.",
            "The equilibrium callback supplies all magnetic and species profiles; no equilibrium is fabricated."
        ],
    }



def solve_cold_plasma_propagating_modes(
    omega_rad_s, propagation_direction, magnetic_field_T, species, *,
    mode_index=None, discriminant_tolerance=1e-10,
):
    """Solve both cold-plasma refractive-index branches and flux-normalize one.

    The Appleton-Hartree biquadratic is solved for N^2 at the specified
    propagation direction.  The selected branch is converted to a laboratory
    wavevector; its electric eigenvector is obtained from the full Maxwell
    wave operator and normalized to unit magnitude of longitudinal Poynting
    flux.  This is a cold, collisionless, local plane-wave normalization.
    """
    omega = _positive(omega_rad_s, "omega_rad_s")
    direction = _vector(propagation_direction, "propagation_direction")
    direction = direction / np.linalg.norm(direction)
    b = _vector(magnetic_field_T, "magnetic_field_T")
    bmag = float(np.linalg.norm(b))
    if bmag == 0:
        raise ValueError("magnetic_field_T must be non-zero")
    if mode_index is None or isinstance(mode_index, bool) or int(mode_index) not in (0, 1):
        raise ValueError("mode_index must be explicitly selected as one of the two propagating branches (0 or 1)")
    epsilon = cold_plasma_dielectric_tensor(omega, b, species)
    # Recover Stix S,D,P in the B-aligned frame.
    bhat = b / bmag
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(trial, bhat))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    e1 = trial - np.dot(trial, bhat) * bhat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(bhat, e1)
    rotation = np.column_stack((e1, e2, bhat))
    eps_stix = rotation.T @ epsilon @ rotation
    sval = float(np.real(0.5 * (eps_stix[0, 0] + eps_stix[1, 1])))
    dval = float(np.real(-0.5j * (eps_stix[1, 0] - eps_stix[0, 1])))
    pval = float(np.real(eps_stix[2, 2]))
    rvalue, lvalue = sval + dval, sval - dval
    cos_theta = float(np.clip(np.dot(direction, bhat), -1.0, 1.0))
    sin2 = max(0.0, 1.0 - cos_theta * cos_theta)
    cos2 = cos_theta * cos_theta
    aa = sval * sin2 + pval * cos2
    bb = rvalue * lvalue * sin2 + pval * sval * (1.0 + cos2)
    cc = pval * rvalue * lvalue
    discriminant = bb * bb - 4.0 * aa * cc
    scale = max(bb * bb, abs(4.0 * aa * cc), 1.0)
    if discriminant < -abs(discriminant_tolerance) * scale:
        raise ValueError("cold-plasma dispersion has no real refractive-index branches for this direction")
    discriminant = max(discriminant, 0.0)
    if abs(aa) <= np.finfo(float).eps * max(abs(bb), abs(cc), 1.0):
        if abs(bb) <= np.finfo(float).eps:
            raise ValueError("degenerate cold-plasma Appleton-Hartree dispersion")
        roots = np.array([cc / bb, np.nan])
    else:
        root = np.sqrt(discriminant)
        roots = np.array([(bb - root) / (2.0 * aa), (bb + root) / (2.0 * aa)])
    roots = np.sort(roots[np.isfinite(roots)])
    if roots.size == 0 or np.any(roots <= 0):
        raise ValueError("selected cold-plasma branch is evanescent (N^2 is not positive)")
    if int(mode_index) >= roots.size:
        raise ValueError("requested propagating mode branch does not exist")
    n_squared = float(roots[int(mode_index)])
    refractive_index = float(np.sqrt(n_squared))
    wavevector = (omega / C) * refractive_index * direction
    operator, _ = cold_plasma_wave_operator(omega, wavevector, b, species)
    _, singular_values, vh = np.linalg.svd(operator)
    electric = vh[-1].conj()
    electric /= np.linalg.norm(electric)
    magnetic = np.cross(wavevector, electric) / (MU0 * omega)
    poynting = 0.5 * np.real(np.cross(electric, magnetic.conj()))
    longitudinal_flux = float(np.dot(poynting, direction))
    if not np.isfinite(longitudinal_flux) or abs(longitudinal_flux) <= np.finfo(float).tiny:
        raise ValueError("selected mode has zero or non-finite longitudinal Poynting flux")
    electric /= np.sqrt(abs(longitudinal_flux))
    magnetic /= np.sqrt(abs(longitudinal_flux))
    poynting /= abs(longitudinal_flux)
    return {
        "refractive_index": refractive_index,
        "refractive_index_squared": n_squared,
        "wavevector_rad_m": wavevector,
        "electric_polarization_unit_flux": electric,
        "magnetic_field_per_sqrt_flux": magnetic,
        "poynting_vector_unit_flux": poynting,
        "signed_longitudinal_flux_before_normalization": longitudinal_flux,
        "wave_operator_singular_values": singular_values,
        "mode_index": int(mode_index),
        "normalization": "absolute longitudinal time-averaged Poynting flux equals 1 W/m^2",
        "assumptions": [
            "Cold, collisionless, homogeneous local plasma.",
            "Plane-wave geometrical optics; no finite-beam or diffraction correction.",
            "The selected branch must be real and propagating; resonant and evanescent branches are rejected."
        ],
    }
