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
from scipy.linalg import expm

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




def _transverse_frame(direction):
    d = _vector(direction, "beam_direction")
    d = d / np.linalg.norm(d)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(trial, d))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    e1 = trial - np.dot(trial, d) * d
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return np.column_stack((e1, e2))


def gaussian_beam_overlap_transverse(
    center_incident_m, direction_incident, covariance_incident_m2,
    center_scattered_m, direction_scattered, covariance_scattered_m2,
):
    """Overlap two normalized transverse Gaussian intensity profiles.

    Each covariance is the propagated 2x2 position block in its own ray frame.
    Both are projected into the plane normal to the mean ray direction before
    evaluating the analytic 2-D Gaussian product integral.
    """
    ci = _vector(center_incident_m, "center_incident_m")
    cs = _vector(center_scattered_m, "center_scattered_m")
    ei = _transverse_frame(direction_incident)
    es = _transverse_frame(direction_scattered)
    di = _vector(direction_incident, "direction_incident")
    ds = _vector(direction_scattered, "direction_scattered")
    common_direction = di / np.linalg.norm(di) + ds / np.linalg.norm(ds)
    if np.linalg.norm(common_direction) <= 1e-10:
        raise ValueError("incident and scattered directions are antiparallel; common transverse overlap plane is undefined")
    common = _transverse_frame(common_direction)
    si = np.asarray(covariance_incident_m2, dtype=float)
    ss = np.asarray(covariance_scattered_m2, dtype=float)
    if si.shape != (2, 2) or ss.shape != (2, 2):
        raise ValueError("transverse beam covariance matrices must be 2x2")
    for matrix, name in ((si, "incident"), (ss, "scattered")):
        if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-13):
            raise ValueError(f"{name} transverse covariance must be finite and symmetric")
        if np.min(np.linalg.eigvalsh(matrix)) <= 0:
            raise ValueError(f"{name} transverse covariance must be positive definite")
    ai = common.T @ ei
    ass = common.T @ es
    si_common = ai @ si @ ai.T
    ss_common = ass @ ss @ ass.T
    total = si_common + ss_common
    delta = common.T @ (ci - cs)
    sign, logdet = np.linalg.slogdet(total)
    if sign <= 0:
        raise ValueError("projected transverse covariance sum must be positive definite")
    exponent = -0.5 * float(delta @ np.linalg.solve(total, delta))
    return float(np.exp(-np.log(2.0 * np.pi) - 0.5 * logdet + exponent))


def trace_cold_plasma_ray(
    initial_position_m, initial_wavevector_rad_m, omega_rad_s, *,
    equilibrium: Callable[[np.ndarray], Mapping],
    mode_index=None, path_parameter_max=1.0, sample_count=256,
    relative_tolerance=1e-8, absolute_tolerance=1e-10,
    initial_mode_amplitudes=None, propagation_constants_rad_m=None,
    initial_beam_phase_space_covariance=None, paraxial_transfer_matrices=None,
    initial_ray_phase_space_covariance=None,
):
    """Trace a cold-plasma eigenmode ray through a supplied 3-D equilibrium.

    equilibrium(x) must return {'magnetic_field_T': vector, 'species': [...]}
    with the species format documented by cold_plasma_dielectric_tensor.
    The ray parameter is normalized so dx/ds = -d(lambda)/dk and
    dk/ds = d(lambda)/dx, where lambda is the selected Hermitian wave-operator
    eigenvalue.  This parameterization traces the Hamiltonian characteristic;
    it is not a physical time or path-length coordinate.

    Mode index is the eigenvalue order returned by numpy.linalg.eigh (0..2).
    Near branch crossings, the caller must inspect the returned branch-continuity
    diagnostics. Optional coherent mode-amplitude propagation is computed only
    when both initial_mode_amplitudes and propagation_constants_rad_m are
    explicitly supplied; it does not change the Hamiltonian ray trajectory.
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
    if initial_ray_phase_space_covariance is not None and initial_beam_phase_space_covariance is not None:
        raise ValueError("choose either a 6-D ray phase-space covariance or a 4-D paraxial covariance with supplied ABCD maps")
    if (initial_beam_phase_space_covariance is None) != (paraxial_transfer_matrices is None):
        raise ValueError("initial_beam_phase_space_covariance and paraxial_transfer_matrices must be supplied together")
    propagated_beam = None
    if initial_beam_phase_space_covariance is not None:
        from .advanced_inference import propagate_paraxial_beam_covariance
        propagated_beam = propagate_paraxial_beam_covariance(
            initial_beam_phase_space_covariance, paraxial_transfer_matrices
        )
        if np.asarray(paraxial_transfer_matrices).shape[0] != int(sample_count) - 1:
            raise ValueError("paraxial_transfer_matrices must contain one symplectic map per adjacent ray-sample pair")

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
    wave_operators = []
    for position, wavevector in zip(positions, wavevectors):
        state = equilibrium(position)
        operator, _ = cold_plasma_wave_operator(
            omega, wavevector, state["magnetic_field_T"], state["species"]
        )
        wave_operators.append(operator)
    wave_operators = np.asarray(wave_operators)
    branch_tracking = track_cold_plasma_modes_from_operators(wave_operators)
    if (initial_mode_amplitudes is None) != (propagation_constants_rad_m is None):
        raise ValueError(
            "initial_mode_amplitudes and propagation_constants_rad_m must be supplied together "
            "to enable coherent mode-conversion propagation"
        )
    mode_conversion = None
    if initial_mode_amplitudes is not None:
        physical_path = np.concatenate((
            [0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        ))
        if np.any(np.diff(physical_path) <= 0):
            raise ValueError("ray path must have strictly positive physical arc-length increments for mode conversion")
        mode_conversion = propagate_cold_plasma_mode_coupling(
            wave_operators, physical_path, propagation_constants_rad_m,
            initial_mode_amplitudes,
        )
    selected_branch_order = branch_tracking["eigenvalue_order_permutation"][:, int(mode_index)]
    fixed_order_branch_consistent = bool(np.all(selected_branch_order == int(mode_index)))
    propagated_ray_covariance = None
    propagated_ray_transverse_covariance = None
    if initial_ray_phase_space_covariance is not None:
        if not fixed_order_branch_consistent:
            raise ValueError("ray phase-space covariance propagation is undefined across a cold-plasma eigenmode branch-order crossing")
        sigma0 = np.asarray(initial_ray_phase_space_covariance, dtype=float)
        if sigma0.shape != (6, 6) or not np.all(np.isfinite(sigma0)) or not np.allclose(sigma0, sigma0.T, rtol=1e-10, atol=1e-13):
            raise ValueError("initial_ray_phase_space_covariance must be a finite symmetric 6x6 matrix for [x,y,z,kx,ky,kz]")
        sigma_eigenvalues = np.linalg.eigvalsh(sigma0)
        sigma_scale = max(float(np.max(np.abs(sigma_eigenvalues))), np.finfo(float).tiny)
        if float(np.min(sigma_eigenvalues)) < -1e-10*sigma_scale:
            raise ValueError("initial_ray_phase_space_covariance must be positive semidefinite")
        canonical = np.block([
            [np.zeros((3,3)), -np.eye(3)],
            [np.eye(3), np.zeros((3,3))],
        ])
        ray_states = np.column_stack((positions, wavevectors))
        fundamental = np.eye(6)
        covariance_states = []
        transverse_states = []
        for sample_index, state_vector in enumerate(ray_states):
            if sample_index > 0:
                midpoint = 0.5*(ray_states[sample_index-1] + state_vector)
                h = np.power(np.finfo(float).eps, 0.25) * np.maximum(
                    np.abs(midpoint),
                    np.r_[np.full(3, max(float(np.linalg.norm(midpoint[:3])), 1.0)),
                          np.full(3, max(float(np.linalg.norm(midpoint[3:])), 1.0))]
                )
                hessian = np.zeros((6,6), dtype=float)
                center_value = eigenvalue(midpoint[:3], midpoint[3:])
                for axis in range(6):
                    delta = np.zeros(6); delta[axis] = h[axis]
                    plus = eigenvalue((midpoint+delta)[:3], (midpoint+delta)[3:])
                    minus = eigenvalue((midpoint-delta)[:3], (midpoint-delta)[3:])
                    hessian[axis,axis] = (plus - 2.0*center_value + minus)/(h[axis]**2)
                for row in range(6):
                    for column in range(row+1,6):
                        dr = np.zeros(6); dc = np.zeros(6)
                        dr[row] = h[row]; dc[column] = h[column]
                        ppv = eigenvalue((midpoint+dr+dc)[:3], (midpoint+dr+dc)[3:])
                        pmv = eigenvalue((midpoint+dr-dc)[:3], (midpoint+dr-dc)[3:])
                        mpv = eigenvalue((midpoint-dr+dc)[:3], (midpoint-dr+dc)[3:])
                        mmv = eigenvalue((midpoint-dr-dc)[:3], (midpoint-dr-dc)[3:])
                        value = (ppv-pmv-mpv+mmv)/(4.0*h[row]*h[column])
                        hessian[row,column] = value
                        hessian[column,row] = value
                hamiltonian_matrix = canonical @ hessian
                ds = float(solution.t[sample_index]-solution.t[sample_index-1])
                fundamental = expm(hamiltonian_matrix*ds) @ fundamental
            sigma = fundamental @ sigma0 @ fundamental.T
            sigma = 0.5*(sigma+sigma.T)
            covariance_states.append(sigma)
            transverse_basis = _transverse_frame(wavevectors[sample_index])
            transverse_states.append(transverse_basis.T @ sigma[:3,:3] @ transverse_basis)
        propagated_ray_covariance = np.asarray(covariance_states)
        propagated_ray_transverse_covariance = np.asarray(transverse_states)
    return {
        "ray_parameter": solution.t,
        "position_m": positions,
        "wavevector_rad_m": wavevectors,
        "dispersion_eigenvalue": eigenvalues,
        "electric_polarization": polarizations,
        "transverse_position_covariance_m2": (
            propagated_ray_transverse_covariance if propagated_ray_transverse_covariance is not None else
            propagated_beam["transverse_position_covariance_m2"] if propagated_beam is not None else None
        ),
        "phase_space_covariance": (
            propagated_ray_covariance if propagated_ray_covariance is not None else
            propagated_beam["phase_space_covariance"] if propagated_beam is not None else None
        ),
        "paraxial_beam_propagation_status": (
            "propagated_from_full_cold_plasma_dispersion_Hessian" if propagated_ray_covariance is not None else
            "propagated_from_supplied_symplectic_ABCD_maps" if propagated_beam is not None else "not_requested"
        ),
        "mode_index": int(mode_index),
        "mode_branch_tracking": branch_tracking,
        "coherent_mode_conversion": mode_conversion,
        "mode_conversion_status": (
            "computed_from_explicit_path_and_propagation_constants"
            if mode_conversion is not None else
            "not_requested_requires_initial_mode_amplitudes_and_propagation_constants"
        ),
        "fixed_eigenvalue_order_tracks_initial_polarization_branch": fixed_order_branch_consistent,
        "parameterization": "Hamiltonian eigenvalue parameter; not physical time",
        "limitations": [
            "Cold, collisionless, local dielectric; not valid at unresolved resonant layers.",
            "The ray ODE follows a fixed eigenvalue-order index. If the tracked eigenvalue-order permutation changes, the ray is not following one continuous polarization branch across the crossing.",
            "Optional lossless coherent mode-amplitude propagation is returned only when its path and propagation constants are explicitly supplied; it does not change the fixed-branch ray trajectory or CTS spectral transfer.",
            "Ray trajectories remain geometrical-optics characteristics; optional finite-beam covariance is a linearized paraxial Gaussian phase-space approximation, not a full wave solution.",
            "The equilibrium callback and its spatial derivatives are supplied by the caller."
        ],
    }



def trace_cts_frequency_resolved_beams(
    frequency_shift_Hz, probe_frequency_Hz, *,
    incident_ray, receiver_rays, equilibrium,
    incident_beam_covariance_m2=None, receiver_beam_covariances_m2=None,
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

    This is cold-plasma geometrical optics. Beam covariance may be supplied as
    local 3-D matrices or propagated from explicit symplectic paraxial ABCD maps
    on each ray; propagated transverse covariances are used in a 2-D Gaussian
    overlap integral. The ray solver does not derive the ABCD maps from the
    dispersion Hessian. Optional coherent mode-amplitude propagation remains
    diagnostic output only and does not alter the ray trajectories or CTS
    spectral transfer. Non-paraxial diffraction and general vector
    quasi-optical mode conversion remain outside this model. The workflow
    assumes uniform fitted ne/Te/Ti across the sampled volume while allowing
    local ray directions and magnetic field to vary.
    """
    if (
        isinstance(mode_index, bool) or not isinstance(mode_index, (int, np.integer))
        or int(mode_index) not in (0, 1, 2)
    ):
        raise ValueError(
            "mode_index must be explicitly selected as 0, 1 or 2 for "
            "frequency-resolved cold-plasma ray tracing"
        )
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
        initial_mode_amplitudes=incident_ray.get("initial_mode_amplitudes"),
        propagation_constants_rad_m=incident_ray.get("propagation_constants_rad_m"),
        initial_beam_phase_space_covariance=incident_ray.get("initial_beam_phase_space_covariance"),
        paraxial_transfer_matrices=incident_ray.get("paraxial_transfer_matrices"),
        initial_ray_phase_space_covariance=incident_ray.get("initial_ray_phase_space_covariance"),
    )
    incident_positions = np.asarray(incident["position_m"], dtype=float)
    if incident.get("transverse_position_covariance_m2") is not None:
        incident_cov = np.asarray(incident["transverse_position_covariance_m2"], dtype=float)
    else:
        if incident_beam_covariance_m2 is None:
            raise ValueError("supply incident_beam_covariance_m2 or incident_ray paraxial covariance propagation inputs")
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
            initial_mode_amplitudes=ray_spec.get("initial_mode_amplitudes"),
            propagation_constants_rad_m=ray_spec.get("propagation_constants_rad_m"),
            initial_beam_phase_space_covariance=ray_spec.get("initial_beam_phase_space_covariance"),
            paraxial_transfer_matrices=ray_spec.get("paraxial_transfer_matrices"),
            initial_ray_phase_space_covariance=ray_spec.get("initial_ray_phase_space_covariance"),
        )
        receiver_positions = np.asarray(receiver["position_m"], dtype=float)
        if receiver.get("transverse_position_covariance_m2") is not None:
            receiver_cov_for_frequency = np.asarray(receiver["transverse_position_covariance_m2"], dtype=float)
        else:
            if receiver_beam_covariances_m2 is None:
                raise ValueError(f"supply receiver_beam_covariances_m2 or receiver_rays[{index}] paraxial covariance propagation inputs")
            receiver_cov = np.asarray(receiver_beam_covariances_m2, dtype=float)
            if receiver_cov.shape == (3, 3):
                receiver_cov_for_frequency = np.broadcast_to(receiver_cov, (receiver_positions.shape[0], 3, 3))
            elif receiver_cov.shape == (shifts.size, 3, 3):
                receiver_cov_for_frequency = np.broadcast_to(receiver_cov[index], (receiver_positions.shape[0], 3, 3))
            elif receiver_cov.shape == (shifts.size, receiver_positions.shape[0], 3, 3):
                receiver_cov_for_frequency = receiver_cov[index]
            else:
                raise ValueError("receiver_beam_covariances_m2 must be (3,3), (n_frequency,3,3), (n_frequency,n_receiver_ray,3,3), or be supplied through ray paraxial propagation");
        if incident_cov.shape[-2:] != receiver_cov_for_frequency.shape[-2:]:
            raise ValueError("incident and receiver beam covariances must both use 2-D propagated transverse blocks or both use 3-D local covariance matrices")
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
            if incident_cov.shape[-2:] == (2, 2) and receiver_cov_for_frequency.shape[-2:] == (2, 2):
                overlap = gaussian_beam_overlap_transverse(
                    xi, ki, incident_cov[incident_index], xs, ks,
                    receiver_cov_for_frequency[receiver_index],
                )
            else:
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
                "beam_overlap": float(overlap),
                "beam_overlap_dimension": 2 if incident_cov.shape[-2:] == (2, 2) else 3,
                "beam_overlap_unit": "m^-2" if incident_cov.shape[-2:] == (2, 2) else "m^-3",
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
            "beam_overlap_path_integral": path_integral,
            "beam_overlap_path_integral_unit": "m^-1" if incident_cov.shape[-2:] == (2, 2) else "m^-2",
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
        "absolute_beam_overlap_path_integral": np.asarray(path_integrals, dtype=float),
        "absolute_beam_overlap_path_integral_unit": "m^-1" if incident_cov.shape[-2:] == (2, 2) else "m^-2",
        "incident_ray": incident,
        "geometry_model": "cold-plasma Hamiltonian ray tracing with path-integrated local Gaussian intensity overlap",
        "scattering_volume_integration": "normalized Gaussian overlap times incident-ray path-length quadrature",
        "scattering_sample_count": np.asarray([len(entry["scattering_samples"]) for entry in entries], dtype=int),
        "limitations": [
            "Cold-plasma ray trajectories remain geometrical optics; finite-beam covariance is propagated paraxially from the dispersion Hessian or supplied ABCD maps, while non-paraxial diffraction and full vector quasi-optics are not solved.",
            "Optional coherent mode-amplitude propagation is diagnostic output only; it does not alter ray trajectories, overlap weights or the full-EM CTS spectral transfer.",
            "Beam covariance is supplied locally, propagated from explicit symplectic paraxial maps, or propagated from the full cold-plasma dispersion Hessian; the latter is valid only along a continuous non-degenerate eigenmode branch.",
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


def track_cold_plasma_modes_from_operators(
    wave_operator_sequence,
    *,
    degeneracy_relative_tolerance=1e-7,
):
    """Track cold-plasma eigenmode branches by eigenvector continuity.

    Parameters
    ----------
    wave_operator_sequence:
        Complex (n_sample, 3, 3) Hermitian cold-plasma wave operators sampled
        along a frequency or ray coordinate. Each operator is constructed by
        :func:`cold_plasma_wave_operator` from the supplied local B field and
        species state.
    degeneracy_relative_tolerance:
        Relative eigenvalue-gap threshold at which an individual eigenvector
        branch is reported as ambiguous.

    Eigenvectors are matched between adjacent samples by maximizing total
    squared polarization overlap (Hungarian assignment), then phase-aligned.
    At an exact/near degeneracy the basis inside the degenerate subspace is not
    unique; the function reports that ambiguity rather than inventing a branch.
    """
    from scipy.optimize import linear_sum_assignment

    operators = np.asarray(wave_operator_sequence, dtype=complex)
    if operators.ndim != 3 or operators.shape[0] < 1 or operators.shape[1:] != (3, 3):
        raise ValueError("wave_operator_sequence must have shape (n_sample, 3, 3)")
    if not np.all(np.isfinite(operators)):
        raise ValueError("wave_operator_sequence must be finite")
    tolerance = float(degeneracy_relative_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("degeneracy_relative_tolerance must be positive and finite")

    values = []
    vectors = []
    permutations = []
    overlaps = []
    ambiguous = []
    previous = None
    for sample_index, operator in enumerate(operators):
        hermitian = 0.5 * (operator + operator.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        gaps = np.abs(np.diff(eigenvalues))
        is_degenerate = bool(np.any(gaps <= tolerance * scale))
        if previous is None:
            order = np.arange(3)
            matched = eigenvectors.copy()
            sample_overlap = np.ones(3, dtype=float)
        else:
            overlap_matrix = np.abs(previous.conj().T @ eigenvectors) ** 2
            rows, columns = linear_sum_assignment(-overlap_matrix)
            order = columns[np.argsort(rows)]
            matched = eigenvectors[:, order].copy()
            sample_overlap = np.empty(3, dtype=float)
            for mode in range(3):
                inner = np.vdot(previous[:, mode], matched[:, mode])
                sample_overlap[mode] = float(np.abs(inner) ** 2)
                if abs(inner) > 0:
                    matched[:, mode] *= np.exp(-1j * np.angle(inner))
        values.append(eigenvalues[order])
        vectors.append(matched)
        permutations.append(order.astype(int))
        overlaps.append(sample_overlap)
        ambiguous.append(is_degenerate)
        previous = matched

    return {
        "tracked_eigenvalues": np.asarray(values, dtype=float),
        "tracked_electric_polarizations": np.asarray(vectors, dtype=complex),
        "eigenvalue_order_permutation": np.asarray(permutations, dtype=int),
        "adjacent_sample_polarization_overlap_squared": np.asarray(overlaps, dtype=float),
        "degenerate_subspace_detected": np.asarray(ambiguous, dtype=bool),
        "branch_tracking_method": "maximum_squared_eigenvector_overlap_with_Hungarian_assignment_and_phase_alignment",
        "branch_identifiability": (
            "ambiguous_at_one_or_more_degenerate_samples"
            if any(ambiguous) else "locally_continuous_eigenvector_branches"
        ),
        "limitations": [
            "At an exact or near eigenvalue degeneracy, individual eigenvectors within the degenerate subspace are not unique; only the subspace is physically well-defined.",
            "This tracks eigenvectors of the supplied sequence of local wave operators; it does not add mode-conversion coupling or solve a coupled-mode propagation equation.",
            "The sequence ordering must correspond to the physical frequency/ray sampling order.",
        ],
    }


def propagate_cold_plasma_mode_coupling(
    wave_operator_sequence,
    path_coordinate_m,
    propagation_constants_rad_m,
    initial_mode_amplitudes,
    *,
    degeneracy_relative_tolerance=1e-7,
):
    """Propagate coherent mode amplitudes through a sampled cold-plasma path.

    The supplied Hermitian Maxwell wave operators define the local polarization
    basis. The caller supplies the physical path coordinate and the three
    local propagation constants (rad/m), since eigenvalues of the wave
    operator are not themselves propagation constants. The evolution is the
    lossless, flux-normalized coupled-mode system

        d a / ds = -[U^H dU/ds + i diag(beta)] a,

    discretized with a unitary midpoint propagator. The polarization frame is
    parallel-transported between samples; exact/near-degenerate eigenspaces
    are aligned as subspaces by unitary Procrustes transport. Individual mode
    labels inside a degenerate subspace remain gauge-dependent, while total
    power in that subspace is invariant.

    This is a local cold-plasma coupled-mode model. It does not include
    absorption, collisions, reflection at turning points, diffraction,
    nonlinear coupling, or unresolved resonant-layer physics. The input path,
    wave operators and propagation constants must describe the same physical
    ray/path. No equilibrium or path is inferred.
    """
    from scipy.linalg import expm, logm
    from scipy.optimize import linear_sum_assignment

    operators = np.asarray(wave_operator_sequence, dtype=complex)
    path = np.asarray(path_coordinate_m, dtype=float)
    beta = np.asarray(propagation_constants_rad_m, dtype=float)
    amplitudes0 = np.asarray(initial_mode_amplitudes, dtype=complex)
    if operators.ndim != 3 or operators.shape[0] < 2 or operators.shape[1:] != (3, 3):
        raise ValueError("wave_operator_sequence must have shape (n_sample>=2, 3, 3)")
    n = operators.shape[0]
    if path.shape != (n,) or not np.all(np.isfinite(path)) or np.any(np.diff(path) <= 0):
        raise ValueError("path_coordinate_m must be finite, strictly increasing and match the operator samples")
    if beta.shape != (n, 3) or not np.all(np.isfinite(beta)):
        raise ValueError("propagation_constants_rad_m must have shape (n_sample, 3) and be finite")
    if amplitudes0.shape != (3,) or not np.all(np.isfinite(amplitudes0)):
        raise ValueError("initial_mode_amplitudes must be a finite complex 3-vector")
    tolerance = float(degeneracy_relative_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("degeneracy_relative_tolerance must be positive and finite")
    if not np.allclose(operators, np.swapaxes(operators.conj(), 1, 2), rtol=1e-10, atol=1e-12):
        raise ValueError("wave operators must be Hermitian; non-Hermitian operators are not silently symmetrized")
    eigvals = np.empty((n, 3), dtype=float)
    frames = np.empty((n, 3, 3), dtype=complex)
    degenerate = np.zeros(n, dtype=bool)
    subspace_power = []
    for j, raw in enumerate(operators):
        h = 0.5 * (raw + raw.conj().T)
        vals, vecs = np.linalg.eigh(h)
        scale = max(float(np.max(np.abs(vals))), np.finfo(float).tiny)
        groups = []
        start = 0
        for q in range(2):
            if abs(vals[q + 1] - vals[q]) > tolerance * scale:
                groups.append(np.arange(start, q + 1))
                start = q + 1
        groups.append(np.arange(start, 3))
        degenerate[j] = any(len(g) > 1 for g in groups)
        if j == 0:
            frame = vecs.copy()
        else:
            # Match individual isolated modes by maximum total polarization overlap.
            overlap = np.abs(frames[j - 1].conj().T @ vecs) ** 2
            rows, cols = linear_sum_assignment(-overlap)
            order = cols[np.argsort(rows)]
            frame = vecs[:, order].copy()
            # Align each vector's arbitrary complex phase.
            for m in range(3):
                z = np.vdot(frames[j - 1, :, m], frame[:, m])
                if abs(z) > 0:
                    frame[:, m] *= np.exp(-1j * np.angle(z))
            # Within a degenerate cluster, transport the whole subspace by
            # the unitary polar factor, avoiding arbitrary eigensolver rotations.
            for raw_group in groups:
                if len(raw_group) > 1:
                    # Convert raw eigensolver columns to their matched-frame
                    # positions before aligning the degenerate subspace.
                    group = np.flatnonzero(np.isin(order, raw_group))
                    previous_group = frames[j - 1][:, group]
                    current_group = frame[:, group]
                    u, _, vh = np.linalg.svd(current_group.conj().T @ previous_group)
                    frame[:, group] = current_group @ (u @ vh)
        frames[j] = frame
        eigvals[j] = vals if j == 0 else vals[order]
        if j == 0:
            order = np.arange(3)
        tracked_groups = [
            np.flatnonzero(np.isin(order, raw_group))
            for raw_group in groups
        ]
        subspace_power.append([
            group.astype(int).tolist()
            for group in tracked_groups if len(group) > 1
        ])
    state = amplitudes0.copy()
    amplitudes = np.empty((n, 3), dtype=complex)
    amplitudes[0] = state
    coupling = np.zeros((n - 1, 3, 3), dtype=complex)
    for j in range(n - 1):
        ds = float(path[j + 1] - path[j])
        overlap = frames[j].conj().T @ frames[j + 1]
        # The unitary polar factor removes finite-step non-unitarity from
        # discretization before taking the anti-Hermitian connection.
        u, _, vh = np.linalg.svd(overlap)
        transport = u @ vh
        connection = logm(transport) / ds
        connection = 0.5 * (connection - connection.conj().T)
        beta_mid = 0.5 * (beta[j] + beta[j + 1])
        generator = -connection - 1j * np.diag(beta_mid)
        step = expm(generator * ds)
        state = step @ state
        amplitudes[j + 1] = state
        coupling[j] = connection
    powers = np.abs(amplitudes) ** 2
    total_power = np.sum(powers, axis=1)
    initial_power = float(np.sum(np.abs(amplitudes0) ** 2))
    conservation_error = float(np.max(np.abs(total_power - initial_power)))
    degenerate_subspace_power = np.zeros(n, dtype=float)
    degenerate_subspace_groups = []
    for sample_index, groups_at_sample in enumerate(subspace_power):
        sample_groups = []
        for indices in groups_at_sample:
            group_power = float(np.sum(powers[sample_index, indices]))
            degenerate_subspace_power[sample_index] += group_power
            sample_groups.append({
                "mode_indices": indices,
                "power": group_power,
            })
        degenerate_subspace_groups.append(sample_groups)
    return {
        "path_coordinate_m": path.copy(),
        "mode_amplitudes": amplitudes,
        "mode_powers": powers,
        "total_mode_power": total_power,
        "mode_conversion_matrix_connection_per_m": coupling,
        "local_wave_operator_eigenvalues": eigvals,
        "parallel_transported_polarization_frames": frames,
        "degenerate_subspace_detected": degenerate,
        "degenerate_subspace_power": degenerate_subspace_power,
        "degenerate_subspace_groups": degenerate_subspace_groups,
        "mode_conversion_model": "lossless_flux_normalized_cold_plasma_coupled_mode_equation",
        "coupled_mode_equation": "d(a)/ds = -[U^H dU/ds + i diag(beta)] a",
        "power_conservation_max_abs_error": conservation_error,
        "assumptions": [
            "Hermitian local cold-plasma Maxwell operator and flux-normalized modes.",
            "Caller-supplied propagation constants and ordered physical path coordinate.",
            "Piecewise-smooth path and sufficiently resolved operator sampling.",
            "No absorption, collisions, reflections, diffraction, nonlinear coupling or unresolved resonant layers."
        ],
        "limitations": [
            "At exact degeneracy, individual labels inside the degenerate subspace are basis-dependent; only subspace power is invariant.",
            "This is not a substitute for full-wave boundary-value modelling at cutoffs, resonances or turning points.",
            "No experimental validation is performed."
        ],
    }
