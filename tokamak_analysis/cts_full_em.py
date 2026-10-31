"""Magnetized full-electromagnetic CTS fluctuation solver.

Implements a collisionless, non-relativistic, isotropic-Maxwellian kinetic
current response in a uniform magnetic field.  The response is evaluated by
a converged gyro-harmonic/velocity-quadrature representation of the linearized
Vlasov equation; Maxwell's wave operator then screens the independent particle
current fluctuations.  Density, current, electric-field and magnetic-field
cross-spectra are retained in one Hermitian covariance matrix.

This is a homogeneous local model.  It does not implement ray tracing, finite
beam overlap, relativistic or anisotropic fast-ion distributions, or the full
three-wave coupling corrections of Bindslev (1993, 1996).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
from scipy.special import roots_laguerre, roots_hermite

E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 1.380649e-23
ME = 9.1093837015e-31
AMU = 1.66053906660e-27
C = 299792458.0


def _vec3(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.all(np.isfinite(a)) or np.linalg.norm(a) == 0:
        raise ValueError(f"{name} must be a finite non-zero 3-vector")
    return a / np.linalg.norm(a)


def _finite_vec3(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return a


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be positive and finite")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _hermitian_psd(matrix, name, tolerance=1e-9):
    h = 0.5 * (matrix + matrix.conj().T)
    eigenvalues = np.linalg.eigvalsh(h)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -tolerance * scale:
        raise ValueError(f"{name} is not positive semidefinite")
    if np.min(eigenvalues) < 0:
        values, vectors = np.linalg.eigh(h)
        h = (vectors * np.maximum(values, 0.0)) @ vectors.conj().T
    return h


def _species_list(electron_density_m3, ion_species, ion_fractions,
                  electron_temperature_eV, ion_temperature_eV,
                  electron_flow_m_s, ion_flow_m_s):
    fractions = np.asarray(ion_fractions, dtype=float)
    if (not isinstance(ion_species, Sequence) or isinstance(ion_species, (str, bytes))
            or len(ion_species) == 0 or fractions.shape != (len(ion_species),)
            or np.any(~np.isfinite(fractions)) or np.any(fractions <= 0)
            or not np.isclose(fractions.sum(), 1.0, rtol=1e-6, atol=1e-8)):
        raise ValueError("ion_species and positive ion_fractions summing to one are required")
    result = [{
        "name": "e-", "density": electron_density_m3, "temperature": electron_temperature_eV,
        "mass": ME, "charge": -E_CHARGE, "flow": _finite_vec3(electron_flow_m_s, "electron_flow_velocity_m_s")
    }]
    for i, item in enumerate(ion_species):
        if not isinstance(item, Mapping):
            raise ValueError("each ion species must be a mapping")
        z = _positive(item.get("charge_state"), f"ion_species[{i}].charge_state")
        mass = _positive(item.get("mass_amu"), f"ion_species[{i}].mass_amu") * AMU
        flow = ion_flow_m_s.get(str(item.get("name", f"ion_{i}")), ion_flow_m_s.get("default"))
        if flow is None:
            raise ValueError(f"ion_flow_velocity_m_s must provide a vector for {item.get('name', f'ion_{i}')}")
        result.append({
            "name": str(item.get("name", f"ion_{i}")),
            "density": fractions[i] * electron_density_m3 / z,
            "temperature": ion_temperature_eV, "mass": mass,
            "charge": z * E_CHARGE, "flow": _finite_vec3(flow, f"ion flow for species {i}")
        })
    return result


def _gyro_basis(magnetic_field_T, k_vector):
    b = np.asarray(magnetic_field_T, dtype=float)
    if b.shape != (3,) or np.any(~np.isfinite(b)) or np.linalg.norm(b) <= 0:
        raise ValueError("magnetic_field_T must be a finite non-zero 3-vector")
    bhat = b / np.linalg.norm(b)
    kpar = float(np.dot(k_vector, bhat))
    kperp_vector = k_vector - kpar * bhat
    kperp = float(np.linalg.norm(kperp_vector))
    if kperp > 1e-14 * np.linalg.norm(k_vector):
        e1 = kperp_vector / kperp
    else:
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, bhat)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(bhat, trial)
        e1 /= np.linalg.norm(e1)
    e2 = np.cross(bhat, e1)
    rotation = np.column_stack((e1, e2, bhat))
    return rotation, kperp, kpar, bhat


def _species_conductivity(omega, k_vector, magnetic_field_T, species,
                          *, harmonic_order, quadrature_order, damping_rate_rad_s):
    """Numerical magnetized-Maxwellian Vlasov conductivity in the B-aligned basis."""
    rotation, kperp, kpar, bhat = _gyro_basis(magnetic_field_T, k_vector)
    mass, charge = species["mass"], species["charge"]
    density, temperature = species["density"], species["temperature"]
    flow = species["flow"]
    omega_c = charge * np.linalg.norm(magnetic_field_T) / mass
    vth = np.sqrt(2.0 * temperature * E_CHARGE / mass)
    omega_d = omega - float(np.dot(k_vector, flow))
    n_h = int(harmonic_order)
    harmonics = np.arange(-n_h, n_h + 1)
    nh = harmonics.size
    xperp, wperp = roots_laguerre(int(quadrature_order))
    xpar, wpar = roots_hermite(int(quadrature_order))
    sigma = np.zeros((3, 3), dtype=complex)
    eye = np.eye(nh, dtype=complex)
    coupling = np.zeros((nh, nh), dtype=complex)
    for j in range(nh - 1):
        coupling[j, j + 1] = 0.5
        coupling[j + 1, j] = 0.5
    for xp, wp in zip(xperp, wperp):
        vperp = vth * np.sqrt(xp)
        matrix_coupling = kperp * vperp * coupling
        for xz, wz in zip(xpar, wpar):
            vpar = vth * xz
            denom = (omega_d - kpar * vpar - harmonics * omega_c
                     + 1j * damping_rate_rad_s)
            operator = np.diag(denom.astype(complex)) + matrix_coupling
            source = np.zeros((nh, 3), dtype=complex)
            # Fourier coefficients of E dot v in exp(i*n*gyro-angle).
            source[n_h + 1, 0] = 0.5 * vperp
            source[n_h - 1, 0] = 0.5 * vperp
            source[n_h + 1, 1] = -0.5j * vperp
            source[n_h - 1, 1] = 0.5j * vperp
            source[n_h, 2] = vpar
            try:
                response = np.linalg.solve(operator, 1j * charge / (temperature * E_CHARGE) * source)
            except np.linalg.LinAlgError as exc:
                raise ValueError("singular magnetized kinetic response; increase harmonic_order or damping_rate_rad_s") from exc
            f0_weight = density / (np.pi ** 1.5 * vth ** 3)
            # d^3v = v_perp dv_perp dphi dv_parallel.
            measure = (vth ** 3 / 2.0) * wp * wz * f0_weight * np.exp(-xz * xz)
            jx = np.pi * vperp * (response[n_h + 1] + response[n_h - 1])
            jy = np.pi * vperp * (response[n_h - 1] - response[n_h + 1]) / 1j
            jz = 2.0 * np.pi * vpar * response[n_h]
            sigma += charge * measure * np.vstack((jx, jy, jz))
    # Transform the conductivity from the B-aligned frame to laboratory axes.
    return rotation @ sigma @ rotation.T


def _cold_mode_polarization(frequency_Hz, direction, magnetic_field_T,
                            species, mode_index):
    """Cold-plasma eigenpolarization for incident/scattered high-frequency modes."""
    omega = 2.0 * np.pi * frequency_Hz
    b_hat = np.asarray(magnetic_field_T, dtype=float)
    b_hat /= np.linalg.norm(b_hat)
    # Cold dielectric tensor is used only for the high-frequency probe modes.
    eps = np.eye(3, dtype=complex)
    cross_b = np.array([[0, -b_hat[2], b_hat[1]],
                        [b_hat[2], 0, -b_hat[0]],
                        [-b_hat[1], b_hat[0], 0]], dtype=complex)
    for sp in species:
        wp2 = sp["density"] * sp["charge"] ** 2 / (EPS0 * sp["mass"])
        wc = sp["charge"] * np.linalg.norm(magnetic_field_T) / sp["mass"]
        eps -= wp2 / (omega * (omega + 1j * 1e-12)) * (
            np.eye(3) + 1j * wc / (omega + 1j * 1e-12) * cross_b
        )
    khat = _vec3(direction, "beam direction")
    n2 = (C * 2.0 * np.pi * frequency_Hz / C) ** 2 / omega ** 2
    wave = eps + n2 * (np.outer(khat, khat) - np.eye(3))
    _, vectors = np.linalg.eig(wave)
    transverse = vectors[:, np.argsort(np.abs(np.linalg.eigvals(wave)))[:2]]
    if mode_index not in (0, 1):
        raise ValueError("polarization mode index must be 0 or 1")
    p = transverse[:, mode_index]
    p = p - khat * np.dot(khat, p)
    norm = np.linalg.norm(p)
    if norm <= 1e-12:
        raise ValueError("selected cold-plasma mode is not transverse to its beam direction")
    return p / norm


def full_em_cts_spectral_density(
    frequency_shift_Hz, probe_frequency_Hz, scattering_geometry, *,
    electron_density_m3, electron_temperature_eV, ion_temperature_eV,
    ion_species, ion_fractions, magnetic_field_T,
    electron_flow_velocity_m_s, ion_flow_velocity_m_s,
    incident_polarization, scattered_polarization,
    harmonic_order=8, quadrature_order=12, damping_rate_rad_s=None,
):
    """Return a local full-EM magnetized CTS spectrum and fluctuation diagnostics.

    Required geometry keys: incident_direction, scattered_direction (unit vectors).
    Polarizations are complex 3-vectors supplied by the diagnostic mode definition.
    Fluctuation covariance includes electron density/current/E/B cross-correlations.
    """
    shifts = np.asarray(frequency_shift_Hz, dtype=float)
    if shifts.ndim != 1 or shifts.size == 0 or np.any(~np.isfinite(shifts)):
        raise ValueError("frequency_shift_Hz must be a finite non-empty vector")
    if np.any(np.diff(shifts) <= 0):
        raise ValueError("frequency_shift_Hz must be strictly increasing")
    f0 = _positive(probe_frequency_Hz, "probe_frequency_Hz")
    ne = _positive(electron_density_m3, "electron_density_m3")
    te = _positive(electron_temperature_eV, "electron_temperature_eV")
    ti = _positive(ion_temperature_eV, "ion_temperature_eV")
    if not isinstance(scattering_geometry, Mapping):
        raise ValueError("scattering_geometry must be a mapping")
    ki_hat = _vec3(scattering_geometry.get("incident_direction"), "incident_direction")
    ks_hat = _vec3(scattering_geometry.get("scattered_direction"), "scattered_direction")
    b = np.asarray(magnetic_field_T, dtype=float)
    if b.shape != (3,) or np.any(~np.isfinite(b)) or np.linalg.norm(b) <= 0:
        raise ValueError("magnetic_field_T must be a finite non-zero 3-vector")
    if not isinstance(ion_flow_velocity_m_s, Mapping):
        raise ValueError("ion_flow_velocity_m_s must map every ion species name to a 3-vector")
    species = _species_list(ne, ion_species, ion_fractions, te, ti,
                            electron_flow_velocity_m_s, ion_flow_velocity_m_s)
    ei = np.asarray(incident_polarization, dtype=complex)
    es = np.asarray(scattered_polarization, dtype=complex)
    if ei.shape != (3,) or es.shape != (3,) or np.any(~np.isfinite(ei)) or np.any(~np.isfinite(es)):
        raise ValueError("incident_polarization and scattered_polarization must be finite complex 3-vectors")
    ei /= np.linalg.norm(ei)
    es /= np.linalg.norm(es)
    if abs(np.vdot(ei, ki_hat)) > 1e-6 or abs(np.vdot(es, ks_hat)) > 1e-6:
        raise ValueError("incident and scattered polarization vectors must be transverse to their beam directions")
    if isinstance(harmonic_order, bool) or int(harmonic_order) < 2:
        raise ValueError("harmonic_order must be an integer >= 2")
    if isinstance(quadrature_order, bool) or int(quadrature_order) < 4:
        raise ValueError("quadrature_order must be an integer >= 4")
    if damping_rate_rad_s is None:
        damping = max(2.0 * np.pi * f0 * 1e-7, 1.0)
    else:
        damping = _positive(damping_rate_rad_s, "damping_rate_rad_s")
    if np.any(f0 + shifts <= 0):
        raise ValueError("scattered frequencies must remain positive")
    # The scattering vector is frequency-dependent because omega_s != omega_i.
    spectra = np.empty(shifts.size, dtype=float)
    component_covariances = np.empty((shifts.size, 10, 10), dtype=complex)
    dielectric_tensors = np.empty((shifts.size, 3, 3), dtype=complex)
    wave_tensors = np.empty_like(dielectric_tensors)
    harmonic_counts = []
    for index, shift in enumerate(shifts):
        fs = f0 + shift
        ki = (2.0 * np.pi * f0 / C) * ki_hat
        ks = (2.0 * np.pi * fs / C) * ks_hat
        kvec = ks - ki
        omega = 2.0 * np.pi * shift
        if abs(omega) < damping:
            omega = np.copysign(damping, omega if omega else 1.0)
        sigmas = [_species_conductivity(omega, kvec, b, sp,
                    harmonic_order=int(harmonic_order),
                    quadrature_order=int(quadrature_order),
                    damping_rate_rad_s=damping) for sp in species]
        sigma_total = np.sum(sigmas, axis=0)
        epsilon = np.eye(3, dtype=complex) + 1j * sigma_total / (EPS0 * omega)
        k_norm = np.linalg.norm(kvec)
        if k_norm <= 0:
            raise ValueError("scattering wavevector must be non-zero")
        khat = kvec / k_norm
        n2 = (C * k_norm / omega) ** 2
        wave_tensor = epsilon + n2 * (np.outer(khat, khat) - np.eye(3))
        try:
            green = np.linalg.inv(wave_tensor)
        except np.linalg.LinAlgError as exc:
            raise ValueError("singular electromagnetic wave tensor at a frequency bin") from exc
        # Independent-particle Langevin currents are related to Re(sigma) by
        # the classical fluctuation-dissipation theorem (two-sided spectrum).
        q_blocks = []
        for sp, sigma in zip(species, sigmas):
            q_blocks.append(_hermitian_psd(
                2.0 * KB * sp["temperature"] * E_CHARGE
                * 0.5 * (sigma + sigma.conj().T),
                f"bare current spectrum for {sp['name']}"
            ))
        source_cov = np.zeros((3 * len(species), 3 * len(species)), dtype=complex)
        for j, qj in enumerate(q_blocks):
            source_cov[3*j:3*j+3, 3*j:3*j+3] = qj
        # E = -i/(omega eps0) Lambda^-1 sum_a j_a0.
        field_map = -1j / (omega * EPS0) * green
        maps = []
        for j, sigma in enumerate(sigmas):
            selector = np.zeros((3, 3 * len(species)), dtype=complex)
            selector[:, 3*j:3*j+3] = np.eye(3)
            maps.append(selector + sigma @ field_map @ np.ones((3, 3 * len(species))))
        electron_current_map = maps[0]
        electron_density_map = (kvec @ electron_current_map) / (omega * species[0]["charge"])
        electric_map = field_map @ np.ones((3, 3 * len(species)))
        magnetic_map = np.array([
            [0, -kvec[2], kvec[1]],
            [kvec[2], 0, -kvec[0]],
            [-kvec[1], kvec[0], 0],
        ], dtype=complex) @ electric_map / omega
        state_map = np.vstack((electron_density_map, electron_current_map,
                               electric_map, magnetic_map))
        state_cov = _hermitian_psd(state_map @ source_cov @ state_map.conj().T,
                                   "electromagnetic fluctuation covariance")
        component_covariances[index] = state_cov
        dielectric_tensors[index] = epsilon
        wave_tensors[index] = wave_tensor
        # The receiver measures the transverse electron-current radiation.
        # Incident polarization sets the Thomson polarization factor; all
        # density/current/E/B cross-correlations remain available in state_cov.
        receiver = np.zeros(10, dtype=complex)
        receiver[1:4] = es.conj()
        received_current_power = float(np.real(receiver @ state_cov @ receiver.conj()))
        polarization_factor = abs(np.vdot(es, ei)) ** 2
        spectra[index] = max(received_current_power * polarization_factor, 0.0)
        harmonic_counts.append(int(harmonic_order))
    return {
        "frequency_shift_Hz": shifts,
        "spectral_density": spectra,
        "fluctuation_covariance_density_current_E_B": component_covariances,
        "dielectric_tensor": dielectric_tensors,
        "wave_tensor": wave_tensors,
        "magnetic_field_T": b,
        "model": "magnetized full-electromagnetic kinetic CTS (local Maxwellian, non-relativistic)",
        "kinetic_response": "linearized Vlasov equation; gyro-harmonic expansion and Gauss-Laguerre/Hermite velocity quadrature",
        "fluctuation_source": "classical fluctuation-dissipation theorem applied species-wise to Hermitian conductivity",
        "retained_cross_correlations": ["electron density-current", "electron density-E", "electron density-B",
            "current-E", "current-B", "E-B"],
        "numerical_controls": {"harmonic_order": int(harmonic_order),
            "quadrature_order": int(quadrature_order), "damping_rate_rad_s": damping},
        "limitations": [
            "Homogeneous local plasma; no ray tracing, refraction along a finite path, or finite scattering-volume beam overlap.",
            "Non-relativistic isotropic Maxwellian distributions only; non-Maxwellian fast-ion VDFs are not yet supported by this full-EM solver.",
            "High-frequency probe propagation and the receiver coupling are represented by supplied transverse polarization vectors; full Bindslev three-wave coupling tensors are not yet included.",
            "Classical fluctuation-dissipation source assumes local thermodynamic equilibrium for each species."
        ],
    }
