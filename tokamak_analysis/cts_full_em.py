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
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
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
        flow = ion_flow_m_s.get(str(item.get("name", f"ion_{i}")))
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


def full_em_cts_spectral_density(
    frequency_shift_Hz, probe_frequency_Hz, scattering_geometry, *,
    electron_density_m3, electron_temperature_eV, ion_temperature_eV,
    ion_species, ion_fractions, magnetic_field_T,
    electron_flow_velocity_m_s, ion_flow_velocity_m_s,
    incident_polarization, scattered_polarization,
    harmonic_order=8, quadrature_order=12, damping_rate_rad_s=None,
    fast_ion_populations=None,
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
    if fast_ion_populations is not None:
        if not isinstance(fast_ion_populations, Sequence) or isinstance(fast_ion_populations, (str, bytes)):
            raise ValueError("fast_ion_populations must be a sequence of explicit gyrotropic populations")
        for index, population in enumerate(fast_ion_populations):
            if not isinstance(population, Mapping):
                raise ValueError(f"fast_ion_populations[{index}] must be a mapping")
            name = str(population.get("name", f"fast_ion_{index}"))
            charge_state = _positive(population.get("charge_state"), f"fast_ion_populations[{index}].charge_state")
            mass_amu = _positive(population.get("mass_amu"), f"fast_ion_populations[{index}].mass_amu")
            density = _positive(population.get("density_m3"), f"fast_ion_populations[{index}].density_m3")
            distribution = population.get("distribution")
            if not isinstance(distribution, Mapping):
                raise ValueError(f"fast_ion_populations[{index}].distribution is required")
            species.append({
                "name": name, "density": density, "temperature": 1.0,
                "mass": mass_amu * AMU, "charge": charge_state * E_CHARGE,
                "flow": _finite_vec3(population.get("flow_velocity_m_s", [0, 0, 0]),
                                     f"fast_ion_populations[{index}].flow_velocity_m_s"),
                "distribution": distribution, "fast_ion": True,
            })
    ei = np.asarray(incident_polarization, dtype=complex).copy()
    es = np.asarray(scattered_polarization, dtype=complex).copy()
    if ei.shape != (3,) or es.shape != (3,) or np.any(~np.isfinite(ei)) or np.any(~np.isfinite(es)):
        raise ValueError("incident_polarization and scattered_polarization must be finite complex 3-vectors")
    if np.linalg.norm(ei) <= 0 or np.linalg.norm(es) <= 0:
        raise ValueError("incident_polarization and scattered_polarization must be non-zero")
    ei /= np.linalg.norm(ei)
    es /= np.linalg.norm(es)
    if abs(np.vdot(ei, ki_hat)) > 1e-6 or abs(np.vdot(es, ks_hat)) > 1e-6:
        raise ValueError("incident and scattered polarization vectors must be transverse to their beam directions")
    if isinstance(harmonic_order, bool) or not isinstance(harmonic_order, (int, np.integer)) or harmonic_order < 2:
        raise ValueError("harmonic_order must be an integer >= 2")
    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, (int, np.integer)) or quadrature_order < 4:
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
    bindslev_operators = []
    bindslev_coefficients = []
    for index, shift in enumerate(shifts):
        fs = f0 + shift
        ki = (2.0 * np.pi * f0 / C) * ki_hat
        ks = (2.0 * np.pi * fs / C) * ks_hat
        kvec = ks - ki
        omega = 2.0 * np.pi * shift
        if abs(omega) < damping:
            omega = np.copysign(damping, omega if omega else 1.0)
        sigmas = []
        fast_bare_covariances = {}
        for species_index, sp in enumerate(species):
            if sp.get("fast_ion", False):
                from .cts_relativistic import relativistic_gyrotropic_species_response
                kinetic = relativistic_gyrotropic_species_response(
                    omega, kvec, b, density_m3=sp["density"], mass_kg=sp["mass"],
                    charge_C=sp["charge"], distribution=sp["distribution"],
                    quadrature_order=int(quadrature_order),
                    gyrophase_points=max(16, 4*int(harmonic_order)),
                    resonance_width_rad_s=damping,
                )
                sigmas.append(kinetic["conductivity_S_m"])
                fast_bare_covariances[species_index] = kinetic["bare_current_spectral_covariance"]
            else:
                sigmas.append(_species_conductivity(
                    omega, kvec, b, sp, harmonic_order=int(harmonic_order),
                    quadrature_order=int(quadrature_order),
                    damping_rate_rad_s=damping
                ))
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
        for species_index, (sp, sigma) in enumerate(zip(species, sigmas)):
            if species_index in fast_bare_covariances:
                q_blocks.append(_hermitian_psd(
                    fast_bare_covariances[species_index],
                    f"Klimontovich bare current spectrum for {sp['name']}"
                ))
            else:
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
        # Apply the complete low-temperature Bindslev source-current tensor
        # to the joint fluctuation covariance.  The fluctuation covariance
        # above is [delta_n, delta_j, delta_E, delta_B]; Bindslev's Eq. (111)
        # is written in [delta_n, delta_particle_flux, delta_E, delta_B].
        from .cts_bindslev import bindslev_source_current_operator, bindslev_correlated_transfer_spectrum
        from .cts_em_geometry import cold_plasma_dielectric_tensor
        electron_cold = [{
            "density_m3": ne, "charge_C": -E_CHARGE, "mass_kg": ME
        }]
        omega_i = 2.0 * np.pi * f0
        omega_s = 2.0 * np.pi * fs
        chi_i = cold_plasma_dielectric_tensor(omega_i, b, electron_cold) - np.eye(3)
        chi_s = cold_plasma_dielectric_tensor(omega_s, b, electron_cold) - np.eye(3)
        source = bindslev_source_current_operator(
            omega_incident_rad_s=omega_i,
            omega_scattered_rad_s=omega_s,
            wavevector_incident_rad_m=ki,
            wavevector_scattered_rad_m=ks,
            incident_electric_field=ei,
            incident_susceptibility=chi_i,
            scattered_susceptibility=chi_s,
            magnetic_field_T=b,
            electron_density_m3=ne,
        )
        # Electron particle flux is j_e/q_e.  This exact unit conversion is
        # required before contracting the Bindslev velocity/flux terms.
        to_flux = np.eye(10, dtype=complex)
        to_flux[1:4, 1:4] /= -E_CHARGE
        covariance_n_v_E_B = to_flux @ state_cov @ to_flux.conj().T
        transfer = bindslev_correlated_transfer_spectrum(
            covariance_n_v_E_B, source["source_current_operator"], es
        )
        spectra[index] = transfer["received_source_current_spectral_power"]
        bindslev_operators.append(source["source_current_operator"])
        projected = es.conj() @ source["source_current_operator"]
        bindslev_coefficients.append({
            "G_n": projected[0],
            "G_v": projected[1:4],
            "G_E": projected[4:7],
            "G_B": projected[7:10],
            "transfer_contributions": transfer["transfer_contributions"],
        })
        harmonic_counts.append(int(harmonic_order))
    return {
        "frequency_shift_Hz": shifts,
        "spectral_density": spectra,
        "fluctuation_covariance_density_current_E_B": component_covariances,
        "dielectric_tensor": dielectric_tensors,
        "wave_tensor": wave_tensors,
        "bindslev_source_current_operator": np.asarray(bindslev_operators),
        "bindslev_coupling_coefficients": bindslev_coefficients,
        "three_wave_coupling": "Bindslev low-temperature magnetized source-current tensor, including Gamma^{-1}, X and all n/v/E/B cross terms",
        "magnetic_field_T": b,
        "fast_ion_populations": fast_ion_populations or [],
        "model": "magnetized full-electromagnetic CTS with Bindslev low-temperature three-wave coupling (thermal response currently Maxwellian, non-relativistic)",
        "kinetic_response": "bulk species: magnetized Maxwellian Vlasov gyro-harmonics; optional fast ions: relativistic gyrotropic momentum-space Vlasov Fourier collocation",
        "fluctuation_source": "bulk thermal species: classical FDT; non-equilibrium fast-ion populations: relativistic Klimontovich orbit shot-noise covariance",
        "retained_cross_correlations": ["electron density-current", "electron density-E", "electron density-B",
            "current-E", "current-B", "E-B"],
        "numerical_controls": {"harmonic_order": int(harmonic_order),
            "quadrature_order": int(quadrature_order), "damping_rate_rad_s": damping},
        "limitations": [
            "Homogeneous local plasma; no ray tracing, refraction along a finite path, or finite scattering-volume beam overlap.",
            "Bulk thermal species use the non-relativistic isotropic Maxwellian solver; optional fast-ion populations use relativistic gyrotropic bi-Gaussian or tabulated momentum distributions.",
            "The Bindslev three-wave source-current closure is the low-temperature magnetized formulation; its relativistic kinetic generalization is not used for the fast-ion population.",
            "Classical FDT is applied only to bulk thermal Maxwellian species; non-equilibrium fast-ion noise uses the uncorrelated Klimontovich orbit spectrum."
        ],
    }



def analyze_full_em_cts_spectrum(
    frequency_shift_Hz, measured_spectrum, *, probe_frequency_Hz,
    scattering_geometry, electron_density_initial_m3,
    electron_temperature_initial_eV, ion_temperature_initial_eV,
    ion_species, ion_fractions, magnetic_field_T,
    electron_flow_velocity_m_s, ion_flow_velocity_m_s,
    incident_polarization, scattered_polarization,
    instrument_response, signal_uncertainty=None, spectral_covariance=None,
    parameter_bounds=None, harmonic_order=8, quadrature_order=12,
    damping_rate_rad_s=None, fast_ion_populations=None,
):
    """Fit n_e, T_e and common T_i using the magnetized full-EM forward model.

    The instrument amplitude and constant background are profiled as nuisance
    parameters. Flow vectors, composition, magnetic field, geometry and mode
    polarizations remain explicit external inputs.
    """
    from scipy.optimize import least_squares

    f = np.asarray(frequency_shift_Hz, dtype=float)
    y = np.asarray(measured_spectrum, dtype=float)
    if f.ndim != 1 or y.shape != f.shape or f.size < 8:
        raise ValueError("frequency_shift_Hz and measured_spectrum must be matching vectors with >=8 bins")
    if np.any(~np.isfinite(f)) or np.any(~np.isfinite(y)) or np.any(np.diff(f) <= 0) or np.any(y < 0):
        raise ValueError("frequency axis must increase; measured spectrum must be finite and non-negative")
    response = np.asarray(instrument_response, dtype=float)
    if response.shape != (f.size, f.size) or np.any(~np.isfinite(response)) or np.any(response < 0) or np.any(response.sum(axis=0) <= 0):
        raise ValueError("instrument_response must be a non-negative square matrix with non-zero columns")
    response = response / response.sum(axis=0, keepdims=True)
    if signal_uncertainty is not None and spectral_covariance is not None:
        raise ValueError("supply signal_uncertainty or spectral_covariance, not both")
    if spectral_covariance is not None:
        cov_y = np.asarray(spectral_covariance, dtype=float)
        if cov_y.shape != (f.size, f.size) or np.any(~np.isfinite(cov_y)) or not np.allclose(cov_y, cov_y.T):
            raise ValueError("spectral_covariance must be a finite symmetric square matrix")
        try:
            chol = np.linalg.cholesky(cov_y)
        except np.linalg.LinAlgError as exc:
            raise ValueError("spectral_covariance must be positive definite") from exc
        whiten = lambda value: np.linalg.solve(chol, value)
    elif signal_uncertainty is not None:
        unc = np.asarray(signal_uncertainty, dtype=float)
        if unc.shape != f.shape or np.any(~np.isfinite(unc)) or np.any(unc <= 0):
            raise ValueError("signal_uncertainty must be positive and match measured_spectrum")
        whiten = lambda value: value / unc
    else:
        raise ValueError("signal_uncertainty or spectral_covariance is required for full-EM inversion")
    bounds = parameter_bounds or {}
    defaults = {
        "electron_density_m3": (1e17, 1e22),
        "electron_temperature_eV": (10.0, 2e4),
        "ion_temperature_eV": (1.0, 1e4),
    }
    names = tuple(defaults)
    initial = np.array([
        _positive(electron_density_initial_m3, "electron_density_initial_m3"),
        _positive(electron_temperature_initial_eV, "electron_temperature_initial_eV"),
        _positive(ion_temperature_initial_eV, "ion_temperature_initial_eV"),
    ])
    limits = np.array([bounds.get(name, defaults[name]) for name in names], dtype=float)
    if limits.shape != (3, 2) or np.any(~np.isfinite(limits)) or np.any(limits[:, 0] <= 0) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("parameter_bounds must contain positive increasing pairs for density and temperatures")
    if np.any(initial <= limits[:, 0]) or np.any(initial >= limits[:, 1]):
        raise ValueError("initial density and temperatures must lie strictly inside parameter_bounds")
    lower, upper = np.log10(limits[:, 0]), np.log10(limits[:, 1])
    p0 = np.log10(initial)

    def model(p):
        result = full_em_cts_spectral_density(
            f, probe_frequency_Hz, scattering_geometry,
            electron_density_m3=10.0**p[0],
            electron_temperature_eV=10.0**p[1],
            ion_temperature_eV=10.0**p[2],
            ion_species=ion_species, ion_fractions=ion_fractions,
            magnetic_field_T=magnetic_field_T,
            electron_flow_velocity_m_s=electron_flow_velocity_m_s,
            ion_flow_velocity_m_s=ion_flow_velocity_m_s,
            incident_polarization=incident_polarization,
            scattered_polarization=scattered_polarization,
            harmonic_order=harmonic_order, quadrature_order=quadrature_order,
            damping_rate_rad_s=damping_rate_rad_s,
            fast_ion_populations=fast_ion_populations,
        )["spectral_density"]
        widths = np.empty_like(f)
        widths[0] = 0.5 * (f[1] - f[0])
        widths[-1] = 0.5 * (f[-1] - f[-2])
        widths[1:-1] = 0.5 * (f[2:] - f[:-2])
        return (response @ (result * widths)) / widths

    def residual(p, return_profile=False):
        z = model(p)
        design = whiten(np.column_stack((z, np.ones_like(z))))
        target = whiten(y)
        amplitude, offset = np.linalg.lstsq(design, target, rcond=None)[0]
        if amplitude < 0:
            amplitude = 0.0
            one = whiten(np.ones_like(y))
            offset = float(one @ target / (one @ one))
        r = whiten(amplitude * z + offset - y)
        return (r, amplitude, offset, z) if return_profile else r

    fit = least_squares(residual, p0, bounds=(lower, upper), x_scale="jac", max_nfev=1200)
    if not fit.success:
        raise ValueError(f"full-EM CTS inversion failed: {fit.message}")
    rank = int(np.linalg.matrix_rank(fit.jac))
    condition = float(np.linalg.cond(fit.jac))
    if rank < 3 or not np.isfinite(condition) or condition > 1e12:
        raise ValueError("full-EM CTS fit rejected: density and temperatures are not jointly identifiable")
    r, amplitude, offset, convolved = residual(fit.x, return_profile=True)
    dof = max(1, f.size - 5)
    covariance_log = np.linalg.pinv(fit.jac.T @ fit.jac)
    if spectral_covariance is None and signal_uncertainty is None:
        covariance_log *= float(r @ r / dof)
    values = 10.0**fit.x
    transform = np.diag(np.log(10.0) * values)
    covariance = transform @ covariance_log @ transform.T
    forward_diagnostics = full_em_cts_spectral_density(
        f, probe_frequency_Hz, scattering_geometry,
        electron_density_m3=float(values[0]),
        electron_temperature_eV=float(values[1]),
        ion_temperature_eV=float(values[2]),
        ion_species=ion_species, ion_fractions=ion_fractions,
        magnetic_field_T=magnetic_field_T,
        electron_flow_velocity_m_s=electron_flow_velocity_m_s,
        ion_flow_velocity_m_s=ion_flow_velocity_m_s,
        incident_polarization=incident_polarization,
        scattered_polarization=scattered_polarization,
        harmonic_order=harmonic_order, quadrature_order=quadrature_order,
        damping_rate_rad_s=damping_rate_rad_s,
        fast_ion_populations=fast_ion_populations,
    )
    return {
        "electron_density_m3": float(values[0]),
        "electron_temperature_eV": float(values[1]),
        "ion_temperature_eV": float(values[2]),
        "parameter_names": list(names),
        "fit_covariance": covariance,
        "parameter_uncertainties": np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        "fitted_spectrum": amplitude * convolved + offset,
        "residual": y - (amplitude * convolved + offset),
        "profiled_amplitude": float(amplitude),
        "profiled_offset": float(offset),
        "reduced_chi_square": float(r @ r / dof),
        "jacobian_rank": rank,
        "jacobian_condition_number": condition,
        "optimizer_success": bool(fit.success),
        "model": "magnetized full-electromagnetic CTS with Bindslev low-temperature coupling and optional relativistic gyrotropic fast-ion populations",
        "external_inputs_fixed": [
            "magnetic_field_T", "electron_flow_velocity_m_s", "ion_flow_velocity_m_s",
            "ion_species", "ion_fractions", "incident_polarization", "scattered_polarization",
        ],
        "inversion_scope": "electron density, electron temperature and one common ion temperature; flow vectors and composition are not fitted",
        "fluctuation_covariance_density_current_E_B": forward_diagnostics["fluctuation_covariance_density_current_E_B"],
        "dielectric_tensor": forward_diagnostics["dielectric_tensor"],
        "wave_tensor": forward_diagnostics["wave_tensor"],
        "forward_model_diagnostics": forward_diagnostics,
    }
