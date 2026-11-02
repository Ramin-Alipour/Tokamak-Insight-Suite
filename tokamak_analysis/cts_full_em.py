"""Magnetized full-electromagnetic CTS fluctuation and transfer solver.

Thermal Maxwellian species use the magnetized kinetic response. Explicitly
supplied relativistic gyrotropic non-Maxwellian electron populations and fast-ion populations
use momentum-space Vlasov response, Klimontovich orbit noise and a second-order
relativistic Vlasov three-wave current vertex. Maxwellian bulk-electron
scattering retains the low-temperature Bindslev Gamma/X source-current
closure. Species-resolved ion temperatures are fitted independently.

Optional frequency-resolved cold-plasma ray tracing and local Gaussian beam
overlap are connected to the spectral transfer. The ray path is geometrical
optics with caller-supplied equilibrium and local beam covariance; it does
not include diffraction, mode conversion or propagated quasi-optical beams.
No empirical validation is performed by this solver.
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


def _real_array(value, name, ndim=None):
    """Validate real numeric API inputs without coercing strings or booleans."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values, not booleans, strings or complex values")
    if ndim is not None and raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    array = raw.astype(float, copy=False)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite non-empty real numeric values")
    return array


def _vec3(value, name):
    a = _real_array(value, name, 1)
    if a.shape != (3,) or not np.all(np.isfinite(a)) or np.linalg.norm(a) == 0:
        raise ValueError(f"{name} must be a finite non-zero 3-vector")
    return a / np.linalg.norm(a)


def _finite_vec3(value, name):
    a = _real_array(value, name, 1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return a


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a positive finite real scalar")
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
                  electron_flow_m_s, ion_flow_m_s=None):
    fractions = _real_array(ion_fractions, "ion_fractions", 1)
    if (not isinstance(ion_species, Sequence) or isinstance(ion_species, (str, bytes))
            or len(ion_species) == 0 or fractions.shape != (len(ion_species),)
            or np.any(~np.isfinite(fractions)) or np.any(fractions <= 0)
            or not np.isclose(fractions.sum(), 1.0, rtol=1e-6, atol=1e-8)):
        raise ValueError("ion_species and positive ion_fractions summing to one are required")
    names = [str(item.get("name", f"ion_{i}")) if isinstance(item, Mapping) else f"ion_{i}"
             for i, item in enumerate(ion_species)]
    if isinstance(ion_temperature_eV, Mapping):
        missing = [name for name in names if name not in ion_temperature_eV]
        extra = [name for name in ion_temperature_eV if name not in names]
        if missing or extra:
            raise ValueError(f"ion_temperature_eV mapping must contain exactly the ion species names; missing={missing}, extra={extra}")
        temperatures = {name: _positive(ion_temperature_eV[name], f"ion_temperature_eV[{name}]")
                        for name in names}
    elif np.isscalar(ion_temperature_eV):
        common = _positive(ion_temperature_eV, "ion_temperature_eV")
        temperatures = {name: common for name in names}
    else:
        values = _real_array(ion_temperature_eV, "ion_temperature_eV", 1)
        if values.shape != (len(names),) or np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("ion_temperature_eV must be a positive scalar, a species-name mapping, or one positive value per ion species")
        temperatures = dict(zip(names, map(float, values)))
    result = [{
        "name": "e-", "density": electron_density_m3, "temperature": electron_temperature_eV,
        "mass": ME, "charge": -E_CHARGE, "flow": _finite_vec3(electron_flow_m_s, "electron_flow_velocity_m_s")
    }]
    for i, item in enumerate(ion_species):
        if not isinstance(item, Mapping):
            raise ValueError("each ion species must be a mapping")
        name = names[i]
        z = _positive(item.get("charge_state"), f"ion_species[{i}].charge_state")
        mass = _positive(item.get("mass_amu"), f"ion_species[{i}].mass_amu") * AMU
        flow = np.zeros(3, dtype=float) if ion_flow_m_s is None else ion_flow_m_s.get(name)
        if flow is None:
            raise ValueError(f"ion_flow_velocity_m_s must provide a vector for {name}")
        result.append({
            "name": name, "density": fractions[i] * electron_density_m3 / z,
            "temperature": temperatures[name], "mass": mass,
            "charge": z * E_CHARGE, "flow": _finite_vec3(flow, f"ion flow for species {name}")
        })
    return result


def _gyro_basis(magnetic_field_T, k_vector):
    b = _real_array(magnetic_field_T, "magnetic_field_T", 1)
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
            measure = (vth ** 3 / 2.0) * wp * wz * f0_weight
            jx = np.pi * vperp * (response[n_h + 1] + response[n_h - 1])
            jy = np.pi * vperp * (response[n_h - 1] - response[n_h + 1]) / 1j
            jz = 2.0 * np.pi * vpar * response[n_h]
            sigma += charge * measure * np.vstack((jx, jy, jz))
    # Transform the conductivity from the B-aligned frame to laboratory axes.
    return rotation @ sigma @ rotation.T



def _select_flux_normalized_cold_mode(omega, direction, magnetic_field, species, requested_polarization):
    """Select the propagating cold-plasma branch matching a supplied polarization."""
    from .cts_em_geometry import solve_cold_plasma_propagating_modes

    candidates=[]
    requested=np.asarray(requested_polarization,dtype=complex)
    requested=requested/np.linalg.norm(requested)
    for branch in (0,1):
        try:
            mode=solve_cold_plasma_propagating_modes(
                omega,direction,magnetic_field,species,mode_index=branch
            )
        except ValueError:
            continue
        field=np.asarray(mode["electric_polarization_unit_flux"],dtype=complex)
        unit=field/np.linalg.norm(field)
        overlap=np.vdot(unit,requested)
        candidates.append((abs(overlap),branch,mode,overlap))
    if not candidates:
        raise ValueError("no real propagating cold-plasma mode exists for the supplied beam direction")
    candidates.sort(key=lambda item:item[0],reverse=True)
    overlap,branch,mode,phase=candidates[0]
    if overlap<0.95:
        raise ValueError(
            "supplied beam polarization is not a cold-plasma eigenmode for this direction; "
            "provide the physical O/X or extraordinary-mode polarization"
        )
    field=np.asarray(mode["electric_polarization_unit_flux"],dtype=complex)
    field=field*np.exp(-1j*np.angle(phase))
    return mode["wavevector_rad_m"],field,{
        "mode_index":branch,"refractive_index":mode["refractive_index"],
        "polarization_overlap":float(overlap),
        "normalized_longitudinal_poynting_flux_W_m2":float(np.dot(mode["poynting_vector_unit_flux"],np.asarray(direction,dtype=float))),
    }


def full_em_cts_spectral_density(
    frequency_shift_Hz, probe_frequency_Hz, scattering_geometry, *,
    electron_density_m3, electron_temperature_eV, ion_temperature_eV,
    ion_species, ion_fractions, magnetic_field_T,
    electron_flow_velocity_m_s=None, ion_flow_velocity_m_s=None,
    incident_polarization=None, scattered_polarization=None,
    electron_flow_projection_m_s=None, ion_flow_projection_m_s=None,
    harmonic_order=8, quadrature_order=12, damping_rate_rad_s=None,
    fast_ion_populations=None, electron_distribution=None,
    bindslev_validity_theta_e_limit=None, instrument_response_uncertainty=None,
    response_mc_samples=100,
):
    """Return a local full-EM magnetized CTS spectrum and fluctuation diagnostics.

    Geometry may be supplied as local incident/scattered unit vectors or as
    frequency_resolved_geometry records produced by trace_cts_frequency_resolved_beams.
    Polarizations are complex 3-vectors supplied by the diagnostic mode definition.
    Electron and per-ion-species flow projections can be supplied as scalar
    components along the scattering wavevector; these are not 3-D flow vectors.
    Fluctuation covariance includes electron density/current/E/B cross-correlations.
    """
    shifts = _real_array(frequency_shift_Hz, "frequency_shift_Hz", 1)
    if shifts.ndim != 1 or shifts.size == 0 or np.any(~np.isfinite(shifts)):
        raise ValueError("frequency_shift_Hz must be a finite non-empty vector")
    if np.any(np.diff(shifts) <= 0):
        raise ValueError("frequency_shift_Hz must be strictly increasing")
    f0 = _positive(probe_frequency_Hz, "probe_frequency_Hz")
    ne = _positive(electron_density_m3, "electron_density_m3")
    te = _positive(electron_temperature_eV, "electron_temperature_eV")
    theta_e = float(te / 510998.95069)
    if electron_distribution is None:
        if bindslev_validity_theta_e_limit is None:
            bindslev_validity_status = "not_quantitatively_screened"
        else:
            bindslev_limit = _positive(
                bindslev_validity_theta_e_limit, "bindslev_validity_theta_e_limit"
            )
            if bindslev_limit >= 1.0:
                raise ValueError("bindslev_validity_theta_e_limit must be a dimensionless positive value below 1")
            if theta_e > bindslev_limit:
                raise ValueError(
                    "Full-EM CTS rejected: electron theta_e exceeds the researcher-supplied "
                    "Bindslev low-temperature closure screening limit. Supply an explicit "
                    "relativistic electron distribution or revise the stated validity limit."
                )
            bindslev_validity_status = "within_researcher_supplied_asymptotic_screening_limit"
    else:
        if bindslev_validity_theta_e_limit is not None:
            raise ValueError("bindslev_validity_theta_e_limit applies only when the Maxwellian Bindslev closure is selected")
        bindslev_validity_status = "not_applicable_explicit_relativistic_electron_distribution"
    if ion_temperature_eV is None:
        raise ValueError("ion_temperature_eV must be supplied for every thermal ion species")
    if not isinstance(scattering_geometry, Mapping):
        raise ValueError("scattering_geometry must be a mapping")
    frequency_geometry = scattering_geometry.get("frequency_resolved_geometry")
    if frequency_geometry is not None and any(
        isinstance(item, Mapping) and "scattering_samples" in item
        for item in frequency_geometry
    ):
        if not isinstance(frequency_geometry, Sequence) or isinstance(frequency_geometry, (str, bytes)) or len(frequency_geometry) != shifts.size:
            raise ValueError("scattering-volume geometry must contain one frequency record per measured spectral bin")
        if "frequency_shift_Hz" in scattering_geometry:
            geometry_shifts = _real_array(scattering_geometry["frequency_shift_Hz"], "scattering_geometry.frequency_shift_Hz", 1)
            if geometry_shifts.shape != shifts.shape or not np.allclose(geometry_shifts, shifts, rtol=1e-10, atol=1e-6):
                raise ValueError("scattering-volume ray geometry must use the same frequency_shift_Hz axis as the measured spectrum")
        integrated_spectrum = np.zeros(shifts.size, dtype=float)
        diagnostic_keys = (
            "fluctuation_covariance_density_current_E_B", "dielectric_tensor",
            "wave_tensor", "bindslev_source_current_operator",
            "total_three_wave_source_current_operator", "bindslev_dielectric_form_factors",
            "bindslev_X_tensor", "bindslev_Gamma_inverse_tensor",
        )
        integrated_diagnostics = {key: [] for key in diagnostic_keys}
        integrated_locations = []
        integrated_fields = []
        integrated_sample_diagnostics = []
        integrated_coefficients = []
        integrated_fast_kinetics = []
        integrated_vertices = []
        first_result = None
        for frequency_index, (shift, frequency_record) in enumerate(zip(shifts, frequency_geometry)):
            if not isinstance(frequency_record, Mapping) or "scattering_samples" not in frequency_record:
                raise ValueError(f"frequency geometry record {frequency_index} is missing scattering_samples")
            samples = frequency_record["scattering_samples"]
            if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)) or len(samples) == 0:
                raise ValueError(f"scattering_samples[{frequency_index}] must be a non-empty sequence")
            if any(not isinstance(sample, Mapping) for sample in samples):
                raise ValueError(f"scattering_samples[{frequency_index}] entries must be mappings")
            sample_weights = np.asarray([sample.get("sample_weight", np.nan) for sample in samples], dtype=float)
            if (np.any(~np.isfinite(sample_weights)) or np.any(sample_weights < 0)
                    or not np.isclose(np.sum(sample_weights), 1.0, rtol=1e-8, atol=1e-10)):
                raise ValueError(f"scattering_samples[{frequency_index}] weights must be finite, non-negative and sum to one")
            local_spectrum = 0.0
            local_diagnostics = {key: None for key in diagnostic_keys}
            local_location = np.zeros(3, dtype=float)
            local_field = np.zeros(3, dtype=float)
            local_sample_results = []
            local_coefficients = []
            local_fast_kinetics = []
            local_vertices = []
            for sample_index, (sample, sample_weight) in enumerate(zip(samples, sample_weights)):
                if not isinstance(sample, Mapping):
                    raise ValueError(f"scattering_samples[{frequency_index}][{sample_index}] must be a mapping")
                sample_geometry = {
                    "frequency_resolved_geometry": [{
                        "incident_direction": sample.get("incident_direction"),
                        "scattered_direction": sample.get("scattered_direction"),
                        "magnetic_field_T": sample.get("magnetic_field_T"),
                        "scattering_location_m": sample.get("scattering_location_m"),
                        "beam_overlap_relative": 1.0,
                    }]
                }
                sample_result = full_em_cts_spectral_density(
                    [float(shift)], f0, sample_geometry,
                    electron_density_m3=ne, electron_temperature_eV=te,
                    ion_temperature_eV=ion_temperature_eV,
                    ion_species=ion_species, ion_fractions=ion_fractions,
                    magnetic_field_T=sample.get("magnetic_field_T"),
                    electron_flow_velocity_m_s=electron_flow_velocity_m_s,
                    electron_flow_projection_m_s=electron_flow_projection_m_s,
                    ion_flow_velocity_m_s=ion_flow_velocity_m_s,
                    ion_flow_projection_m_s=ion_flow_projection_m_s,
                    incident_polarization=incident_polarization,
                    scattered_polarization=scattered_polarization,
                    harmonic_order=harmonic_order, quadrature_order=quadrature_order,
                    damping_rate_rad_s=damping_rate_rad_s,
                    fast_ion_populations=fast_ion_populations,
                    electron_distribution=electron_distribution,
                )
                local_spectrum += float(sample_weight)*float(sample_result["spectral_density"][0])
                local_location += float(sample_weight)*_finite_vec3(sample["scattering_location_m"], "scattering sample location")
                local_field += float(sample_weight)*_finite_vec3(sample["magnetic_field_T"], "scattering sample magnetic field")
                for key in diagnostic_keys:
                    value = np.asarray(sample_result[key][0])
                    if local_diagnostics[key] is None:
                        local_diagnostics[key] = float(sample_weight)*value
                    else:
                        local_diagnostics[key] = local_diagnostics[key]+float(sample_weight)*value
                local_sample_results.append({
                    "sample_index": sample_index,
                    "sample_weight": float(sample_weight),
                    "scattering_location_m": np.asarray(sample["scattering_location_m"], dtype=float),
                    "magnetic_field_T": np.asarray(sample["magnetic_field_T"], dtype=float),
                    "ray_separation_m": sample.get("ray_separation_m"),
                    "spectral_density": float(sample_result["spectral_density"][0]),
                })
                local_coefficients.append({
                    "sample_weight": float(sample_weight),
                    "coefficients": sample_result["bindslev_coupling_coefficients"][0],
                })
                local_fast_kinetics.append(sample_result["fast_ion_kinetic_diagnostics"])
                local_vertices.append(sample_result["relativistic_three_wave_vertex_diagnostics"])
                if first_result is None:
                    first_result = sample_result
            path_weight = _positive(frequency_record.get("beam_overlap_relative"), "beam_overlap_relative")
            integrated_spectrum[frequency_index] = path_weight*local_spectrum
            for key in diagnostic_keys:
                integrated_diagnostics[key].append(local_diagnostics[key])
            integrated_locations.append(local_location)
            integrated_fields.append(local_field)
            integrated_sample_diagnostics.append(local_sample_results)
            integrated_coefficients.append({"scattering_volume_samples": local_coefficients})
            integrated_fast_kinetics.append(local_fast_kinetics)
            integrated_vertices.append(local_vertices)
        result = dict(first_result)
        result.update({
            "frequency_shift_Hz": shifts,
            "spectral_density": integrated_spectrum,
            **{key: np.asarray(values) for key, values in integrated_diagnostics.items()},
            "bindslev_coupling_coefficients": integrated_coefficients,
            "fast_ion_kinetic_diagnostics": integrated_fast_kinetics,
            "kinetic_population_diagnostics": integrated_fast_kinetics,
            "relativistic_three_wave_vertex_diagnostics": integrated_vertices,
            "relativistic_kinetic_three_wave_vertex_used": any(
                len(vertex_group) > 0
                for frequency_samples in integrated_vertices
                for sample_diagnostics in frequency_samples
                for vertex_group in sample_diagnostics
            ),
            "frequency_resolved_magnetic_field_T": np.asarray(integrated_fields),
            "scattering_locations_m": np.asarray(integrated_locations),
            "relative_gaussian_beam_overlap": np.asarray([
                item["beam_overlap_relative"] for item in frequency_geometry
            ], dtype=float),
            "ray_tracing_geometry_used": True,
            "scattering_volume_integration_used": True,
            "scattering_volume_sample_count": np.asarray([
                len(item["scattering_samples"]) for item in frequency_geometry
            ], dtype=int),
            "scattering_volume_sample_diagnostics": integrated_sample_diagnostics,
            "scattering_volume_weighting": "normalized local Gaussian beam overlap times incident-ray path-length quadrature",
            "magnetic_field_T": np.mean(np.asarray(integrated_fields), axis=0),
            "limitations": list(first_result.get("limitations", []))+[
                "Scattering-volume integration uses local Gaussian overlap samples and assumes fitted ne, Te and thermal-ion temperatures are uniform across the volume.",
                "Absolute beam-overlap path integrals are used only as relative spectral-bin weights; absolute launcher/receiver power coupling is not normalized."
            ],
        })
        result["relativistic_fast_ion_three_wave_vertex_used"] = any(
            item.get("population_role") == "fast ion"
            for frequency_samples in integrated_vertices
            for sample_diagnostics in frequency_samples
            for vertex_group in sample_diagnostics
            for item in vertex_group
        )
        return result
    if frequency_geometry is None:
        ki_hat = _vec3(scattering_geometry.get("incident_direction"), "incident_direction")
        ks_hat = _vec3(scattering_geometry.get("scattered_direction"), "scattered_direction")
    else:
        if not isinstance(frequency_geometry, Sequence) or isinstance(frequency_geometry, (str, bytes)) or len(frequency_geometry) != shifts.size:
            raise ValueError("frequency_resolved_geometry must contain one traced geometry record per frequency bin")
        if not isinstance(frequency_geometry[0], Mapping):
            raise ValueError("frequency_resolved_geometry entries must be mappings")
        if "frequency_shift_Hz" in scattering_geometry:
            geometry_shifts = np.asarray(scattering_geometry["frequency_shift_Hz"], dtype=float)
            if geometry_shifts.shape != shifts.shape or not np.allclose(geometry_shifts, shifts, rtol=1e-10, atol=1e-6):
                raise ValueError("ray-traced geometry must use the same frequency_shift_Hz axis as the measured spectrum")
        ki_hat = _vec3(frequency_geometry[0].get("incident_direction"), "first frequency-resolved incident direction")
        ks_hat = _vec3(frequency_geometry[0].get("scattered_direction"), "first frequency-resolved scattered direction")
    b = np.asarray(magnetic_field_T, dtype=float)
    if b.shape != (3,) or np.any(~np.isfinite(b)) or np.linalg.norm(b) <= 0:
        raise ValueError("magnetic_field_T must be a finite non-zero 3-vector")
    if frequency_geometry is not None:
        if not isinstance(frequency_geometry, Sequence) or isinstance(frequency_geometry, (str, bytes)) or len(frequency_geometry) != shifts.size:
            raise ValueError("frequency_resolved_geometry must contain one traced geometry record per frequency bin")
        for geometry_index, item in enumerate(frequency_geometry):
            if not isinstance(item, Mapping):
                raise ValueError(f"frequency_resolved_geometry[{geometry_index}] must be a mapping")
            _vec3(item.get("incident_direction"), f"frequency_resolved_geometry[{geometry_index}].incident_direction")
            _vec3(item.get("scattered_direction"), f"frequency_resolved_geometry[{geometry_index}].scattered_direction")
            local_b = np.asarray(item.get("magnetic_field_T"), dtype=float)
            if local_b.shape != (3,) or np.any(~np.isfinite(local_b)) or np.linalg.norm(local_b) <= 0:
                raise ValueError(f"frequency_resolved_geometry[{geometry_index}].magnetic_field_T must be a finite non-zero 3-vector")
            overlap = float(item.get("beam_overlap_relative"))
            if not np.isfinite(overlap) or overlap < 0.0 or overlap > 1.0 + 1e-10:
                raise ValueError("relative Gaussian beam-overlap weights must be finite and lie in [0, 1]")
    if electron_flow_projection_m_s is not None:
        if electron_flow_velocity_m_s is not None:
            raise ValueError(
                "supply electron_flow_projection_m_s or electron_flow_velocity_m_s, not both"
            )
        electron_flow_projection_value = _real_array(
            electron_flow_projection_m_s, "electron_flow_projection_m_s"
        )
        if electron_flow_projection_value.ndim != 0:
            raise ValueError("electron_flow_projection_m_s must be a scalar")
        electron_flow_projection_m_s = float(electron_flow_projection_value)
        if not np.isfinite(electron_flow_projection_m_s):
            raise ValueError("electron_flow_projection_m_s must be finite")
        electron_flow_seed = np.zeros(3, dtype=float)
    else:
        if electron_flow_velocity_m_s is None:
            raise ValueError(
                "electron_flow_projection_m_s is required when no external electron flow vector is supplied"
            )
        electron_flow_seed = _finite_vec3(
            electron_flow_velocity_m_s, "electron_flow_velocity_m_s"
        )
    ion_names = [str(item.get("name", f"ion_{i}")) for i, item in enumerate(ion_species)]
    if ion_flow_projection_m_s is not None:
        if ion_flow_velocity_m_s is not None:
            raise ValueError("supply ion_flow_projection_m_s or ion_flow_velocity_m_s, not both")
        if not isinstance(ion_flow_projection_m_s, Mapping):
            raise ValueError("ion_flow_projection_m_s must map every ion species name to a scalar projection")
        missing = [name for name in ion_names if name not in ion_flow_projection_m_s]
        extra = [name for name in ion_flow_projection_m_s if name not in ion_names]
        if missing or extra:
            raise ValueError(
                f"ion_flow_projection_m_s must contain exactly the ion species names; missing={missing}, extra={extra}"
            )
        ion_flow_projections = {}
        for name in ion_names:
            projection = _real_array(
                ion_flow_projection_m_s[name], f"ion_flow_projection_m_s[{name}]"
            )
            if projection.ndim != 0:
                raise ValueError(f"ion_flow_projection_m_s[{name}] must be a scalar")
            ion_flow_projections[name] = float(projection)
        if any(not np.isfinite(value) for value in ion_flow_projections.values()):
            raise ValueError("ion-flow projections must be finite")
        ion_flow_vectors = None
    else:
        if not isinstance(ion_flow_velocity_m_s, Mapping):
            raise ValueError(
                "ion_flow_projection_m_s is required when ion flow is to be inferred from the measured spectrum"
            )
        missing = [name for name in ion_names if name not in ion_flow_velocity_m_s]
        extra = [name for name in ion_flow_velocity_m_s if name not in ion_names]
        if missing or extra:
            raise ValueError(
                f"ion_flow_velocity_m_s must contain exactly the ion species names; missing={missing}, extra={extra}"
            )
        ion_flow_projections = None
        ion_flow_vectors = ion_flow_velocity_m_s
    species = _species_list(ne, ion_species, ion_fractions, te, ion_temperature_eV,
                            electron_flow_seed, ion_flow_vectors)
    electron_distribution_used = None
    if electron_distribution is not None:
        if electron_flow_projection_m_s is not None and abs(electron_flow_projection_m_s) > 0.0:
            raise ValueError(
                "Full-EM electron-flow inference with an explicit relativistic electron distribution "
                "requires a drifted relativistic VDF model; this solver does not silently approximate "
                "that distribution by a frequency shift"
            )
        if not isinstance(electron_distribution, Mapping):
            raise ValueError("electron_distribution must be an explicit relativistic gyrotropic distribution mapping")
        if electron_distribution.get("kind") not in {"relativistic_maxwell_juttner", "relativistic_bigaussian", "tabulated_gyrotropic"}:
            raise ValueError("electron_distribution kind must be relativistic_maxwell_juttner, relativistic_bigaussian or tabulated_gyrotropic")
        electron_distribution_used = dict(electron_distribution)
        if electron_distribution_used["kind"] == "relativistic_maxwell_juttner":
            distribution_temperature = _positive(electron_distribution_used.get("temperature_eV", te), "electron_distribution.temperature_eV")
            if not np.isclose(distribution_temperature, te, rtol=1e-8, atol=0.0):
                raise ValueError("relativistic Maxwell-Juttner electron distribution temperature must equal electron_temperature_eV")
            electron_distribution_used["temperature_eV"] = te
        species[0].update({
            "distribution": electron_distribution_used,
            "fast_ion": True,
            "nonthermal_electron": True,
        })
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
    # In an anisotropic magnetized plasma, cold-plasma eigenmodes can have a longitudinal E component.
    # These vectors are branch-selection seeds; the solver replaces them with flux-normalized eigenmodes.
    cold_wave_species = [{
        "density_m3": sp["density"], "charge_C": sp["charge"], "mass_kg": sp["mass"]
    } for sp in species]
    if frequency_geometry is None:
        incident_wavevector, incident_mode_field, incident_mode_diagnostics = _select_flux_normalized_cold_mode(
            2.0*np.pi*f0, ki_hat, b, cold_wave_species, ei
        )
        ei = incident_mode_field
    else:
        incident_wavevector, incident_mode_diagnostics = None, None
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
    total_three_wave_operators = []
    bindslev_coefficients = []
    bindslev_form_factors = []
    bindslev_x_tensors = []
    bindslev_gamma_inverse = []
    fast_ion_kinetic_diagnostics = []
    relativistic_three_wave_vertex_diagnostics = []
    cold_mode_diagnostics = []
    local_magnetic_fields = []
    beam_overlap_weights = []
    for index, shift in enumerate(shifts):
        fs = f0 + shift
        if frequency_geometry is None:
            ki_hat_bin, ks_hat_bin, b_bin = ki_hat, ks_hat, b
            overlap_weight = 1.0
            incident_wavevector_bin, ei_bin = incident_wavevector, ei
            incident_diag_bin = incident_mode_diagnostics
        else:
            geometry_bin = frequency_geometry[index]
            ki_hat_bin = _vec3(geometry_bin["incident_direction"], "frequency-resolved incident direction")
            ks_hat_bin = _vec3(geometry_bin["scattered_direction"], "frequency-resolved scattered direction")
            b_bin = np.asarray(geometry_bin["magnetic_field_T"], dtype=float)
            overlap_weight = float(geometry_bin["beam_overlap_relative"])
            if not np.isfinite(overlap_weight) or overlap_weight < 0.0 or overlap_weight > 1.0 + 1e-10:
                raise ValueError("beam_overlap_relative must be finite and lie in [0, 1]")
            incident_wavevector_bin, ei_bin, incident_diag_bin = _select_flux_normalized_cold_mode(
                2.0*np.pi*f0, ki_hat_bin, b_bin, cold_wave_species, ei
            )
        scattered_wavevector, scattered_mode_field, scattered_mode_diagnostics = _select_flux_normalized_cold_mode(
            2.0*np.pi*fs, ks_hat_bin, b_bin, cold_wave_species, es
        )
        ki = incident_wavevector_bin
        ks = scattered_wavevector
        ei_local = ei_bin
        es_local = scattered_mode_field
        local_magnetic_fields.append(b_bin.copy())
        beam_overlap_weights.append(overlap_weight)
        cold_mode_diagnostics.append({
            "incident": incident_diag_bin,
            "scattered": scattered_mode_diagnostics,
        })
        kvec = ks - ki
        kvec_norm_for_flow = float(np.linalg.norm(kvec))
        if (electron_flow_projection_m_s is not None or ion_flow_projections is not None):
            if not np.isfinite(kvec_norm_for_flow) or kvec_norm_for_flow <= 0.0:
                raise ValueError("scattering wavevector must be non-zero to infer flow projections")
            khat_flow = kvec / kvec_norm_for_flow
            if electron_flow_projection_m_s is not None:
                # One geometry identifies only the projection along k, not a 3-D vector.
                species[0]["flow"] = electron_flow_projection_m_s * khat_flow
            if ion_flow_projections is not None:
                for ion_index, ion_name in enumerate(ion_names, start=1):
                    species[ion_index]["flow"] = ion_flow_projections[ion_name] * khat_flow
        omega = 2.0 * np.pi * shift
        if abs(omega) < damping:
            omega = np.copysign(damping, omega if omega else 1.0)
        sigmas = []
        fast_bare_covariances = {}
        fast_kinetic_at_frequency = []
        for species_index, sp in enumerate(species):
            if sp.get("fast_ion", False):
                from .cts_relativistic import relativistic_gyrotropic_species_response
                kinetic = relativistic_gyrotropic_species_response(
                    omega, kvec, b_bin, density_m3=sp["density"], mass_kg=sp["mass"],
                    charge_C=sp["charge"], distribution=sp["distribution"],
                    quadrature_order=int(quadrature_order),
                    gyrophase_points=max(16, 4*int(harmonic_order)),
                    resonance_width_rad_s=damping,
                )
                sigmas.append(kinetic["conductivity_S_m"])
                fast_bare_covariances[species_index] = kinetic["bare_current_spectral_covariance"]
                fast_kinetic_at_frequency.append({"name": sp["name"], "distribution_kind": kinetic["distribution_kind"], "relativistic_kinematics": kinetic["relativistic_kinematics"], "velocity_max_fraction_c": kinetic["velocity_max_fraction_c"], "bare_current_spectral_covariance": kinetic["bare_current_spectral_covariance"]})
            else:
                sigmas.append(_species_conductivity(
                    omega, kvec, b_bin, sp, harmonic_order=int(harmonic_order),
                    quadrature_order=int(quadrature_order),
                    damping_rate_rad_s=damping
                ))
        fast_ion_kinetic_diagnostics.append(fast_kinetic_at_frequency)
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
        from .cts_bindslev import (bindslev_source_current_operator, bindslev_correlated_transfer_spectrum, bindslev_dielectric_form_factors)
        from .cts_em_geometry import cold_plasma_dielectric_tensor
        electron_cold = [{
            "density_m3": ne, "charge_C": -E_CHARGE, "mass_kg": ME
        }]
        omega_i = 2.0 * np.pi * f0
        omega_s = 2.0 * np.pi * fs
        chi_i = cold_plasma_dielectric_tensor(omega_i, b_bin, electron_cold) - np.eye(3)
        chi_s = cold_plasma_dielectric_tensor(omega_s, b_bin, electron_cold) - np.eye(3)
        source = bindslev_source_current_operator(
            omega_incident_rad_s=omega_i,
            omega_scattered_rad_s=omega_s,
            wavevector_incident_rad_m=ki,
            wavevector_scattered_rad_m=ks,
            incident_electric_field=ei_local,
            incident_susceptibility=chi_i,
            scattered_susceptibility=chi_s,
            magnetic_field_T=b_bin,
            electron_density_m3=ne,
        )
        # Electron particle flux is j_e/q_e.  This exact unit conversion is
        # required before contracting the Bindslev velocity/flux terms.
        to_flux = np.eye(10, dtype=complex)
        to_flux[1:4, 1:4] /= -E_CHARGE
        covariance_n_v_E_B = to_flux @ state_cov @ to_flux.conj().T
        total_source_operator = (
            np.zeros_like(source["source_current_operator"], dtype=complex)
            if electron_distribution is not None
            else np.asarray(source["source_current_operator"], dtype=complex).copy()
        )
        relativistic_vertex_at_frequency = []
        for fast_species in species:
            if not fast_species.get("fast_ion", False):
                continue
            from .cts_relativistic import relativistic_gyrotropic_three_wave_vertex
            vertex = relativistic_gyrotropic_three_wave_vertex(
                omega_i, ki, ei_local,
                omega, kvec, b_bin,
                density_m3=fast_species["density"], mass_kg=fast_species["mass"],
                charge_C=fast_species["charge"], distribution=fast_species["distribution"],
                quadrature_order=int(quadrature_order),
                gyrophase_points=max(16, 4*int(harmonic_order)),
                resonance_width_rad_s=damping,
            )
            # The relativistic kinetic vertex is a map from the local fluctuation
            # electric field to J^(2); embed it in the shared [n,j,E,B] state
            # so all measured cross-correlations are retained in the contraction.
            fast_operator = np.zeros((3, 10), dtype=complex)
            fast_operator[:, 4:7] = vertex["current_vertex_A_m2_per_V_m"]
            total_source_operator += fast_operator
            relativistic_vertex_at_frequency.append({
                "name": fast_species["name"],
                "population_role": (
                    "relativistic Maxwell-Juttner electron"
                    if fast_species.get("nonthermal_electron") and fast_species["distribution"].get("kind") == "relativistic_maxwell_juttner"
                    else "non-Maxwellian electron" if fast_species.get("nonthermal_electron")
                    else "fast ion"
                ),
                "current_vertex_A_m2_per_V_m": vertex["current_vertex_A_m2_per_V_m"],
                "distribution_kind": vertex["distribution_kind"],
                "second_order_vlasov": vertex["second_order_vlasov"],
            })
        transfer = bindslev_correlated_transfer_spectrum(
            covariance_n_v_E_B, total_source_operator, es_local
        )
        form_factors = bindslev_dielectric_form_factors(
            source["source_current_operator"], ei_local, es_local, ki, ks,
            omega_i, omega_s, ne,
        )
        spectra[index] = overlap_weight * transfer["received_source_current_spectral_power"]
        bindslev_operators.append(source["source_current_operator"])
        total_three_wave_operators.append(total_source_operator)
        bindslev_form_factors.append(form_factors["G_alpha_beta"])
        bindslev_x_tensors.append(source["X_tensor"])
        bindslev_gamma_inverse.append(source["Gamma_inverse_tensor"])
        relativistic_three_wave_vertex_diagnostics.append(relativistic_vertex_at_frequency)
        projected = es_local.conj() @ source["source_current_operator"]
        total_projected = es_local.conj() @ total_source_operator
        bindslev_coefficients.append({
            "N": projected[0],
            "U_k": projected[1:4],
            "E_k": projected[4:7],
            "B_k": projected[7:10],
            "normalized_incident_flux_S_i": form_factors["normalized_incident_flux_S_i"],
            "normalized_scattered_flux_S_s": form_factors["normalized_scattered_flux_S_s"],
            "transfer_contributions": transfer["transfer_contributions"],
            "total_three_wave_receiver_projection": total_projected,
        })
        harmonic_counts.append(int(harmonic_order))
    return {
        "frequency_shift_Hz": shifts,
        "spectral_density": spectra,
        "fluctuation_covariance_density_current_E_B": component_covariances,
        "dielectric_tensor": dielectric_tensors,
        "wave_tensor": wave_tensors,
        "bindslev_source_current_operator": np.asarray(bindslev_operators),
        "total_three_wave_source_current_operator": np.asarray(total_three_wave_operators),
        "bindslev_coupling_coefficients": bindslev_coefficients,
        "relativistic_three_wave_vertex_diagnostics": relativistic_three_wave_vertex_diagnostics,
        "relativistic_fast_ion_three_wave_vertex_used": any(
            any(item.get("population_role") == "fast ion" for item in frequency_items)
            for frequency_items in relativistic_three_wave_vertex_diagnostics
        ),
        "relativistic_kinetic_three_wave_vertex_used": any(relativistic_three_wave_vertex_diagnostics),
        "kinetic_population_diagnostics": fast_ion_kinetic_diagnostics,
        "nonthermal_electron_distribution": electron_distribution_used,
        "relativistic_electron_distribution": electron_distribution_used,
        "bindslev_dielectric_form_factors": np.asarray(bindslev_form_factors),
        "bindslev_X_tensor": np.asarray(bindslev_x_tensors),
        "bindslev_Gamma_inverse_tensor": np.asarray(bindslev_gamma_inverse),
        "fast_ion_kinetic_diagnostics": fast_ion_kinetic_diagnostics,
        "three_wave_coupling": (
            "Relativistic second-order gyrotropic Vlasov vertices for supplied relativistic electrons/fast ions; Maxwellian bulk-electron Bindslev vertex replaced"
            if electron_distribution is not None
            else "Bindslev low-temperature bulk-electron source-current tensor plus relativistic second-order gyrotropic Vlasov vertices for supplied fast-ion populations"
            if any(relativistic_three_wave_vertex_diagnostics)
            else "Bindslev low-temperature magnetized source-current tensor, including Gamma^{-1}, X and all n/v/E/B cross terms"
        ),
        "magnetic_field_T": b,
        "frequency_resolved_magnetic_field_T": np.asarray(local_magnetic_fields),
        "relative_gaussian_beam_overlap": np.asarray(beam_overlap_weights),
        "ray_tracing_geometry_used": frequency_geometry is not None,
        "scattering_volume_integration_used": False,
        "scattering_volume_weighting": "single local scattering point; no path-integrated scattering-volume quadrature",
        "scattering_locations_m": (np.asarray([item["scattering_location_m"] for item in frequency_geometry])
                                   if frequency_geometry is not None and all("scattering_location_m" in item for item in frequency_geometry)
                                   else None),
        "fast_ion_populations": fast_ion_populations or [],
        "model": ("magnetized full-electromagnetic CTS with relativistic gyrotropic non-Maxwellian electron and fast-ion response and second-order kinetic vertices"
                  if electron_distribution is not None
                  else "magnetized full-electromagnetic CTS with Bindslev low-temperature bulk-electron coupling and optional relativistic gyrotropic fast-ion kinetics"),
        "kinetic_response": "Maxwellian thermal species: magnetized Vlasov gyro-harmonics; explicitly supplied relativistic Maxwell-Juttner, bi-Gaussian or tabulated gyrotropic electrons/fast ions: relativistic momentum-space Vlasov Fourier collocation",
        "fluctuation_source": "Maxwellian thermal species: classical FDT; explicitly supplied relativistic electron/fast-ion populations: relativistic Klimontovich orbit shot-noise covariance",
        "retained_cross_correlations": ["electron density-current", "electron density-E", "electron density-B",
            "current-E", "current-B", "E-B"],
        "numerical_controls": {"harmonic_order": int(harmonic_order),
            "quadrature_order": int(quadrature_order), "damping_rate_rad_s": damping,
            "damping_rate_source": ("caller_supplied" if damping_rate_rad_s is not None else "documented_numerical_default_max_2pi_probe_frequency_times_1e-7_or_1_rad_per_s")},
        "electron_thermal_relativistic_parameter": theta_e,
        "bulk_electron_closure": ("explicit_relativistic_gyrotropic_distribution" if electron_distribution is not None else "Maxwellian_Bindslev_low_temperature_closure"),
        "bulk_electron_closure_validity_status": bindslev_validity_status,
        "bulk_electron_closure_validity_theta_e_limit": (
            None if electron_distribution is not None or bindslev_validity_theta_e_limit is None
            else float(bindslev_validity_theta_e_limit)
        ),
        "bulk_electron_closure_validity_note": (
            "Bindslev is an asymptotic low-temperature closure. theta_e is reported; no universal numerical cutoff is assumed. Supply bindslev_validity_theta_e_limit to enforce a researcher-selected screening threshold."
            if electron_distribution is None and bindslev_validity_theta_e_limit is None
            else "The fitted theta_e is within the researcher-supplied Bindslev asymptotic screening limit."
            if electron_distribution is None
            else "The explicit electron distribution replaces the Maxwellian Bindslev closure."
        ),
        "limitations": [
            *( [] if frequency_geometry is not None else ["Homogeneous local geometry; supply trace_cts_frequency_resolved_beams output to include frequency-resolved ray paths and Gaussian overlap."] ),
            "Thermal ions and Maxwellian electrons use the non-relativistic isotropic Maxwellian solver; explicitly supplied electron distributions may use relativistic Maxwell-Juttner, bi-Gaussian or tabulated gyrotropic momentum distributions, and fast ions use bi-Gaussian or tabulated distributions.",
            "When electron_distribution is supplied, the Maxwellian electron response and Bindslev electron nonlinear source are replaced by a relativistic gyrotropic electron Vlasov response and second-order kinetic vertex; supplied fast-ion populations add their own relativistic vertices.",
            "Classical FDT is applied only to bulk thermal Maxwellian species; explicitly supplied relativistic electron/fast-ion populations use the Klimontovich orbit spectrum."
        ],
    }



def _apply_frequency_response(response, spectrum, frequency):
    """Apply a bin-probability response while conserving integrated spectral power."""
    f = _real_array(frequency, "frequency_shift_Hz", 1)
    y = _real_array(spectrum, "spectral_density", 1)
    r = _real_array(response, "instrument_response", 2)
    if f.size < 2 or y.shape != f.shape or r.shape != (f.size, f.size):
        raise ValueError("response, spectrum and frequency axis must share a square frequency grid")
    if np.any(r < 0.0) or np.any(r.sum(axis=0) <= 0.0):
        raise ValueError("instrument_response must be non-negative with non-zero columns")
    if not np.allclose(r.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("measured instrument_response columns must already sum to one; the calibrated response operator is not renormalized")
    edges = np.empty(f.size + 1, dtype=float)
    edges[1:-1] = 0.5*(f[:-1] + f[1:])
    edges[0] = f[0] - 0.5*(f[1]-f[0])
    edges[-1] = f[-1] + 0.5*(f[-1]-f[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise ValueError("frequency axis must define positive bin widths")
    return (r @ (y*widths))/widths


def analyze_full_em_cts_spectrum(
    frequency_shift_Hz, measured_spectrum, *, probe_frequency_Hz,
    scattering_geometry, electron_density_initial_m3=None,
    electron_temperature_initial_eV=None, ion_temperature_initial_eV=None,
    ion_species, ion_fractions, magnetic_field_T,
    incident_polarization, scattered_polarization, ion_flow_projection_initial_m_s=None,
    instrument_response, electron_flow_projection_initial_m_s=None,
    signal_uncertainty=None, spectral_covariance=None,
    parameter_bounds=None, harmonic_order=8, quadrature_order=12,
    damping_rate_rad_s=None, fast_ion_populations=None, electron_distribution=None,
    bindslev_validity_theta_e_limit=None, instrument_response_uncertainty=None,
    response_mc_samples=100, external_input_covariance=None,
    external_input_parameter_order=None, external_input_mc_samples=100,
    external_input_mc_seed=84622,
    numerical_convergence=True, convergence_harmonic_orders=None,
    convergence_quadrature_orders=None, convergence_relative_tolerance=1e-3,
):
    """Fit n_e, T_e, electron flow, and each supplied ion species' T_i and flow projection.

    The instrument amplitude and constant background are profiled as nuisance
    parameters. The electron-flow component along the local scattering
    wavevector is fitted from the measured spectrum; transverse components
    are not identifiable from a single geometry. Ion composition, magnetic
    field, geometry and mode polarizations remain external inputs. The flow
    projection of every supplied ion species is independently fitted
    from the measured spectrum; transverse components are not identifiable
    from a single CTS geometry. Each species temperature is also fitted.
    """
    from scipy.optimize import least_squares
    from tokamak_analysis.cts_analysis import estimate_cts_optimizer_initial_parameters

    external_order = None
    external_covariance = None
    external_draw_vectors = None
    if (external_input_covariance is None) != (external_input_parameter_order is None):
        raise ValueError(
            "external_input_covariance and external_input_parameter_order must be supplied together"
        )
    if external_input_covariance is not None:
        from .cts_external_uncertainty import (
            validate_external_input_covariance, sample_external_input_vectors,
        )
        if not isinstance(scattering_geometry, Mapping) or "frequency_resolved_geometry" in scattering_geometry:
            raise ValueError(
                "external_input_covariance currently supports local incident/scattered geometry only; "
                "frequency-resolved ray/equilibrium covariance is not represented by this input contract"
            )
        external_order, external_covariance = validate_external_input_covariance(
            external_input_covariance, external_input_parameter_order
        )
        external_base = {
            "magnetic_field_T_x": float(np.asarray(magnetic_field_T, dtype=float)[0]),
            "magnetic_field_T_y": float(np.asarray(magnetic_field_T, dtype=float)[1]),
            "magnetic_field_T_z": float(np.asarray(magnetic_field_T, dtype=float)[2]),
        }
        for kind, vector in (
            ("incident_direction", scattering_geometry.get("incident_direction")),
            ("scattered_direction", scattering_geometry.get("scattered_direction")),
        ):
            vec = _vec3(vector, kind)
            for axis, value in zip("xyz", vec):
                external_base[f"{kind}_{axis}"] = float(value)
        for kind, vector in (
            ("incident", incident_polarization),
            ("scattered", scattered_polarization),
        ):
            vec = np.asarray(vector, dtype=complex)
            if vec.shape != (3,) or not np.all(np.isfinite(vec)):
                raise ValueError(f"{kind}_polarization must be a finite complex 3-vector")
            for axis, value in zip("xyz", vec):
                external_base[f"{kind}_polarization_real_{axis}"] = float(value.real)
                external_base[f"{kind}_polarization_imag_{axis}"] = float(value.imag)
        external_draw_vectors = sample_external_input_vectors(
            np.asarray([external_base[name] for name in external_order], dtype=float),
            external_covariance, samples=external_input_mc_samples,
            seed=external_input_mc_seed,
        )

    if electron_distribution is not None:
        raise ValueError(
            "Full-EM spectrum inversion cannot currently fit electron flow with an explicit "
            "relativistic electron distribution because a drifted relativistic VDF is not "
            "implemented; use the Maxwellian bulk-electron branch rather than fixing or "
            "silently assuming an electron-flow vector"
        )

    f = _real_array(frequency_shift_Hz, "frequency_shift_Hz", 1)
    y = _real_array(measured_spectrum, "measured_spectrum", 1)
    if f.ndim != 1 or y.shape != f.shape or f.size < 8:
        raise ValueError("frequency_shift_Hz and measured_spectrum must be matching vectors with >=8 bins")
    if np.any(~np.isfinite(f)) or np.any(~np.isfinite(y)) or np.any(np.diff(f) <= 0) or np.any(y < 0):
        raise ValueError("frequency axis must increase; measured spectrum must be finite and non-negative")
    response = _real_array(instrument_response, "instrument_response", 2)
    if response.shape != (f.size, f.size) or np.any(~np.isfinite(response)) or np.any(response < 0) or np.any(response.sum(axis=0) <= 0):
        raise ValueError("instrument_response must be a non-negative square matrix with non-zero columns")
    if not np.allclose(response.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("measured instrument_response columns must already sum to one; the calibrated response operator is not renormalized")
    response_sigma = None
    if instrument_response_uncertainty is not None:
        response_sigma = _real_array(instrument_response_uncertainty, "instrument_response_uncertainty", 2)
        if response_sigma.shape != response.shape or np.any(~np.isfinite(response_sigma)) or np.any(response_sigma < 0):
            raise ValueError("instrument_response_uncertainty must be finite, non-negative, and match instrument_response")
        if isinstance(response_mc_samples, bool) or not isinstance(response_mc_samples, (int, np.integer)) or response_mc_samples < 20:
            raise ValueError("response_mc_samples must be an integer >= 20 when response uncertainty is supplied")
    if signal_uncertainty is not None and spectral_covariance is not None:
        raise ValueError("supply signal_uncertainty or spectral_covariance, not both")
    if spectral_covariance is not None:
        cov_y = _real_array(spectral_covariance, "spectral_covariance", 2)
        if cov_y.shape != (f.size, f.size) or np.any(~np.isfinite(cov_y)) or not np.allclose(cov_y, cov_y.T):
            raise ValueError("spectral_covariance must be a finite symmetric square matrix")
        try:
            chol = np.linalg.cholesky(cov_y)
        except np.linalg.LinAlgError as exc:
            raise ValueError("spectral_covariance must be positive definite") from exc
        whiten = lambda value: np.linalg.solve(chol, value)
    elif signal_uncertainty is not None:
        unc = _real_array(signal_uncertainty, "signal_uncertainty", 1)
        if unc.shape != f.shape or np.any(~np.isfinite(unc)) or np.any(unc <= 0):
            raise ValueError("signal_uncertainty must be positive and match measured_spectrum")
        whiten = lambda value: value / unc
    else:
        # Calibrated spectra remain analyzable when pointwise measurement
        # uncertainty is unavailable. Use unweighted least squares; the local
        # Jacobian covariance is residual-variance-scaled below and explicitly
        # labelled as not measurement-noise-calibrated.
        whiten = lambda value: value
    if not isinstance(ion_species, Sequence) or isinstance(ion_species, (str, bytes)) or len(ion_species) == 0:
        raise ValueError("ion_species must be a non-empty sequence for species-resolved CTS inversion")
    species_names = [str(item.get("name", f"ion_{i}")) for i, item in enumerate(ion_species)]
    if len(set(species_names)) != len(species_names):
        raise ValueError("ion species names must be unique for species-resolved temperature inference")
    automatically_estimated_parameters = [
        name for name, value in (
            ("electron_density_m3", electron_density_initial_m3),
            ("electron_temperature_eV", electron_temperature_initial_eV),
            ("ion_temperature_eV", ion_temperature_initial_eV),
        ) if value is None
    ]
    automatic_seeds = estimate_cts_optimizer_initial_parameters(
        f, y, probe_frequency_Hz=probe_frequency_Hz,
        scattering_geometry=scattering_geometry, ion_species=ion_species,
        ion_fractions=ion_fractions,
    ) if automatically_estimated_parameters else None
    if electron_density_initial_m3 is None:
        electron_density_initial_m3 = automatic_seeds["electron_density_m3"]
    if electron_temperature_initial_eV is None:
        electron_temperature_initial_eV = automatic_seeds["electron_temperature_eV"]
    if ion_temperature_initial_eV is None:
        ion_temperature_initial_eV = {
            name: automatic_seeds["ion_temperature_eV"] for name in species_names
        }
    if isinstance(ion_temperature_initial_eV, Mapping):
        if set(ion_temperature_initial_eV) != set(species_names):
            raise ValueError("ion_temperature_initial_eV mapping must contain exactly the supplied ion species names")
        ion_initials = np.array([_positive(ion_temperature_initial_eV[name], f"ion_temperature_initial_eV[{name}]") for name in species_names])
    elif np.isscalar(ion_temperature_initial_eV):
        ion_initials = np.full(len(species_names), _positive(ion_temperature_initial_eV, "ion_temperature_initial_eV"))
    else:
        ion_initials = _real_array(ion_temperature_initial_eV, "ion_temperature_initial_eV", 1)
        if ion_initials.shape != (len(species_names),) or np.any(~np.isfinite(ion_initials)) or np.any(ion_initials <= 0):
            raise ValueError("ion_temperature_initial_eV must be a positive scalar, species-name mapping, or one positive value per species")
    names = (
        "electron_density_m3", "electron_temperature_eV",
        "electron_flow_projection_m_s",
    ) + tuple(f"ion_temperature_eV:{name}" for name in species_names) + tuple(
        f"ion_flow_projection_m_s:{name}" for name in species_names
    )
    if parameter_bounds is not None and not isinstance(parameter_bounds, Mapping):
        raise ValueError("parameter_bounds must map parameter names to finite increasing pairs (positive for density and temperatures)")
    bounds = {} if parameter_bounds is None else dict(parameter_bounds)
    allowed_bound_names = set(names) | {"ion_temperature_eV", "ion_flow_projection_m_s"}
    unknown_bound_names = set(bounds)-allowed_bound_names
    if unknown_bound_names:
        raise ValueError(f"parameter_bounds contains unknown parameter names: {sorted(unknown_bound_names)}")
    if electron_flow_projection_initial_m_s is None:
        flow_seed = 0.0
    else:
        flow_seed_value = _real_array(
            electron_flow_projection_initial_m_s, "electron_flow_projection_initial_m_s"
        )
        if flow_seed_value.ndim != 0:
            raise ValueError("electron_flow_projection_initial_m_s must be a scalar")
        flow_seed = float(flow_seed_value)
    if not np.isfinite(flow_seed):
        raise ValueError("electron_flow_projection_initial_m_s must be finite")
    if ion_flow_projection_initial_m_s is None:
        ion_flow_seeds = np.zeros(len(species_names), dtype=float)
    else:
        if not isinstance(ion_flow_projection_initial_m_s, Mapping):
            raise ValueError("ion_flow_projection_initial_m_s must map every ion species name to an optional numerical seed")
        missing = [name for name in species_names if name not in ion_flow_projection_initial_m_s]
        extra = [name for name in ion_flow_projection_initial_m_s if name not in species_names]
        if missing or extra:
            raise ValueError(
                f"ion_flow_projection_initial_m_s must contain exactly the ion species names; missing={missing}, extra={extra}"
            )
        ion_flow_seed_values = []
        for name in species_names:
            seed = _real_array(
                ion_flow_projection_initial_m_s[name],
                f"ion_flow_projection_initial_m_s[{name}]",
            )
            if seed.ndim != 0:
                raise ValueError(f"ion_flow_projection_initial_m_s[{name}] must be a scalar")
            ion_flow_seed_values.append(float(seed))
        ion_flow_seeds = np.asarray(ion_flow_seed_values, dtype=float)
        if np.any(~np.isfinite(ion_flow_seeds)):
            raise ValueError("ion-flow projection initial estimates must be finite")
    initial = np.concatenate((
        [_positive(electron_density_initial_m3, "electron_density_initial_m3"),
         _positive(electron_temperature_initial_eV, "electron_temperature_initial_eV"),
         flow_seed],
        ion_initials,
        ion_flow_seeds,
    ))
    lower = np.full(len(names), -np.inf, dtype=float)
    upper = np.full(len(names), np.inf, dtype=float)
    applied_bounds = {}
    for index, name in enumerate(names):
        pair_value = bounds.get(name)
        if pair_value is None and name.startswith("ion_temperature_eV:"):
            pair_value = bounds.get("ion_temperature_eV")
        if pair_value is None and name.startswith("ion_flow_projection_m_s:"):
            pair_value = bounds.get("ion_flow_projection_m_s")
        if pair_value is None:
            continue
        pair = _real_array(pair_value, f"parameter_bounds[{name!r}]", 1)
        is_flow = "flow_projection_m_s" in name
        if (pair.shape != (2,) or np.any(~np.isfinite(pair))
                or (not is_flow and np.any(pair <= 0)) or pair[0] >= pair[1]):
            qualifier = "finite increasing" if is_flow else "positive finite increasing"
            raise ValueError(f"parameter_bounds[{name!r}] must be a {qualifier} pair")
        if not pair[0] < initial[index] < pair[1]:
            automatic_flow_seed = (
                name == "electron_flow_projection_m_s"
                and electron_flow_projection_initial_m_s is None
            ) or (
                name.startswith("ion_flow_projection_m_s:")
                and ion_flow_projection_initial_m_s is None
            )
            if is_flow and automatic_flow_seed:
                initial[index] = 0.5 * (pair[0] + pair[1])
            else:
                raise ValueError(f"initial value for {name} must lie strictly inside its supplied parameter bounds")
        if is_flow:
            lower[index], upper[index] = pair / 1.0e6
        else:
            lower[index], upper[index] = np.log10(pair)
        applied_bounds[name] = pair.tolist()
    n_species = len(species_names)
    p0 = np.concatenate((
        np.log10(initial[:2]),
        [initial[2] / 1.0e6],
        np.log10(initial[3:3+n_species]),
        initial[3+n_species:] / 1.0e6,
    ))

    def electron_flow_projection(p):
        return float(p[2] * 1.0e6)

    def unpack_temperatures(p):
        return {
            name: float(10.0**np.clip(p[3+i], -300.0, 300.0))
            for i, name in enumerate(species_names)
        }

    def unpack_ion_flow_projections(p):
        return {
            name: float(p[3+n_species+i] * 1.0e6)
            for i, name in enumerate(species_names)
        }

    def model(p, response_matrix=None, external_values=None):
        active_geometry = scattering_geometry
        active_field = magnetic_field_T
        active_incident_polarization = incident_polarization
        active_scattered_polarization = scattered_polarization
        if external_values is not None:
            values = dict(zip(external_order, np.asarray(external_values, dtype=float)))
            active_field = np.asarray([
                values.get(f"magnetic_field_T_{axis}", np.asarray(magnetic_field_T, dtype=float)[i])
                for i, axis in enumerate("xyz")
            ], dtype=float)
            active_geometry = dict(scattering_geometry)
            for kind in ("incident_direction", "scattered_direction"):
                if any(f"{kind}_{axis}" in values for axis in "xyz"):
                    vector = np.asarray([
                        values.get(f"{kind}_{axis}", scattering_geometry[kind][i])
                        for i, axis in enumerate("xyz")
                    ], dtype=float)
                    norm = float(np.linalg.norm(vector))
                    if not np.isfinite(norm) or norm <= 0.0:
                        raise ValueError(f"sampled {kind} has zero or invalid norm")
                    active_geometry[kind] = vector / norm
            for kind, base in (
                ("incident", incident_polarization),
                ("scattered", scattered_polarization),
            ):
                if any(f"{kind}_polarization_{part}_{axis}" in values
                       for part in ("real", "imag") for axis in "xyz"):
                    vector = np.asarray([
                        complex(
                            values.get(f"{kind}_polarization_real_{axis}", complex(base[i]).real),
                            values.get(f"{kind}_polarization_imag_{axis}", complex(base[i]).imag),
                        )
                        for i, axis in enumerate("xyz")
                    ], dtype=complex)
                    norm = float(np.linalg.norm(vector))
                    base_norm = float(np.linalg.norm(np.asarray(base, dtype=complex)))
                    if not np.isfinite(norm) or norm <= 0.0:
                        raise ValueError(f"sampled {kind} polarization has zero or invalid norm")
                    vector *= base_norm / norm
                    if kind == "incident":
                        active_incident_polarization = vector
                    else:
                        active_scattered_polarization = vector
        result = full_em_cts_spectral_density(
            f, probe_frequency_Hz, active_geometry,
            electron_density_m3=10.0**np.clip(p[0], -300.0, 300.0),
            electron_temperature_eV=10.0**np.clip(p[1], -300.0, 300.0),
            ion_temperature_eV=unpack_temperatures(p),
            ion_species=ion_species, ion_fractions=ion_fractions,
            magnetic_field_T=active_field,
            electron_flow_projection_m_s=electron_flow_projection(p),
            ion_flow_projection_m_s=unpack_ion_flow_projections(p),
            incident_polarization=active_incident_polarization,
            scattered_polarization=active_scattered_polarization,
            harmonic_order=harmonic_order, quadrature_order=quadrature_order,
            damping_rate_rad_s=damping_rate_rad_s,
            fast_ion_populations=fast_ion_populations,
            electron_distribution=electron_distribution,
            bindslev_validity_theta_e_limit=bindslev_validity_theta_e_limit,
        )["spectral_density"]
        return _apply_frequency_response(response if response_matrix is None else response_matrix, result, f)

    def residual(p, return_profile=False, response_matrix=None, external_values=None):
        z = model(p, response_matrix=response_matrix, external_values=external_values)
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
    if rank < len(names) or not np.isfinite(condition) or condition > 1e12:
        raise ValueError("full-EM CTS fit rejected: density, electron temperature, electron-flow projection, species-resolved ion temperatures and ion-flow projections are not jointly identifiable from this spectrum")
    r, amplitude, offset, convolved = residual(fit.x, return_profile=True)
    dof = max(1, f.size - (len(names) + 2))
    covariance_log = np.linalg.pinv(fit.jac.T @ fit.jac)
    if spectral_covariance is None and signal_uncertainty is None:
        covariance_log *= float(r @ r / dof)
    def physical_parameter_values(parameters):
        return np.concatenate((
            10.0**np.clip(parameters[:2], -300.0, 300.0),
            [electron_flow_projection(parameters)],
            10.0**np.clip(parameters[3:3+n_species], -300.0, 300.0),
            np.asarray(list(unpack_ion_flow_projections(parameters).values()), dtype=float),
        ))
    values = physical_parameter_values(fit.x)
    transform_diagonal = np.concatenate((
        np.log(10.0) * values[:2],
        [1.0e6],
        np.log(10.0) * values[3:3+n_species],
        np.full(n_species, 1.0e6),
    ))
    transform = np.diag(transform_diagonal)
    covariance = transform @ covariance_log @ transform.T
    forward_diagnostics = full_em_cts_spectral_density(
        f, probe_frequency_Hz, scattering_geometry,
        electron_density_m3=float(values[0]),
        electron_temperature_eV=float(values[1]),
        ion_temperature_eV=unpack_temperatures(fit.x),
        ion_species=ion_species, ion_fractions=ion_fractions,
        magnetic_field_T=magnetic_field_T,
        electron_flow_projection_m_s=electron_flow_projection(fit.x),
        ion_flow_projection_m_s=unpack_ion_flow_projections(fit.x),
        incident_polarization=incident_polarization,
        scattered_polarization=scattered_polarization,
        harmonic_order=harmonic_order, quadrature_order=quadrature_order,
        damping_rate_rad_s=damping_rate_rad_s,
        fast_ion_populations=fast_ion_populations,
        electron_distribution=electron_distribution,
        bindslev_validity_theta_e_limit=bindslev_validity_theta_e_limit,
    )
    out = {
        "optimizer_initial_estimates": {
            "electron_density_m3": float(initial[0]),
            "electron_temperature_eV": float(initial[1]),
            "ion_temperature_eV": ion_initials.tolist(),
            "method": automatic_seeds["method"] if automatic_seeds is not None else "researcher_supplied",
            "automatically_estimated_parameters": automatically_estimated_parameters,
            "density_seed_assumption": (
                automatic_seeds["density_seed_assumption"] if automatic_seeds is not None else None
            ),
            "role": "optimizer_initial_estimates_only",
        },
        "electron_density_m3": float(values[0]),
        "electron_temperature_eV": float(values[1]),
        "electron_flow_projection_m_s": float(values[2]),
        "electron_flow_projection_uncertainty_m_s": float(np.sqrt(max(covariance[2, 2], 0.0))),
        "electron_flow_projection_is_fitted": True,
        "ion_temperature_eV": (float(values[3]) if len(species_names) == 1 else unpack_temperatures(fit.x)),
        "ion_temperature_eV_by_species": unpack_temperatures(fit.x),
        "ion_flow_projection_m_s_by_species": unpack_ion_flow_projections(fit.x),
        "ion_flow_projection_uncertainty_m_s_by_species": {
            name: float(np.sqrt(max(covariance[3+n_species+i, 3+n_species+i], 0.0)))
            for i, name in enumerate(species_names)
        },
        "ion_flow_projection_is_fitted": True,
        "parameter_names": list(names),
        "parameter_bounds": applied_bounds,
        "fit_covariance": covariance,
        "parameter_uncertainties": np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        "fitted_spectrum": amplitude * convolved + offset,
        "residual": y - (amplitude * convolved + offset),
        "profiled_amplitude": float(amplitude),
        "profiled_offset": float(offset),
        "reduced_chi_square": (
            float(r @ r / dof)
            if spectral_covariance is not None or signal_uncertainty is not None
            else None
        ),
        "fit_residual_rms": float(np.sqrt(np.mean((y - (amplitude * convolved + offset)) ** 2))),
        "spectral_uncertainty_supplied": bool(
            spectral_covariance is not None or signal_uncertainty is not None
        ),
        "spectral_weighting_status": (
            "full_frequency_bin_covariance_whitened"
            if spectral_covariance is not None else
            "diagonal_pointwise_measurement_uncertainty_weighted"
            if signal_uncertainty is not None else
            "unweighted_least_squares_no_measurement_uncertainty_supplied"
        ),
        "parameter_covariance_interpretation": (
            "local_jacobian_covariance_from_supplied_spectral_uncertainty"
            if spectral_covariance is not None or signal_uncertainty is not None
            else "residual_variance_scaled_local_covariance_not_measurement_noise_calibrated"
        ),
        "parameter_covariance_scope": (
            "local_jacobian_covariance_conditional_on_fixed_external_inputs; "
            "does_not_include_external_geometry_field_composition_polarization_ray_or_beam_covariance_uncertainty"
        ),
        "numerical_convergence_status": "not_assessed_in_single_inversion; use assess_full_em_cts_numerical_convergence",
        "jacobian_rank": rank,
        "jacobian_condition_number": condition,
        "optimizer_success": bool(fit.success),
        "model": ("magnetized full-EM CTS with relativistic gyrotropic electron/fast-ion response and second-order kinetic vertices"
                  if electron_distribution is not None
                  else "magnetized full-EM CTS with Bindslev low-temperature coupling and optional relativistic gyrotropic fast-ion populations"),
        "external_inputs_fixed": [
            "magnetic_field_T", "ion_species", "ion_fractions",
            "incident_polarization", "scattered_polarization",
        ],
        "electron_flow_inference": (
            "fitted scalar component along the local scattering wavevector; "
            "transverse components are not identifiable from a single CTS geometry"
        ),
        "ion_flow_inference": (
            "one fitted scalar projection per supplied ion species along the local "
            "scattering wavevector; transverse components are not identifiable "
            "from a single CTS geometry"
        ),
        "inversion_scope": "electron density, electron temperature, electron-flow projection, one ion-temperature and one ion-flow projection per supplied thermal ion species are fitted from the measured spectrum; composition, magnetic field, geometry, mode polarizations and supplied non-Maxwellian VDF shapes remain fixed external inputs",
        "fluctuation_covariance_density_current_E_B": forward_diagnostics["fluctuation_covariance_density_current_E_B"],
        "dielectric_tensor": forward_diagnostics["dielectric_tensor"],
        "wave_tensor": forward_diagnostics["wave_tensor"],
        "forward_model_diagnostics": forward_diagnostics,
    }
    # Preserve the fit-only covariance before adding any supplied external-response component.
    # This lets downstream users distinguish conditional fit uncertainty from propagated inputs.
    out["conditional_fit_covariance"] = covariance.copy()
    out["external_uncertainty_covariance_components"] = {}
    out["instrument_response_uncertainty_supplied"] = response_sigma is not None
    out["instrument_response_uncertainty_propagated"] = False
    if response_sigma is not None:
        rng = np.random.default_rng(73129)
        response_draws = []
        for _ in range(int(response_mc_samples)):
            sampled_response = np.maximum(response + rng.normal(0.0, response_sigma), 0.0)
            column_sums = sampled_response.sum(axis=0)
            if np.any(column_sums <= 0.0):
                continue
            sampled_response = sampled_response / column_sums[None, :]
            try:
                draw = least_squares(
                    lambda parameters: residual(parameters, response_matrix=sampled_response),
                    fit.x, bounds=(lower, upper), x_scale="jac", max_nfev=1200,
                )
                if draw.success and np.all(np.isfinite(draw.x)):
                    response_draws.append(physical_parameter_values(draw.x))
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
        out["instrument_response_mc_successes"] = len(response_draws)
        out["instrument_response_mc_requested"] = int(response_mc_samples)
        if len(response_draws) >= max(12, int(np.ceil(0.7*response_mc_samples))):
            response_covariance = np.cov(np.asarray(response_draws), rowvar=False, ddof=1)
            covariance = covariance + response_covariance
            out["instrument_response_parameter_covariance"] = response_covariance
            out["external_uncertainty_covariance_components"]["instrument_response"] = response_covariance
            out["fit_covariance"] = covariance
            out["parameter_covariance_scope"] = (
                "total covariance = conditional local-fit covariance + supplied instrument-response Monte Carlo covariance; "
                "cross-covariances with other external inputs are not included"
            )
            out["parameter_uncertainties"] = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            out["electron_flow_projection_uncertainty_m_s"] = float(np.sqrt(max(covariance[2,2], 0.0)))
            out["ion_flow_projection_uncertainty_m_s_by_species"] = {
                name: float(np.sqrt(max(covariance[3+n_species+i,3+n_species+i],0.0)))
                for i, name in enumerate(species_names)
            }
            out["instrument_response_uncertainty_propagated"] = True
        else:
            out["instrument_response_uncertainty_warning"] = (
                "Too few successful response-matrix perturbation refits; no response covariance was added."
            )
    out["external_input_covariance_supplied"] = external_covariance is not None
    out["external_input_covariance_order"] = list(external_order) if external_order is not None else []
    out["external_input_covariance_propagated"] = False
    out["external_input_mc_requested"] = (
        int(external_input_mc_samples) if external_covariance is not None else 0
    )
    out["external_input_mc_successes"] = 0
    if external_draw_vectors is not None:
        external_draws = []
        for draw_values in external_draw_vectors:
            try:
                draw = least_squares(
                    lambda parameters: residual(
                        parameters, external_values=draw_values
                    ),
                    fit.x, bounds=(lower, upper), x_scale="jac", max_nfev=1200,
                )
                if draw.success and np.all(np.isfinite(draw.x)):
                    external_draws.append(physical_parameter_values(draw.x))
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
        out["external_input_mc_successes"] = len(external_draws)
        if len(external_draws) >= max(12, int(np.ceil(0.7*external_input_mc_samples))):
            external_parameter_covariance = np.cov(
                np.asarray(external_draws), rowvar=False, ddof=1
            )
            covariance = covariance + external_parameter_covariance
            out["external_input_parameter_covariance"] = external_parameter_covariance
            out["external_uncertainty_covariance_components"]["local_external_inputs"] = (
                external_parameter_covariance
            )
            out["fit_covariance"] = covariance
            out["parameter_uncertainties"] = np.sqrt(
                np.maximum(np.diag(covariance), 0.0)
            )
            out["electron_flow_projection_uncertainty_m_s"] = float(
                np.sqrt(max(covariance[2, 2], 0.0))
            )
            out["ion_flow_projection_uncertainty_m_s_by_species"] = {
                name: float(np.sqrt(max(covariance[3+n_species+i, 3+n_species+i], 0.0)))
                for i, name in enumerate(species_names)
            }
            out["external_input_covariance_propagated"] = True
            out["parameter_covariance_scope"] = (
                "conditional local-fit covariance plus separately propagated supplied "
                "instrument-response and local magnetic-field/geometry/polarization covariance "
                "components; cross-covariance between distinct external covariance sources "
                "is not included"
            )
        else:
            out["external_input_covariance_warning"] = (
                "Too few successful local external-input perturbation refits; "
                "no local external-input covariance was added."
            )
    if not isinstance(numerical_convergence, (bool, np.bool_)):
        raise ValueError("numerical_convergence must be boolean")
    if numerical_convergence:
        default_harmonics = (int(harmonic_order), int(harmonic_order) + 4, int(harmonic_order) + 8)
        default_quadratures = (int(quadrature_order), int(quadrature_order) + 6, int(quadrature_order) + 12)
        selected_harmonics = default_harmonics if convergence_harmonic_orders is None else convergence_harmonic_orders
        selected_quadratures = default_quadratures if convergence_quadrature_orders is None else convergence_quadrature_orders
        convergence = assess_full_em_cts_numerical_convergence(
            f, probe_frequency_Hz, scattering_geometry,
            harmonic_orders=selected_harmonics,
            quadrature_orders=selected_quadratures,
            relative_tolerance=convergence_relative_tolerance,
            electron_density_m3=float(values[0]),
            electron_temperature_eV=float(values[1]),
            ion_temperature_eV=unpack_temperatures(fit.x),
            ion_species=ion_species, ion_fractions=ion_fractions,
            magnetic_field_T=magnetic_field_T,
            electron_flow_projection_m_s=electron_flow_projection(fit.x),
            ion_flow_projection_m_s=unpack_ion_flow_projections(fit.x),
            incident_polarization=incident_polarization,
            scattered_polarization=scattered_polarization,
            damping_rate_rad_s=damping_rate_rad_s,
            fast_ion_populations=fast_ion_populations,
            electron_distribution=electron_distribution,
            bindslev_validity_theta_e_limit=bindslev_validity_theta_e_limit,
        )
        out["numerical_convergence"] = convergence
        out["numerical_convergence_status"] = convergence["convergence_status"]
        out["numerical_convergence_is_uncertainty"] = False
    else:
        if convergence_harmonic_orders is not None or convergence_quadrature_orders is not None:
            raise ValueError("convergence refinement orders cannot be supplied when numerical_convergence=False")
        out["numerical_convergence"] = {
            "convergence_status": "not_assessed_by_request",
            "converged": None,
            "interpretation": "Numerical convergence was explicitly disabled; no convergence claim is made.",
        }
        out["numerical_convergence_status"] = "not_assessed_by_request"
        out["numerical_convergence_is_uncertainty"] = False
    from .thomson_identifiability import cts_full_em_identifiability
    out["parameter_identifiability"] = cts_full_em_identifiability(species_names)
    return out


def assess_full_em_cts_numerical_convergence(
    frequency_shift_Hz, probe_frequency_Hz, scattering_geometry, *,
    harmonic_orders=(4, 6, 8), quadrature_orders=(6, 8, 10),
    relative_tolerance=1e-3, **model_kwargs,
):
    """Run a paired harmonic/quadrature refinement study for the Full-EM forward model.

    This is an explicit numerical sensitivity study, not an experimental
    validation or a substitute for physical-model uncertainty. The same
    physical inputs and damping prescription are held fixed at every level.
    """
    harmonics = np.asarray(harmonic_orders)
    quadratures = np.asarray(quadrature_orders)
    if harmonics.ndim != 1 or quadratures.ndim != 1 or harmonics.size != quadratures.size or harmonics.size < 2:
        raise ValueError("harmonic_orders and quadrature_orders must be paired 1-D sequences with at least two levels")
    if harmonics.dtype.kind not in "iu" or quadratures.dtype.kind not in "iu":
        raise ValueError("harmonic_orders and quadrature_orders must contain integers")
    if np.any(harmonics < 2) or np.any(np.diff(harmonics) <= 0):
        raise ValueError("harmonic_orders must be strictly increasing integers >=2")
    if np.any(quadratures < 4) or np.any(np.diff(quadratures) <= 0):
        raise ValueError("quadrature_orders must be strictly increasing integers >=4")
    if isinstance(relative_tolerance, (bool, np.bool_)) or not isinstance(relative_tolerance, (int, float, np.integer, np.floating)):
        raise ValueError("relative_tolerance must be a positive finite real scalar")
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("relative_tolerance must be a positive finite real scalar")
    if "harmonic_order" in model_kwargs or "quadrature_order" in model_kwargs:
        raise ValueError("Set refinement levels through harmonic_orders and quadrature_orders, not model_kwargs")
    levels = []
    previous = None
    for harmonic_order, quadrature_order in zip(harmonics, quadratures):
        result = full_em_cts_spectral_density(
            frequency_shift_Hz, probe_frequency_Hz, scattering_geometry,
            harmonic_order=int(harmonic_order), quadrature_order=int(quadrature_order),
            **model_kwargs,
        )
        spectrum = _real_array(result["spectral_density"], "spectral_density", 1)
        if previous is None:
            relative_l2 = None
            relative_linf = None
        else:
            difference = spectrum - previous
            relative_l2 = float(np.linalg.norm(difference) / max(float(np.linalg.norm(spectrum)), np.finfo(float).tiny))
            relative_linf = float(np.max(np.abs(difference)) / max(float(np.max(np.abs(spectrum))), np.finfo(float).tiny))
        levels.append({
            "harmonic_order": int(harmonic_order),
            "quadrature_order": int(quadrature_order),
            "relative_l2_change_from_previous": relative_l2,
            "relative_linf_change_from_previous": relative_linf,
        })
        previous = spectrum
    final = levels[-1]
    converged = (
        final["relative_l2_change_from_previous"] <= tolerance
        and final["relative_linf_change_from_previous"] <= tolerance
    )
    return {
        "levels": levels,
        "relative_tolerance": tolerance,
        "converged": bool(converged),
        "convergence_status": "converged_at_final_refinement" if converged else "not_converged_at_final_refinement",
        "final_spectrum": previous,
        "interpretation": (
            "Numerical convergence under the tested paired harmonic/quadrature refinements only; "
            "does not quantify physical-model discrepancy or external-input uncertainty."
        ),
    }
