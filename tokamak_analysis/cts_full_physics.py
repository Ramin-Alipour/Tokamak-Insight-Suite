"""Integrated advanced CTS/ECE full-physics entry points.

This module connects the independently testable Landau--FDT, vector Maxwell,
mode-conversion and polarized kinetic-ECE kernels. It deliberately exposes the
actual approximation boundary of each solver; it does not silently replace
missing relativistic, collisional or arbitrary-3D physics with fitted damping.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

from .cts_landau_fdt import build_linearized_landau_fdt, landau_cts_dynamic_structure_factor
from .cts_landau_em import landau_full_em_current_response
from .cts_full_wave import solve_stratified_mode_conversion, solve_stratified_full_wave_oblique
from .ece_kinetic_modes import solve_two_mode_kinetic_ece_profile
from .ece_kinetic import (
    relativistic_kinetic_ece_absorption_coefficient,
    solve_relativistic_kinetic_ece_profile,
)
from .cts_em_background import integrate_thermal_microwave_background


def analyze_landau_fdt_cts(
    angular_frequency_rad_s,
    normalized_velocity_x,
    normalized_velocity_y,
    normalized_velocity_z,
    species,
    *,
    common_temperature_J,
    wave_number_m_inverse,
    propagation_direction=(1.0, 0.0, 0.0),
    common_bulk_flow_m_s=(0.0, 0.0, 0.0),
    coulomb_logarithm=15.0,
):
    """Build and apply the conserving multispecies Landau--FDT CTS branch.

    This is an electrostatic longitudinal CTS closure for a common-temperature,
    common-flow, isotropic Maxwellian mixture. The finite 3-D velocity grid is
    returned through its metadata; grid/domain convergence remains mandatory.
    """
    system = build_linearized_landau_fdt(
        normalized_velocity_x, normalized_velocity_y, normalized_velocity_z,
        species, common_temperature_J=common_temperature_J,
        common_bulk_flow_m_s=common_bulk_flow_m_s,
        coulomb_logarithm=coulomb_logarithm,
    )
    result = landau_cts_dynamic_structure_factor(
        system, angular_frequency_rad_s, species,
        common_temperature_J=common_temperature_J,
        wave_number_m_inverse=wave_number_m_inverse,
        propagation_direction=propagation_direction,
    )
    return {
        **result,
        "landau_system": system,
        "operator_metadata": dict(system.metadata),
        "integration_status": "landau_operator_and_matching_FDT_noise_connected_to_electrostatic_CTS_response",
        "quantitative_release_gate": {
            "velocity_grid_convergence_required": True,
            "velocity_domain_convergence_required": True,
            "frequency_resolution_convergence_required": True,
            "experimental_validation_performed": False,
        },
    }


def propagate_stratified_nonparaxial_vector_field(
    x_m,
    y_m,
    electric_field_V_m,
    z_m,
    epsilon_r,
    frequency_Hz,
    *,
    evanescent="discard",
    rtol=1e-7,
    atol=1e-9,
    max_step_m=None,
):
    """Propagate a 2-D vector beam through a z-stratified anisotropic plasma.

    Each transverse Fourier component is solved with the oblique 4x4 Berreman
    Maxwell boundary-value solver. Thus diffraction is non-paraxial and the
    off-diagonal Jones coupling, reflection and transmission are retained for
    every propagating angular component. The dielectric may be complex.

    The medium is invariant in x/y and both exterior ports are vacuum. This is
    not a general 3-D inhomogeneous-interface solver. Evanescent angular
    components are explicitly discarded; they are counted in the result.
    """
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    eps = np.asarray(epsilon_r, dtype=complex)
    field = np.asarray(electric_field_V_m, dtype=complex)
    f = float(frequency_Hz)
    if x.ndim != 1 or y.ndim != 1 or x.size < 4 or y.size < 4:
        raise ValueError("x_m and y_m must be 1-D grids with at least four points")
    for axis, name in ((x, "x_m"), (y, "y_m")):
        delta = np.diff(axis)
        if np.any(~np.isfinite(axis)) or np.any(delta <= 0) or not np.allclose(delta, delta[0]):
            raise ValueError(f"{name} must be finite, increasing and uniformly spaced")
    if field.shape != (y.size, x.size, 3) or np.any(~np.isfinite(field)):
        raise ValueError("electric_field_V_m must have shape (ny,nx,3) and be finite")
    if z.ndim != 1 or z.size < 2 or np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0):
        raise ValueError("z_m must be finite, strictly increasing and contain at least two points")
    if eps.shape != (z.size, 3, 3) or np.any(~np.isfinite(eps)):
        raise ValueError("epsilon_r must have shape (nz,3,3) and be finite")
    if not np.isfinite(f) or f <= 0:
        raise ValueError("frequency_Hz must be positive and finite")
    if evanescent != "discard":
        raise ValueError("evanescent must be 'discard'; evanescent scattering ports are not yet solved")
    omega = 2.0 * np.pi * f
    c0 = 299792458.0
    k0 = omega / c0
    kx = 2.0 * np.pi * fftfreq(x.size, d=float(x[1] - x[0]))
    ky = 2.0 * np.pi * fftfreq(y.size, d=float(y[1] - y[0]))
    KX, KY = np.meshgrid(kx, ky)
    spectrum = np.stack([fft2(field[..., j]) for j in range(3)], axis=-1)
    transmitted = np.zeros_like(spectrum)
    reflected = np.zeros_like(spectrum)
    propagating = KX*KX + KY*KY < k0*k0
    conversion = np.zeros((y.size, x.size, 2, 2), dtype=complex)
    solved = 0
    for iy, ix in zip(*np.nonzero(propagating)):
        q = float(np.sqrt(k0*k0 - KX[iy, ix]**2 - KY[iy, ix]**2))
        kv = np.array([KX[iy, ix], KY[iy, ix], q], dtype=float)
        ev = spectrum[iy, ix].copy()
        # Enforce the incident vacuum Maxwell constraint before boundary matching.
        ev -= kv * (np.dot(kv, ev) / (k0*k0))
        if np.linalg.norm(ev[:2]) <= np.finfo(float).tiny:
            continue
        conversion_result = solve_stratified_mode_conversion(
            omega, z, eps, kx_rad_m=float(KX[iy, ix]),
            ky_rad_m=float(KY[iy, ix]), rtol=rtol, atol=atol,
            max_step_m=max_step_m,
        )
        et = conversion_result["transmitted_jones_matrix"] @ ev[:2]
        er = conversion_result["reflected_jones_matrix"] @ ev[:2]
        transmitted[iy, ix, :2] = et
        transmitted[iy, ix, 2] = -(KX[iy, ix]*et[0] + KY[iy, ix]*et[1]) / q
        reflected[iy, ix, :2] = er
        reflected[iy, ix, 2] = (KX[iy, ix]*er[0] + KY[iy, ix]*er[1]) / q
        conversion[iy, ix] = conversion_result["transmitted_jones_matrix"]
        solved += 1
    output = np.stack([ifft2(transmitted[..., j]) for j in range(3)], axis=-1)
    reflected_field = np.stack([ifft2(reflected[..., j]) for j in range(3)], axis=-1)
    scale = max(float(np.linalg.norm(transmitted)), np.finfo(float).tiny)
    div = (KX*transmitted[..., 0] + KY*transmitted[..., 1]
           + np.sqrt(np.maximum(k0*k0-KX*KX-KY*KY, 0.0))*transmitted[..., 2])
    return {
        "x_m": x.copy(), "y_m": y.copy(), "z_m": z.copy(),
        "transmitted_electric_field_V_m": output,
        "reflected_electric_field_V_m": reflected_field,
        "transmitted_angular_spectrum_V_m": transmitted,
        "reflected_angular_spectrum_V_m": reflected,
        "transmitted_jones_matrix_by_angular_component": conversion,
        "kx_rad_m": kx, "ky_rad_m": ky,
        "propagating_component_mask": propagating,
        "evanescent_component_count": int(np.count_nonzero(~propagating)),
        "solved_angular_component_count": solved,
        "transversality_relative_residual": float(np.linalg.norm(div[propagating]) / scale),
        "solver": "nonparaxial_2D_angular_spectrum_plus_oblique_Berreman_4x4",
        "mode_conversion": "coherent_reflection_and_transmission_Jones_matrices_per_angular_component",
        "paraxial_approximation_used": False,
        "approximation_scope": "linear_anisotropic_dielectric_stratified_only_in_z; vacuum_ports; propagating_angular_spectrum",
        "limitations": [
            "The dielectric tensor is supplied; this routine does not infer plasma equilibrium or dielectric response.",
            "The medium is translationally invariant in x and y; arbitrary 3-D interfaces and lateral inhomogeneity are not represented.",
            "Evanescent angular components are discarded, not matched through the slab.",
            "No nonlinear wave coupling or experimental validation is performed.",
        ],
    }


def solve_polarized_kinetic_ece_with_mode_conversion(
    frequency_Hz,
    segment_length_m,
    *,
    wavevector_rad_m,
    magnetic_field_T,
    electron_density_m3,
    electron_temperature_eV,
    mode_electric_field_unit_flux,
    dielectric_z_m,
    dielectric_tensor_r,
    kx_rad_m=0.0,
    ky_rad_m=0.0,
    mode_coupling_matrix_rad_m=None,
    incident_coherency_K=None,
    harmonic_order=8,
    quadrature_order=48,
    damping_rate_rad_s=0.0,
    rtol=1e-8,
    atol=1e-10,
):
    """Run polarized kinetic ECE transfer and stratified full-wave conversion.

    The local kinetic opacity is evaluated for the supplied unit-normal-flux
    eigenmode basis; the independent Berreman solve returns coherent boundary
    reflection/transmission and mode conversion. Both outputs are returned
    together, without conflating a boundary Jones matrix with a distributed
    mode-coupling generator.
    """
    f = np.asarray(frequency_Hz, dtype=float)
    if f.ndim == 0:
        f = f.reshape(1)
    z = np.asarray(dielectric_z_m, dtype=float)
    eps = np.asarray(dielectric_tensor_r, dtype=complex)
    if f.ndim != 1 or f.size == 0 or np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequency_Hz must be a positive finite vector")
    if z.ndim != 1 or z.size < 2 or np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0):
        raise ValueError("dielectric_z_m must be finite and strictly increasing")
    if eps.shape == (z.size, 3, 3):
        eps = np.broadcast_to(eps[None, ...], (f.size, z.size, 3, 3))
    if eps.shape != (f.size, z.size, 3, 3) or np.any(~np.isfinite(eps)):
        raise ValueError("dielectric_tensor_r must have shape (nz,3,3) or (nf,nz,3,3)")
    kx = np.broadcast_to(np.asarray(kx_rad_m, dtype=float), f.shape)
    ky = np.broadcast_to(np.asarray(ky_rad_m, dtype=float), f.shape)
    if np.any(~np.isfinite(kx)) or np.any(~np.isfinite(ky)):
        raise ValueError("kx_rad_m and ky_rad_m must be finite")
    kinetic = solve_two_mode_kinetic_ece_profile(
        f, segment_length_m, wavevector_rad_m=wavevector_rad_m,
        magnetic_field_T=magnetic_field_T,
        electron_density_m3=electron_density_m3,
        electron_temperature_eV=electron_temperature_eV,
        mode_electric_field_unit_flux=mode_electric_field_unit_flux,
        mode_coupling_matrix_rad_m=mode_coupling_matrix_rad_m,
        incident_coherency_K=incident_coherency_K,
        harmonic_order=harmonic_order, quadrature_order=quadrature_order,
        damping_rate_rad_s=damping_rate_rad_s,
    )
    conversion = [
        solve_stratified_mode_conversion(
            2.0*np.pi*f[i], z, eps[i], kx_rad_m=float(kx[i]),
            ky_rad_m=float(ky[i]), rtol=rtol, atol=atol,
        )
        for i in range(f.size)
    ]
    return {
        **kinetic,
        "stratified_full_wave_mode_conversion": conversion,
        "integration_status": "kinetic_two_mode_opacity_and_ordered_LTE_transfer_with_independent_Berreman_Jones_conversion",
        "dielectric_tensor_source": "explicit_user_supplied_local_tensor",
        "limitations": list(kinetic.get("limitations", [])) + [
            "The Jones boundary solution and distributed kinetic opacity are coupled in the returned analysis, but are not algebraically collapsed into one transfer operator.",
            "Mode basis and distributed coupling generator remain explicit inputs and must be derived consistently from the supplied dielectric.",
            "The kinetic electron response is non-relativistic Maxwellian; no conserving collisional Landau correction or relativistic emissivity is included.",
            "LTE Rayleigh-Jeans source function is assumed.",
        ],
    }



def analyze_cts_microwave_background(
    frequency_Hz,
    path_length_m,
    electron_temperature_eV,
    *,
    electron_density_m3,
    ion_densities_m3,
    ion_charge_states,
    ece_absorption_coefficient_m_inverse=None,
    ece_mode_weights=None,
    gaunt_factor=1.0,
    incident_radiance_W_m2_sr_Hz=0.0,
):
    """Evaluate the physical CTS microwave-background radiance on a ray path.

    This public route connects the free-free/ECE radiative-transfer kernel to
    the advanced Analysis-1 dispatcher. ECE opacity, Gaunt factors, mode
    weights and incident radiance remain explicit inputs; this routine does
    not infer missing dielectric or relativistic corrections.
    """
    result = integrate_thermal_microwave_background(
        frequency_Hz,
        path_length_m,
        electron_temperature_eV,
        electron_density_m3=electron_density_m3,
        ion_densities_m3=ion_densities_m3,
        ion_charge_states=ion_charge_states,
        ece_absorption_coefficient_m_inverse=ece_absorption_coefficient_m_inverse,
        ece_mode_weights=ece_mode_weights,
        gaunt_factor=gaunt_factor,
        incident_radiance_W_m2_sr_Hz=incident_radiance_W_m2_sr_Hz,
    )
    return {
        **result,
        "public_physics_entrypoint": "microwave_background",
        "background_inputs_are_explicit": True,
        "automatic_model_substitution": False,
    }


def run_advanced_thomson_physics(model, **kwargs):
    """Public, explicit dispatcher for the advanced Analysis-1 physics solvers.

    Supported models:
      - landau_fdt_cts: multispecies electrostatic CTS with the discrete
        Landau operator, matching Langevin source and Poisson screening.
      - polarized_kinetic_ece: magnetized kinetic ECE opacity projected onto
        two unit-flux modes, polarized LTE transfer and Berreman conversion.
      - stratified_nonparaxial_vector: 2-D vector-beam propagation through a
        z-stratified anisotropic dielectric using angular-spectrum and
        oblique Berreman solutions.
      - stratified_mode_conversion: coherent Jones reflection/transmission
        matrices for a supplied stratified dielectric.

    No physical model is substituted automatically. Equilibrium, geometry,
    dielectric, mode-basis and calibration inputs remain explicit caller data.
    The returned limitations and convergence gates are preserved.
    """
    if not isinstance(model, str):
        raise ValueError("model must be an explicit supported model name")
    if model == "relativistic_kinetic_ece_profile":
        solver = solve_relativistic_kinetic_ece_profile
    elif model == "relativistic_kinetic_ece":
        solver = relativistic_kinetic_ece_absorption_coefficient
    elif model == "microwave_background":
        solver = analyze_cts_microwave_background
    elif model == "landau_fdt_cts":
        solver = analyze_landau_fdt_cts
    elif model == "landau_fdt_full_em_response":
        solver = landau_full_em_current_response
    elif model == "polarized_kinetic_ece":
        solver = solve_polarized_kinetic_ece_with_mode_conversion
    elif model == "stratified_nonparaxial_vector":
        solver = propagate_stratified_nonparaxial_vector_field
    elif model == "stratified_mode_conversion":
        solver = solve_stratified_mode_conversion
    else:
        raise ValueError(
            "model must be one of: relativistic_kinetic_ece_profile, relativistic_kinetic_ece, microwave_background, landau_fdt_cts, landau_fdt_full_em_response, polarized_kinetic_ece, "
            "stratified_nonparaxial_vector, stratified_mode_conversion; "
            "no automatic model substitution is performed"
        )
    result = solver(**kwargs)
    if not isinstance(result, Mapping):
        raise TypeError("advanced physics solver must return a mapping")
    return {
        **dict(result),
        "public_physics_entrypoint": model,
        "automatic_model_substitution": False,
    }


__all__ = [
    "analyze_cts_microwave_background",
    "analyze_landau_fdt_cts",
    "propagate_stratified_nonparaxial_vector_field",
    "solve_polarized_kinetic_ece_with_mode_conversion",
    "run_advanced_thomson_physics",
]
