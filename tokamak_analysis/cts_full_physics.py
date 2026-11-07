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

from .cts_landau_fdt import LandauFDTSystem, build_linearized_landau_fdt, landau_cts_dynamic_structure_factor
from .cts_fdt import (
    quantum_fdt_dynamic_structure_factor,
    quantum_unsymmetrized_dynamic_structure_factor,
)
from .cts_landau_em import landau_full_em_current_response, landau_full_em_electric_field_fluctuation_spectrum
from .cts_full_wave import (solve_stratified_mode_conversion, solve_stratified_full_wave_oblique, solve_3d_periodic_full_wave)
from .ece_kinetic_modes import solve_two_mode_kinetic_ece_profile
from .ece_kinetic import (
    relativistic_kinetic_ece_absorption_coefficient,
    solve_relativistic_kinetic_ece_profile,
)
from .cts_em_background import (
    integrate_thermal_microwave_background,
    plasma_charge_state_moments,
    radiance_to_receiver_spectral_power,
    propagate_receiver_spectral_power_covariance,
    integrate_receiver_spectral_power,
    brightness_temperature_from_radiance,
)
from .cts_kappa import kappa_electron_susceptibility, bi_kappa_electron_susceptibility
from .cts_kappa import maxwellian_electron_susceptibility
from .cts_kappa_dispersion import kappa_dispersion_function
from .thomson_generalized_dispersion import (
    collective_tabulated_electron_spectral_density,
    collective_multi_population_tabulated_electron_spectral_density,
    generalized_electron_susceptibility,
)
from .cts_analysis import analyze_cts_spectrum, fit_cts_ion_species_flows
from .cts_collisions import multispecies_thermal_coulomb_collision_matrix
from .cts_bgk_fdt import analyze_conserving_bgk_cts
from .cts_stokes_transfer import (
    propagate_polarized_lte_stokes,
    propagate_stratified_polarized_lte_stokes,
)
from .thomson_evdf import reconstruct_thomson_evdf
from .thomson_scattering import infer_thomson_path_averaged_density
from .thomson_joint_spatial_spectral import reconstruct_joint_thomson_spatial_spectra
from .formula_library import (
    thomson_electron_temperature_from_spectral_width,
    thomson_velocity_projection_relativistic_exact,
)

from .advanced_inference import reconstruct_multiview_3d_evdf
from .thomson_anisotropic_inference import (
    fit_gyrotropic_bi_maxwellian_thomson,
    fit_gyrotropic_bi_kappa_thomson,
)
from .thomson_gap_closure import (
    reconstruct_relativistic_projected_thomson_evdf,
    reconstruct_psf_aware_thomson_profile,
    map_thomson_channels_to_normalized_poloidal_flux,
    relativistic_projected_compton_kinematics,
)
from .thomson_kinetic_models import (
    infer_gyrotropic_bi_maxwellian_temperatures,
    infer_three_dimensional_flow_from_thomson_projections,
    project_gyrotropic_bi_kappa_electron_vdf,
    collective_kappa_spectral_density,
    fit_collective_kappa,
)
from .thomson_temperature_tensor import infer_full_temperature_tensor_from_thomson_projections
from .cts_magnetoionic import (
    cold_plasma_dielectric_tensor,
    cold_plasma_magnetoionic_modes,
    cold_plasma_absorbed_power_density,
    cold_plasma_mode_radiative_coefficients,
    cold_plasma_mode_group_velocity,
)


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



def analyze_landau_fdt_full_em_constitutive(
    angular_frequency_rad_s,
    normalized_velocity_x=None,
    normalized_velocity_y=None,
    normalized_velocity_z=None,
    species=None,
    *,
    common_temperature_J,
    wave_number_m_inverse,
    propagation_direction=(1.0, 0.0, 0.0),
    common_bulk_flow_m_s=(0.0, 0.0, 0.0),
    coulomb_logarithm=15.0,
    system=None,
):
    """Build and apply the Landau--FDT Full-EM constitutive response.

    Unlike the lower-level response routine, this public Analysis-1 entry
    point assembles the conserving multispecies collision operator from the
    calibrated equilibrium inputs and then projects its resolvent and matching
    FDT noise onto all three Cartesian current components.

    Scope is deliberately limited to the unmagnetized, zero-flow,
    common-temperature isotropic Maxwellian reference equilibrium. The output
    is a local constitutive tensor/noise kernel, NOT a Full-EM CTS three-wave
    scattering vertex or a complete measured-spectrum inversion.
    """
    if system is None:
        if any(value is None for value in (
            normalized_velocity_x, normalized_velocity_y, normalized_velocity_z, species
        )):
            raise ValueError(
                "supply either a preassembled LandauFDTSystem or all three normalized velocity axes and species"
            )
        system = build_linearized_landau_fdt(
            normalized_velocity_x, normalized_velocity_y, normalized_velocity_z,
            species, common_temperature_J=common_temperature_J,
            common_bulk_flow_m_s=common_bulk_flow_m_s,
            coulomb_logarithm=coulomb_logarithm,
        )
    elif not isinstance(system, LandauFDTSystem):
        raise TypeError("system must be a LandauFDTSystem assembled by build_linearized_landau_fdt")
    if species is None:
        raise ValueError("species metadata is required with a preassembled LandauFDTSystem")
    response = landau_full_em_current_response(
        system, angular_frequency_rad_s, species=species,
        common_temperature_J=common_temperature_J,
        wave_number_m_inverse=wave_number_m_inverse,
        propagation_direction=propagation_direction,
    )
    return {
        **response,
        "landau_system": system,
        "operator_metadata": dict(system.metadata),
        "integration_status": (
            "assembled_conserving_Landau_operator_and_matching_FDT_noise_"
            "projected_to_cartesian_Full_EM_constitutive_response"
        ),
        "cts_three_wave_vertex_connected": False,
        "quantitative_release_gate": {
            "velocity_grid_convergence_required": True,
            "velocity_domain_convergence_required": True,
            "frequency_resolution_convergence_required": True,
            "magnetized_response_required_for_magnetized_plasma": True,
            "experimental_validation_performed": False,
        },
    }



def analyze_landau_fdt_full_em_spectrum(
    angular_frequency_rad_s,
    normalized_velocity_x=None,
    normalized_velocity_y=None,
    normalized_velocity_z=None,
    species=None,
    *,
    common_temperature_J,
    wave_number_m_inverse,
    propagation_direction=(1.0, 0.0, 0.0),
    common_bulk_flow_m_s=(0.0, 0.0, 0.0),
    coulomb_logarithm=15.0,
    relative_dielectric_tensor=None,
    condition_limit=1.0e12,
    system=None,
):
    """Run the conserving Landau--FDT operator through the Full-EM Green dyadic.

    This is the end-to-end local homogeneous-medium field-fluctuation path:
    multispecies Landau collision operator -> matched Langevin current noise
    -> Cartesian conductivity/dielectric -> Maxwell Green tensor -> electric,
    magnetic, cross-spectral and transverse Stokes spectra.

    The reference equilibrium is restricted to an unmagnetized, isotropic,
    common-temperature Maxwellian mixture with zero bulk flow. It is not a
    magnetized CTS three-wave vertex, a ray-integrated receiver spectrum, or
    an experimental inversion. A supplied dielectric tensor can represent a
    separately validated passive background; otherwise the constitutive
    dielectric generated by the same Landau response is used.
    """
    constitutive = analyze_landau_fdt_full_em_constitutive(
        angular_frequency_rad_s,
        normalized_velocity_x, normalized_velocity_y, normalized_velocity_z,
        species,
        common_temperature_J=common_temperature_J,
        wave_number_m_inverse=wave_number_m_inverse,
        propagation_direction=propagation_direction,
        common_bulk_flow_m_s=common_bulk_flow_m_s,
        coulomb_logarithm=coulomb_logarithm,
        system=system,
    )
    field = landau_full_em_electric_field_fluctuation_spectrum(
        constitutive,
        wave_number_m_inverse,
        propagation_direction=propagation_direction,
        relative_dielectric_tensor=relative_dielectric_tensor,
        condition_limit=condition_limit,
    )
    return {
        **field,
        "constitutive_response": constitutive,
        "integration_status": (
            "conserving_Landau_operator_and_matching_FDT_noise_connected_"
            "through_frequency_resolved_Maxwell_Green_dyadic"
        ),
        "cts_three_wave_vertex_connected": False,
        "physical_scope": (
            "local_unmagnetized_zero_flow_common_temperature_Maxwellian_"
            "Full_EM_field_fluctuation_spectrum"
        ),
        "quantitative_release_gate": {
            "velocity_grid_convergence_required": True,
            "velocity_domain_convergence_required": True,
            "frequency_resolution_convergence_required": True,
            "magnetized_response_required_for_magnetized_plasma": True,
            "ray_and_receiver_calibration_required_for_measured_spectrum": True,
            "experimental_validation_performed": False,
        },
    }



def analyze_landau_fdt_full_em_spectrum_converged(
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
    relative_dielectric_tensor=None,
    condition_limit=1.0e12,
    convergence_tolerance=2.0e-2,
    max_refinements=2,
    max_velocity_nodes=512,
):
    r"""Run a grid-refined Landau--FDT Full-EM calculation with a quantitative gate.

    At refinement level j, every normalized-velocity interval is bisected:
    N_{j+1}=2 N_j-1 per axis, with the velocity-domain endpoints held fixed.
    For each returned tensor X, the inter-grid error is
        delta_X = ||X_j-X_{j-1}||_F / max(||X_j||_F,||X_{j-1}||_F, tiny).
    Convergence is accepted only when the maximum delta_X over conductivity,
    current-noise, electric-field, magnetic-field and Stokes spectra is below
    the requested tolerance. This checks grid resolution only; it does not
    certify velocity-domain, frequency-resolution or experimental convergence.
    """
    axes = []
    for values, name in (
        (normalized_velocity_x, "normalized_velocity_x"),
        (normalized_velocity_y, "normalized_velocity_y"),
        (normalized_velocity_z, "normalized_velocity_z"),
    ):
        axis = np.asarray(values, dtype=float)
        if axis.ndim != 1 or axis.size < 3 or np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
            raise ValueError(f"{name} must be finite, strictly increasing and contain at least three points")
        axes.append(axis)
    tolerance = float(convergence_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("convergence_tolerance must be positive and finite")
    if isinstance(max_refinements, (bool, np.bool_)) or not isinstance(max_refinements, (int, np.integer)) or max_refinements < 1:
        raise ValueError("max_refinements must be a positive integer")
    if isinstance(max_velocity_nodes, (bool, np.bool_)) or not isinstance(max_velocity_nodes, (int, np.integer)) or max_velocity_nodes < 27:
        raise ValueError("max_velocity_nodes must be an integer >= 27")
    history = []
    previous = None
    final = None
    converged = False
    limiting_reason = "maximum_refinements_reached"
    compared_keys = (
        "conductivity_tensor_S_m",
        "current_fluctuation_spectral_matrix",
        "electric_field_fluctuation_spectral_matrix_V2_m2_per_rad_s",
        "magnetic_field_fluctuation_spectral_matrix_T2_per_rad_s",
        "transverse_stokes_spectral_density_IQUV",
    )

    def relative_change(current, reference):
        a = np.asarray(current)
        b = np.asarray(reference)
        if a.shape != b.shape or np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
            raise ArithmeticError("successive Landau--FDT grids returned incompatible or non-finite spectra")
        numerator = float(np.linalg.norm(a - b))
        denominator = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), np.finfo(float).tiny)
        return numerator / denominator

    for level in range(int(max_refinements) + 1):
        node_count = int(np.prod([axis.size for axis in axes]))
        if node_count > max_velocity_nodes:
            limiting_reason = "max_velocity_nodes_reached"
            break
        final = analyze_landau_fdt_full_em_spectrum(
            angular_frequency_rad_s,
            *axes,
            species,
            common_temperature_J=common_temperature_J,
            wave_number_m_inverse=wave_number_m_inverse,
            propagation_direction=propagation_direction,
            common_bulk_flow_m_s=common_bulk_flow_m_s,
            coulomb_logarithm=coulomb_logarithm,
            relative_dielectric_tensor=relative_dielectric_tensor,
            condition_limit=condition_limit,
        )
        entry = {"level": level, "axis_nodes": [int(axis.size) for axis in axes], "velocity_nodes_per_species": node_count}
        if previous is not None:
            changes = {
                key: relative_change(final["constitutive_response"][key] if key in ("conductivity_tensor_S_m", "current_fluctuation_spectral_matrix") else final[key],
                                    previous["constitutive_response"][key] if key in ("conductivity_tensor_S_m", "current_fluctuation_spectral_matrix") else previous[key])
                for key in compared_keys
            }
            entry["relative_changes"] = changes
            entry["maximum_relative_change"] = max(changes.values())
            if entry["maximum_relative_change"] <= tolerance:
                converged = True
                limiting_reason = "velocity_grid_converged"
                history.append(entry)
                break
        history.append(entry)
        previous = final
        refined = [np.sort(np.concatenate((axis, 0.5*(axis[:-1] + axis[1:])))) for axis in axes]
        if int(np.prod([axis.size for axis in refined])) > max_velocity_nodes:
            limiting_reason = "max_velocity_nodes_reached"
            break
        axes = refined

    if final is None:
        raise ValueError("initial velocity grid exceeds max_velocity_nodes")
    gate = dict(final["quantitative_release_gate"])
    gate.update({
        "velocity_grid_convergence_required": not converged,
        "velocity_grid_converged": converged,
        "velocity_domain_convergence_required": True,
        "frequency_resolution_convergence_required": True,
        "experimental_validation_performed": False,
    })
    final["quantitative_release_gate"] = gate
    final["velocity_grid_convergence"] = {
        "status": "converged" if converged else "not_converged",
        "converged": converged,
        "tolerance": tolerance,
        "maximum_refinements": int(max_refinements),
        "maximum_velocity_nodes": int(max_velocity_nodes),
        "limiting_reason": limiting_reason,
        "history": history,
        "compared_spectral_quantities": list(compared_keys),
        "domain_endpoints_preserved": True,
        "relative_error_definition": "Frobenius norm of successive-grid difference divided by the larger successive-grid Frobenius norm",
    }
    final["reportable_as_grid_converged"] = converged
    final["public_physics_entrypoint"] = "landau_fdt_full_em_spectrum_converged"
    return final


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
    ece_emissivity_W_m3_sr_Hz=None,
    ece_lte=False,
    gaunt_factor=1.0,
    incident_radiance_W_m2_sr_Hz=0.0,
    receiver_etendue_m2_sr=None,
    end_to_end_efficiency=None,
    receiver_input_covariance=None,
):
    """Evaluate CTS plasma-background transfer and optional receiver coupling.

    The path solver returns spectral radiance. If receiver etendue and
    end-to-end efficiency are both supplied, the radiance is converted to
    receiver spectral power exactly once. An optional full joint covariance
    for [radiance bins, etendue, efficiency] is propagated through that
    multiplicative measurement equation; frequency integration then retains
    all supplied spectral-bin correlations.
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
        ece_emissivity_W_m3_sr_Hz=ece_emissivity_W_m3_sr_Hz,
        ece_lte=ece_lte,
        gaunt_factor=gaunt_factor,
        incident_radiance_W_m2_sr_Hz=incident_radiance_W_m2_sr_Hz,
    )
    has_etendue = receiver_etendue_m2_sr is not None
    has_efficiency = end_to_end_efficiency is not None
    if has_etendue != has_efficiency:
        raise ValueError(
            "receiver_etendue_m2_sr and end_to_end_efficiency must be supplied together"
        )
    if receiver_input_covariance is not None and not has_etendue:
        raise ValueError(
            "receiver_input_covariance requires receiver etendue and end-to-end efficiency"
        )
    result["rayleigh_jeans_brightness_temperature_K"] = (
        brightness_temperature_from_radiance(
            frequency_Hz, result["spectral_radiance_W_m2_sr_Hz"],
            convention="rayleigh_jeans",
        )
    )
    result["planck_brightness_temperature_K"] = (
        brightness_temperature_from_radiance(
            frequency_Hz, result["spectral_radiance_W_m2_sr_Hz"],
            convention="planck",
        )
    )
    result["brightness_temperature_interpretation"] = (
        "Equivalent radiometric brightness temperature; not electron_temperature_eV"
    )
    if has_etendue:
        if receiver_input_covariance is None:
            receiver_power = radiance_to_receiver_spectral_power(
                result["spectral_radiance_W_m2_sr_Hz"],
                receiver_etendue_m2_sr=receiver_etendue_m2_sr,
                end_to_end_efficiency=end_to_end_efficiency,
            )
            result["receiver_spectral_power_W_Hz"] = receiver_power
            result["receiver_power_uncertainty_status"] = (
                "not_propagated; full joint receiver_input_covariance not supplied"
            )
            integrated = integrate_receiver_spectral_power(
                frequency_Hz, receiver_power
            )
        else:
            propagated = propagate_receiver_spectral_power_covariance(
                result["spectral_radiance_W_m2_sr_Hz"],
                receiver_etendue_m2_sr=receiver_etendue_m2_sr,
                end_to_end_efficiency=end_to_end_efficiency,
                input_covariance=receiver_input_covariance,
            )
            result.update(propagated)
            integrated = integrate_receiver_spectral_power(
                frequency_Hz,
                propagated["receiver_spectral_power_W_Hz"],
                spectral_power_covariance_W2_Hz2=(
                    propagated["receiver_spectral_power_covariance_W2_Hz2"]
                ),
            )
        result.update(integrated)
        result["receiver_conversion_model"] = (
            "P_nu = L_nu * receiver_etendue * end_to_end_efficiency"
        )
        result["receiver_conversion_applied_once"] = True
    else:
        result["receiver_conversion_model"] = "not_applied; radiance output retained"
        result["receiver_conversion_applied_once"] = False
    return {
        **result,
        "public_physics_entrypoint": "microwave_background",
        "background_inputs_are_explicit": True,
        "automatic_model_substitution": False,
    }



def collective_non_gyrotropic_gaussian_electron_spectrum(
    wavelength_nm, laser_wavelength_nm, scattering_angle_rad, *,
    electron_density_m3, electron_temperature_tensor_eV,
    electron_velocity_m_s, electron_velocity_grid_m_s,
    ion_species, ion_fractions, ion_temperature_eV=None,
    ion_velocity_m_s, incident_direction, scattered_direction,
    minimum_velocity_grid_probability=0.999,
    **response_options,
):
    """Electrostatic CTS spectrum for a general 3-D Gaussian electron tensor.

    The 3-D Gaussian is marginalized exactly along q_hat=(n_s-n_i)/|n_s-n_i|.
    Its projected variance is sigma_q^2=(e/m_e) q_hat.T T_e q_hat and its
    projected mean is q_hat.u_e. The tabulated-VDF Landau solver evaluates
    the kinetic susceptibility, including finite-support endpoint terms.
    A single view measures T_q, not all six independent tensor components.
    """
    tensor = np.asarray(electron_temperature_tensor_eV, dtype=float)
    flow = np.asarray(electron_velocity_m_s, dtype=float)
    velocity = np.asarray(electron_velocity_grid_m_s, dtype=float)
    incident = np.asarray(incident_direction, dtype=float)
    scattered = np.asarray(scattered_direction, dtype=float)
    angle = float(scattering_angle_rad)
    if tensor.shape != (3, 3) or np.any(~np.isfinite(tensor)):
        raise ValueError("electron_temperature_tensor_eV must be a finite 3x3 tensor")
    scale = max(float(np.linalg.norm(tensor, ord=2)), 1.0)
    if np.linalg.norm(tensor-tensor.T, ord=2) > 1e-10*scale:
        raise ValueError("electron_temperature_tensor_eV must be symmetric")
    tensor = 0.5*(tensor+tensor.T)
    eigenvalues = np.linalg.eigvalsh(tensor)
    if eigenvalues[0] <= 0.0:
        raise ValueError("electron_temperature_tensor_eV must be positive definite")
    if flow.shape != (3,) or np.any(~np.isfinite(flow)) or np.linalg.norm(flow) >= 299792458.0:
        raise ValueError("electron_velocity_m_s must be a finite subluminal 3-vector")
    if incident.shape != (3,) or scattered.shape != (3,) or np.any(~np.isfinite(incident)) or np.any(~np.isfinite(scattered)):
        raise ValueError("incident_direction and scattered_direction must be finite 3-vectors")
    if np.linalg.norm(incident) == 0.0 or np.linalg.norm(scattered) == 0.0:
        raise ValueError("incident and scattered directions must be non-zero")
    incident = incident/np.linalg.norm(incident)
    scattered = scattered/np.linalg.norm(scattered)
    measured_angle = float(np.arccos(np.clip(incident @ scattered, -1.0, 1.0)))
    if not np.isfinite(angle) or angle <= 0.0 or angle > np.pi or not np.isclose(measured_angle, angle, atol=1e-5, rtol=0.0):
        raise ValueError("scattering_angle_rad must agree with the supplied unit propagation directions")
    qhat = scattered-incident
    qhat /= np.linalg.norm(qhat)
    temperature_q = float(qhat @ tensor @ qhat)
    sigma_q = float(np.sqrt(1.602176634e-19*temperature_q/9.1093837015e-31))
    if velocity.ndim != 1 or velocity.size < 9 or np.any(~np.isfinite(velocity)) or np.any(np.diff(velocity) <= 0):
        raise ValueError("electron_velocity_grid_m_s must be a finite strictly increasing 1-D grid with >=9 points")
    if np.any(np.abs(velocity) >= 299792458.0):
        raise ValueError("the non-relativistic Gaussian Landau kernel requires |v_q|<c throughout the supplied grid")
    # Keep the thermal marginal centred at zero: the public Landau forward
    # model applies q.u exactly once through its resonant phase velocity.
    projected_pdf = np.exp(-0.5*(velocity/sigma_q)**2)/(np.sqrt(2.0*np.pi)*sigma_q)
    captured = float(np.trapz(projected_pdf, velocity))
    minimum = float(minimum_velocity_grid_probability)
    if not np.isfinite(minimum) or not 0.0 < minimum < 1.0:
        raise ValueError("minimum_velocity_grid_probability must lie strictly between zero and one")
    if not np.isfinite(captured) or captured < minimum:
        raise ValueError(
            f"electron_velocity_grid_m_s captures only {captured:.8g} of the projected Gaussian; "
            f"expand/refine the grid to reach {minimum:.8g} before using the truncated Landau response"
        )
    result = collective_tabulated_electron_spectral_density(
        wavelength_nm, laser_wavelength_nm, scattering_angle_rad,
        electron_density_m3=electron_density_m3,
        electron_velocity_grid_m_s=velocity,
        electron_probability_density_s_m=projected_pdf,
        ion_species=ion_species, ion_fractions=ion_fractions,
        ion_temperature_eV=ion_temperature_eV,
        electron_velocity_m_s=flow, ion_velocity_m_s=ion_velocity_m_s,
        incident_direction=incident, scattered_direction=scattered,
        **response_options,
    )
    result.update({
        "electron_distribution_model": "non_gyrotropic_3D_Gaussian_temperature_tensor_exact_q_marginal",
        "electron_temperature_tensor_eV": tensor.copy(),
        "electron_temperature_tensor_eigenvalues_eV": eigenvalues.copy(),
        "scattering_vector_unit": qhat.copy(),
        "projected_directional_temperature_eV": temperature_q,
        "projected_thermal_standard_deviation_m_s": sigma_q,
        "projected_flow_velocity_m_s": float(qhat @ flow),
        "projected_velocity_grid_probability": captured,
        "single_view_identifiability": {
            "identified": ["q_projected_mean_velocity", "q_projected_second_moment_temperature"],
            "not_identified": ["six independent temperature-tensor components"],
            "reason": "one scattering vector samples q_hat.T @ T_e @ q_hat; tensor reconstruction requires independent geometries",
        },
        "projection_relation": "T_q = q_hat.T @ T_e @ q_hat; sigma_q^2 = e*T_q/m_e; f_q(v) = exp(-v^2/(2*sigma_q^2))/sqrt(2*pi*sigma_q^2)",
        "flow_convention": "thermal marginal is zero-centred; q_hat dot electron_velocity_m_s is applied once as the Doppler shift by the Landau forward model",
        "relativistic_scope": "non-relativistic Gaussian marginal; use the relativistic projected-EVDF route when thermal or drift beta is not small",
    })
    return result



# Canonical list shared with the advanced Dashboard selector. Keep this set in
# lockstep with the explicit solver dispatch below; it also drives complete,
# actionable errors for unsupported model names.
ADVANCED_THOMSON_PHYSICS_MODELS = frozenset({
    "thomson_kinetic_background_fit",
    "thomson_n_population_relativistic_fit",
    "thomson_multiview_3d_evdf",
    "kappa_electrostatic_susceptibility",
    "kappa_dispersion_function",
    "bi_kappa_electrostatic_susceptibility",
    "maxwellian_electrostatic_susceptibility",
    "cts_collective_kappa_spectrum",
    "cts_collective_kappa_inference",
    "cts_tabulated_evdf_susceptibility",
    "cts_tabulated_evdf_landau",
    "cts_non_gyrotropic_gaussian_tensor_spectrum",
    "cts_multi_ion_flow_inference",
    "cts_electrostatic_measured_spectrum_inference",
    "thomson_projected_evdf",
    "thomson_projected_bi_kappa_evdf",
    "thomson_temperature_from_spectral_width",
    "thomson_relativistic_exact_flow_projection",
    "thomson_relativistic_projected_evdf",
    "thomson_psf_profile_inverse",
    "thomson_joint_spatial_spectra",
    "thomson_three_dimensional_flow",
    "thomson_full_temperature_tensor",
    "thomson_full_temperature_tensor_spectral_fit",
    "gyrotropic_bi_maxwellian_anisotropy",
    "gyrotropic_bi_maxwellian_spectral_fit",
    "gyrotropic_bi_kappa_spectral_fit",
    "cts_conserving_bgk_fdt",
    "landau_fdt_cts",
    "landau_fdt_full_em_constitutive",
    "landau_fdt_full_em_spectrum_converged",
    "landau_fdt_full_em_field_fluctuations",
    "landau_fdt_full_em_response",
    "polarized_kinetic_ece",
    "stratified_nonparaxial_vector",
    "periodic_3d_full_wave",
    "stratified_mode_conversion",
    "relativistic_kinetic_ece_profile",
    "relativistic_kinetic_ece",
    "microwave_background",
    "multispecies_thermal_collision_matrix",
    "quantum_fdt",
    "quantum_fdt_unsymmetrized",
    "polarized_lte_stokes",
    "stratified_polarized_lte_stokes",
    "cold_plasma_dielectric",
    "cold_plasma_magnetoionic_modes",
    "cold_plasma_absorbed_power_density",
    "cold_plasma_mode_radiative_coefficients",
    "cold_plasma_mode_group_velocity",
    "plasma_charge_state_moments",
    "thomson_normalized_poloidal_flux",
})

def run_advanced_thomson_physics(model, *args, **kwargs):
    """Public, explicit dispatcher for the advanced Analysis-1 physics solvers.

    Supported models:
      - cts_multi_ion_flow_inference: conditional fit of independent projected ion
        species flows from a calibrated electrostatic CTS spectrum with fixed
        plasma/composition inputs; the species-flow Jacobian rank gates reporting.
      - cts_non_gyrotropic_gaussian_tensor_spectrum: exact q-marginal of a general 3-D Gaussian temperature tensor, passed to the tabulated Landau CTS forward solver; a single view identifies only the q-projected temperature.
      - cts_tabulated_evdf_landau: electrostatic CTS dynamic structure spectrum using
        a supplied normalized 1-D projected electron EVDF, analytic piecewise-linear
        Landau-pole integration (including finite-support endpoint jumps), and
        either Maxwellian or species-resolved tabulated ion EVDFs.
      - cts_multi_population_tabulated_evdf_landau: density-weighted sum of independent
        electron-population Landau susceptibilities and Klimontovich fluctuation sources,
        coupled through one multispecies dielectric; population fractions are explicit inputs.
      - maxwellian_electrostatic_susceptibility: exact Fried-Conte Maxwellian electron susceptibility using the Faddeeva function.
      - kappa_electrostatic_susceptibility: collisionless longitudinal susceptibility
        for the exact 1-D marginal of an isotropic 3-D Kappa distribution; this is
        a kinetic response kernel, not a complete Kappa CTS fluctuation spectrum.
      - bi_kappa_electrostatic_susceptibility: exact gyrotropic bi-Kappa directional
        marginal with full-tail Landau response; one view identifies only T_q.
      - cts_collective_kappa_spectrum: collisionless electrostatic dynamic-structure
        spectrum using independently selected Maxwellian/Kappa electron populations
        and Maxwellian/Kappa multispecies ions with species-resolved Landau response.
      - cts_collective_kappa_inference: joint measured-spectrum fit of electron density,
        second-moment temperature and Kappa index with fixed supplied ion state, optional
        per-species Maxwellian/Kappa ion models, and measured response.
      - landau_fdt_cts: multispecies electrostatic CTS with the discrete
        Landau operator, matching Langevin source and Poisson screening.
      - quantum_fdt: symmetrized quantum FDT applied to a supplied causal response.
      - quantum_fdt_unsymmetrized: quantum S-greater spectrum with KMS detailed balance.
        Neither FDT route quantum-corrects the supplied susceptibility.
      - polarized_kinetic_ece: magnetized kinetic ECE opacity projected onto
        two unit-flux modes, polarized LTE transfer and Berreman conversion.
      - stratified_nonparaxial_vector: 2-D vector-beam propagation through a
        z-stratified anisotropic dielectric using angular-spectrum and
        oblique Berreman solutions.
      - periodic_3d_full_wave: full-vector 3-D Bloch-periodic Maxwell FDFD solve
        with tensor dielectric coupling, magnetic fields, Poynting flux and
        source-to-absorption energy-balance diagnostics.
      - stratified_mode_conversion: coherent Jones reflection/transmission
        matrices for a supplied stratified dielectric.
      - polarized_lte_stokes: exact constant-coefficient Stokes transfer with
        dichroic Kirchhoff emissivity and Faraday rotation/conversion.
      - stratified_polarized_lte_stokes: ordered segment-by-segment polarized
        LTE ray transfer; non-commuting layer operators are not averaged.
      - cold_plasma_dielectric: local multispecies Stix R/L/S/D/P tensor.
      - cold_plasma_magnetoionic_modes: two local cold-plasma n^2 branches
        and electric polarization eigenvectors for a supplied wave-normal.
      - cold_plasma_absorbed_power_density: local time-averaged collisional
        wave absorption from the Hermitian dissipative dielectric part.
      - cold_plasma_mode_radiative_coefficients: modal attenuation from
        absorbed power divided by modal Poynting flux, with optional explicit-LTE
        Kirchhoff emissivity j_nu,m=alpha_nu,m B_nu(T_e).
      - cold_plasma_mode_group_velocity: local 3-D group/ray velocity and
        wave-normal walk-off from the Hellmann-Feynman derivative of the
        anisotropic Maxwell dispersion matrix; collisionless simple modes only.
      - thomson_multiview_3d_evdf: non-negative 3-D velocity-space inversion from multiple measured scattering views; the data-only stacked kernel rank gates whether a 3-D EVDF is reportable.
      - thomson_relativistic_exact_flow_projection: exact relativistic Doppler inversion for one fitted q-parallel bulk-flow projection (not a distribution mean or 3-D flow).
      - thomson_n_population_relativistic_fit: measured-spectrum inversion of a finite 2–8 component relativistic Maxwell-Juttner mixture, with ordered temperatures, simplex number fractions, optional component-specific q-flow, measured response, structured background, joint measurement/model-discrepancy covariance and a scaled-Jacobian identifiability gate.
      - thomson_kinetic_background_fit: finite-bin Kappa or bin-averaged relativistic
        Maxwell-Juttner physical fit with detector background, instrument response,
        measurement/model-discrepancy/systematic covariance and joint nuisance covariance.
      - thomson_projected_evdf: response-aware non-negative inversion of the
        measured optical Thomson electron feature to its 1-D q-projected VDF;
        absolute density and the unobserved 3-D distribution are not inferred.
      - thomson_relativistic_projected_evdf: exact relativistic projected
        inversion in the Thomson limit, or exact Compton four-momentum
        kinematics with polarized Klein-Nishina weighting when recoil is enabled.
      - thomson_psf_profile_inverse: explicit PSF/mixing-aware regularized spatial inversion.
      - thomson_joint_spatial_spectra: joint wavelength-resolved inversion of measured channel spectra through a supplied spatial mixing operator, with shared physical profiles and joint covariance.
      - thomson_normalized_poloidal_flux: equilibrium-supplied mapping of Thomson channel fluxes to rho_pol with full joint flux/axis/separatrix covariance propagation.
      - gyrotropic_bi_maxwellian_anisotropy: GLS inversion of directional Thomson temperatures at independent magnetic pitch angles.
      - thomson_full_temperature_tensor_spectral_fit: joint raw-spectrum inversion of a general non-gyrotropic positive-definite 3-D temperature tensor with either Gaussian Maxwellian or ellipsoidal Kappa closure; Kappa uses exact finite-bin Student-t marginals, fitted/fixed kappa and a projected superluminal-tail validity gate.
      - optical_collective_kappa_multi_population_inference: measured-spectrum optical collective dielectric fit with explicit electron and ion populations, species-resolved Landau susceptibilities and shared dielectric/FDT screening.
      - thomson_three_dimensional_flow: covariance-weighted GLS inversion of measured flow projections from three or more independent scattering-vector directions; rank-3 geometry is additionally gated by the condition number of the whitened design, and overdetermined view sets are checked against the shared-flow chi-square residual.
      - plasma_charge_state_moments: Zeff, ion charge balance and net charge
        density from explicit multispecies densities and charge states.

    No physical model is substituted automatically. Equilibrium, geometry,
    dielectric, mode-basis and calibration inputs remain explicit caller data.
    The returned limitations and convergence gates are preserved.
    """
    if not isinstance(model, str):
        raise ValueError("model must be an explicit supported model name")
    if model == "landau_fdt_full_em_spectrum":
        solver = analyze_landau_fdt_full_em_spectrum
    elif model == "landau_fdt_full_em_spectrum_converged":
        solver = analyze_landau_fdt_full_em_spectrum_converged
    elif model == "thomson_path_averaged_density":
        solver = infer_thomson_path_averaged_density
    elif model == "thomson_multiview_3d_evdf":
        solver = reconstruct_multiview_3d_evdf
    elif model == "thomson_kinetic_background_fit":
        from .thomson_kinetic_background import fit_noncollective_kinetic_thomson_with_background
        solver = fit_noncollective_kinetic_thomson_with_background
    elif model == "thomson_n_population_relativistic_fit":
        from .thomson_n_population import fit_n_population_relativistic_thomson_spectrum
        solver = fit_n_population_relativistic_thomson_spectrum
    elif model == "relativistic_kinetic_ece_profile":
        solver = solve_relativistic_kinetic_ece_profile
    elif model == "relativistic_kinetic_ece":
        solver = relativistic_kinetic_ece_absorption_coefficient
    elif model == "microwave_background":
        solver = analyze_cts_microwave_background
    elif model == "maxwellian_electrostatic_susceptibility":
        solver = maxwellian_electron_susceptibility
    elif model == "kappa_electrostatic_susceptibility":
        solver = kappa_electron_susceptibility
    elif model == "kappa_dispersion_function":
        solver = kappa_dispersion_function
    elif model == "bi_kappa_electrostatic_susceptibility":
        solver = bi_kappa_electron_susceptibility
    elif model == "cts_collective_kappa_spectrum":
        solver = collective_kappa_spectral_density
    elif model == "cts_collective_kappa_inference":
        solver = fit_collective_kappa
    elif model == "cts_tabulated_evdf_susceptibility":
        solver = generalized_electron_susceptibility
    elif model == "cts_tabulated_evdf_landau":
        solver = collective_tabulated_electron_spectral_density
    elif model == "cts_multi_population_tabulated_evdf_landau":
        solver = collective_multi_population_tabulated_electron_spectral_density
    elif model == "cts_non_gyrotropic_gaussian_tensor_spectrum":
        solver = collective_non_gyrotropic_gaussian_electron_spectrum
    elif model == "cts_electrostatic_measured_spectrum_inference":
        # Dashboard/API contract names the calibrated observation
        # measured_spectrum; the electrostatic fitter's internal keyword is
        # spectrum. Normalize the alias once at this public boundary.
        if "measured_spectrum" in kwargs and "spectrum" in kwargs:
            raise ValueError("Supply the measured CTS spectrum once, as 'measured_spectrum' or 'spectrum', not both")
        if "measured_spectrum" in kwargs:
            kwargs["spectrum"] = kwargs.pop("measured_spectrum")
        solver = analyze_cts_spectrum
    elif model == "cts_multi_ion_flow_inference":
        solver = fit_cts_ion_species_flows
    elif model == "multispecies_thermal_collision_matrix":
        solver = multispecies_thermal_coulomb_collision_matrix
    elif model == "landau_fdt_cts":
        solver = analyze_landau_fdt_cts
    elif model == "cts_conserving_bgk_fdt":
        solver = analyze_conserving_bgk_cts
    elif model == "quantum_fdt":
        solver = quantum_fdt_dynamic_structure_factor
    elif model == "quantum_fdt_unsymmetrized":
        solver = quantum_unsymmetrized_dynamic_structure_factor
    elif model in {"landau_fdt_full_em_response", "landau_fdt_full_em_constitutive"}:
        # Keep the documented response alias executable through the same
        # assembled Landau-operator + matching-FDT constitutive solver.
        solver = analyze_landau_fdt_full_em_constitutive
    elif model == "landau_fdt_full_em_field_fluctuations":
        solver = landau_full_em_electric_field_fluctuation_spectrum
    elif model == "polarized_kinetic_ece":
        solver = solve_polarized_kinetic_ece_with_mode_conversion
    elif model == "stratified_nonparaxial_vector":
        solver = propagate_stratified_nonparaxial_vector_field
    elif model == "periodic_3d_full_wave":
        solver = solve_3d_periodic_full_wave
    elif model == "polarized_lte_stokes":
        solver = propagate_polarized_lte_stokes
    elif model == "stratified_polarized_lte_stokes":
        solver = propagate_stratified_polarized_lte_stokes
    elif model == "stratified_mode_conversion":
        solver = solve_stratified_mode_conversion
    elif model == "cold_plasma_dielectric":
        solver = cold_plasma_dielectric_tensor
    elif model == "cold_plasma_magnetoionic_modes":
        solver = cold_plasma_magnetoionic_modes
    elif model == "cold_plasma_absorbed_power_density":
        solver = cold_plasma_absorbed_power_density
    elif model == "cold_plasma_mode_radiative_coefficients":
        solver = cold_plasma_mode_radiative_coefficients
    elif model == "cold_plasma_mode_group_velocity":
        solver = cold_plasma_mode_group_velocity
    elif model == "thomson_temperature_from_spectral_width":
        solver = thomson_electron_temperature_from_spectral_width
    elif model == "thomson_relativistic_exact_flow_projection":
        solver = thomson_velocity_projection_relativistic_exact
    elif model == "thomson_projected_evdf":
        solver = reconstruct_thomson_evdf
    elif model == "thomson_projected_bi_kappa_evdf":
        solver = project_gyrotropic_bi_kappa_electron_vdf
    elif model == "thomson_relativistic_projected_evdf":
        solver = reconstruct_relativistic_projected_thomson_evdf
    elif model == "thomson_psf_profile_inverse":
        solver = reconstruct_psf_aware_thomson_profile
    elif model == "thomson_joint_spatial_spectra":
        solver = reconstruct_joint_thomson_spatial_spectra
    elif model == "thomson_normalized_poloidal_flux":
        solver = map_thomson_channels_to_normalized_poloidal_flux
    elif model == "gyrotropic_bi_maxwellian_anisotropy":
        solver = infer_gyrotropic_bi_maxwellian_temperatures
    elif model == "gyrotropic_bi_maxwellian_spectral_fit":
        solver = fit_gyrotropic_bi_maxwellian_thomson
    elif model == "gyrotropic_bi_kappa_spectral_fit":
        solver = fit_gyrotropic_bi_kappa_thomson
    elif model == "thomson_three_dimensional_flow":
        solver = infer_three_dimensional_flow_from_thomson_projections
    elif model == "thomson_full_temperature_tensor":
        solver = infer_full_temperature_tensor_from_thomson_projections
    elif model == "thomson_full_temperature_tensor_spectral_fit":
        from .thomson_temperature_tensor import fit_general_temperature_tensor_thomson_spectra
        solver = fit_general_temperature_tensor_thomson_spectra
    elif model == "plasma_charge_state_moments":
        solver = plasma_charge_state_moments
    else:
        supported = ", ".join(sorted(ADVANCED_THOMSON_PHYSICS_MODELS))
        raise ValueError(
            f"model must be one of: {supported}; "
            "no automatic model substitution is performed"
        )
    # Geometry-covariance and reporting-gate inputs belong to the derived
    # 3-D flow closure, not to the per-view spectral forward-fit signature.
    # Split them at this public API boundary so they reach the correct solver.
    flow_geometry_options = {}
    solver_kwargs = kwargs
    if model in {"gyrotropic_bi_maxwellian_spectral_fit", "gyrotropic_bi_kappa_spectral_fit"}:
        solver_kwargs = dict(kwargs)
        for option_name in (
            "scattering_vector_covariance",
            "projection_geometry_cross_covariance",
            "maximum_geometry_condition_number",
            "minimum_projection_consistency_p_value",
        ):
            if option_name in solver_kwargs:
                flow_geometry_options[option_name] = solver_kwargs.pop(option_name)

    result = solver(*args, **solver_kwargs)
    if not isinstance(result, Mapping):
        raise TypeError("advanced physics solver must return a mapping")
    result = dict(result)

    # A gyrotropic multi-view fit already estimates one q-parallel flow
    # projection per view. Close the physical path by solving v_q = q_hat.v
    # jointly in 3-D, using the *fit-derived cross-view projection covariance*.
    # Never replace a missing/singular covariance with an invented identity.
    if model in {"gyrotropic_bi_maxwellian_spectral_fit", "gyrotropic_bi_kappa_spectral_fit"}:
        directions = kwargs.get("scattering_vector_unit")
        if directions is not None:
            projections = np.asarray(result.get("electron_velocity_projection_m_s"), dtype=float)
            projection_covariance = result.get("flow_projection_covariance_m2_s2")
            if projection_covariance is None:
                result["three_dimensional_flow_inference"] = {
                    "identifiability": "not_reported",
                    "reportable_as_inferred_parameter": False,
                    "electron_velocity_vector_m_s": None,
                    "electron_velocity_covariance_m2_s2": None,
                    "reason": (
                        "The spectral-fit Jacobian nullspace overlaps one or more "
                        "view-flow projections; correlated flow uncertainty is not identifiable."
                    ),
                }
            else:
                projection_covariance = np.asarray(projection_covariance, dtype=float)
                if projection_covariance.shape != (projections.size, projections.size):
                    raise ValueError("gyrotropic fit flow-projection covariance does not match the fitted views")
                try:
                    result["three_dimensional_flow_inference"] = infer_three_dimensional_flow_from_thomson_projections(
                        projections,
                        directions,
                        projection_covariance_m2_s2=projection_covariance,
                        scattering_vector_covariance=flow_geometry_options.get("scattering_vector_covariance"),
                        projection_geometry_cross_covariance=flow_geometry_options.get(
                            "projection_geometry_cross_covariance"
                        ),
                        maximum_geometry_condition_number=flow_geometry_options.get(
                            "maximum_geometry_condition_number", 1.0e3
                        ),
                        minimum_projection_consistency_p_value=flow_geometry_options.get(
                            "minimum_projection_consistency_p_value", 1.0e-3
                        ),
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    result["three_dimensional_flow_inference"] = {
                        "identifiability": "not_reported",
                        "reportable_as_inferred_parameter": False,
                        "electron_velocity_vector_m_s": None,
                        "electron_velocity_covariance_m2_s2": None,
                        "reason": f"3-D flow covariance/geometry solve was not reportable: {exc}",
                    }
    return {
        **result,
        "public_physics_entrypoint": model,
        "automatic_model_substitution": False,
    }


__all__ = [
    "kappa_electron_susceptibility",
    "kappa_dispersion_function",
    "maxwellian_electron_susceptibility",
    "collective_tabulated_electron_spectral_density",
    "collective_non_gyrotropic_gaussian_electron_spectrum",
    "quantum_fdt_dynamic_structure_factor",
    "quantum_unsymmetrized_dynamic_structure_factor",
    "analyze_cts_microwave_background",
    "analyze_landau_fdt_cts",
    "fit_cts_ion_species_flows",
    "analyze_landau_fdt_full_em_constitutive",
    "analyze_landau_fdt_full_em_spectrum",
    "propagate_stratified_nonparaxial_vector_field",
    "solve_3d_periodic_full_wave",
    "solve_polarized_kinetic_ece_with_mode_conversion",
    "propagate_polarized_lte_stokes",
    "propagate_stratified_polarized_lte_stokes",
    "cold_plasma_dielectric_tensor",
    "cold_plasma_magnetoionic_modes",
    "cold_plasma_absorbed_power_density",
    "cold_plasma_mode_radiative_coefficients",
    "cold_plasma_mode_group_velocity",
    "plasma_charge_state_moments",
    "thomson_velocity_projection_relativistic_exact",
    "reconstruct_thomson_evdf",
    "reconstruct_relativistic_projected_thomson_evdf",
    "relativistic_projected_compton_kinematics",
    "reconstruct_psf_aware_thomson_profile",
    "infer_gyrotropic_bi_maxwellian_temperatures",
    "fit_gyrotropic_bi_maxwellian_thomson",
    "fit_gyrotropic_bi_kappa_thomson",
    "infer_three_dimensional_flow_from_thomson_projections",
    "infer_full_temperature_tensor_from_thomson_projections",
    "run_advanced_thomson_physics",
]
