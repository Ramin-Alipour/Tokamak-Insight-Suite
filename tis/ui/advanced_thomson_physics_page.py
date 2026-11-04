"""Streamlit entry point for explicit advanced Analysis-1 physics solvers.

The numerical kernels live in tokamak_analysis.cts_full_physics. This page only
collects already-calibrated/user-prepared arrays and forwards them unchanged;
it performs no calibration, geometry reconstruction, or hidden model fallback.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
import numpy as np

from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics


MODEL_LABELS = {
    "kappa_electrostatic_susceptibility": "CTS — full-tail collisionless isotropic-Kappa Landau susceptibility",
    "thomson_projected_evdf": "Optical Thomson — response-aware 1-D projected electron velocity distribution inversion",
    "thomson_relativistic_projected_evdf": "Optical Thomson — relativistic projected EVDF (Thomson or Compton/Klein–Nishina)",
    "thomson_psf_profile_inverse": "Thomson profile — PSF/mixing-aware regularized spatial inversion",
    "thomson_three_dimensional_flow": "Optical Thomson — 3-D electron flow from multiple scattering views",
    "gyrotropic_bi_maxwellian_anisotropy": "Optical Thomson — gyrotropic electron temperature anisotropy from multiple pitch angles",
    "landau_fdt_cts": "Multispecies Landau operator + matching FDT + Poisson-screened electrostatic CTS",
    "landau_fdt_full_em_constitutive": "Full-EM CTS — conserving Landau–FDT Cartesian current-response kernel (not a three-wave spectrum)",
    "polarized_kinetic_ece": "Magnetized kinetic ECE opacity + two-mode polarized transfer + stratified mode conversion",
    "stratified_nonparaxial_vector": "Non-paraxial vector angular spectrum + anisotropic Berreman propagation",
    "stratified_mode_conversion": "Coherent oblique-incidence Berreman/Jones mode conversion",
}

EXAMPLES = {
    "thomson_relativistic_projected_evdf": {
        "wavelength_nm": np.linspace(499.8, 500.2, 81).tolist(),
        "signal": (2.0 + 100.0*np.exp(-0.5*((np.linspace(499.8, 500.2, 81)-500.0)/0.07)**2)).tolist(),
        "laser_wavelength_nm": 500.0,
        "scattering_angle_rad": 1.5707963267948966,
        "instrument_response_matrix": np.eye(81).tolist(),
        "background_signal": np.full(81, 2.0).tolist(),
        "velocity_grid_m_s": np.linspace(-80000.0, 80000.0, 81).tolist(),
        "include_compton_recoil": False,
        "maximum_recoil_correction_fraction": 0.01,
        "incident_polarization": [0.0, 0.0, 1.0],
        "regularization_strength": 0.001
    },
    "thomson_psf_profile_inverse": {
        "measured_channel_values": [2.0, 3.0, 4.0, 3.0, 2.0],
        "spatial_mixing_matrix": [[0.85,0.10,0.00,0.00,0.00],[0.10,0.80,0.10,0.00,0.00],[0.00,0.10,0.80,0.10,0.00],[0.00,0.00,0.10,0.80,0.10],[0.00,0.00,0.00,0.10,0.85]],
        "regularization_strength": 0.01,
        "measurement_uncertainty": [0.1,0.1,0.1,0.1,0.1]
    },
    "thomson_projected_evdf": {
        "wavelength_nm": np.linspace(499.98, 500.02, 41).tolist(),
        "signal": (2.0 + 100.0*np.exp(-0.5*((np.linspace(499.98, 500.02, 41)-500.001)/0.003)**2)).tolist(),
        "laser_wavelength_nm": 500.0,
        "scattering_angle_rad": 1.5707963267948966,
        "instrument_response_matrix": np.eye(41).tolist(),
        "background_signal": np.full(41, 2.0).tolist(),
        "regularization_strength": 0.001,
        "velocity_limit_m_s": 8000.0,
        "velocity_points": 81
    },
    "kappa_electrostatic_susceptibility": {
        "v": np.linspace(-5.0e7, 5.0e7, 2001).tolist(),
        "phase_velocity_m_s": 1.0e6,
        "electron_density_m3": 2.0e18,
        "electron_temperature_eV": 250.0,
        "kappa": 4.0,
        "wavenumber_m_inverse": 120.0,
        "pole_imaginary_part_m_s": 2.0e3,
        "minimum_velocity_grid_probability_coverage": 0.999
    },
    "thomson_three_dimensional_flow": {
        "velocity_projection_m_s": [100000.0, 200000.0, 300000.0],
        "scattering_vector_unit": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "projection_covariance_m2_s2": [[1000000.0, 0.0, 0.0], [0.0, 1000000.0, 0.0], [0.0, 0.0, 1000000.0]],
        "scattering_vector_covariance": (1.0e-6*np.eye(9)).tolist()
    },
    "gyrotropic_bi_maxwellian_anisotropy": {
        "projected_temperature_eV": [180.0, 105.0, 80.0],
        "magnetic_pitch_cosine": [1.0, 0.5, 0.0],
        "projected_temperature_covariance_eV2": [[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 25.0]],
        "magnetic_pitch_cosine_covariance": [[0.0001, 0.0, 0.0], [0.0, 0.0001, 0.0], [0.0, 0.0, 0.0001]]
    },
    "landau_fdt_cts": {
        "angular_frequency_rad_s": [1.0e5, 2.0e5],
        "normalized_velocity_x": [-4.0, 0.0, 4.0],
        "normalized_velocity_y": [-4.0, 0.0, 4.0],
        "normalized_velocity_z": [-4.0, 0.0, 4.0],
        "species": [
            {"name": "electrons", "density_m3": 1.0e18, "mass_kg": 9.1093837015e-31, "charge_C": -1.602176634e-19},
            {"name": "ions", "density_m3": 1.0e18, "mass_kg": 1.67262192369e-27, "charge_C": 1.602176634e-19}
        ],
        "common_temperature_J": 1.602176634e-17,  # 100 eV in joules
        "wave_number_m_inverse": 0.001,
        "coulomb_logarithm": 12.0
    },
    "landau_fdt_full_em_constitutive": {
        "angular_frequency_rad_s": [1.0e5, 2.0e5],
        "normalized_velocity_x": [-4.0, 0.0, 4.0],
        "normalized_velocity_y": [-4.0, 0.0, 4.0],
        "normalized_velocity_z": [-4.0, 0.0, 4.0],
        "species": [
            {"name": "electrons", "density_m3": 1.0e18, "mass_kg": 9.1093837015e-31, "charge_C": -1.602176634e-19},
            {"name": "ions", "density_m3": 1.0e18, "mass_kg": 1.67262192369e-27, "charge_C": 1.602176634e-19}
        ],
        "common_temperature_J": 1.602176634e-17,
        "wave_number_m_inverse": 0.001,
        "propagation_direction": [1.0, 0.0, 0.0],
        "common_bulk_flow_m_s": [0.0, 0.0, 0.0],
        "coulomb_logarithm": 12.0
    },
    "polarized_kinetic_ece": {
        "frequency_Hz": [45.0e9],
        "segment_length_m": [0.001],
        "wavevector_rad_m": [[0.0, 0.0, 942.0]],
        "magnetic_field_T": [[0.0, 0.0, 2.0]],
        "electron_density_m3": [1.0e19],
        "electron_temperature_eV": [1000.0],
        "mode_electric_field_unit_flux": {"real": [[[27.449, 0.0], [0.0, 27.449], [0.0, 0.0]]], "imag": [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]},
        "dielectric_z_m": [0.0, 0.001],
        "dielectric_tensor_r": {"real": [[[1,0,0],[0,1,0],[0,0,1]], [[1,0,0],[0,1,0],[0,0,1]]], "imag": [[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]},
        "harmonic_order": 2,
        "quadrature_order": 16,
        "damping_rate_rad_s": 2.0e7
    },
    "stratified_nonparaxial_vector": {
        "x_m": [-0.006, -0.004, -0.002, 0.0, 0.002, 0.004, 0.006, 0.008],
        "y_m": [-0.006, -0.004, -0.002, 0.0, 0.002, 0.004, 0.006, 0.008],
        "electric_field_V_m": {"real": [[[0,0,0] for _ in range(8)] for _ in range(8)], "imag": [[[0,0,0] for _ in range(8)] for _ in range(8)]},
        "z_m": [0.0, 0.001],
        "epsilon_r": {"real": [[[1,0,0],[0,1,0],[0,0,1]], [[1,0,0],[0,1,0],[0,0,1]]], "imag": [[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]},
        "frequency_Hz": 10.0e9
    },
    "stratified_mode_conversion": {
        "omega_rad_s": 6.283185307179586e10,
        "z_m": [0.0, 0.001],
        "epsilon_r": {"real": [[[1,0,0],[0,1,0],[0,0,1]], [[1,0,0],[0,1,0],[0,0,1]]], "imag": [[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]},
        "kx_rad_m": 0.0,
        "ky_rad_m": 0.0
    }
}


def decode_complex_json(value):
    """Decode nested JSON arrays; {real: ..., imag: ...} denotes complex data."""
    if isinstance(value, Mapping):
        if set(value) == {"real", "imag"}:
            real = np.asarray(decode_complex_json(value["real"]), dtype=float)
            imag = np.asarray(decode_complex_json(value["imag"]), dtype=float)
            if real.shape != imag.shape:
                raise ValueError("Complex JSON real and imag arrays must have identical shapes.")
            return real + 1j * imag
        return {key: decode_complex_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_complex_json(item) for item in value]
    return value


def json_safe(value):
    """Convert numerical solver outputs to JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items() if key != "landau_system"}
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Sparse matrices and internal solver objects are deliberately not expanded.
    return str(value)


def render_advanced_thomson_physics_page(st) -> None:
    """Render the explicit advanced Analysis-1 solver interface."""
    st.subheader("Analysis 1 — advanced kinetic and electromagnetic physics")
    st.info(
        "These are explicit numerical solvers, not automatic fits to an arbitrary uploaded file. "
        "Supply calibrated/prepared physical inputs and geometry. No calibration, equilibrium "
        "reconstruction, or missing-input estimation is performed."
    )
    model = st.selectbox(
        "Select the physical solver",
        list(MODEL_LABELS),
        format_func=lambda key: MODEL_LABELS[key],
        key="ts_advanced_physics_model",
    )
    st.caption(
        "Enter one JSON object containing the keyword arguments for the selected solver. "
        "For complex-valued arrays use {\"real\": [...], \"imag\": [...]}; both arrays must have identical shapes. "
        "The example is an input-shape template, not experimental data or a recommended physical operating point."
    )
    raw = st.text_area(
        "Prepared physical inputs (JSON)",
        value=json.dumps(EXAMPLES[model], indent=2),
        height=360,
        key=f"ts_advanced_physics_config_{model}",
    )
    if st.button("Run selected advanced physics solver", type="primary", key="run_ts_advanced_physics"):
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("The solver configuration must be a JSON object.")
            config = decode_complex_json(parsed)
            result = run_advanced_thomson_physics(model, **config)
            safe = json_safe(result)
            st.success("The selected numerical solver completed. Review its limitations and convergence gates before interpreting results.")
            st.json(safe)
            st.download_button(
                "Export advanced physics result (JSON)",
                data=json.dumps(safe, indent=2, allow_nan=False),
                file_name=f"tis_{model}_result.json",
                mime="application/json",
                key="download_ts_advanced_physics_result",
            )
        except Exception as exc:
            st.error(f"Advanced physics solver failed: {exc}")
