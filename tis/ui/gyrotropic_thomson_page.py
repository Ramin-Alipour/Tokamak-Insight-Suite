"""Measured multi-view Dashboard for gyrotropic Thomson inference.

The page deliberately consumes calibrated spectra, measured instrument response,
and explicit view geometry. It does not infer missing viewing directions or
silently promote a single-view directional temperature to 3-D anisotropy.
"""
from __future__ import annotations

import json
import numpy as np


def _read_numeric_csv(upload, label):
    if upload is None:
        return None
    try:
        import pandas as pd
        values = pd.read_csv(upload, header=None).to_numpy(dtype=float)
    except Exception as exc:
        raise ValueError(f"{label} must be a numeric CSV without a header: {exc}") from exc
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain finite numeric values")
    return values


def render_gyrotropic_thomson_page(st) -> None:
    """Render measured-spectrum bi-Maxwellian / bi-Kappa multi-view inference."""
    st.subheader("Analysis 1 — gyrotropic anisotropic optical Thomson inference")
    st.info(
        "Jointly fit calibrated spectra from independent scattering views. "
        "Parallel/perpendicular second-moment temperatures are reported separately "
        "only when the supplied pitch-angle design identifies them."
    )
    st.caption(
        "Required: one common calibrated wavelength grid, one measured response matrix, "
        "and independently measured view geometry. A single view identifies only a "
        "directional temperature, not T_parallel and T_perpendicular separately. "
        "With three linearly independent scattering directions, the API also reconstructs "
        "the 3-D flow vector from the jointly fitted projections and their full covariance. "
        "Optional scattering_vector_covariance in the geometry JSON propagates direction uncertainty."
    )
    spectrum_file = st.file_uploader(
        "Measured multi-view spectra CSV: wavelength_nm followed by one intensity column per view",
        type=["csv", "txt"], key="ts_gyro_spectra",
    )
    response_file = st.file_uploader(
        "Measured wavelength-response matrix CSV (square; no synthetic response)",
        type=["csv", "txt"], key="ts_gyro_response",
    )
    geometry_default = json.dumps({
        "magnetic_field_unit": [0.0, 0.0, 1.0],
        "scattering_vector_unit": [
            [0.0, 0.0, 1.0],
            [0.8660254037844386, 0.0, 0.5],
            [1.0, 0.0, 0.0],
        ],
        # Optional 3*n_view square Cartesian covariance can be supplied here
        # in view-major [qx,qy,qz] ordering for tangent-plane EIV propagation.
    }, indent=2)
    geometry_text = st.text_area(
        "Measured geometry (JSON)", value=geometry_default,
        height=150, key="ts_gyro_geometry",
    )
    model = st.selectbox(
        "Gyrotropic physical distribution",
        ["bi_maxwellian", "bi_kappa"],
        format_func=lambda value: "Gyrotropic bi-Maxwellian" if value == "bi_maxwellian" else "Gyrotropic finite-variance bi-Kappa",
        key="ts_gyro_model",
    )
    c1, c2 = st.columns(2)
    with c1:
        laser_nm = st.number_input("Laser wavelength (nm)", min_value=1e-9, value=1064.0, format="%.9g", key="ts_gyro_laser")
    with c2:
        angle = st.number_input("Scattering angle (degrees)", min_value=0.001, max_value=179.999, value=90.0, key="ts_gyro_angle")
    uncertainty_file = st.file_uploader(
        "Optional pointwise uncertainty CSV (one row per view, one column per wavelength)",
        type=["csv", "txt"], key="ts_gyro_uncertainty",
    )
    covariance_file = st.file_uploader(
        "Optional full measurement covariance CSV (view-major ordering)",
        type=["csv", "txt"], key="ts_gyro_covariance",
    )
    discrepancy_file = st.file_uploader(
        "Optional full model-discrepancy covariance CSV (view-major ordering)",
        type=["csv", "txt"], key="ts_gyro_discrepancy",
    )
    initial_parallel = st.number_input("Initial T_parallel (eV)", min_value=1e-6, value=800.0, key="ts_gyro_tpar")
    initial_perpendicular = st.number_input("Initial T_perpendicular (eV)", min_value=1e-6, value=400.0, key="ts_gyro_tperp")
    initial_kappa = st.number_input("Initial kappa (bi-Kappa only)", min_value=1.5001, value=4.0, key="ts_gyro_kappa")
    if st.button("Run gyrotropic multi-view inference", type="primary", key="run_ts_gyrotropic"):
        try:
            spectrum = _read_numeric_csv(spectrum_file, "Measured multi-view spectra")
            response = _read_numeric_csv(response_file, "Measured response matrix")
            if spectrum is None or response is None:
                raise ValueError("Upload both measured multi-view spectra and the measured response matrix.")
            if spectrum.ndim != 2 or spectrum.shape[1] < 2:
                raise ValueError("Spectrum CSV must contain wavelength_nm plus at least one view-intensity column.")
            wavelength = spectrum[:, 0]
            measured = spectrum[:, 1:].T
            if wavelength.size < 9 or np.any(np.diff(wavelength) <= 0.0):
                raise ValueError("The common wavelength grid must contain at least nine strictly increasing samples.")
            n_view, n_bin = measured.shape
            if response.shape != (n_bin, n_bin):
                raise ValueError(f"Measured response matrix must have shape ({n_bin}, {n_bin}).")
            geometry = json.loads(geometry_text)
            if not isinstance(geometry, dict) or "magnetic_field_unit" not in geometry or "scattering_vector_unit" not in geometry:
                raise ValueError("Geometry JSON must contain magnetic_field_unit and scattering_vector_unit.")
            magnetic = np.asarray(geometry["magnetic_field_unit"], dtype=float)
            q_vectors = np.asarray(geometry["scattering_vector_unit"], dtype=float)
            q_covariance = geometry.get("scattering_vector_covariance")
            if q_covariance is not None:
                q_covariance = np.asarray(q_covariance, dtype=float)
                expected_geometry_covariance_shape = (3 * n_view, 3 * n_view)
                if q_covariance.shape != expected_geometry_covariance_shape or not np.all(np.isfinite(q_covariance)):
                    raise ValueError(
                        "scattering_vector_covariance in geometry JSON must be a finite "
                        f"{expected_geometry_covariance_shape} Cartesian covariance matrix in view-major order."
                    )
            if magnetic.shape not in {(3,), (n_view, 3)} or q_vectors.shape != (n_view, 3):
                raise ValueError(f"Geometry must provide one or {n_view} magnetic-field unit vector(s) and exactly {n_view} scattering-vector unit vectors.")
            uncertainty = _read_numeric_csv(uncertainty_file, "Pointwise uncertainty")
            covariance = _read_numeric_csv(covariance_file, "Measurement covariance")
            discrepancy = _read_numeric_csv(discrepancy_file, "Model-discrepancy covariance")
            if uncertainty is not None and uncertainty.shape != measured.shape:
                raise ValueError(f"Pointwise uncertainty must have shape {measured.shape}.")
            total_size = n_view * n_bin
            if covariance is not None and covariance.shape != (total_size, total_size):
                raise ValueError(f"Measurement covariance must have shape ({total_size}, {total_size}).")
            if discrepancy is not None and discrepancy.shape != (total_size, total_size):
                raise ValueError(f"Model-discrepancy covariance must have shape ({total_size}, {total_size}).")
            if covariance is not None and uncertainty is not None:
                raise ValueError("Supply measurement covariance or pointwise uncertainty, not both.")
            from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics
            if model == "bi_maxwellian":
                api_model = "gyrotropic_bi_maxwellian_spectral_fit"
                options = {}
            else:
                api_model = "gyrotropic_bi_kappa_spectral_fit"
                options = {"initial_kappa": float(initial_kappa)}
            if uncertainty is not None:
                options["signal_uncertainty"] = uncertainty
            if covariance is not None:
                options["measurement_covariance"] = covariance
            if discrepancy is not None:
                options["model_discrepancy_covariance"] = discrepancy
            result = run_advanced_thomson_physics(api_model,
                wavelength, measured,
                laser_wavelength_nm=float(laser_nm),
                scattering_angle_rad=float(np.deg2rad(angle)),
                magnetic_field_unit=magnetic,
                scattering_vector_unit=q_vectors,
                scattering_vector_covariance=q_covariance,
                instrument_response_matrix=response,
                initial_temperature_parallel_eV=float(initial_parallel),
                initial_temperature_perpendicular_eV=float(initial_perpendicular),
                **options,
            )
            st.subheader("Gyrotropic inference result")
            flow_result = result.get("three_dimensional_flow_inference") or {}
            if flow_result.get("reportable_as_inferred_parameter") is True:
                st.success(
                    "The fitted view projections identify a subluminal 3-D electron-flow vector. "
                    "Its covariance includes the full cross-view spectral-fit covariance."
                )
                st.markdown("**Three-dimensional electron-flow inference**")
                st.json({
                    "electron_velocity_vector_m_s": flow_result.get("electron_velocity_vector_m_s"),
                    "electron_speed_m_s": flow_result.get("electron_speed_m_s"),
                    "electron_velocity_covariance_m2_s2": flow_result.get("electron_velocity_covariance_m2_s2"),
                    "fit_residual_projection_m_s": flow_result.get("fit_residual_projection_m_s"),
                    "reduced_chi_square": flow_result.get("reduced_chi_square"),
                    "geometry_covariance_method": flow_result.get("geometry_covariance_method"),
                })
            elif flow_result:
                st.warning(
                    "A 3-D flow vector was withheld: "
                    + str(flow_result.get("reason", "the supplied view geometry or fit covariance is insufficient."))
                )
            if result.get("reportable_as_inferred_parameter") is True:
                st.success("The multi-view data identify the reported anisotropic temperature parameters within the solver's declared model scope.")
            else:
                st.warning("The data do not identify separate parallel/perpendicular temperatures. Directional results are retained; non-identifiable anisotropy parameters are withheld.")
            st.json(result)
            st.download_button(
                "Export gyrotropic inference (JSON)",
                data=json.dumps(result, default=lambda value: np.asarray(value).tolist(), indent=2, allow_nan=False),
                file_name="tis_gyrotropic_thomson_inference.json",
                mime="application/json",
                key="download_ts_gyrotropic_result",
            )
        except Exception as exc:
            st.error(f"Gyrotropic Thomson inference failed: {exc}")
