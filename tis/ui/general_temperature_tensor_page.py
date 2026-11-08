"""Dashboard for raw-spectrum, non-gyrotropic 3-D Thomson temperature-tensor inference."""
from __future__ import annotations

import json
import numpy as np


def _numeric_csv(upload, label):
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


def render_general_temperature_tensor_page(st) -> None:
    st.subheader("Analysis 1 — general non-gyrotropic 3-D Thomson temperature tensor")
    st.info(
        "This workflow fits raw calibrated spectra jointly. It does not assume a magnetic "
        "axis or gyrotropy. Six independent q-vector dyads are required to identify the "
        "six components of a symmetric 3-D temperature tensor."
    )
    st.caption(
        "Forward model: T_q = q_hat.T @ T_e @ q_hat; sigma_v^2 = e*T_q/m_e; "
        "lambda_c = lambda_L*(1 - 2*sin(theta/2)*u_q/c). Maxwellian projections "
        "are Gaussian; the ellipsoidal 3-D Kappa closure projects exactly to a "
        "Student-t marginal with nu=2*kappa-1. The measured response acts on "
        "bin-integrated spectral power. Neither closure is an unrestricted EVDF."
    )
    distribution_model=st.selectbox(
        "3-D electron distribution closure",["maxwellian","kappa"],
        index=0,key="ts_tensor_distribution_model",
    )
    initial_kappa_index=None
    fit_kappa_index=False
    kappa_bounds=None
    maximum_projected_superluminal_tail_probability=1.0e-6
    if distribution_model=="kappa":
        k1,k2=st.columns(2)
        with k1:
            initial_kappa_index=st.number_input(
                "Initial Kappa index (>1.5)",min_value=1.5001,max_value=100.0,
                value=4.0,step=0.1,key="ts_tensor_kappa_initial",
            )
        with k2:
            fit_kappa_index=st.checkbox(
                "Fit Kappa index jointly with the tensor",value=True,
                key="ts_tensor_fit_kappa",
            )
        if fit_kappa_index:
            kb1,kb2=st.columns(2)
            with kb1:
                kappa_lower=st.number_input(
                    "Kappa lower bound",min_value=1.5001,max_value=99.0,
                    value=2.0,step=0.1,key="ts_tensor_kappa_lower",
                )
            with kb2:
                kappa_upper=st.number_input(
                    "Kappa upper bound",min_value=1.5002,max_value=100.0,
                    value=12.0,step=0.1,key="ts_tensor_kappa_upper",
                )
            kappa_bounds=[float(kappa_lower),float(kappa_upper)]
        maximum_projected_superluminal_tail_probability=st.number_input(
            "Maximum projected superluminal tail probability (non-relativistic validity gate)",
            min_value=1.0e-12,max_value=0.1,value=1.0e-6,format="%.2e",
            key="ts_tensor_kappa_superluminal_tail_limit",
        )
    spectra_file = st.file_uploader(
        "Measured spectra CSV: wavelength_nm followed by one intensity column per view",
        type=["csv", "txt"], key="ts_tensor_spectra",
    )
    response_file = st.file_uploader(
        "Measured wavelength response matrix CSV", type=["csv", "txt"],
        key="ts_tensor_response",
    )
    geometry_default = json.dumps({"scattering_vector_unit": [
        [1,0,0], [0,1,0], [0,0,1], [0.7071067812,0.7071067812,0],
        [0.7071067812,0,0.7071067812], [0,0.7071067812,0.7071067812],
        [0.5773502692,0.5773502692,0.5773502692]
    ]}, indent=2)
    geometry_text = st.text_area("Measured scattering vectors (JSON)", geometry_default,
                                 height=190, key="ts_tensor_geometry")
    c1,c2=st.columns(2)
    with c1:
        laser=st.number_input("Laser wavelength (nm)",min_value=1e-9,value=532.0,key="ts_tensor_laser")
    with c2:
        angle=st.number_input("Scattering angle (degrees)",min_value=0.001,max_value=179.999,value=90.0,key="ts_tensor_angle")
    uncertainty_file=st.file_uploader("Optional pointwise uncertainty CSV",type=["csv","txt"],key="ts_tensor_uncertainty")
    covariance_file=st.file_uploader("Optional full measurement covariance CSV (view-major)",type=["csv","txt"],key="ts_tensor_covariance")
    geometry_covariance_file=st.file_uploader("Optional scattering-direction covariance CSV (3*n_view Cartesian tangent coordinates)",type=["csv","txt"],key="ts_tensor_geometry_covariance")
    geometry_cross_file=st.file_uploader("Optional measured-spectrum / scattering-geometry cross-covariance CSV",type=["csv","txt"],key="ts_tensor_geometry_cross_covariance")
    discrepancy_file=st.file_uploader("Optional model-discrepancy covariance CSV",type=["csv","txt"],key="ts_tensor_discrepancy")
    systematic_file=st.file_uploader("Optional systematic covariance CSV",type=["csv","txt"],key="ts_tensor_systematic")
    if st.button("Run general 3-D temperature-tensor inference",type="primary",key="run_ts_tensor"):
        try:
            table=_numeric_csv(spectra_file,"Measured spectra")
            response=_numeric_csv(response_file,"Measured response")
            if table is None or response is None:
                raise ValueError("Upload measured spectra and measured response.")
            if table.ndim!=2 or table.shape[1]<7:
                raise ValueError("At least six view-intensity columns plus wavelength are required.")
            wavelength=table[:,0]
            measured=table[:,1:].T
            geometry=json.loads(geometry_text)
            q=np.asarray(geometry.get("scattering_vector_unit"),dtype=float)
            if q.shape!=(measured.shape[0],3):
                raise ValueError("Provide one measured 3-D scattering vector per spectrum.")
            if response.shape not in {(wavelength.size,wavelength.size),(measured.shape[0],wavelength.size,wavelength.size)}:
                raise ValueError("Measured response must be a shared square matrix or one square matrix per view.")
            uncertainty=_numeric_csv(uncertainty_file,"Pointwise uncertainty")
            covariance=_numeric_csv(covariance_file,"Measurement covariance")
            discrepancy=_numeric_csv(discrepancy_file,"Model-discrepancy covariance")
            systematic=_numeric_csv(systematic_file,"Systematic covariance")
            geometry_covariance=_numeric_csv(geometry_covariance_file,"Scattering-direction covariance")
            geometry_cross=_numeric_csv(geometry_cross_file,"Spectrum/geometry cross-covariance")
            if covariance is not None and uncertainty is not None:
                raise ValueError("Supply measurement covariance or pointwise uncertainty, not both.")
            kwargs={}
            for name,value in (("signal_uncertainty",uncertainty),("measurement_covariance",covariance),
                               ("model_discrepancy_covariance",discrepancy),("systematic_covariance",systematic),
                               ("scattering_vector_covariance",geometry_covariance),
                               ("spectral_geometry_cross_covariance",geometry_cross)):
                if value is not None: kwargs[name]=value
            from tokamak_analysis.thomson_workflow import analyze_thomson_spectra
            api_model = (
                "general_temperature_tensor_maxwellian"
                if distribution_model == "maxwellian"
                else "general_temperature_tensor_kappa"
            )
            result=analyze_thomson_spectra(
                wavelength,measured,model=api_model,laser_wavelength_nm=float(laser),
                scattering_angle_rad=float(np.deg2rad(angle)),
                scattering_vector_unit=q,instrument_response_matrix=response,
                initial_temperature_tensor_eV=np.eye(3)*500.0,
                distribution_model=distribution_model,
                initial_kappa_index=initial_kappa_index,
                fit_kappa_index=fit_kappa_index,
                kappa_bounds=kappa_bounds,
                maximum_projected_superluminal_tail_probability=maximum_projected_superluminal_tail_probability,
                **kwargs)
            if result.get("reportable_as_inferred_parameter") is True:
                st.success(f"The general temperature tensor is identifiable within the selected non-relativistic {distribution_model} model.")
            else:
                st.warning("The solver withheld the tensor because the data do not identify it.")
            st.json({key:result.get(key) for key in (
                "electron_temperature_tensor_eV","principal_temperature_eigenvalues_eV",
                "distribution_model","kappa_index","kappa_index_uncertainty","kappa_fit_status",
"projected_superluminal_tail_probability_per_view","maximum_projected_superluminal_tail_probability","model_validity_status",
                "electron_velocity_projection_m_s","view_dyad_rank","view_dyad_condition_number",
                "jacobian_rank","jacobian_parameter_count","temperature_tensor_parameter_covariance_eV2",
                "covariance_provenance","covariance_status",
                "scattering_vector_covariance_used","spectral_geometry_cross_covariance_used",
                "geometry_parameter_covariance","geometry_cross_parameter_covariance",
                "geometry_covariance_method","reportable_as_inferred_parameter",
                "fitted_spectra","residual_spectra") if key in result})
            st.download_button("Export tensor inference JSON",data=json.dumps(result,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x),indent=2),
                               file_name="thomson_general_temperature_tensor.json",mime="application/json")
        except Exception as exc:
            st.error(f"General temperature-tensor inference failed: {exc}")
