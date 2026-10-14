"""Streamlit workflow for comparing TIS outputs with paper/device reference results."""
import pandas as pd
import streamlit as st

from tis.validation import compare_reference_results, compare_reference_series


def render_validation_page():
    st.title("Reference Results Validation")
    st.write(
        "Compare TIS outputs with independently reported paper/device results. "
        "The software never invents a tolerance: provide uncertainties or a documented tolerance. "
        "Agreement is conditional on matching the physical quantity, units, calibration and analysis settings."
    )
    comparison_type = st.radio(
        "Comparison type",
        ["Scalar results / reported parameters", "Curves / time series / spectra"],
        horizontal=True,
        key="tis_validation_type",
    )
    if comparison_type == "Scalar results / reported parameters":
        st.caption(
            "CSV columns: required reference_value, software_value; recommended parameter, unit, source; "
            "optional reference_uncertainty, software_uncertainty, absolute_tolerance, relative_tolerance."
        )
        sample = pd.DataFrame({
            "parameter": ["Example only — replace with a real result"],
            "unit": ["eV"],
            "source": ["Paper DOI / figure / table / dataset"],
            "reference_value": [10.0],
            "software_value": [10.2],
            "reference_uncertainty": [0.3],
            "software_uncertainty": [0.2],
        })
        st.download_button("Download scalar comparison CSV template", sample.to_csv(index=False).encode("utf-8"), "tis_reference_comparison_template.csv", "text/csv")
        uploaded = st.file_uploader("Upload reference-vs-TIS scalar CSV", type=["csv"], key="tis_validation_scalar_csv")
        k = st.number_input("Uncertainty coverage factor k", min_value=0.1, value=2.0, step=0.5, help="Used when both standard uncertainties are supplied.")
        if uploaded is None:
            st.info("No comparison has been run. The template values are illustrative, not an experimental benchmark.")
            return
        try:
            frame = pd.read_csv(uploaded)
            result = compare_reference_results(frame, coverage_factor=float(k))
        except Exception as exc:
            st.error(f"Cannot compare scalar results: {exc}")
            return
        counts = result["agreement_status"].value_counts().to_dict()
        a, b, c = st.columns(3)
        a.metric("Rows compared", len(result))
        b.metric("Agree within supplied criterion", int(sum(v for key, v in counts.items() if key.startswith("AGREES"))))
        c.metric("Unassessed", int(counts.get("UNASSESSED", 0)))
        st.dataframe(result, use_container_width=True)
        st.download_button("Download scalar comparison results", result.to_csv(index=False).encode("utf-8"), "tis_reference_comparison_results.csv", "text/csv")
        st.warning("Agreement under a supplied rule is not proof of physical-model validity or independent device validation.")
        return

    st.caption(
        "Curve CSV columns: x_reference, reference_value, x_software, software_value. "
        "The two curves may use different x grids; TIS interpolates the reference only inside their overlapping domain. "
        "Optional uncertainty columns are reference_uncertainty and software_uncertainty, each aligned with its own curve columns."
    )
    curve_template = pd.DataFrame({
        "x_reference": [0.0, 1.0, 2.0, 3.0],
        "reference_value": [0.0, 1.0, 4.0, 9.0],
        "reference_uncertainty": [0.1, 0.1, 0.2, 0.3],
        "x_software": [0.0, 0.5, 1.5, 3.0],
        "software_value": [0.0, 0.3, 2.3, 9.1],
        "software_uncertainty": [0.1, 0.1, 0.2, 0.3],
    })
    st.download_button("Download curve comparison CSV template", curve_template.to_csv(index=False).encode("utf-8"), "tis_curve_comparison_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload reference-curve and software-curve CSV", type=["csv"], key="tis_validation_curve_csv")
    criterion = st.selectbox(
        "Curve comparison criterion",
        ["No pass/fail criterion — report error metrics", "Combined standard uncertainty", "Absolute tolerance", "Relative tolerance"],
        key="tis_curve_criterion",
    )
    k = st.number_input("Uncertainty coverage factor k", min_value=0.1, value=2.0, step=0.5, key="tis_curve_k")
    tolerance = None
    if criterion == "Absolute tolerance":
        tolerance = st.number_input("Absolute tolerance (same units as y)", min_value=0.0, value=0.0, format="%.8g", key="tis_curve_abs_tol")
    elif criterion == "Relative tolerance":
        tolerance = st.number_input("Relative tolerance (fraction; 0.05 = 5%)", min_value=0.0, max_value=1.0, value=0.05, format="%.8g", key="tis_curve_rel_tol")
    if uploaded is None:
        st.info("No curve comparison has been run. The template values are illustrative only.")
        return
    try:
        frame = pd.read_csv(uploaded)
        if criterion == "Combined standard uncertainty":
            if not {"reference_uncertainty", "software_uncertainty"}.issubset(frame.columns):
                raise ValueError("Combined-uncertainty mode requires both uncertainty columns.")
            result = compare_reference_series(frame, coverage_factor=float(k))
        elif criterion == "Absolute tolerance":
            frame = frame.drop(columns=["reference_uncertainty", "software_uncertainty"], errors="ignore")
            result = compare_reference_series(frame, coverage_factor=float(k), absolute_tolerance=float(tolerance))
        elif criterion == "Relative tolerance":
            frame = frame.drop(columns=["reference_uncertainty", "software_uncertainty"], errors="ignore")
            result = compare_reference_series(frame, coverage_factor=float(k), relative_tolerance=float(tolerance))
        else:
            frame = frame.drop(columns=["reference_uncertainty", "software_uncertainty"], errors="ignore")
            result = compare_reference_series(frame, coverage_factor=float(k))
    except Exception as exc:
        st.error(f"Cannot compare curves: {exc}")
        return
    st.markdown("### Curve comparison summary")
    summary = result["summary"]
    a, b, c = st.columns(3)
    a.metric("Compared points", summary["n_compared_points"])
    b.metric("RMSE", f"{summary['root_mean_square_error']:.6g}")
    c.metric("Within criterion", "Not assessed" if summary["fraction_within_criterion"] is None else f"{summary['fraction_within_criterion']:.1%}")
    st.json(summary)
    st.dataframe(result["aligned_comparison"], use_container_width=True)
    st.line_chart(result["aligned_comparison"].set_index("x")[["reference_value_interpolated", "software_value"]])
    st.download_button("Download aligned curve comparison", result["aligned_comparison"].to_csv(index=False).encode("utf-8"), "tis_curve_comparison_results.csv", "text/csv")
    st.warning(summary["warning"] + " Digitization and axis-calibration uncertainty should be estimated and included; no extrapolation is used.")
