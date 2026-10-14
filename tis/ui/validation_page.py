"""Streamlit workflow for comparing TIS outputs with paper/device reference results."""
import pandas as pd
import streamlit as st

from tis.validation import compare_reference_results


def render_validation_page():
    st.title("Reference Results Validation")
    st.write(
        "Compare independently reported paper/device results against values calculated by TIS. "
        "The software never invents a tolerance: provide uncertainties or a documented tolerance "
        "for each comparison. A numerical match alone does not validate the physical model."
    )
    st.markdown("### Reference comparison table")
    st.caption(
        "Upload CSV with required columns reference_value, software_value. Optional columns: "
        "parameter, unit, source, reference_uncertainty, software_uncertainty, absolute_tolerance, relative_tolerance. "
        "Use consistent units and compare the same physical quantity, shot/time window, calibration and processing definition."
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
    st.download_button("Download comparison CSV template", sample.to_csv(index=False).encode("utf-8"), "tis_reference_comparison_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload reference-vs-TIS comparison CSV", type=["csv"], key="tis_validation_csv")
    k = st.number_input("Uncertainty coverage factor k", min_value=0.1, value=2.0, step=0.5, help="Used only when both reference and software standard uncertainties are supplied. k=2 is a common approximate 95% interval for near-normal uncertainties.")
    if uploaded is None:
        st.info("No comparison has been run yet. The template is illustrative only and is not an experimental benchmark.")
        return
    try:
        frame = pd.read_csv(uploaded)
        result = compare_reference_results(frame, coverage_factor=float(k))
    except Exception as exc:
        st.error(f"Cannot compare reference results: {exc}")
        return
    counts = result["agreement_status"].value_counts().to_dict()
    a, b, c = st.columns(3)
    a.metric("Rows compared", len(result))
    b.metric("Agreement within supplied criterion", int(sum(v for key, v in counts.items() if key.startswith("AGREES"))))
    c.metric("Unassessed", int(counts.get("UNASSESSED", 0)))
    st.dataframe(result, use_container_width=True)
    st.warning("Interpretation: agreement is conditional on matching units, definitions, calibration and uncertainty/tolerance assumptions. It is not independent device validation, and graph-digitized points may carry additional digitization uncertainty.")
    st.download_button("Download comparison results CSV", result.to_csv(index=False).encode("utf-8"), "tis_reference_comparison_results.csv", "text/csv")
