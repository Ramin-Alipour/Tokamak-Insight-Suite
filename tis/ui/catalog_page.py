"""Diagnostic and formula catalogue page, separated from the app entry point."""
from __future__ import annotations
from typing import Callable, Iterable, Mapping, Any

from tis.catalog import maturity_label


def render_catalog_page(st: Any, diagnostics: Iterable[Mapping[str, Any]],
                        formulas: Iterable[Mapping[str, Any]],
                        diagnostic_families: Callable[[], list[str]],
                        json_download: Callable[[str, Any, str], None],
                        analyses: Iterable[Mapping[str, Any]] = (),
                        analysis_families: Iterable[Mapping[str, Any]] = ()) -> None:
    """Render the catalogue from canonical registry records."""
    diagnostics = list(diagnostics)
    formulas = list(formulas)
    analyses = list(analyses)
    analysis_families = list(analysis_families)
    st.title("Diagnostic Knowledge Base")
    # Diagnostic records and callable entry points are separate registry layers.
    from collections import Counter
    import hashlib
    from tis.catalog import _DIAGNOSTICS_PATH

    diagnostic_status = Counter((row.get("execution_contract") or {}).get("completeness", "incomplete") for row in diagnostics)
    analysis_status = Counter((row.get("execution_contract") or {}).get("completeness", "incomplete") for row in analyses)
    diagnostic_levels = Counter((row.get("execution_contract") or {}).get("execution_level", "reference_only") for row in diagnostics)
    registry_sha256 = hashlib.sha256(_DIAGNOSTICS_PATH.read_bytes()).hexdigest()
    completeness_total = sum(diagnostic_status.values())
    if completeness_total != len(diagnostics):
        st.error("Diagnostic completeness counts do not reconcile with the loaded registry; do not trust this status summary.")
    if len(diagnostics) != 80:
        st.error(f"Expected exactly 80 diagnostic records but loaded {len(diagnostics)}.")
    if len(analyses) != 80 or [item.get("id") for item in analyses] != list(range(1, 81)):
        st.error("The analysis registry must contain IDs 1–80 in canonical order.")
    if [item.get("name") for item in analyses] != [item.get("diagnostic") for item in diagnostics]:
        st.error("Analysis titles/order do not match the canonical diagnostic registry.")
    diagnostic_positions = {id(row): position for position, row in enumerate(diagnostics, start=1)}
    expected_completeness = {"complete": 7, "complete_with_declared_scope": 4, "incomplete": 69}
    expected_levels = {"analysis_pipeline": 11, "limited_calculation": 17, "reference_only": 52}
    if dict(diagnostic_status) != expected_completeness or dict(diagnostic_levels) != expected_levels:
        st.warning(
            "Loaded registry metadata differs from the reconciled main-branch baseline "
            "(completeness 7/4/69; execution levels 11/17/52). Verify the deployed commit and registry SHA-256."
        )
    st.caption(
        f"Canonical registry: {len(diagnostics)} diagnostic records · {len(analyses)} callable analysis entry points. "
        "Completeness describes declared software scope, not independent experimental validation."
    )
    st.caption(f"Registry SHA-256: `{registry_sha256}`")
    st.write(
        "Execution levels — " +
        " · ".join(f"{key}: {value}" for key, value in sorted(diagnostic_levels.items()))
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Diagnostic records**")
        st.write(
            f"Complete: {diagnostic_status.get('complete', 0)} · "
            f"Scoped complete: {diagnostic_status.get('complete_with_declared_scope', 0)} · "
            f"Incomplete: {diagnostic_status.get('incomplete', 0)}"
        )
    with right:
        st.markdown("**Callable analysis entry points**")
        st.write(
            f"Complete: {analysis_status.get('complete', 0)} · "
            f"Scoped complete: {analysis_status.get('complete_with_declared_scope', 0)} · "
            f"Incomplete: {analysis_status.get('incomplete', 0)}"
        )
    if analysis_families and (len(analysis_families) != 20 or sorted(i for family in analysis_families for i in family.get("entry_ids", [])) != list(range(1, 81))):
        st.error("Primary-analysis family mapping is invalid; verify ANALYSIS_FAMILIES.json before relying on the grouped catalogue.")
    st.caption("Primary navigation: 20 analysis families · underlying callable entry points: 80")
    view = st.radio("Catalog", ["20 primary analyses", "Diagnostic families", "Analysis entry points", "Formula reference"], horizontal=True)
    query = st.text_input("Search by diagnostic, observable, formula, method or caveat", "").strip().lower()
    if view == "20 primary analyses":
        analysis_by_id = {item["id"]: item for item in analyses}
        visible_families = []
        for family in analysis_families:
            children = [analysis_by_id[i] for i in family.get("entry_ids", []) if i in analysis_by_id]
            haystack = " ".join([str(family), *(str(item) for item in children)]).lower()
            if not query or query in haystack:
                visible_families.append((family, children))
        st.write(f"{len(visible_families)} primary analysis families")
        for family, children in visible_families:
            with st.expander(f"{family['id']:02d} · {family['name']} · {len(children)} subanalyses"):
                st.caption(family.get("description", ""))
                if family.get("category") == "instrumentation_support":
                    st.info("Shared instrumentation and measurement-support toolkit; not an additional physical diagnostic family.")
                rows = [{
                    "Original ID": item["id"],
                    "Subanalysis / workflow": item["name"],
                    "Callable": f"tokamak_analysis.{item['module']}.{item['function']}",
                    "Maturity": maturity_label(item),
                } for item in children]
                if rows:
                    st.dataframe(rows, width="stretch", hide_index=True)
        json_download("Export 20 primary analysis families", analysis_families, "tis_primary_analysis_families.json")
        return
    if view == "Diagnostic families":
        family = st.selectbox("Diagnostic family", ["All families"] + diagnostic_families())
        entries = [d for d in diagnostics
                   if (family == "All families" or d.get("family") == family)
                   and (not query or query in " ".join(str(v) for v in d.values()).lower())]
        st.write(f"{len(entries)} entries")
        for diagnostic in entries:
            with st.expander(f"{diagnostic_positions[id(diagnostic)]:02d} · {diagnostic['diagnostic']} · {diagnostic['family']}"):
                contract = diagnostic.get("execution_contract", {})
                level = contract.get("execution_level", "reference_only")
                completeness = contract.get("completeness", "incomplete")
                inputs = contract.get("input_specs", [])
                outputs = contract.get("output_specs", [])
                functions = ", ".join(diagnostic.get("implementation_functions", []))
                if not functions:
                    functions = "No direct callable workflow is mapped; registry description only."
                raw_to_physics = contract.get("raw_data_to_physics", False) and completeness in {"complete", "complete_with_declared_scope"}
                st.markdown(
                    f"**Observables:** {diagnostic['quantities']}\n\n"
                    f"**Quantity class (M=measured, D=derived, I=inferred):** {diagnostic['status']}\n\n"
                    f"**Execution level:** `{level}`\n\n"
                    f"**Contract completeness:** {completeness}\n\n"
                    f"**Maturity level:** {maturity_label(diagnostic)}\n\n"
                    f"**Declared inputs / outputs:** {len(inputs)} / {len(outputs)}\n\n"
                    f"**Raw data → physical result:** {'Declared' if raw_to_physics else 'Not claimed'}\n\n"
                    f"**Implementation status:** {diagnostic.get('implementation_status', 'Registry reference only')}\n\n"
                    f"**Mapped functions:** `{functions}`\n\n"
                    f"**Method:** {diagnostic['method']}\n\n"
                    f"**Limitations:** {diagnostic['caveat']}"
                )
        json_download("Export diagnostic registry JSON", entries, "tis_diagnostic_registry.json")
        return
    if view == "Analysis entry points":
        entries = [item for item in analyses
                   if not query or query in " ".join(str(v) for v in item.values()).lower()]
        st.write(f"{len(entries)} callable analysis entry points")
        for analysis in entries:
            contract = analysis.get("execution_contract", {})
            with st.expander(f"{analysis['id']:02d} · {analysis['name']}"):
                st.markdown(
                    f"**Callable:** `{analysis['module']}.{analysis['function']}`\n\n"
                    f"**Execution level:** `{contract.get('execution_level', 'reference_only')}`\n\n"
                    f"**Contract completeness:** {contract.get('completeness', 'incomplete')}\n\n"
                    f"**Maturity level:** {maturity_label(analysis)}\n\n"
                    f"**Declared input / output specifications:** {len(contract.get('input_specs', []))} / {len(contract.get('output_specs', []))}\n\n"
                    f"**Raw data → physical result:** {'Declared' if contract.get('raw_data_to_physics', False) and contract.get('completeness') in {'complete', 'complete_with_declared_scope'} else 'Not claimed'}\n\n"
                    f"**Limitations:** {'; '.join(contract.get('limitations', [])) or 'Not yet documented'}"
                )
        json_download("Export analysis entry-point contracts JSON", entries, "tis_analysis_contracts.json")
        return
    entries = [formula for formula in formulas
               if not query or query in " ".join(str(v) for v in formula.values()).lower()]
    st.write(f"{len(entries)} formula entries")
    for formula in entries:
        with st.expander(formula["name"]):
            execution = formula.get("execution_contract", {})
            st.markdown(
                f"**Equation:** `{formula['equation']}`\n\n"
                f"**Inputs:** {formula['inputs']}\n\n"
                f"**Output:** {formula['output']}\n\n"
                f"**Status:** {formula.get('status', 'unspecified')}\n\n"
                f"**Implementation link:** {execution.get('implementation_link_status', 'not_mapped')}\n\n"
                f"**Assumptions:** {formula['assumptions']}\n\n"
                f"**Execution note:** {execution.get('note', 'No implementation link declared.')}"
            )
    json_download("Export formula registry JSON", entries, "tis_formula_registry.json")
