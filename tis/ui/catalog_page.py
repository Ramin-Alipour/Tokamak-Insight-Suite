"""Diagnostic and formula catalogue page, separated from the app entry point."""
from __future__ import annotations
from typing import Callable, Iterable, Mapping, Any

from tis.catalog import maturity_label


def render_catalog_page(st: Any, diagnostics: Iterable[Mapping[str, Any]],
                        formulas: Iterable[Mapping[str, Any]],
                        diagnostic_families: Callable[[], list[str]],
                        json_download: Callable[[str, Any, str], None],
                        analyses: Iterable[Mapping[str, Any]] = ()) -> None:
    """Render the catalogue from canonical registry records."""
    diagnostics = list(diagnostics)
    formulas = list(formulas)
    analyses = list(analyses)
    st.title("Diagnostic Knowledge Base")
    view = st.radio("Catalog", ["Diagnostic families", "Analysis entry points", "Formula reference"], horizontal=True)
    query = st.text_input("Search by diagnostic, observable, formula, method or caveat", "").strip().lower()
    if view == "Diagnostic families":
        family = st.selectbox("Diagnostic family", ["All families"] + diagnostic_families())
        entries = [d for d in diagnostics
                   if (family == "All families" or d.get("family") == family)
                   and (not query or query in " ".join(str(v) for v in d.values()).lower())]
        st.write(f"{len(entries)} entries")
        for diagnostic in entries:
            with st.expander(f"{diagnostic['diagnostic']} · {diagnostic['family']}"):
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
