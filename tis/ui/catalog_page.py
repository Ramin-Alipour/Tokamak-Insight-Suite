"""Diagnostic and formula catalogue page, separated from the app entry point."""
from __future__ import annotations
from typing import Callable, Iterable, Mapping, Any


def render_catalog_page(st: Any, diagnostics: Iterable[Mapping[str, Any]],
                        formulas: Iterable[Mapping[str, Any]],
                        diagnostic_families: Callable[[], list[str]],
                        json_download: Callable[[str, Any, str], None]) -> None:
    """Render the catalogue from canonical registry records."""
    diagnostics = list(diagnostics)
    formulas = list(formulas)
    st.title("Diagnostic Knowledge Base")
    view = st.radio("Catalog", ["Diagnostic families", "Formula reference"], horizontal=True)
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
                raw_to_physics = contract.get("raw_data_to_physics", False) and completeness == "complete"
                st.markdown(
                    f"**Observables:** {diagnostic['quantities']}\n\n"
                    f"**Quantity class (M=measured, D=derived, I=inferred):** {diagnostic['status']}\n\n"
                    f"**Execution level:** `{level}`\n\n"
                    f"**Contract completeness:** {completeness}\n\n"
                    f"**Declared inputs / outputs:** {len(inputs)} / {len(outputs)}\n\n"
                    f"**Raw data → physical result:** {'Declared' if raw_to_physics else 'Not claimed'}\n\n"
                    f"**Implementation status:** {diagnostic.get('implementation_status', 'Registry reference only')}\n\n"
                    f"**Mapped functions:** `{functions}`\n\n"
                    f"**Method:** {diagnostic['method']}\n\n"
                    f"**Limitations:** {diagnostic['caveat']}"
                )
        json_download("Export diagnostic registry JSON", entries, "tis_diagnostic_registry.json")
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
