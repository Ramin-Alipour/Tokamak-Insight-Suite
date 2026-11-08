"""Product-oriented dashboard view over the canonical analysis registry."""
from __future__ import annotations

from typing import Any

from tis.products import get_product, list_products


def render_product_workspace_page(st: Any) -> None:
    """Render product ownership, callable workflows, and scientific contracts."""
    products = list_products()
    st.title("Specialist Product Workspace")
    st.caption(
        "Twenty specialist products share the TIS platform. Each listed workflow "
        "is resolved through the canonical executable registry; physics remains "
        "in its domain solver module."
    )
    selected_id = st.selectbox(
        "Specialist product",
        options=[item["id"] for item in products],
        format_func=lambda product_id: (
            f"{product_id:02d} · "
            next(item["name"] for item in products if item["id"] == product_id)
        ),
        key="tis_specialist_product",
    )
    product = get_product(selected_id)
    st.subheader(product["name"])
    st.caption(
        f"Product ID: {product['product_id']} · "
        f"Family: {product.get('category', 'physical_diagnostic')} · "
        f"{product['execution_entry_point_count']} registered workflows"
    )
    st.write(product.get("description", ""))
    if product["id"] == 20:
        st.info(
            "This is a cross-cutting instrumentation-support product. Its "
            "calibration and measurement services are shared dependencies, not "
            "a substitute for diagnostic-specific physical inference."
        )
    rows = []
    for entry in product["entry_points"]:
        contract = entry.get("execution_contract") or {}
        rows.append({
            "ID": entry["id"],
            "Workflow": entry["name"],
            "Callable": f"tokamak_analysis.{entry['module']}.{entry['function']}",
            "Execution level": contract.get("execution_level", "reference_only"),
            "Completeness": contract.get("completeness", "incomplete"),
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    st.markdown("### Selected workflow contract")
    selected_analysis_id = st.selectbox(
        "Workflow",
        options=product["entry_ids"],
        format_func=lambda analysis_id: (
            f"{analysis_id} · "
            next(item["name"] for item in product["entry_points"]
                 if item["id"] == analysis_id)
        ),
        key=f"tis_product_workflow_{selected_id}",
    )
    entry = next(
        item for item in product["entry_points"]
        if item["id"] == selected_analysis_id
    )
    contract = entry.get("execution_contract") or {}
    st.caption(f"Callable: tokamak_analysis.{entry['module']}.{entry['function']}")
    st.markdown(contract.get("algorithm", "No algorithm description declared."))
    left, right = st.columns(2)
    with left:
        st.markdown("**Inputs**")
        st.json(contract.get("input_specs", []))
    with right:
        st.markdown("**Outputs**")
        st.json(contract.get("output_specs", []))
    if contract.get("limitations"):
        with st.expander("Scientific limitations and assumptions"):
            st.write({"assumptions": contract.get("assumptions", []),
                      "limitations": contract.get("limitations", [])})
    st.info(
        "Execution is available in Advanced Analysis → Registered analysis modules. "
        "This workspace and the registered execution form use the same family IDs "
        "and callable registry."
    )
