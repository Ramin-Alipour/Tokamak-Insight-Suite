"""Product-oriented dashboard view over the canonical analysis registry."""
from __future__ import annotations

from typing import Any

from tis.products import get_product, list_products, build_dependency_inventory, load_product_api


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
            + next(item["name"] for item in products if item["id"] == product_id)
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
    product_api = load_product_api(product["product_id"])
    st.markdown("**Public Python API**")
    st.code(product["api_module"], language="python")
    st.caption(
        "Exported capabilities: "
        + ", ".join(getattr(product_api, "__all__", ()))
    )
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
    inventory = build_dependency_inventory()
    dependency_row = next(
        item for item in inventory["products"]
        if item["product_id"] == product["product_id"]
    )
    with st.expander("Code ownership and dependency boundary"):
        st.write({
            "registered_solver_modules": dependency_row["entry_point_modules"],
            "product_local_domain_modules": dependency_row["product_local_domain_modules"],
            "shared_domain_modules": dependency_row["shared_domain_modules"],
            "shared_platform_dependencies": dependency_row["shared_platform_dependencies"],
        })
        st.caption(
            "Static AST import closure. Dynamic imports, runtime plugins, "
            "non-Python resources and third-party package constraints are not included."
        )
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
