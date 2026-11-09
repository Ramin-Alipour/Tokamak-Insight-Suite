"""Stable Python execution API shared by specialist products."""
from __future__ import annotations

from inspect import signature
from importlib import import_module
from typing import Any

from .catalog import get_product


def resolve_product_analysis(product_id: int | str, analysis_id: int):
    """Resolve one executable solver inside its owning specialist product.

    Ownership is checked before importing code. The returned callable is the
    exact solver used by :func:`execute_product_analysis`, so Dashboard
    introspection and execution cannot silently resolve different legacy paths.
    """
    product = get_product(product_id)
    if isinstance(analysis_id, bool) or not isinstance(analysis_id, int):
        raise ValueError("analysis_id must be an integer")
    if analysis_id not in set(product["entry_ids"]):
        raise ValueError(
            f"Analysis ID {analysis_id} is not owned by product {product['product_id']}"
        )
    entry = next(
        (item for item in product["entry_points"] if item["id"] == analysis_id),
        None,
    )
    if entry is None:
        raise RuntimeError(
            f"Product catalog for {product['product_id']} owns analysis {analysis_id} "
            "but does not provide its executable entry-point contract"
        )
    if product["id"] == 1:
        # Thomson execution is resolved by its specialist package, not by the
        # legacy tokamak_analysis callable registry. Keep the product's own
        # allow-list authoritative even if legacy registry metadata drifts.
        from tis.products.thomson.registry import THOMSON_ENTRY_POINTS

        try:
            module_name, function_name = THOMSON_ENTRY_POINTS[analysis_id]
        except KeyError as exc:
            raise RuntimeError(
                f"Thomson product owns analysis {analysis_id} but has no "
                "product-native executable mapping"
            ) from exc
        module_path = product["api_module"] + ".physics." + module_name
        module = import_module(module_path)
        try:
            return getattr(module, function_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"Thomson solver {module_path}.{function_name} is not importable"
            ) from exc

    module_path = product["api_module"] + ".physics." + entry["module"]
    module = import_module(module_path)
    return getattr(module, entry["function"])


def execute_product_analysis(
    product_id: int | str,
    analysis_id: int,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute one registered workflow through its owning specialist product.

    Arguments are passed unchanged to the registered scientific callable. This
    adapter performs ownership and Python-signature validation only; it does not
    calibrate, preprocess, coerce, or otherwise modify measurement inputs.
    """
    solver = resolve_product_analysis(product_id, analysis_id)
    signature(solver).bind(*args, **kwargs)
    return solver(*args, **kwargs)



def load_product_api(product_id: int | str):
    """Import and return the stable public Python module for one product."""
    product = get_product(product_id)
    return import_module(product["api_module"])
