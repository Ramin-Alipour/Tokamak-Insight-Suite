"""Stable Python execution API shared by specialist products."""
from __future__ import annotations

from inspect import signature
from importlib import import_module
from typing import Any

from tokamak_analysis.analysis_registry import get_analysis
from .catalog import get_product


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
    product = get_product(product_id)
    allowed_ids = set(product["entry_ids"])
    if isinstance(analysis_id, bool) or not isinstance(analysis_id, int):
        raise ValueError("analysis_id must be an integer")
    if analysis_id not in allowed_ids:
        raise ValueError(
            f"Analysis ID {analysis_id} is not owned by product {product['product_id']}"
        )
    entry = get_analysis(analysis_id)
    module = import_module(entry["module"])
    solver = getattr(module, entry["function"])
    signature(solver).bind(*args, **kwargs)
    return solver(*args, **kwargs)



def load_product_api(product_id: int | str):
    """Import and return the stable public Python module for one product."""
    product = get_product(product_id)
    return import_module(product["api_module"])
