"""Specialist product layer for Tokamak Insight Suite.

Product metadata and execution are adapters over the canonical analysis registry;
physics implementations remain in their domain modules.
"""
from .catalog import get_product, list_products, list_product_analyses
from .api import execute_product_analysis

__all__ = [
    "get_product", "list_products", "list_product_analyses",
    "execute_product_analysis",
]
