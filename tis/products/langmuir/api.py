"""Public execution API for Langmuir and electrostatic probe workflows."""
from __future__ import annotations

from typing import Any

from ..api import execute_product_analysis

PRODUCT_ID = "langmuir-electrostatic-probes"


def run_probe_analysis(analysis_id: int, *args: Any, **kwargs: Any) -> Any:
    """Execute a registered Langmuir/electrostatic probe workflow (IDs 2, 16–20)."""
    return execute_product_analysis(PRODUCT_ID, analysis_id, *args, **kwargs)
