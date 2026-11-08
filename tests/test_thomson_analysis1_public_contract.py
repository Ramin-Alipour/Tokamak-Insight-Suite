import json
from pathlib import Path

from tokamak_analysis import thomson_model_comparison


def test_thomson_model_comparison_is_exposed_in_package_api():
    assert callable(thomson_model_comparison.compare_thomson_forward_models)


def test_thomson_registry_does_not_advertise_raw_count_likelihood():
    registry = json.loads(
        (Path(__file__).resolve().parents[1] / "DIAGNOSTIC_REGISTRY.json").read_text(
            encoding="utf-8"
        )
    )
    entry = next(item for item in registry if item.get("diagnostic") == "Thomson scattering")
    functions = entry.get("implementation_functions", [])
    assert (
        "tokamak_analysis.thomson_model_comparison.compare_thomson_forward_models"
        in functions
    )
    assert (
        "tokamak_analysis.advanced_inference.poisson_deviance_residual"
        not in functions
    )
    assert "No raw-count Poisson likelihood" in entry.get("implementation_status", "")


def test_thomson_model_comparison_requires_same_measured_bins_and_parameter_counts():
    import numpy as np
    import pytest

    y = np.linspace(0.0, 1.0, 8)
    with pytest.raises(ValueError, match="missing parameter count"):
        thomson_model_comparison.compare_thomson_forward_models(
            y, {"maxwellian": y, "kappa": y + 0.01}, {"maxwellian": 3}
        )
    with pytest.raises(ValueError, match="must match measured_spectrum"):
        thomson_model_comparison.compare_thomson_forward_models(
            y, {"maxwellian": y, "kappa": y[:-1]}, {"maxwellian": 3, "kappa": 4}
        )
