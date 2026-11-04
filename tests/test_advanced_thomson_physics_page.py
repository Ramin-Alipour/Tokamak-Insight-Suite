import numpy as np
import pytest

from tis.ui.advanced_thomson_physics_page import decode_complex_json, json_safe


def test_complex_json_arrays_decode_without_shape_or_phase_loss():
    value = {"real": [[1.0, 0.0]], "imag": [[0.0, -2.0]]}
    decoded = decode_complex_json(value)
    np.testing.assert_allclose(decoded, [[1.0 + 0.0j, -2.0j]])
    assert decoded.dtype.kind == "c"


def test_complex_json_rejects_mismatched_real_and_imag_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        decode_complex_json({"real": [1.0, 2.0], "imag": [1.0]})


def test_json_safe_preserves_complex_arrays_and_omits_internal_landau_operator():
    value = {
        "field": np.array([1.0 + 2.0j]),
        "landau_system": object(),
        "count": np.int64(3),
    }
    safe = json_safe(value)
    assert safe == {
        "field": {"real": [1.0], "imag": [2.0]},
        "count": 3,
    }


def test_full_tail_kappa_cts_solver_is_exposed_in_dashboard_and_formula_registry():
    import json
    from pathlib import Path
    from tis.ui.advanced_thomson_physics_page import EXAMPLES, MODEL_LABELS
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    model = "kappa_electrostatic_susceptibility"
    assert model in MODEL_LABELS
    configuration = json.loads(json.dumps(EXAMPLES[model]))
    result = run_advanced_thomson_physics(model, **configuration)
    assert result["landau_integral_tail_complete"] is True
    assert result["public_physics_entrypoint"] == model

    root = Path(__file__).resolve().parents[1]
    formula = json.loads((root / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    packaged = json.loads((root / "tis" / "data" / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    key = "cts_isotropic_kappa_landau_susceptibility"
    matches = [item for item in formula if item.get("execution_contract", {}).get("formula_key") == key]
    assert len(matches) == 1
    assert formula == packaged
