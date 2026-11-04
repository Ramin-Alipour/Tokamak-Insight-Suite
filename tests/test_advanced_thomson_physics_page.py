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
