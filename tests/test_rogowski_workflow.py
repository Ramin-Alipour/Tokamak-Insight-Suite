import numpy as np
import pytest

from tis.magnetic import analyze_rogowski_current


def test_rogowski_current_integration_uses_mutual_inductance():
    result = analyze_rogowski_current(
        [0, 1, 2], [0.5, 0.5, 0.5], mutual_inductance_H=0.25,
    )
    np.testing.assert_allclose(result["plasma_current_A"], [0, 2, 4])
    assert result["mutual_inductance_H"] == pytest.approx(0.25)


def test_rogowski_rejects_invalid_mutual_inductance():
    with pytest.raises(ValueError, match="positive and finite"):
        analyze_rogowski_current([0, 1], [0, 1], mutual_inductance_H=0)
