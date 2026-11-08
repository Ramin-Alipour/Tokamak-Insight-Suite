import numpy as np
import pytest

from tokamak_analysis.thomson_background import (
    fit_structured_background,
    fit_multichannel_structured_background,
)


def test_structured_background_template_fit_and_covariance():
    x = np.linspace(500.0, 510.0, 41)
    templates = np.column_stack((np.ones(x.size), (x - 505.0) / 5.0,
                                 np.exp(-0.5 * ((x - 507.0) / 0.8) ** 2)))
    y = templates @ np.array([2.0, -0.3, 0.7])
    result = fit_structured_background(x, y, model="templates", templates=templates)
    np.testing.assert_allclose(result["background"], y, atol=1e-11)
    assert result["background_role"] == "detector_space_nuisance_not_plasma_emission"
    assert result["coefficient_covariance"].shape == (3, 3)


def test_multichannel_structured_background_retains_cross_channel_covariance():
    x = np.linspace(500.0, 510.0, 21)
    y = np.vstack((2.0 + 0.1 * (x - 505.0), 3.0 - 0.2 * (x - 505.0)))
    covariance = np.eye(y.size)
    covariance[0, x.size] = covariance[x.size, 0] = 0.2
    result = fit_multichannel_structured_background(x, y, covariance=covariance)
    np.testing.assert_allclose(result["background"], y, atol=1e-10)
    assert result["cross_channel_covariance_used"] is True
    assert result["covariance_ordering"] == "channel_major"
    assert result["coefficient_covariance"].shape == (4, 4)
    assert abs(result["coefficient_covariance"][0, 2]) > 0.0


@pytest.mark.parametrize("covariance,match", [
    (np.ones((5, 5)), "positive definite"),
    (np.eye(4), "match"),
])
def test_multichannel_background_rejects_invalid_covariance(covariance, match):
    x = np.linspace(500.0, 510.0, 5)
    y = np.vstack((x, x))
    with pytest.raises(ValueError, match=match):
        fit_multichannel_structured_background(x, y, covariance=covariance)
