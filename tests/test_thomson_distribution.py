import numpy as np
import pytest

from tis.thomson_distribution import invert_thomson_distribution


def test_thomson_distribution_inversion_recovers_synthetic_projected_distribution():
    velocity = np.linspace(-3e6, 3e6, 81)
    spectral_bins = np.linspace(-2.5, 2.5, 61)
    kernel = np.exp(-0.5 * ((spectral_bins[:, None] - velocity[None, :] / 1.2e6) / 0.45) ** 2)
    true_distribution = np.exp(-0.5 * (velocity / 8e5) ** 2)
    spectrum = kernel @ true_distribution
    result = invert_thomson_distribution(
        spectrum, kernel, velocity, regularization=1e-5,
        uncertainty=np.full_like(spectrum, 1e-3),
    )
    assert np.all(result["distribution_raw"] >= 0)
    assert np.trapezoid(result["distribution_normalized"], velocity) == pytest.approx(1.0, rel=1e-5)
    assert np.all(np.isfinite(result["predicted_spectrum"]))
    assert result["weighted_chi_squared"] >= 0


def test_thomson_distribution_rejects_incompatible_kernel_shape():
    with pytest.raises(ValueError, match="shape"):
        invert_thomson_distribution([1, 2, 3], np.ones((3, 4)), [0, 1, 2])


def test_thomson_distribution_rejects_nonmonotonic_velocity_grid():
    with pytest.raises(ValueError, match="strictly increasing"):
        invert_thomson_distribution([1, 2, 3], np.ones((3, 3)), [0, 2, 1])
