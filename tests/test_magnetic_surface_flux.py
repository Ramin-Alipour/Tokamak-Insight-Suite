import pytest
from tis.magnetic import integrate_magnetic_flux_surface


def test_magnetic_surface_flux_and_uncertainty():
    result = integrate_magnetic_flux_surface([1.0, 2.0], [0.1, 0.2], field_uncertainty_T=0.01)
    assert result['magnetic_flux_Wb'] == pytest.approx(0.5)
    assert result['standard_uncertainty_Wb'] == pytest.approx((0.001**2 + 0.002**2) ** 0.5)


def test_magnetic_surface_flux_rejects_negative_area():
    with pytest.raises(ValueError, match='nonnegative'):
        integrate_magnetic_flux_surface([1.0], [-1.0])
