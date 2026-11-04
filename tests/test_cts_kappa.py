import numpy as np
import pytest
from tokamak_analysis.cts_kappa import kappa_electron_susceptibility

def args(k=4.0):
    return dict(v=np.linspace(-2e7,2e7,1001),phase_velocity_m_s=1e6,
        electron_density_m3=2e18,electron_temperature_eV=250.0,
        kappa=k,wavenumber_m_inverse=120.0,pole_imaginary_part_m_s=2e3)

def test_kappa_response_is_finite_and_normalized():
    a=args(); r=kappa_electron_susceptibility(**a)
    assert np.isclose(np.trapz(r["normalized_probability_density_s_m"],a["v"]),1,rtol=1e-8)
    assert np.isfinite(r["electron_susceptibility"].real)
    assert r["pole"].imag>0

def test_kappa_requires_kappa_above_three_halves():
    with pytest.raises(ValueError,match="greater than 3/2"):
        kappa_electron_susceptibility(**args(1.5))


@pytest.mark.parametrize(
    "field,value",
    [
        ("electron_density_m3", True),
        ("electron_temperature_eV", "250"),
        ("kappa", False),
        ("wavenumber_m_inverse", 0j),
        ("mass_kg", "9.1e-31"),
        ("phase_velocity_m_s", True),
        ("pole_imaginary_part_m_s", False),
    ],
)
def test_kappa_rejects_boolean_string_and_complex_scalar_coercion(field, value):
    parameters = args()
    if field == "phase_velocity_m_s":
        parameters["phase_velocity_m_s"] = value
    elif field == "pole_imaginary_part_m_s":
        parameters[field] = value
    else:
        parameters[field] = value
    with pytest.raises(ValueError):
        kappa_electron_susceptibility(**parameters)


def test_kappa_reports_nonrelativistic_finite_support_scope():
    result = kappa_electron_susceptibility(**args())
    assert result["distribution_model"] == "exact_1D_marginal_of_isotropic_3D_Kappa"
    assert result["finite_velocity_grid_truncation"] is True
    assert any("non-relativistic" in item for item in result["limitations"])
