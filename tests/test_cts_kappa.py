import numpy as np
import pytest
from tokamak_analysis.cts_kappa import kappa_electron_susceptibility

def args(k=4.0):
    return dict(v=np.linspace(-5e7,5e7,2001),phase_velocity_m_s=1e6,
        electron_density_m3=2e18,electron_temperature_eV=250.0,
        kappa=k,wavenumber_m_inverse=120.0,pole_imaginary_part_m_s=2e3)

def test_kappa_response_is_finite_and_normalized():
    a=args(); r=kappa_electron_susceptibility(**a)
    assert np.isclose(np.trapz(r["normalized_probability_density_s_m"],a["v"]),1,rtol=1e-8)
    assert np.isfinite(r["electron_susceptibility"].real)
    assert r["pole"].imag>0
    assert r["velocity_grid_probability_coverage"] >= 0.999
    assert np.trapz(r["analytic_kappa_pdf_s_m"], a["v"]) == pytest.approx(r["velocity_grid_probability_coverage"], rel=2e-4)
    assert r["finite_support_probability_mass"] == pytest.approx(r["velocity_grid_probability_coverage"])
    assert r["finite_grid_normalization_bias_prevented"] is True

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


def test_kappa_heavy_tail_truncation_is_rejected_before_finite_grid_renormalization():
    parameters = args()
    parameters["v"] = np.linspace(-5.0e6, 5.0e6, 1001)
    with pytest.raises(ValueError, match="analytic Kappa probability mass"):
        kappa_electron_susceptibility(**parameters)


def test_kappa_coverage_can_be_relaxed_only_by_explicit_user_choice():
    parameters = args()
    parameters["v"] = np.linspace(-5.0e6, 5.0e6, 1001)
    parameters["minimum_velocity_grid_probability_coverage"] = 0.5
    result = kappa_electron_susceptibility(**parameters)
    assert result["velocity_grid_probability_coverage"] < 0.99
    assert result["minimum_velocity_grid_probability_coverage"] == 0.5


def test_kappa_susceptibility_is_reachable_through_public_advanced_physics_dispatcher():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    result = run_advanced_thomson_physics(
        "kappa_electrostatic_susceptibility", **args()
    )
    assert result["public_physics_entrypoint"] == "kappa_electrostatic_susceptibility"
    assert result["automatic_model_substitution"] is False
    assert result["velocity_grid_probability_coverage"] >= 0.999


def test_analytic_kappa_tail_does_not_receive_artificial_zero_extension_edge_jumps():
    result = kappa_electron_susceptibility(**args())
    assert result["finite_support_boundary_terms_included"] is False
    assert result["finite_support_extension"] == "analytic_Kappa_tail_integrated_to_infinity"
    assert result["finite_support_boundary_integral"] == 0.0j
    assert result["tail_model"].startswith("analytic_isotropic_Kappa_marginal")


def test_kappa_landau_susceptibility_integrates_the_analytic_tail_not_the_finite_grid():
    narrow = args()
    wide = args()
    wide["v"] = np.linspace(-1.2e8, 1.2e8, 4001)
    first = kappa_electron_susceptibility(**narrow)
    second = kappa_electron_susceptibility(**wide)
    assert first["landau_integral_tail_complete"] is True
    assert first["susceptibility_uses_complete_analytic_kappa_tail"] is True
    assert first["quadrature"].endswith("minus_infinity_to_infinity")
    assert first["integral"] == pytest.approx(second["integral"], rel=2e-7, abs=1e-12)
    assert first["electron_susceptibility"] == pytest.approx(
        second["electron_susceptibility"], rel=2e-7, abs=1e-12
    )
    assert first["landau_integral_real_absolute_error"] >= 0.0
    assert first["landau_integral_imag_absolute_error"] >= 0.0
