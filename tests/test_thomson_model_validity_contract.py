import pytest

from tokamak_analysis.thomson_identifiability import optical_model_validity


def test_nonrelativistic_model_reports_asymptotic_warning_without_switching():
    result = optical_model_validity(
        "non_collective_maxwellian",
        electron_temperature_eV=20_000.0,
        electron_velocity_projection_m_s=0.2 * 299_792_458.0,
    )
    assert result["status"] == "asymptotic_screen_exceeded"
    assert result["quantitative_reporting_allowed"] is True
    assert result["requires_scope_warning"] is True
    assert result["automatic_model_switch"] is False


@pytest.mark.parametrize(
    "temperature,flow",
    [
        (-1.0, 0.0),
        (0.0, 0.0),
        (float("nan"), 0.0),
        (100.0, 299_792_458.0),
        (100.0, -1.01 * 299_792_458.0),
    ],
)
def test_unphysical_temperature_or_superluminal_flow_invalidates_reporting(temperature, flow):
    result = optical_model_validity(
        "non_collective_maxwellian",
        electron_temperature_eV=temperature,
        electron_velocity_projection_m_s=flow,
    )
    assert result["status"] == "outside_declared_model_domain"
    assert result["quantitative_reporting_allowed"] is False


@pytest.mark.parametrize("kappa", [1.5, 1.2, float("nan")])
def test_kappa_finite_variance_domain_is_a_hard_contract(kappa):
    result = optical_model_validity(
        "non_collective_kappa",
        electron_temperature_eV=100.0,
        electron_velocity_projection_m_s=0.0,
        kappa_index=kappa,
    )
    assert result["quantitative_reporting_allowed"] is False
    assert result["status"] == "outside_declared_model_domain"


def test_missing_fit_parameter_is_not_misreported_as_valid_domain():
    result = optical_model_validity(
        "non_collective_maxwellian",
        electron_temperature_eV=None,
        electron_velocity_projection_m_s=0.0,
    )
    assert result["status"] == "incomplete_domain_diagnostics"
    assert result["domain_assessment_complete"] is False
    assert result["quantitative_reporting_allowed"] is False


def test_relativistic_model_does_not_receive_nonrelativistic_temperature_warning():
    result = optical_model_validity(
        "non_collective_relativistic_maxwell_juttner",
        electron_temperature_eV=100_000.0,
        electron_velocity_projection_m_s=0.5 * 299_792_458.0,
    )
    assert result["status"] == "within_declared_screen"
    assert result["requires_scope_warning"] is False
