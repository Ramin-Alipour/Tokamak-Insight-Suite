import numpy as np
import math
import pytest
from tokamak_analysis.formula_library import thomson_electron_temperature_from_spectral_width, thomson_velocity_projection, thomson_density_from_scattered_power

def test_thomson_temperature_formula_reference():
    out=thomson_electron_temperature_from_spectral_width(0.2,532.0,math.pi/2,sigma_intrinsic_sigma_nm=0.002,sigma_laser_wavelength_nm=0.001,sigma_scattering_angle_rad=1e-4)
    expected=(0.2/532.0)**2*510998.95069/(4*math.sin(math.pi/4)**2)
    assert out["electron_temperature_eV"]==pytest.approx(expected,rel=1e-12)
    assert out["uncertainty_electron_temperature_eV"]>0

def test_thomson_velocity_formula_reference():
    out=thomson_velocity_projection(532.01,532.0,math.pi/2)
    assert out["electron_velocity_projection_m_s"]==pytest.approx(-299792458.0*0.01/(2*532.0*math.sin(math.pi/4)),rel=1e-12)

def test_thomson_density_formula_reference():
    out=thomson_density_from_scattered_power(1e-9,1.0,0.1,math.pi/2,1e-3,optical_efficiency=0.5)
    assert out["electron_density_m3"]>0

def test_thomson_formulas_reject_invalid_physics():
    with pytest.raises(ValueError): thomson_electron_temperature_from_spectral_width(0.2,532.0,0.0)
    with pytest.raises(ValueError,match="(0, pi)"): thomson_electron_temperature_from_spectral_width(0.2,532.0,math.pi)
    with pytest.raises(ValueError,match="superluminal"): thomson_velocity_projection(1500.0,500.0,math.pi/2)


def test_thomson_density_correlated_input_covariance():
    from tokamak_analysis.thomson_scattering import density_from_thomson_scattered_power
    covariance = np.eye(6) * 1e-4
    out = density_from_thomson_scattered_power(
        1e-9, 1.0, 0.1, math.pi/2, 1e-3,
        optical_efficiency=0.5,
        relative_input_covariance=covariance,
    )
    assert out["electron_density_uncertainty_m3"] == pytest.approx(
        out["electron_density_m3"] * math.sqrt(6e-4), rel=1e-12
    )
    assert out["density_interpretation"] == "interaction_length_averaged_effective_density"


def test_thomson_density_uses_supplied_effective_relativistic_cross_section():
    from tokamak_analysis.thomson_scattering import density_from_thomson_scattered_power

    effective_cross_section = 1.7e-29
    cross_section_uncertainty = 0.1*effective_cross_section
    result = density_from_thomson_scattered_power(
        1e-12, 2.0, 0.2, math.pi/2, 0.01,
        optical_efficiency=0.5,
        effective_differential_cross_section_m2_sr=effective_cross_section,
        effective_differential_cross_section_uncertainty_m2_sr=cross_section_uncertainty,
    )
    expected = 1e-12/(2.0*0.2*effective_cross_section*0.01*0.5)
    assert result["electron_density_m3"] == pytest.approx(expected, rel=1e-12)
    assert result["differential_cross_section_m2_sr"] == pytest.approx(effective_cross_section)
    assert result["electron_density_uncertainty_m3"] == pytest.approx(0.1*expected, rel=1e-12)


def test_thomson_density_propagates_fitted_power_cross_section_covariance():
    from tokamak_analysis.thomson_scattering import density_from_thomson_scattered_power

    power = 1e-12
    cross_section = 1.7e-29
    power_sigma = 0.1*power
    cross_section_sigma = 0.2*cross_section
    relative_covariance = 0.01
    result = density_from_thomson_scattered_power(
        power, 2.0, 0.2, math.pi/2, 0.01,
        optical_efficiency=0.5,
        scattered_power_uncertainty_W=power_sigma,
        relative_input_covariance=np.zeros((6, 6)),
        effective_differential_cross_section_m2_sr=cross_section,
        effective_differential_cross_section_uncertainty_m2_sr=cross_section_sigma,
        power_cross_section_relative_covariance=relative_covariance,
    )
    expected_density = power/(2.0*0.2*cross_section*0.01*0.5)
    expected_relative_variance = 0.1**2 + 0.2**2 - 2.0*relative_covariance
    assert result["electron_density_uncertainty_m3"] == pytest.approx(
        expected_density*np.sqrt(expected_relative_variance), rel=1e-12
    )


def test_relativistic_mj_effective_cross_section_formula_library_link():
    from tokamak_analysis.formula_library import relativistic_mj_effective_differential_cross_section

    wavelength = np.linspace(532.0-40.0, 532.0+40.0, 801)
    value = relativistic_mj_effective_differential_cross_section(
        wavelength, 532.0, math.pi/2, 100.0,
        flow_projection_m_s=0.0,
        incident_polarization=np.array([0.0, 0.0, 1.0]),
    )
    assert np.isfinite(value)
    assert value > 0.0


def test_thomson_covariance_psd_guard_rejects_asymmetric_matrices():
    from tokamak_analysis.thomson_scattering import _covariance_is_positive_semidefinite

    # eigvalsh reads one triangle and can report positive eigenvalues for this
    # materially asymmetric input; an asymmetric matrix is not a covariance.
    assert not _covariance_is_positive_semidefinite(np.array([[1.0, 0.9], [0.1, 1.0]]))


def test_thomson_covariance_psd_guard_is_scale_relative_for_symmetry():
    from tokamak_analysis.thomson_scattering import _covariance_is_positive_semidefinite

    scale = 1e-24
    symmetric_psd = scale * np.array([[1.0, 0.25], [0.25, 1.0]])
    asymmetric = scale * np.array([[1.0, 0.25], [0.249, 1.0]])
    assert _covariance_is_positive_semidefinite(symmetric_psd)
    assert not _covariance_is_positive_semidefinite(asymmetric)


def test_thomson_temperature_formula_contract_preserves_measured_response_forward_fit():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    root_registry = json.loads((root / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    packaged_registry = json.loads(
        (root / "tis" / "data" / "FORMULA_REGISTRY.json").read_text(encoding="utf-8")
    )
    assert root_registry == packaged_registry
    entry = next(
        item for item in root_registry
        if item["execution_contract"]["formula_key"]
        == "thomson_electron_temperature_from_spectral_width"
    )
    contract = entry["execution_contract"]
    assert "instrument-corrected width" not in " ".join(contract["prerequisites"]).lower()
    assert "measured-response-convolved physical forward fit" in contract["note"]
    assert "does not correct or deconvolve measured spectra" in contract["note"]


def test_nonrelativistic_thomson_flow_propagates_joint_wavelength_geometry_covariance():
    from tokamak_analysis.formula_library import thomson_velocity_projection

    covariance = [
        [4.0e-10, 1.0e-10, 2.0e-12],
        [1.0e-10, 9.0e-10, -1.0e-12],
        [2.0e-12, -1.0e-12, 4.0e-8],
    ]
    result = thomson_velocity_projection(
        531.999, 532.0, 1.1, input_covariance_nm2_rad2=covariance
    )
    j = result["velocity_jacobian_m_s_per_input_unit"]
    expected = math.sqrt(sum(j[i] * covariance[i][k] * j[k]
                              for i in range(3) for k in range(3)))
    assert result["uncertainty_electron_velocity_projection_m_s"] == pytest.approx(expected)
    assert "full_joint_covariance" in result["uncertainty_model"]


def test_temperature_from_thomson_width_propagates_joint_covariance():
    from tokamak_analysis.formula_library import thomson_electron_temperature_from_spectral_width
    covariance = [
        [1.0e-8, 1.0e-11, 2.0e-12],
        [1.0e-11, 4.0e-12, -1.0e-12],
        [2.0e-12, -1.0e-12, 1.0e-8],
    ]
    result = thomson_electron_temperature_from_spectral_width(
        0.02, 532.0, math.pi / 2.0, input_covariance_nm2_rad2=covariance
    )
    j = result["temperature_jacobian_eV_per_input_unit"]
    expected = math.sqrt(sum(j[i] * covariance[i][k] * j[k]
                              for i in range(3) for k in range(3)))
    assert result["uncertainty_electron_temperature_eV"] == pytest.approx(expected)
    assert result["temperature_identifiability"].startswith("conditional_on_")
