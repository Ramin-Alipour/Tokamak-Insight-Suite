import numpy as np

from tokamak_analysis.thomson_inference_closure import (
    compton_recoil_contract,
    fit_thomson_with_structured_background,
    reconstruct_joint_thomson_profiles,
)


def test_joint_background_fit_preserves_joint_parameter_covariance():
    x = np.linspace(525.0, 535.0, 81)
    z = 2 * (x - x.min()) / np.ptp(x) - 1
    design = np.column_stack((np.ones(x.size), z))
    def signal(p, axis):
        amp, center, width = p
        return amp * np.exp(-0.5 * ((axis - center) / width) ** 2)
    truth = np.array([3.2, 530.1, 0.75])
    y = signal(truth, x) + design @ np.array([0.12, -0.025])
    result = fit_thomson_with_structured_background(
        x, y, forward_model=signal, physical_initial=[3.0, 530.0, 0.9],
        background_design=design, measurement_uncertainty=np.full(x.size, 0.01),
    )
    assert result["optimizer_success"]
    assert result["physical_background_cross_covariance_retained"]
    assert result["joint_parameter_covariance"].shape == (5, 5)
    assert np.allclose(result["physical_parameters"], truth, rtol=2e-3, atol=2e-3)


def test_compton_contract_records_cross_section_and_recoil_screen():
    elastic = compton_recoil_contract(1064.0, include_compton_recoil=False)
    recoil = compton_recoil_contract(0.1, include_compton_recoil=True)
    assert elastic["cross_section"] == "polarized_Thomson"
    assert recoil["cross_section"] == "polarized_Klein_Nishina"
    import pytest
    with pytest.raises(ValueError):
        compton_recoil_contract(0.1, include_compton_recoil=False)


def test_joint_profile_inversion_reports_data_identifiability_separately():
    x = np.linspace(-3.0, 3.0, 61)
    radius = np.array([0.10, 0.17, 0.29, 0.38, 0.55])
    profiles = np.vstack((
        np.linspace(1.0, 1.4, 5),
        np.linspace(0.8, 1.1, 5),
        np.linspace(0.25, 0.4, 5),
    ))
    def forward(i, axis, ne, te, ti):
        electron = ne * np.exp(-0.5 * (axis / te) ** 2)
        ion = 0.2 * ti * np.exp(-0.5 * ((axis - 1.5) / (0.4 + ti)) ** 2)
        return electron + ion
    measured = np.vstack([forward(i, x, *profiles[:, i]) for i in range(radius.size)])
    lower = np.full_like(profiles, 0.05)
    upper = np.full_like(profiles, 3.0)
    result = reconstruct_joint_thomson_profiles(
        x, measured, radius, forward_model=forward,
        initial_profiles=profiles * 0.98, profile_bounds=(lower, upper),
        measurement_uncertainty=np.full(measured.size, 0.01),
        regularization_strength=1e-4,
    )
    assert result["optimizer_success"]
    assert result["profiles"]["electron_density_m3"].shape == radius.shape
    assert result["data_identifiability"]["parameter_count"] == 15
    assert result["regularization_is_not_measurement"]
