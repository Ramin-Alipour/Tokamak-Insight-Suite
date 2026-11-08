import numpy as np
import pytest

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



def test_structured_background_fit_does_not_report_covariance_after_optimizer_failure(monkeypatch):
    from types import SimpleNamespace
    import tokamak_analysis.thomson_inference_closure as closure

    x = np.linspace(-1.0, 1.0,  nine := 9)
    design = np.column_stack((np.ones(nine), x))
    def forward(parameters, axis):
        return parameters[0] * np.exp(-axis**2)

    # A full-rank Jacobian isolates the failure-state contract: numerical
    # rank alone must not turn a non-converged fit into a reported covariance.
    jacobian = np.zeros((nine, 3))
    jacobian[:3, :] = np.eye(3)
    fake = SimpleNamespace(
        x=np.array([1.0, 0.0, 0.0]),
        jac=jacobian,
        fun=np.ones(nine),
        success=False,
        message="iteration limit reached",
        cost=4.5,
    )
    monkeypatch.setattr(closure, "least_squares", lambda *args, **kwargs: fake)
    result = fit_thomson_with_structured_background(
        x, np.ones(nine), forward_model=forward, physical_initial=[1.0],
        background_design=design,
    )
    assert result["optimizer_success"] is False
    assert result["joint_parameter_covariance"] is None
    assert result["covariance_status"] == "not_reported_optimizer_failed"
    assert result["fit_acceptance"] == "optimizer_failed"


def test_structured_background_fit_uses_feasible_one_sided_nuisance_initialization():
    x = np.linspace(-1.0, 1.0, 31)
    design = np.column_stack((np.ones(x.size), x))
    def forward(parameters, axis):
        return parameters[0] * np.exp(-axis**2)
    result = fit_thomson_with_structured_background(
        x, forward([1.2], x) + design @ np.array([0.15, 0.02]),
        forward_model=forward, physical_initial=[1.0],
        background_design=design, background_bounds=([0.0, 0.0], [1.0, 1.0]),
        measurement_uncertainty=np.full(x.size, 0.01),
    )
    assert result["optimizer_success"]
    assert result["background_coefficients"][0] == pytest.approx(0.15, abs=1e-3)
    assert result["background_coefficients"][1] == pytest.approx(0.02, abs=1e-3)
