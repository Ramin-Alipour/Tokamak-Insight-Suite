"""Independent regression tests for new Thomson path and CTS mode utilities."""
import numpy as np
import pytest

from tokamak_analysis.thomson_scattering import infer_thomson_path_averaged_density
from tokamak_analysis.cts_em_geometry import track_cold_plasma_modes_from_operators


def test_path_integrated_density_recovers_weighted_uniform_density():
    s = np.linspace(0.0, 0.4, 101)
    incident = np.linspace(1.0, 0.8, s.size)
    cross = np.full(s.size, 6.0e-29)
    solid_angle = np.linspace(0.02, 0.03, s.size)
    efficiency = np.linspace(0.4, 0.5, s.size)
    truth = 2.5e19
    integrate = getattr(np, "trapezoid", None) or np.trapz
    denominator = integrate(incident * cross * solid_angle * efficiency, s)
    measured_power = truth * denominator
    result = infer_thomson_path_averaged_density(
        measured_power, s, incident, cross, solid_angle, efficiency,
        scattered_power_uncertainty_W=0.02 * measured_power,
    )
    assert result["electron_density_path_averaged_m3"] == pytest.approx(truth, rel=1e-12)
    assert result["electron_density_path_averaged_uncertainty_m3"] == pytest.approx(0.02 * truth)
    assert result["identifiability"]["status"] == "weighted_path_average_identifiable"
    assert "arbitrary pointwise electron_density_profile_m3" in result["identifiability"]["not_identifiable"]


def test_path_integrated_density_rejects_unmatched_or_unphysical_profiles():
    s = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="must be finite and match"):
        infer_thomson_path_averaged_density(1.0, s, np.ones(4), np.ones(5), np.ones(5), np.ones(5))
    with pytest.raises(ValueError, match="efficiency must not exceed one"):
        infer_thomson_path_averaged_density(1.0, s, np.ones(5), np.ones(5), np.ones(5), np.full(5, 1.1))


def test_mode_tracking_follows_polarization_through_eigenvalue_order_crossing():
    angle = np.linspace(0.0, np.pi / 2.0, 31)
    operators = []
    for a in angle:
        rotation = np.array([
            [np.cos(a), -np.sin(a), 0.0],
            [np.sin(a), np.cos(a), 0.0],
            [0.0, 0.0, 1.0],
        ])
        diagonal = np.diag([-1.0 + 4.0 * a / np.pi, 1.0 - 4.0 * a / np.pi, 3.0])
        operators.append(rotation @ diagonal @ rotation.T)
    result = track_cold_plasma_modes_from_operators(np.asarray(operators))
    permutations = result["eigenvalue_order_permutation"]
    assert np.any(np.diff(permutations[:, 0]) != 0)
    overlap = result["adjacent_sample_polarization_overlap_squared"]
    assert np.all(np.isfinite(overlap)) and np.all((overlap >= 0.0) & (overlap <= 1.0 + 1e-12))
    assert result["branch_identifiability"] == "ambiguous_at_one_or_more_degenerate_samples"
    assert np.any(result["degenerate_subspace_detected"])


def test_mode_tracking_reports_exact_degeneracy_as_ambiguous():
    operators = np.asarray([
        np.diag([0.0, 1.0, 2.0]),
        np.diag([0.0, 0.0, 2.0]),
        np.diag([0.0, 1.0, 2.0]),
    ], dtype=complex)
    result = track_cold_plasma_modes_from_operators(operators)
    assert result["degenerate_subspace_detected"].tolist() == [False, True, False]
    assert result["branch_identifiability"] == "ambiguous_at_one_or_more_degenerate_samples"



def test_cold_plasma_ray_includes_branch_tracking_diagnostics(monkeypatch):
    from types import SimpleNamespace
    import tokamak_analysis.cts_em_geometry as geometry

    omega = 2.0 * np.pi * 140.0e9
    k0 = omega / 299792458.0
    positions = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    wavevectors = np.array([[k0, 0.0, 0.0], [k0, 0.01 * k0, 0.0], [k0, 0.02 * k0, 0.0]])
    fake_solution = SimpleNamespace(
        success=True,
        message="synthetic solver path for software unit test",
        t=np.array([0.0, 0.5, 1.0]),
        y=np.vstack((positions.T, wavevectors.T)),
    )
    monkeypatch.setattr(geometry, "solve_ivp", lambda *args, **kwargs: fake_solution)
    equilibrium = lambda position: {
        "magnetic_field_T": np.array([0.0, 0.0, 2.0]),
        "species": [{
            "density_m3": 1.0e19,
            "charge_C": -1.602176634e-19,
            "mass_kg": 9.1093837015e-31,
        }],
    }
    result = geometry.trace_cold_plasma_ray(
        positions[0], wavevectors[0], omega,
        equilibrium=equilibrium, mode_index=0, sample_count=3,
    )
    assert result["mode_branch_tracking"]["tracked_eigenvalues"].shape == (3, 3)
    assert isinstance(result["fixed_eigenvalue_order_tracks_initial_polarization_branch"], bool)
    assert "mode conversion" in " ".join(result["limitations"]).lower()



def test_coupled_mode_propagation_conserves_power_and_reports_conversion():
    from tokamak_analysis.cts_em_geometry import propagate_cold_plasma_mode_coupling
    s = np.linspace(0.0, 1.0, 101)
    angles = np.linspace(0.0, np.pi / 2.0, s.size)
    operators = []
    for a in angles:
        c, q = np.cos(a), np.sin(a)
        rotation = np.array([[c, -q, 0.0], [q, c, 0.0], [0.0, 0.0, 1.0]])
        operators.append(rotation @ np.diag([-1.0, 1.0, 3.0]) @ rotation.T)
    result = propagate_cold_plasma_mode_coupling(
        np.asarray(operators), s, np.tile([10.0, 12.0, 15.0], (s.size, 1)),
        np.array([1.0 + 0j, 0j, 0j]),
    )
    assert result["mode_powers"].shape == (s.size, 3)
    assert result["power_conservation_max_abs_error"] < 1e-10
    assert np.max(result["mode_powers"][-1, 1:]) > 1e-3
    assert result["mode_conversion_model"] == "lossless_flux_normalized_cold_plasma_coupled_mode_equation"


def test_coupled_mode_propagation_rejects_missing_or_misaligned_physical_path():
    from tokamak_analysis.cts_em_geometry import propagate_cold_plasma_mode_coupling
    operators = np.asarray([np.diag([0.0, 1.0, 2.0])] * 3, dtype=complex)
    with pytest.raises(ValueError, match="strictly increasing"):
        propagate_cold_plasma_mode_coupling(
            operators, [0.0, 1.0, 1.0], np.ones((3, 3)), [1.0, 0.0, 0.0]
        )


def test_path_density_propagates_correlated_optical_profile_covariance():
    s = np.linspace(0.0, 0.4, 21)
    incident = np.linspace(1.0, 0.8, s.size)
    cross = np.full(s.size, 6.0e-29)
    solid = np.linspace(0.02, 0.03, s.size)
    efficiency = np.linspace(0.4, 0.5, s.size)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    sensitivity = integrate(incident * cross * solid * efficiency, s)
    density = 2.5e19
    power = density * sensitivity

    # A common 5% calibration mode correlates every incident-power sample.
    n = s.size
    covariance = np.zeros((4*n, 4*n))
    common_mode = 0.05 * incident
    covariance[:n, :n] = np.outer(common_mode, common_mode)
    result = infer_thomson_path_averaged_density(
        power, s, incident, cross, solid, efficiency,
        scattered_power_uncertainty_W=0.02*power,
        optical_profile_covariance=covariance,
    )
    assert result["electron_density_optical_profile_uncertainty_m3"] == pytest.approx(
        0.05*density, rel=1e-10
    )
    assert result["electron_density_path_averaged_uncertainty_m3"] == pytest.approx(
        density*np.sqrt(0.05**2 + 0.02**2), rel=1e-10
    )
    assert result["optical_profile_covariance_order"][0] == "incident_power_profile_W"


def test_path_density_rejects_non_psd_optical_profile_covariance():
    s = np.linspace(0.0, 1.0, 4)
    covariance = np.zeros((4*s.size, 4*s.size))
    covariance[0, 0] = 1.0
    covariance[1, 1] = 1.0
    covariance[0, 1] = covariance[1, 0] = 2.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        infer_thomson_path_averaged_density(
            1.0, s, np.ones(4), np.ones(4), np.ones(4), np.full(4, 0.5),
            optical_profile_covariance=covariance,
        )


def test_path_density_propagates_scattered_power_profile_cross_covariance():
    s = np.linspace(0.0, 0.4, 21)
    incident = np.linspace(1.0, 0.8, s.size)
    cross = np.full(s.size, 6.0e-29)
    solid = np.linspace(0.02, 0.03, s.size)
    efficiency = np.linspace(0.4, 0.5, s.size)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    sensitivity = integrate(incident * cross * solid * efficiency, s)
    density = 2.5e19
    power = density * sensitivity
    sigma_power = 0.02 * power

    # A shared calibration mode drives both scattered power and the incident
    # power profile. The complete block covariance is rank-one PSD.
    n = s.size
    profile_mode = np.zeros(4*n)
    profile_mode[:n] = 0.05 * incident
    profile_covariance = np.outer(profile_mode, profile_mode)
    cross_covariance = sigma_power * profile_mode
    result = infer_thomson_path_averaged_density(
        power, s, incident, cross, solid, efficiency,
        scattered_power_uncertainty_W=sigma_power,
        optical_profile_covariance=profile_covariance,
        scattered_power_optical_profile_cross_covariance=cross_covariance,
    )
    # The density gradient has opposite signs for scattered power and
    # incident-power calibration, so their positive covariance cancels part
    # of the otherwise independent uncertainty.
    assert result["electron_density_path_averaged_uncertainty_m3"] == pytest.approx(
        0.03*density, rel=1e-10
    )
    assert result["scattered_power_optical_profile_cross_covariance_supplied"] is True


def test_path_density_rejects_non_psd_power_profile_joint_covariance():
    s = np.linspace(0.0, 1.0, 5)
    n = s.size
    profile_covariance = np.eye(4*n)
    cross_covariance = np.full(4*n, 10.0)
    with pytest.raises(ValueError, match="joint scattered-power/optical-profile covariance"):
        infer_thomson_path_averaged_density(
            2.0, s, np.ones(n), np.full(n, 1e-28), np.full(n, 0.02),
            np.full(n, 0.5),
            scattered_power_uncertainty_W=0.01,
            optical_profile_covariance=profile_covariance,
            scattered_power_optical_profile_cross_covariance=cross_covariance,
        )


def test_correlated_path_density_is_a_registered_thomson_subanalysis():
    from tokamak_analysis.analysis_registry import get_analysis, get_analysis_family

    entry = get_analysis(87)
    assert entry["function"] == "infer_thomson_path_averaged_density"
    assert entry["module"] == "thomson_scattering"
    assert any(
        item["name"] == "scattered_power_optical_profile_cross_covariance"
        for item in entry["execution_contract"]["input_specs"]
    )
    assert 87 in get_analysis_family(1)["entry_ids"]


def test_path_density_formula_registry_points_to_the_physical_solver():
    import json
    from pathlib import Path

    registry = json.loads(
        (Path(__file__).resolve().parents[1] / "FORMULA_REGISTRY.json").read_text()
    )
    formula = registry["108"]
    assert formula["execution_contract"]["function_link"] == (
        "tokamak_analysis.thomson_scattering.infer_thomson_path_averaged_density"
    )
    assert "Var(n̄_e)=gᵀ C_joint g" in formula["equation"]
