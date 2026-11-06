import numpy as np
import pytest
from tokamak_analysis.cts_kappa import kappa_electron_susceptibility

def args(k=4.0):
    return dict(v=np.linspace(-5e7,5e7,2001),phase_velocity_m_s=1e6,
        electron_density_m3=2e18,electron_temperature_eV=250.0,
        kappa=k,wavenumber_m_inverse=120.0,pole_imaginary_part_m_s=2e3)

def test_kappa_response_is_finite_and_normalized():
    a=args(); r=kappa_electron_susceptibility(**a)
    # The analytic PDF integrates to the represented probability mass;
    # only the explicitly named grid-conditioned view is renormalized.
    assert np.trapz(r["normalized_probability_density_s_m"], a["v"]) == pytest.approx(
        r["velocity_grid_probability_coverage"], rel=2e-4
    )
    assert np.isclose(
        np.trapz(r["grid_conditioned_probability_density_s_m"], a["v"]),
        1.0, rtol=2e-4,
    )
    assert r["grid_conditioning_applied_to_physical_response"] is False
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
    # A positive phase velocity samples the negative Kappa slope; the
    # retarded-pole imaginary part must resolve that resonant damping term.
    assert first["integral"].imag < 0.0
    assert abs(first["integral"].imag) > first["landau_integral_imag_absolute_error"]


def test_collective_kappa_cts_inverse_is_exposed_by_cts_dispatcher(monkeypatch):
    import tokamak_analysis.thomson_kinetic_models as kinetic
    from tokamak_analysis.cts_analysis import analyze_cts_spectrum_model

    sentinel = {"model": "collective_kappa_test_route"}
    def fake_fit(**kwargs):
        assert kwargs["initial_kappa"] == 4.0
        assert kwargs["ion_temperature_eV"] == [20.0]
        return sentinel

    monkeypatch.setattr(kinetic, "fit_collective_kappa", fake_fit)
    result = analyze_cts_spectrum_model(
        model="electrostatic_kappa",
        wavelength_nm=np.linspace(500.0, 501.0, 12), signal=np.ones(12),
        laser_wavelength_nm=500.0, scattering_angle_rad=1.0,
        response=np.eye(12), incident_direction=[0.0, 0.0, 1.0],
        scattered_direction=[np.sin(1.0), 0.0, np.cos(1.0)],
        ion_species=[{"name": "D", "charge_state": 1, "mass_amu": 2.0}],
        ion_fractions=[1.0], ion_temperature_eV=[20.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        initial_density_m3=1e19, initial_temperature_eV=100.0,
        initial_kappa=4.0,
    )
    assert result is sentinel


def test_collective_kappa_dispatcher_converts_canonical_frequency_schema(monkeypatch):
    import tokamak_analysis.thomson_kinetic_models as kinetic
    from tokamak_analysis.cts_analysis import analyze_cts_spectrum_model

    captured = {}
    def fake_fit(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(kinetic, "fit_collective_kappa", fake_fit)
    frequency = np.linspace(-2.0e6, 2.0e6, 12)
    result = analyze_cts_spectrum_model(
        model="electrostatic_kappa", frequency_shift_Hz=frequency,
        spectrum=np.ones(12), probe_frequency_Hz=1.0e11,
        scattering_geometry={"scattering_angle_rad": 1.0},
        ion_species=[{"name": "D", "charge_state": 1, "mass_amu": 2.0}],
        ion_fractions=[1.0], instrument_response=np.eye(12),
        ion_temperature_eV=[20.0], electron_velocity_projection_m_s=1.0e4,
        ion_velocity_projection_m_s=2.0e3, electron_density_initial_m3=1e19,
        electron_temperature_initial_eV=100.0, initial_kappa=4.0,
    )
    assert result == {"ok": True}
    assert np.all(np.diff(captured["wavelength_nm"]) > 0.0)
    assert captured["signal"].shape == (12,)
    assert captured["response"].shape == (12, 12)
    assert captured["electron_velocity_m_s"].shape == (3,)
    assert captured["ion_velocity_m_s"].shape == (1, 3)



def test_collective_kappa_dispatcher_passes_multi_population_contract_to_physical_fit(monkeypatch):
    import tokamak_analysis.thomson_kinetic_models as kinetic
    from tokamak_analysis.cts_analysis import analyze_cts_spectrum_model

    populations = [
        {"density_fraction": 0.8, "temperature_ratio": 0.75,
         "kappa": 5.0, "flow_projection_m_s": -2.0e4},
        {"density_fraction": 0.2, "temperature_ratio": 3.0,
         "kappa": 2.4, "flow_projection_m_s": 4.0e5},
    ]
    captured = {}

    def fake_fit(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(kinetic, "fit_collective_kappa", fake_fit)
    result = analyze_cts_spectrum_model(
        model="electrostatic_kappa",
        wavelength_nm=np.linspace(500.0, 501.0, 12), signal=np.ones(12),
        laser_wavelength_nm=500.0, scattering_angle_rad=1.0,
        response=np.eye(12), incident_direction=[0.0, 0.0, 1.0],
        scattered_direction=[np.sin(1.0), 0.0, np.cos(1.0)],
        ion_species=[{"name": "D", "charge_state": 1, "mass_amu": 2.0}],
        ion_fractions=[1.0], ion_temperature_eV=[20.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        initial_density_m3=1e19, initial_temperature_eV=100.0,
        initial_kappa=4.0, electron_populations=populations,
    )
    assert result == {"ok": True}
    assert captured["electron_populations"] == populations



def test_analytic_maxwellian_susceptibility_matches_landau_quadrature_reference():
    from tokamak_analysis.cts_kappa import maxwellian_electron_susceptibility
    from tokamak_analysis.thomson_kinetic_models import _landau_susceptibility

    ne, te, wave_number = 2.0e19, 850.0, 420.0
    sigma = np.sqrt(1.602176634e-19 * te / 9.1093837015e-31)
    omega_pe2 = ne * 1.602176634e-19**2 / (8.8541878128e-12 * 9.1093837015e-31)
    phase = np.array([-1.3, -0.4, 0.0, 0.7, 1.8]) * sigma
    exact = maxwellian_electron_susceptibility(
        phase, electron_density_m3=ne, electron_temperature_eV=te,
        wavenumber_m_inverse=wave_number,
    )["electron_susceptibility"]
    reference = np.array([
        _landau_susceptibility(v, omega_pe2, wave_number, sigma)
        for v in phase
    ])
    np.testing.assert_allclose(exact, reference, rtol=3e-5, atol=2e-8)


def test_analytic_maxwellian_susceptibility_has_correct_zero_and_high_phase_limits():
    from tokamak_analysis.cts_kappa import maxwellian_electron_susceptibility

    ne, te, k = 1.5e19, 500.0, 300.0
    result = maxwellian_electron_susceptibility(
        0.0, electron_density_m3=ne, electron_temperature_eV=te,
        wavenumber_m_inverse=k,
    )
    omega_pe2 = result["plasma_frequency_squared_s_minus2"]
    sigma2 = result["one_dimensional_velocity_variance_m2_s2"]
    expected_zero = omega_pe2 / (k**2 * sigma2)
    assert result["electron_susceptibility"].real == pytest.approx(expected_zero, rel=1e-12)
    assert result["electron_susceptibility"].imag == pytest.approx(0.0, abs=1e-14)

    phase = 80.0 * result["thermal_speed_m_s"]
    high = maxwellian_electron_susceptibility(
        phase, electron_density_m3=ne, electron_temperature_eV=te,
        wavenumber_m_inverse=k,
    )["electron_susceptibility"]
    cold_asymptote = -omega_pe2 / (k**2 * phase**2)
    assert high.real == pytest.approx(cold_asymptote, rel=5e-4)
    assert abs(high.imag) < abs(high.real) * 1e-10


def test_analytic_maxwellian_susceptibility_is_available_through_public_dispatcher_and_dashboard():
    from pathlib import Path
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    result = run_advanced_thomson_physics(
        "maxwellian_electrostatic_susceptibility",
        phase_velocity_m_s=[0.0, 1.0e5],
        electron_density_m3=1.0e19,
        electron_temperature_eV=100.0,
        wavenumber_m_inverse=100.0,
    )
    assert result["public_physics_entrypoint"] == "maxwellian_electrostatic_susceptibility"
    assert np.asarray(result["electron_susceptibility"]).shape == (2,)
    dashboard = (Path(__file__).resolve().parents[1] / "tis" / "ui" / "advanced_thomson_physics_page.py").read_text()
    assert "maxwellian_electrostatic_susceptibility" in dashboard



def test_analytic_maxwellian_susceptibility_derivative_matches_central_difference():
    from tokamak_analysis.cts_kappa import maxwellian_electron_susceptibility

    args = dict(
        electron_density_m3=1.2e19,
        electron_temperature_eV=720.0,
        wavenumber_m_inverse=350.0,
    )
    phase = 0.83 * np.sqrt(
        2.0 * 1.602176634e-19 * args["electron_temperature_eV"]
        / 9.1093837015e-31
    )
    result = maxwellian_electron_susceptibility(phase, **args)
    step = 1.0e-5 * result["thermal_speed_m_s"]
    plus = maxwellian_electron_susceptibility(phase + step, **args)["electron_susceptibility"]
    minus = maxwellian_electron_susceptibility(phase - step, **args)["electron_susceptibility"]
    finite_difference = (plus - minus) / (2.0 * step)
    assert result["susceptibility_phase_velocity_derivative_s_m"] == pytest.approx(
        finite_difference, rel=2e-8, abs=1e-14
    )


def test_kappa_phase_derivative_formula_registry_is_synchronized_and_executable():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    formula = json.loads((root / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    packaged = json.loads((root / "tis" / "data" / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    key = "cts_kappa_landau_phase_velocity_derivative"
    matches = [item for item in formula if item.get("execution_contract", {}).get("formula_key") == key]
    assert len(matches) == 1
    assert matches == [item for item in packaged if item.get("execution_contract", {}).get("formula_key") == key]
    assert matches[0]["execution_contract"]["function_link"] == "tokamak_analysis.cts_kappa.kappa_electron_susceptibility"
    assert "susceptibility_phase_velocity_derivative_s_m" in {
        item["name"] for item in matches[0]["execution_contract"]["output_specs"]
    }


def test_kappa_susceptibility_phase_derivative_matches_central_finite_difference():
    parameters = args(k=4.5)
    parameters["phase_velocity_m_s"] = 0.73e6
    result = kappa_electron_susceptibility(**parameters)
    step = 2.0e-4 * np.sqrt(
        1.602176634e-19 * parameters["electron_temperature_eV"]
        / 9.1093837015e-31
    )
    plus = kappa_electron_susceptibility(
        **{**parameters, "phase_velocity_m_s": parameters["phase_velocity_m_s"] + step}
    )["electron_susceptibility"]
    minus = kappa_electron_susceptibility(
        **{**parameters, "phase_velocity_m_s": parameters["phase_velocity_m_s"] - step}
    )["electron_susceptibility"]
    finite_difference = (plus - minus) / (2.0 * step)
    assert result["susceptibility_phase_velocity_derivative_s_m"] == pytest.approx(
        finite_difference, rel=3e-4, abs=2e-10
    )
    assert np.isfinite(result["integral_phase_velocity_derivative_s3_m3"].real)
    assert np.isfinite(result["integral_phase_velocity_derivative_s3_m3"].imag)
    assert result["integral_phase_velocity_derivative_real_absolute_error"] >= 0.0
    assert result["integral_phase_velocity_derivative_imag_absolute_error"] >= 0.0
    assert result["susceptibility_angular_frequency_derivative_s"] == pytest.approx(
        result["susceptibility_phase_velocity_derivative_s_m"]
        / parameters["wavenumber_m_inverse"],
        rel=1e-12,
    )


def test_kappa_phase_derivative_formula_registry_is_synchronized_and_executable():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    formula = json.loads((root / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    packaged = json.loads((root / "tis" / "data" / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    key = "cts_kappa_landau_phase_velocity_derivative"
    matches = [item for item in formula if item.get("execution_contract", {}).get("formula_key") == key]
    assert len(matches) == 1
    assert matches == [item for item in packaged if item.get("execution_contract", {}).get("formula_key") == key]
    assert matches[0]["execution_contract"]["function_link"] == "tokamak_analysis.cts_kappa.kappa_electron_susceptibility"
    assert "susceptibility_phase_velocity_derivative_s_m" in {
        item["name"] for item in matches[0]["execution_contract"]["output_specs"]
    }



def test_kappa_default_uses_exact_retarded_landau_limit_and_preserves_resonant_damping():
    parameters = args()
    parameters.pop("pole_imaginary_part_m_s")
    result = kappa_electron_susceptibility(**parameters)
    assert result["landau_contour_method"] == "exact_eta_to_zero_plus"
    assert result["finite_pole_regularization_used"] is False
    assert result["pole_imaginary_part_m_s"] == 0.0
    assert result["landau_integral_tail_complete"] is True
    assert result["integral"].imag < 0.0
    assert result["integral_phase_velocity_derivative_s3_m3"].imag != 0.0


def test_kappa_exact_landau_susceptibility_converges_to_maxwellian_as_kappa_increases():
    from tokamak_analysis.cts_kappa import maxwellian_electron_susceptibility
    parameters = args(k=300.0)
    parameters.pop("pole_imaginary_part_m_s")
    thermal_speed = np.sqrt(2.0*1.602176634e-19*parameters["electron_temperature_eV"]/9.1093837015e-31)
    parameters["phase_velocity_m_s"] = 0.5*thermal_speed
    kappa_result = kappa_electron_susceptibility(**parameters)
    maxwellian_result = maxwellian_electron_susceptibility(
        parameters["phase_velocity_m_s"],
        electron_density_m3=parameters["electron_density_m3"],
        electron_temperature_eV=parameters["electron_temperature_eV"],
        wavenumber_m_inverse=parameters["wavenumber_m_inverse"],
    )
    relative_error = abs(kappa_result["electron_susceptibility"]-maxwellian_result["electron_susceptibility"])/abs(maxwellian_result["electron_susceptibility"])
    assert relative_error < 4.0e-3


def test_kappa_exact_landau_phase_derivative_matches_finite_difference():
    parameters = args(k=5.0)
    parameters.pop("pole_imaginary_part_m_s")
    parameters["phase_velocity_m_s"] = 0.71e6
    result = kappa_electron_susceptibility(**parameters)
    step = 1.0e-5*np.sqrt(1.602176634e-19*parameters["electron_temperature_eV"]/9.1093837015e-31)
    plus = kappa_electron_susceptibility(**{**parameters,"phase_velocity_m_s":parameters["phase_velocity_m_s"]+step})["electron_susceptibility"]
    minus = kappa_electron_susceptibility(**{**parameters,"phase_velocity_m_s":parameters["phase_velocity_m_s"]-step})["electron_susceptibility"]
    finite_difference = (plus-minus)/(2.0*step)
    assert result["susceptibility_phase_velocity_derivative_s_m"] == pytest.approx(finite_difference,rel=2.0e-6,abs=1.0e-10)


def test_explicit_kappa_pole_displacement_retains_finite_pole_mode():
    result = kappa_electron_susceptibility(**args())
    assert result["landau_contour_method"] == "finite_positive_imaginary_pole_regularization"
    assert result["finite_pole_regularization_used"] is True
    assert result["pole_imaginary_part_m_s"] == 2.0e3


def test_kappa_drift_shifts_landau_resonance_to_electron_rest_frame():
    parameters = args(k=4.5)
    parameters.pop("pole_imaginary_part_m_s", None)
    flow = 2.6e5
    drifting = kappa_electron_susceptibility(
        **parameters, electron_flow_projection_m_s=flow
    )
    rest_frame = kappa_electron_susceptibility(
        **{**parameters, "phase_velocity_m_s": parameters["phase_velocity_m_s"]-flow}
    )
    assert drifting["electron_susceptibility"] == pytest.approx(
        rest_frame["electron_susceptibility"], rel=2e-12, abs=1e-13
    )
    assert drifting["susceptibility_phase_velocity_derivative_s_m"] == pytest.approx(
        rest_frame["susceptibility_phase_velocity_derivative_s_m"], rel=2e-12, abs=1e-13
    )
    assert drifting["phase_velocity_in_electron_rest_frame_m_s"] == parameters["phase_velocity_m_s"]-flow


def test_maxwellian_drift_shifts_fried_conte_argument_and_derivative():
    from tokamak_analysis.cts_kappa import maxwellian_electron_susceptibility

    parameters = {
        "phase_velocity_m_s": np.array([-1.2e6, 0.0, 1.7e6]),
        "electron_density_m3": 2.0e19,
        "electron_temperature_eV": 800.0,
        "wavenumber_m_inverse": 250.0,
    }
    flow = -3.1e5
    drifting = maxwellian_electron_susceptibility(
        **parameters, electron_flow_projection_m_s=flow
    )
    rest_frame = maxwellian_electron_susceptibility(
        **{**parameters, "phase_velocity_m_s": parameters["phase_velocity_m_s"]-flow}
    )
    np.testing.assert_allclose(
        drifting["electron_susceptibility"], rest_frame["electron_susceptibility"],
        rtol=2e-13, atol=1e-14,
    )
    np.testing.assert_allclose(
        drifting["susceptibility_phase_velocity_derivative_s_m"],
        rest_frame["susceptibility_phase_velocity_derivative_s_m"],
        rtol=2e-13, atol=1e-14,
    )
    np.testing.assert_allclose(
        drifting["phase_velocity_in_electron_rest_frame_m_s"],
        parameters["phase_velocity_m_s"]-flow,
    )


def test_isotropic_kappa_and_maxwellian_reject_superluminal_flow():
    from tokamak_analysis.cts_kappa import maxwellian_electron_susceptibility

    parameters = args()
    with pytest.raises(ValueError, match="strictly subluminal"):
        kappa_electron_susceptibility(**parameters, electron_flow_projection_m_s=299792458.0)
    with pytest.raises(ValueError, match="strictly subluminal"):
        maxwellian_electron_susceptibility(
            0.0, electron_density_m3=2e19, electron_temperature_eV=800.0,
            wavenumber_m_inverse=250.0, electron_flow_projection_m_s=299792458.0
        )


def test_bi_kappa_drift_uses_projected_rest_frame_and_isotropic_limit():
    from tokamak_analysis.cts_kappa import bi_kappa_electron_susceptibility

    parameters = dict(
        v=np.linspace(-5e7, 5e7, 2001),
        phase_velocity_m_s=1.0e6,
        electron_density_m3=2.0e18,
        electron_temperature_parallel_eV=250.0,
        electron_temperature_perpendicular_eV=250.0,
        magnetic_field_unit=[0.0, 0.0, 1.0],
        scattering_vector_unit=[1.0, 0.0, 0.0],
        kappa=4.5,
        wavenumber_m_inverse=120.0,
        pole_imaginary_part_m_s=2.0e3,
    )
    flow = 2.6e5
    drifting = bi_kappa_electron_susceptibility(
        **parameters, electron_flow_projection_m_s=flow
    )
    rest_frame = kappa_electron_susceptibility(
        parameters["v"],
        parameters["phase_velocity_m_s"] - flow,
        electron_density_m3=parameters["electron_density_m3"],
        electron_temperature_eV=250.0,
        kappa=parameters["kappa"],
        wavenumber_m_inverse=parameters["wavenumber_m_inverse"],
        pole_imaginary_part_m_s=parameters["pole_imaginary_part_m_s"],
    )
    assert drifting["electron_susceptibility"] == pytest.approx(
        rest_frame["electron_susceptibility"], rel=2e-12, abs=1e-13
    )
    assert drifting["phase_velocity_in_electron_rest_frame_m_s"] == pytest.approx(
        parameters["phase_velocity_m_s"] - flow
    )
    assert drifting["projected_temperature_eV"] == pytest.approx(250.0)


def test_bi_kappa_rejects_superluminal_projected_electron_flow():
    from tokamak_analysis.cts_kappa import bi_kappa_electron_susceptibility

    with pytest.raises(ValueError, match="strictly subluminal"):
        bi_kappa_electron_susceptibility(
            np.linspace(-5e7, 5e7, 2001), 1.0e6,
            electron_density_m3=2.0e18,
            electron_temperature_parallel_eV=250.0,
            electron_temperature_perpendicular_eV=250.0,
            magnetic_field_unit=[0.0, 0.0, 1.0],
            scattering_vector_unit=[1.0, 0.0, 0.0],
            kappa=4.0, wavenumber_m_inverse=120.0,
            electron_flow_projection_m_s=299792458.0,
        )

def test_kappa_moment_hierarchy_reports_fourth_moment_only_when_it_exists():
    finite = kappa_electron_susceptibility(**args(4.0))
    assert finite["fourth_moment_exists"] is True
    assert finite["kurtosis"] == pytest.approx(5.0)
    assert finite["excess_kurtosis"] == pytest.approx(2.0)
    expected_variance = 1.602176634e-19 * 250.0 / 9.1093837015e-31
    assert finite["fourth_central_moment_m4_s4"] == pytest.approx(
        5.0 * expected_variance**2
    )

    infinite = kappa_electron_susceptibility(**args(2.0))
    assert infinite["fourth_moment_exists"] is False
    assert infinite["kurtosis"] is None
    assert infinite["excess_kurtosis"] is None
    assert infinite["fourth_central_moment_m4_s4"] is None


def test_kappa_analytic_pdf_is_not_conditioned_on_velocity_grid_coverage():
    narrow = args()
    narrow["v"] = np.linspace(-2.0e7, 2.0e7, 1001)
    narrow["minimum_velocity_grid_probability_coverage"] = 0.5
    result = kappa_electron_susceptibility(**narrow)
    assert result["velocity_grid_probability_coverage"] < 1.0
    np.testing.assert_allclose(
        result["normalized_probability_density_s_m"],
        result["analytic_kappa_pdf_s_m"],
        rtol=0.0, atol=0.0,
    )

@pytest.mark.parametrize("eta_fraction", [0.0, 0.17])
def test_cts_kappa_susceptibility_uses_shared_generalized_dispersion_kernel(eta_fraction):
    from tokamak_analysis.cts_kappa_dispersion import kappa_dispersion_function

    kappa = 4.3
    ne = 2.0e18
    te = 250.0
    mass = 9.1093837015e-31
    charge = 1.602176634e-19
    eps0 = 8.8541878128e-12
    wave_number = 120.0
    # The generalized dispersion coordinate is normalized by
    # sqrt(((2*kappa-3)/kappa) * k_B*T/m), not by the Student-t scale.
    velocity_scale = np.sqrt(
        ((2.0*kappa-3.0)/kappa) * charge*te/mass
    )
    relative_phase = 0.63*velocity_scale
    flow = -0.21*velocity_scale
    eta = eta_fraction*velocity_scale
    parameters = dict(
        v=np.linspace(-18.0*velocity_scale, 18.0*velocity_scale, 2401),
        phase_velocity_m_s=relative_phase+flow,
        electron_density_m3=ne,
        electron_temperature_eV=te,
        kappa=kappa,
        wavenumber_m_inverse=wave_number,
        electron_flow_projection_m_s=flow,
        minimum_velocity_grid_probability_coverage=0.99,
    )
    if eta_fraction > 0.0:
        parameters["pole_imaginary_part_m_s"] = eta
    result = kappa_electron_susceptibility(**parameters)
    kernel = kappa_dispersion_function(
        relative_phase/velocity_scale, eta/velocity_scale, kappa
    )
    expected_integral = kernel["kappa_dispersion_integral"]/velocity_scale**2
    expected_derivative = kernel["kappa_dispersion_integral_derivative"]/velocity_scale**3
    omega_pe_squared = ne*charge**2/(eps0*mass)
    expected_susceptibility = -omega_pe_squared/wave_number**2*expected_integral
    expected_susceptibility_derivative = (
        -omega_pe_squared/wave_number**2*expected_derivative
    )
    assert result["dispersion_kernel"] == "shared_full_tail_generalized_kappa_landau_integral"
    assert result["normalized_phase_velocity"] == pytest.approx(
        complex(relative_phase/velocity_scale, eta/velocity_scale)
    )
    assert result["integral"] == pytest.approx(expected_integral, rel=2e-11, abs=1e-18)
    assert result["integral_phase_velocity_derivative_s3_m3"] == pytest.approx(
        expected_derivative, rel=2e-11, abs=1e-24
    )
    assert result["electron_susceptibility"] == pytest.approx(
        expected_susceptibility, rel=2e-11, abs=1e-14
    )
    assert result["susceptibility_phase_velocity_derivative_s_m"] == pytest.approx(
        expected_susceptibility_derivative, rel=2e-11, abs=1e-18
    )

