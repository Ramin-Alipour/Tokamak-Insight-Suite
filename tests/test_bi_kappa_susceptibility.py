import numpy as np

from tokamak_analysis.cts_kappa import (
    bi_kappa_electron_susceptibility,
    kappa_electron_susceptibility,
)


def _common():
    return {
        "v": np.linspace(-2.0e8, 2.0e8, 1601),
        "phase_velocity_m_s": 1.1e6,
        "electron_density_m3": 2.0e19,
        "kappa": 4.0,
        "wavenumber_m_inverse": 250.0,
        "minimum_velocity_grid_probability_coverage": 0.999,
    }


def test_bi_kappa_isotropic_limit_matches_existing_kappa_solver():
    args = _common()
    common_temperature = 900.0
    bi = bi_kappa_electron_susceptibility(
        **args,
        electron_temperature_parallel_eV=common_temperature,
        electron_temperature_perpendicular_eV=common_temperature,
        magnetic_field_unit=[0.0, 0.0, 1.0],
        scattering_vector_unit=[0.6, 0.0, 0.8],
    )
    isotropic = kappa_electron_susceptibility(
        args["v"], args["phase_velocity_m_s"],
        electron_density_m3=args["electron_density_m3"],
        electron_temperature_eV=common_temperature,
        kappa=args["kappa"],
        wavenumber_m_inverse=args["wavenumber_m_inverse"],
        minimum_velocity_grid_probability_coverage=args["minimum_velocity_grid_probability_coverage"],
    )
    np.testing.assert_allclose(
        bi["electron_susceptibility"],
        isotropic["electron_susceptibility"],
        rtol=2e-11,
        atol=1e-13,
    )
    assert bi["projected_temperature_eV"] == common_temperature


def test_bi_kappa_directional_temperature_obeys_gyrotropic_projection():
    args = _common()
    result = bi_kappa_electron_susceptibility(
        **args,
        electron_temperature_parallel_eV=1500.0,
        electron_temperature_perpendicular_eV=500.0,
        magnetic_field_unit=[0.0, 0.0, 1.0],
        scattering_vector_unit=[np.sqrt(0.5), 0.0, np.sqrt(0.5)],
    )
    np.testing.assert_allclose(result["magnetic_pitch_cosine"], np.sqrt(0.5))
    np.testing.assert_allclose(result["projected_temperature_eV"], 1000.0)
    assert result["separate_parallel_perpendicular_temperatures_identifiable_from_one_view"] is False
    assert result["distribution_model"] == "gyrotropic_3D_bi_Kappa_exact_1D_directional_marginal"


def test_bi_kappa_rejects_non_unit_geometry():
    args = _common()
    try:
        bi_kappa_electron_susceptibility(
            **args,
            electron_temperature_parallel_eV=1500.0,
            electron_temperature_perpendicular_eV=500.0,
            magnetic_field_unit=[0.0, 0.0, 2.0],
            scattering_vector_unit=[0.0, 0.0, 1.0],
        )
    except ValueError as exc:
        assert "unit vectors" in str(exc)
    else:
        raise AssertionError("non-unit geometry must be rejected")


def test_bi_kappa_is_executable_through_public_advanced_physics_dispatcher():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    args = _common()
    result = run_advanced_thomson_physics(
        "bi_kappa_electrostatic_susceptibility",
        **args,
        electron_temperature_parallel_eV=1200.0,
        electron_temperature_perpendicular_eV=800.0,
        magnetic_field_unit=[0.0, 0.0, 1.0],
        scattering_vector_unit=[0.0, 0.0, 1.0],
    )
    assert result["distribution_model"] == "gyrotropic_3D_bi_Kappa_exact_1D_directional_marginal"
    np.testing.assert_allclose(result["projected_temperature_eV"], 1200.0)


def test_drifting_bi_kappa_uses_electron_rest_frame_landau_resonance():
    args = _common()
    flow = 3.7e5
    drifting = bi_kappa_electron_susceptibility(
        **args,
        electron_temperature_parallel_eV=1500.0,
        electron_temperature_perpendicular_eV=500.0,
        magnetic_field_unit=[0.0, 0.0, 1.0],
        scattering_vector_unit=[np.sqrt(0.5), 0.0, np.sqrt(0.5)],
        electron_flow_projection_m_s=flow,
    )
    rest_frame = bi_kappa_electron_susceptibility(
        **{**args, "phase_velocity_m_s": args["phase_velocity_m_s"] - flow},
        electron_temperature_parallel_eV=1500.0,
        electron_temperature_perpendicular_eV=500.0,
        magnetic_field_unit=[0.0, 0.0, 1.0],
        scattering_vector_unit=[np.sqrt(0.5), 0.0, np.sqrt(0.5)],
    )
    np.testing.assert_allclose(
        drifting["electron_susceptibility"],
        rest_frame["electron_susceptibility"],
        rtol=2e-12, atol=1e-13,
    )
    np.testing.assert_allclose(
        drifting["susceptibility_phase_velocity_derivative_s_m"],
        rest_frame["susceptibility_phase_velocity_derivative_s_m"],
        rtol=2e-12, atol=1e-13,
    )
    assert drifting["phase_velocity_lab_m_s"] == args["phase_velocity_m_s"]
    assert drifting["phase_velocity_in_electron_rest_frame_m_s"] == args["phase_velocity_m_s"] - flow
    assert drifting["electron_flow_projection_m_s"] == flow


def test_drifting_bi_kappa_rejects_superluminal_projected_flow():
    args = _common()
    try:
        bi_kappa_electron_susceptibility(
            **args,
            electron_temperature_parallel_eV=1500.0,
            electron_temperature_perpendicular_eV=500.0,
            magnetic_field_unit=[0.0, 0.0, 1.0],
            scattering_vector_unit=[0.0, 0.0, 1.0],
            electron_flow_projection_m_s=299792458.0,
        )
    except ValueError as exc:
        assert "strictly subluminal" in str(exc)
    else:
        raise AssertionError("superluminal projected flow must be rejected")
