import numpy as np
import pytest

from tokamak_analysis.cts_magnetoionic import (
    cold_plasma_dielectric_tensor,
    cold_plasma_magnetoionic_modes,
)

E = 1.602176634e-19
EPS0 = 8.8541878128e-12
ME = 9.1093837015e-31


def _electron_only(density=1.0e18):
    return [{"name": "e-", "mass_kg": ME, "charge_C": -E, "density_m3": density}]


def test_unmagnetized_limit_is_isotropic_and_matches_plasma_frequency_relation():
    frequency = 100e9
    density = 1.0e18
    result = cold_plasma_dielectric_tensor(
        frequency, [0.0, 0.0, 0.0], _electron_only(density)
    )
    omega = 2*np.pi*frequency
    omega_p2 = density*E**2/(EPS0*ME)
    expected = 1.0 - omega_p2/omega**2
    np.testing.assert_allclose(result["dielectric_tensor_lab"], expected*np.eye(3), rtol=2e-14, atol=2e-14)
    assert result["stix_R"] == pytest.approx(expected)
    assert result["stix_L"] == pytest.approx(expected)
    assert result["stix_D"] == pytest.approx(0.0, abs=1e-14)


def test_parallel_propagation_recovers_right_and_left_circular_branches():
    frequency = 100e9
    field = [0.0, 0.0, 2.0]
    result = cold_plasma_magnetoionic_modes(
        frequency, field, _electron_only(),
        wave_normal_direction=[0.0, 0.0, 1.0],
    )
    roots = sorted((m["refractive_index_squared"].real for m in result["modes"]))
    expected = sorted([result["stix_R"].real, result["stix_L"].real])
    np.testing.assert_allclose(roots, expected, rtol=2e-12, atol=2e-12)
    assert all(m["dispersion_residual"] < 1e-10 for m in result["modes"])


def test_perpendicular_ordinary_mode_is_stix_P():
    result = cold_plasma_magnetoionic_modes(
        100e9, [0.0, 0.0, 2.0], _electron_only(),
        wave_normal_direction=[1.0, 0.0, 0.0],
    )
    roots = np.asarray([m["refractive_index_squared"] for m in result["modes"]])
    assert np.min(np.abs(roots - result["stix_P"])) < 1e-9
    assert result["mode_conversion_performed"] is False


@pytest.mark.parametrize(
    "frequency,field,species,direction",
    [
        (0.0, [0, 0, 1], _electron_only(), [0, 0, 1]),
        (100e9, [0, 0], _electron_only(), [0, 0, 1]),
        (100e9, [0, 0, 1], [], [0, 0, 1]),
        (100e9, [0, 0, 1], _electron_only(), [0, 0, 0]),
    ],
)
def test_magnetoionic_solver_rejects_invalid_physical_inputs(frequency, field, species, direction):
    with pytest.raises(ValueError):
        cold_plasma_magnetoionic_modes(
            frequency, field, species, wave_normal_direction=direction
        )


def test_cyclotron_pole_is_not_hidden_by_artificial_damping():
    gyrofrequency = E*2.0/ME
    with pytest.raises(ValueError, match="cyclotron pole"):
        cold_plasma_dielectric_tensor(
            gyrofrequency/(2*np.pi), [0.0, 0.0, 2.0], _electron_only()
        )



def test_cold_plasma_modes_are_reachable_through_advanced_physics_dispatcher():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    result = run_advanced_thomson_physics(
        "cold_plasma_magnetoionic_modes",
        frequency_Hz=100e9,
        magnetic_field_T=[0.0, 0.0, 2.0],
        species=_electron_only(),
        wave_normal_direction=[0.0, 0.0, 1.0],
    )
    assert result["public_physics_entrypoint"] == "cold_plasma_magnetoionic_modes"
    assert result["automatic_model_substitution"] is False
    assert len(result["modes"]) == 2
    assert result["mode_conversion_performed"] is False



def test_magnetoionic_mode_solver_does_not_mutate_wave_normal_input():
    direction = np.array([2.0, 0.0, 0.0])
    original = direction.copy()
    cold_plasma_magnetoionic_modes(
        100e9, [0.0, 0.0, 2.0], _electron_only(),
        wave_normal_direction=direction,
    )
    np.testing.assert_array_equal(direction, original)



def test_species_resolved_cold_fluid_collisions_modify_all_stix_responses():
    frequency = 100e9
    density = 1.0e18
    collision = 2.0e8
    species = _electron_only(density)
    species[0]["collision_frequency_rad_s"] = collision
    result = cold_plasma_dielectric_tensor(
        frequency, [0.0, 0.0, 2.0], species
    )
    omega = 2*np.pi*frequency
    omega_p2 = density*E**2/(EPS0*ME)
    gyro = -E*2.0/ME
    expected_p = 1.0 - omega_p2/(omega*(omega + 1j*collision))
    expected_r = 1.0 - omega_p2/(omega*(omega + 1j*collision + gyro))
    expected_l = 1.0 - omega_p2/(omega*(omega + 1j*collision - gyro))
    assert result["stix_P"] == pytest.approx(expected_p)
    assert result["stix_R"] == pytest.approx(expected_r)
    assert result["stix_L"] == pytest.approx(expected_l)
    assert result["species"][0]["collision_frequency_rad_s"] == collision
    assert np.linalg.norm(
        result["dielectric_tensor_lab"] - result["dielectric_tensor_lab"].conj().T
    ) > 0.0


def test_zero_collision_parameter_recovers_collisionless_tensor_exactly():
    species = _electron_only()
    baseline = cold_plasma_dielectric_tensor(100e9, [0.0, 0.0, 2.0], species)
    species[0]["collision_frequency_rad_s"] = 0.0
    explicit_zero = cold_plasma_dielectric_tensor(100e9, [0.0, 0.0, 2.0], species)
    np.testing.assert_allclose(
        explicit_zero["dielectric_tensor_lab"],
        baseline["dielectric_tensor_lab"],
        rtol=0.0,
        atol=0.0,
    )


def test_negative_or_nonfinite_collision_frequency_is_rejected():
    for value in (-1.0, np.nan, np.inf, 1.0 + 2.0j, True):
        species = _electron_only()
        species[0]["collision_frequency_rad_s"] = value
        with pytest.raises(ValueError, match="collision_frequency_rad_s"):
            cold_plasma_dielectric_tensor(100e9, [0.0, 0.0, 2.0], species)


def test_cold_fluid_collisional_absorption_matches_drude_work_relation():
    from tokamak_analysis.cts_magnetoionic import cold_plasma_absorbed_power_density

    frequency = 80e9
    omega = 2*np.pi*frequency
    density = 2.5e18
    collision = 3.0e8
    field = np.array([2.0+1.0j, -0.5j, 1.25])
    species = _electron_only(density)
    species[0]["collision_frequency_rad_s"] = collision
    result = cold_plasma_absorbed_power_density(
        frequency, [0.0, 0.0, 0.0], species, field,
    )
    omega_p2 = density*E**2/(EPS0*ME)
    epsilon_imaginary = omega_p2*collision/(omega*(omega**2+collision**2))
    expected = 0.5*EPS0*omega*epsilon_imaginary*np.vdot(field, field).real
    assert result["absorbed_power_density_W_m3"] == pytest.approx(expected, rel=2e-13)
    assert np.linalg.eigvalsh(
        result["hermitian_dissipative_dielectric_tensor"]
    ).min() >= -1e-13
    assert "cold_fluid_collisional_absorption" in result["absorption_scope"]


def test_cold_fluid_absorption_is_zero_in_collisionless_limit():
    from tokamak_analysis.cts_magnetoionic import cold_plasma_absorbed_power_density

    result = cold_plasma_absorbed_power_density(
        100e9, [0.0, 0.0, 2.0], _electron_only(),
        [1.0+0.5j, 0.25, -0.75j],
    )
    assert result["absorbed_power_density_W_m3"] == pytest.approx(0.0, abs=1e-30)


def test_cold_fluid_absorption_rejects_invalid_electric_field():
    from tokamak_analysis.cts_magnetoionic import cold_plasma_absorbed_power_density

    for field in ([1.0, 2.0], [1.0, np.nan, 2.0], [1.0, 2.0, 1.0+1.0j, 3.0]):
        with pytest.raises(ValueError, match="electric_field_V_m"):
            cold_plasma_absorbed_power_density(
                100e9, [0.0, 0.0, 1.0], _electron_only(), field,
            )


def test_cold_fluid_absorption_is_available_from_public_dispatcher():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    result = run_advanced_thomson_physics(
        "cold_plasma_absorbed_power_density",
        frequency_Hz=100e9,
        magnetic_field_T=[0.0, 0.0, 0.0],
        species=[{
            "mass_kg": ME, "charge_C": -E, "density_m3": 1e18,
            "collision_frequency_rad_s": 1e8,
        }],
        electric_field_V_m=[1.0, 0.0, 0.0],
    )
    assert result["public_physics_entrypoint"] == "cold_plasma_absorbed_power_density"
    assert result["absorbed_power_density_W_m3"] > 0.0
    assert result["automatic_model_substitution"] is False


def test_magnetized_multispecies_collisional_dissipation_is_passive():
    from tokamak_analysis.cts_magnetoionic import (
        cold_plasma_absorbed_power_density,
        cold_plasma_dielectric_tensor,
    )

    species = [
        {
            "name": "e-", "mass_kg": ME, "charge_C": -E,
            "density_m3": 3.0e19, "collision_frequency_rad_s": 4.0e7,
        },
        {
            "name": "D+", "mass_kg": 2.0*ME*1836.15267343, "charge_C": E,
            "density_m3": 1.5e19, "collision_frequency_rad_s": 8.0e6,
        },
    ]
    dielectric = cold_plasma_dielectric_tensor(
        75e9, [0.3, -0.4, 2.0], species,
    )
    assert dielectric["model_scope"].endswith(
        "_with_species_resolved_cold_fluid_collisions"
    )
    electric = np.array([1.0+0.2j, -0.3+0.7j, 0.5-0.1j])
    result = cold_plasma_absorbed_power_density(
        75e9, [0.3, -0.4, 2.0], species, electric,
    )
    eigenvalues = np.linalg.eigvalsh(
        result["hermitian_dissipative_dielectric_tensor"]
    )
    assert eigenvalues.min() >= -1e-11 * max(eigenvalues.max(), 1.0)
    assert result["absorbed_power_density_W_m3"] > 0.0


def test_magnetized_mode_opacity_and_lte_emissivity_follow_energy_balance():
    from tokamak_analysis.cts_magnetoionic import (
        cold_plasma_mode_radiative_coefficients,
    )
    from tokamak_analysis.cts_em_background import planck_spectral_radiance

    species = [
        {
            "name": "e-", "mass_kg": ME, "charge_C": -E,
            "density_m3": 3.0e19, "collision_frequency_rad_s": 4.0e7,
        },
        {
            "name": "D+", "mass_kg": 2.0*ME*1836.15267343, "charge_C": E,
            "density_m3": 1.5e19, "collision_frequency_rad_s": 8.0e6,
        },
    ]
    frequency = 75.0e9
    temperature = 1800.0
    field = np.array([1.0+0.2j, -0.3+0.7j, 0.5-0.1j])
    flux = 2.5e6
    result = cold_plasma_mode_radiative_coefficients(
        frequency, [0.3, -0.4, 2.0], species, field,
        mode_poynting_flux_W_m2=flux,
        electron_temperature_eV=temperature,
        assume_lte=True,
    )
    expected_alpha = result["absorbed_power_density_W_m3"] / flux
    expected_planck = planck_spectral_radiance(
        np.array([frequency]), temperature * E / 1.380649e-23
    )[0]
    assert result["absorption_coefficient_m_inverse"] == pytest.approx(
        expected_alpha, rel=2e-15
    )
    assert result["thermal_emissivity_W_m3_sr_Hz"] == pytest.approx(
        expected_alpha * expected_planck, rel=2e-15
    )
    assert result["lte_assumed"] is True
    assert result["absorption_coefficient_m_inverse"] > 0.0


def test_magnetized_mode_emissivity_is_not_invented_without_explicit_lte():
    from tokamak_analysis.cts_magnetoionic import (
        cold_plasma_mode_radiative_coefficients,
    )

    species = [
        {
            "name": "e-", "mass_kg": ME, "charge_C": -E,
            "density_m3": 3.0e19, "collision_frequency_rad_s": 4.0e7,
        },
    ]
    result = cold_plasma_mode_radiative_coefficients(
        75.0e9, [0.0, 0.0, 2.0], species,
        np.array([1.0+0.0j, 0.2+0.1j, 0.0+0.0j]),
        mode_poynting_flux_W_m2=1.0e6,
        electron_temperature_eV=1000.0,
        assume_lte=False,
    )
    assert result["thermal_emissivity_W_m3_sr_Hz"] is None
    assert result["planck_source_radiance_W_m2_sr_Hz"] is None
    assert result["emissivity_relation"].startswith("not_evaluated")


def test_magnetized_mode_radiative_closure_is_publicly_dispatched():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    result = run_advanced_thomson_physics(
        "cold_plasma_mode_radiative_coefficients",
        frequency_Hz=75.0e9,
        magnetic_field_T=[0.0, 0.0, 2.0],
        species=[{
            "name": "e-", "mass_kg": ME, "charge_C": -E,
            "density_m3": 3.0e19, "collision_frequency_rad_s": 4.0e7,
        }],
        electric_field_V_m=np.array([1.0+0.0j, 0.2+0.1j, 0.0+0.0j]),
        mode_poynting_flux_W_m2=1.0e6,
        electron_temperature_eV=1000.0,
        assume_lte=True,
    )
    assert result["public_physics_entrypoint"] == "cold_plasma_mode_radiative_coefficients"
    assert result["thermal_emissivity_W_m3_sr_Hz"] > 0.0
    assert result["automatic_model_substitution"] is False



def test_parallel_magnetoionic_group_velocity_matches_implicit_dispersion_derivative():
    from tokamak_analysis.cts_magnetoionic import cold_plasma_mode_group_velocity
    from scipy.constants import c

    frequency = 100.0e9
    field = [0.0, 0.0, 2.0]
    species = _electron_only(1.0e18)
    result = cold_plasma_mode_group_velocity(
        frequency, field, species, wave_normal_direction=[0.0, 0.0, 1.0],
        mode_index=0,
    )
    omega = 2.0 * np.pi * frequency
    omega_p2 = species[0]["density_m3"] * E**2 / (EPS0 * ME)
    # The branch order is the polynomial root order; independently verify
    # the returned derivative against a finite-difference derivative of its
    # own selected n(omega) branch rather than assuming R/L ordering.
    def selected_k(freq):
        from tokamak_analysis.cts_magnetoionic import cold_plasma_magnetoionic_modes
        modes = cold_plasma_magnetoionic_modes(
            freq, field, species, wave_normal_direction=[0.0, 0.0, 1.0]
        )
        root = modes["modes"][0]["refractive_index_squared"]
        return np.sqrt(root.real) * 2.0 * np.pi * freq / c
    delta = frequency * 1.0e-5
    dk_domega = (selected_k(frequency + delta) - selected_k(frequency - delta)) / (
        4.0 * np.pi * delta
    )
    expected = 1.0 / dk_domega
    assert result["group_velocity_m_s"][2] == pytest.approx(expected, rel=3e-4)
    assert result["group_velocity_m_s"][0] == pytest.approx(0.0, abs=1e-7)
    assert result["group_velocity_m_s"][1] == pytest.approx(0.0, abs=1e-7)
    assert result["wave_normal_to_ray_walkoff_angle_rad"] == pytest.approx(0.0, abs=1e-7)
    assert 0.0 < result["group_speed_m_s"] < c
    assert result["brillouin_energy_density_J_m3_per_V2_m2"] > 0.0
    np.testing.assert_allclose(
        result["energy_velocity_m_s"], result["group_velocity_m_s"],
        rtol=3e-4, atol=1e-5,
    )
    assert result["relative_group_energy_velocity_difference"] < 3e-4


def test_oblique_magnetoionic_group_velocity_reports_ray_walkoff():
    from tokamak_analysis.cts_magnetoionic import cold_plasma_mode_group_velocity

    result = cold_plasma_mode_group_velocity(
        100.0e9, [0.0, 0.0, 2.0], _electron_only(1.0e18),
        wave_normal_direction=[0.8, 0.0, 0.6], mode_index=0,
    )
    assert result["group_velocity_m_s"].shape == (3,)
    assert np.all(np.isfinite(result["group_velocity_m_s"]))
    assert result["ray_direction"].shape == (3,)
    assert np.linalg.norm(result["ray_direction"]) == pytest.approx(1.0, rel=1e-12)
    assert 0.0 <= result["wave_normal_to_ray_walkoff_angle_rad"] <= np.pi
    assert result["brillouin_energy_density_J_m3_per_V2_m2"] > 0.0
    np.testing.assert_allclose(
        result["energy_velocity_m_s"], result["group_velocity_m_s"],
        rtol=5e-4, atol=1e-5,
    )


def test_group_velocity_rejects_collisional_and_degenerate_modes():
    from tokamak_analysis.cts_magnetoionic import cold_plasma_mode_group_velocity

    collisional = _electron_only()
    collisional[0]["collision_frequency_rad_s"] = 1.0e7
    with pytest.raises(ValueError, match="collisionless"):
        cold_plasma_mode_group_velocity(
            100.0e9, [0.0, 0.0, 2.0], collisional,
            wave_normal_direction=[0.0, 0.0, 1.0],
        )
    # In an unmagnetized isotropic plasma the two transverse modes are
    # exactly degenerate; a branch-specific derivative is not unique.
    with pytest.raises(ValueError, match="degenerate"):
        cold_plasma_mode_group_velocity(
            100.0e9, [0.0, 0.0, 0.0], _electron_only(),
            wave_normal_direction=[0.0, 0.0, 1.0],
        )


def test_group_velocity_is_publicly_dispatched():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    result = run_advanced_thomson_physics(
        "cold_plasma_mode_group_velocity",
        frequency_Hz=100.0e9,
        magnetic_field_T=[0.0, 0.0, 2.0],
        species=_electron_only(),
        wave_normal_direction=[0.0, 0.0, 1.0],
        mode_index=0,
    )
    assert result["public_physics_entrypoint"] == "cold_plasma_mode_group_velocity"
    assert result["automatic_model_substitution"] is False
    assert result["group_velocity_relation"].startswith("v_g,j")
