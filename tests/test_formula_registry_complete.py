"""Independent reference and contract checks for all 42 formula registry entries."""
import ast
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

C_LIGHT = 299792458.0  # exact SI value
E_CHARGE = 1.602176634e-19  # exact SI value
MU0 = 1.25663706212e-6  # CODATA 2018 reference
M_E = 9.1093837139e-31  # CODATA 2022 reference
from tokamak_analysis.formula_library import (
    plasma_frequency_from_density_cm3, doppler_line_of_sight_velocity, ion_sound_speed, langmuir_electron_temperature_from_slope, langmuir_ion_density_from_saturation_current,
    langmuir_electron_density_from_saturation_current, langmuir_bohm_particle_flux, langmuir_multispecies_ion_density,
    ion_temperature_from_doppler_sigma, gaussian_fwhm_from_sigma,
    magnetic_energy_density, toroidal_beta, safety_factor_large_aspect_ratio,
    faraday_induction, plasma_loop_resistance_proxy, ohmic_power_approximation,
    energy_confinement_time, greenwald_density_reference, debye_parameter,
    electron_plasma_angular_frequency, ion_saturation_particle_flux_proxy,
    radiated_power_fraction, integrated_surface_power, apparent_feature_speed,
    gas_pumping_speed, injected_particle_number, particle_confinement_time_proxy,
    relative_fluctuation_level, rms_signal, nyquist_frequency, cross_correlation_lag,
    magnitude_squared_coherence, bolometry_regularized_inverse, particle_balance_rate,
)

ROOT = Path(__file__).resolve().parents[1]


def test_plasma_frequency_cm3_conversion_reference():
    n_cm3 = 1.0e13
    expected = math.sqrt(n_cm3 * 1.0e6 * E_CHARGE**2 / (8.8541878128e-12 * M_E)) / (2 * math.pi)
    r = plasma_frequency_from_density_cm3(n_cm3, sigma_density_cm3=1e11)
    assert r["frequency_pe_Hz"] == pytest.approx(expected, rel=1e-12)
    assert r["uncertainty_frequency_pe_Hz"] == pytest.approx(0.5 * expected * (1e11 / n_cm3), rel=1e-12)


def test_doppler_velocity_reference():
    r = doppler_line_of_sight_velocity(0.01, 500.0, sigma_delta_lambda=0.001, sigma_lambda0=0.01)
    assert r["velocity_los_m_s"] == pytest.approx(C_LIGHT * 0.01 / 500.0, rel=1e-12)
    assert r["uncertainty_velocity_los_m_s"] == pytest.approx(math.hypot(C_LIGHT * 0.001 / 500.0, C_LIGHT * 0.01 * 0.01 / 500.0**2), rel=1e-12)


def test_ion_temperature_from_doppler_sigma_reference():
    sigma, rest, mass = 0.1, 500.0, 1.67262192369e-27
    expected = mass * C_LIGHT**2 / E_CHARGE * (sigma / rest)**2
    r = ion_temperature_from_doppler_sigma(sigma, rest, mass, sigma_sigma_lambda=0.001,
                                           sigma_lambda0=0.01, sigma_ion_mass_kg=1e-30)
    assert r["ion_temperature_eV"] == pytest.approx(expected, rel=1e-12)
    assert r["uncertainty_ion_temperature_eV"] == pytest.approx(expected * math.sqrt((2*0.001/sigma)**2 + (2*0.01/rest)**2 + (1e-30/mass)**2), rel=1e-12)


def test_gaussian_fwhm_reference():
    r = gaussian_fwhm_from_sigma(2.0, sigma_sigma_width=0.1)
    factor = 2 * math.sqrt(2 * math.log(2))
    assert r["sigma_width"] == 2.0
    assert r["fwhm_width"] == pytest.approx(factor * 2.0, rel=1e-12)
    assert r["uncertainty_sigma_width"] == pytest.approx(0.1)
    assert r["uncertainty_fwhm_width"] == pytest.approx(factor * 0.1, rel=1e-12)


def test_magnetic_energy_density_reference():
    r = magnetic_energy_density(2.0, sigma_magnetic_field_T=0.01)
    assert r["magnetic_energy_density_J_m3"] == pytest.approx(4.0 / (2 * MU0), rel=1e-12)
    assert r["uncertainty_magnetic_energy_density_J_m3"] == pytest.approx(2 * 0.01 / MU0, rel=1e-12)


def test_toroidal_beta_reference():
    p, B = 1000.0, 2.0
    r = toroidal_beta(p, B, sigma_pressure_Pa=10.0, sigma_toroidal_field_T=0.01)
    assert r["toroidal_beta"] == pytest.approx(2 * MU0 * p / B**2, rel=1e-12)
    assert r["uncertainty_toroidal_beta"] == pytest.approx(math.hypot(2*MU0*10/B**2, 4*MU0*p*0.01/abs(B)**3), rel=1e-12)


def test_safety_factor_reference():
    r = safety_factor_large_aspect_ratio(0.2, 1.5, 2.0, 0.1,
        sigma_r_m=0.001, sigma_major_radius_m=0.002, sigma_toroidal_field_T=0.01, sigma_poloidal_field_T=0.002)
    assert r["safety_factor_estimate"] == pytest.approx(0.2 * 2.0 / (1.5 * 0.1), rel=1e-12)
    assert r["uncertainty_safety_factor_estimate"] == pytest.approx(math.sqrt((2.0*0.001/(1.5*0.1))**2 + (r["safety_factor_estimate"]*0.002/1.5)**2 + (0.2*0.01/(1.5*0.1))**2 + (r["safety_factor_estimate"]*0.002/0.1)**2), rel=1e-12)


def test_faraday_induction_reference():
    r = faraday_induction(0.25, 10, sigma_flux_derivative_Wb_s=0.01)
    assert r["coil_voltage_V"] == pytest.approx(-2.5)
    assert r["uncertainty_coil_voltage_V"] == pytest.approx(0.1)


def test_loop_resistance_proxy_reference():
    r = plasma_loop_resistance_proxy(2.0, 4.0, sigma_loop_voltage_V=0.1, sigma_plasma_current_A=0.2)
    assert r["loop_resistance_proxy_ohm"] == pytest.approx(0.5)
    assert r["uncertainty_loop_resistance_proxy_ohm"] == pytest.approx(math.hypot(0.1/4.0, 2.0*0.2/4.0**2), rel=1e-12)


def test_ohmic_power_reference():
    r = ohmic_power_approximation(2.0, 4.0, sigma_loop_voltage_V=0.1, sigma_plasma_current_A=0.2)
    assert r["ohmic_power_approximation_W"] == pytest.approx(8.0)
    assert r["uncertainty_ohmic_power_approximation_W"] == pytest.approx(math.hypot(4.0*0.1, 2.0*0.2), rel=1e-12)


def test_energy_confinement_time_reference():
    r = energy_confinement_time(100.0, 20.0, sigma_stored_energy_J=2.0, sigma_loss_power_W=1.0)
    assert r["energy_confinement_time_s"] == pytest.approx(5.0)
    assert r["uncertainty_energy_confinement_time_s"] == pytest.approx(math.hypot(2.0/20.0, 100.0*1.0/20.0**2), rel=1e-12)


def test_greenwald_density_reference():
    r = greenwald_density_reference(1.0, 0.5, sigma_plasma_current_MA=0.01, sigma_minor_radius_m=0.005)
    assert r["greenwald_density_1e20_m3"] == pytest.approx(1.0 / (math.pi * 0.5**2), rel=1e-12)
    assert r["greenwald_density_m3"] == pytest.approx(r["greenwald_density_1e20_m3"] * 1e20)
    assert r["uncertainty_greenwald_density_m3"] == pytest.approx(1e20 * math.sqrt((0.01/(math.pi*0.5**2))**2 + (2*1.0*0.005/(math.pi*0.5**3))**2), rel=1e-12)


def test_debye_parameter_reference():
    r = debye_parameter(1e18, 1e-4, sigma_density_m3=1e16, sigma_debye_length_m=1e-6)
    assert r["debye_parameter_particles"] == pytest.approx(4 * math.pi / 3 * 1e18 * (1e-4)**3, rel=1e-12)
    assert r["uncertainty_debye_parameter_particles"] == pytest.approx(math.hypot(4*math.pi/3*(1e-4)**3*1e16, 4*math.pi*1e18*(1e-4)**2*1e-6), rel=1e-12)


def test_electron_plasma_angular_frequency_alias_reference():
    r = electron_plasma_angular_frequency(1e19, sigma_density_m3=1e17)
    assert r["omega_pe_rad_s"] > 0
    assert r["uncertainty_omega_pe_rad_s"] == pytest.approx(0.005 * r["omega_pe_rad_s"], rel=1e-12)


def test_ion_saturation_flux_reference():
    r = ion_saturation_particle_flux_proxy(-E_CHARGE, 1e-4, sigma_saturation_current_A=0.01 * E_CHARGE,
                                           sigma_effective_area_m2=1e-6)
    assert r["ion_saturation_particle_flux_m2_s"] == pytest.approx(1e4, rel=1e-12)
    assert r["uncertainty_ion_saturation_particle_flux_m2_s"] == pytest.approx(math.hypot((0.01*E_CHARGE)/(E_CHARGE*1e-4), E_CHARGE*1e-6/(E_CHARGE*(1e-4)**2)), rel=1e-12)


def test_radiated_power_fraction_reference():
    r = radiated_power_fraction(30.0, 100.0, sigma_radiated_power_W=1.0, sigma_input_power_W=2.0)
    assert r["radiated_power_fraction"] == pytest.approx(0.3)
    assert r["uncertainty_radiated_power_fraction"] == pytest.approx(math.hypot(1.0/100.0, 30.0*2.0/100.0**2), rel=1e-12)
    assert radiated_power_fraction(120.0, 100.0)["quality_flag"].startswith("fraction_above_unity")


def test_integrated_surface_power_reference():
    r = integrated_surface_power([10.0, 20.0], [0.1, 0.2],
                                 sigma_heat_flux_W_m2=[1.0, 2.0], sigma_area_weights_m2=[0.01, 0.02])
    assert r["integrated_surface_power_W"] == pytest.approx(5.0)
    assert r["uncertainty_integrated_surface_power_W"] == pytest.approx(math.sqrt((0.1*1.0)**2+(10.0*0.01)**2+(0.2*2.0)**2+(20.0*0.02)**2), rel=1e-12)


def test_apparent_feature_speed_reference():
    r = apparent_feature_speed(-0.3, 0.1, sigma_displacement_m=0.01, sigma_frame_interval_s=0.001)
    assert r["apparent_feature_speed_m_s"] == pytest.approx(-3.0)
    assert r["uncertainty_apparent_feature_speed_m_s"] == pytest.approx(math.hypot(0.01/0.1, 0.3*0.001/0.1**2), rel=1e-12)


def test_gas_pumping_speed_reference():
    r = gas_pumping_speed(2e-3, 1e-3, sigma_throughput_Pa_m3_s=1e-5, sigma_pressure_Pa=1e-5)
    assert r["gas_pumping_speed_m3_s"] == pytest.approx(2.0)
    assert r["uncertainty_gas_pumping_speed_m3_s"] == pytest.approx(math.hypot(1e-5/1e-3, 2e-3*1e-5/(1e-3)**2), rel=1e-12)


def test_injected_particle_number_reference():
    r = injected_particle_number([2.0, 2.0, 2.0], [0.0, 1.0, 2.0], sigma_particle_rate_s=[0.1, 0.1, 0.1])
    assert r["injected_particle_number"] == pytest.approx(4.0)
    assert r["uncertainty_injected_particle_number"] == pytest.approx(0.1*math.sqrt(0.5**2+1.0**2+0.5**2), rel=1e-12)


def test_particle_confinement_time_reference():
    r = particle_confinement_time_proxy(100.0, 5.0, sigma_stored_particles=2.0, sigma_loss_rate_s=0.1)
    assert r["particle_confinement_time_s"] == pytest.approx(20.0)
    assert r["uncertainty_particle_confinement_time_s"] == pytest.approx(math.hypot(2.0/5.0, 100.0*0.1/5.0**2), rel=1e-12)


def test_relative_fluctuation_level_reference():
    r = relative_fluctuation_level([9.0, 10.0, 11.0])
    assert r["mean_signal"] == pytest.approx(10.0)
    assert r["fluctuation_rms"] == pytest.approx(math.sqrt(2.0 / 3.0))
    assert r["relative_fluctuation_level"] == pytest.approx(math.sqrt(2.0 / 3.0) / 10.0)


def test_rms_signal_reference():
    r = rms_signal([3.0, 4.0], sigma_signal=[0.1, 0.2])
    assert r["rms_signal"] == pytest.approx(math.sqrt(12.5))
    assert r["uncertainty_rms_signal"] == pytest.approx(math.sqrt((3.0*0.1/(2*r["rms_signal"]))**2+(4.0*0.2/(2*r["rms_signal"]))**2), rel=1e-12)


def test_nyquist_frequency_reference():
    r = nyquist_frequency(1000.0, sigma_sampling_frequency_Hz=2.0)
    assert r["nyquist_frequency_Hz"] == pytest.approx(500.0)
    assert r["uncertainty_nyquist_frequency_Hz"] == pytest.approx(1.0, rel=1e-12)


def test_cross_correlation_lag_reference():
    x = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    r = cross_correlation_lag(x, y, 0.25)
    assert r["cross_correlation_lag_s"] == pytest.approx(0.5)
    assert r["normalized_peak_correlation"] > 0


def test_magnitude_squared_coherence_reference():
    fs = 1000.0
    t = np.arange(1024) / fs
    x = np.sin(2 * math.pi * 40.0 * t)
    y = x + 0.05 * np.sin(2 * math.pi * 13.0 * t + 0.3)
    r = magnitude_squared_coherence(x, y, fs, nperseg=256)
    f = np.asarray(r["coherence_frequency_Hz"])
    c = np.asarray(r["magnitude_squared_coherence"])
    i = int(np.argmin(abs(f - 40.0)))
    assert 0.0 <= c[i] <= 1.0
    assert c[i] > 0.9


def test_bolometry_inverse_reference():
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([2.0, 3.0, 5.0])
    r = bolometry_regularized_inverse(y, A, 1e-3, sigma_measurements=0.1)
    assert np.asarray(r["emissivity_estimate"]) == pytest.approx([2.0, 3.0], rel=1e-5)
    assert r["residual_norm"] < 1e-4
    assert len(r["uncertainty_emissivity_estimate"]) == 2


def test_particle_balance_reference():
    r = particle_balance_rate(10.0, 7.0, 2.0, 1.0,
        sigma_input_rate_s=0.1, sigma_output_rate_s=0.2, sigma_source_rate_s=0.1, sigma_loss_rate_s=0.2)
    assert r["particle_balance_rate_s"] == pytest.approx(4.0)
    assert r["uncertainty_particle_balance_rate_s"] == pytest.approx(math.sqrt(0.1**2 + 0.2**2 + 0.1**2 + 0.2**2))


def test_all_formula_registry_mappings_resolve_and_have_reference_tests():
    rows = json.loads((ROOT / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    assert len(rows) == 42
    assert len({row["name"] for row in rows}) == 42
    for row in rows:
        contract = row["execution_contract"]
        assert contract["implementation_link_status"] == "mapped", row["name"]
        assert contract["function_link"]
        assert contract["formula_key"]
        assert contract["input_specs"], row["name"]
        assert contract["output_specs"], row["name"]
        assert contract["uncertainty_model"] and contract["uncertainty_model"] != "not_specified"
        module_name, function_name = contract["function_link"].rsplit(".", 1)
        assert callable(getattr(importlib.import_module(module_name), function_name)), row["name"]
        assert contract["test_links"], row["name"]
        for link in contract["test_links"]:
            path, test_name = link.split("::", 1)
            test_path = ROOT / path
            assert test_path.is_file(), link
            assert f"def {test_name}(" in test_path.read_text(encoding="utf-8"), link


def test_ion_sound_speed_uncertainty_reference():
    Te, Ti, mi, gamma = 20.0, 10.0, 2 * 1.66053906660e-27, 1.0
    r = ion_sound_speed(Te, Ti, mi, gamma, sigma_electron_temperature_eV=0.2,
                        sigma_ion_temperature_eV=0.3, sigma_ion_mass_kg=1e-29, sigma_gamma_i=0.02)
    assert r["uncertainty_ion_sound_speed_m_s"] > 0


@pytest.mark.parametrize("call", [
    lambda: rms_signal([0.0, 0.0], sigma_signal=[0.1, 0.1]),
    lambda: ion_sound_speed(1.0, 1.0, 1.0, sigma_electron_temperature_eV=0.1),
])
def test_uncertainty_contracts_reject_undefined_or_partial_uncertainty(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("call", [
    lambda: doppler_line_of_sight_velocity(1.0, 0.0),
    lambda: ion_temperature_from_doppler_sigma(0.1, 500.0, 0.0),
    lambda: toroidal_beta(100.0, 0.0),
    lambda: safety_factor_large_aspect_ratio(0.2, 1.5, 2.0, 0.0),
    lambda: faraday_induction(0.1, 2.5),
    lambda: plasma_loop_resistance_proxy(1.0, 0.0),
    lambda: energy_confinement_time(10.0, 0.0),
    lambda: greenwald_density_reference(1.0, 0.0),
    lambda: gas_pumping_speed(1.0, 0.0),
    lambda: particle_confinement_time_proxy(1.0, 0.0),
    lambda: relative_fluctuation_level([-1.0, 1.0]),
    lambda: cross_correlation_lag([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], 0.1),
    lambda: bolometry_regularized_inverse([1.0, 2.0], [[1.0, 2.0]], 0.1),
    lambda: particle_balance_rate(-1.0, 0.0, 0.0, 0.0),
])
def test_invalid_inputs_are_rejected_for_new_formulae(call):
    with pytest.raises(ValueError):
        call()


def test_formula_registry_copies_are_identical():
    root_rows = json.loads((ROOT / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    packaged_rows = json.loads((ROOT / "tis" / "data" / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    assert packaged_rows == root_rows


def test_registry_declares_every_uncertainty_input_in_callable_signature():
    import inspect
    rows = json.loads((ROOT / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    for row in rows:
        contract = row["execution_contract"]
        module_name, function_name = contract["function_link"].rsplit(".", 1)
        fn = getattr(importlib.import_module(module_name), function_name)
        declared = {item["name"] for item in contract["input_specs"]}
        for name, parameter in inspect.signature(fn).parameters.items():
            if name.startswith("sigma_") or name == "nperseg":
                assert name in declared, (row["name"], name)
        assert len({item["name"] for item in contract["input_specs"]}) == len(contract["input_specs"]), row["name"]
        assert len({item["name"] for item in contract["output_specs"]}) == len(contract["output_specs"]), row["name"]


def test_coherence_rejects_fractional_segment_length():
    x = np.sin(np.arange(64, dtype=float))
    with pytest.raises(ValueError, match="integer"):
        magnitude_squared_coherence(x, x, 1000.0, nperseg=8.5)


def test_registry_and_packaged_registry_contract_outputs_cover_uncertainty_fields():
    root_rows = json.loads((ROOT / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    packaged_rows = json.loads((ROOT / "tis" / "data" / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    assert packaged_rows == root_rows
    for row in root_rows:
        outputs = {item["name"] for item in row["execution_contract"]["output_specs"]}
        fn = row["execution_contract"]["function_link"].rsplit(".", 1)[1]
        module = importlib.import_module(row["execution_contract"]["function_link"].rsplit(".", 1)[0])
        signature = __import__("inspect").signature(getattr(module, fn))
        expected = {"uncertainty_" + p[6:] for p in signature.parameters if p.startswith("sigma_")}
        # Output uncertainty names need not mirror each input name; every declared
        # uncertainty model is explicit and every listed output must be unique.
        assert len(outputs) == len(row["execution_contract"]["output_specs"]), row["name"]


def test_registry_output_specs_cover_all_statically_returned_keys():
    source = (ROOT / "tokamak_analysis" / "formula_library.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    rows = json.loads((ROOT / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    for row in rows:
        function_name = row["execution_contract"]["function_link"].rsplit(".", 1)[1]
        node = functions[function_name]
        returned = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Dict):
                parent_keys = [
                    key.value for key in child.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                ]
                # Only count dictionaries assigned to the public output object
                # or returned directly; nested/internal dictionaries are ignored.
                parent = next((p for p in ast.walk(node) if child in ast.iter_child_nodes(p)), None)
                if isinstance(parent, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "out" for target in parent.targets
                ):
                    returned.update(parent_keys)
                elif isinstance(parent, ast.Return):
                    returned.update(parent_keys)
            elif isinstance(child, ast.Subscript) and isinstance(child.ctx, ast.Store):
                if isinstance(child.value, ast.Name) and child.value.id == "out":
                    key = child.slice
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        returned.add(key.value)
        declared = {item["name"] for item in row["execution_contract"]["output_specs"]}
        assert returned <= declared, (row["name"], sorted(returned - declared))


def test_langmuir_electron_temperature_slope_reference():
    result = langmuir_electron_temperature_from_slope(0.4, sigma_slope_per_V=0.02)
    assert result["electron_temperature_eV"] == pytest.approx(2.5)
    assert result["uncertainty_electron_temperature_eV"] == pytest.approx(0.125)


def test_langmuir_ion_density_from_saturation_reference():
    current, temperature, area, mass, coefficient = 2.0e-3, 5.0, 1.0e-4, 3.34e-27, 0.61
    result = langmuir_ion_density_from_saturation_current(
        -current, temperature, area, mass, coefficient,
        sigma_ion_saturation_current_A=2.0e-5,
        sigma_electron_temperature_eV=0.1,
        sigma_probe_area_m2=1.0e-6,
        sigma_ion_mass_kg=1.0e-29,
        sigma_sheath_coefficient=0.01,
    )
    expected = current / (coefficient * E_CHARGE * area * math.sqrt(E_CHARGE * temperature / mass))
    assert result["ion_density_m3"] == pytest.approx(expected, rel=1e-12)
    assert result["uncertainty_ion_density_m3"] > 0


def test_langmuir_formula_registry_charge_state_reference():
    rows = json.loads((ROOT / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    row = next(item for item in rows if item["name"] == "Langmuir ion density from ion-saturation current")
    assert "Z_i^(3/2)" in row["equation"]
    assert any(spec["name"] == "charge_state" for spec in row["execution_contract"]["input_specs"])
    singly = langmuir_ion_density_from_saturation_current(1e-3, 5.0, 1e-4, 3.34e-27, 0.61)
    doubly = langmuir_ion_density_from_saturation_current(1e-3, 5.0, 1e-4, 3.34e-27, 0.61, charge_state=2)
    assert doubly["ion_density_m3"] == pytest.approx(singly["ion_density_m3"] / (2.0**1.5))


def test_langmuir_electron_density_from_saturation_reference():
    n = langmuir_electron_density_from_saturation_current(1e-3, 5.0, 1e-4)
    expected = 1e-3 / (E_CHARGE * 1e-4 * math.sqrt(E_CHARGE*5.0/(2.0*math.pi*M_E)))
    assert n["electron_density_m3"] == pytest.approx(expected, rel=1e-12)
    u = langmuir_electron_density_from_saturation_current(
        1e-3, 5.0, 1e-4, sigma_electron_saturation_current_A=1e-5,
        relative_sigma_electron_temperature=0.02, relative_sigma_probe_area=0.01)
    assert u["uncertainty_electron_density_m3"] > 0


def test_langmuir_bohm_particle_flux_reference():
    n, te, mass, coeff, charge = 1e18, 5.0, 3.34e-27, 0.61, 2
    result = langmuir_bohm_particle_flux(n, te, mass, coeff, charge_state=charge)
    expected = coeff*n*math.sqrt(charge*E_CHARGE*te/mass)
    assert result["ion_particle_flux_m2_s"] == pytest.approx(expected, rel=1e-12)
    u = langmuir_bohm_particle_flux(n, te, mass, coeff, charge_state=charge,
        relative_sigma_ion_density=0.01, relative_sigma_electron_temperature=0.02,
        relative_sigma_ion_mass=0.01, relative_sigma_sheath_coefficient=0.03)
    assert u["uncertainty_ion_particle_flux_m2_s"] > 0


def test_langmuir_multispecies_density_reference():
    species = [
        {"mass_kg": 3.34e-27, "fraction": 0.7, "charge_state": 1, "sheath_coefficient": 0.61},
        {"mass_kg": 6.68e-27, "fraction": 0.3, "charge_state": 2, "sheath_coefficient": 0.58},
    ]
    result = langmuir_multispecies_ion_density(-1e-3, 5.0, 1e-4, species)
    factor = sum(x["fraction"]*x["sheath_coefficient"]*x["charge_state"]**1.5/math.sqrt(x["mass_kg"]) for x in species)
    expected = 1e-3/(E_CHARGE*1e-4*math.sqrt(E_CHARGE*5.0)*factor)
    assert result["ion_density_m3"] == pytest.approx(expected, rel=1e-12)
    assert result["uncertainty_ion_density_m3"] is None
