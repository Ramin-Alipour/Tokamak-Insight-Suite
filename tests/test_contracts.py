import json
from pathlib import Path
import pytest
from tis.contracts import ContractError, ExecutionContract, ExecutionLevel, QuantitySpec, get_diagnostic_contract, get_diagnostic_contracts


def test_all_eighty_entries_have_explicit_conservative_contracts():
    contracts = get_diagnostic_contracts()
    assert len(contracts) == 81
    assert len({item.name for item in contracts}) == 81
    assert sum(item.completeness == "complete" for item in contracts) == 6
    assert sum(item.completeness == "complete_with_declared_scope" for item in contracts) == 12
    assert sum(item.completeness == "incomplete" for item in contracts) == 62
    assert all(item.completeness in {"incomplete", "complete", "complete_with_declared_scope"} for item in contracts)
    assert sum(item.level is ExecutionLevel.REFERENCE_ONLY for item in contracts) == 47
    assert sum(item.level is ExecutionLevel.LIMITED_CALCULATION for item in contracts) == 16
    assert sum(item.level is ExecutionLevel.ANALYSIS_PIPELINE for item in contracts) == 17
    assert sum(item.level is ExecutionLevel.PHYSICAL_SOLVER for item in contracts) == 1
    assert any(item.supports_raw_data for item in contracts)
    assert all(
        not item.supports_raw_data
        or (item.level is ExecutionLevel.ANALYSIS_PIPELINE and item.completeness == "complete")
        for item in contracts
    )


def test_thomson_contract_is_limited_to_calibrated_profiles():
    contract = get_diagnostic_contract("Thomson scattering")
    assert contract.level is ExecutionLevel.ANALYSIS_PIPELINE
    assert contract.completeness == "complete_with_declared_scope"
    assert not contract.raw_data_to_physics
    assert not contract.supports_raw_data
    assert contract.calibrated_measurements_to_physics
    assert contract.supports_calibrated_measurements


def test_input_validation_checks_required_names_unknown_names_and_bounds():
    contract = ExecutionContract("test", "test", ExecutionLevel.LIMITED_CALCULATION,
        input_specs=(QuantitySpec("density", "1 / meter ** 3", minimum=0),))
    contract.validate_inputs({"density": 1e19})
    with pytest.raises(ContractError, match="Missing required"):
        contract.validate_inputs({})
    with pytest.raises(ContractError, match="Undeclared input"):
        contract.validate_inputs({"density": 1e19, "temperature": 10})
    with pytest.raises(ContractError, match="bounds"):
        contract.validate_inputs({"density": -1})


def test_catalogue_contract_fields_are_machine_readable():
    path = Path(__file__).resolve().parents[1] / "DIAGNOSTIC_REGISTRY.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert all("execution_contract" in row for row in rows)
    assert all("execution_level" in row["execution_contract"] for row in rows)


def test_registry_loader_falls_back_to_packaged_data_when_root_copy_is_absent(tmp_path, monkeypatch):
    import tis.contracts as contracts

    monkeypatch.setattr(contracts, "_ROOT_REGISTRY_PATH", tmp_path / "missing" / "DIAGNOSTIC_REGISTRY.json")
    monkeypatch.setattr(
        contracts,
        "_PACKAGE_REGISTRY_PATH",
        Path(__file__).resolve().parents[1] / "tis" / "data" / "DIAGNOSTIC_REGISTRY.json",
    )
    contracts.clear_registry_cache()
    try:
        rows = contracts.load_diagnostic_registry()
        assert len(rows) == 81
        assert rows[0]["diagnostic"] == "Thomson scattering"
    finally:
        contracts.clear_registry_cache()


def test_root_and_packaged_diagnostic_and_formula_registries_are_synchronized():
    root = Path(__file__).resolve().parents[1]
    assert (root / "DIAGNOSTIC_REGISTRY.json").read_bytes() == (
        root / "tis" / "data" / "DIAGNOSTIC_REGISTRY.json"
    ).read_bytes()
    assert (root / "FORMULA_REGISTRY.json").read_bytes() == (
        root / "tis" / "data" / "FORMULA_REGISTRY.json"
    ).read_bytes()


def test_collective_thomson_contract_declares_calibrated_measurements():
    contract = get_diagnostic_contract("Collective Thomson / coherent scattering (CTS)")
    assert contract.supports_calibrated_measurements
    assert not contract.supports_raw_data
    assert "electron_velocity_projection_m_s" in {spec.name for spec in contract.input_specs}


def test_all_thomson_and_cts_contracts_declare_parameter_identifiability_output():
    path = Path(__file__).resolve().parents[1] / "DIAGNOSTIC_REGISTRY.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    names = {
        "Thomson scattering",
        "Collective Thomson / coherent scattering (CTS)",
        "Thomson spatial profile reconstruction",
        "Thomson spectral distribution",
    }
    selected = [row for row in rows if row["diagnostic"] in names]
    assert len(selected) == len(names)
    for row in selected:
        outputs = {item["name"]: item for item in row["execution_contract"]["output_specs"]}
        assert "parameter_identifiability" in outputs
        assert "primary/conditional/non-identifiable" in outputs["parameter_identifiability"]["unit"]


def test_thomson_spectral_distribution_contract_excludes_separate_evdf_inverse():
    path = Path(__file__).resolve().parents[1] / "DIAGNOSTIC_REGISTRY.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    entry = next(row for row in rows if row["diagnostic"] == "Thomson spectral distribution")
    contract = entry["execution_contract"]
    input_names = [item["name"] for item in contract["input_specs"]]
    output_names = {item["name"] for item in contract["output_specs"]}
    assert len(input_names) == len(set(input_names))
    assert set(input_names) == {
        "wavelength_nm", "signal", "instrument_response_matrix",
        "background_signal", "background_mask", "background_fraction",
        "signal_uncertainty", "signal_covariance", "moment_mc_samples",
    }
    assert not output_names.intersection({
        "velocity_parallel_m_s", "evdf_parallel_s_per_m",
        "regularization_strength", "parametric_model_comparison",
    })
    assert contract["calibrated_measurements_to_physics"] is True
    assert contract["function_links"] == [
        "tokamak_analysis.thomson_distribution.analyze_thomson_spectral_distribution"
    ]
    primary = rows[0]
    assert "electron_velocity_distribution" in {
        item["name"] for item in primary["execution_contract"]["output_specs"]
    }
    assert "tokamak_analysis.thomson_evdf.reconstruct_thomson_evdf" in (
        primary["execution_contract"]["function_links"]
    )
    primary_contract = primary["execution_contract"]
    evdf_input = next(
        item for item in primary_contract["input_specs"]
        if item["name"] == "enable_evdf_reconstruction"
    )
    evdf_output = next(
        item for item in primary_contract["output_specs"]
        if item["name"] == "electron_velocity_distribution"
    )
    assert "only for the non-collective Maxwellian and Kappa" in evdf_input["description"]
    assert "explicitly unavailable" in evdf_output["description"]


def test_all_thomson_and_cts_entry_points_are_subanalyses_of_primary_analysis_1():
    path = Path(__file__).resolve().parents[1] / "DIAGNOSTIC_REGISTRY.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    names = {
        "Thomson scattering",
        "Collective Thomson / coherent scattering (CTS)",
        "Thomson spatial profile reconstruction",
        "Thomson spectral distribution",
    }
    selected = {row["diagnostic"]: row for row in rows if row["diagnostic"] in names}
    assert set(selected) == names
    assert all(row["primary_analysis_id"] == 1 for row in selected.values())
    families_path = Path(__file__).resolve().parents[1] / "ANALYSIS_FAMILIES.json"
    families = json.loads(families_path.read_text(encoding="utf-8"))["families"]
    analysis_1 = next(item for item in families if item["id"] == 1)
    assert analysis_1["entry_ids"] == [1, 30, 46, 81]


def test_physical_solver_execution_level_is_runnable_and_calibrated():
    contract = get_diagnostic_contract("General relativistic 3-D optical Thomson EVDF reconstruction")
    assert contract.level is ExecutionLevel.PHYSICAL_SOLVER
    assert contract.runnable
    assert contract.supports_calibrated_measurements


def test_spectral_distribution_registry_matches_public_physical_geometry_arguments():
    import inspect
    from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution

    contract = get_diagnostic_contract("Thomson spectral distribution")
    declared = {item.name for item in contract.input_specs}
    public = set(inspect.signature(analyze_thomson_spectral_distribution).parameters)
    assert {"laser_wavelength_nm", "scattering_angle_rad"} <= declared
    assert {"laser_wavelength_nm", "scattering_angle_rad"} <= public
    assert {"physical_initial", "incident_polarization", "include_compton_recoil"} <= declared
    assert {"physical_initial", "incident_polarization", "include_compton_recoil"} <= public
