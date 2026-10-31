import json
from pathlib import Path
import pytest
from tis.contracts import ContractError, ExecutionContract, ExecutionLevel, QuantitySpec, get_diagnostic_contract, get_diagnostic_contracts


def test_all_eighty_entries_have_explicit_conservative_contracts():
    contracts = get_diagnostic_contracts()
    assert len(contracts) == 69
    assert len({item.name for item in contracts}) == 79
    assert sum(item.completeness == "complete" for item in contracts) == 7
    assert sum(item.completeness == "complete_with_declared_scope" for item in contracts) == 11
    assert sum(item.completeness == "incomplete" for item in contracts) == 62
    assert all(item.completeness in {"incomplete", "complete", "complete_with_declared_scope"} for item in contracts)
    assert sum(item.level is ExecutionLevel.REFERENCE_ONLY for item in contracts) == 47
    assert sum(item.level is ExecutionLevel.LIMITED_CALCULATION for item in contracts) == 16
    assert sum(item.level is ExecutionLevel.ANALYSIS_PIPELINE for item in contracts) == 16
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
        assert len(rows) == 79
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
