import json
from pathlib import Path
import pytest
from tis.contracts import ContractError, ExecutionContract, ExecutionLevel, QuantitySpec, get_diagnostic_contract, get_diagnostic_contracts


def test_all_eighty_entries_have_explicit_conservative_contracts():
    contracts = get_diagnostic_contracts()
    assert len(contracts) == 80
    assert len({item.name for item in contracts}) == 80
    assert sum(item.completeness == "complete" for item in contracts) == 7
    assert sum(item.completeness == "complete_with_declared_scope" for item in contracts) == 9
    assert sum(item.completeness == "incomplete" for item in contracts) == 64
    assert all(item.completeness in {"incomplete", "complete", "complete_with_declared_scope"} for item in contracts)
    assert sum(item.level is ExecutionLevel.REFERENCE_ONLY for item in contracts) == 49
    assert sum(item.level is ExecutionLevel.LIMITED_CALCULATION for item in contracts) == 16
    assert sum(item.level is ExecutionLevel.ANALYSIS_PIPELINE for item in contracts) == 15
    assert any(item.supports_raw_data for item in contracts)
    assert all(
        not item.supports_raw_data
        or (item.level is ExecutionLevel.ANALYSIS_PIPELINE and item.completeness == "complete")
        for item in contracts
    )


def test_thomson_contract_marks_scoped_raw_spectrum_analysis_explicitly():
    contract = get_diagnostic_contract("Thomson scattering")
    assert contract.level is ExecutionLevel.ANALYSIS_PIPELINE
    assert contract.completeness == "complete_with_declared_scope"
    assert contract.raw_data_to_physics
    assert not contract.supports_raw_data


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
        assert len(rows) == 80
        assert rows[0]["diagnostic"] == "Thomson scattering"
    finally:
        contracts.clear_registry_cache()
