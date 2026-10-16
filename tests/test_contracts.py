import json
from pathlib import Path
import pytest
from tis.contracts import ContractError, ExecutionContract, ExecutionLevel, QuantitySpec, get_diagnostic_contract, get_diagnostic_contracts

def test_all_eighty_entries_have_explicit_conservative_contracts():
    contracts = get_diagnostic_contracts()
    assert len(contracts) == 80
    assert len({item.name for item in contracts}) == 80
    assert sum(item.completeness == "complete" for item in contracts) == 1
    assert all(item.completeness in {"incomplete", "complete"} for item in contracts)
    assert sum(item.level is ExecutionLevel.REFERENCE_ONLY for item in contracts) == 59
    assert sum(item.level is ExecutionLevel.LIMITED_CALCULATION for item in contracts) == 20
    assert sum(item.level is ExecutionLevel.ANALYSIS_PIPELINE for item in contracts) == 1
    assert not any(item.supports_raw_data for item in contracts)

def test_thomson_catalogue_entry_does_not_claim_raw_data_to_physics():
    contract = get_diagnostic_contract("Thomson scattering")
    assert contract.level is ExecutionLevel.ANALYSIS_PIPELINE
    assert contract.completeness == "complete"
    assert not contract.raw_data_to_physics
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
