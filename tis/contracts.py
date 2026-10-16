"""Typed execution contracts for the diagnostic catalogue and its consumers."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

class ExecutionLevel(str, Enum):
    REFERENCE_ONLY = "reference_only"
    LIMITED_CALCULATION = "limited_calculation"
    ANALYSIS_PIPELINE = "analysis_pipeline"
    INVERSE_RECONSTRUCTION = "inverse_reconstruction"

class ContractError(ValueError):
    """An analysis request violates a declared contract."""

@dataclass(frozen=True)
class QuantitySpec:
    name: str
    unit: str
    description: str = ""
    required: bool = True
    accepted_units: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

@dataclass(frozen=True)
class ExecutionContract:
    diagnostic_id: str
    name: str
    level: ExecutionLevel
    input_specs: tuple[QuantitySpec, ...] = ()
    output_specs: tuple[QuantitySpec, ...] = ()
    assumptions: tuple[str, ...] = ()
    preprocessing: tuple[str, ...] = ()
    algorithm: str = ""
    uncertainty_model: str = "not_specified"
    limitations: tuple[str, ...] = ()
    function_links: tuple[str, ...] = ()
    raw_data_to_physics: bool = False
    completeness: str = "incomplete"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def runnable(self) -> bool:
        return self.level is not ExecutionLevel.REFERENCE_ONLY

    @property
    def supports_raw_data(self) -> bool:
        return self.raw_data_to_physics and self.completeness == "complete"

    def validate_inputs(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping):
            raise ContractError("inputs must be a mapping keyed by declared input name")
        declared = {item.name: item for item in self.input_specs}
        unknown = set(values) - set(declared)
        if unknown:
            raise ContractError(f"Undeclared input(s): {', '.join(sorted(unknown))}")
        missing = [item.name for item in self.input_specs if item.required and item.name not in values]
        if missing:
            raise ContractError(f"Missing required input(s): {', '.join(missing)}")
        for name, value in values.items():
            spec = declared[name]
            if spec.minimum is None and spec.maximum is None:
                continue
            try:
                if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
                    numbers = [float(value)]
                else:
                    numbers = [float(v) for v in value]
            except (TypeError, ValueError) as exc:
                raise ContractError(f"Input '{name}' must be numeric") from exc
            if any((spec.minimum is not None and v < spec.minimum) or
                   (spec.maximum is not None and v > spec.maximum) for v in numbers):
                raise ContractError(f"Input '{name}' is outside its declared bounds")

_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "DIAGNOSTIC_REGISTRY.json"

@lru_cache(maxsize=1)
def load_diagnostic_registry() -> tuple[dict[str, Any], ...]:
    with _REGISTRY_PATH.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ContractError("DIAGNOSTIC_REGISTRY.json must contain a JSON list")
    return tuple(rows)

def contract_from_registry(row: Mapping[str, Any]) -> ExecutionContract:
    raw = row.get("execution_contract") or {}
    try:
        level = ExecutionLevel(raw.get("execution_level", "reference_only"))
    except ValueError as exc:
        raise ContractError(f"Unknown execution level for {row.get('diagnostic', '<unnamed>')}") from exc
    def specs(key: str) -> tuple[QuantitySpec, ...]:
        return tuple(QuantitySpec(
            name=str(item["name"]), unit=str(item.get("unit", "unspecified")),
            description=str(item.get("description", "")), required=bool(item.get("required", True)),
            accepted_units=tuple(item.get("accepted_units", ())),
            minimum=item.get("minimum"), maximum=item.get("maximum"))
            for item in raw.get(key, []))
    return ExecutionContract(
        diagnostic_id=str(row.get("id", row.get("diagnostic", "unknown"))),
        name=str(row.get("diagnostic", row.get("name", "Unnamed diagnostic"))),
        level=level, input_specs=specs("input_specs"), output_specs=specs("output_specs"),
        assumptions=tuple(raw.get("assumptions", ())),
        preprocessing=tuple(raw.get("preprocessing", ())),
        algorithm=str(raw.get("algorithm", row.get("method", ""))),
        uncertainty_model=str(raw.get("uncertainty_model", "not_specified")),
        limitations=tuple(raw.get("limitations", ())),
        function_links=tuple(raw.get("function_links", row.get("implementation_functions", ()))),
        raw_data_to_physics=bool(raw.get("raw_data_to_physics", False)),
        completeness=str(raw.get("completeness", "incomplete")),
        metadata={"family": row.get("family"), "quantities": row.get("quantities"),
                  "catalogue_status": row.get("implementation_status")})

@lru_cache(maxsize=1)
def get_diagnostic_contracts() -> tuple[ExecutionContract, ...]:
    return tuple(contract_from_registry(row) for row in load_diagnostic_registry())

def get_diagnostic_contract(name_or_id: str | int) -> ExecutionContract:
    needle = str(name_or_id).strip().casefold()
    for contract in get_diagnostic_contracts():
        if contract.diagnostic_id.casefold() == needle or contract.name.casefold() == needle:
            return contract
    raise KeyError(f"No diagnostic contract matches {name_or_id!r}")

def clear_registry_cache() -> None:
    load_diagnostic_registry.cache_clear()
    get_diagnostic_contracts.cache_clear()
