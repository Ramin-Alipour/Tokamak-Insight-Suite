"""Canonical, provenance-aware measurement objects for multi-diagnostic workflows."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
import numpy as np
import pint

_UNIT_REGISTRY = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

class MeasurementError(ValueError):
    """Invalid measurement metadata, shape or unit conversion."""

@dataclass
class Measurement:
    values: Any
    quantity: str
    unit: str
    shot_id: str | int | None = None
    time_s: Any | None = None
    location_m: Any | None = None
    coordinate_system: str | None = None
    sample_rate_hz: float | None = None
    calibration: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: Any | None = None
    quality_flags: tuple[str, ...] = ()
    status: str = "valid"
    invalid_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.quantity.strip():
            raise MeasurementError("quantity must be a non-empty name")
        if not self.unit.strip():
            raise MeasurementError("unit must be explicit; use 'dimensionless' if applicable")
        try:
            _UNIT_REGISTRY.Unit(self.unit)
        except (pint.errors.UndefinedUnitError, TypeError, ValueError) as exc:
            raise MeasurementError(f"Unrecognized unit {self.unit!r}") from exc
        self.values = np.asarray(self.values)
        if self.values.ndim == 0:
            self.values = self.values.reshape(1)
        if self.time_s is not None:
            self.time_s = np.asarray(self.time_s, dtype=float)
            if self.time_s.ndim != 1 or self.time_s.size != self.values.shape[0]:
                raise MeasurementError("time_s must match the first dimension of values")
            if not np.all(np.isfinite(self.time_s)):
                raise MeasurementError("time_s contains NaN or infinite values")
            if self.time_s.size > 1 and np.any(np.diff(self.time_s) <= 0):
                raise MeasurementError("time_s must be strictly increasing")
        if self.uncertainty is not None:
            self.uncertainty = np.asarray(self.uncertainty, dtype=float)
            if self.uncertainty.shape not in (self.values.shape, ()):
                raise MeasurementError("uncertainty must be scalar or match values shape")
            if not np.all(np.isfinite(self.uncertainty)) or np.any(self.uncertainty < 0):
                raise MeasurementError("uncertainty must be finite and non-negative")
        if self.sample_rate_hz is not None:
            if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
                raise MeasurementError("sample_rate_hz must be finite and positive")
        if self.status != "valid" and not self.invalid_reason:
            raise MeasurementError("non-valid measurements must state invalid_reason")

    def to(self, target_unit: str) -> "Measurement":
        """Return a copy converted explicitly to target_unit."""
        try:
            values = _UNIT_REGISTRY.Quantity(self.values, self.unit).to(target_unit).magnitude
            uncertainty = None
            if self.uncertainty is not None:
                uncertainty = _UNIT_REGISTRY.Quantity(self.uncertainty, self.unit).to(target_unit).magnitude
        except (pint.errors.PintError, TypeError, ValueError) as exc:
            raise MeasurementError(f"Cannot convert {self.unit!r} to {target_unit!r}") from exc
        return Measurement(
            values=values, quantity=self.quantity, unit=target_unit, shot_id=self.shot_id,
            time_s=self.time_s, location_m=self.location_m, coordinate_system=self.coordinate_system,
            sample_rate_hz=self.sample_rate_hz, calibration=dict(self.calibration),
            provenance=dict(self.provenance), uncertainty=uncertainty,
            quality_flags=self.quality_flags, status=self.status, invalid_reason=self.invalid_reason,
            metadata=dict(self.metadata), created_at=self.created_at)

    def as_dict(self) -> dict[str, Any]:
        def plain(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, Mapping):
                return {str(k): plain(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(v) for v in value]
            return value
        return {key: plain(value) for key, value in vars(self).items()}
