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
    """A measurement plus explicit units, quality state and acquisition provenance.

    Non-finite sample values are retained so that raw-data gaps can be represented;
    callers must mark/interpret them through quality_flags or status.
    Time coordinates themselves must always be finite and strictly increasing.
    """

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
        if not isinstance(self.quantity, str) or not self.quantity.strip():
            raise MeasurementError("quantity must be a non-empty name")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise MeasurementError("unit must be explicit; use 'dimensionless' if applicable")
        try:
            _UNIT_REGISTRY.Unit(self.unit)
        except (pint.errors.PintError, TypeError, ValueError) as exc:
            raise MeasurementError(f"Unrecognized unit {self.unit!r}") from exc

        for name in ("calibration", "provenance", "metadata"):
            if not isinstance(getattr(self, name), Mapping):
                raise MeasurementError(f"{name} must be a mapping")
        if not isinstance(self.quality_flags, (tuple, list)) or any(
            not isinstance(flag, str) or not flag.strip() for flag in self.quality_flags
        ):
            raise MeasurementError("quality_flags must be a sequence of non-empty strings")
        if not isinstance(self.status, str) or self.status not in {
            "valid", "warning", "invalid", "missing", "unvalidated"
        }:
            raise MeasurementError(f"Unsupported measurement status: {self.status!r}")
        if self.status != "valid" and (
            not isinstance(self.invalid_reason, str) or not self.invalid_reason.strip()
        ):
            raise MeasurementError("non-valid measurements must state invalid_reason")

        if self.sample_rate_hz is not None:
            try:
                rate = float(self.sample_rate_hz)
            except (TypeError, ValueError) as exc:
                raise MeasurementError("sample_rate_hz must be finite and positive") from exc
            if isinstance(self.sample_rate_hz, (bool, np.bool_)) or not np.isfinite(rate) or rate <= 0:
                raise MeasurementError("sample_rate_hz must be finite and positive")
            self.sample_rate_hz = rate

        try:
            self.values = np.asarray(self.values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise MeasurementError("values must be numeric") from exc
        if self.values.ndim == 0:
            self.values = self.values.reshape(1)
        if self.values.ndim < 1 or self.values.shape[0] < 1:
            raise MeasurementError("values must contain at least one sample")

        if self.time_s is not None:
            try:
                self.time_s = np.asarray(self.time_s, dtype=float)
            except (TypeError, ValueError) as exc:
                raise MeasurementError("time_s must be numeric") from exc
            if self.time_s.ndim != 1 or self.time_s.size != self.values.shape[0]:
                raise MeasurementError("time_s must match the first dimension of values")
            if not np.all(np.isfinite(self.time_s)):
                raise MeasurementError("time_s contains NaN or infinite values")
            if self.time_s.size > 1 and np.any(np.diff(self.time_s) <= 0):
                raise MeasurementError("time_s must be strictly increasing")
            if self.sample_rate_hz is not None and self.time_s.size > 1:
                intervals = np.diff(self.time_s)
                median_rate = 1.0 / float(np.median(intervals))
                if not np.isclose(self.sample_rate_hz, median_rate, rtol=0.02, atol=0.0):
                    raise MeasurementError(
                        "sample_rate_hz conflicts with time_s median sampling interval (>2%); "
                        "omit it for irregularly sampled data"
                    )

        if self.location_m is not None:
            try:
                self.location_m = np.asarray(self.location_m, dtype=float)
            except (TypeError, ValueError) as exc:
                raise MeasurementError("location_m must be numeric") from exc
            if not np.all(np.isfinite(self.location_m)):
                raise MeasurementError("location_m must contain only finite coordinates")
            if self.location_m.ndim > 2:
                raise MeasurementError("location_m must be a coordinate vector or 2-D coordinate array")
        if self.coordinate_system is not None:
            if not isinstance(self.coordinate_system, str) or not self.coordinate_system.strip():
                raise MeasurementError("coordinate_system must be a non-empty string when supplied")

        if self.uncertainty is not None:
            try:
                self.uncertainty = np.asarray(self.uncertainty, dtype=float)
            except (TypeError, ValueError) as exc:
                raise MeasurementError("uncertainty must be numeric") from exc
            if self.uncertainty.shape not in (self.values.shape, ()):
                raise MeasurementError("uncertainty must be scalar or match values shape")
            if not np.all(np.isfinite(self.uncertainty)) or np.any(self.uncertainty < 0):
                raise MeasurementError("uncertainty must be finite and non-negative")

    def to(self, target_unit: str) -> "Measurement":
        """Return a copy converted explicitly to target_unit.

        Standard uncertainties are converted by the absolute unit scale, not by
        applying affine offsets (important for Celsius/Fahrenheit temperatures).
        """
        try:
            source_unit = _UNIT_REGISTRY.Unit(self.unit)
            destination_unit = _UNIT_REGISTRY.Unit(target_unit)
            values = _UNIT_REGISTRY.Quantity(self.values, source_unit).to(destination_unit).magnitude
            uncertainty = None
            if self.uncertainty is not None:
                zero = _UNIT_REGISTRY.Quantity(0.0, source_unit).to(destination_unit).magnitude
                one = _UNIT_REGISTRY.Quantity(1.0, source_unit).to(destination_unit).magnitude
                scale = abs(float(one) - float(zero))
                uncertainty = np.asarray(self.uncertainty) * scale
        except (pint.errors.PintError, TypeError, ValueError) as exc:
            raise MeasurementError(f"Cannot convert {self.unit!r} to {target_unit!r}") from exc
        return Measurement(
            values=values, quantity=self.quantity, unit=target_unit, shot_id=self.shot_id,
            time_s=self.time_s, location_m=self.location_m, coordinate_system=self.coordinate_system,
            sample_rate_hz=self.sample_rate_hz, calibration=dict(self.calibration),
            provenance=dict(self.provenance), uncertainty=uncertainty,
            quality_flags=tuple(self.quality_flags), status=self.status, invalid_reason=self.invalid_reason,
            metadata=dict(self.metadata), created_at=self.created_at)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot; array values become nested lists."""
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
