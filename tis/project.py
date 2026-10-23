"""Portable project and reproducible analysis-run persistence (JSON schema v1)."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4
import numpy as np

PROJECT_SCHEMA_VERSION = "1.0"

class ProjectError(ValueError):
    """Invalid project file or analysis record."""

def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"__nonfinite_float__": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported project value type: {type(value).__name__}")

def stable_digest(value: Any) -> str:
    canonical = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

@dataclass
class AnalysisRun:
    analysis_name: str
    inputs: Mapping[str, Any]
    settings: Mapping[str, Any]
    algorithm_version: str
    outputs: Mapping[str, Any] = field(default_factory=dict)
    input_provenance: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    status: str = "completed"
    run_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    software_version: str = "unknown"
    environment: Mapping[str, str] = field(default_factory=dict)
    input_hashes: Mapping[str, str] = field(default_factory=dict)
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.analysis_name, str) or not self.analysis_name.strip():
            raise ProjectError("analysis_name must be a nonempty string")
        if not isinstance(self.algorithm_version, str) or not self.algorithm_version.strip():
            raise ProjectError("algorithm_version must be a nonempty string")
        for field_name in (
            "inputs", "settings", "outputs", "input_provenance", "units",
            "environment", "input_hashes",
        ):
            if not isinstance(getattr(self, field_name), Mapping):
                raise ProjectError(f"{field_name} must be a mapping")
        if not isinstance(self.software_version, str) or not self.software_version.strip():
            raise ProjectError("software_version must be a nonempty string")
        if any(not isinstance(key, str) or not isinstance(value, str) or not value.strip()
               for key, value in self.environment.items()):
            raise ProjectError("environment must map string names to nonempty string values")
        if any(not isinstance(key, str) or not isinstance(value, str)
               or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value)
               for key, value in self.input_hashes.items()):
            raise ProjectError("input_hashes must map input names to 64-character SHA-256 hex digests")
        if isinstance(self.random_seed, bool) or (
            self.random_seed is not None and
            (not isinstance(self.random_seed, int) or self.random_seed < 0)
        ):
            raise ProjectError("random_seed must be a non-negative integer or None")
        if not isinstance(self.warnings, (list, tuple)) or any(not isinstance(item, str) for item in self.warnings):
            raise ProjectError("warnings must be a sequence of strings")
        if not isinstance(self.status, str) or self.status not in {"completed", "warning", "failed", "invalid_input"}:
            raise ProjectError(f"Unsupported analysis run status: {self.status}")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ProjectError("run_id must be a nonempty string")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ProjectError("created_at must be a nonempty string")

    @property
    def reproducibility_fingerprint(self) -> str:
        """Stable fingerprint of inputs and execution configuration (not the output)."""
        return stable_digest({
            "analysis_name": self.analysis_name,
            "inputs": self.inputs,
            "settings": self.settings,
            "algorithm_version": self.algorithm_version,
            "software_version": self.software_version,
            "environment": self.environment,
            "input_hashes": self.input_hashes,
            "units": self.units,
            "random_seed": self.random_seed,
        })

    @property
    def output_digest(self) -> str:
        """SHA-256 digest of normalized outputs for regression comparison."""
        return stable_digest(self.outputs)

@dataclass
class Project:
    title: str
    project_id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: str = PROJECT_SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    shots: list[dict[str, Any]] = field(default_factory=list)
    analysis_runs: list[AnalysisRun] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title.strip():
            raise ProjectError("Project title must be a nonempty string.")
        if not isinstance(self.project_id, str) or not self.project_id.strip():
            raise ProjectError("project_id must be a nonempty string.")
        if self.schema_version != PROJECT_SCHEMA_VERSION:
            raise ProjectError(f"Unsupported project schema version: {self.schema_version!r}")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ProjectError("created_at must be a nonempty string.")
        if not isinstance(self.updated_at, str) or not self.updated_at.strip():
            raise ProjectError("updated_at must be a nonempty string.")
        if not isinstance(self.shots, list) or any(not isinstance(shot, dict) for shot in self.shots):
            raise ProjectError("shots must be a list of mappings.")
        if not isinstance(self.analysis_runs, list) or any(not isinstance(run, AnalysisRun) for run in self.analysis_runs):
            raise ProjectError("analysis_runs must be a list of AnalysisRun objects.")
        run_ids = [run.run_id for run in self.analysis_runs]
        if len(run_ids) != len(set(run_ids)):
            raise ProjectError("Project contains duplicate analysis run IDs.")
        if not isinstance(self.metadata, dict):
            raise ProjectError("metadata must be a dictionary.")

    def add_run(self, run: AnalysisRun) -> None:
        if not isinstance(run, AnalysisRun):
            raise ProjectError("run must be an AnalysisRun instance.")
        if any(existing.run_id == run.run_id for existing in self.analysis_runs):
            raise ProjectError(f"Duplicate analysis run id: {run.run_id}")
        self.analysis_runs.append(run)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def compare_runs(self, run_ids: list[str]) -> list[AnalysisRun]:
        by_id = {run.run_id: run for run in self.analysis_runs}
        missing = [run_id for run_id in run_ids if run_id not in by_id]
        if missing:
            raise ProjectError(f"Unknown analysis run id(s): {', '.join(missing)}")
        return [by_id[run_id] for run_id in run_ids]

class ProjectStore:
    """Save and load portable JSON projects using atomic replacement."""

    @staticmethod
    def save(project: Project, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = _jsonable(project)
        payload["schema_version"] = PROJECT_SCHEMA_VERSION
        payload["content_sha256"] = stable_digest({k: v for k, v in payload.items()
                                                   if k != "content_sha256"})
        data = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                             delete=False) as stream:
                temp_name = stream.name
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, target)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
        return target

    @staticmethod
    def load(path: str | Path) -> Project:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectError(f"Cannot read project file {source}") from exc
        if not isinstance(payload, dict):
            raise ProjectError("Project file root must be a JSON object.")
        schema_version = payload.get("schema_version")
        if isinstance(schema_version, bool) or schema_version != PROJECT_SCHEMA_VERSION:
            raise ProjectError(f"Unsupported project schema version: {schema_version!r}")
        if not isinstance(payload.get("title"), str) or not payload["title"].strip():
            raise ProjectError("Project title must be a nonempty string.")
        shots = payload.get("shots", [])
        if not isinstance(shots, list) or any(not isinstance(shot, dict) for shot in shots):
            raise ProjectError("Project shots must be a JSON array of objects.")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ProjectError("Project metadata must be a JSON object.")
        expected = payload.pop("content_sha256", None)
        if not expected or stable_digest(payload) != expected:
            raise ProjectError("Project integrity check failed (SHA-256 mismatch)")
        raw_runs = payload.pop("analysis_runs", [])
        if not isinstance(raw_runs, list) or any(not isinstance(item, dict) for item in raw_runs):
            raise ProjectError("Project analysis_runs must be a JSON array of objects.")
        try:
            runs = [AnalysisRun(**item) for item in raw_runs]
        except (TypeError, ValueError) as exc:
            raise ProjectError("Project contains an invalid analysis run record.") from exc
        run_ids = [run.run_id for run in runs]
        if len(run_ids) != len(set(run_ids)):
            raise ProjectError("Project contains duplicate analysis run IDs.")
        return Project(analysis_runs=runs, **payload)
