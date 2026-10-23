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

    def __post_init__(self) -> None:
        if not self.analysis_name.strip():
            raise ProjectError("analysis_name must not be empty")
        if not self.algorithm_version.strip():
            raise ProjectError("algorithm_version must be recorded")
        if self.status not in {"completed", "warning", "failed", "invalid_input"}:
            raise ProjectError(f"Unsupported analysis run status: {self.status}")

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

    def add_run(self, run: AnalysisRun) -> None:
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
        if payload.get("schema_version") != PROJECT_SCHEMA_VERSION:
            raise ProjectError(f"Unsupported project schema version: {payload.get('schema_version')!r}")
        expected = payload.pop("content_sha256", None)
        if not expected or stable_digest(payload) != expected:
            raise ProjectError("Project integrity check failed (SHA-256 mismatch)")
        runs = [AnalysisRun(**item) for item in payload.pop("analysis_runs", [])]
        return Project(analysis_runs=runs, **payload)
