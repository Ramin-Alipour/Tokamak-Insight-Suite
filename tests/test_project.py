import json
import numpy as np
import pytest
from tis.project import AnalysisRun, Project, ProjectError, ProjectStore, stable_digest

def test_project_roundtrip_persists_reproducible_run(tmp_path):
    run = AnalysisRun(analysis_name="FFT power spectrum",
        inputs={"signal_ref": "signals/shot-12.csv"},
        settings={"window": "hann", "nperseg": 256}, algorithm_version="0.5.1",
        outputs={"frequency_Hz": np.array([0.0, 10.0]), "peak_Hz": 10.0},
        input_provenance={"shot_id": 12, "sha256": "source-digest"},
        units={"frequency_Hz": "Hz", "peak_Hz": "Hz"},
        warnings=("synthetic test input",), status="warning")
    project = Project(title="reproducibility test", shots=[{"shot_id": 12}])
    project.add_run(run)
    path = ProjectStore.save(project, tmp_path / "test.tis-project.json")
    restored = ProjectStore.load(path)
    assert restored.title == project.title
    assert restored.analysis_runs[0].run_id == run.run_id
    assert restored.analysis_runs[0].algorithm_version == "0.5.1"
    assert restored.analysis_runs[0].outputs["frequency_Hz"] == [0.0, 10.0]
    assert restored.analysis_runs[0].warnings == ["synthetic test input"]

def test_project_detects_tampering(tmp_path):
    path = ProjectStore.save(Project(title="integrity test"), tmp_path / "project.json")
    payload = json.loads(path.read_text())
    payload["title"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ProjectError, match="integrity"):
        ProjectStore.load(path)

def test_project_compares_runs_in_requested_order():
    project = Project(title="comparison")
    first = AnalysisRun("density", {}, {}, "1", outputs={"n": 1})
    second = AnalysisRun("temperature", {}, {}, "1", outputs={"T": 10})
    project.add_run(first)
    project.add_run(second)
    assert [x.analysis_name for x in project.compare_runs([second.run_id, first.run_id])] == ["temperature", "density"]
    with pytest.raises(ProjectError, match="Unknown analysis run"):
        project.compare_runs(["not-a-run"])


def test_project_serializes_nonfinite_numpy_values(tmp_path):
    project = Project(title="non-finite values", metadata={"scalar": np.float64(np.nan), "array": np.array([1.0, np.inf, -np.inf])})
    path = ProjectStore.save(project, tmp_path / "non-finite.tis-project.json")
    payload = json.loads(path.read_text())
    assert payload["metadata"]["scalar"] == {"__nonfinite_float__": "nan"}
    assert payload["metadata"]["array"] == [1.0, {"__nonfinite_float__": "inf"}, {"__nonfinite_float__": "-inf"}]
    restored = ProjectStore.load(path)
    assert restored.metadata == payload["metadata"]


def test_project_load_rejects_non_object_json_root(tmp_path):
    path = tmp_path / "invalid-project.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ProjectError, match="root must be a JSON object"):
        ProjectStore.load(path)


def test_project_load_rejects_duplicate_run_ids(tmp_path):
    first = AnalysisRun("first", {}, {}, "1")
    path = ProjectStore.save(Project(title="duplicates", analysis_runs=[first]), tmp_path / "duplicates.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    duplicate = dict(payload["analysis_runs"][0])
    duplicate["analysis_name"] = "second"
    payload["analysis_runs"].append(duplicate)
    payload.pop("content_sha256")
    payload["content_sha256"] = stable_digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProjectError, match="duplicate analysis run IDs"):
        ProjectStore.load(path)


def test_project_load_rejects_malformed_analysis_runs_schema(tmp_path):
    path = ProjectStore.save(Project(title="malformed runs"), tmp_path / "malformed-runs.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["analysis_runs"] = {"not": "an array"}
    payload.pop("content_sha256")
    payload["content_sha256"] = stable_digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProjectError, match="analysis_runs must be a JSON array"):
        ProjectStore.load(path)


def test_project_load_rejects_invalid_metadata_type(tmp_path):
    path = ProjectStore.save(Project(title="invalid metadata"), tmp_path / "invalid-metadata.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"] = []
    payload.pop("content_sha256")
    payload["content_sha256"] = stable_digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProjectError, match="metadata must be a JSON object"):
        ProjectStore.load(path)


def test_analysis_run_rejects_invalid_field_types_cleanly():
    with pytest.raises(ProjectError, match="analysis_name must be a nonempty string"):
        AnalysisRun(None, {}, {}, "0.5.1")
    with pytest.raises(ProjectError, match="inputs must be a mapping"):
        AnalysisRun("analysis", [], {}, "0.5.1")


def test_project_model_rejects_invalid_title_and_duplicate_runs():
    with pytest.raises(ProjectError, match="title must be a nonempty string"):
        Project(title="")
    run = AnalysisRun("same", {}, {}, "0.5.1")
    with pytest.raises(ProjectError, match="duplicate analysis run IDs"):
        Project(title="duplicate", analysis_runs=[run, run])
