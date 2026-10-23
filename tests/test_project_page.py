from tis.project import AnalysisRun, Project
from tis.ui.project_page import _export_context_signature, _history_item_key, _sync_analysis_history, _update_project_metadata


def test_project_export_syncs_new_history_into_existing_project_without_duplicates():
    project = Project(
        title="existing",
        analysis_runs=[AnalysisRun("older", {"history_key": "older-key"}, {}, "0.5.1")],
    )
    # Session history is stored newest-first by the Streamlit download callback.
    history = [
        {
            "name": "another analysis",
            "created_at_utc": "2026-10-23T10:01:00+00:00",
            "result_sha256": "same-result-digest",
            "parameter_snapshot": {"window": "boxcar"},
            "result_summary": {"peak_Hz": 50.0},
            "source_filename": "shot.csv",
            "source_sha256": "abc",
            "shot_id": "42",
        },
        {
            "name": "new analysis",
            "created_at_utc": "2026-10-23T10:00:00+00:00",
            "result_sha256": "same-result-digest",
            "parameter_snapshot": {"window": "hann"},
            "result_summary": {"peak_Hz": 50.0},
            "source_filename": "shot.csv",
            "source_sha256": "abc",
            "shot_id": "42",
        },
    ]
    _sync_analysis_history(project, history, "0.5.1")
    _sync_analysis_history(project, history, "0.5.1")
    assert [run.analysis_name for run in project.analysis_runs] == ["older", "new analysis", "another analysis"]
    assert project.analysis_runs[1].inputs["history_key"]
    assert project.analysis_runs[2].inputs["history_key"]


def test_project_export_excludes_history_that_predates_loading_a_project():
    project = Project(title="loaded project")
    old_item = {
        "name": "previous project analysis",
        "created_at_utc": "2026-10-23T09:00:00+00:00",
        "result_summary": {"value": 1},
    }
    new_item = {
        "name": "analysis after loading",
        "created_at_utc": "2026-10-23T11:00:00+00:00",
        "result_summary": {"value": 2},
    }
    _sync_analysis_history(
        project,
        [new_item, old_item],
        "0.5.1",
        excluded_history_keys={_history_item_key(old_item)},
    )
    assert [run.analysis_name for run in project.analysis_runs] == ["analysis after loading"]


def test_existing_project_records_new_shot_ids_without_duplicates():
    from tis.ui.project_page import _ensure_project_shot
    project = Project(title="multi-shot", shots=[{"shot_id": "100"}])
    _ensure_project_shot(project, "200")
    _ensure_project_shot(project, "200")
    assert project.shots == [{"shot_id": "100"}, {"shot_id": "200"}]


def test_project_export_signature_changes_when_context_changes():
    history = [{"name": "analysis", "created_at_utc": "2026-10-23T10:00:00+00:00"}]
    base = _export_context_signature("project", "42", "analyst", "shot.csv", "abc", history)
    assert _export_context_signature("renamed", "42", "analyst", "shot.csv", "abc", history) != base
    assert _export_context_signature("project", "43", "analyst", "shot.csv", "abc", history) != base
    assert _export_context_signature("project", "42", "analyst", "shot.csv", "abc", history + [{"name": "new"}]) != base


def test_project_metadata_update_preserves_saved_provenance_when_workspace_is_empty():
    project = Project(title="saved", metadata={
        "source_filename": "old-shot.csv",
        "source_sha256": "old-digest",
        "source_manifest": {"source_url": "https://example.org/old.csv"},
        "import_parameters": {"delimiter": ","},
        "analyst": "previous analyst",
    })
    _update_project_metadata(
        project, software_version="0.5.1", shot_id="", analyst="", filename="",
        source_sha256="", source_manifest=None, import_parameters=None,
    )
    assert project.metadata["source_filename"] == "old-shot.csv"
    assert project.metadata["source_sha256"] == "old-digest"
    assert project.metadata["source_manifest"]["source_url"] == "https://example.org/old.csv"
    assert project.metadata["import_parameters"] == {"delimiter": ","}
    assert project.metadata["analyst"] == "previous analyst"
