from tis.project import AnalysisRun, Project
from tis.ui.project_page import _history_item_key, _sync_analysis_history


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
