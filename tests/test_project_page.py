from tis.project import AnalysisRun, Project
from tis.ui.project_page import _sync_analysis_history


def test_project_export_syncs_new_history_into_existing_project_without_duplicates():
    project = Project(
        title="existing",
        analysis_runs=[AnalysisRun("older", {}, {}, "0.5.1", inputs={"history_key": "older-key"})],
    )
    history = [
        {
            "name": "new analysis",
            "created_at_utc": "2026-10-23T10:00:00+00:00",
            "parameter_snapshot": {"window": "hann"},
            "result_summary": {"peak_Hz": 50.0},
            "source_filename": "shot.csv",
            "source_sha256": "abc",
            "shot_id": "42",
        }
    ]
    _sync_analysis_history(project, history, "0.5.1")
    _sync_analysis_history(project, history, "0.5.1")
    assert [run.analysis_name for run in project.analysis_runs] == ["older", "new analysis"]
    assert project.analysis_runs[1].inputs["history_key"]
