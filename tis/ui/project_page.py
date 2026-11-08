"""Persistent project manager page, independent from the Streamlit app entry point."""
from __future__ import annotations
from pathlib import Path
import tempfile
from typing import Any
import pandas as pd
from tis.project import AnalysisRun, Project, ProjectStore, stable_digest


def _ensure_project_shot(active_project: Project, shot_id: str | None) -> None:
    """Keep each selected discharge represented in the project's shot list."""
    if shot_id and not any(str(shot.get("shot_id", "")) == shot_id for shot in active_project.shots):
        active_project.shots.append({"shot_id": shot_id})


def _update_project_metadata(active_project: Project, *, software_version: str,
                             shot_id: str, analyst: str, filename: str,
                             source_sha256: str, source_manifest: dict[str, Any] | None,
                             import_parameters: dict[str, Any] | None) -> None:
    """Update available workspace metadata without erasing saved provenance."""
    metadata_update: dict[str, Any] = {
        "software": "Tokamak Insight Suite",
        "software_version": software_version,
        "project_export_note": "Raw source data is referenced by provenance and is not embedded.",
    }
    if shot_id:
        metadata_update["current_shot_id"] = shot_id
    if analyst:
        metadata_update["analyst"] = analyst
    if filename:
        metadata_update["source_filename"] = filename
    if source_sha256:
        metadata_update["source_sha256"] = source_sha256
    if source_manifest is not None:
        metadata_update["source_manifest"] = source_manifest
    if import_parameters is not None:
        metadata_update["import_parameters"] = import_parameters
    active_project.metadata.update(metadata_update)


def _history_item_key(item: dict[str, Any]) -> str:
    return stable_digest({
        key: item.get(key) for key in (
            "name", "created_at_utc", "result_sha256", "parameter_snapshot",
            "result_summary", "source_filename", "source_sha256", "shot_id",
            "source_manifest", "import_parameters"
        )
    })


def _export_context_signature(project_title: str, shot_id: str, analyst: str,
                              filename: str, source_sha256: str,
                              history: list[dict[str, Any]]) -> str:
    return stable_digest({
        "project_title": project_title,
        "shot_id": shot_id,
        "analyst": analyst,
        "source_filename": filename,
        "source_sha256": source_sha256,
        "history_keys": [_history_item_key(item) for item in history],
    })


def _sync_analysis_history(active_project: Project, history: list[dict[str, Any]],
                           software_version: str,
                           excluded_history_keys: set[str] | None = None) -> None:
    """Append new session history entries without importing unrelated pre-load history."""
    excluded = excluded_history_keys or set()
    existing_keys = {
        run.inputs.get("history_key")
        for run in active_project.analysis_runs
        if isinstance(run.inputs, dict) and run.inputs.get("history_key")
    }
    for item in reversed(history):
        history_key = _history_item_key(item)
        if history_key in existing_keys or history_key in excluded:
            continue
        active_project.add_run(AnalysisRun(
            analysis_name=str(item.get("name", "Unnamed UI result")),
            inputs={
                "history_key": history_key,
                "source_filename": item.get("source_filename"),
                "shot_id": item.get("shot_id")
            },
            settings=item.get("parameter_snapshot", {}),
            algorithm_version=str(item.get("algorithm_version", software_version)),
            outputs=item.get("result_summary", {}),
            input_provenance={
                "source_filename": item.get("source_filename"),
                "source_sha256": item.get("source_sha256"),
                "shot_id": item.get("shot_id"),
                "source_manifest": item.get("source_manifest"),
                "import_parameters": item.get("import_parameters")
            },
            warnings=tuple(item.get("warnings", ())) + (
                "Raw input data is referenced, not embedded in the project.",
            ),
            status="warning"
        ))
        existing_keys.add(history_key)


def render_project_page(st: Any, *, session_state: Any, project_title: str, shot_id: str,
                        analyst: str, filename: str, source_sha256: str, software_version: str) -> None:
    pending_title = session_state.pop("tis_pending_project_title", None)
    if pending_title is not None:
        session_state["tis_project_title"] = pending_title
    active_project = session_state.get("tis_active_project")
    default_project_title = active_project.title if active_project is not None else project_title
    project_title = st.text_input("Project title", value=default_project_title, key="tis_project_title")
    current_history = list(session_state.get("tis_analysis_history", []))
    current_signature = _export_context_signature(
        project_title, shot_id, analyst, filename, source_sha256, current_history
    )
    if session_state.get("tis_project_export_bytes") and session_state.get("tis_project_export_signature") != current_signature:
        session_state.pop("tis_project_export_bytes", None)
        session_state.pop("tis_project_export_signature", None)
    st.title("Research Project Manager")
    st.caption("Save and reopen project metadata and analysis history independently of browser/session state.")
    uploaded_project = st.file_uploader("Open TIS project JSON", type=["json"], key="tis_project_upload")
    if st.button("Load project", key="tis_load_project", disabled=uploaded_project is None):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                source_path = Path(temp_dir) / "uploaded-project.json"
                source_path.write_bytes(uploaded_project.getvalue())
                loaded_project = ProjectStore.load(source_path)
            session_state["tis_active_project"] = loaded_project
            session_state["tis_pending_project_title"] = loaded_project.title
            session_state["tis_project_history_excluded_keys"] = [
                _history_item_key(item) for item in session_state.get("tis_analysis_history", [])
            ]
            session_state.pop("tis_project_export_bytes", None)
            session_state.pop("tis_project_export_signature", None)
            st.success(f"Loaded project: {loaded_project.title} ({len(loaded_project.analysis_runs)} saved analysis runs)")
        except Exception as exc:
            st.error(f"Project could not be loaded: {exc}")

    active_project = session_state.get("tis_active_project")
    if active_project is not None:
        st.markdown("### Open project")
        st.write({"title": active_project.title, "project_id": active_project.project_id,
                  "saved_shots": len(active_project.shots),
                  "saved_analysis_runs": len(active_project.analysis_runs),
                  "updated_at": active_project.updated_at})
        if active_project.analysis_runs:
            st.dataframe(pd.DataFrame([{
                "analysis": run.analysis_name, "status": run.status,
                "algorithm_version": run.algorithm_version, "created_at": run.created_at,
                "warnings": "; ".join(run.warnings)
            } for run in active_project.analysis_runs]), width="stretch")

    st.markdown("### Save current workspace")
    st.caption("The project records source references and SHA-256 metadata; large raw files are not embedded.")
    if st.button("Prepare project export", key="tis_prepare_project_export"):
        try:
            active_project = session_state.get("tis_active_project")
            if active_project is None:
                active_project = Project(title=project_title, shots=([{"shot_id": shot_id}] if shot_id else []))
                session_state.pop("tis_project_history_excluded_keys", None)
            history = list(session_state.get("tis_analysis_history", []))
            excluded = set(session_state.get("tis_project_history_excluded_keys", []))
            _sync_analysis_history(active_project, history, software_version, excluded)
            if project_title and project_title != "Untitled tokamak analysis":
                active_project.title = project_title
            _ensure_project_shot(active_project, shot_id)
            _update_project_metadata(
                active_project, software_version=software_version, shot_id=shot_id,
                analyst=analyst, filename=filename, source_sha256=source_sha256,
                source_manifest=session_state.get("tis_source_manifest"),
                import_parameters=session_state.get("tis_import_parameters"),
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                target = Path(temp_dir) / "tokamak-insight-project.tis-project.json"
                ProjectStore.save(active_project, target)
                project_bytes = target.read_bytes()
            session_state["tis_active_project"] = active_project
            session_state["tis_project_export_bytes"] = project_bytes
            session_state["tis_project_export_signature"] = _export_context_signature(
                project_title, shot_id, analyst, filename, source_sha256,
                list(session_state.get("tis_analysis_history", []))
            )
            st.success(f"Project prepared with {len(active_project.analysis_runs)} analysis run(s).")
        except Exception as exc:
            st.error(f"Project export failed: {exc}")

    project_bytes = session_state.get("tis_project_export_bytes")
    if project_bytes:
        st.download_button("Download project JSON", data=project_bytes,
            file_name="tokamak-insight-project.tis-project.json", mime="application/json",
            key="tis_download_project_json")
    st.info("A project preserves settings, result summaries, algorithm version, recorded units, warnings and provenance. Raw data must remain available at its recorded source path for exact reruns.")
