"""Interactive regression tests for the equilibrium reconstruction workspace."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_equilibrium_workspace_exposes_grad_shafranov_solver_without_exceptions():
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    app.radio(key="workspace_page").set_value("Equilibrium & Reconstruction").run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("Grad–Shafranov solver" in str(item.value) for item in app.markdown)
    assert any(item.label == "Solve Grad–Shafranov equilibrium" for item in app.button)
    expected_inputs = {
        "gs_R", "gs_Z", "gs_boundary", "gs_profile", "gs_pprime", "gs_ffprime"
    }
    actual_inputs = {item.key for item in app.get("file_uploader")}
    assert expected_inputs <= actual_inputs
    assert any("not a full EFIT replacement" in str(item.value) for item in app.warning)


def test_equilibrium_workspace_rejects_run_when_required_csv_inputs_are_missing():
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.radio(key="workspace_page").set_value("Equilibrium & Reconstruction").run()
    next(item for item in app.button if item.label == "Solve Grad–Shafranov equilibrium").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("Upload all six CSV inputs" in str(item.value) for item in app.error)


def test_equilibrium_workspace_runs_solver_and_renders_downloadable_solution():
    import io

    import numpy as np
    from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec

    def upload(name, values):
        buffer = io.StringIO()
        np.savetxt(buffer, np.asarray(values, dtype=float), delimiter=",")
        payload = buffer.getvalue().encode("utf-8")
        return UploadedFile(
            record=UploadedFileRec(file_id=name, name=name, type="text/csv", data=payload),
            file=io.BytesIO(payload),
        )

    R = np.linspace(1.0, 2.0, 5)
    Z = np.linspace(-0.5, 0.5, 5)
    boundary = np.broadcast_to(R[None, :] ** 2, (Z.size, R.size)).copy()
    files = {
        "gs_R": upload("R.csv", R),
        "gs_Z": upload("Z.csv", Z),
        "gs_boundary": upload("boundary.csv", boundary),
        "gs_profile": upload("profile.csv", [0.0, 1.0]),
        "gs_pprime": upload("pprime.csv", [0.0, 0.0]),
        "gs_ffprime": upload("ffprime.csv", [0.0, 0.0]),
    }
    app = AppTest.from_file(APP_PATH, default_timeout=90).run()
    app.radio(key="workspace_page").set_value("Equilibrium & Reconstruction").run()
    for key, value in files.items():
        next(item for item in app.get("file_uploader") if item.key == key).set_value(value)
    next(item for item in app.button if item.label == "Solve Grad–Shafranov equilibrium").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("Converged" in str(item.value) for item in app.metric)
    assert any("tis_grad_shafranov_equilibrium.csv" in str(item.value) for item in app.get("download_button"))
