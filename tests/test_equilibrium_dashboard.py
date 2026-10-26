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
