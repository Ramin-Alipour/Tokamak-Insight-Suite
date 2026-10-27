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
    assert any("public DIII-D EFIT flux map" in str(item.value) for item in app.caption)
    assert any(item.key == "eq_inverse_bundle" for item in app.get("file_uploader"))
    assert any(item.label == "Fit equilibrium to magnetic measurements" for item in app.button)
    assert any("machine-specific JSON bundle" in str(item.value) for item in app.caption)


def test_equilibrium_workspace_rejects_run_when_required_csv_inputs_are_missing():
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.radio(key="workspace_page").set_value("Equilibrium & Reconstruction").run()
    next(item for item in app.button if item.label == "Solve Grad–Shafranov equilibrium").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("Upload all six CSV inputs" in str(item.value) for item in app.error)


def test_equilibrium_workspace_runs_solver_and_renders_downloadable_solution():
    import io

    import numpy as np

    def upload(name, values):
        buffer = io.StringIO()
        np.savetxt(buffer, np.asarray(values, dtype=float), delimiter=",")
        payload = buffer.getvalue().encode("utf-8")
        return (name, payload, "text/csv")

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
    assert any(item.label == "Download Grad–Shafranov solution grid" for item in app.get("download_button"))



def test_inverse_dashboard_requires_machine_calibrated_bundle():
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.radio(key="workspace_page").set_value("Equilibrium & Reconstruction").run()
    next(item for item in app.button if item.label == "Fit equilibrium to magnetic measurements").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("machine-specific calibrated JSON bundle" in str(item.value) for item in app.error)



def test_inverse_dashboard_runs_with_explicit_synthetic_acceptance_fixture():
    import io
    import json

    import numpy as np

    R = np.linspace(1.0, 2.0, 9)
    Z = np.linspace(-0.5, 0.5, 9)
    rr, zz = np.meshgrid(R, Z)
    basis = np.stack((-(rr - 1.5) ** 2 - zz**2,
                      0.2 * (rr - 1.5) ** 2 + 0.1 * zz**2))
    coeff = np.array([2.0, 0.5])
    probe_response = np.array([[1.0, 0.2], [0.1, 1.2], [0.7, -0.4]])
    probe_coil_response = np.array([[0.01], [0.02], [-0.01]])
    probe_currents = np.array([10.0])
    loop_response = np.array([[0.4, 0.3], [-0.2, 0.8]])
    loop_coil_response = np.array([[0.002], [0.004]])
    bundle = {
        "R_m": R.tolist(), "Z_m": Z.tolist(), "basis_psi_Wb_rad": basis.tolist(),
        "magnetic_probe_response_T_per_coefficient": probe_response.tolist(),
        "magnetic_probe_values_T": (probe_response @ coeff + probe_coil_response @ probe_currents).tolist(),
        "magnetic_probe_sigma_T": [0.01, 0.02, 0.01],
        "coil_probe_response_T_per_A": probe_coil_response.tolist(),
        "coil_probe_currents_A": probe_currents.tolist(),
        "flux_loop_response_Wb_rad_per_coefficient": loop_response.tolist(),
        "flux_loop_values_Wb_rad": (loop_response @ coeff + loop_coil_response @ probe_currents).tolist(),
        "flux_loop_sigma_Wb_rad": [0.001, 0.002],
        "coil_flux_response_Wb_rad_per_A": loop_coil_response.tolist(),
        "coil_flux_currents_A": probe_currents.tolist(),
    }
    payload = json.dumps(bundle).encode("utf-8")
    app = AppTest.from_file(APP_PATH, default_timeout=90).run()
    app.radio(key="workspace_page").set_value("Equilibrium & Reconstruction").run()
    next(item for item in app.get("file_uploader") if item.key == "eq_inverse_bundle").set_value(
        ("synthetic_acceptance_fixture.json", payload, "application/json")
    )
    next(item for item in app.button if item.label == "Fit equilibrium to magnetic measurements").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any(item.label == "Download inverse equilibrium grid" for item in app.get("download_button"))
    assert any("Magnetic measurement count" in str(item.label) for item in app.metric)
