from streamlit.testing.v1 import AppTest


def test_flux_loop_workflow_is_available_and_renders_without_data():
    app = AppTest.from_file("app.py", default_timeout=120).run()
    assert not app.exception

    app.radio(key="workspace_page").set_value("Advanced Analysis").run()
    assert not app.exception

    workflow = app.selectbox(label="Select workflow")
    workflow.set_value("Flux-loop magnetic-flux reconstruction").run()
    assert not app.exception
    assert any("calibrated flux-loop voltage" in item.value.lower() for item in app.info)
    assert any("initial flux" in item.value.lower() for item in app.warning)
