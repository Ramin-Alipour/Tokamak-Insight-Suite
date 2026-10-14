from streamlit.testing.v1 import AppTest


def test_dashboard_renders_home_without_streamlit_exceptions():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("Tokamak Insight Suite" in str(item.value) for item in app.markdown)
