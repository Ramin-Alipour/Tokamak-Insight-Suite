from tis.report import build_html_report


def test_report_escapes_figure_payload_as_html_attribute():
    malicious = "abc' onerror='alert(1)"
    report = build_html_report({}, sections=[{"title": "test", "figures": [malicious]}])
    assert "onerror='alert(1)" not in report
    assert "&#x27; onerror=&#x27;alert(1)" in report

def test_report_includes_maturity_summary_and_validation_limit():
    report = build_html_report({
        "diagnostic_maturity_counts": {"Reference only": 52},
        "analysis_maturity_counts": {"Analysis pipeline — incomplete": 70},
        "analysis_maturity_registry": [
            {"id": 1, "name": "Thomson Scattering", "maturity_level": "Analysis pipeline — incomplete"}
        ],
        "scientific_validation_status": "Not independently validated",
    })
    assert "Implementation maturity" in report
    assert "Diagnostic categories" in report
    assert "Callable analysis entry points" in report
    assert "not independent experimental validation" in report
    assert "Thomson Scattering" in report
    assert "Analysis pipeline — incomplete" in report
