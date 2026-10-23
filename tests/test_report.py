from tis.report import build_html_report


def test_report_escapes_figure_payload_as_html_attribute():
    malicious = "abc' onerror='alert(1)"
    report = build_html_report({}, sections=[{"title": "test", "figures": [malicious]}])
    assert "onerror='alert(1)" not in report
    assert "&#x27; onerror=&#x27;alert(1)" in report
