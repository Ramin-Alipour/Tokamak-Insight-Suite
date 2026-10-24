import json
import subprocess
import sys


def test_status_report_uses_diagnostic_catalog_and_separates_callable_entries(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tis.cli",
            "catalog",
            "--kind",
            "status",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "DIAGNOSTIC_STATUS_REPORT.json" in result.stdout

    report = json.loads((tmp_path / "DIAGNOSTIC_STATUS_REPORT.json").read_text(encoding="utf-8"))
    assert report["source_of_truth"] == "DIAGNOSTIC_REGISTRY.json"
    assert report["diagnostic_record_count"] == 80
    assert report["diagnostic_execution_level_counts"] == {
        "analysis_pipeline": 11,
        "limited_calculation": 17,
        "reference_only": 52,
    }
    assert report["diagnostic_completeness_counts"] == {
        "complete": 8,
        "complete_with_declared_scope": 2,
        "incomplete": 70,
    }
    assert report["callable_analysis_entry_point_count"] == 81
    assert sum(report["callable_analysis_completeness_counts"].values()) == 81

    first_ten = report["first_ten_diagnostics"]
    assert len(first_ten) == 10
    assert [row["catalog_position"] for row in first_ten] == list(range(1, 11))
    assert first_ten[0]["diagnostic"] == "Thomson scattering"
    assert first_ten[0]["completeness"] == "incomplete"
    assert first_ten[1]["diagnostic"] == "Langmuir probe"
    assert first_ten[1]["completeness"] == "complete"
    assert first_ten[9]["diagnostic"] == "Mirnov / pickup coils"
    assert first_ten[9]["completeness"] == "complete"
    assert report["interpretation"]["completeness_is_software_scope_not_independent_experimental_validation"] is True
