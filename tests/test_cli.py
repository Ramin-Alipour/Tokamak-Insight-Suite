import json
import sys

import pytest

from tis.cli import main


def test_cli_qc_exports_missing_counts_as_json_integers(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    source.write_text("time,signal\n0,1\n1,\n2,3\n", encoding="utf-8")
    output = tmp_path / "reports" / "qc.json"
    monkeypatch.setattr(sys, "argv", ["tis", "qc", str(source), "--output", str(output)])
    assert main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["missing_by_column"]["signal"] == 1
    assert isinstance(report["missing_by_column"]["signal"], int)


def test_cli_scalings_creates_output_parent_directory(tmp_path, monkeypatch):
    output = tmp_path / "reports" / "scalings.json"
    monkeypatch.setattr(sys, "argv", [
        "tis", "scalings", "--ip-A", "1000000", "--bt-T", "2", "--r-m", "1.5",
        "--a-m", "0.5", "--kappa", "1.7", "--ne-line-avg-m3", "1e19",
        "--p-loss-W", "1000000", "--output", str(output),
    ])
    assert main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))


def test_cli_fft_requires_exactly_one_sampling_source(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    monkeypatch.setattr(sys, "argv", ["tis", "fft", str(source), "--signal", "signal"])
    with pytest.raises(SystemExit):
        main()
    monkeypatch.setattr(sys, "argv", [
        "tis", "fft", str(source), "--signal", "signal", "--time", "time", "--fs", "1000"
    ])
    with pytest.raises(SystemExit):
        main()


def test_cli_catalog_exports_analysis_maturity(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "tis", "catalog", "--kind", "analyses", "--output-dir", str(tmp_path)
    ])
    assert main() == 0
    registry = json.loads((tmp_path / "ANALYSIS_REGISTRY.json").read_text(encoding="utf-8"))
    assert len(registry) == 80
    assert all(row.get("maturity_level") for row in registry)
    assert registry[0]["execution_contract"]["completeness"] == "complete_with_declared_scope"
    assert registry[0]["maturity_level"] == "Complete within declared scope — not independently validated"
    assert any(row["maturity_level"] == "Complete within declared scope — not independently validated" for row in registry)


def test_cli_catalog_status_reports_canonical_completeness_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "tis", "catalog", "--kind", "status", "--output-dir", str(tmp_path)
    ])
    assert main() == 0
    report = json.loads(
        (tmp_path / "DIAGNOSTIC_STATUS_REPORT.json").read_text(encoding="utf-8")
    )
    assert report["diagnostic_record_count"] == 80
    assert report["callable_analysis_entry_point_count"] == 80
    assert report["diagnostic_completeness_counts"] == {
        "complete": 7,
        "complete_with_declared_scope": 9,
        "incomplete": 64,
    }
    first_ten = report["first_ten_diagnostics"]
    assert len(first_ten) == 10
    assert [row["catalog_position"] for row in first_ten] == list(range(1, 11))
    assert [row["completeness"] for row in first_ten] == [
        "complete_with_declared_scope", "complete_with_declared_scope",
        "complete_with_declared_scope", "complete_with_declared_scope",
        "complete", "complete", "complete", "complete", "complete", "complete",
    ]
    assert all(row["maturity_level"] for row in first_ten)
