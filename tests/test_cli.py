import json
import sys

from tis.cli import main


def test_cli_qc_exports_missing_counts_as_json_integers(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    source.write_text("time,signal\\n0,1\\n1,\\n2,3\\n", encoding="utf-8")
    output = tmp_path / "reports" / "qc.json"
    monkeypatch.setattr(sys, "argv", ["tis", "qc", str(source), "--output", str(output)])
    assert main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["missing_by_column"]["signal"] == 1
    assert isinstance(report["missing_by_column"]["signal"], int)
