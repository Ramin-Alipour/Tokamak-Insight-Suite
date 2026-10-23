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


def test_cli_scalings_creates_output_parent_directory(tmp_path, monkeypatch):
    output = tmp_path / "reports" / "scalings.json"
    monkeypatch.setattr(sys, "argv", [
        "tis", "scalings", "--ip-A", "1000000", "--bt-T", "2", "--r-m", "1.5",
        "--a-m", "0.5", "--kappa", "1.7", "--ne-line-avg-m3", "1e19",
        "--p-loss-W", "1000000", "--output", str(output),
    ])
    assert main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))
