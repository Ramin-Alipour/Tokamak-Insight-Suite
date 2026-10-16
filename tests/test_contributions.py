import json

import pytest

from tis.contributions import build_contribution_bundle, bundle_json


def test_bundle_has_provenance_and_does_not_embed_raw_data():
    bundle = build_contribution_bundle(
        analysis_id=2,
        analysis_name="Langmuir probe",
        software_version="0.5.1",
        parameters={"voltage": [0.0, 1.0], "current": {"column": "I"}},
        result={"te_eV": 12.0, "bad": float("nan")},
        dataset_sha256="a" * 64,
        dataset_name="shot.csv",
        tokamak="Example Tokamak",
        shot_id=123,
        diagnostic="Langmuir probe",
        reference_result={"te_eV": 11.8},
        reference_description="Published value from the same shot",
    )
    assert bundle["schema"] == "tokamak-insight-contribution/1.0"
    assert bundle["dataset"]["raw_data_embedded"] is False
    assert bundle["dataset"]["sha256"] == "a" * 64
    assert bundle["software_result"]["bad"] is None
    assert bundle["reference"]["result"]["te_eV"] == 11.8
    assert "raw_data" not in bundle
    parsed = json.loads(bundle_json(bundle))
    assert parsed["analysis"]["id"] == 2
    assert len(parsed["bundle_sha256"]) == 64


@pytest.mark.parametrize("kwargs", [
    {"analysis_id": 0, "analysis_name": "x", "software_version": "0.5.1", "parameters": {}, "result": {}},
    {"analysis_id": 1, "analysis_name": "", "software_version": "0.5.1", "parameters": {}, "result": {}},
    {"analysis_id": 1, "analysis_name": "x", "software_version": "0.5.1", "parameters": {}, "result": {}, "dataset_sha256": "bad"},
])
def test_invalid_bundle_metadata_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build_contribution_bundle(**kwargs)
