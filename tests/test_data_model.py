import pytest
from tis.data_model import Measurement, MeasurementError

def test_measurement_preserves_metadata_uncertainty_and_provenance():
    m = Measurement(values=[1.0, 2.0], quantity="electron temperature", unit="eV",
        shot_id=123, time_s=[0.0, 0.1], uncertainty=[0.1, 0.2],
        provenance={"source": "test.csv", "sha256": "abc"}, quality_flags=("calibrated",))
    assert m.shot_id == 123
    assert m.provenance["source"] == "test.csv"
    assert m.as_dict()["values"] == [1.0, 2.0]

def test_unit_conversion_is_explicit_and_does_not_mutate_original():
    m = Measurement(values=[1000.0], quantity="length", unit="mm")
    converted = m.to("m")
    assert converted.unit == "m"
    assert converted.values[0] == pytest.approx(1.0)
    assert m.unit == "mm"

@pytest.mark.parametrize("time", [[0.0, 0.0], [0.0, float("nan")], [0.0]])
def test_invalid_timebase_is_rejected(time):
    with pytest.raises(MeasurementError):
        Measurement(values=[1.0, 2.0], quantity="signal", unit="V", time_s=time)

def test_invalid_status_requires_reason():
    with pytest.raises(MeasurementError, match="invalid_reason"):
        Measurement(values=[1.0], quantity="signal", unit="V", status="invalid")

def test_negative_uncertainty_is_rejected():
    with pytest.raises(MeasurementError, match="non-negative"):
        Measurement(values=[1.0], quantity="signal", unit="V", uncertainty=[-0.1])


def test_temperature_uncertainty_conversion_does_not_apply_affine_offset():
    measurement = Measurement(
        values=[20.0], quantity="temperature", unit="degC", uncertainty=[2.0]
    )
    converted = measurement.to("kelvin")
    assert converted.values[0] == pytest.approx(293.15)
    assert converted.uncertainty[0] == pytest.approx(2.0)


def test_measurement_rejects_non_numeric_values_and_metadata():
    with pytest.raises(MeasurementError, match="values must be numeric"):
        Measurement(values=["not a number"], quantity="signal", unit="V")
    with pytest.raises(MeasurementError, match="provenance must be a mapping"):
        Measurement(values=[1.0], quantity="signal", unit="V", provenance=["bad"])


def test_measurement_rejects_sampling_rate_inconsistent_with_timebase():
    with pytest.raises(MeasurementError, match="conflicts with time_s"):
        Measurement(
            values=[1.0, 2.0, 3.0], quantity="signal", unit="V",
            time_s=[0.0, 0.1, 0.2], sample_rate_hz=50.0,
        )


def test_measurement_preserves_nonfinite_samples_for_explicit_quality_handling():
    measurement = Measurement(
        values=[1.0, float("nan"), 3.0], quantity="signal", unit="V",
        quality_flags=("contains_gap",),
    )
    assert measurement.quality_flags == ("contains_gap",)
    assert measurement.values[1] != measurement.values[1]
