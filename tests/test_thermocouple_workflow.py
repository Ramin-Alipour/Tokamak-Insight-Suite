from tis.thermocouple import analyze_thermocouple


def test_thermocouple_temperature_conversion():
    r = analyze_thermocouple([0, 1, 2], temperature_calibration_coefficients=[100, 0], time_s=[0, 1, 2])
    assert list(r['temperature_C']) == [0, 100, 200]
