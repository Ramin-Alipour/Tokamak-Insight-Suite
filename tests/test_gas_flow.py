from tis.gas_flow import analyze_gas_flow_meter


def test_gas_flow_sccm_conversion():
    r = analyze_gas_flow_meter([1.0], calibration_sccm_per_unit=10.0)
    assert r['flow_sccm'][0] == 10.0
    assert r['gas_throughput_Pa_m3_s'][0] > 0
    assert r['particle_rate_s-1'][0] > 0
