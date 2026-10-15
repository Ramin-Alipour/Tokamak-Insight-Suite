from tis.neutron_yield import analyze_neutron_yield_monitor


def test_neutron_monitor_yield():
    r = analyze_neutron_yield_monitor([10, 20, 30], bin_width_s=0.5, dead_time_s=0.001, neutrons_per_count=100)
    assert r['neutron_yield'] > 0
    assert r['dead_time_corrected_count_rate_s-1'][0] > r['observed_count_rate_s-1'][0]
