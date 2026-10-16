import numpy as np
from tis.channel_sync import analyze_channel_synchronization


def test_channel_synchronization_finds_relative_delay():
    t = np.arange(256)/1000
    a = np.sin(2*np.pi*40*t)
    b = np.r_[np.zeros(3), a[:-3]]
    r = analyze_channel_synchronization(t, np.column_stack([a, b]), max_lag_s=0.01)
    assert r['relative_delay_s'].shape == (2,)
    assert r['relative_delay_s'][0] == 0
