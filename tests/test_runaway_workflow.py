import numpy as np
from tis.runaway import analyze_runaway_electron_signatures


def test_hxr_event_detection():
    t = np.linspace(0, 1, 101)
    h = np.ones(101)*10
    h[40:46] = 100
    r = analyze_runaway_electron_signatures(t, h, threshold_sigma=3)
    assert len(r['candidate_events']) == 1
