import numpy as np
import pytest
from tokamak_analysis.thomson_spectral_distribution import analyze_thomson_spectral_distribution

def test_symmetric_distribution_recovers_centroid_and_width():
    x=np.linspace(500,504,401); y=10*np.exp(-0.5*((x-502)/0.4)**2)+2
    o=analyze_thomson_spectral_distribution(x,y,background=2,signal_uncertainty=np.full(x.size,.1))
    assert o["centroid_nm"]==pytest.approx(502,abs=1e-3)
    assert o["rms_width_nm"]==pytest.approx(.4,rel=.01)
    assert abs(o["skewness"])<1e-2
    assert o["integrated_positive_area_uncertainty"]>0
    assert o["centroid_uncertainty_nm"]>0

def test_negative_bins_are_preserved():
    x=np.linspace(0,4,9); y=np.array([0,-1,1,4,8,4,1,-1,0],float)
    o=analyze_thomson_spectral_distribution(x,y)
    assert o["negative_net_bin_count"]==2 and np.any(o["net_signal"]<0)
    assert np.all(o["positive_net_signal"]>=0)

@pytest.mark.parametrize("x,y", [([1,2,3,4,5],[1,2,np.nan,2,1]),([1,2,3,3,5],[1,2,3,2,1]),([1,2,3,4],[1,2,3,4])])
def test_invalid_spectra_rejected(x,y):
    with pytest.raises(ValueError): analyze_thomson_spectral_distribution(x,y)

def test_invalid_uncertainty_and_insufficient_positive_bins_rejected():
    x=np.linspace(1,5,9); y=np.array([0,0,0,0,5,0,0,0,0],float)
    with pytest.raises(ValueError,match="positive finite"):
        analyze_thomson_spectral_distribution(x,y,signal_uncertainty=np.zeros_like(y))
    with pytest.raises(ValueError,match="positive net-signal bins"):
        analyze_thomson_spectral_distribution(x,y)
