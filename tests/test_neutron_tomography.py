import numpy as np
from tis.neutron_tomography import analyze_neutron_tomography


def test_neutron_tomography_image_shape():
    A = np.eye(4)
    r = analyze_neutron_tomography([1, 2, 3, 4], A, (2, 2), pixel_area_m2=0.01, regularization=0)
    assert r['reconstructed_emission_image'].shape == (2, 2)
    assert r['integrated_emission_proxy'] == 0.1
