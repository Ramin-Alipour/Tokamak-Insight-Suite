import numpy as np
from tis.sxr import analyze_soft_xray_array


def test_soft_xray_array_reconstruction_shape():
    A = np.eye(3)
    y = np.array([[1., 2., 3.], [2., 3., 4.]])
    result = analyze_soft_xray_array([0., 1.], y, A, regularization=0)
    assert result['emissivity_reconstruction'].shape == (2, 3)
    assert result['peak_cell_by_time'].tolist() == [2, 2]
