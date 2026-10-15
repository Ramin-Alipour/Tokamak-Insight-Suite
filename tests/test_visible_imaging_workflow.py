import numpy as np
from tis.visible_imaging import analyze_visible_imaging


def test_image_centroid_speed():
    frames = np.zeros((3, 10, 10))
    frames[0, 4, 2] = 10
    frames[1, 4, 3] = 10
    frames[2, 4, 4] = 10
    result = analyze_visible_imaging(frames, [0, 1, 2], pixel_size_m=0.01)
    np.testing.assert_allclose(result['centroid_x_pixel'], [2, 3, 4])
    np.testing.assert_allclose(result['apparent_centroid_speed_m_s'], [0.01, 0.01, 0.01])
