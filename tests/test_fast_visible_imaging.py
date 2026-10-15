import numpy as np
from tis.visible_imaging import analyze_fast_visible_imaging


def test_fast_visible_imaging_frame_difference_outputs():
    frames = np.zeros((4, 8, 8))
    frames[1, 2:4, 2:4] = 10
    frames[2, 2:4, 3:5] = 10
    frames[3, 2:4, 4:6] = 10
    result = analyze_fast_visible_imaging(frames, [0, 0.001, 0.002, 0.003], pixel_size_m=0.001)
    assert result['frame_difference_rms'].shape == (3,)
    assert result['frame_difference_max_abs'].max() > 0
