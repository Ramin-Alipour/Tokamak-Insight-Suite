import numpy as np
from tis.fild import analyze_fild_image


def test_fild_image_energy_pitch_calibration():
    image = np.zeros((4, 5))
    image[2, 3] = 10
    result = analyze_fild_image(image, energy_calibration_coefficients=[2, 1], pitch_calibration_coefficients=[5, 10])
    assert result['intensity_weighted_mean_energy_keV'] == 7
    assert result['intensity_weighted_mean_pitch_angle_deg'] == 20
