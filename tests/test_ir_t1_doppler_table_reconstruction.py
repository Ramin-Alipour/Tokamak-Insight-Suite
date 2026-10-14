"""Independent reconstruction of the IR-T1 paper's Doppler-temperature table.

Source: Alipour et al., AIP Advances 7, 115303 (2017), Table VI and VII,
DOI 10.1063/1.4990869. Widths are the paper's Doppler FWHM in nm, not Gaussian
standard deviations. This test uses the published rounded values and independently
converts FWHM to sigma before calling the software's sigma-based routine.
"""
import math

import pytest

from tis.core import doppler_temperature_eV


# Table VII masses (amu), wavelengths (nm), Doppler FWHM from Table VI (nm),
# and published ion temperatures (eV). These are figure/article reference values,
# not original spectrometer samples.
PUBLISHED_LINES = [
    ("N III", 14.0, 409.731, 0.04955, 34.44),
    ("O V", 16.0, 412.390, 0.07361, 85.75),
    ("Cr I", 52.0, 425.434, 0.00968, 4.53),
    ("C II", 12.0, 426.700, 0.05047, 28.24),
    ("Mo I", 96.0, 427.726, 0.00714, 4.50),
    ("O II", 16.0, 441.490, 0.04858, 32.59),
    ("Ar II", 40.0, 443.018, 0.02665, 24.35),
    ("N II", 14.0, 444.703, 0.05024, 30.06),
    ("Ni I", 58.7, 446.264, 0.01144, 6.48),
    ("Ti I", 47.9, 453.323, 0.00966, 3.66),
    ("C III", 12.0, 464.742, 0.06775, 42.90),
    ("Mn I", 55.0, 476.237, 0.01211, 5.98),
    ("O IV", 16.0, 479.825, 0.07594, 67.41),
    ("W I", 183.8, 484.382, 0.00578, 4.41),
    ("Fe I", 55.8, 537.149, 0.01164, 4.40),
]

FWHM_TO_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


@pytest.mark.parametrize(
    "species,mass_amu,wavelength_nm,doppler_fwhm_nm,published_temperature_eV",
    PUBLISHED_LINES,
)
def test_reconstruct_published_ir_t1_doppler_temperature_table(
    species, mass_amu, wavelength_nm, doppler_fwhm_nm, published_temperature_eV
):
    """Reproduce each Table VII value from Table VI's corrected Doppler FWHM."""
    sigma_lambda_nm = doppler_fwhm_nm / FWHM_TO_SIGMA

    # Independent physics check using SI constants and the 1-D Maxwellian Gaussian.
    elementary_charge = 1.602176634e-19
    atomic_mass_unit = 1.66053906660e-27
    speed_of_light = 299792458.0
    independent_temperature_eV = (
        mass_amu * atomic_mass_unit * speed_of_light**2 / elementary_charge
    ) * (sigma_lambda_nm / wavelength_nm) ** 2

    software_temperature_eV = doppler_temperature_eV(
        sigma_lambda_nm, wavelength_nm, mass_amu
    )

    assert software_temperature_eV == pytest.approx(
        independent_temperature_eV, rel=1e-12
    ), species
    # The paper reports B=1.68216831e8 in Eq. (16). Using exact c=299792458 m/s
    # gives B ~= 1.67982740e8; using the commonly rounded c=3.00e8 m/s gives
    # B ~= 1.68215405e8, within 0.001% of the paper's coefficient. Thus the
    # apparent ~0.139% difference is explained by rounding c, not evidence of
    # a physically inconsistent coefficient. Rounded tabulated widths/masses
    # account for remaining small line-by-line differences.
    assert software_temperature_eV == pytest.approx(
        published_temperature_eV, rel=0.004
    ), species


def test_fwhm_must_not_be_passed_as_gaussian_sigma():
    """Catch the common factor-of-2.35482 error in paper-to-code conversion."""
    _, mass_amu, wavelength_nm, doppler_fwhm_nm, expected_eV = PUBLISHED_LINES[9]
    correct = doppler_temperature_eV(
        doppler_fwhm_nm / FWHM_TO_SIGMA, wavelength_nm, mass_amu
    )
    wrong_if_fwhm_passed_as_sigma = doppler_temperature_eV(
        doppler_fwhm_nm, wavelength_nm, mass_amu
    )
    assert correct == pytest.approx(expected_eV, abs=0.02)
    assert wrong_if_fwhm_passed_as_sigma / correct == pytest.approx(
        FWHM_TO_SIGMA**2, rel=1e-12
    )
