"""Regression checks for the published-width Doppler-temperature reproduction."""
import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "aditya_u_ti_reproduction.py"
spec = importlib.util.spec_from_file_location("aditya_u_ti_reproduction", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AdityaUPublishedWidthTests(unittest.TestCase):
    def test_reported_width_reproduces_published_temperature(self):
        ti = module.doppler_ti_ev(module.WIDTH_NM)
        self.assertAlmostEqual(ti, 39.6754, places=3)
        self.assertLess(abs(ti - module.PUBLISHED_TI_EV) / module.PUBLISHED_TI_EV, 0.02)

    def test_instrument_subtraction_is_material(self):
        from math import sqrt
        corrected = sqrt(module.WIDTH_NM**2 - module.INSTRUMENT_FWHM_NM**2)
        ti = module.doppler_ti_ev(corrected)
        self.assertAlmostEqual(ti, 25.4797, places=3)
        self.assertLess(ti, 0.7 * module.PUBLISHED_TI_EV)

    def test_published_uncertainty_is_not_reproduced_by_width_uncertainty(self):
        ti = module.doppler_ti_ev(module.WIDTH_NM)
        sigma_ti = 2 * ti * module.WIDTH_SIGMA_NM / module.WIDTH_NM
        # The paper prints ±0.003 eV, but the stated ±0.0006 nm width error
        # propagates to about ±0.73 eV under T proportional to width squared.
        self.assertAlmostEqual(sigma_ti, 0.7302, places=3)
        self.assertGreater(sigma_ti, 100 * 0.003)


if __name__ == "__main__":
    unittest.main()
