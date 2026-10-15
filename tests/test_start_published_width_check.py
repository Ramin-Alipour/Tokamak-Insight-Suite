"""Regression check for the approximate START paper width."""
import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "start_published_width_check.py"
spec = importlib.util.spec_from_file_location("start_published_width_check", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StartPublishedWidthTests(unittest.TestCase):
    def test_approximate_width_gives_expected_temperature_scale(self):
        ti = module.doppler_ti_ev(module.APPROX_FWHM_NM)
        self.assertAlmostEqual(ti, 288.08, places=2)
        self.assertLess(abs(ti / module.PUBLISHED_APPROX_TI_EV - 1), 0.05)


if __name__ == "__main__":
    unittest.main()
