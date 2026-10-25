"""Physics-aware CXRS line-profile fitting for calibrated spectra.

This module estimates ion temperature and line-of-sight velocity from an isolated,
optically thin Gaussian emission line. It does not perform atomic-population,
neutral-beam attenuation, impurity-density, or toroidal/poloidal inversion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import curve_fit

_E_CHARGE = 1.602176634e-19
_C = 299792458.0
_K_B = 1.380649e-23
_U = 1.66053906660e-27


@dataclass(frozen=True)
class CXRSResult:
    wavelength_center_nm: float
    wavelength_center_uncertainty_nm: float
    line_sigma_nm: float
    intrinsic_sigma_nm: float
    ion_temperature_eV: float
    ion_temperature_uncertainty_eV: float
    instrument_sigma_uncertainty_nm: float
    line_of_sight_velocity_m_s: float
    velocity_uncertainty_m_s: float
    viewing_angle_rad: float
    amplitude: float
    background_intercept: float
    background_slope_per_nm: float
    fit_r_squared: float
    reduced_chi_square: float | None
    degrees_of_freedom: int
    fitted_signal: list[float]
    residual: list[float]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _line_model(x, amplitude, center, sigma, intercept, slope):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + intercept + slope * (x - center)


def analyze_cxrs_line(
    wavelength_nm,
    signal,
    *,
    rest_wavelength_nm: float,
    ion_mass_amu: float,
    instrument_sigma_nm: float,
    viewing_angle_rad: float = 0.0,
    signal_uncertainty=None,
    initial_center_nm: float | None = None,
    instrument_sigma_uncertainty_nm: float = 0.0,
) -> CXRSResult:
    """Fit one calibrated CXRS line and derive Ti and line-of-sight velocity.

    Parameters
    ----------
    wavelength_nm : array-like
        Strictly increasing wavelength axis in nm.
    signal : array-like
        Background-containing spectral signal in arbitrary or calibrated units.
    rest_wavelength_nm : float
        Rest wavelength in nm for the correctly identified charge-exchange line.
    ion_mass_amu : float
        Emitting ion mass in atomic mass units.
    instrument_sigma_nm : float
        Independently measured Gaussian instrumental standard deviation in nm.
    viewing_angle_rad : float, default 0
        Angle between the measured line of sight and the velocity component being
        interpreted. The function reports line-of-sight velocity directly and
        does not divide by this angle's cosine.
    signal_uncertainty : array-like, optional
        Positive 1-sigma uncertainty per sample; if supplied, used as fit weights.
    initial_center_nm : float, optional
        Initial line-center estimate. Defaults to wavelength at maximum signal.
    instrument_sigma_uncertainty_nm : float, default 0
        1-sigma uncertainty of the independently measured instrumental sigma.

    Notes
    -----
    Ion temperature assumes a single Maxwellian emitter and Gaussian Doppler
    broadening: sigma_intrinsic^2 = sigma_observed^2 - sigma_instrument^2 and
    Ti = m c^2 (sigma_intrinsic/lambda0)^2 / e.
    """
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(signal, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 7:
        raise ValueError("wavelength_nm and signal must be matching 1-D arrays with at least 7 samples")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("wavelength_nm and signal must contain only finite values")
    if np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm must be strictly increasing")
    if not x[0] <= rest_wavelength_nm <= x[-1]:
        raise ValueError("rest_wavelength_nm must lie within the measured wavelength range")
    for name, value in (("rest_wavelength_nm", rest_wavelength_nm),
                        ("ion_mass_amu", ion_mass_amu),
                        ("instrument_sigma_nm", instrument_sigma_nm)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(viewing_angle_rad) or not 0 <= viewing_angle_rad <= np.pi:
        raise ValueError("viewing_angle_rad must be finite and within [0, pi]")
    if not np.isfinite(instrument_sigma_uncertainty_nm) or instrument_sigma_uncertainty_nm < 0:
        raise ValueError("instrument_sigma_uncertainty_nm must be finite and non-negative")
    sy = None
    if signal_uncertainty is not None:
        sy = np.asarray(signal_uncertainty, dtype=float)
        if sy.shape != y.shape or not np.all(np.isfinite(sy)) or np.any(sy <= 0):
            raise ValueError("signal_uncertainty must match signal and be finite and positive")

    edge_n = max(2, x.size // 8)
    edge_x = np.r_[x[:edge_n], x[-edge_n:]]
    edge_y = np.r_[y[:edge_n], y[-edge_n:]]
    slope0, intercept0 = np.polyfit(edge_x - np.mean(x), edge_y, 1)
    center0 = float(x[np.argmax(y)]) if initial_center_nm is None else float(initial_center_nm)
    if not np.isfinite(center0) or center0 < x[0] or center0 > x[-1]:
        raise ValueError("initial_center_nm must lie within the wavelength axis")
    baseline = intercept0 + slope0 * (x - np.mean(x))
    amp0 = float(np.max(y - baseline))
    if not np.isfinite(amp0) or amp0 <= 0:
        raise ValueError("spectrum must contain a positive emission feature above the edge baseline")
    spacing = float(np.median(np.diff(x)))
    span = float(x[-1] - x[0])
    p0 = [amp0, center0, max(2 * spacing, min(span / 8, rest_wavelength_nm * 1e-4)), intercept0 + slope0 * (center0 - np.mean(x)), slope0]
    lower = [0.0, x[0], spacing * 0.25, -np.inf, -np.inf]
    upper = [np.inf, x[-1], span, np.inf, np.inf]
    try:
        popt, pcov = curve_fit(_line_model, x, y, p0=p0, sigma=sy, absolute_sigma=sy is not None,
                               bounds=(lower, upper), maxfev=30000)
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        raise ValueError(f"CXRS line fit failed: {exc}") from exc
    amp, center, observed_sigma, intercept, slope = map(float, popt)
    if observed_sigma <= instrument_sigma_nm:
        raise ValueError("observed line width does not exceed instrumental width; ion temperature is unresolved")
    intrinsic_sigma = float(np.sqrt(observed_sigma**2 - instrument_sigma_nm**2))
    mass = ion_mass_amu * _U
    ti = mass * _C**2 * (intrinsic_sigma / rest_wavelength_nm) ** 2 / _E_CHARGE
    center_unc = float(np.sqrt(max(0.0, pcov[1, 1]))) if np.isfinite(pcov[1, 1]) else float("nan")
    sigma_unc = float(np.sqrt(max(0.0, pcov[2, 2]))) if np.isfinite(pcov[2, 2]) else float("nan")
    ti_unc_fit = 2 * ti * observed_sigma * sigma_unc / intrinsic_sigma**2
    ti_unc_instrument = 2 * ti * instrument_sigma_nm * instrument_sigma_uncertainty_nm / intrinsic_sigma**2
    ti_unc = float(np.hypot(ti_unc_fit, ti_unc_instrument))
    velocity = _C * (center - rest_wavelength_nm) / rest_wavelength_nm
    velocity_unc = _C * center_unc / rest_wavelength_nm
    fitted = _line_model(x, *popt)
    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    dof = int(x.size - len(popt))
    redchi = float(np.sum((residual / sy) ** 2) / dof) if sy is not None and dof > 0 else None
    return CXRSResult(
        wavelength_center_nm=center,
        wavelength_center_uncertainty_nm=center_unc,
        line_sigma_nm=observed_sigma,
        intrinsic_sigma_nm=intrinsic_sigma,
        ion_temperature_eV=float(ti),
        ion_temperature_uncertainty_eV=ti_unc,
        instrument_sigma_uncertainty_nm=float(instrument_sigma_uncertainty_nm),
        line_of_sight_velocity_m_s=float(velocity),
        velocity_uncertainty_m_s=float(velocity_unc),
        viewing_angle_rad=float(viewing_angle_rad),
        amplitude=amp,
        background_intercept=intercept,
        background_slope_per_nm=slope,
        fit_r_squared=r2,
        reduced_chi_square=redchi,
        degrees_of_freedom=dof,
        fitted_signal=fitted.tolist(),
        residual=residual.tolist(),
        assumptions=("calibrated wavelength axis", "isolated optically thin line", "single Maxwellian emitting-ion population",
                     "Gaussian Doppler and instrumental line shapes", "known rest wavelength, ion mass and instrumental width"),
        limitations=("No atomic-rate or neutral-beam attenuation model; no impurity density estimate",
                     "Velocity is line-of-sight only; toroidal/poloidal inversion requires multiple calibrated views and geometry",
                     "Fit covariance excludes wavelength, rest-wavelength, instrument-width and calibration systematics",
                     "Blends, opacity, non-Maxwellian distributions and non-Gaussian instrument response are not modelled",
                     "viewing_angle_rad is metadata only and does not imply a geometry inversion"),
    )
