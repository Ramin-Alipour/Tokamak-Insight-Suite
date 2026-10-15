"""Auditable Doppler ion-temperature calculation with explicit instrument correction.

This module intentionally refuses to return a physical temperature when the
instrumental FWHM is missing, inconsistent, or at least as broad as the
observed line. It assumes Gaussian Doppler and instrumental profiles and
requires the caller to establish that non-Doppler broadening is negligible.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

C_M_S = 299_792_458.0
E_CHARGE_J = 1.602_176_634e-19
AMU_KG = 1.660_539_066_60e-27


@dataclass(frozen=True)
class DopplerTemperatureResult:
    temperature_eV: float
    doppler_fwhm_nm: float
    observed_fwhm_nm: float
    instrumental_fwhm_nm: float
    uncertainty_eV: float | None
    status: str
    assumptions: tuple[str, ...]


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def doppler_temperature_eV(
    wavelength_nm: float,
    observed_fwhm_nm: float,
    ion_mass_amu: float,
    *,
    instrumental_fwhm_nm: float | None,
    observed_fwhm_uncertainty_nm: float | None = None,
    instrumental_fwhm_uncertainty_nm: float | None = None,
) -> DopplerTemperatureResult:
    """Infer ion temperature from Gaussian Doppler FWHM.

    Width deconvolution is performed in quadrature:
      FWHM_D = sqrt(FWHM_observed**2 - FWHM_instrument**2)

    Temperature uses T[eV] = (m*c^2/e) * (FWHM_D/lambda)^2/(8 ln 2).
    A result is never produced without an explicitly supplied instrument width.
    Uncertainty, when both width uncertainties are provided, is first-order
    propagation assuming independent Gaussian width errors.
    """
    lam_nm = _positive_finite("wavelength_nm", wavelength_nm)
    obs_nm = _positive_finite("observed_fwhm_nm", observed_fwhm_nm)
    mass_amu = _positive_finite("ion_mass_amu", ion_mass_amu)
    if instrumental_fwhm_nm is None:
        raise ValueError(
            "instrumental_fwhm_nm is required; uncorrected width cannot be "
            "reported as a Doppler width"
        )
    inst_nm = float(instrumental_fwhm_nm)
    if not math.isfinite(inst_nm) or inst_nm < 0:
        raise ValueError("instrumental_fwhm_nm must be finite and >= 0")
    if obs_nm <= inst_nm:
        raise ValueError(
            "observed FWHM must exceed instrumental FWHM; Doppler width is "
            "unresolved and temperature is not identifiable"
        )

    doppler_nm = math.sqrt(obs_nm * obs_nm - inst_nm * inst_nm)
    mass_kg = mass_amu * AMU_KG
    width_ratio = doppler_nm / lam_nm
    temp_ev = (mass_kg * C_M_S**2 / E_CHARGE_J) * width_ratio**2 / (8 * math.log(2))

    uncertainty = None
    if observed_fwhm_uncertainty_nm is not None or instrumental_fwhm_uncertainty_nm is not None:
        if observed_fwhm_uncertainty_nm is None or instrumental_fwhm_uncertainty_nm is None:
            raise ValueError("provide both width uncertainties, or neither")
        u_obs = float(observed_fwhm_uncertainty_nm)
        u_inst = float(instrumental_fwhm_uncertainty_nm)
        if not (math.isfinite(u_obs) and math.isfinite(u_inst) and u_obs >= 0 and u_inst >= 0):
            raise ValueError("width uncertainties must be finite and >= 0")
        # dT/T = 2*dD/D, D=sqrt(obs^2-inst^2)
        var_doppler = ((obs_nm / doppler_nm) * u_obs) ** 2 + ((inst_nm / doppler_nm) * u_inst) ** 2
        uncertainty = temp_ev * 2 * math.sqrt(var_doppler) / doppler_nm

    return DopplerTemperatureResult(
        temperature_eV=temp_ev,
        doppler_fwhm_nm=doppler_nm,
        observed_fwhm_nm=obs_nm,
        instrumental_fwhm_nm=inst_nm,
        uncertainty_eV=uncertainty,
        status="estimate_requires_physical_review",
        assumptions=(
            "Gaussian Doppler profile",
            "Gaussian instrumental line-spread function",
            "independent widths deconvolved in quadrature",
            "Stark, Zeeman, opacity, unresolved blends, and rotation are negligible or separately removed",
            "ion mass and rest wavelength correspond to the fitted transition",
        ),
    )


def doppler_fwhm_nm_for_temperature(
    wavelength_nm: float, ion_mass_amu: float, temperature_eV: float
) -> float:
    """Return theoretical Gaussian Doppler FWHM in nm for a given ion temperature."""
    lam_nm = _positive_finite("wavelength_nm", wavelength_nm)
    mass_amu = _positive_finite("ion_mass_amu", ion_mass_amu)
    temp_ev = _positive_finite("temperature_eV", temperature_eV)
    return lam_nm * math.sqrt(
        8 * math.log(2) * temp_ev * E_CHARGE_J
        / (mass_amu * AMU_KG * C_M_S**2)
    )
