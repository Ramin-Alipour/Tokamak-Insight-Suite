"""Electrical-probe analysis routines for tokamak edge plasma diagnostics."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def fit_swept_ball_pen_ion_temperature(
    voltage_V,
    current_A,
    *,
    alpha_bpp: float = 0.6,
    sigma_A=None,
    fit_min_voltage_V: float | None = None,
    initial_ti_eV: float = 20.0,
    plasma_potential_guess_V: float | None = None,
) -> dict[str, Any]:
    """Fit ion temperature from the electron branch of a swept ball-pen-probe I-V curve.

    Model from Cipciar, Ion temperature measurement on tokamak Golem (2020):
        I(V) = I_sat_minus * [1 + R*(V-Phi)]
               - I_sat_plus * exp((Phi-V)/Ti)
        I_sat_minus = exp(alpha_bpp) * I_sat_plus

    current_A may be in A or mA; the fitted saturation-current parameter uses
    the same current unit. R is in inverse volts. Temperatures are in eV.

    This fit is valid only after parasitic capacitive current has been removed and
    after selecting the electron branch. The caller must supply a voltage cut or
    an independently justified branch selection. This function does not clean raw
    signals, infer a branch from an uncleaned waveform, or certify plasma conditions.
    """
    from scipy.optimize import curve_fit

    v = np.asarray(voltage_V, dtype=float).reshape(-1)
    i = np.asarray(current_A, dtype=float).reshape(-1)
    if v.size != i.size:
        raise ValueError("voltage_V and current_A must have equal lengths.")
    if not math.isfinite(float(alpha_bpp)):
        raise ValueError("alpha_bpp must be finite.")
    good = np.isfinite(v) & np.isfinite(i)
    if sigma_A is not None:
        sig = np.asarray(sigma_A, dtype=float).reshape(-1)
        if sig.size != v.size:
            raise ValueError("sigma_A must have the same length as voltage_V.")
        good &= np.isfinite(sig) & (sig > 0)
    else:
        sig = None
    if fit_min_voltage_V is not None:
        good &= v >= float(fit_min_voltage_V)
    v, i = v[good], i[good]
    if sig is not None:
        sig = sig[good]
    if v.size < 8:
        raise ValueError("At least eight finite electron-branch samples are required.")
    order = np.argsort(v)
    v, i = v[order], i[order]
    if sig is not None:
        sig = sig[order]
    span = float(np.ptp(v))
    scale = float(max(np.ptp(i), np.max(np.abs(i)), 1e-12))
    if span <= 0 or scale <= 0:
        raise ValueError("The selected I-V data must have nonzero voltage and current range.")

    phi0 = float(plasma_potential_guess_V) if plasma_potential_guess_V is not None else float(np.percentile(v, 10))
    phi_low = float(np.min(v) - max(1.0, 0.2 * span))
    phi_high = float(np.max(v))
    ti0 = float(np.clip(initial_ti_eV, 0.2, 500.0))
    isat0 = float(max(np.median(i[-max(3, len(i)//5):]) / math.exp(alpha_bpp), scale * 0.05, 1e-12))
    r0 = max(0.0, min(0.01, scale / max(span, 1e-9) / max(isat0, 1e-12) * 0.01))

    def model(x, ti, phi, isat_plus, r):
        isat_minus = math.exp(alpha_bpp) * isat_plus
        return isat_minus * (1.0 + r * (x - phi)) - isat_plus * np.exp(np.clip((phi - x) / ti, -100, 100))

    lower = [0.2, phi_low, max(scale * 1e-9, 1e-15), 0.0]
    upper = [1000.0, phi_high, scale * 100.0, 1.0]
    p0 = [ti0, float(np.clip(phi0, phi_low + 1e-6, phi_high - 1e-6)), isat0, r0]
    kwargs = {"maxfev": 100000, "bounds": (lower, upper)}
    if sig is not None:
        kwargs.update(sigma=sig, absolute_sigma=True)
    params, covariance = curve_fit(model, v, i, p0=p0, **kwargs)
    ti, phi, isat_plus, r = [float(x) for x in params]
    fitted = model(v, *params)
    residual = i - fitted
    dof = max(1, v.size - len(params))
    rms = float(np.sqrt(np.mean(residual**2)))
    covariance_ok = bool(np.all(np.isfinite(covariance)))
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0)) if covariance_ok else np.full(4, np.nan)
    ss_tot = float(np.sum((i - np.mean(i))**2))
    ss_res = float(np.sum(residual**2))
    return {
        "method": "swept BPP electron-branch four-parameter nonlinear fit",
        "model": "I=exp(alpha)*Isat_plus*(1+R*(V-Phi))-Isat_plus*exp((Phi-V)/Ti)",
        "alpha_bpp": float(alpha_bpp),
        "Ti_eV": ti,
        "Ti_std_eV": float(errors[0]) if np.isfinite(errors[0]) else None,
        "plasma_potential_V": phi,
        "plasma_potential_std_V": float(errors[1]) if np.isfinite(errors[1]) else None,
        "I_sat_plus_same_units_as_input": isat_plus,
        "I_sat_minus_same_units_as_input": float(math.exp(alpha_bpp) * isat_plus),
        "electron_current_slope_per_V": r,
        "parameter_std": [float(x) if np.isfinite(x) else None for x in errors],
        "n_points_used": int(v.size),
        "voltage_fit_range_V": [float(v.min()), float(v.max())],
        "rms_residual_same_units_as_current": rms,
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "degrees_of_freedom": int(dof),
        "covariance_finite": covariance_ok,
        "fit_is_physical_validation": False,
        "required_preconditions": [
            "Remove parasitic capacitive current proportional to dV/dt.",
            "Select the electron branch and document the voltage cut.",
            "Check fit residuals and covariance; compare only with same-shot/time reference.",
        ],
    }


def golem_bpp_alpha_from_toroidal_field(bt_T: float) -> dict[str, float | str]:
    """Empirical GOLEM BPP calibration reported by Cipciar (2021).

    alpha_BPP(Bt) = (-2.7 ± 0.5) * Bt[T] + (2.04 ± 0.2).
    The coefficient is campaign/setup-specific; it must not be generalized to
    other tokamaks or assumed valid outside the reported GOLEM magnetic-field
    range. The returned uncertainty combines slope and intercept uncertainties
    in quadrature, treating them as independent.
    """
    bt = float(bt_T)
    if not math.isfinite(bt) or bt < 0 or bt > 1.0:
        raise ValueError("bt_T must be finite and within the documented GOLEM range [0, 1] T.")
    alpha = -2.7 * bt + 2.04
    sigma = math.sqrt((0.5 * bt) ** 2 + 0.2 ** 2)
    return {
        "alpha_bpp": alpha,
        "alpha_bpp_std": sigma,
        "bt_T": bt,
        "source": "Cipciar, Ion temperature measurement on tokamak Golem (2021), Eq. (3)",
        "note": "Empirical GOLEM campaign calibration; confirm applicability to the specific probe/setup before using for a reported Ti.",
    }


def fit_swept_ball_pen_with_alpha_uncertainty(
    voltage_V, current_A, *, alpha_bpp: float, alpha_bpp_std: float, **kwargs
) -> dict[str, Any]:
    """Fit swept BPP Ti and quantify sensitivity to the empirical alpha calibration.

    Reports the nominal fit and the range obtained by refitting at alpha ± 1 sigma.
    This is a sensitivity envelope, not a full joint posterior or a replacement for
    uncertainty from raw-signal cleanup, hysteresis, and branch selection.
    """
    sigma = float(alpha_bpp_std)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("alpha_bpp_std must be finite and nonnegative.")
    nominal = fit_swept_ball_pen_ion_temperature(
        voltage_V, current_A, alpha_bpp=alpha_bpp, **kwargs
    )
    refit_kwargs = dict(kwargs)
    refit_kwargs.pop("initial_ti_eV", None)
    refit_kwargs.pop("plasma_potential_guess_V", None)
    low = fit_swept_ball_pen_ion_temperature(
        voltage_V, current_A, alpha_bpp=alpha_bpp - sigma,
        initial_ti_eV=nominal["Ti_eV"],
        plasma_potential_guess_V=nominal["plasma_potential_V"], **refit_kwargs
    )
    high = fit_swept_ball_pen_ion_temperature(
        voltage_V, current_A, alpha_bpp=alpha_bpp + sigma,
        initial_ti_eV=nominal["Ti_eV"],
        plasma_potential_guess_V=nominal["plasma_potential_V"], **refit_kwargs
    )
    values = [low["Ti_eV"], nominal["Ti_eV"], high["Ti_eV"]]
    nominal["alpha_bpp_std"] = sigma
    nominal["Ti_range_from_alpha_plusminus_1sigma_eV"] = [float(min(values)), float(max(values))]
    nominal["Ti_alpha_sensitivity_half_range_eV"] = float((max(values) - min(values)) / 2.0)
    nominal["alpha_uncertainty_treatment"] = "Refit at alpha - 1 sigma and alpha + 1 sigma; not a joint confidence interval."
    return nominal
