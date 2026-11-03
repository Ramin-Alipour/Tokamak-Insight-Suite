"""Formal radiative transfer for ECE along a prescribed 1-D ray.

Solves dI_nu/ds = j_nu - alpha_nu I_nu exactly for piecewise-constant cells.
Ray tracing, refraction, reflection, scattering and mode coupling are outside
this transfer step and must be handled by the propagation solver.
"""
from __future__ import annotations
import numpy as np
from scipy.constants import h, k as K_B, c


def planck_spectral_radiance(frequency_Hz, temperature_K):
    """Planck source radiance B_nu [W m^-2 sr^-1 Hz^-1]."""
    nu = np.asarray(frequency_Hz, dtype=float)
    temp = np.asarray(temperature_K, dtype=float)
    if nu.ndim != 1 or nu.size == 0 or np.any(~np.isfinite(nu)) or np.any(nu <= 0):
        raise ValueError("frequency_Hz must be a finite positive vector")
    if np.any(~np.isfinite(temp)) or np.any(temp <= 0):
        raise ValueError("temperature_K must be finite and positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = 2.0*h*nu**3/c**2 / np.expm1(h*nu/(K_B*temp))
    if np.any(~np.isfinite(result)) or np.any(result < 0):
        raise FloatingPointError("Planck radiance outside representable range")
    return result


def rayleigh_jeans_spectral_radiance(frequency_Hz, temperature_K):
    """Rayleigh-Jeans approximation; valid for h nu much less than k_B T."""
    nu = np.asarray(frequency_Hz, dtype=float)
    temp = np.asarray(temperature_K, dtype=float)
    if np.any(~np.isfinite(nu)) or np.any(nu <= 0) or np.any(~np.isfinite(temp)) or np.any(temp <= 0):
        raise ValueError("frequency and temperature must be finite and positive")
    return 2.0*K_B*temp*nu**2/c**2


def solve_ece_radiative_transfer(
    frequency_Hz, path_edges_m, absorption_coefficient_m_inverse, *,
    emissivity_W_m3_sr_Hz=None, electron_temperature_K=None,
    incident_radiance_W_m2_sr_Hz=0.0, source_function="planck",
):
    """Propagate intensity through cells using the exact formal solution.

    alpha has shape (cell,) or (frequency, cell). Supply exactly one of an
    explicit emissivity j_nu with that shape or LTE electron_temperature_K
    with shape (cell,) or (frequency, cell). In LTE, j_nu=alpha_nu B_nu.
    """
    nu = np.asarray(frequency_Hz, dtype=float)
    edges = np.asarray(path_edges_m, dtype=float)
    alpha = np.asarray(absorption_coefficient_m_inverse, dtype=float)
    if nu.ndim != 1 or nu.size == 0 or np.any(~np.isfinite(nu)) or np.any(nu <= 0):
        raise ValueError("frequency_Hz must be a finite positive vector")
    if edges.ndim != 1 or edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("path_edges_m must be finite and strictly increasing")
    ncell = edges.size - 1
    if alpha.shape not in {(ncell,), (nu.size, ncell)} or np.any(~np.isfinite(alpha)) or np.any(alpha < 0):
        raise ValueError("absorption must be non-negative with cell or frequency-cell shape")
    if alpha.ndim == 1:
        alpha = np.broadcast_to(alpha[None, :], (nu.size, ncell))
    if (emissivity_W_m3_sr_Hz is None) == (electron_temperature_K is None):
        raise ValueError("supply exactly one of emissivity or LTE electron_temperature_K")
    if electron_temperature_K is not None:
        temp = np.asarray(electron_temperature_K, dtype=float)
        if temp.shape not in {(ncell,), (nu.size, ncell)} or np.any(~np.isfinite(temp)) or np.any(temp <= 0):
            raise ValueError("temperature must be positive with cell or frequency-cell shape")
        if temp.ndim == 1:
            temp = np.broadcast_to(temp[None, :], (nu.size, ncell))
        if source_function == "planck":
            source = np.column_stack([planck_spectral_radiance(nu, temp[:, j]) for j in range(ncell)])
        elif source_function == "rayleigh_jeans":
            source = rayleigh_jeans_spectral_radiance(nu[:, None], temp)
        else:
            raise ValueError("source_function must be planck or rayleigh_jeans")
        emissivity = alpha * source
        model = "LTE_Kirchhoff_" + source_function
    else:
        emissivity = np.asarray(emissivity_W_m3_sr_Hz, dtype=float)
        if emissivity.shape == (ncell,):
            emissivity = np.broadcast_to(emissivity[None, :], (nu.size, ncell))
        if emissivity.shape != (nu.size, ncell) or np.any(~np.isfinite(emissivity)) or np.any(emissivity < 0):
            raise ValueError("emissivity must be non-negative finite with cell or frequency-cell shape")
        model = "user_supplied_emissivity"
    incident = np.asarray(incident_radiance_W_m2_sr_Hz, dtype=float)
    if incident.ndim == 0:
        incident = np.full(nu.size, float(incident))
    if incident.shape != (nu.size,) or np.any(~np.isfinite(incident)) or np.any(incident < 0):
        raise ValueError("incident radiance must be a non-negative scalar or frequency vector")
    intensity = incident.copy()
    optical_depth = np.zeros(nu.size)
    for j, ds in enumerate(np.diff(edges)):
        tau = alpha[:, j] * ds
        attenuation = np.exp(-tau)
        source = np.divide(emissivity[:, j], alpha[:, j], out=np.zeros(nu.size), where=alpha[:, j] > 0)
        intensity = intensity*attenuation + source*(-np.expm1(-tau))
        zero_alpha = alpha[:, j] == 0
        intensity[zero_alpha] += emissivity[zero_alpha, j]*ds
        optical_depth += tau
    return {
        "frequency_Hz": nu.copy(),
        "spectral_radiance_W_m2_sr_Hz": intensity,
        "optical_depth": optical_depth,
        "transmission": np.exp(-optical_depth),
        "absorption_coefficient_m_inverse": alpha.copy(),
        "emission_model": model,
        "transfer_equation": "dI_nu/ds = j_nu - alpha_nu*I_nu",
        "solver": "exact_piecewise_constant_formal_solution",
        "scope": "1-D prescribed ray; no refraction, reflection, scattering, or mode coupling",
    }
