"""Standard tokamak plasma parameters, empirical scalings and threshold laws.

Every function states its formula, units and source convention. These are *empirical
or textbook* relations: they are meant for comparison and sanity checks, not as
validated predictions for a specific device.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .core import AMU, E_CHARGE, EPS0, MU0

# --- helpers ---------------------------------------------------------------

def _pos(name: str, v: float) -> float:
    v = float(v)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return v


def coulomb_logarithm_electron(ne_m3: float, te_eV: float) -> float:
    """NRL electron-ion Coulomb logarithm for Z=1 (n_e in m^-3, T_e in eV).

    T_e >= 10 eV: 24 - ln(sqrt(n_e[cm^-3]) / T_e);  T_e < 10 eV: 23 - ln(sqrt(n_e[cm^-3]) T_e^-1.5).
    """
    ne = _pos("ne_m3", ne_m3); te = _pos("te_eV", te_eV)
    ne_cc = ne * 1e-6
    lam = 24.0 - math.log(math.sqrt(ne_cc) / te) if te >= 10.0 else 23.0 - math.log(math.sqrt(ne_cc) * te ** -1.5)
    return float(max(lam, 1.0))


def spitzer_resistivity(te_eV: float, zeff: float = 1.0, ne_m3: float | None = None,
                        ln_lambda: float | None = None) -> dict[str, Any]:
    """Classical Spitzer resistivity (NRL): eta_perp = 1.03e-4 Z lnL T^-3/2 [ohm m]; eta_par = eta_perp/1.96.

    Neoclassical trapped-particle corrections are NOT included (they raise eta in a
    tokamak by 1/(1-1.95 sqrt(eps)+...)), and Zeff is a single-species proxy.
    """
    te = _pos("te_eV", te_eV); z = _pos("zeff", zeff)
    if ln_lambda is None:
        if ne_m3 is None:
            raise ValueError("Provide ne_m3 or ln_lambda; no Coulomb logarithm is assumed.")
        ln_lambda = coulomb_logarithm_electron(ne_m3, te)
    ln_lambda = _pos("ln_lambda", ln_lambda)
    eta_perp = 1.03e-4 * z * ln_lambda * te ** -1.5
    return {"eta_perp_ohm_m": eta_perp, "eta_parallel_ohm_m": eta_perp / 1.96,
            "coulomb_logarithm": float(ln_lambda), "zeff": z,
            "note": "Classical Spitzer; no neoclassical correction."}


def electron_collision_frequency(ne_m3: float, te_eV: float, ln_lambda: float | None = None) -> float:
    """NRL nu_e = 2.91e-6 n_e[cm^-3] lnL T_e^-3/2  [s^-1]."""
    ne = _pos("ne_m3", ne_m3); te = _pos("te_eV", te_eV)
    ll = coulomb_logarithm_electron(ne, te) if ln_lambda is None else _pos("ln_lambda", ln_lambda)
    return float(2.91e-6 * (ne * 1e-6) * ll * te ** -1.5)


def larmor_radii_and_speeds(te_eV: float, ti_eV: float, b_T: float, ne_m3: float | None = None,
                            ion_mass_amu: float = 2.0) -> dict[str, float]:
    """Gyro-radius family. rho_s = c_s/Omega_i with c_s = sqrt(Te/m_i); rho_i uses v_th = sqrt(2 Ti/m_i);
    rho_e uses v_th = sqrt(2 Te/m_e). Alfven speed needs ne (quasi-neutral, n_i=n_e, single species)."""
    te = _pos("te_eV", te_eV); ti = _pos("ti_eV", ti_eV); b = _pos("|B|", abs(b_T)); mi = _pos("ion_mass_amu", ion_mass_amu) * AMU
    me = 9.1093837139e-31
    omega_i = E_CHARGE * b / mi
    out = {"ion_cyclotron_frequency_Hz": omega_i / (2 * math.pi),
           "electron_cyclotron_frequency_Hz": E_CHARGE * b / me / (2 * math.pi),
           "rho_s_m": math.sqrt(E_CHARGE * te / mi) / omega_i,
           "rho_i_m": math.sqrt(2 * E_CHARGE * ti / mi) / omega_i,
           "rho_e_m": math.sqrt(2 * E_CHARGE * te / me) / (E_CHARGE * b / me)}
    if ne_m3 is not None:
        out["alfven_speed_m_s"] = b / math.sqrt(MU0 * _pos("ne_m3", ne_m3) * mi)
    return out


# --- geometry / MHD --------------------------------------------------------

def shaping_surface_area(r_m: float, a_m: float, kappa: float) -> float:
    """Approximate plasma surface area S ~ 4 pi^2 R a sqrt((1+kappa^2)/2) [m^2] (ITER Physics Basis)."""
    return float(4 * math.pi ** 2 * _pos("R", r_m) * _pos("a", a_m) * math.sqrt((1 + _pos("kappa", kappa) ** 2) / 2))


def q95_iter_formula(ip_MA: float, bt_T: float, r_m: float, a_m: float, kappa95: float,
                     delta95: float = 0.0) -> float:
    """ITER Physics Basis q95 (Uckan): 5 a^2 B/(R I) * (1+k^2(1+2d^2-1.2d^3))/2 * (1.17-0.65e)/(1-e^2)^2.

    Units: I [MA], B [T], R, a [m]. kappa95 and delta95 are the 95%-flux-surface shaping values.
    """
    ip = _pos("|Ip| (MA)", abs(ip_MA)); b = _pos("|Bt|", abs(bt_T)); r = _pos("R", r_m); a = _pos("a", a_m); k = _pos("kappa95", kappa95)
    eps = a / r
    if not 0 < eps < 1:
        raise ValueError("Inverse aspect ratio a/R must be in (0,1).")
    d = float(delta95)
    if not math.isfinite(d) or not -1 < d < 1:
        raise ValueError("delta95 must be finite and lie strictly between -1 and 1.")
    shape = (1 + k ** 2 * (1 + 2 * d ** 2 - 1.2 * d ** 3)) / 2
    return float(5 * a ** 2 * b / (r * ip) * shape * (1.17 - 0.65 * eps) / (1 - eps ** 2) ** 2)


def q_star_cylindrical(ip_MA: float, bt_T: float, r_m: float, a_m: float, kappa: float) -> float:
    """Cylindrical-equivalent edge safety factor q* = 5 a^2 B (1+kappa^2)/(2 R I[MA]).

    The large-aspect-ratio cylindrical approximation requires 0 < a/R < 1.
    Reject geometrically invalid inputs rather than returning a plausible-looking q*.
    """
    r = _pos("R", r_m)
    a = _pos("a", a_m)
    eps = a / r
    if not 0 < eps < 1:
        raise ValueError("Inverse aspect ratio a/R must be in (0,1).")
    return float(5 * a ** 2 * _pos("|Bt|", abs(bt_T)) * (1 + _pos("kappa", kappa) ** 2)
                 / (2 * r * _pos("|Ip| (MA)", abs(ip_MA))))


def beta_metrics(ne_m3: float, te_eV: float, ti_eV: float, ip_A: float, bt_T: float,
                 a_m: float, kappa: float = 1.0, ni_m3: float | None = None) -> dict[str, Any]:
    """beta_t, beta_p and normalized beta_N from *volume-averaged* n and T supplied by the user.

    beta_t = 2 mu0 <p>/B_t^2;  <B_p> = mu0 I/L with L = 2 pi a sqrt((1+k^2)/2);  beta_p = 2 mu0 <p>/<B_p>^2;
    beta_N[%] = beta_t[%] a[m] B[T] / I[MA].  Uses the *given* averages; it does not integrate profiles.
    """
    ne = _pos("ne_m3", ne_m3); te = _pos("te_eV", te_eV); ti = _pos("ti_eV", ti_eV)
    ni = ne if ni_m3 is None else _pos("ni_m3", ni_m3)
    ip = _pos("|Ip| (A)", abs(ip_A)); b = _pos("|Bt|", abs(bt_T)); a = _pos("a", a_m)
    p = E_CHARGE * (ne * te + ni * ti)
    beta_t = 2 * MU0 * p / b ** 2
    circ = 2 * math.pi * a * math.sqrt((1 + _pos("kappa", kappa) ** 2) / 2)
    bp = MU0 * ip / circ
    beta_p = 2 * MU0 * p / bp ** 2
    beta_n = 100.0 * beta_t * a * b / (ip / 1e6)
    return {"pressure_Pa": p, "beta_t": beta_t, "beta_p": beta_p, "beta_N_percent_m_T_per_MA": beta_n,
            "mean_poloidal_field_T": bp, "troyon_reference_beta_N": 2.8,
            "note": "beta_N ~2.8 is the conventional Troyon reference for no-wall limits; device limits differ. "
                    "Inputs must be volume averages for these to be meaningful."}


def collisionality_nu_star(ne_m3: float, te_eV: float, q: float, r_m: float, a_m: float,
                           ln_lambda: float | None = None) -> float:
    """Electron banana-regime collisionality nu*_e = 6.921e-18 q R n_e lnL / (eps^1.5 T_e^2)  (n in m^-3, T in eV)."""
    ne = _pos("ne_m3", ne_m3); te = _pos("te_eV", te_eV); qq = _pos("q", q); r = _pos("R", r_m); a = _pos("a", a_m)
    eps = a / r
    if not 0 < eps < 1:
        raise ValueError("Inverse aspect ratio a/R must be in (0,1).")
    ll = coulomb_logarithm_electron(ne, te) if ln_lambda is None else _pos("ln_lambda", ln_lambda)
    return float(6.921e-18 * qq * r * ne * ll / (eps ** 1.5 * te ** 2))


# --- confinement scalings and thresholds ------------------------------------

def greenwald(ne_line_avg_m3: float, ip_A: float, a_m: float) -> dict[str, float]:
    """Greenwald density n_G = I[MA]/(pi a^2) [1e20 m^-3] and fraction n_e/n_G (line-averaged n_e)."""
    ng = _pos("|Ip|", abs(ip_A)) / 1e6 / (math.pi * _pos("a", a_m) ** 2)
    ne20 = _pos("ne_line_avg_m3", ne_line_avg_m3) / 1e20
    return {"greenwald_density_1e20_m-3": ng, "greenwald_fraction": ne20 / ng}


def ipb98y2_confinement_time(ip_MA: float, bt_T: float, p_loss_MW: float, ne_19: float,
                             mass_amu: float, r_m: float, a_m: float, kappa_a: float) -> dict[str, Any]:
    """IPB98(y,2) H-mode thermal energy confinement scaling (ITER Physics Basis, 1999):

    tau_E = 0.0562 Ip^0.93 Bt^0.15 P^-0.69 n19^0.41 M^0.19 R^1.97 eps^0.58 kappa_a^0.78
    with Ip [MA], Bt [T], P_loss [MW], line-avg n_e [1e19 m^-3], M [amu], R [m], eps=a/R,
    kappa_a = V/(2 pi^2 R a^2). Derived from a *conventional-aspect-ratio* H-mode database;
    it is known to under-/over-predict spherical tokamaks, so use H98 there as a rough guide only.
    """
    ip = _pos("|Ip| (MA)", abs(ip_MA)); b = _pos("|Bt|", abs(bt_T)); p = _pos("p_loss_MW", p_loss_MW)
    n = _pos("ne_19", ne_19); m = _pos("mass_amu", mass_amu); r = _pos("R", r_m); a = _pos("a", a_m); k = _pos("kappa_a", kappa_a)
    eps = a / r
    tau = 0.0562 * ip ** 0.93 * b ** 0.15 * p ** -0.69 * n ** 0.41 * m ** 0.19 * r ** 1.97 * eps ** 0.58 * k ** 0.78
    return {"tau_E_IPB98y2_s": float(tau), "scaling": "IPB98(y,2)",
            "note": "Empirical H-mode scaling; P_loss must be the net loss power (P_heat - dW/dt, optionally minus radiated core power by convention)."}


def h_factor(tau_exp_s: float, tau_scaling_s: float) -> float:
    """H = tau_E,exp / tau_E,scaling."""
    return float(_pos("tau_exp_s", tau_exp_s) / _pos("tau_scaling_s", tau_scaling_s))


def lh_threshold_martin08(ne_line_avg_m3: float, bt_T: float, surface_m2: float,
                          ion_mass_amu: float = 2.0) -> dict[str, Any]:
    """Martin (2008) L-H power threshold: P_LH[MW] = 0.0488 n20^0.717 B^0.803 S^0.941 (deuterium).

    Hydrogen plasmas need roughly a factor 2 more power (scaled here as 2/M for M<2 only).
    The scaling has ~±30% scatter and does not capture low-density branch/ isotope/ divertor effects.
    """
    n20 = _pos("ne_line_avg_m3", ne_line_avg_m3) / 1e20
    p = 0.0488 * n20 ** 0.717 * _pos("|Bt|", abs(bt_T)) ** 0.803 * _pos("surface_m2", surface_m2) ** 0.941
    m = _pos("ion_mass_amu", ion_mass_amu)
    if m < 2.0:
        p *= 2.0 / m
    return {"P_LH_MW": float(p), "scaling": "Martin 2008 (J. Phys.: Conf. Ser. 123 012033)",
            "uncertainty_note": "Typical regression scatter about ±30%; not a sharp threshold."}


def fusion_triple_product(ni0_m3: float, ti0_keV: float, tau_e_s: float) -> dict[str, float]:
    """n_i(0) T_i(0) tau_E in keV s m^-3 (compare with ~3e21 for ignition-relevant D-T)."""
    return {"triple_product_keV_s_m-3": float(_pos("ni0_m3", ni0_m3) * _pos("ti0_keV", ti0_keV) * _pos("tau_e_s", tau_e_s)),
            "ignition_reference_keV_s_m-3": 3e21}


def confinement_summary(*, ip_A: float, bt_T: float, r_m: float, a_m: float, kappa: float,
                        ne_line_avg_m3: float, p_loss_W: float, mass_amu: float = 2.0,
                        stored_energy_J: float | None = None, delta: float = 0.0,
                        kappa_a: float | None = None) -> dict[str, Any]:
    """One-call bundle: Greenwald fraction, q*, q95 (ITER formula, kappa95 ~ kappa/1.12 estimate flagged),
    IPB98(y,2) tau_E, H98 (if stored energy supplied) and the Martin L-H threshold."""
    ip_MA = abs(ip_A) / 1e6
    out: dict[str, Any] = {"greenwald": greenwald(ne_line_avg_m3, ip_A, a_m),
                           "q_star_cylindrical": q_star_cylindrical(ip_MA, bt_T, r_m, a_m, kappa)}
    k95 = kappa / 1.12
    out["q95_iter_formula"] = q95_iter_formula(ip_MA, bt_T, r_m, a_m, k95, delta)
    out["q95_note"] = "kappa95 estimated as kappa_sep/1.12; supply measured 95% shaping for a real q95."
    ka = kappa if kappa_a is None else kappa_a
    scal = ipb98y2_confinement_time(ip_MA, bt_T, p_loss_W / 1e6, ne_line_avg_m3 / 1e19, mass_amu, r_m, a_m, ka)
    out["ipb98y2"] = scal
    if stored_energy_J is not None:
        tau = _pos("stored_energy_J", stored_energy_J) / _pos("p_loss_W", p_loss_W)
        out["tau_E_measured_s"] = tau
        out["H98y2"] = h_factor(tau, scal["tau_E_IPB98y2_s"])
    S = shaping_surface_area(r_m, a_m, kappa)
    out["surface_area_m2"] = S
    out["lh_threshold"] = lh_threshold_martin08(ne_line_avg_m3, bt_T, S, mass_amu)
    out["P_loss_over_P_LH"] = (p_loss_W / 1e6) / out["lh_threshold"]["P_LH_MW"]
    return out
