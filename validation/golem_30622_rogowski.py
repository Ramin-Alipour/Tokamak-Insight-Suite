#!/usr/bin/env python3
"""Independent GOLEM shot 30622 Rogowski-to-plasma-current reconstruction.

Implements the public GOLEM BasicDiagnostics method:
  Ip+ch = integral(dIp_dt) * K_RogowskiCoil
  dIch/dt = (U_loop - R_chamber*Ich)/L_chamber
  Ip = Ip+ch - Ich
Then compares against the archived processed Ip.csv and IplaMean=2010 A.
"""
from __future__ import annotations
import json, math, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d

SHOT = 30622
BASES = ["https://golem.fjfi.cvut.cz", "http://golem.fjfi.cvut.cz"]
OUT = Path("golem_30622_rogowski_result.json")
EXPECTED_MEAN_A = 2010.0
EXPECTED_ROGOWSKI_MAX_A = 3330.0
T_START_S, T_END_S = 0.00709, 0.01570

def get_text(paths, timeout=25):
    import urllib.request
    errors=[]
    for path in paths:
        for base in BASES:
            url=base + path
            try:
                req=urllib.request.Request(url, headers={"User-Agent":"Tokamak-Insight-Suite-validation/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data=r.read().decode("utf-8", errors="replace").strip()
                if data and not data.lower().startswith("<!doctype html") and "404 Not Found" not in data:
                    return data, url
                errors.append(f"{url}: empty/HTML response")
            except Exception as e:
                errors.append(f"{url}: {type(e).__name__}: {e}")
    raise RuntimeError("Could not fetch any candidate URL:\n" + "\n".join(errors[-8:]))

def read_scalar(candidates):
    text, url = get_text(candidates)
    m = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not m: raise ValueError(f"No numeric scalar at {url}: {text[:100]!r}")
    return float(m.group()), url

def read_signal(candidates):
    import io
    raw, url = get_text(candidates, timeout=40)
    # GOLEM CSVs are normally two columns (time,value), often without headers.
    df = pd.read_csv(io.StringIO(raw), sep=None, engine="python", header=None, comment="#")
    if df.shape[1] < 2:
        df = pd.read_csv(io.StringIO(raw), delim_whitespace=True, header=None, comment="#")
    df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 100: raise ValueError(f"Too few numeric samples at {url}: {len(df)}")
    df = df.sort_values(df.columns[0]).drop_duplicates(df.columns[0])
    t = df.iloc[:,0].to_numpy(float)
    y = df.iloc[:,1].to_numpy(float)
    # GOLEM signal time axes are seconds; handle millisecond axes if clearly so.
    if np.nanmax(np.abs(t)) > 1 and np.nanmax(np.abs(t)) < 100:
        t = t * 1e-3
    return t, y, url

def main():
    notes=[]
    # Standard GOLEM raw diagnostic identifiers (new pipeline); fall back to the
    # older raw Rogowski/loop-voltage naming where available.
    dIp_t, dIp, dIp_url = read_signal([
        f"/shots/{SHOT}/Diagnostics/PlasmaDetection/dIp_dt.csv",
        f"/shots/{SHOT}/Diagnostics/PlasmaDetection/U_RogCoil.csv",
        f"/shots/{SHOT}/Diagnostics/BasicDiagnostics/dIp_dt.csv",
    ])
    ul_t, ul, ul_url = read_signal([
        f"/shots/{SHOT}/Diagnostics/PlasmaDetection/V_loop.csv",
        f"/shots/{SHOT}/Diagnostics/PlasmaDetection/U_Loop.csv",
        f"/shots/{SHOT}/Diagnostics/BasicDiagnostics/Results/U_loop.csv",
        f"/shots/{SHOT}/Diagnostics/BasicDiagnostics/Results/U_Loop.csv",
    ])
    K, k_url = read_scalar([
        f"/shots/{SHOT}/Production/Parameters/SystemParameters/K_RogowskiCoil",
        f"/shots/{SHOT}/Production/Parameters/K_RogowskiCoil",
    ])
    R, r_url = read_scalar([
        f"/shots/{SHOT}/Production/Parameters/SystemParameters/R_chamber",
        f"/shots/{SHOT}/Production/Parameters/SystemParameters/ChamberResistance",
    ])
    L, l_url = read_scalar([
        f"/shots/{SHOT}/Production/Parameters/SystemParameters/L_chamber",
        f"/shots/{SHOT}/Production/Parameters/SystemParameters/ChamberInduktance",
    ])
    # Prefer archived plasma window; official Data.php publishes these exact scalars.
    try:
        t0, t0_url = read_scalar([
            f"/shots/{SHOT}/Diagnostics/PlasmaDetection/Results/t_plasma_start",
            f"/shots/{SHOT}/Diagnostics/PlasmaDetection/Results/PlasmaStart",
        ])
        t1, t1_url = read_scalar([
            f"/shots/{SHOT}/Diagnostics/PlasmaDetection/Results/t_plasma_end",
            f"/shots/{SHOT}/Diagnostics/PlasmaDetection/Results/PlasmaEnd",
        ])
        if t0 > 1: t0 *= 1e-3
        if t1 > 1: t1 *= 1e-3
    except Exception as e:
        notes.append(f"Used published Data.php plasma interval because scalar endpoint retrieval failed: {e}")
        t0, t1 = T_START_S, T_END_S
        t0_url = t1_url = "Data.php published scalars"

    # Offset correction: use the pre-current-drive baseline when present.
    try:
        tcd_raw, tcd_url = read_scalar([
            f"/shots/{SHOT}/Production/Parameters/Tcd",
            f"/shots/{SHOT}/Production/Parameters/t_cd",
        ])
        tcd = tcd_raw * (1e-6 if tcd_raw > 1 else 1.0)
        notes.append(f"Current-drive time read as {tcd_raw:g} from {tcd_url}; interpreted as {tcd:g} s.")
    except Exception as e:
        tcd = max(0.0, t0 - 0.002)
        notes.append(f"Could not fetch Tcd; used t0-2 ms for baseline/cutoff: {e}")
    baseline_end = min(tcd, float(t0)) - 1e-4
    baseline = dIp_t < baseline_end
    if baseline.sum() < 20:
        baseline = dIp_t < max(0.004, float(t0)-0.0002)
    if baseline.sum() < 5:
        raise RuntimeError("Insufficient pre-discharge samples for Rogowski offset correction.")
    polarity = 1.0
    try:
        pol, pol_url = get_text([
            f"/shots/{SHOT}/Production/Parameters/CD_orientation",
            f"/shots/{SHOT}/Production/Parameters/SystemParameters/CD_orientation",
        ])
        if pol.strip().upper() != "CW": polarity = -1.0
        notes.append(f"Polarity={pol.strip()} from {pol_url}.")
    except Exception as e:
        notes.append(f"Could not read CD_orientation; both signal signs will be assessed: {e}")

    dIp = polarity * dIp
    dIp = np.where(np.isfinite(dIp), dIp, np.nan)
    dIp = pd.Series(dIp).interpolate(limit_direction="both").to_numpy()
    offset = float(np.mean(dIp[baseline]))
    dIp -= offset
    dIp[dIp_t <= tcd] = 0.0
    total = cumulative_trapezoid(dIp, x=dIp_t, initial=0.0) * K

    # Chamber-current ODE on loop-voltage timebase, then interpolate to Rogowski timebase.
    ul = np.asarray(ul, float)
    ul = np.where(np.isfinite(ul), ul, np.nan)
    ul = pd.Series(ul).interpolate(limit_direction="both").to_numpy()
    # Current polarity correction consistent with the Rogowski signal.
    ul_interp = interp1d(ul_t, ul, bounds_error=False, fill_value=(ul[0], ul[-1]))
    def rhs(t, y):
        return [(float(ul_interp(t)) - R*y[0]) / L]
    sol = solve_ivp(rhs, (float(ul_t[0]), float(ul_t[-1])), [0.0],
                    t_eval=ul_t, rtol=1e-7, atol=1e-8, max_step=max(np.median(np.diff(ul_t))*5, 1e-7))
    if not sol.success: raise RuntimeError("Chamber-current ODE failed: " + sol.message)
    ich_interp = interp1d(ul_t, sol.y[0], bounds_error=False, fill_value="extrapolate")
    ich = ich_interp(dIp_t)
    ip = total - ich

    # Compare reconstructed plasma current with official processed Ip.csv when accessible.
    ref = None
    ref_error = None
    try:
        rt, ry, ref_url = read_signal([
            f"/shots/{SHOT}/Diagnostics/BasicDiagnostics/Results/Ip.csv",
            f"/shots/{SHOT}/Diagnostics/BasicDiagnostics/Results/Iplasma.csv",
        ])
        ref = {"url": ref_url, "samples": int(len(ry))}
        refmask = (rt >= t0) & (rt <= t1)
        if refmask.sum() > 0:
            interp_ip = interp1d(dIp_t, ip, bounds_error=False, fill_value="extrapolate")(rt[refmask])
            rv = ry[refmask]
            denom = np.linalg.norm(rv)
            ref["reconstruction_vs_archived_Ip_rmse_A"] = float(np.sqrt(np.mean((interp_ip-rv)**2)))
            ref["reconstruction_vs_archived_Ip_nrmse_percent"] = float(100*np.sqrt(np.mean((interp_ip-rv)**2))/(np.sqrt(np.mean(rv**2)) or 1))
            ref["archived_Ip_mean_A"] = float(np.mean(rv))
            ref["reconstructed_mean_on_reference_timebase_A"] = float(np.mean(interp_ip))
            ref["archived_Ip_max_A"] = float(np.max(rv))
        else:
            ref_error = "Ip.csv has no points within the published plasma interval."
    except Exception as e:
        ref_error = str(e)

    mask = (dIp_t >= t0) & (dIp_t <= t1)
    if mask.sum() < 2: raise RuntimeError(f"No reconstructed samples in plasma window {t0}..{t1}s")
    mean_ip = float(np.mean(ip[mask]))
    max_ip = float(np.max(ip[mask]))
    max_total = float(np.max(total[mask]))
    result = {
        "shot": SHOT,
        "method": "GOLEM standard Rogowski integration + chamber-current ODE correction",
        "status": "computed_from_raw_signals",
        "sample_counts": {"dIp_dt": int(len(dIp_t)), "loop_voltage": int(len(ul_t)), "plasma_window": int(mask.sum())},
        "plasma_window_s": [float(t0), float(t1)],
        "calibration": {"K_RogowskiCoil_A_per_Vs": K, "R_chamber_ohm": R, "L_chamber_H": L},
        "sources": {"dIp_dt": dIp_url, "loop_voltage": ul_url, "K": k_url, "R": r_url, "L": l_url, "plasma_start": t0_url, "plasma_end": t1_url},
        "offset_removed_dIp_dt_V": offset,
        "reconstructed": {"mean_plasma_current_A": mean_ip, "max_plasma_current_A": max_ip, "max_total_plasma_plus_chamber_current_A": max_total},
        "published_references": {"mean_plasma_current_A": EXPECTED_MEAN_A, "max_rogowski_current_A": EXPECTED_ROGOWSKI_MAX_A},
        "mean_current_relative_error_percent": float(100*abs(mean_ip-EXPECTED_MEAN_A)/EXPECTED_MEAN_A),
        "archived_Ip_comparison": ref,
        "archived_Ip_fetch_error": ref_error,
        "notes": notes,
    }
    # The raw signal's max may be total current; report separately rather than conflate it with plasma current.
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"RESULT_JSON={OUT}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        payload={"shot":SHOT,"status":"failed_to_fetch_or_compute","error":f"{type(e).__name__}: {e}"}
        OUT.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        sys.exit(1)
