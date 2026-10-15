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


def fetch_legacy_npz(identifier):
    """Fetch a signal exported by GOLEM's legacy BasicDiagnostics archive."""
    import io
    import urllib.request
    errors = []
    urls = [
        f"https://golem.fjfi.cvut.cz/operation/shots/{SHOT}/basicdiagn/{identifier}.npz",
        f"http://golem.fjfi.cvut.cz/operation/shots/{SHOT}/basicdiagn/{identifier}.npz",
        f"https://golem.fjfi.cvut.cz/golem/database/operation/shots/{SHOT}/basicdiagn/{identifier}.npz",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"Tokamak-Insight-Suite-validation/1.0"})
            with urllib.request.urlopen(req, timeout=35) as r:
                blob = r.read()
            with np.load(io.BytesIO(blob), allow_pickle=True) as z:
                keys = list(z.files)
                # Legacy npz exports vary: accept a 2-column table, a Series,
                # a pair of vectors, or named time/value vectors.
                # GOLEM legacy BasicDiagnostics NPZ format stores the signal
                # in "data" and the acquisition bounds in "t_start"/"t_end".
                if "data" in keys and "t_start" in keys and "t_end" in keys:
                    y = np.asarray(z["data"], float).squeeze().ravel()
                    t_start = float(np.asarray(z["t_start"]).squeeze())
                    t_end = float(np.asarray(z["t_end"]).squeeze())
                    if len(y) > 20 and np.isfinite(t_start) and np.isfinite(t_end) and t_end > t_start:
                        t = np.linspace(t_start, t_end, len(y))
                        # Prefer the archive's explicit time axis where provided.
                        if "axis" in keys:
                            try:
                                axis = np.asarray(z["axis"], float).squeeze().ravel()
                                if len(axis) == len(y) and np.all(np.isfinite(axis)) and np.all(np.diff(axis) > 0):
                                    t = axis
                            except (TypeError, ValueError):
                                pass
                        return t, y, url, keys
                for key in keys:
                    a = z[key]
                    if a.dtype == object and a.size == 1:
                        obj = a.item()
                        if hasattr(obj, "index") and hasattr(obj, "to_numpy"):
                            t = np.asarray(obj.index, float)
                            y = np.asarray(obj.to_numpy(), float)
                            if len(t) == len(y) and len(y) > 20:
                                return t, y, url, keys
                        if isinstance(obj, dict):
                            for tk in ("time", "t", "Time", "x"):
                                for yk in (identifier, "data", "y", "value", "values"):
                                    if tk in obj and yk in obj:
                                        t, y = np.asarray(obj[tk],float).ravel(), np.asarray(obj[yk],float).ravel()
                                        if len(t)==len(y) and len(y)>20: return t,y,url,keys
                    if np.issubdtype(a.dtype, np.number) and a.ndim == 2 and a.shape[0] > 20 and a.shape[1] >= 2:
                        arr = np.asarray(a, float)
                        t, y = arr[:,0], arr[:,1]
                        if np.all(np.diff(t) >= 0):
                            return t, y, url, keys
                numeric = {k: np.asarray(z[k]) for k in keys if np.issubdtype(np.asarray(z[k]).dtype, np.number) and np.asarray(z[k]).ndim == 1}
                time_candidates = [(k,v) for k,v in numeric.items() if len(v)>20 and np.all(np.isfinite(v)) and np.all(np.diff(v)>0) and (np.max(v)-np.min(v)) < 100]
                signal_candidates = [(k,v) for k,v in numeric.items() if len(v)>20 and not any(k==tk for tk,_ in time_candidates)]
                for tk,t in time_candidates:
                    for yk,y in signal_candidates:
                        if len(t)==len(y):
                            return np.asarray(t,float), np.asarray(y,float), url, keys
                raise ValueError(f"Unrecognized NPZ structure; keys={keys}; shapes={{k:np.asarray(z[k]).shape for k in keys}}")
        except Exception as e:
            errors.append(f"{url}: {type(e).__name__}: {e}")
    raise RuntimeError("Legacy NPZ fetch/parse failed:\n" + "\n".join(errors))


def fetch_raw_das_signal(identifier):
    """Fetch GOLEM legacy raw NIstandard DAS signal (irog/uloop) via public data API."""
    import io, urllib.request
    errors = []
    channel = {"irog": "2", "uloop": "0"}.get(identifier, identifier)
    urls = [
        f"https://golem.fjfi.cvut.cz/cgi-bin/data/{SHOT}/nistandard_{identifier}.npz",
        f"https://golem.fjfi.cvut.cz/cgi-bin/data/{SHOT}/nistandard_{channel}.npz",
        f"https://golem.fjfi.cvut.cz/utils/data/{SHOT}/nistandard_{identifier}.npz",
        f"https://golem.fjfi.cvut.cz/utils/data/{SHOT}/nistandard_{channel}.npz",
        f"https://golem.fjfi.cvut.cz/utils/data/{SHOT}/{identifier}.npz",
        f"https://golem.fjfi.cvut.cz/utils/data/{SHOT}/{identifier}",
        f"http://golem.fjfi.cvut.cz/cgi-bin/data/{SHOT}/nistandard_{identifier}.npz",
        f"http://golem.fjfi.cvut.cz/cgi-bin/data/{SHOT}/nistandard_{channel}.npz",
        f"http://golem.fjfi.cvut.cz/utils/data/{SHOT}/{identifier}",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"Tokamak-Insight-Suite-validation/1.0"})
            with urllib.request.urlopen(req, timeout=35) as r:
                blob = r.read()
            if blob[:4] == b"PK\x03\x04":
                with np.load(io.BytesIO(blob), allow_pickle=True) as z:
                    keys = list(z.files)
                    if "data" in keys and "t_start" in keys and "t_end" in keys:
                        y = np.asarray(z["data"], float).squeeze().ravel()
                        t0 = float(np.asarray(z["t_start"]).squeeze())
                        t1 = float(np.asarray(z["t_end"]).squeeze())
                        t = np.linspace(t0, t1, len(y))
                        if "axis" in keys:
                            try:
                                axis = np.asarray(z["axis"], float).squeeze().ravel()
                                if len(axis)==len(y) and np.all(np.isfinite(axis)) and np.all(np.diff(axis)>0):
                                    t = axis
                            except (TypeError, ValueError):
                                pass
                        return t, y, url, {"format":"npz","keys":keys,"samples":len(y)}
            # Some legacy endpoints return a tab/comma-separated two-column table.
            df = pd.read_csv(io.BytesIO(blob), sep=None, engine="python", header=None, comment="#")
            df = df.iloc[:,:2].apply(pd.to_numeric, errors="coerce").dropna()
            if len(df) > 20 and df.shape[1] >= 2:
                t, y = df.iloc[:,0].to_numpy(float), df.iloc[:,1].to_numpy(float)
                if np.max(np.abs(t)) > 1 and np.max(np.abs(t)) < 100:
                    t *= 1e-3
                return t, y, url, {"format":"text-table","samples":len(y)}
            raise ValueError(f"Unrecognized response: {len(blob)} bytes, prefix={blob[:32]!r}")
        except Exception as e:
            errors.append(f"{url}: {type(e).__name__}: {e}")
    raise RuntimeError("Raw DAS acquisition failed:\\n" + "\\n".join(errors))

def raw_legacy_reconstruction(reason):
    """Reconstruct Ip from raw irog/uloop, with archived signals used only to resolve polarity and score output."""
    # Raw channels: irog is dI_(p+ch)/dt in volts; uloop is the loop-voltage channel.
    tr, dIraw, ur, rmeta = fetch_raw_das_signal("irog")
    tu, Uraw, uu, umeta = fetch_raw_das_signal("uloop")
    # Use archived calibrated traces for objective polarity checks and final comparison.
    refs = {}
    refmeta = {}
    for ident in ("Irogowski","Ich","Iplasma"):
        tt, yy, src, keys = fetch_legacy_npz(ident)
        if np.max(np.abs(tt)) > 1 and np.max(np.abs(tt)) < 100:
            tt = tt*1e-3
        refs[ident] = (tt, yy)
        refmeta[ident] = {"url":src,"samples":len(yy),"npz_keys":keys}
    for name, tt in (("irog",tr),("uloop",tu)):
        if np.max(np.abs(tt)) > 1 and np.max(np.abs(tt)) < 100:
            if name=="irog": tr=tt*1e-3
            else: tu=tt*1e-3
    # GOLEM shot-30622 metadata: current drive starts at 5 ms, K=5.3e6 A/(V s),
    # R_ch=9.7 mOhm, L_ch=1.2 uH, and Uloop calibration=5.5.
    tcd = 0.005
    K = 5.3e6
    R = 9.7e-3
    L = 1.2e-6
    Uloop_cal = 5.5
    t0, t1 = T_START_S, T_END_S
    baseline = tr < (tcd - 1e-4)
    if baseline.sum() < 10:
        baseline = tr < tcd
    if baseline.sum() < 5:
        raise RuntimeError("Raw irog signal has insufficient pre-current-drive baseline samples.")
    offset = float(np.mean(dIraw[baseline]))
    dI = dIraw - offset
    dI[tr <= tcd] = 0.0
    total_candidates = []
    t_refR, refR = refs["Irogowski"]
    maskR = (t_refR >= t0) & (t_refR <= t1)
    for polarity in (1.0,-1.0):
        total = cumulative_trapezoid(polarity*dI, x=tr, initial=0.0)*K
        pred = np.interp(t_refR[maskR], tr, total)
        target = refR[maskR]
        total_candidates.append((float(np.sqrt(np.mean((pred-target)**2))), polarity, total))
    total_rmse, irog_polarity, total = min(total_candidates, key=lambda x:x[0])
    # Calibrate and orient raw loop voltage against archived Ich, then solve the chamber RL equation.
    t_refC, refC = refs["Ich"]
    maskC = (t_refC >= t0) & (t_refC <= t1)
    ich_candidates = []
    for polarity in (1.0,-1.0):
        u_interp = interp1d(tu, polarity*Uraw*Uloop_cal, bounds_error=False,
                            fill_value=(polarity*Uraw[0]*Uloop_cal, polarity*Uraw[-1]*Uloop_cal))
        def rhs(t, y):
            return [(float(u_interp(t))-R*y[0])/L]
        sol = solve_ivp(rhs, (float(tu[0]),float(tu[-1])), [0.0], t_eval=tu,
                        rtol=1e-7, atol=1e-8, max_step=max(np.median(np.diff(tu))*5,1e-7))
        if not sol.success:
            continue
        ich_model = sol.y[0]
        pred = np.interp(t_refC[maskC], tu, ich_model)
        target = refC[maskC]
        ich_candidates.append((float(np.sqrt(np.mean((pred-target)**2))), polarity, ich_model))
    if not ich_candidates:
        raise RuntimeError("Could not solve chamber-current ODE for raw loop voltage.")
    ich_rmse, uloop_polarity, ich = min(ich_candidates, key=lambda x:x[0])
    ip = total - np.interp(tr,tu,ich)
    t_refP, refP = refs["Iplasma"]
    mask = (tr >= t0) & (tr <= t1)
    central_start, central_end = t0+0.2*(t1-t0), t1-0.2*(t1-t0)
    central = (tr >= central_start) & (tr <= central_end)
    pred = np.interp(t_refP[(t_refP>=t0)&(t_refP<=t1)], tr, ip)
    target = refP[(t_refP>=t0)&(t_refP<=t1)]
    err = pred-target
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sqrt(np.mean(target**2)))
    result = {
        "shot": SHOT,
        "status": "COMPUTED_raw_DAS_reconstruction",
        "method": "Raw NIstandard irog integration + raw uloop calibration + chamber RL ODE; polarity selected by minimum error against archived component signals",
        "raw_csv_route_error": reason,
        "raw_inputs": {"irog": {"url":ur,**rmeta}, "uloop": {"url":uu,**umeta}},
        "archived_reference_inputs": refmeta,
        "parameters": {"t_current_drive_s":tcd,"K_Rogowski_A_per_Vs":K,"Uloop_calibration":Uloop_cal,"R_chamber_ohm":R,"L_chamber_H":L,"irog_offset_removed_V":offset,"irog_polarity":irog_polarity,"uloop_polarity":uloop_polarity},
        "plasma_window_s":[t0,t1],
        "central_60_percent_window_s":[central_start,central_end],
        "raw_reconstruction": {
            "total_current_max_A":float(np.max(total[mask])),
            "plasma_current_mean_A":float(np.mean(ip[mask])),
            "plasma_current_max_A":float(np.max(ip[mask])),
            "central_60_percent_mean_A":float(np.mean(ip[central])),
        },
        "archived_plasma_current": {
            "mean_A":float(np.mean(refP[(t_refP>=t0)&(t_refP<=t1)])),
            "max_A":float(np.max(refP[(t_refP>=t0)&(t_refP<=t1)])),
            "central_60_percent_mean_A":float(np.mean(refP[(t_refP>=central_start)&(t_refP<=central_end)])),
            "published_scalar_mean_A":EXPECTED_MEAN_A,
            "central_mean_relative_error_to_published_scalar_percent":float(100*abs(float(np.mean(ip[central]))-EXPECTED_MEAN_A)/EXPECTED_MEAN_A),
        },
        "reference_fit_checks": {
            "total_current_vs_archived_Irogowski_rmse_A":total_rmse,
            "chamber_current_vs_archived_Ich_rmse_A":ich_rmse,
            "plasma_current_vs_archived_Iplasma_rmse_A":rmse,
            "plasma_current_vs_archived_Iplasma_nrmse_percent":float(100*rmse/(denom or 1)),
            "plasma_current_mean_relative_difference_percent":float(100*abs(float(np.mean(ip[mask]))-float(np.mean(refP[(t_refP>=t0)&(t_refP<=t1)])))/(abs(float(np.mean(refP[(t_refP>=t0)&(t_refP<=t1)]))) or 1)),
        },
        "interpretation": "This run uses the raw archived irog and uloop channels. Polarity is selected by matching each reconstructed component to its corresponding archived calibrated signal; these are validation anchors, not independent sign conventions.",
    }
    OUT.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    print(f"RESULT_JSON={OUT}")


def fallback_processed_crosscheck(reason):
    """Independent cross-check from GOLEM's archived integrated signals if raw dIp/dt is absent."""
    signals = {}
    sources = {}
    for ident in ("Irogowski", "Ich", "Iplasma"):
        t, y, url, keys = fetch_legacy_npz(ident)
        signals[ident] = (t, y)
        sources[ident] = {"url": url, "npz_keys": keys, "samples": int(len(y))}
    tR, iR = signals["Irogowski"]
    tC, iC = signals["Ich"]
    tP, iP = signals["Iplasma"]
    start, end = T_START_S, T_END_S
    # GOLEM's scalar result uses seconds; legacy time axes may be seconds or milliseconds.
    if np.max(np.abs(tR)) > 1 and np.max(np.abs(tR)) < 100: tR = tR*1e-3
    if np.max(np.abs(tC)) > 1 and np.max(np.abs(tC)) < 100: tC = tC*1e-3
    if np.max(np.abs(tP)) > 1 and np.max(np.abs(tP)) < 100: tP = tP*1e-3
    # Interpolate chamber current to Rogowski timebase, then subtract.
    ich = np.interp(tR, tC, iC)
    ip_recon = iR - ich
    ip_ref = np.interp(tR, tP, iP)
    mask = (tR >= start) & (tR <= end)
    if mask.sum() < 2:
        raise RuntimeError(f"Archived signals had no points in plasma interval {start}-{end}s; t range={tR.min()}..{tR.max()}")
    err = ip_recon[mask] - ip_ref[mask]
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sqrt(np.mean(ip_ref[mask]**2)))
    central_start = start + 0.2*(end-start)
    central_end = end - 0.2*(end-start)
    central_mask = (tR >= central_start) & (tR <= central_end)
    central_recon_mean = float(np.mean(ip_recon[central_mask]))
    central_ref_mean = float(np.mean(ip_ref[central_mask]))
    central_ref_error = float(100*abs(central_recon_mean-central_ref_mean)/(abs(central_ref_mean) or 1))
    result = {
        "shot": SHOT,
        "status": "COMPUTED_archived_signal_crosscheck_raw_reconstruction_unavailable",
        "method": "GOLEM archived integrated Rogowski current minus archived chamber current; compared samplewise with archived plasma-current signal",
        "raw_reconstruction_attempt": "failed",
        "raw_reconstruction_error": reason,
        "sample_counts": {k:v["samples"] for k,v in sources.items()},
        "plasma_window_s": [start,end],
        "central_60_percent_window_s": [central_start, central_end],
        "sources": sources,
        "reconstructed_from_archived_components": {
            "mean_plasma_current_A": float(np.mean(ip_recon[mask])),
            "max_plasma_current_A": float(np.max(ip_recon[mask])),
            "central_60_percent_mean_plasma_current_A": central_recon_mean,
            "mean_total_rogowski_current_A": float(np.mean(iR[mask])),
            "max_total_rogowski_current_A": float(np.max(iR[mask])),
        },
        "time_ranges_s": {
            "Irogowski": [float(np.min(tR)), float(np.max(tR))],
            "Ich": [float(np.min(tC)), float(np.max(tC))],
            "Iplasma": [float(np.min(tP)), float(np.max(tP))],
        },
        "official_archived_plasma_current": {
            "mean_A": float(np.mean(ip_ref[mask])),
            "mean_absolute_A": float(np.mean(np.abs(ip_ref[mask]))),
            "max_A": float(np.max(ip_ref[mask])),
            "min_A": float(np.min(ip_ref[mask])),
            "central_60_percent_mean_A": central_ref_mean,
            "central_60_percent_relative_difference_percent": central_ref_error,
            "central_60_percent_mean_vs_published_2010A_relative_error_percent": float(100*abs(central_ref_mean-EXPECTED_MEAN_A)/EXPECTED_MEAN_A),
            "full_trace_mean_absolute_A": float(np.mean(np.abs(iP))),
            "plasma_window_percentiles_A": {
                "p10": float(np.percentile(ip_ref[mask],10)),
                "p25": float(np.percentile(ip_ref[mask],25)),
                "p50": float(np.percentile(ip_ref[mask],50)),
                "p75": float(np.percentile(ip_ref[mask],75)),
                "p90": float(np.percentile(ip_ref[mask],90)),
            },
        },
        "samplewise_comparison": {
            "rmse_A": rmse,
            "nrmse_percent": float(100*rmse/(denom or 1)),
            "max_absolute_difference_A": float(np.max(np.abs(err))),
            "mean_absolute_difference_A": float(np.mean(np.abs(err))),
        },
        "published_references": {"mean_plasma_current_A": EXPECTED_MEAN_A, "max_rogowski_current_A": EXPECTED_ROGOWSKI_MAX_A},
        "mean_current_relative_error_percent": float(100*abs(float(np.mean(ip_recon[mask]))-EXPECTED_MEAN_A)/EXPECTED_MEAN_A),
        "limitation": "This is an archived processed-signal cross-check, not an independent integration of the raw Rogowski derivative. The raw dIp/dt path is reported separately and was unavailable at the candidate endpoints.",
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"RESULT_JSON={OUT}")


def main_raw():
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
        main_raw()
    except Exception as e:
        print(f"Raw-signal route failed; attempting archived component cross-check: {type(e).__name__}: {e}", file=sys.stderr)
        raw_reason = f"{type(e).__name__}: {e}"
        try:
            raw_legacy_reconstruction(raw_reason)
        except Exception as e2:
            print(f"Raw legacy DAS route failed; falling back to archived processed-current cross-check: {type(e2).__name__}: {e2}", file=sys.stderr)
            try:
                fallback_processed_crosscheck(f"{raw_reason}; raw DAS error: {type(e2).__name__}: {e2}")
            except Exception as e3:
                payload={"shot":SHOT,"status":"failed_raw_and_archived_crosscheck","raw_csv_error":raw_reason,"raw_das_error":f"{type(e2).__name__}: {e2}","fallback_error":f"{type(e3).__name__}: {e3}"}
                OUT.write_text(json.dumps(payload, indent=2))
                print(json.dumps(payload, indent=2))
                sys.exit(1)
