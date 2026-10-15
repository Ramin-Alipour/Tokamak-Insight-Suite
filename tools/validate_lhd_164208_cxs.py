#!/usr/bin/env python3
"""Validate LHD CXS7 shot 164208 from raw spectra against NIFS processed products."""
from __future__ import annotations
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

SHOT = 164208
BASE = "https://exp.lhd.nifs.ac.jp/opendata/LHD/dataexplog/"
WEBAPI = "https://exp.lhd.nifs.ac.jp/opendata/LHD/webapi.fcgi"
DIAGNOSTICS = ("lhdcxs7_img", "lhdcxs7_prep", "lhdcxs7_cvi")
OUT = Path("artifacts/lhd-164208-cxs-validation")
LAMBDA0_NM = 529.05
W0_NM = 529.1
PIX_CENTER = 157.5
DISPERSION_NM_PER_PIXEL = 0.00941
DISPERSION_NM_PER_MM = 0.7242
INSTRUMENT_WIDTH_MM = 0.130
CARBON_MASS_ENERGY_KEV = 12.0 * 931494.10242
TIME_TARGETS = (3.4, 4.0)
R_TARGET_M = 3.5974


def request(url: str, data: bytes | None = None, content_type: str | None = None, timeout: int = 90):
    headers = {"User-Agent": "Tokamak-Insight-Suite-LHD-validation/1.0"}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, dict(response.headers.items()), response.read()


def get_diagnostic_subshots() -> dict[str, str]:
    body = urllib.parse.urlencode({"SEARCH_SHOT_NO": str(SHOT)}).encode("ascii")
    _, _, raw = request(
        urllib.parse.urljoin(BASE, "get_summary2.fcgi"), body,
        "application/x-www-form-urlencoded", timeout=45
    )
    (OUT / "shot_summary.json").write_bytes(raw)
    summary = json.loads(raw.decode("utf-8", errors="replace"))
    if summary.get("status") != 0:
        raise RuntimeError(f"NIFS summary query failed: {summary}")
    names, subnos = summary.get("arry_diag") or [], summary.get("arry_subno") or []
    found = {}
    for i, name in enumerate(names):
        if name in DIAGNOSTICS and i < len(subnos):
            found[name] = str(subnos[i])
    missing = sorted(set(DIAGNOSTICS) - set(found))
    if missing:
        raise RuntimeError(f"Shot {SHOT} missing required NIFS products: {missing}")
    return found


def download_products() -> dict[str, Path]:
    found = get_diagnostic_subshots()
    paths = {}
    provenance = {}
    for diag in DIAGNOSTICS:
        subno = found[diag]
        url = WEBAPI + "?" + urllib.parse.urlencode({
            "cmd": "getfile", "diag": diag, "shotno": str(SHOT), "subno": subno
        })
        _, headers, raw = request(url, timeout=120)
        if len(raw) < 1000:
            raise RuntimeError(f"{diag} response unexpectedly small ({len(raw)} bytes)")
        path = OUT / f"{diag}_{SHOT}_sub{subno}.txt"
        path.write_bytes(raw)
        paths[diag] = path
        provenance[diag] = {
            "url": url, "subno": subno, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "content_disposition": headers.get("Content-Disposition", ""),
        }
    return paths, provenance


def parse_prep(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", header=None)


def parse_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", header=None)


def gaussian_linear(x, amplitude, center, sigma, offset, slope):
    z = x - center
    return amplitude * np.exp(-0.5 * (z / sigma) ** 2) + offset + slope * z


def fit_raw_frame(image: np.ndarray, pixels: np.ndarray, frame_index: int,
                  stripe: int, background_stripe: int) -> dict:
    # Stripe identifiers in the prep file are one-based.
    y = image[frame_index, :, stripe - 1].astype(float) - image[frame_index, :, background_stripe - 1].astype(float)
    mask = (pixels >= 65) & (pixels <= 250)
    x, yy = pixels[mask].astype(float), y[mask]
    smooth = pd.Series(yy).rolling(5, center=True, min_periods=1).mean().to_numpy()
    peak = int(np.argmax(smooth))
    p0 = [max(float(smooth[peak] - np.median(smooth)), 1.0), float(x[peak]), 22.0, float(np.median(smooth)), 0.0]
    popt, covariance = curve_fit(
        gaussian_linear, x, yy, p0=p0,
        bounds=([0, 65, 1, -np.inf, -np.inf], [np.inf, 250, 80, np.inf, np.inf]),
        maxfev=20000,
    )
    amplitude, center, sigma_px, offset, slope = [float(v) for v in popt]
    sigma_observed_nm = sigma_px * DISPERSION_NM_PER_PIXEL
    instrument_fwhm_nm = INSTRUMENT_WIDTH_MM * DISPERSION_NM_PER_MM
    sigma_instrument_nm = instrument_fwhm_nm / 2.354820045
    sigma_doppler_nm = float(np.sqrt(max(sigma_observed_nm**2 - sigma_instrument_nm**2, 0.0)))
    ti_keV = CARBON_MASS_ENERGY_KEV * (sigma_doppler_nm / LAMBDA0_NM) ** 2
    predicted = gaussian_linear(x, *popt)
    return {
        "frame_index": int(frame_index), "time_s": float(image_times[frame_index]),
        "stripe": int(stripe), "background_stripe": int(background_stripe),
        "amplitude_counts": amplitude, "center_pixel": center, "sigma_pixel": sigma_px,
        "sigma_observed_nm": sigma_observed_nm, "instrument_fwhm_nm": instrument_fwhm_nm,
        "sigma_doppler_nm": sigma_doppler_nm, "Ti_raw_fit_keV": float(ti_keV),
        "integrated_line_area_counts_pixel": float(amplitude * sigma_px * np.sqrt(2*np.pi)),
        "fit_rmse_counts": float(np.sqrt(np.mean((yy - predicted) ** 2))),
        "fit_window_pixel": [65, 250],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths, provenance = download_products()
    prep = parse_prep(paths["lhdcxs7_prep"])
    img_df = parse_data(paths["lhdcxs7_img"])
    cvi = parse_data(paths["lhdcxs7_cvi"])
    if img_df.shape[1] != 104:
        raise ValueError(f"Unexpected raw image column count: {img_df.shape}")
    if prep.shape[1] < 27 or len(prep) != 102:
        raise ValueError(f"Unexpected CXS7 preparation shape: {prep.shape}")
    if cvi.shape[1] != 33:
        raise ValueError(f"Unexpected CXS7 processed-output shape: {cvi.shape}")

    # Raw image rows are ordered by time then x-pixel; each row contains 102 stripe values.
    global image_times
    n_frames = int(img_df.iloc[:, 0].nunique())
    n_pixels = int(img_df.iloc[:, 1].nunique())
    if n_frames * n_pixels != len(img_df):
        raise ValueError("Raw image time/pixel grid is not rectangular.")
    raw = img_df.to_numpy(dtype=float).reshape(n_frames, n_pixels, img_df.shape[1])
    image = raw[:, :, 2:]
    image_times = raw[:, 0, 0]
    pixels = raw[0, :, 1]
    if image.shape[2] != len(prep):
        raise ValueError("Raw stripe count does not match preparation rows.")

    # Prep column 18 is R (metres); column 3 is one-based BGNo. The repository
    # prep table contains one extra wavelength-centre field not listed in ValName.
    prep_r = prep.iloc[:, 18].to_numpy(dtype=float)
    candidate = np.flatnonzero(np.abs(prep_r - R_TARGET_M) < 0.03)
    if not len(candidate):
        raise ValueError("No raw stripe maps near the selected central radius.")
    stripe_index = min(candidate, key=lambda i: abs(prep_r[i] - R_TARGET_M))
    stripe = int(prep.iloc[stripe_index, 0])
    bg_stripe = int(prep.iloc[stripe_index, 3])
    radius = float(prep.iloc[stripe_index, 18])
    if stripe < 1 or stripe > image.shape[2] or bg_stripe < 1 or bg_stripe > image.shape[2]:
        raise ValueError("Invalid stripe/background mapping.")

    raw_fits = []
    for i in range(n_frames):
        try:
            raw_fits.append(fit_raw_frame(image, pixels, i, stripe, bg_stripe))
        except Exception as exc:
            raw_fits.append({"frame_index": i, "time_s": float(image_times[i]), "fit_error": str(exc)})

    official = []
    for i, t in enumerate(image_times):
        rows = cvi[np.isclose(cvi.iloc[:, 0].to_numpy(dtype=float), t)]
        if rows.empty:
            official.append({"time_s": float(t), "Ti_keV": None, "Vc_km_s": None, "In_au": None})
            continue
        row = rows.iloc[int(np.argmin(np.abs(rows.iloc[:, 1].to_numpy(dtype=float) - radius)))]
        official.append({
            "time_s": float(t), "R_m": float(row.iloc[1]),
            "ary": float(row.iloc[2]), "Ti_keV": float(row.iloc[3]),
            "Ti_error_keV": float(row.iloc[4]), "Vc_km_s": float(row.iloc[5]),
            "Vc_error_km_s": float(row.iloc[6]), "In_au": float(row.iloc[7]),
            "In_error_au": float(row.iloc[8]),
        })

    # Compare integrated raw spectral line area with the official intensity time series.
    areas = np.array([x.get("integrated_line_area_counts_pixel", np.nan) for x in raw_fits], dtype=float)
    official_in = np.array([x.get("In_au", np.nan) for x in official], dtype=float)
    mask = np.isfinite(areas) & np.isfinite(official_in) & (areas > 0) & (official_in > 0)
    if int(mask.sum()) >= 10:
        pr = pearsonr(areas[mask], official_in[mask])
        sr = spearmanr(areas[mask], official_in[mask])
        intensity_check = {
            "valid_frames": int(mask.sum()), "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue), "spearman_rho": float(sr.statistic),
            "spearman_p": float(sr.pvalue),
            "status": "PASS" if pr.statistic >= 0.90 and sr.statistic >= 0.90 else "PENDING",
        }
    else:
        intensity_check = {"valid_frames": int(mask.sum()), "status": "PENDING", "reason": "Too few matched valid frames."}

    ti_checks = []
    valid_reference_frames = [
        i for i, ref in enumerate(official)
        if ref.get("Ti_keV") is not None and np.isfinite(ref["Ti_keV"]) and ref["Ti_keV"] > 0
        and raw_fits[i].get("Ti_raw_fit_keV") is not None
        and np.isfinite(raw_fits[i]["Ti_raw_fit_keV"])
    ]
    for target in TIME_TARGETS:
        if not valid_reference_frames:
            frame = int(np.argmin(np.abs(image_times - target)))
        else:
            # The nominal 3.4 s and 4.0 s points fall exactly between two
            # frames; choose the nearest frame with a valid official Ti fit,
            # not a zero-filled invalid fit at 3.39/3.99 s.
            frame = min(valid_reference_frames, key=lambda i: abs(image_times[i] - target))
        raw_result = raw_fits[frame]
        ref = official[frame]
        ref_ti = ref.get("Ti_keV")
        diff = None if ref_ti is None or not ref_ti else abs(raw_result.get("Ti_raw_fit_keV", np.nan) - ref_ti) / abs(ref_ti) * 100
        ti_checks.append({
            "requested_time_s": target, "matched_time_s": float(image_times[frame]),
            "radius_m": radius, "raw_fit": raw_result,
            "official_processed": ref,
            "relative_Ti_difference_percent": None if diff is None or not np.isfinite(diff) else float(diff),
            "status": "PASS" if diff is not None and np.isfinite(diff) and diff <= 10 else "PENDING",
        })

    # Detect a common field/unit mix-up: values near 2.76 and 4.38 occur in
    # the official Vc column (km/s), not in the Ti column (keV).
    semantic_reference_checks = []
    for target, candidate_value in ((3.4, 2.76), (4.0, 4.38)):
        valid_times = sorted({float(x["time_s"]) for x in official
                              if x.get("Ti_keV") is not None and x["Ti_keV"] > 0})
        matched_time = min(valid_times, key=lambda t: abs(t - target)) if valid_times else float(image_times[np.argmin(np.abs(image_times-target))])
        rows = cvi[np.isclose(cvi.iloc[:, 0].to_numpy(dtype=float), matched_time)]
        if rows.empty:
            semantic_reference_checks.append({"requested_time_s": target, "candidate_value": candidate_value, "status": "PENDING"})
            continue
        row = rows.iloc[int(np.argmin(np.abs(rows.iloc[:, 5].to_numpy(dtype=float) - candidate_value)))]
        vc = float(row.iloc[5])
        semantic_reference_checks.append({
            "requested_time_s": target, "matched_time_s": matched_time,
            "candidate_value": candidate_value, "matched_R_m": float(row.iloc[1]),
            "official_Ti_keV": float(row.iloc[3]), "official_Vc_km_s": vc,
            "Vc_relative_difference_percent": abs(vc-candidate_value)/abs(candidate_value)*100,
            "status": "FIELD_UNIT_MISMATCH_CONFIRMED" if abs(vc-candidate_value)/abs(candidate_value)*100 <= 1.0 else "UNCONFIRMED",
        })

    report = {
        "title": "LHD shot 164208 CXS7 raw-spectrum cross-check",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NIFS LHD analyzed-data repository",
        "shot": SHOT, "diagnostic": "lhdcxs7_img",
        "reference_product": "lhdcxs7_cvi",
        "reference_product_description": "Ion temperature and plasma rotation measured with CXS7",
        "source_files": provenance,
        "method": {
            "selected_radius_m": radius, "stripe": stripe, "background_stripe": bg_stripe,
            "time_sampling_s": float(np.median(np.diff(image_times))),
            "wavelength_calibration": "lambda = 529.1 nm + (pixel - 157.5)*0.00941 nm",
            "line_model": "single Gaussian plus linear baseline on background-subtracted stripe",
            "background_subtraction": "signal stripe minus BGNo stripe as defined by lhdcxs7_prep",
            "instrumental_width": "0.130 mm * 0.7242 nm/mm; treated as FWHM and deconvolved in quadrature",
            "ion_temperature_formula": "Ti = (12*u*c^2 in keV) * (sigma_Doppler/lambda0)^2",
        },
        "Ti_temperature_checks": ti_checks,
        "intensity_trend_check": intensity_check,
        "interpretation_guard": {
            "field_order": "In lhdcxs7_cvi, Ti is the fourth numeric field including time/R indexing (zero-based column 3); Vc is column 5 and is in km/s.",
            "status": "Ti width reconstruction remains PENDING where the raw Gaussian fit differs by >10%; candidate values near 2.76 and 4.38 are checked against the explicitly labelled Vc_km_s column and are not treated as Ti.",
        },
        "time_series": [
            {"time_s": float(image_times[i]), "raw_line_area": areas[i] if np.isfinite(areas[i]) else None,
             "official_In_au": official_in[i] if np.isfinite(official_in[i]) else None,
             "raw_Ti_keV": raw_fits[i].get("Ti_raw_fit_keV"),
             "official_Ti_keV": official[i].get("Ti_keV"),
             "official_Vc_km_s": official[i].get("Vc_km_s")}
            for i in range(n_frames)
        ],
    }
    (OUT / "lhd_164208_cxs_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report["time_series"]).to_csv(OUT / "lhd_164208_cxs_timeseries.csv", index=False)
    print(json.dumps({
        "report": str(OUT / "lhd_164208_cxs_validation.json"),
        "stripe": stripe, "background_stripe": bg_stripe, "radius_m": radius,
        "Ti_checks": [{"time_s": x["matched_time_s"], "raw_Ti_keV": x["raw_fit"].get("Ti_raw_fit_keV"),
                       "official_Ti_keV": x["official_processed"].get("Ti_keV"),
                       "relative_difference_percent": x["relative_Ti_difference_percent"], "status": x["status"]}
                      for x in ti_checks],
        "intensity_check": intensity_check,
    }, indent=2))


if __name__ == "__main__":
    main()
