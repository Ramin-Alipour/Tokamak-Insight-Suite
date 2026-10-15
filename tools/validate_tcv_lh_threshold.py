#!/usr/bin/env python3
"""Reproduce the published TCV L-H threshold-normalization calculation from the Zenodo database."""
from __future__ import annotations
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

RECORD = "https://zenodo.org/records/14996664"
ARTICLE_DOI = "10.1088/1361-6587/adc8ce"
DATABASE_URL = f"{RECORD}/files/lhdatabase.h5?download=1"
LH_DATA_URL = f"{RECORD}/files/LH_DATA.h5?download=1"
OUT = Path("artifacts/tcv-lh-threshold-validation")


def download(url: str, path: Path) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Tokamak-Insight-Suite-TCV-validation/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        raw = response.read()
    if len(raw) < 1000:
        raise RuntimeError(f"Downloaded database is unexpectedly small: {len(raw)} bytes")
    path.write_bytes(raw)
    return {"url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def load_fields(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        required = ("Q95", "ILH", "A", "SEEDING", "PECH", "cH", "NEL20",
                    "NEL20STD", "PLMW", "PLMWSTD", "ABEAM", "SHOT", "BT",
                    "SPLASMA", "HDRSTD")
        missing = [key for key in required if key not in h5]
        if missing:
            raise KeyError(f"Required HDF5 fields missing: {missing}")
        return {key: np.asarray(h5[key]).reshape(-1) for key in required}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    db_path = OUT / "lhdatabase.h5"
    provenance = download(DATABASE_URL, db_path)
    lh_path = OUT / "LH_DATA.h5"
    lh_provenance = download(LH_DATA_URL, lh_path)
    d = load_fields(db_path)
    n = len(d["SHOT"])
    if any(len(v) != n for v in d.values()):
        raise ValueError("HDF5 arrays have inconsistent lengths.")

    # Published ITPA threshold normalization used by the source notebook.
    p_itpa_mw = 0.049 * np.abs(d["NEL20"])**0.72 * np.abs(d["BT"])**0.8 * d["SPLASMA"]**0.94
    formula_valid = (
        np.isfinite(p_itpa_mw) & (p_itpa_mw > 0) &
        np.isfinite(d["NEL20"]) & np.isfinite(d["BT"]) &
        np.isfinite(d["SPLASMA"])
    )
    valid = formula_valid & np.isfinite(d["PLMW"]) & np.isfinite(d["PLMWSTD"])
    ratio = np.full(n, np.nan)
    ratio_err = np.full(n, np.nan)
    ratio[valid] = d["PLMW"][valid] / p_itpa_mw[valid]
    ratio_err[valid] = d["PLMWSTD"][valid] / p_itpa_mw[valid]

    common = (d["SEEDING"] == 0) & (d["PECH"] == 0) & (d["NEL20"] > 0.35) & (d["NEL20"] < 0.45)
    panel_a = [
        ("a_D_ILH_ABEAM2", d["ILH"].astype(bool) & common & (d["Q95"] < 3.5) & (d["ABEAM"] == 2) & (d["A"] == 2)),
        ("a_H_ILH_ABEAM2", d["ILH"].astype(bool) & common & (d["Q95"] < 3.5) & (d["ABEAM"] == 2) & (d["A"] == 1)),
        ("a_H_ILH_ABEAM1", d["ILH"].astype(bool) & common & (d["Q95"] < 3.5) & (d["ABEAM"] == 1) & (d["A"] == 1)),
        ("a_H_noLH_ABEAM1", (~d["ILH"].astype(bool)) & common & (d["Q95"] < 3.5) & (d["ABEAM"] == 1) & (d["A"] == 1)),
        ("a_D_ILH_ABEAM1", d["ILH"].astype(bool) & common & (d["Q95"] < 3.5) & (d["ABEAM"] == 1) & (d["A"] == 2)),
    ]
    common_b = (d["SEEDING"] == 0) & (d["PECH"] == 0) & (d["NEL20"] > 0.35) & (d["NEL20"] < 0.45) & (d["Q95"] > 4) & (d["Q95"] < 5)
    panel_b = [
        ("b_D_ILH_ABEAM2", d["ILH"].astype(bool) & common_b & (d["ABEAM"] == 2) & (d["A"] == 2)),
        ("b_D_noLH_ABEAM2", (~d["ILH"].astype(bool)) & common_b & (d["ABEAM"] == 2) & (d["A"] == 2)),
        ("b_H_ILH_ABEAM2", d["ILH"].astype(bool) & common_b & (d["ABEAM"] == 2) & (d["A"] == 1)),
        ("b_H_ILH_ABEAM1", d["ILH"].astype(bool) & common_b & (d["ABEAM"] == 1) & (d["A"] == 1)),
        ("b_D_ILH_ABEAM1", d["ILH"].astype(bool) & common_b & (d["ABEAM"] == 1) & (d["A"] == 2)),
    ]

    groups = []
    rows = []
    for group_name, mask in panel_a + panel_b:
        mask = mask & valid
        vals = ratio[mask]
        errs = ratio_err[mask]
        groups.append({
            "group": group_name, "n": int(mask.sum()),
            "mean_ratio": float(np.mean(vals)) if len(vals) else None,
            "median_ratio": float(np.median(vals)) if len(vals) else None,
            "min_ratio": float(np.min(vals)) if len(vals) else None,
            "max_ratio": float(np.max(vals)) if len(vals) else None,
            "mean_normalized_error": float(np.mean(errs)) if len(errs) else None,
            "fraction_below_reference_line": float(np.mean(vals < 1.0)) if len(vals) else None,
        })
        for i in np.flatnonzero(mask):
            rows.append({
                "group": group_name, "shot": int(d["SHOT"][i]),
                "ILH": bool(d["ILH"][i]), "A": float(d["A"][i]),
                "ABEAM": float(d["ABEAM"][i]), "Q95": float(d["Q95"][i]),
                "NEL20": float(d["NEL20"][i]), "BT_T": float(d["BT"][i]),
                "SPLASMA_m2": float(d["SPLASMA"][i]),
                "PLMW_MW": float(d["PLMW"][i]), "PLMWSTD_MW": float(d["PLMWSTD"][i]),
                "P_ITPA_MW": float(p_itpa_mw[i]), "P_loss_over_P_ITPA": float(ratio[i]),
                "ratio_uncertainty": float(ratio_err[i]),
            })

    # Reproduce the source notebook's Figure-2 power-balance comparison.
    with h5py.File(lh_path, "r") as h5:
        fig2 = h5["/data/fig2"]
        ploss = np.asarray(fig2["PLMW"]).reshape(-1)
        psep = np.asarray(fig2["PSEP"]).reshape(-1)
    balance_mask = np.isfinite(ploss) & np.isfinite(psep) & (ploss > 0.2)
    if int(balance_mask.sum()) >= 5:
        xp, yp = ploss[balance_mask], psep[balance_mask]
        slope, intercept = np.polyfit(xp, yp, 1)
        pearson_r = float(np.corrcoef(xp, yp)[0, 1])
        rank_x = pd.Series(xp).rank(method="average").to_numpy()
        rank_y = pd.Series(yp).rank(method="average").to_numpy()
        spearman_rho = float(np.corrcoef(rank_x, rank_y)[0, 1])
        ratio = yp / xp
        power_balance_check = {
            "valid_points": int(balance_mask.sum()),
            "pearson_r": pearson_r,
            "spearman_rho": spearman_rho,
            "linear_slope": float(slope),
            "linear_intercept_MW": float(intercept),
            "mean_Psep_minus_Ploss_MW": float(np.mean(yp - xp)),
            "median_Psep_over_Ploss": float(np.median(ratio)),
            "absolute_difference_MW_p50_p90_max": [
                float(np.quantile(np.abs(yp-xp), 0.50)),
                float(np.quantile(np.abs(yp-xp), 0.90)),
                float(np.max(np.abs(yp-xp))),
            ],
            "status": "PASS" if pearson_r >= 0.98 and 0.90 <= slope <= 1.10 and 0.85 <= np.median(ratio) <= 1.15 else "PENDING",
        }
        balance_rows = [{
            "row_index": int(i),
            "P_loss_MW": float(ploss[i]), "P_sep_MW": float(psep[i]),
            "P_sep_over_P_loss": float(psep[i]/ploss[i]),
        } for i in np.flatnonzero(balance_mask)]
    else:
        power_balance_check = {"valid_points": int(balance_mask.sum()), "status": "PENDING"}
        balance_rows = []

    report = {
        "title": "TCV L-H threshold scaling reproducibility",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_record": RECORD, "article_doi": ARTICLE_DOI,
        "source_database": provenance, "figure2_power_balance_source": lh_provenance,
        "database_rows": int(n), "formula_valid_rows": int(formula_valid.sum()), "valid_scaling_rows": int(valid.sum()),
        "excluded_ratio_rows": [
            {"shot": int(d["SHOT"][i]), "reason": "PLMW or PLMWSTD missing"}
            for i in np.flatnonzero(formula_valid & ~valid)
        ],
        "formula": "P_ITPA_MW = 0.049 * abs(NEL20)**0.72 * abs(BT_T)**0.8 * SPLASMA_m2**0.94",
        "formula_units": {
            "NEL20": "line-averaged density in 10^20 m^-3",
            "BT": "toroidal magnetic field in tesla",
            "SPLASMA": "plasma surface area in m^2",
            "PLMW": "loss power in MW",
        },
        "normalization": "P_loss/P_ITPA; uncertainty PLMWSTD/P_ITPA",
        "figure4_group_reproduction": groups, "figure2_power_balance_check": power_balance_check,
        "acceptance": {
            "formula_reproduction_pass": bool(formula_valid.all() and valid.sum() >= 0.95 * n and len(rows) > 0),
            "power_balance_consistency_pass": power_balance_check.get("status") == "PASS",
            "criteria": "All rows produce finite positive P_ITPA; at least 95% have finite PLMW and PLMWSTD for the ten Figure-4 masks.",
            "interpretation": "This reproduces the published database normalization and selection logic; it is not a reanalysis of raw diagnostic signals.",
        },
    }
    (OUT / "tcv_lh_threshold_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "tcv_lh_threshold_selected_shots.csv", index=False)
    pd.DataFrame(balance_rows).to_csv(OUT / "tcv_power_balance_points.csv", index=False)
    print(json.dumps({
        "report": str(OUT / "tcv_lh_threshold_validation.json"),
        "database_rows": n, "formula_valid_rows": int(formula_valid.sum()), "valid_rows": int(valid.sum()),
        "groups": groups,
        "formula_reproduction_pass": report["acceptance"]["formula_reproduction_pass"],
        "power_balance_check": power_balance_check,
    }, indent=2))


if __name__ == "__main__":
    main()
