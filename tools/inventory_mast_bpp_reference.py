#!/usr/bin/env python3
"""Inventory FAIR-MAST groups for a published BPP/Thomson cross-diagnostic shot.

The target discharge is MAST 28830 from Walkden et al., arXiv:1411.7298.
This is discovery only: no measurement is labelled validated unless raw inputs,
time/radius correspondence, and a same-observable reference are established.
"""
from __future__ import annotations

import json
from pathlib import Path

import xarray as xr
import pandas as pd
import numpy as np

from tis.data_sources import FAIR_MAST_BASE

SHOT = 28830
GROUPS = ("arp", "ayc", "efm", "ane")


def main() -> None:
    out = Path("artifacts/mast-bpp-discovery")
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "shot": SHOT,
        "publication": "Walkden et al., Profile measurements in the plasma edge of MAST using a ball pen probe, arXiv:1411.7298",
        "publication_shots": [28819, 28830, 28834],
        "groups": [],
        "classification": "availability inventory only; not a validation result",
    }
    url = f"{FAIR_MAST_BASE}/{SHOT}.zarr"
    for group in GROUPS:
        item = {"group": group, "status": "unavailable"}
        ds = None
        try:
            ds = xr.open_zarr(url, group=group, storage_options={"timeout": 12})
            item.update(
                status="available",
                dimensions={k: int(v) for k, v in ds.sizes.items()},
                variables={
                    name: {
                        "dims": list(ds[name].dims),
                        "shape": list(ds[name].shape),
                        "dtype": str(ds[name].dtype),
                        "units": ds[name].attrs.get("units"),
                        "long_name": ds[name].attrs.get("long_name"),
                    }
                    for name in ds.data_vars
                },
                coordinates=list(ds.coords),
            )
            # Preserve measured profiles and independent diagnostics for offline
            # comparison against the paper's tabulated shot-28830 values.
            if group == "ayc" and all(name in ds for name in ("time", "radius", "te", "ne")):
                t = np.asarray(ds["time"].values, dtype=float)
                radius = np.asarray(ds["radius"].values, dtype=float)
                te = np.asarray(ds["te"].values, dtype=float)
                ne = np.asarray(ds["ne"].values, dtype=float)
                te_err = np.asarray(ds["te_error"].values, dtype=float) if "te_error" in ds else np.full_like(te, np.nan)
                ne_err = np.asarray(ds["ne_error"].values, dtype=float) if "ne_error" in ds else np.full_like(ne, np.nan)
                rows = []
                for it, time_s in enumerate(t):
                    for ir in range(radius.shape[-1]):
                        rows.append((time_s, ir, radius[it, ir], te[it, ir], te_err[it, ir], ne[it, ir], ne_err[it, ir]))
                pd.DataFrame(rows, columns=["time_s", "radial_index", "radius_m", "te_eV", "te_error_eV", "ne_m3", "ne_error"]).to_csv(out / "mast_28830_thomson_profiles.csv", index=False)
            if group == "ane" and "density" in ds:
                time_name = next((name for name in ("time", "time1") if name in ds and ds[name].ndim == 1), None)
                if time_name is not None:
                    pd.DataFrame({"time_s": np.asarray(ds[time_name].values, dtype=float), "line_density_raw": np.asarray(ds["density"].values, dtype=float)}).to_csv(out / "mast_28830_interferometer_density.csv", index=False)
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if ds is not None:
                ds.close()
        report["groups"].append(item)
        print(f"{group}: {item['status']}")
    report["possible_cross_diagnostic_reconstruction"] = (
        "Proceed only if probe-potential and Thomson Te data are present with enough timing/position metadata."
    )
    (out / "inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({g["group"]: g["status"] for g in report["groups"]}, indent=2))


if __name__ == "__main__":
    main()
