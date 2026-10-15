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

from tis.data_sources import FAIR_MAST_BASE

SHOT = 28830
GROUPS = ("arp", "ayc", "xmo")


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
