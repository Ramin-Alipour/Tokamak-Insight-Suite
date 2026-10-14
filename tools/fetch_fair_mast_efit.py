#!/usr/bin/env python3
"""Export one public FAIR-MAST EFIT equilibrium snapshot for reproducible inspection.

This exports an existing archive reconstruction; it does not solve the
Grad-Shafranov equation or validate the equilibrium reconstruction itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

SHOT = 30421
URL = f"https://s3.echo.stfc.ac.uk/mast/level1/shots/{SHOT}.zarr"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mast-efit"))
    parser.add_argument("--time-s", type=float, default=0.2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_zarr(URL, group="efm")
    try:
        required = {"plasma_current_rz", "r", "z", "time"}
        missing = required.difference(ds.variables)
        if missing:
            raise KeyError(f"EFIT archive missing required variables: {sorted(missing)}")

        field = ds["plasma_current_rz"]
        if "time" not in field.dims or "r" not in field.dims or "z" not in field.dims:
            raise ValueError(f"Unexpected plasma_current_rz dimensions: {field.dims}")
        times = np.asarray(ds["time"].values, dtype=float)
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError("EFIT time coordinate is not a finite one-dimensional array.")
        index = int(np.argmin(np.abs(times - args.time_s)))
        snapshot = field.isel(time=index).transpose("z", "r")
        grid = np.asarray(snapshot.values, dtype=float)
        r = np.asarray(ds["r"].values, dtype=float).ravel()
        z = np.asarray(ds["z"].values, dtype=float).ravel()
        if grid.shape != (z.size, r.size):
            raise ValueError(f"Grid/coordinate mismatch: field={grid.shape}, z={z.size}, r={r.size}")
        finite = np.isfinite(grid)
        if not np.any(finite):
            raise ValueError("Selected equilibrium snapshot contains no finite grid values.")

        # Wide-format CSV: first column is Z [m], remaining columns are R [m].
        table = np.column_stack([z, grid])
        csv_path = args.output_dir / "mast_30421_efit_plasma_current_rz.csv"
        np.savetxt(csv_path, table, delimiter=",",
                   header="z_m," + ",".join(f"R={v:.9g}_m" for v in r), comments="")
        summary = {
            "dataset": "UKAEA FAIR-MAST Level-1",
            "shot": SHOT,
            "source_group": "efm",
            "source_signal": "plasma_current_rz",
            "source_url": URL,
            "requested_time_s": float(args.time_s),
            "selected_time_s": float(times[index]),
            "time_index": index,
            "grid_shape_z_by_r": [int(grid.shape[0]), int(grid.shape[1])],
            "r_min_m": float(np.min(r)), "r_max_m": float(np.max(r)),
            "z_min_m": float(np.min(z)), "z_max_m": float(np.max(z)),
            "finite_grid_fraction": float(np.mean(finite)),
            "grid_min_finite": float(np.min(grid[finite])),
            "grid_max_finite": float(np.max(grid[finite])),
            "grid_mean_finite": float(np.mean(grid[finite])),
            "csv_file": csv_path.name,
            "scientific_limit": (
                "Exports an existing EFIT-derived archive grid only. No equilibrium "
                "reconstruction is performed, and coordinate conventions/units must be "
                "checked against the source metadata before scientific use."
            ),
        }
        if "current_centrd_r" in ds and "current_centrd_z" in ds:
            summary["current_centroid_r_m"] = float(ds["current_centrd_r"].isel(time=index).values)
            summary["current_centroid_z_m"] = float(ds["current_centrd_z"].isel(time=index).values)
        (args.output_dir / "mast_30421_efit_manifest.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(
            f"PASS: MAST shot {SHOT} EFIT grid at t={times[index]:.6f} s; "
            f"shape={grid.shape}; finite={np.mean(finite):.3f}"
        )
    finally:
        ds.close()


if __name__ == "__main__":
    main()
