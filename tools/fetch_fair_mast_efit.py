#!/usr/bin/env python3
"""Export one public FAIR-MAST EFIT equilibrium snapshot for reproducible inspection.

This exports an existing archive reconstruction; it does not solve the
Grad-Shafranov equation or validate the equilibrium reconstruction itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tis.data_sources import fetch_fair_mast_efit_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mast-efit"))
    parser.add_argument("--time-s", type=float, default=0.2)
    parser.add_argument("--shot", type=int, default=30421)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, manifest = fetch_fair_mast_efit_snapshot(shot=args.shot, time_s=args.time_s)
    csv_name = f"mast_{args.shot}_efit_plasma_current_rz.csv"
    frame.to_csv(args.output_dir / csv_name, index=False)
    manifest["csv_file"] = csv_name
    (args.output_dir / f"mast_{args.shot}_efit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"PASS: MAST shot {args.shot} EFIT grid at t={manifest['selected_time_s']:.6f} s; "
        f"shape={manifest['grid_shape_z_by_r']}; finite={manifest['finite_grid_fraction']:.3f}"
    )


if __name__ == "__main__":
    main()
