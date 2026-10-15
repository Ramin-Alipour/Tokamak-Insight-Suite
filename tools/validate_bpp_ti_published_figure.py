#!/usr/bin/env python3
"""Compare the BPP Ti fitter with a plot-digitized published GOLEM benchmark.

This is a method/regression check only: the CSV points are digitized approximately
from Figure 3, not the original oscilloscope samples.
"""
from __future__ import annotations
import csv, json
from datetime import datetime, timezone
from pathlib import Path
from tis.probes import fit_swept_ball_pen_ion_temperature

ROOT = Path(__file__).resolve().parents[1]
fixture = ROOT / "validation" / "golem_bpp_31992_figure3_digitized.csv"
outdir = ROOT / "artifacts" / "golem-bpp-31992-fit"
outdir.mkdir(parents=True, exist_ok=True)
with fixture.open(newline="", encoding="utf-8") as f:
    rows = [row for row in csv.DictReader(line for line in f if not line.startswith("#"))]
v = [float(row["voltage_V"]) for row in rows]
i = [float(row["current_mA"]) for row in rows]
fit = fit_swept_ball_pen_ion_temperature(
    v, i, alpha_bpp=0.6, fit_min_voltage_V=10.0,
    initial_ti_eV=21.0, plasma_potential_guess_V=7.15,
)
reference = {"shot": 31992, "time_ms": 7.268177445738422,
             "weighted_Ti_eV": 21.00, "weighted_sigma_eV": 3.27,
             "unweighted_Ti_eV": 21.13, "unweighted_sigma_eV": 4.14,
             "source": "https://golem.fjfi.cvut.cz/wiki/Experiments/EdgePlasmaPhysics/ParticleFlux/BallPenProbe/Reports/0220_DC_Report_Golem.pdf"}
report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_classification": "approximate plot-digitized curve; NOT raw experimental samples",
    "reference": reference,
    "fit_settings": {"alpha_bpp": 0.6, "fit_min_voltage_V": 10.0,
                     "voltage_points": len(v), "points_in_fit": sum(x >= 10.0 for x in v)},
    "fit": fit,
    "comparison_to_published_weighted_fit": {
        "absolute_difference_eV": abs(fit["Ti_eV"] - reference["weighted_Ti_eV"]),
        "relative_difference_percent": 100*abs(fit["Ti_eV"]-reference["weighted_Ti_eV"])/reference["weighted_Ti_eV"],
        "within_10_eV_digitization_tolerance": abs(fit["Ti_eV"]-reference["weighted_Ti_eV"]) <= 10.0,
        "interpretation": "Approximate consistency of the model implementation with the published plotted curve; not raw-data reproduction or validation of shot 30622."
    }
}
(outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
if not report["comparison_to_published_weighted_fit"]["within_10_eV_digitization_tolerance"]:
    raise SystemExit("BPP fit did not reproduce the plot-digitized published benchmark within 10 eV.")
