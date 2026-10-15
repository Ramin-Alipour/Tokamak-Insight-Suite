#!/usr/bin/env python3
"""Acquire raw swept BPP signals from public GOLEM shot 35362."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np

SHOT = 35362
ROOT = f"http://golem.fjfi.cvut.cz/shots/{SHOT}"
OUT = Path("artifacts/golem-bpp-35362-raw")
OUT.mkdir(parents=True, exist_ok=True)
products = {
    "probe_bias_voltage": f"{ROOT}/Diagnostics/PetiProbe/U_bias",
    "probe_current_signal": f"{ROOT}/Diagnostics/PetiProbe/U_current",
    "sweep_frequency_Hz": f"{ROOT}/Diagnostics/PetiProbe/Parameters/f_fg",
    "current_resistor_ohm": f"{ROOT}/Diagnostics/PetiProbe/Parameters/r_i",
    "bias_amplitude_V": f"{ROOT}/Diagnostics/PetiProbe/Parameters/u_fg",
    "current_drive_trigger_us": f"{ROOT}/Production/Parameters/Tcd",
    "toroidal_field_trigger_us": f"{ROOT}/Production/Parameters/TBt",
    "toroidal_field_series": f"http://golem.fjfi.cvut.cz/utils/data/{SHOT}/toroidal_field",
    "plasma_start_ms": f"{ROOT}/Diagnostics/BasicDiagnostics/Results/t_plasma_start",
    "plasma_end_ms": f"{ROOT}/Diagnostics/BasicDiagnostics/Results/t_plasma_end",
}
report = {
    "shot": SHOT,
    "shot_page": f"{ROOT}/",
    "published_reference": "https://golem.fjfi.cvut.cz/wiki/Experiments/EdgePlasmaPhysics/ParticleFlux/BallPenProbe/Reports/0221_DC_Golem_report_3.pdf",
    "report_says": "BPP swept ±150 V at 50 kHz; ion-temperature distribution peaks near 15 eV for shots 35346/35362/35364.",
    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    "products": {},
}
def get(url):
    req = Request(url, headers={"User-Agent": "Tokamak-Insight-Suite-BPP-raw-acquisition/1.0"})
    with urlopen(req, timeout=10) as r:
        return r.read(), r.geturl(), r.headers.get("Content-Type", "")

for name, url in products.items():
    try:
        raw, final_url, ctype = get(url)
        (OUT / f"{name}.txt").write_bytes(raw)
        decoded = raw.decode("utf-8", "replace")
        arr = None
        try:
            arr = np.loadtxt(decoded.splitlines())
        except Exception:
            pass
        if arr is not None and np.size(arr) > 0:
            arr = np.asarray(arr)
            info = {"status": "numeric_data", "shape": list(arr.shape), "min": float(np.nanmin(arr)),
                    "max": float(np.nanmax(arr)), "mean": float(np.nanmean(arr))}
        else:
            info = {"status": "downloaded_not_numeric_or_empty"}
        info.update({"url": url, "final_url": final_url, "content_type": ctype, "bytes": len(raw),
                     "sha256": hashlib.sha256(raw).hexdigest()})
        report["products"][name] = info
    except Exception as exc:
        report["products"][name] = {"url": url, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

report["raw_probe_pair_available"] = all(
    report["products"].get(k, {}).get("status") == "numeric_data"
    for k in ("probe_bias_voltage", "probe_current_signal")
)
(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
