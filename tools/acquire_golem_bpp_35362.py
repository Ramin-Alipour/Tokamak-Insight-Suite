#!/usr/bin/env python3
"""Acquire and discover raw swept BPP signals for GOLEM shot 35362."""
from __future__ import annotations
import hashlib, json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np

SHOT = 35362
HTTP = f"http://golem.fjfi.cvut.cz/shots/{SHOT}"
HTTPS = f"https://golem.fjfi.cvut.cz/shots/{SHOT}"
OUT = Path("artifacts/golem-bpp-35362-raw")
OUT.mkdir(parents=True, exist_ok=True)
signal_candidates = {
    "probe_bias_voltage": [
        f"{HTTP}/Diagnostics/PetiProbe/U_bias", f"{HTTP}/Diagnostics/PetiProbe/U_bias.txt",
        f"{HTTP}/Diagnostics/PetiProbe/data/U_bias", f"{HTTP}/Diagnostics/PetiProbe/data/U_bias.txt",
        f"{HTTP}/diagnostics/PetiProbe/U_bias", f"{HTTP}/Diagnostics/PetiProbe/U_bias.csv",
        f"http://golem.fjfi.cvut.cz/utils/data/{SHOT}/U_bias",
        f"http://golem.fjfi.cvut.cz/utils/data/{SHOT}/PetiProbe:U_bias",
        f"http://golem.fjfi.cvut.cz/utils/data/{SHOT}/PetiProbe/U_bias",
        f"http://golem.fjfi.cvut.cz/utils/data/{SHOT}/PetiProbe_U_bias",
    ],
    "probe_current_signal": [
        f"{HTTP}/Diagnostics/PetiProbe/U_current", f"{HTTP}/Diagnostics/PetiProbe/U_current.txt",
        f"{HTTP}/Diagnostics/PetiProbe/data/U_current", f"{HTTP}/Diagnostics/PetiProbe/data/U_current.txt",
        f"{HTTP}/diagnostics/PetiProbe/U_current", f"{HTTP}/Diagnostics/PetiProbe/U_current.csv",
        f"http://golem.fjfi.cvut.cz/utils/data/{SHOT}/U_current",
        f"http://golem.fjfi.cvut.cz/utils/data/{SHOT}/PetiProbe:U_current",
        f"http://golem.fjfi.cvut.cz/utils/data/{SHOT}/PetiProbe/U_current",
        f"http://golem.fjfi.cvut.cz/utils/data/{SHOT}/PetiProbe_U_current",
    ],
}
parameters = {
    "sweep_frequency_Hz": f"{HTTP}/Diagnostics/PetiProbe/Parameters/f_fg",
    "current_resistor_ohm": f"{HTTP}/Diagnostics/PetiProbe/Parameters/r_i",
    "bias_amplitude_V": f"{HTTP}/Diagnostics/PetiProbe/Parameters/u_fg",
    "current_drive_trigger_us": f"{HTTP}/Production/Parameters/Tcd",
    "toroidal_field_trigger_us": f"{HTTP}/Production/Parameters/TBt",
    "toroidal_field_series": f"http://golem.fjfi.cvut.cz/utils/data/{SHOT}/toroidal_field",
    "plasma_start_ms": f"{HTTP}/Diagnostics/BasicDiagnostics/Results/t_plasma_start",
    "plasma_end_ms": f"{HTTP}/Diagnostics/BasicDiagnostics/Results/t_plasma_end",
}
report = {
    "shot": SHOT,
    "shot_page": f"{HTTPS}/",
    "published_reference": "https://golem.fjfi.cvut.cz/wiki/Experiments/EdgePlasmaPhysics/ParticleFlux/BallPenProbe/Reports/0221_DC_Golem_report_3.pdf",
    "published_setup": {"sweep_range_V": [-150, 150], "sweep_frequency_Hz": 50000, "current_resistor_ohm": 47},
    "report_says": "For shots 35346/35362/35364 the ion-temperature distribution peaks near 15 eV; the BPP is swept and the LP is floating.",
    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    "signal_candidates": {}, "parameters": {},
}
def get(url, timeout=6):
    req = Request(url, headers={"User-Agent": "Tokamak-Insight-Suite-BPP-raw-acquisition/1.1"})
    with urlopen(req, timeout=timeout) as r:
        return r.read(), r.geturl(), r.headers.get("Content-Type", "")

def parse(raw):
    try:
        arr = np.loadtxt(raw.decode("utf-8", "replace").splitlines())
        arr = np.asarray(arr)
        if arr.size < 10:
            return None
        return arr
    except Exception:
        return None

def probe_one(args):
    name, url = args
    rec = {"url": url}
    try:
        raw, final_url, ctype = get(url)
        arr = parse(raw)
        rec.update({"final_url": final_url, "content_type": ctype, "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest()})
        if arr is None:
            rec["status"] = "not_numeric_or_too_short"
        else:
            rec.update({"status": "numeric_data", "shape": list(arr.shape),
                        "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr))})
            (OUT / f"{name}.txt").write_bytes(raw)
    except Exception as exc:
        rec.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return rec

tasks = [(name, url) for name, urls in signal_candidates.items() for url in urls]
with ThreadPoolExecutor(max_workers=16) as pool:
    all_results = list(pool.map(probe_one, tasks))
for (name, url), rec in zip(tasks, all_results):
    report["signal_candidates"].setdefault(name, []).append(rec)
for name, url in parameters.items():
    try:
        raw, final_url, ctype = get(url)
        arr = parse(raw)
        if arr is not None:
            info = {"status": "numeric_data", "shape": list(arr.shape),
                    "values": np.asarray(arr).reshape(-1)[:8].tolist()}
        else:
            try:
                val = float(raw.decode("utf-8", "replace").strip())
                info = {"status": "numeric_scalar", "value": val}
            except Exception:
                info = {"status": "not_numeric_or_empty"}
        info.update({"url": url, "final_url": final_url, "content_type": ctype,
                     "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        report["parameters"][name] = info
    except Exception as exc:
        report["parameters"][name] = {"url": url, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
for name, results in report["signal_candidates"].items():
    successes = [x for x in results if x["status"] == "numeric_data"]
    report.setdefault("chosen_signal_urls", {})[name] = successes[0]["final_url"] if successes else None
report["raw_probe_pair_available"] = all(report.get("chosen_signal_urls", {}).get(k) for k in signal_candidates)
report["numeric_candidate_count"] = sum(x["status"] == "numeric_data" for v in report["signal_candidates"].values() for x in v)
(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
