#!/usr/bin/env python3
"""Probe public GOLEM shot 30622 for a swept BPP voltage/current channel."""
from __future__ import annotations
import hashlib, json, re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np

SHOT = 30622
ROOT = "http://golem.fjfi.cvut.cz"
OUT = Path("artifacts/golem-probe-30622")
OUT.mkdir(parents=True, exist_ok=True)
identifiers = [
    "bpp", "lp", "BPP", "LP", "U_bias", "U_current", "bpp_voltage", "bpp_current",
    "probe_voltage", "probe_current", "PetiProbe:U_bias", "PetiProbe:U_current",
    "PetiProbe/U_bias", "PetiProbe/U_current", "nistandard",
    "electron_temperature", "toroidal_field",
]
urls = [f"{ROOT}/utils/data/{SHOT}/{x}" for x in identifiers] + [
    f"{ROOT}/shots/{SHOT}/diagnostics/ParticleFlux/PetiProbe/{x}" for x in
    ("U_bias", "U_current", "bpp", "lp")
] + [
    f"{ROOT}/operation/shots/{SHOT}/diagnostics/ParticleFlux/PetiProbe.ON/data/{x}" for x in
    ("U_bias", "U_current", "bpp", "lp")
]
report = {"shot": SHOT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
          "purpose": "Determine whether shot 30622 exposes swept BPP raw voltage/current suitable for direct Ti fitting.",
          "candidates": []}
def get(url):
    req = Request(url, headers={"User-Agent": "Tokamak-Insight-Suite-probe-discovery/1.0"})
    with urlopen(req, timeout=5) as r:
        return r.read(), r.geturl(), r.headers.get("Content-Type", "")
def try_one(url):
    rec = {"url": url}
    try:
        raw, final, ctype = get(url)
        rec.update(final_url=final, content_type=ctype, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        try:
            arr = np.loadtxt(raw.decode("utf-8", "replace").splitlines())
            arr = np.asarray(arr)
        except Exception:
            arr = np.asarray([])
        if arr.size < 10:
            rec["status"] = "empty_or_non_numeric"
        else:
            rec.update(status="numeric_data", shape=list(arr.shape), min=float(np.nanmin(arr)), max=float(np.nanmax(arr)))
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", url.rstrip("/").split("/")[-1])
            (OUT / f"{safe}_{hashlib.sha256(url.encode()).hexdigest()[:8]}.txt").write_bytes(raw)
            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] > 100:
                t = arr[:, 0]; y = arr[:, 1]
                good = np.isfinite(t) & np.isfinite(y)
                t, y = t[good], y[good]
                if t.size > 100 and np.all(np.diff(t) > 0):
                    dt = float(np.median(np.diff(t)))
                    rec.update(sample_count=int(t.size), median_dt_s=dt, sample_rate_Hz=1/dt,
                               duration_s=float(t[-1]-t[0]), signal_range=float(np.ptp(y)))
                    if t.size > 1000:
                        y0 = y - np.mean(y)
                        freq = np.fft.rfftfreq(len(y0), dt)
                        power = np.abs(np.fft.rfft(y0))**2
                        mask = (freq >= 1000) & (freq <= min(2e6, 0.45/dt))
                        if np.any(mask):
                            k = np.argmax(power[mask])
                            rec["dominant_frequency_Hz"] = float(freq[mask][k])
        return rec
    except Exception as exc:
        return {"url": url, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
with ThreadPoolExecutor(max_workers=24) as pool:
    report["candidates"] = list(pool.map(try_one, list(dict.fromkeys(urls))))
numeric = [x for x in report["candidates"] if x["status"] == "numeric_data"]
report["numeric_candidate_count"] = len(numeric)
report["possible_swept_signal_candidates"] = [
    x for x in numeric if 30000 <= x.get("dominant_frequency_Hz", 0) <= 70000
]
report["direct_Ti_fit_possible_from_found_channels"] = any(
    "U_bias" in x["url"] for x in numeric
) and any("U_current" in x["url"] for x in numeric)
(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
