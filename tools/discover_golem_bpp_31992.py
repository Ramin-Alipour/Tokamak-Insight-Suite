#!/usr/bin/env python3
"""Discover published swept ball-pen-probe signals for GOLEM shot 31992.

The 2020 GOLEM ion-temperature report uses shot 31992 and reports a weighted
BPP I-V fit at 7.268177 ms of Ti = 21.00 ± 3.27 eV. This script searches the
shot archive for the original bias/current signals so the fit can be reproduced
from measured data rather than digitized plots.
"""
from __future__ import annotations
import hashlib, json, re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SHOT = 31992
ROOT = "https://golem.fjfi.cvut.cz"
OUT = Path("artifacts/golem-bpp-31992-discovery")
OUT.mkdir(parents=True, exist_ok=True)

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            a = dict(attrs)
            href = a.get("href", "")
            text = a.get("title", "")
            if href:
                self.links.append({"href": href, "title": text})

def get(url, timeout=12):
    req = Request(url, headers={"User-Agent": "Tokamak-Insight-Suite-BPP-validation/1.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read(), r.geturl(), r.headers.get("Content-Type", "")

pages = [
    f"{ROOT}/shots/{SHOT}/",
    f"{ROOT}/shots/{SHOT}/Data.php",
    f"{ROOT}/shots/{SHOT}/Diagnostics/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/ParticleFlux/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/BallPenProbe/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/ParticleFlux/BallPenProbe/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/PetiProbe/",
    f"{ROOT}/shots/{SHOT}/diagnostics/ParticleFlux/",
    f"{ROOT}/shots/{SHOT}/diagnostics/ParticleFlux/BallPenProbe/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/BallPenProbe/analysis.html",
    f"{ROOT}/shots/{SHOT}/Diagnostics/ParticleFlux/BallPenProbe/analysis.html",
]
report = {
    "shot": SHOT,
    "report_reference": "https://golem.fjfi.cvut.cz/wiki/Experiments/EdgePlasmaPhysics/ParticleFlux/BallPenProbe/Reports/0220_DC_Report_Golem.pdf",
    "reported_fit": {"time_ms": 7.268177445738422, "Ti_weighted_eV": 21.00, "Ti_weighted_uncertainty_eV": 3.27,
                     "Ti_unweighted_eV": 21.13, "Ti_unweighted_uncertainty_eV": 4.14},
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "page_attempts": [],
    "discovered_probe_links": [],
    "signal_attempts": [],
}
for url in pages:
    rec = {"url": url}
    try:
        raw, final_url, ctype = get(url)
        rec.update(status="downloaded", final_url=final_url, content_type=ctype,
                   bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        (OUT / f"page_{len(report['page_attempts']):02d}.html").write_bytes(raw)
        if "html" in ctype.lower() or b"<html" in raw[:500].lower() or b"<!doctype" in raw[:500].lower():
            parser = Links()
            parser.feed(raw.decode("utf-8", "replace"))
            for link in parser.links:
                href = link["href"]
                if re.search(r"bpp|ball.?pen|probe|particleflux|u_bias|u_current|current|voltage", href + " " + link["title"], re.I):
                    report["discovered_probe_links"].append({"page": final_url, **link})
    except Exception as exc:
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    report["page_attempts"].append(rec)

# Known signal naming conventions used by GOLEM probe pages and PyGolem.
bases = [
    f"{ROOT}/shots/{SHOT}/Diagnostics/BallPenProbe/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/ParticleFlux/BallPenProbe/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/PetiProbe/",
    f"{ROOT}/shots/{SHOT}/diagnostics/ParticleFlux/BallPenProbe/",
    f"{ROOT}/shots/{SHOT}/diagnostics/ParticleFlux/PetiProbe/",
    f"{ROOT}/operation/shots/{SHOT}/diagnostics/ParticleFlux/BallPenProbe.ON/data/",
    f"{ROOT}/operation/shots/{SHOT}/diagnostics/ParticleFlux/1012RakeProbe.ON/data/",
    f"{ROOT}/operation/shots/{SHOT}/diagnostics/ParticleFlux/1111BallPenProbe.ON/data/",
]
names = ["U_bias", "U_current", "I", "V", "voltage", "current", "bpp", "lp", "BPP", "LP",
         "U_bias.txt", "U_current.txt", "U_bias.csv", "U_current.csv"]
candidates = []
for link in report["discovered_probe_links"]:
    href = link["href"]
    if href.startswith("http://") or href.startswith("https://"):
        candidates.append(href)
    elif href.startswith("/"):
        candidates.append(ROOT + href)
    else:
        candidates.append(link["page"].rsplit("/", 1)[0] + "/" + href)
for base in bases:
    candidates.extend(base + name for name in names)
seen = set()
for url in candidates:
    if url in seen:
        continue
    seen.add(url)
    rec = {"url": url}
    try:
        raw, final_url, ctype = get(url, timeout=6)
        if not raw or b"<html" in raw[:500].lower() or b"<!doctype" in raw[:500].lower():
            rec.update(status="not_numeric_payload", content_type=ctype, bytes=len(raw), final_url=final_url)
        else:
            try:
                import numpy as np
                arr = np.loadtxt(raw.decode("utf-8", "replace").splitlines())
                rec.update(status="numeric_payload", content_type=ctype, bytes=len(raw), final_url=final_url,
                           shape=list(arr.shape), sha256=hashlib.sha256(raw).hexdigest())
                filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", url.rstrip("/").split("/")[-1]) or "signal"
                (OUT / f"signal_{len(report['signal_attempts']):03d}_{filename}.txt").write_bytes(raw)
            except Exception:
                rec.update(status="downloaded_unparsed", content_type=ctype, bytes=len(raw), final_url=final_url)
    except Exception as exc:
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    report["signal_attempts"].append(rec)

report["summary"] = {
    "pages_downloaded": sum(x["status"] == "downloaded" for x in report["page_attempts"]),
    "probe_links_found": len(report["discovered_probe_links"]),
    "numeric_signals_found": sum(x["status"] == "numeric_payload" for x in report["signal_attempts"]),
}
(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
