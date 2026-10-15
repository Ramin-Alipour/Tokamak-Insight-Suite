#!/usr/bin/env python3
"""Discover raw swept ball-pen-probe signals for GOLEM shot 31992."""
from __future__ import annotations
import hashlib, json, re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np

SHOT = 31992
ROOT = "https://golem.fjfi.cvut.cz"
OUT = Path("artifacts/golem-bpp-31992-discovery")
OUT.mkdir(parents=True, exist_ok=True)
REFERENCE = {
    "report": "https://golem.fjfi.cvut.cz/wiki/Experiments/EdgePlasmaPhysics/ParticleFlux/BallPenProbe/Reports/0220_DC_Report_Golem.pdf",
    "time_ms": 7.268177445738422, "weighted_Ti_eV": 21.00, "weighted_sigma_eV": 3.27,
    "unweighted_Ti_eV": 21.13, "unweighted_sigma_eV": 4.14,
}

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            a = dict(attrs)
            if a.get("href"):
                self.links.append({"href": a["href"], "title": a.get("title", "")})

def get(url, timeout=4):
    req = Request(url, headers={"User-Agent": "Tokamak-Insight-Suite-BPP-validation/1.1"})
    with urlopen(req, timeout=timeout) as r:
        return r.read(), r.geturl(), r.headers.get("Content-Type", "")

pages = [
    f"{ROOT}/shots/{SHOT}/", f"{ROOT}/shots/{SHOT}/Data.php",
    f"{ROOT}/shots/{SHOT}/Diagnostics/", f"{ROOT}/shots/{SHOT}/Diagnostics/ParticleFlux/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/BallPenProbe/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/ParticleFlux/BallPenProbe/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/PetiProbe/",
    f"{ROOT}/shots/{SHOT}/diagnostics/",
    f"{ROOT}/shots/{SHOT}/diagnostics/ParticleFlux/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/Probe/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/ParticleFlux/BPP/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/BPP/",
]
report = {"shot": SHOT, "reference": REFERENCE, "created_at_utc": datetime.now(timezone.utc).isoformat(),
          "page_attempts": [], "discovered_probe_links": [], "signal_attempts": []}

def fetch_page(url):
    try:
        raw, final, ctype = get(url)
        links = []
        if "html" in ctype.lower() or b"<html" in raw[:500].lower() or b"<!doctype" in raw[:500].lower():
            parser = Links(); parser.feed(raw.decode("utf-8", "replace"))
            links = [x for x in parser.links if re.search(r"bpp|ball.?pen|probe|particleflux|u_bias|u_current|current|voltage", x["href"]+" "+x["title"], re.I)]
        return {"url": url, "status": "downloaded", "final_url": final, "content_type": ctype,
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "links": links, "raw": raw}
    except Exception as exc:
        return {"url": url, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "links": []}

with ThreadPoolExecutor(max_workers=12) as pool:
    page_results = list(pool.map(fetch_page, pages))
for i, rec in enumerate(page_results):
    raw = rec.pop("raw", None)
    if raw is not None:
        (OUT / f"page_{i:02d}.html").write_bytes(raw)
    report["discovered_probe_links"].extend({"page": rec["url"], **link} for link in rec.pop("links", []))
    report["page_attempts"].append(rec)

bases = [
    f"{ROOT}/shots/{SHOT}/Diagnostics/BallPenProbe/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/ParticleFlux/BallPenProbe/",
    f"{ROOT}/shots/{SHOT}/Diagnostics/PetiProbe/",
    f"{ROOT}/shots/{SHOT}/diagnostics/ParticleFlux/BallPenProbe/",
    f"{ROOT}/shots/{SHOT}/diagnostics/ParticleFlux/PetiProbe/",
    f"{ROOT}/operation/shots/{SHOT}/diagnostics/ParticleFlux/BallPenProbe.ON/data/",
    f"{ROOT}/operation/shots/{SHOT}/diagnostics/ParticleFlux/1012RakeProbe.ON/data/",
    f"{ROOT}/operation/shots/{SHOT}/diagnostics/ParticleFlux/1111BallPenProbe.ON/data/",
    f"{ROOT}/utils/data/{SHOT}/",
    f"{ROOT}/utils/data/{SHOT}/BPP/",
    f"{ROOT}/utils/data/{SHOT}/PetiProbe/",
]
names = ["U_bias", "U_current", "I", "V", "voltage", "current", "bpp", "lp",
         "U_bias.txt", "U_current.txt", "U_bias.csv", "U_current.csv",
         "PetiProbe:U_bias", "PetiProbe:U_current", "bpp:U_bias", "bpp:U_current",
         "U_bias.npy", "U_current.npy"]
candidates = []
for link in report["discovered_probe_links"]:
    href = link["href"]
    if href.startswith(("http://", "https://")): candidates.append(href)
    elif href.startswith("/"): candidates.append(ROOT + href)
    else: candidates.append(link["page"].rsplit("/", 1)[0] + "/" + href)
for base in bases: candidates.extend(base + name for name in names)
candidates = list(dict.fromkeys(candidates))

def fetch_signal(url):
    rec = {"url": url}
    try:
        raw, final, ctype = get(url, timeout=3)
        if not raw or b"<html" in raw[:500].lower() or b"<!doctype" in raw[:500].lower():
            rec.update(status="not_numeric_payload", content_type=ctype, bytes=len(raw), final_url=final)
        else:
            try:
                arr = np.loadtxt(raw.decode("utf-8", "replace").splitlines())
                if np.size(arr) < 8:
                    rec.update(status="empty_or_too_short", content_type=ctype, bytes=len(raw), final_url=final,
                               shape=list(np.shape(arr)))
                    return rec
                rec.update(status="numeric_payload", content_type=ctype, bytes=len(raw), final_url=final,
                           shape=list(arr.shape), sha256=hashlib.sha256(raw).hexdigest())
                filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", url.rstrip("/").split("/")[-1]) or "signal"
                (OUT / f"signal_{hashlib.sha256(url.encode()).hexdigest()[:8]}_{filename}.txt").write_bytes(raw)
            except Exception:
                rec.update(status="downloaded_unparsed", content_type=ctype, bytes=len(raw), final_url=final)
    except Exception as exc:
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return rec

with ThreadPoolExecutor(max_workers=24) as pool:
    report["signal_attempts"] = list(pool.map(fetch_signal, candidates))

report["summary"] = {
    "pages_downloaded": sum(x["status"] == "downloaded" for x in report["page_attempts"]),
    "probe_links_found": len(report["discovered_probe_links"]),
    "numeric_signals_found": sum(x["status"] == "numeric_payload" for x in report["signal_attempts"]),
    "raw_signals_available_for_fit": sum(x["status"] == "numeric_payload" for x in report["signal_attempts"]) >= 2,
    "warning": "An empty one-byte response is not a raw signal and is never counted as numeric data.",
}
(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
