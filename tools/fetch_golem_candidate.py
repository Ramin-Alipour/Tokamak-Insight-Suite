#!/usr/bin/env python3
"""Acquire GOLEM spectrum and temperature candidates with reproducible provenance.

Acquisition is not scientific validation: the script never labels a candidate as
raw or treats a published summary value as a matched reference automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

BASE = "https://golem.fjfi.cvut.cz"
API_BASE = "http://golem.fjfi.cvut.cz"
USER_AGENT = "Tokamak-Insight-Suite empirical-validation inventory/1.1"


def get(url: str) -> tuple[bytes, str, str]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=20) as response:
        data = response.read()
        content_type = response.headers.get("Content-Type", "")
        return data, response.geturl(), content_type


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, default=30622)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/golem-empirical-candidate"))
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    shot = args.shot
    page_url = f"{BASE}/shots/{shot}/Data.php"
    page, final_page_url, page_type = get(page_url)
    (out / "shot_page.html").write_bytes(page)

    # The GOLEM Data page displays filenames, while its documented data API
    # accepts identifiers under /utils/data/<shot>/<identifier>[.txt].
    # Try both documented identifiers and legacy operation-directory paths.
    candidates = [
        ("spectrum_identifier", f"{API_BASE}/utils/data/{shot}/spectra.txt_Data_.txt"),
        ("spectrum_identifier_no_extension", f"{API_BASE}/utils/data/{shot}/spectra.txt_Data_"),
        ("spectrum_config_identifier", f"{API_BASE}/utils/data/{shot}/spectrometr:data"),
        ("calibrated_spectrum_identifier", f"{API_BASE}/utils/data/{shot}/spectrometr:data_calib"),
        ("calibrated_spectrum_database_path", f"{API_BASE}/golem/database/operation/shots/{shot}/diagnostics/Radiation/1111Spectrometer.ON/data/spectrum_calibrated"),
        ("spectrum_legacy_path", f"{BASE}/operation/shots/{shot}/diagnostics/Radiation/1111Spectrometer.ON/data/spectra.txt_Data_.txt"),
        ("impurity_temperature_identifier", f"{API_BASE}/utils/data/{shot}/spectrometr:temperature"),
        ("impurity_temperature_xls", f"{API_BASE}/utils/data/{shot}/spectrometr:temperature.xls"),
        ("impurity_temperature_database_path", f"{API_BASE}/golem/database/operation/shots/{shot}/analysis/Radiation/1212Impurities_TO.ON/data/temperature"),
        ("electron_temperature_mean_database_path", f"{API_BASE}/golem/database/operation/shots/{shot}/basicdiagn/ElectronTemperatureMean"),
        ("temperature_identifier", f"{API_BASE}/utils/data/{shot}/temperature"),
        ("temperature_mean_identifier", f"{API_BASE}/utils/data/{shot}/temperature_mean"),
        ("temperature_legacy_path", f"{BASE}/operation/shots/{shot}/temperature"),
        ("temperature_mean_legacy_path", f"{BASE}/operation/shots/{shot}/temperature_mean"),
    ]
    attempts, downloaded = [], []
    seen = set()
    for label, url in candidates:
        if url in seen:
            continue
        seen.add(url)
        record = {"label": label, "url": url}
        try:
            data, final_url, content_type = get(url)
            # Avoid mistaking an HTML error page served with status 200 for data.
            if "text/html" in content_type.lower() and b"<html" in data[:1000].lower():
                record.update(status="rejected_html", final_url=final_url, content_type=content_type,
                              bytes=len(data), reason="HTML response is not a data file")
            elif len(data) < 8:
                record.update(status="rejected_too_small", final_url=final_url, bytes=len(data))
            else:
                suffix = Path(urlparse(final_url).path).suffix or ".txt"
                filename = f"{label}{suffix}"
                (out / filename).write_bytes(data)
                record.update(status="downloaded", final_url=final_url, content_type=content_type,
                              file=filename, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                downloaded.append(record.copy())
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        attempts.append(record)

    manifest = {
        "shot": shot,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "page_url": page_url,
        "final_page_url": final_page_url,
        "page_content_type": page_type,
        "page_bytes": len(page),
        "page_sha256": hashlib.sha256(page).hexdigest(),
        "attempts": attempts,
        "downloaded_files": downloaded,
        "public_page_reference_temperature_eV": 20.8 if shot == 30622 else None,
        "classification": "candidate files only; raw/calibrated/processed status not established; no empirical validation claimed",
        "acceptance_checks": [
            "Confirm downloaded file format and whether the spectrum is raw counts or corrected/calibrated data.",
            "Establish temporal/spatial correspondence between spectrum and the published temperature reference.",
            "Independently derive the same observable with documented uncertainty and calculate absolute/relative discrepancy.",
            "Add the verified input and numerical comparison as a deterministic regression test before declaring validation.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not downloaded:
        raise SystemExit("No candidate data files downloaded; this run is not an empirical-validation pass.")


if __name__ == "__main__":
    main()
