#!/usr/bin/env python3
"""Fetch a public FAIR-MAST Mirnov signal and exercise TIS analysis routines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Make the repository root importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.signal import welch

from tis.core import fft_spectrum, welch_psd
from tis.data_sources import fetch_fair_mast_signal


def fetch_sample(output_dir: Path, max_samples: int = 70000) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, manifest = fetch_fair_mast_signal(
        shot=29790, group="xmo", signal_name="omaha_1lz", max_samples=max_samples
    )
    values = frame["signal_raw"].to_numpy(dtype=float)
    fs = float(manifest["estimated_sampling_rate_hz_after_stride"])
    freq, psd = welch(values - np.mean(values), fs=fs, nperseg=min(8192, values.size))
    peak_hz = (
        float(freq[1 + int(np.argmax(psd[1:]))])
        if freq.size > 1 and np.any(psd[1:] > 0)
        else None
    )
    csv_name = f"mast_29790_mirnov_{manifest['signal']}.csv"
    frame.to_csv(output_dir / csv_name, index=False)
    manifest["welch_peak_frequency_hz_excluding_dc"] = peak_hz
    manifest["csv_file"] = csv_name
    manifest["scientific_limit"] = (
        "Raw-signal software integration test only; not a calibrated physical result "
        "or validation of all TIS diagnostics."
    )
    (output_dir / "mast_29790_mirnov_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mast-real-data"))
    parser.add_argument("--max-samples", type=int, default=70000)
    args = parser.parse_args()
    print(json.dumps(fetch_sample(args.output_dir, args.max_samples), indent=2))


if __name__ == "__main__":
    main()
