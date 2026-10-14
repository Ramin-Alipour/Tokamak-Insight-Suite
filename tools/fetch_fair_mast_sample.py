#!/usr/bin/env python3
"""Fetch a real, public MAST Level-1 Mirnov signal for TIS integration testing.

Source documented by the UKAEA FAIR-MAST / MAST Data Catalog:
https://mastapp.site/level1-data.html
Shot 29790, source group xmo, signal omaha_1lz. The source catalog labels this
signal as raw and reports 1,400,000 time samples. Values are in the source's
recorded units (shown as arbitrary units); this script does not claim absolute
magnetic calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import welch

DEFAULT_URL = "https://s3.echo.stfc.ac.uk/mast/level1/shots/29790.zarr"
DEFAULT_GROUP = "xmo"
DEFAULT_SIGNAL = "omaha_1lz"
SOURCE_PAGE = "https://mastapp.site/level1-data.html"


def fetch_sample(output_dir: Path, max_samples: int = 70000) -> dict:
    if max_samples < 1024:
        raise ValueError("max_samples must be at least 1024")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the published Zarr store remotely; do not download the full 100 MB
    # dataset or commit third-party experimental data into the source repository.
    ds = xr.open_zarr(DEFAULT_URL, group=DEFAULT_GROUP, consolidated=False)
    try:
        if "time" not in ds.coords or DEFAULT_SIGNAL not in ds:
            raise KeyError(f"Expected time coordinate and {DEFAULT_SIGNAL!r} signal in source group {DEFAULT_GROUP!r}")
        time = np.asarray(ds["time"].values, dtype=np.float64).ravel()
        signal = np.asarray(ds[DEFAULT_SIGNAL].values, dtype=np.float64).ravel()
        if time.size != signal.size or time.size < 1024:
            raise ValueError(f"Unexpected signal/time dimensions: {signal.size} and {time.size}")
        finite = np.isfinite(time) & np.isfinite(signal)
        time, signal = time[finite], signal[finite]
        if time.size < 1024 or not np.all(np.diff(time) > 0):
            raise ValueError("Source must contain at least 1024 finite samples with strictly increasing time")
        original_count = int(time.size)
        stride = max(1, int(np.ceil(original_count / max_samples)))
        time, signal = time[::stride], signal[::stride]
        if time.size < 1024:
            raise ValueError("Downsampled signal unexpectedly has fewer than 1024 samples")

        frame = pd.DataFrame({"time_s": time, "signal_raw": signal})
        csv_path = output_dir / "mast_29790_mirnov_omaha_1lz.csv"
        frame.to_csv(csv_path, index=False)

        dt = float(np.median(np.diff(time)))
        fs = 1.0 / dt
        demeaned = signal - np.mean(signal)
        freq, psd = welch(demeaned, fs=fs, nperseg=min(8192, signal.size))
        if freq.size > 1 and np.any(psd[1:] > 0):
            peak_hz = float(freq[1 + int(np.argmax(psd[1:]))])
        else:
            peak_hz = None
        manifest = {
            "dataset": "FAIR-MAST / UKAEA public MAST Level-1 data",
            "shot": 29790,
            "source_group": DEFAULT_GROUP,
            "signal": DEFAULT_SIGNAL,
            "source_url": DEFAULT_URL,
            "source_documentation": SOURCE_PAGE,
            "source_signal_class": "Raw",
            "source_units_note": "Catalog reports arbitrary units; no absolute calibration is applied.",
            "original_finite_samples": original_count,
            "exported_samples": int(signal.size),
            "downsample_stride": stride,
            "time_start_s": float(time[0]),
            "time_end_s": float(time[-1]),
            "estimated_sampling_rate_hz_after_stride": fs,
            "signal_min": float(np.min(signal)),
            "signal_max": float(np.max(signal)),
            "signal_mean": float(np.mean(signal)),
            "signal_std": float(np.std(signal)),
            "welch_peak_frequency_hz_excluding_dc": peak_hz,
            "csv_file": csv_path.name,
            "scientific_limit": "Raw-signal software integration test only; not a calibrated physical result or validation of all TIS diagnostics.",
        }
        manifest_path = output_dir / "mast_29790_mirnov_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    finally:
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mast-real-data"))
    parser.add_argument("--max-samples", type=int, default=70000)
    args = parser.parse_args()
    manifest = fetch_sample(args.output_dir, args.max_samples)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
