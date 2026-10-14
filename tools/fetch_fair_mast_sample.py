#!/usr/bin/env python3
"""Fetch a real, public MAST Level-1 Mirnov signal for TIS integration testing.

Source: UKAEA FAIR-MAST / MAST Data Catalog, shot 29790, group xmo.
The source catalog labels the group Raw and reports 1,400,000 time samples.
No absolute magnetic calibration is claimed.
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
    # Open the public Zarr store remotely; don't commit the full experimental data.
    ds = xr.open_zarr(DEFAULT_URL, group=DEFAULT_GROUP, consolidated=False)
    try:
        # Tolerate catalog/store revisions in coordinate and channel naming.
        time_name = next((name for name in ("time", "time1") if name in ds.variables), None)
        if time_name is None:
            raise KeyError(f"No time coordinate found; available variables: {list(ds.variables)}")
        candidates = [DEFAULT_SIGNAL, "omaha_3lz", "omaha_2lz", "omaha_1lz"]
        signal_name = next((name for name in candidates if name in ds.data_vars and ds[name].dims == (time_name,)), None)
        if signal_name is None:
            signal_name = next((name for name in ds.data_vars if name.startswith("omaha_") and ds[name].dims == (time_name,)), None)
        if signal_name is None:
            raise KeyError(f"No one-dimensional omaha Mirnov signal on {time_name!r}; data variables: {list(ds.data_vars)}")
        time = np.asarray(ds[time_name].values, dtype=np.float64).ravel()
        signal = np.asarray(ds[signal_name].values, dtype=np.float64).ravel()
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

        csv_name = f"mast_29790_mirnov_{signal_name}.csv"
        frame = pd.DataFrame({"time_s": time, "signal_raw": signal})
        frame.to_csv(output_dir / csv_name, index=False)
        dt = float(np.median(np.diff(time)))
        fs = 1.0 / dt
        freq, psd = welch(signal - np.mean(signal), fs=fs, nperseg=min(8192, signal.size))
        peak_hz = float(freq[1 + int(np.argmax(psd[1:]))]) if freq.size > 1 and np.any(psd[1:] > 0) else None
        manifest = {
            "dataset": "FAIR-MAST / UKAEA public MAST Level-1 data",
            "shot": 29790,
            "source_group": DEFAULT_GROUP,
            "signal": signal_name,
            "source_url": DEFAULT_URL,
            "source_documentation": SOURCE_PAGE,
            "source_signal_class": "Raw",
            "source_units_note": "Raw source signal; interpret units/calibration using original channel metadata.",
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
            "csv_file": csv_name,
            "scientific_limit": "Raw-signal software integration test only; not a calibrated physical result or validation of all TIS diagnostics.",
        }
        (output_dir / "mast_29790_mirnov_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    finally:
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mast-real-data"))
    parser.add_argument("--max-samples", type=int, default=70000)
    args = parser.parse_args()
    print(json.dumps(fetch_sample(args.output_dir, args.max_samples), indent=2))


if __name__ == "__main__":
    main()
