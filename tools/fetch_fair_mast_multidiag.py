#!/usr/bin/env python3
"""Fetch multiple FAIR-MAST diagnostics and validate the common TIS analysis path.

The reference shot/group/signal combinations are taken from the public FAIR-MAST
Level-1 documentation for shot 30421. Values are exported without calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tis.core import fft_spectrum, welch_psd, signal_quality
from tis.data_sources import fetch_fair_mast_signal


CASES = (
    ("amc", "plasma_current", "plasma_current"),
    ("ane", "density", "interferometer_density"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mast-multidiag"))
    parser.add_argument("--max-samples", type=int, default=50000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for group, signal, label in CASES:
        frame, manifest = fetch_fair_mast_signal(
            shot=30421, group=group, signal_name=signal, max_samples=args.max_samples
        )
        t = frame["time_s"].to_numpy(dtype=float)
        y = frame["signal_raw"].to_numpy(dtype=float)
        fs = float(manifest["estimated_sampling_rate_hz_after_stride"])
        assert len(frame) >= 1024, f"{label}: too few samples"
        assert np.all(np.isfinite(t)) and np.all(np.isfinite(y)), f"{label}: nonfinite values"
        assert np.all(np.diff(t) > 0), f"{label}: time not strictly increasing"

        # Generic numerical checks only: do not infer calibrated physics from raw/archive units.
        freq, amp = fft_spectrum(y, fs)
        freq_w, psd = welch_psd(y, fs)
        quality = signal_quality(y)
        assert len(freq) > 10 and len(freq_w) > 10, f"{label}: spectral output too short"
        assert np.all(np.isfinite(amp)) and np.all(np.isfinite(psd)), f"{label}: invalid spectrum"

        csv_name = f"mast_30421_{label}.csv"
        manifest_name = f"mast_30421_{label}_manifest.json"
        frame.to_csv(args.output_dir / csv_name, index=False)
        manifest.update({
            "analysis_case": label,
            "csv_file": csv_name,
            "quality_metrics": quality,
            "analysis_checks": ["finite_values", "strictly_increasing_time", "FFT", "Welch PSD"],
            "scientific_limit": (
                "Archive signal imported without calibration. Numerical pipeline passed; "
                "this is not a calibrated physical result or diagnostic-specific validation."
            ),
        })
        (args.output_dir / manifest_name).write_text(
            json.dumps(manifest, indent=2, default=lambda x: float(x)), encoding="utf-8"
        )
        results.append({
            "shot": 30421, "group": group, "signal": signal,
            "samples": len(frame), "estimated_fs_hz": fs,
            "csv": csv_name, "manifest": manifest_name,
        })
        print(
            f"PASS: MAST shot 30421 {group}/{signal}; "
            f"samples={len(frame)}; fs={fs:.3f} Hz; FFT and Welch PSD"
        )

    summary = {
        "dataset": "UKAEA FAIR-MAST Level-1",
        "cases": results,
        "status": "PASS",
        "limit": "Data retrieval and generic numerical pipeline only; no calibration validation.",
    }
    (args.output_dir / "multidiag_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"PASS: {len(results)} additional diagnostic traces exported.")


if __name__ == "__main__":
    main()
