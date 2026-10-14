"""Public tokamak data access helpers with explicit provenance.

Only public archive endpoints are used. The caller remains responsible for
checking calibration, units, quality flags, and the validity of any inference.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

FAIR_MAST_BASE = "https://s3.echo.stfc.ac.uk/mast/level1/shots"
FAIR_MAST_DOCS = "https://mastapp.site/level1-data.html"


def fetch_fair_mast_signal(
    shot: int,
    group: str,
    signal_name: str | None = None,
    max_samples: int = 70000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch one 1-D signal from the public FAIR-MAST Zarr archive.

    Uses the archive's documented direct shot URL pattern. Returns a
    time_s/signal_raw table and a provenance manifest. Data are downsampled
    by stride only when necessary; no calibration is applied.
    """
    if int(shot) < 1:
        raise ValueError("Shot number must be a positive integer.")
    group = str(group).strip()
    if not group or "/" in group or "\\" in group:
        raise ValueError("Group must be a single FAIR-MAST group name.")
    if max_samples < 1024:
        raise ValueError("max_samples must be at least 1024.")

    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "FAIR-MAST access needs optional dependencies. Install with: "
            "python -m pip install -r requirements-data.txt"
        ) from exc

    shot_url = f"{FAIR_MAST_BASE}/{int(shot)}.zarr"
    ds = xr.open_zarr(shot_url, group=group)
    try:
        # The documented FAIR-MAST groups usually expose time as a coordinate;
        # time1 can also be present as a data variable on the same dimension.
        time_name = next(
            (
                name
                for name in ("time", "time1")
                if name in ds.variables and ds[name].ndim == 1
            ),
            None,
        )
        if time_name is None:
            raise KeyError(
                f"No 1-D time coordinate found in shot {shot}, group {group}. "
                f"Available variables: {list(ds.variables)}"
            )
        time_dim = ds[time_name].dims[0]

        if signal_name:
            if signal_name not in ds.data_vars:
                raise KeyError(
                    f"Signal {signal_name!r} not found in shot {shot}, group {group}. "
                    f"Available signals: {list(ds.data_vars)}"
                )
            selected = signal_name
            if ds[selected].dims != (time_dim,):
                raise ValueError(
                    f"Signal {selected!r} is not a 1-D trace on dimension {time_dim!r}; "
                    f"dimensions are {ds[selected].dims}."
                )
        else:
            candidates = list(ds.data_vars)
            if group.lower() == "xmo":
                preferred = ["omaha_1lz", "omaha_3lz", "omaha_2lz"]
                candidates = preferred + [name for name in candidates if name not in preferred]
            selected = next(
                (
                    name for name in candidates
                    if ds[name].dims == (time_dim,)
                    and np.issubdtype(ds[name].dtype, np.number)
                    and name not in {"time", "time1"}
                ),
                None,
            )
            if selected is None:
                raise KeyError(
                    f"No 1-D numeric signal found on dimension {time_dim!r}. "
                    f"Available data variables: {list(ds.data_vars)}"
                )

        time = np.asarray(ds[time_name].values, dtype=np.float64).ravel()
        values = np.asarray(ds[selected].values, dtype=np.float64).ravel()
        if time.size != values.size or time.size < 1024:
            raise ValueError(
                f"Unexpected time/signal sizes: {time.size} and {values.size}."
            )
        finite = np.isfinite(time) & np.isfinite(values)
        time, values = time[finite], values[finite]
        if time.size < 1024 or not np.all(np.diff(time) > 0):
            raise ValueError(
                "Source must contain at least 1024 finite samples with strictly increasing time."
            )

        original_count = int(time.size)
        stride = max(1, int(math.ceil(original_count / max_samples)))
        time, values = time[::stride], values[::stride]
        if time.size < 1024:
            raise ValueError("Downsampled signal unexpectedly has fewer than 1024 samples.")

        dt = float(np.median(np.diff(time)))
        fs = 1.0 / dt
        frame = pd.DataFrame({"time_s": time, "signal_raw": values})
        attrs = dict(ds.attrs)
        signal_attrs = dict(ds[selected].attrs)
        manifest = {
            "dataset": "UKAEA FAIR-MAST / MAST Level-1",
            "shot": int(shot),
            "source_group": group,
            "signal": selected,
            "source_url": shot_url,
            "source_documentation": FAIR_MAST_DOCS,
            "source_signal_class": attrs.get("signal_type", "Unknown"),
            "source_quality": attrs.get("quality", "Unspecified"),
            "source_units": signal_attrs.get("units", "Not specified in variable attributes"),
            "original_finite_samples": original_count,
            "exported_samples": int(values.size),
            "downsample_stride": stride,
            "time_start_s": float(time[0]),
            "time_end_s": float(time[-1]),
            "estimated_sampling_rate_hz_after_stride": fs,
            "signal_min": float(np.min(values)),
            "signal_max": float(np.max(values)),
            "signal_mean": float(np.mean(values)),
            "signal_std": float(np.std(values)),
            "scientific_limit": (
                "Data are exported without calibration. Verify source units, quality, "
                "instrument response, and diagnostic-specific assumptions before physical interpretation."
            ),
        }
        return frame, manifest
    finally:
        ds.close()
