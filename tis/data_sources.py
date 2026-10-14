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



def fetch_public_table_bytes(url: str, max_bytes: int = 100_000_000, timeout: int = 30) -> tuple[bytes, dict[str, str]]:
    """Download one public HTTP(S) tabular file with bounded size and provenance.

    Intended for direct CSV/TXT/Parquet links such as GOLEM's public per-shot
    exports. It deliberately does not crawl websites or scrape HTML pages.
    """
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlparse
    from urllib.request import Request, urlopen

    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a direct public http:// or https:// file URL.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not allowed.")
    if max_bytes < 1024:
        raise ValueError("Maximum download size must be at least 1024 bytes.")

    request = Request(
        parsed.geturl(),
        headers={"User-Agent": "Tokamak-Insight-Suite/0.4 (public scientific data import)"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "unknown")
            final_url = response.geturl()
            payload = response.read(max_bytes + 1)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Public data download failed: {exc}") from exc

    if len(payload) > max_bytes:
        raise ValueError(
            f"File exceeds the {max_bytes:,}-byte download limit; use a smaller export."
        )
    if not payload:
        raise ValueError("The downloaded file is empty.")
    preview = payload[:512].lstrip().lower()
    if b"<html" in preview or b"<!doctype html" in preview:
        raise ValueError(
            "The URL returned an HTML page, not a direct data file. Use the archive's direct download URL."
        )
    import hashlib
    from datetime import datetime, timezone

    metadata = {
        "source_url": str(url).strip(),
        "final_url": final_url,
        "content_type": content_type,
        "downloaded_bytes": str(len(payload)),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return payload, metadata


def fetch_fair_mast_efit_snapshot(
    shot: int = 30421,
    time_s: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch one existing EFIT-derived plasma-current R-Z grid from FAIR-MAST.

    This exports archive output; it does not perform a Grad-Shafranov solve.
    The returned table has z_m plus one column per R coordinate.
    """
    if int(shot) < 1:
        raise ValueError("Shot number must be a positive integer.")
    if not np.isfinite(time_s):
        raise ValueError("Requested EFIT time must be finite.")
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "FAIR-MAST EFIT access needs optional dependencies. "
            "Install with: python -m pip install -r requirements-data.txt"
        ) from exc

    shot_url = f"{FAIR_MAST_BASE}/{int(shot)}.zarr"
    ds = xr.open_zarr(shot_url, group="efm")
    try:
        required = {"plasma_current_rz", "r", "z", "time"}
        missing = required.difference(ds.variables)
        if missing:
            raise KeyError(f"EFIT archive missing required variables: {sorted(missing)}")
        field = ds["plasma_current_rz"]
        if not {"time", "r", "z"}.issubset(field.dims):
            raise ValueError(f"Unexpected plasma_current_rz dimensions: {field.dims}")
        times = np.asarray(ds["time"].values, dtype=np.float64).ravel()
        if not times.size or not np.all(np.isfinite(times)):
            raise ValueError("EFIT time coordinate is empty or nonfinite.")
        index = int(np.argmin(np.abs(times - float(time_s))))
        grid = np.asarray(field.isel(time=index).transpose("z", "r").values, dtype=np.float64)
        r = np.asarray(ds["r"].values, dtype=np.float64).ravel()
        z = np.asarray(ds["z"].values, dtype=np.float64).ravel()
        if grid.shape != (z.size, r.size):
            raise ValueError(f"Grid/coordinate mismatch: field={grid.shape}, z={z.size}, r={r.size}")
        finite = np.isfinite(grid)
        if not np.any(finite):
            raise ValueError("Selected EFIT snapshot contains no finite grid values.")
        frame = pd.DataFrame(grid, columns=[f"R={v:.9g}_m" for v in r])
        frame.insert(0, "z_m", z)
        attrs = dict(ds.attrs)
        manifest = {
            "dataset": "UKAEA FAIR-MAST / MAST Level-1",
            "shot": int(shot),
            "source_group": "efm",
            "signal": "plasma_current_rz",
            "source_url": shot_url,
            "source_documentation": FAIR_MAST_DOCS,
            "source_quality": attrs.get("quality", "Unspecified"),
            "requested_time_s": float(time_s),
            "selected_time_s": float(times[index]),
            "time_index": index,
            "grid_shape_z_by_r": [int(grid.shape[0]), int(grid.shape[1])],
            "r_min_m": float(np.min(r)),
            "r_max_m": float(np.max(r)),
            "z_min_m": float(np.min(z)),
            "z_max_m": float(np.max(z)),
            "finite_grid_fraction": float(np.mean(finite)),
            "grid_min_finite": float(np.min(grid[finite])),
            "grid_max_finite": float(np.max(grid[finite])),
            "grid_mean_finite": float(np.mean(grid[finite])),
            "scientific_limit": (
                "Exports an existing EFIT-derived archive grid only. TIS does not solve "
                "the Grad-Shafranov equation here; verify coordinate conventions, units, "
                "quality flags, and reconstruction assumptions before scientific use."
            ),
        }
        for var, key in (("current_centrd_r", "current_centroid_r_m"),
                         ("current_centrd_z", "current_centroid_z_m")):
            if var in ds:
                value = np.asarray(ds[var].isel(time=index).values).item()
                if np.isfinite(value):
                    manifest[key] = float(value)
        return frame, manifest
    finally:
        ds.close()
