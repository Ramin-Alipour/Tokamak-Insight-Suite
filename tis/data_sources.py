"""Public tokamak data access helpers with explicit provenance.

Only public archive endpoints are used. The caller remains responsible for
checking calibration, units, quality flags, and the validity of any inference.
"""
from __future__ import annotations

import math
import ipaddress
import socket
from urllib.parse import urlparse
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

FAIR_MAST_BASE = "https://s3.echo.stfc.ac.uk/mast/level1/shots"
FAIR_MAST_DOCS = "https://mastapp.site/level1-data.html"



def _is_finite_real_scalar(value: Any) -> bool:
    """Return whether a public numeric parameter is a finite real scalar (not bool)."""
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _safe_provenance_url(value: str) -> str:
    """Remove query strings and fragments that may contain signed-link secrets."""
    parsed = urlparse(str(value).strip())
    return parsed._replace(query="", fragment="").geturl()


def _open_zarr_with_retry(xr: Any, url: str, group: str, attempts: int = 3,
                          timeout_s: float = 30.0):
    """Open a public Zarr group with bounded request timeout and transient retries."""
    import time
    if not _is_finite_real_scalar(attempts) or int(attempts) != attempts or attempts < 1 or attempts > 5:
        raise ValueError("attempts must be an integer in [1, 5].")
    if not _is_finite_real_scalar(timeout_s) or timeout_s <= 0 or timeout_s > 120:
        raise ValueError("timeout_s must be positive and no greater than 120 seconds.")
    last_error = None
    for attempt in range(int(attempts)):
        try:
            return xr.open_zarr(url, group=group, storage_options={"timeout": float(timeout_s)})
        except Exception as exc:
            last_error = exc
            if attempt + 1 < int(attempts):
                time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(
        f"Could not open public Zarr group {group!r} after {attempts} bounded attempts: {last_error}"
    ) from last_error



def anti_alias_decimate(time_s: Any, values: Any, max_samples: int = 70000) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Anti-alias-filter a uniformly sampled trace before integer-factor downsampling."""
    t = np.asarray(time_s, dtype=np.float64).ravel()
    x = np.asarray(values, dtype=np.float64).ravel()
    if len(t) != len(x) or len(t) < 1024:
        raise ValueError("Time and signal must have equal lengths and at least 1024 samples.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)) or not np.all(np.diff(t) > 0):
        raise ValueError("Time and signal must be finite and timestamps strictly increasing.")
    if not _is_finite_real_scalar(max_samples) or int(max_samples) != max_samples or max_samples < 1024:
        raise ValueError("max_samples must be an integer of at least 1024.")
    dt = np.diff(t)
    median_dt = float(np.median(dt))
    jitter = float(np.std(dt) / median_dt)
    if np.max(dt) > 1.05 * median_dt or jitter > 0.02:
        raise ValueError("Source timestamps are irregular or contain gaps; resample explicitly before anti-aliased decimation.")
    stride = max(1, int(math.ceil(len(x) / int(max_samples))))
    original_fs = 1.0 / median_dt
    if stride > 1:
        x_out = resample_poly(x, up=1, down=stride, window=("kaiser", 8.6), padtype="line")
        t_out = t[::stride][:len(x_out)]
    else:
        t_out, x_out = t.copy(), x.copy()
    metadata = {
        "original_samples": int(len(x)),
        "exported_samples": int(len(x_out)),
        "stride": stride,
        "source_sampling_rate_hz": original_fs,
        "exported_sampling_rate_hz": 1.0 / float(np.median(np.diff(t_out))),
        "anti_alias_filter_applied": bool(stride > 1),
        "method": "polyphase FIR Kaiser-window low-pass" if stride > 1 else "none",
    }
    return t_out, x_out, metadata


def fetch_fair_mast_signal(
    shot: int,
    group: str,
    signal_name: str | None = None,
    max_samples: int = 70000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch one 1-D signal from the public FAIR-MAST Zarr archive.

    Uses the archive's documented direct shot URL pattern. Returns a
    time_s/signal_raw table and provenance manifest. Downsampling uses polyphase
    FIR anti-alias filtering; no calibration is applied.
    """
    if isinstance(shot, bool) or not isinstance(shot, (int, np.integer)) or shot < 1:
        raise ValueError("Shot number must be a positive integer.")
    group = str(group).strip()
    if not group or "/" in group or "\\" in group:
        raise ValueError("Group must be a single FAIR-MAST group name.")
    if isinstance(max_samples, bool) or not isinstance(max_samples, (int, np.integer)) or max_samples < 1024:
        raise ValueError("max_samples must be an integer of at least 1024.")

    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "FAIR-MAST access needs optional dependencies. Install with: "
            "python -m pip install -r requirements-data.txt"
        ) from exc

    shot_url = f"{FAIR_MAST_BASE}/{int(shot)}.zarr"
    ds = _open_zarr_with_retry(xr, shot_url, group)
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
        time, values, decimation = anti_alias_decimate(time, values, max_samples=max_samples)
        stride = int(decimation["stride"])
        fs = float(decimation["exported_sampling_rate_hz"])
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
            "downsampling_method": decimation["method"],
            "anti_alias_filter_applied": decimation["anti_alias_filter_applied"],
            "source_sampling_rate_hz": decimation["source_sampling_rate_hz"],
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
    """Download public HTTP(S) data with bounded reads and DNS-pinned connections."""
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, build_opener, HTTPRedirectHandler, HTTPHandler, HTTPSHandler, ProxyHandler
    import http.client
    import ssl

    parsed=urlparse(str(url).strip())
    if parsed.scheme not in {"http","https"} or not parsed.hostname:
        raise ValueError("Enter a direct public http:// or https:// file URL.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not allowed.")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1024 <= max_bytes <= 1_000_000_000:
        raise ValueError("Maximum download size must be an integer between 1024 bytes and 1,000,000,000 bytes.")
    if not _is_finite_real_scalar(timeout) or timeout <= 0:
        raise ValueError("Timeout must be a positive finite number of seconds.")
    pinned_hosts: dict[str,str]={}

    def pin_destination(target: str) -> str:
        parts=urlparse(target); host=(parts.hostname or "").rstrip(".").lower()
        if parts.scheme not in {"http","https"} or not host or parts.username or parts.password:
            raise ValueError("Only credential-free HTTP(S) destinations are allowed.")
        if host in {"localhost","localhost.localdomain","metadata.google.internal"} or host.endswith((".localhost",".local",".internal")):
            raise ValueError("Local and internal network destinations are blocked.")
        port=parts.port or (443 if parts.scheme=="https" else 80)
        try:
            addresses=[ipaddress.ip_address(host)]
        except ValueError:
            try:
                resolved=socket.getaddrinfo(host,port,type=socket.SOCK_STREAM)
                addresses=sorted({ipaddress.ip_address(item[4][0]) for item in resolved},key=lambda ip:(ip.version,int(ip)))
            except (OSError,ValueError) as exc:
                raise ValueError(f"Could not resolve host {host!r}.") from exc
        if not addresses or any(not ip.is_global for ip in addresses):
            raise ValueError("Private, loopback, link-local, reserved and non-global destinations are blocked.")
        pinned_hosts[host]=str(addresses[0])
        return host

    pin_destination(parsed.geturl())

    class PinnedHTTPConnection(http.client.HTTPConnection):
        def __init__(self,host,pinned_ip,**kwargs):
            kwargs.pop("check_hostname",None); self._pinned_ip=pinned_ip
            super().__init__(host,**kwargs)
        def connect(self):
            self.sock=socket.create_connection((self._pinned_ip,self.port),self.timeout,self.source_address)

    class PinnedHTTPSConnection(http.client.HTTPSConnection):
        def __init__(self,host,pinned_ip,**kwargs):
            kwargs.pop("check_hostname",None); self._pinned_ip=pinned_ip
            super().__init__(host,**kwargs)
        def connect(self):
            raw_sock=socket.create_connection((self._pinned_ip,self.port),self.timeout,self.source_address)
            try: self.sock=self._context.wrap_socket(raw_sock,server_hostname=self.host)
            except Exception:
                raw_sock.close(); raise

    class PinnedHTTPHandler(HTTPHandler):
        def http_open(self,req):
            host=(urlparse(req.full_url).hostname or "").rstrip(".").lower()
            return self.do_open(lambda connection_host,**kwargs:
                PinnedHTTPConnection(connection_host,pinned_hosts[host],**kwargs),req)

    class PinnedHTTPSHandler(HTTPSHandler):
        def https_open(self,req):
            host=(urlparse(req.full_url).hostname or "").rstrip(".").lower()
            return self.do_open(lambda connection_host,**kwargs:
                PinnedHTTPSConnection(connection_host,pinned_hosts[host],**kwargs),req,
                context=ssl.create_default_context(),check_hostname=True)

    class SafeRedirectHandler(HTTPRedirectHandler):
        def redirect_request(self,req,fp,code,msg,headers,newurl):
            pin_destination(newurl)
            return super().redirect_request(req,fp,code,msg,headers,newurl)

    request=Request(parsed.geturl(),headers={"User-Agent":"Tokamak-Insight-Suite/0.5 (public scientific data import)"})
    opener=build_opener(ProxyHandler({}),PinnedHTTPHandler(),PinnedHTTPSHandler(),SafeRedirectHandler())
    try:
        with opener.open(request,timeout=timeout) as response:
            content_type=response.headers.get("Content-Type","unknown")
            final_url=response.geturl()
            final_host=(urlparse(final_url).hostname or "").rstrip(".").lower()
            if final_host not in pinned_hosts: pin_destination(final_url)
            payload=response.read(max_bytes+1)
    except (HTTPError,URLError,TimeoutError,OSError,ssl.SSLError,ValueError) as exc:
        if isinstance(exc,ValueError): raise
        raise RuntimeError(f"Public data download failed: {exc}") from exc
    if len(payload)>max_bytes:
        raise ValueError(f"File exceeds the {max_bytes:,}-byte download limit; use a smaller export.")
    if not payload: raise ValueError("The downloaded file is empty.")
    preview=payload[:512].lstrip().lower()
    if b"<html" in preview or b"<!doctype html" in preview:
        raise ValueError("The URL returned an HTML page, not a direct data file. Use the archive's direct download URL.")
    import hashlib
    from datetime import datetime,timezone
    source_parsed = urlparse(str(url).strip())
    final_parsed = urlparse(final_url)
    metadata={"source_url":_safe_provenance_url(str(url).strip()),"final_url":_safe_provenance_url(final_url),
              "query_parameters_redacted":bool(source_parsed.query or final_parsed.query),
              "content_type":content_type,"downloaded_bytes":str(len(payload)),
              "downloaded_at_utc":datetime.now(timezone.utc).isoformat(),
              "sha256":hashlib.sha256(payload).hexdigest()}
    return payload,metadata


def fetch_fair_mast_efit_snapshot(
    shot: int = 30421,
    time_s: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch one existing EFIT-derived plasma-current R-Z grid from FAIR-MAST.

    This exports archive output; it does not perform a Grad-Shafranov solve.
    The returned table has z_m plus one column per R coordinate.
    """
    if isinstance(shot, bool) or not isinstance(shot, (int, np.integer)) or shot < 1:
        raise ValueError("Shot number must be a positive integer.")
    if not _is_finite_real_scalar(time_s):
        raise ValueError("Requested EFIT time must be a finite real scalar.")
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "FAIR-MAST EFIT access needs optional dependencies. "
            "Install with: python -m pip install -r requirements-data.txt"
        ) from exc

    shot_url = f"{FAIR_MAST_BASE}/{int(shot)}.zarr"
    ds = _open_zarr_with_retry(xr, shot_url, "efm")
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
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(z)):
            raise ValueError("EFIT R and Z coordinates must contain only finite values.")
        if len(np.unique(r)) != len(r) or len(np.unique(z)) != len(z):
            raise ValueError("EFIT R and Z coordinates must be unique.")
        # Normalize descending archive axes to increasing coordinates without
        # detaching the current-density grid from its physical coordinates.
        r_order = np.argsort(r)
        z_order = np.argsort(z)
        r = r[r_order]
        z = z[z_order]
        grid = grid[np.ix_(z_order, r_order)]
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


def list_fair_mast_signals(shot: int, group: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """List available variables in a FAIR-MAST group without downloading signal arrays.

    The table marks numeric one-dimensional time traces that can be imported by
    :func:`fetch_fair_mast_signal`. Multidimensional profiles/images are listed
    for discovery but are not misrepresented as supported by the 1-D importer.
    """
    if isinstance(shot, bool) or not isinstance(shot, (int, np.integer)) or shot < 1:
        raise ValueError("Shot number must be a positive integer.")
    group = str(group).strip()
    if not group or "/" in group or "\\" in group:
        raise ValueError("Group must be a single FAIR-MAST group name.")
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "FAIR-MAST access needs optional dependencies. Install with: "
            "python -m pip install -r requirements-data.txt"
        ) from exc

    shot_url = f"{FAIR_MAST_BASE}/{int(shot)}.zarr"
    ds = _open_zarr_with_retry(xr, shot_url, group)
    try:
        time_name = next(
            (name for name in ("time", "time1")
             if name in ds.variables and ds[name].ndim == 1),
            None,
        )
        time_dim = ds[time_name].dims[0] if time_name else None
        rows = []
        for name, variable in ds.data_vars.items():
            dims = tuple(str(d) for d in variable.dims)
            numeric = bool(np.issubdtype(variable.dtype, np.number))
            is_trace = bool(numeric and time_dim and dims == (time_dim,))
            rows.append({
                "signal": str(name),
                "dimensions": " × ".join(dims),
                "shape": " × ".join(str(int(n)) for n in variable.shape),
                "dtype": str(variable.dtype),
                "units": str(variable.attrs.get("units", "Not specified")),
                "description": str(variable.attrs.get("description", "")),
                "numeric": numeric,
                "time_series_import_supported": is_trace,
            })
        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(
                ["time_series_import_supported", "signal"],
                ascending=[False, True],
            ).reset_index(drop=True)
        manifest = {
            "dataset": "UKAEA FAIR-MAST / MAST Level-1",
            "shot": int(shot),
            "source_group": group,
            "source_url": shot_url,
            "source_documentation": FAIR_MAST_DOCS,
            "source_quality": ds.attrs.get("quality", "Unspecified"),
            "source_signal_class": ds.attrs.get("signal_type", "Unknown"),
            "time_coordinate": time_name,
            "variable_count": int(len(table)),
            "one_dimensional_numeric_time_traces": int(
                table["time_series_import_supported"].sum()
            ) if not table.empty else 0,
            "note": (
                "Metadata inventory only. Multidimensional profiles/images may require "
                "dedicated diagnostic-specific loaders; this inventory does not load or "
                "validate their full data arrays."
            ),
        }
        return table, manifest
    finally:
        ds.close()



def fetch_public_efit_flux_map(
    sample_index: int = 0,
    time_s: float = 0.2,
    *,
    config: str = "diii_d_train",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fetch a real EFIT poloidal-flux map from the open Fusion Equilibrium Challenge.

    The dataset provides EFIT target maps, physical R/Z coordinates, and EFIT times.
    Shot identifiers are intentionally anonymized; sample_index selects a row in
    the public training split. Requested time is in seconds; the source clock is ms.
    """
    if isinstance(sample_index, (bool, np.bool_)) or not isinstance(sample_index, (int, np.integer)) or sample_index < 0:
        raise ValueError("sample_index must be a non-negative integer")
    if not _is_finite_real_scalar(time_s) or time_s < 0:
        raise ValueError("time_s must be a finite non-negative number")
    if config not in {"diii_d_train"}:
        raise ValueError("Only the public DIII-D training split contains EFIT target maps.")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Public EFIT dataset access needs the optional data dependencies. "
            "Install with: python -m pip install -r requirements-data.txt"
        ) from exc
    from itertools import islice
    dataset = load_dataset(
        "Sophelio/fusion-equilibrium-challenge", config, split="train", streaming=True
    )
    row = next(islice(iter(dataset), int(sample_index), int(sample_index) + 1), None)
    if row is None:
        raise IndexError(f"Public EFIT sample index {sample_index} is outside the available split.")
    required = {"efit_times", "efit_grid_R", "efit_grid_Z", "efit_psirz"}
    missing = required.difference(row)
    if missing:
        raise KeyError(f"Public EFIT dataset sample is missing fields: {sorted(missing)}")
    times_ms = np.asarray(row["efit_times"], dtype=float).ravel()
    R = np.asarray(row["efit_grid_R"], dtype=float).ravel()
    Z = np.asarray(row["efit_grid_Z"], dtype=float).ravel()
    flux_stack = np.asarray(row["efit_psirz"], dtype=float)
    if (times_ms.size == 0 or not np.all(np.isfinite(times_ms))
            or not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z))
            or np.any(np.diff(R) <= 0) or np.any(np.diff(Z) <= 0)):
        raise ValueError("Public EFIT time and coordinate arrays must be finite and strictly increasing.")
    if flux_stack.ndim != 3 or flux_stack.shape[0] != times_ms.size:
        raise ValueError("Public EFIT flux must have shape (n_times, n_Z, n_R).")
    if flux_stack.shape[1:] == (R.size, Z.size) and (R.size != Z.size):
        flux_stack = np.transpose(flux_stack, (0, 2, 1))
    if flux_stack.shape[1:] != (Z.size, R.size):
        raise ValueError(
            f"Public EFIT flux shape {flux_stack.shape[1:]} does not match (n_Z, n_R)={(Z.size, R.size)}."
        )
    selected = int(np.argmin(np.abs(times_ms - float(time_s) * 1000.0)))
    psi = np.asarray(flux_stack[selected], dtype=float)
    if not np.all(np.isfinite(psi)) or float(np.ptp(psi)) <= np.finfo(float).tiny:
        raise ValueError("Selected public EFIT flux map is non-finite or degenerate.")
    manifest = {
        "dataset": "Sophelio/Fusion Equilibrium Challenge",
        "source_url": "https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge",
        "license": "CC BY 4.0",
        "config": config,
        "sample_index": int(sample_index),
        "requested_time_s": float(time_s),
        "selected_time_ms": float(times_ms[selected]),
        "time_index": selected,
        "grid_shape_z_by_r": [int(Z.size), int(R.size)],
        "R_range_m": [float(R[0]), float(R[-1])],
        "Z_range_m": [float(Z[0]), float(Z[-1])],
        "flux_min_Wb_rad": float(np.min(psi)),
        "flux_max_Wb_rad": float(np.max(psi)),
        "scientific_limit": (
            "The reference is an existing EFIT reconstruction. The TIS profile-estimation "
            "workflow derives a source profile from this same map and reports an in-sample "
            "comparison, not an independent experimental validation."
        ),
    }
    return R, Z, psi, manifest
