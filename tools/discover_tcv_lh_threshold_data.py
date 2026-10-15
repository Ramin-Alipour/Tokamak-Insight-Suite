#!/usr/bin/env python3
"""Fetch the public TCV L-H transition dataset and inventory its HDF5/notebook structure."""
from __future__ import annotations
import hashlib
import json
import urllib.request
from pathlib import Path

import h5py

RECORD = "https://zenodo.org/records/14996664"
FILES = ("LH_DATA.h5", "lhdatabase.h5", "LH_analysis.ipynb", "README.md", "requirements.txt")
OUT = Path("artifacts/tcv-lh-transition-source")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Tokamak-Insight-Suite-validation/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} for {url}")
        return response.read()


def hdf5_inventory(path: Path) -> dict:
    inventory = {"file": path.name, "datasets": [], "groups": []}
    with h5py.File(path, "r") as h5:
        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                inventory["groups"].append(name)
            elif isinstance(obj, h5py.Dataset):
                inventory["datasets"].append({
                    "path": name, "shape": list(obj.shape), "dtype": str(obj.dtype),
                    "attrs": {str(k): str(v)[:300] for k, v in obj.attrs.items()},
                })
        h5.visititems(visit)
        inventory["root_attrs"] = {str(k): str(v)[:300] for k, v in h5.attrs.items()}
    return inventory


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"record": RECORD, "reference_doi": "10.1088/1361-6587/adc8ce", "files": []}
    for name in FILES:
        url = f"{RECORD}/files/{name}?download=1"
        raw = fetch(url)
        path = OUT / name
        path.write_bytes(raw)
        item = {"name": name, "url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if name.endswith(".h5"):
            item["hdf5_inventory"] = hdf5_inventory(path)
        elif name.endswith(".ipynb"):
            notebook = json.loads(raw.decode("utf-8"))
            cells = []
            for i, cell in enumerate(notebook.get("cells", [])):
                source = "".join(cell.get("source", []))
                cells.append({"index": i, "cell_type": cell.get("cell_type"),
                              "source_chars": len(source), "source": source[:5000]})
            item["notebook_metadata"] = notebook.get("metadata", {})
            item["cells"] = cells
        manifest["files"].append(item)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "record": RECORD,
        "file_count": len(manifest["files"]),
        "hdf5_files": [x["name"] for x in manifest["files"] if "hdf5_inventory" in x],
        "manifest": str(OUT / "manifest.json"),
        "artifact_dir": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
