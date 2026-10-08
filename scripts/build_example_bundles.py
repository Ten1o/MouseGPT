#!/usr/bin/env python3
"""Maintainer utility: package the MouseGPT example data into reproducible ZIP bundles."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def add_file(bundle, name, path=None, content=None):
    info = zipfile.ZipInfo(name, date_time=(2024, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    if content is not None:
        bundle.writestr(info, content, compresslevel=1)
    else:
        with path.open("rb") as source, bundle.open(info, "w", force_zip64=True) as dest:
            shutil.copyfileobj(source, dest, 1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    source = args.source
    vlm = source / "2_VLM/prompt_test/data"
    rows = [json.loads(line) for line in (vlm / "test_data.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 10:
        raise ValueError(f"Expected 10 VLM records; found {len(rows)}")
    images = set()
    for row in rows:
        normalized = []
        for path in row["images"]:
            name = Path(path).name
            if not (vlm / "imgs" / name).is_file():
                raise FileNotFoundError(name)
            images.add(name)
            normalized.append("imgs/" + name)
        row["images"] = normalized
    if len(images) != 10:
        raise ValueError(f"Expected 10 distinct VLM images; found {len(images)}")
    nlui = source / "4_NLUI/data"
    hdf5 = nlui / "240419_C11M4.h5"
    preview_dir = nlui / "240419_C11M4_imgs"
    previews = sorted(preview_dir.glob("*.jpg"))
    if not hdf5.is_file() or hdf5.stat().st_size == 0:
        raise ValueError("Missing or empty HDF5 session")
    if not preview_dir.is_dir() or len(previews) != 6000:
        raise ValueError(f"Expected 6000 NLUI previews; found {len(previews)}")
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output / "vlm-demo.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        add_file(z, "test_data.jsonl", content=("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode())
        for name in sorted(images):
            add_file(z, "imgs/" + name, path=vlm / "imgs" / name)
    selected = [hdf5, *previews]
    with zipfile.ZipFile(args.output / "nlui-demo.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in selected:
            add_file(z, path.relative_to(nlui).as_posix(), path=path)
    assets = []
    for name, destination, description in [
        ("vlm-demo", "2_VLM/prompt_test/data", "10 image/prompt examples for VLM inference"),
        ("nlui-demo", "4_NLUI/data", "Prepared 240419_C11M4 session and 6000 preview frames")]:
        path = args.output / (name + ".zip")
        with zipfile.ZipFile(path) as z:
            files = [entry for entry in z.infolist() if not entry.is_dir()]
            size = sum(entry.file_size for entry in files)
        assets.append({"id": name, "repo_id": "Ten1o/MouseGPT-examples", "repo_type": "dataset",
                       "revision": None, "filename": path.name, "destination": destination,
                       "description": description, "sha256": digest(path), "bytes": path.stat().st_size,
                       "file_count": len(files), "uncompressed_bytes": size})
    manifest = {"schema_version": 1, "assets": assets}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
