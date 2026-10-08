#!/usr/bin/env python3
"""Download pinned MouseGPT examples from HF; no model inference is performed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets" / "manifest.json"
RECEIPT = ".mousegpt-asset.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    part = PurePosixPath(relative)
    if not relative or part.is_absolute() or ".." in part.parts or "\\" in relative:
        raise ValueError(f"Unsafe relative path: {relative!r}")
    result = root.joinpath(*part.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes destination: {relative!r}")
    return result


def verify_install(target: Path, asset: dict) -> int:
    receipt_path = target / RECEIPT
    if not receipt_path.is_file():
        raise ValueError(f"No managed asset receipt at {target}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("archive_sha256") != asset["sha256"]:
        raise ValueError(f"Installed asset version differs at {target}")
    for entry in receipt["files"]:
        path = checked_path(target, entry["path"])
        if not path.is_file() or path.is_symlink() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Missing or modified sample: {path}")
    if len(receipt["files"]) != asset["file_count"]:
        raise ValueError("Installed file count differs from asset manifest")
    return len(receipt["files"])


def install_archive(archive: Path, target: Path, asset: dict) -> int:
    if sha256(archive) != asset["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive.name}")
    if target.exists():
        return verify_install(target, asset)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mousegpt-extract-", dir=target.parent))
    entries = []
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            files = [entry for entry in members if not entry.is_dir()]
            if len(files) != asset["file_count"]:
                raise ValueError("Archive file count differs from asset manifest")
            if sum(entry.file_size for entry in files) != asset["uncompressed_bytes"]:
                raise ValueError("Archive expanded size differs from asset manifest")
            seen = set()
            for entry in members:
                destination = checked_path(stage, entry.filename)
                if entry.filename in seen or entry.filename == RECEIPT:
                    raise ValueError("Duplicate or reserved archive path")
                seen.add(entry.filename)
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Symlinks are not allowed in example bundles")
                if entry.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with bundle.open(entry) as source, destination.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                entries.append({"path": entry.filename, "sha256": digest.hexdigest()})
        receipt = {"asset": asset["id"], "archive_sha256": asset["sha256"],
                   "repo_id": asset["repo_id"], "revision": asset.get("revision"),
                   "files": entries}
        (stage / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
        # Never overwrite an existing user dataset.
        if target.exists():
            raise ValueError(f"Destination appeared during extraction: {target}")
        stage.rename(target)
        return len(entries)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="*", help="Asset IDs, or all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Verify installed sample bytes offline")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository layout root")
    parser.add_argument("--local-bundle-dir", type=Path, help="Use already downloaded archives")
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text())
    by_id = {asset["id"]: asset for asset in manifest["assets"]}
    if args.list or not args.assets:
        for asset in by_id.values():
            print(f'{asset["id"]}: {asset["description"]} ({asset["bytes"] / 1e6:.1f} MB download)')
        return 0
    names = list(by_id) if args.assets == ["all"] else args.assets
    unknown = set(names) - set(by_id)
    if unknown:
        parser.error("Unknown asset IDs: " + ", ".join(sorted(unknown)))
    try:
        for name in names:
            asset = by_id[name]
            target = checked_path(args.root.resolve(), asset["destination"])
            if args.verify:
                count = verify_install(target, asset)
            else:
                if args.local_bundle_dir:
                    archive = args.local_bundle_dir / asset["filename"]
                else:
                    if not asset.get("revision"):
                        raise ValueError(f"{name}: HF upload is not pinned yet; use --local-bundle-dir")
                    try:
                        from huggingface_hub import hf_hub_download
                    except ImportError:
                        raise ValueError("Install assets dependencies: pip install -r requirements-assets.txt") from None
                    archive = Path(hf_hub_download(repo_id=asset["repo_id"],
                        repo_type="dataset", filename=asset["filename"], revision=asset["revision"],
                        token=os.environ.get("HF_TOKEN") or None))
                count = install_archive(archive, target, asset)
            print(f"{name}: verified {count} files at {target}")
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Asset error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
