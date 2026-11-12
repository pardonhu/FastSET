#!/usr/bin/env python3
"""Download a versioned artifact, verify SHA256, and extract regular files only."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", choices=["weights", "a2b-data"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archive", type=Path, help="Use an already downloaded archive")
    a = p.parse_args()
    filename = "model_bundle.json" if a.asset == "weights" else "a2b_bundle.json"
    meta = json.loads((ROOT / "artifacts" / filename).read_text())
    out = a.output.resolve()
    if out.exists():
        raise ValueError("Choose a new output directory")
    archive = a.archive or out.parent / meta["asset"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        url = f'https://github.com/pardonhu/FastSET/releases/download/{meta["tag"]}/{meta["asset"]}'
        partial = archive.with_suffix(archive.suffix + ".partial")
        print("DOWNLOAD", url, flush=True)
        with urllib.request.urlopen(url) as response, partial.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        partial.rename(archive)
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    if archive.stat().st_size != meta["bytes"] or digest.hexdigest() != meta["sha256"]:
        raise ValueError("Archive hash/size mismatch")
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            path = out / member.name
            if not member.isfile() or not path.resolve().is_relative_to(out):
                raise ValueError("Unsafe archive member")
        out.mkdir(parents=True)
        for member in members:
            path = out / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, path.open("xb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    print("VERIFIED_AND_EXTRACTED", out)


if __name__ == "__main__":
    main()
