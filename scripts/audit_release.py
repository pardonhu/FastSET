"""CPU-only release hygiene and included-artifact integrity check."""

import ast
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
excluded = {".git", ".venv", "results", "__pycache__", ".pytest_cache", "build", "dist"}
failures = []
files = []
patterns = [
    re.compile(r"(?:/" + "home/|/" + "mnt/)[A-Za-z0-9_]"),
    re.compile(r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?:password|passwd|token)\s*=\s*['\"][^'\"]+['\"]", re.I),
]
for path in root.rglob("*"):
    relative = path.relative_to(root)
    if any(p in excluded or p.endswith(".egg-info") for p in relative.parts):
        continue
    if path.is_symlink():
        failures.append(f"symlink: {relative}")
    if not path.is_file():
        continue
    files.append(path)
    if path.suffix in {".sock", ".pem", ".key", ".log", ".pkl"}:
        failures.append(f"unexpected private/large-data file: {relative}")
    if path.suffix in {".py", ".md", ".json", ".csv", ".yml", ".toml", ".txt", ".cff"}:
        content = path.read_text()
        if any(p.search(content) for p in patterns):
            failures.append(f"private path/address or credential assignment: {relative}")
    if path.suffix == ".py":
        ast.parse(path.read_text(), filename=str(relative))
for item in json.loads((root / "artifacts/INDEX.json").read_text()):
    path = root / item["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        failures.append("artifact hash mismatch: " + item["path"])
print(
    json.dumps(
        {
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "hygiene_and_integrity_pass": not failures,
            "failures": failures,
            "scope": "not a license approval or GPU numerical acceptance",
        },
        indent=2,
    )
)
raise SystemExit(bool(failures))
