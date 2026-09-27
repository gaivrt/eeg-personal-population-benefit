"""Explicitly download pinned upstream source files; no EEG, weights or execution."""
from pathlib import Path
import hashlib, json, urllib.request
root = Path(__file__).resolve().parent
base = root / "docs/cross_model_audit_2026-09-27"
for row in json.loads((base / "source_manifest.json").read_text())["files"]:
    target = base / row["file"]
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]:
        continue
    content = urllib.request.urlopen(row["url"], timeout=60).read()
    assert hashlib.sha256(content).hexdigest() == row["sha256"], row["file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    print("Verified", row["file"])
