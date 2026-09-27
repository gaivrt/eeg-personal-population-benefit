"""Read-only release integrity check. No training, inference or statistical testing."""
from pathlib import Path
import csv, hashlib, json, math
root=Path(__file__).resolve().parent
release=json.loads((root/"RELEASE_MANIFEST.json").read_text())
for name, expected in release["files"].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected, name
p=root/"paper/nature_portfolio"
m=json.loads((p/"build_manifest.json").read_text())
for name, expected in m["sources"].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected, name
rows=list(csv.DictReader((p/"numbers.csv").open(encoding="utf-8-sig")))
assert len(rows)==37265
assert all(math.isfinite(float(r["value"])) for r in rows)
assert all((root/name).is_file() for r in rows for name in r["source_file"].split(";"))
tests=list(csv.DictReader((p/"supplementary_data/Supplementary_Data_1.csv").open(encoding="utf-8-sig")))
assert [r["print_id"] for r in tests]==[f"T{i:04d}" for i in range(1,905)]
assert len(tests)==904
print(f"Verified {len(release['files'])} files, {len(rows)} numeric records, and {len(tests)} stable comparison IDs.")
