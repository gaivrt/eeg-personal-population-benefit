"""Bounded stage-0a downloads; only subject 1, at most two sessions per dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/data_access_audit_2026-09-25/sources"
SESSION = requests.Session()
SESSION.trust_env = False


def download(item):
    target = ROOT / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    expected = item.get("bytes")
    if not expected and not target.exists():
        response = SESSION.head(item["url"], timeout=(20, 30), allow_redirects=True)
        response.raise_for_status()
        expected = int(response.headers.get("Content-Length", 0)) or None
    if not target.exists() or (expected and target.stat().st_size != expected):
        part = target.with_suffix(target.suffix + ".part")
        if expected and expected > 10 * 1024 * 1024:
            chunk_dir = target.parent / (target.name + ".chunks")
            chunk_dir.mkdir(exist_ok=True)
            size = 4 * 1024 * 1024
            ranges = [(start, min(start + size, expected) - 1) for start in range(0, expected, size)]

            def fetch_range(bounds):
                start, end = bounds
                file = chunk_dir / str(start)
                if file.exists() and file.stat().st_size == end - start + 1:
                    return file
                for attempt in range(3):
                    try:
                        with SESSION.get(item["url"], headers={"Range": f"bytes={start}-{end}"},
                                         timeout=(20, 30)) as r:
                            r.raise_for_status()
                            assert r.status_code == 206
                            assert r.headers["Content-Range"] == f"bytes {start}-{end}/{expected}"
                            assert len(r.content) == end - start + 1
                            file.write_bytes(r.content)
                        return file
                    except Exception:
                        if attempt == 2:
                            raise
                raise RuntimeError("unreachable")

            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = [pool.submit(fetch_range, bounds) for bounds in ranges]
                for completed, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if completed % 16 == 0:
                        print(f"  {target.name}: {completed}/{len(ranges)} chunks", flush=True)
            with part.open("wb") as stream:
                for start, end in ranges:
                    stream.write((chunk_dir / str(start)).read_bytes())
            assert part.stat().st_size == expected
            part.replace(target)
            for start, _ in ranges:
                chunk = (chunk_dir / str(start)).resolve()
                assert chunk.is_relative_to(ROOT)
                chunk.unlink()
            chunk_dir.rmdir()
        else:
          with SESSION.get(item["url"], timeout=(20, 30), stream=True) as response:
            response.raise_for_status()
            declared = int(response.headers.get("Content-Length", 0))
            if expected and declared and declared != expected:
                raise ValueError(f"Unexpected byte count: {declared} vs {expected}")
            count = 0
            with part.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
                    count += len(chunk)
            if (expected or declared) and count != (expected or declared):
                raise ValueError(f"Truncated download: {count}")
          part.replace(target)
    digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
    if item.get("sha256") and digest != item["sha256"]:
        raise ValueError("SHA256 mismatch")
    return {**item, "status": "ok", "bytes": target.stat().st_size,
            "sha256": digest, "seconds": round(time.perf_counter() - started, 3)}


def make_manifest():
    assets = []
    hf = SESSION.get("https://huggingface.co/api/models/weighting666/CBraMod", timeout=30).json()
    assets.append({"dataset": "CBraMod", "revision": hf["sha"],
                   "url": f"https://huggingface.co/weighting666/CBraMod/resolve/{hf['sha']}/pretrained_weights.pth",
                   "path": "weights/cbramod.pth"})
    for run in [1, 2, 4, 8, 12]:
        name = f"S001R{run:02d}.edf"
        assets.append({"dataset": "PhysionetMI", "subject": 1, "session": 1,
                       "url": f"https://physionet.org/files/eegmmidb/1.0.0/S001/{name}",
                       "path": f"data/samples/PhysionetMI/{name}"})
    fs = json.loads((AUDIT / "stieger_figshare.json").read_text(encoding="utf-8"))
    for item in fs["files"]:
        if item["name"] in ["S1_Session_1.mat", "S1_Session_2.mat"]:
            assets.append({"dataset": "Stieger2021", "subject": 1,
                           "session": int(item["name"].split("_")[-1].split(".")[0]),
                           "url": item["download_url"], "bytes": item["size"],
                           "path": "data/samples/Stieger2021/" + item["name"]})
    base = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/"
    for dataset, xmlname, suffixes in [
        ("Lee2019_MI", "lee_object_list.xml", ["session1/s1/sess01_subj01_EEG_MI.mat", "session2/s1/sess02_subj01_EEG_MI.mat"]),
        ("Cho2017", "cho_object_list.xml", ["s01.mat"]),
    ]:
        tree = ET.fromstring((AUDIT / xmlname).read_text(encoding="utf-8"))
        for item in tree.findall("{*}Contents"):
            key = item.find("{*}Key").text
            if any(key.endswith(suffix) for suffix in suffixes):
                assets.append({"dataset": dataset, "subject": 1,
                               "url": base + key, "bytes": int(item.find("{*}Size").text),
                               "path": f"data/samples/{dataset}/{key.rsplit('/', 1)[-1]}"})
    rows = list(csv.DictReader((AUDIT / "dreyer_manifest.tsv").open(encoding="utf-8"), delimiter="\t"))
    for row in rows:
        if row["filename"] == "sub-01.zip":
            assets.append({"dataset": "Dreyer2023", "subject": 1,
                           "url": row["url"], "path": "data/samples/Dreyer2023/sub-01.zip"})
    expected_counts = {"CBraMod": 1, "PhysionetMI": 5, "Stieger2021": 2,
                       "Lee2019_MI": 2, "Cho2017": 1, "Dreyer2023": 1}
    for dataset, n in expected_counts.items():
        assert sum(a["dataset"] == dataset for a in assets) == n, dataset
    return assets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all")
    args = parser.parse_args()
    manifest = ROOT / "configs/assets.json"
    assets = json.loads(manifest.read_text()) if manifest.exists() else make_manifest()
    manifest.write_text(json.dumps(assets, indent=2), encoding="utf-8")
    receipt_path = ROOT / "reports/stage0a/download_receipts.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else []
    receipts = {r["path"]: r for r in previous}
    for item in assets:
        if args.only != "all" and item["dataset"] != args.only:
            continue
        print("Downloading", item["path"], flush=True)
        try:
            result = download(item)
        except Exception as error:
            result = {**item, "status": "error", "error": repr(error)}
        receipts[item["path"]] = result
        receipt_path.write_text(json.dumps(list(receipts.values()), indent=2), encoding="utf-8")
        print(result, flush=True)
    if any(r["status"] != "ok" for r in receipts.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
