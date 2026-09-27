"""Storage-checked Lee session-1 MI download and unchanged Stage A preprocessing.

Runs on a CPU allocation. Only the pre-train rest is used; FCz is not fabricated.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.data import make_splits
from subject_context.stage_a_common import ROOT, require_compute, save_json, scratch_root, sha256
from subject_context.stage_a_preprocess import canonical_channel, filtered_array, save_signal, segment_rest
from subject_context.stage_b import temporal_halves
from subject_context.stage_w import config_w

BASE = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/"
PREFIX = "live/pub/10.5524/100001_101000/100542/"
DATASET = "Lee2019_MI"


def manifest(xml):
    doc = ET.fromstring(xml)
    assert doc.findtext("{*}IsTruncated") == "false", "Incomplete object listing"
    rows = []
    pattern = re.compile(re.escape(PREFIX) + r"session1/s(\d+)/sess01_subj(\d+)_EEG_MI\.mat$")
    for item in doc.findall("{*}Contents"):
        key = item.findtext("{*}Key")
        match = pattern.fullmatch(key)
        if match:
            number, repeated = map(int, match.groups())
            assert number == repeated
            rows.append({"subject": number, "key": key, "url": BASE+key,
                         "bytes": int(item.findtext("{*}Size"))})
    rows.sort(key=lambda r: r["subject"])
    assert [r["subject"] for r in rows] == list(range(1, 55))
    return rows


def get_manifest(root):
    response = requests.get(BASE, params={"list-type": "2", "prefix": PREFIX}, timeout=(30, 120))
    response.raise_for_status()
    rows = manifest(response.content)
    available = shutil.disk_usage(root).free
    required = config_w()["external"]["minimum_free_bytes"]
    if available < required:
        raise RuntimeError(f"Need {required} free bytes before download; have {available}")
    audit = root / "stage_w/lee"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "object_list.xml").write_bytes(response.content)
    record = {"utc": datetime.now(timezone.utc).isoformat(), "available_bytes_before": available,
              "minimum_free_bytes": required, "download_bytes": sum(r["bytes"] for r in rows), "files": rows}
    save_json(audit / "download_manifest.json", record)
    print(json.dumps({k:v for k,v in record.items() if k != "files"}), flush=True)
    return rows


def download(root, row):
    folder = root / "raw/Lee2019_MI/session1"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / Path(row["key"]).name
    receipt = root / "stage_w/lee/download" / f"sub-{row['subject']:03d}.json"
    if dest.exists():
        assert dest.stat().st_size == row["bytes"]
        digest = sha256(dest)
        if receipt.exists():
            assert json.loads(receipt.read_text())["sha256"] == digest
        else:
            save_json(receipt, {**row, "sha256": digest, "status": "complete", "existing_file": True})
        return dest
    part = dest.with_suffix(".mat.part")
    offset = part.stat().st_size if part.exists() else 0
    assert offset <= row["bytes"]
    if offset < row["bytes"]:
        if shutil.disk_usage(root).free < row["bytes"] - offset + 20_000_000_000:
            raise RuntimeError("Storage reserve would be exhausted")
        with requests.get(row["url"], headers={"Range": f"bytes={offset}-"} if offset else {},
                          stream=True, timeout=(30, 120)) as r:
            r.raise_for_status()
            if offset:
                assert r.status_code == 206 and r.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
            with part.open("ab" if offset else "wb") as f:
                for chunk in r.iter_content(4*1024*1024):
                    f.write(chunk)
    assert part.stat().st_size == row["bytes"]
    digest = sha256(part)
    part.rename(dest)
    save_json(receipt, {**row, "sha256": digest, "status": "complete", "resumed_bytes": offset})
    print(f"Downloaded Lee session1 subject {row['subject']:02d}", flush=True)
    return dest


def preprocess(root, subject, path):
    cfg = config_w()["external"]
    out = root / "processed" / DATASET / f"sub-{subject:03d}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "audit_w.json").exists():
        previous = json.loads((out / "audit_w.json").read_text())
        assert previous["raw_sha256"] == sha256(path)
        assert previous["w_config_sha256"] == sha256(ROOT / "configs/stage_w.yaml")
        return previous
    data = loadmat(path, simplify_cells=True)
    train, test = data["EEG_MI_train"], data["EEG_MI_test"]
    channels = [canonical_channel(str(c)) for c in train["chan"]]
    assert len(channels) == cfg["channels"] and len(set(channels)) == cfg["channels"]
    common = json.loads((ROOT / "configs/stage_a_channels.json").read_text())["readout_common"]
    assert set(common) - set(channels) == {"FCz"}
    tasks, rows = [], []
    for order, (run, d) in enumerate([("train", train), ("test", test)], 1):
        assert [canonical_channel(str(c)) for c in d["chan"]] == channels
        assert int(d["fs"]) == cfg["sfreq"]
        mapping = {int(number): str(label).lower() for number, label in d["class"]}
        assert mapping == {1: "right", 2: "left"}
        labels, times = np.asarray(d["y_dec"], dtype=int), np.asarray(d["t"], dtype=int)
        assert len(labels) == len(times) == 100 and np.all(np.diff(times) > 0)
        assert np.bincount(labels, minlength=3).tolist() == [0, 50, 50]
        # Official MOABB reader uses the published event indices directly.
        signal = filtered_array(np.asarray(d["x"], dtype=float).T * 1e-6, d["fs"], channels, DATASET, cfg)
        for index, (sample, label) in enumerate(zip(times, labels)):
            onset = float(sample) / cfg["sfreq"]
            start = round(onset * 200)
            assert 0 <= start and start + 800 <= signal.shape[1]
            tasks.append(signal[:, start:start+800])
            rows.append({"trial_id": f"{DATASET}:{subject:03d}:{len(rows):04d}", "label": int(label == 1),
                         "source_file": str(path), "source_field": f"EEG_MI_{run}.x", "run": order,
                         "source_event_index": index, "source_event_sample": int(sample),
                         "start_seconds": onset, "stop_seconds": onset+4, "protocol_order": order,
                         "quality_valid": True})
        del signal
    tasks = np.stack(tasks)
    assert np.isfinite(tasks).all()
    save_signal(out / "tasks.npy", tasks)
    table = pd.DataFrame(rows)
    table.to_csv(out / "trials.csv", index=False)
    np.save(out / "labels.npy", table.label.to_numpy(dtype=np.int64))
    np.save(out / "task_valid.npy", np.ones(len(table), dtype=bool))
    fit_rows, query, counts = temporal_halves(table)
    assert fit_rows.tolist() == list(range(100)) and query.tolist() == list(range(100, 200))
    rest_meta, rest_error = {}, None
    try:
        rest = np.asarray(train["pre_rest"], dtype=float)
        assert rest.shape == (60000, 62) and np.isfinite(rest).all()
        x = filtered_array(rest.T * 1e-6, 1000, channels, DATASET, cfg)
        windows = segment_rest(x)
        save_signal(out / "rest_open.npy", windows)
        np.save(out / "rest_open_valid.npy", np.ones(len(windows), dtype=bool))
        rest_meta["open"] = {"source_file": str(path), "source_field": "EEG_MI_train.pre_rest", "eye_state": "open",
                              "duration_seconds": 60, "usable_seconds": len(windows)*4, "protocol_order": 0}
    except (KeyError, AssertionError, ValueError) as error:
        rest_error = f"{type(error).__name__}: {error}"
    result = {"dataset": DATASET, "subject": subject, "session": 1, "status": "complete", "counts": counts,
              "raw_sha256": sha256(path), "w_config_sha256": sha256(ROOT / "configs/stage_w.yaml"),
              "trial_table_sha256": sha256(out / "trials.csv"), "fit_indices": fit_rows.tolist(),
              "query_indices": query.tolist(), "rest_error": rest_error, "rest_available": bool(rest_meta),
              "readout_channels": [c for c in common if c != "FCz"], "missing_FCz_user_approved": True}
    save_json(out / "metadata.json", {"dataset": DATASET, "subject": subject, "session": 1,
        "channels": channels, "sfreq": 200, "units": "uV/100", "dtype": "float16", "rest": rest_meta,
        "source_files": [{"path": str(path), "sha256": result["raw_sha256"]}], "checks": result})
    save_json(out / "audit_w.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    require_compute()
    root = scratch_root()
    rows = get_manifest(root)
    if args.manifest_only:
        return
    # Two download streams, no GPU allocation, resumable .part files.
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda row: download(root, row), rows))
    audits, errors = [], []
    for row, path in zip(rows, paths):
        try:
            audits.append(preprocess(root, row["subject"], path))
            print(f"Preprocessed Lee subject {row['subject']:02d}", flush=True)
        except Exception:
            errors.append({"subject": row["subject"], "error": traceback.format_exc()})
        save_json(root / "stage_w/lee/preprocessing.json", {"subjects": audits, "errors": errors})
    assert not errors and len(audits) == 54, "Preserve errors and resolve before training; do not silently exclude subjects"
    splits = make_splits([(DATASET, s) for s in range(1, 55)], 5, .10, 20260925)
    save_json(root / "stage_w/lee/splits.json", {"folds": [dict(zip(["train", "validation", "test"], s)) for s in splits]})
    save_json(root / "stage_w/lee/halves.json", {"subjects": audits})
    save_json(root / "stage_w/lee/prepared.json", {"status": "complete", "subjects": 54,
        "downloaded_bytes": sum(p.stat().st_size for p in paths), "available_bytes_after": shutil.disk_usage(root).free,
        "rest_available_subjects": sum(r["rest_available"] for r in audits),
        "w_config_sha256": sha256(ROOT / "configs/stage_w.yaml")})


if __name__ == "__main__":
    main()
