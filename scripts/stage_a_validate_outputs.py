"""Verify actual saved arrays and hash them for subsequent read-only reuse."""
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.stage_a_common import config, participants, provenance, require_compute, save_json, scratch_root, sha256

require_compute()
root = scratch_root()
start = time.monotonic()
record = {"provenance": provenance(), "subjects": [], "files": [], "excluded_from_array_verification": []}
for dataset, parameters in config()["datasets"].items():
    receipt = json.loads((root / "receipts/preprocess" / f"{dataset}.json").read_text())
    assert receipt["status"] == "complete"
    assert {r["subject"] for r in receipt["subjects"]} == set(participants(dataset))
    for row in receipt["subjects"]:
        subject = row["subject"]
        if "processed_signal_bytes" not in row:
            record["excluded_from_array_verification"].append({"dataset": dataset, "subject": subject, "reason": row["failures"]})
            continue
        directory = root / "processed" / dataset / f"sub-{subject:03d}"
        metadata = json.loads((directory / "metadata.json").read_text())
        assert metadata["preprocessing_config_sha256"] == sha256(Path(__file__).resolve().parents[1] / "configs/stage_a.yaml")
        tasks = np.load(directory / "tasks.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(directory / "labels.npy", allow_pickle=False)
        mask = np.load(directory / "task_valid.npy", allow_pickle=False)
        assert tasks.dtype == np.float16 and labels.dtype == np.int64 and mask.dtype == bool
        assert tasks.shape == (len(labels), parameters["channels"], 200 * parameters["task_seconds"])
        assert mask.shape == labels.shape
        assert np.bincount(labels, minlength=2).tolist() == [row["left_trials"], row["right_trials"]]
        assert np.bincount(labels[mask], minlength=2).tolist() == [row["usable_left"], row["usable_right"]]
        for eye in metadata["rest"]:
            rest = np.load(directory / f"rest_{eye}.npy", mmap_mode="r", allow_pickle=False)
            valid = np.load(directory / f"rest_{eye}_valid.npy", allow_pickle=False)
            assert rest.dtype == np.float16 and valid.dtype == bool
            assert rest.shape == (len(valid), parameters["channels"], 800)
            assert int(valid.sum()) * 4 == row["rest"][eye]["usable_seconds"]
        for path in sorted(directory.glob("*.npy")):
            record["files"].append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)})
        record["subjects"].append({"dataset": dataset, "subject": subject, "status": "pass"})
record.update(status="pass", elapsed_seconds=time.monotonic()-start,
              total_bytes=sum(f["bytes"] for f in record["files"]))
save_json(root / "receipts/processed_arrays_manifest.json", record)
print(json.dumps({"status": "pass", "verified_subjects": len(record["subjects"]), "unprocessed_subjects": len(record["excluded_from_array_verification"]), "files": len(record["files"]), "bytes": record["total_bytes"], "seconds": record["elapsed_seconds"]}))
