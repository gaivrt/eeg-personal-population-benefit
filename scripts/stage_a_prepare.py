"""Use all finite stored trials; recover four readable Dreyer records."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.stage_a_common import config, participants, provenance, require_compute, save_json, scratch_root
from subject_context.stage_a_preprocess import preprocess_subject, alpha_comparison

require_compute()
root = scratch_root()
record = {"provenance": provenance(), "policy": "all_finite_no_amplitude_or_balance_exclusion", "subjects": []}
start = time.monotonic()
for dataset in config()["datasets"]:
    for subject in participants(dataset):
        out = root / "processed" / dataset / f"sub-{subject:03d}"
        if not (out / "tasks.npy").exists():
            preprocess_subject(root, dataset, subject)
        meta = json.loads((out / "metadata.json").read_text())
        tasks = np.load(out / "tasks.npy", mmap_mode="r")
        labels = np.load(out / "labels.npy")
        valid = np.isfinite(tasks).all(axis=(1, 2))
        assert valid.all() and len(labels) == len(tasks)
        np.save(out / "task_valid.npy", valid)
        trials = pd.read_csv(out / "trials.csv")
        trials["quality_valid"] = valid
        trials.to_csv(out / "trials.csv", index=False)
        rest, masks = {}, {}
        for eye in meta["rest"]:
            rest[eye] = np.load(out / f"rest_{eye}.npy", mmap_mode="r")
            masks[eye] = np.isfinite(rest[eye]).all(axis=(1, 2))
            assert masks[eye].all() and len(rest[eye])
            np.save(out / f"rest_{eye}_valid.npy", masks[eye])
            meta["rest"][eye]["usable_seconds"] = len(rest[eye]) * 4
            meta["rest"][eye]["rejected_segment_seconds"] = 0
        meta["active_trial_policy"] = record["policy"]
        save_json(out / "metadata.json", meta)
        alpha = alpha_comparison(rest["open"], rest.get("closed"), meta["channels"], masks["open"], masks.get("closed"))
        counts = np.bincount(labels, minlength=2)
        row = {"dataset": dataset, "subject": subject, "left_trials": int(counts[0]),
               "right_trials": int(counts[1]), "rest": meta["rest"], "alpha": alpha,
               "channels": meta["channels"], "checks": meta["checks"],
               "processed_bytes": sum(p.stat().st_size for p in out.glob("*.npy"))}
        save_json(out / "audit_v2.json", row)
        record["subjects"].append(row)
record.update(status="complete", elapsed_seconds=time.monotonic()-start)
save_json(root / "receipts/prepared_v2.json", record)
print(json.dumps({"subjects":len(record["subjects"]),"trials":sum(r["left_trials"]+r["right_trials"] for r in record["subjects"]),"seconds":record["elapsed_seconds"]}))
