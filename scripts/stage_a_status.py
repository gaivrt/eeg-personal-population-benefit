"""Read compact metadata only. Safe on the login node; no EEG loading."""
import json
import os
from pathlib import Path

root = Path(os.environ["STAGE_A_ROOT"])
summary = {}
for dataset in ["PhysionetMI", "Dreyer2023", "Cho2017"]:
    entry = {}
    directory = root / "receipts/download" / dataset
    files = sorted(directory.glob("subject-*.json"))
    entry["downloaded_subjects"] = len(files)
    if (directory / "run.json").exists():
        run = json.loads((directory / "run.json").read_text())
        entry["download_status"] = run["status"]
        entry["tls_verify"] = run.get("tls_verify", False)
        entry["download_job"] = run["slurm_job_id"]
        if "error" in run:
            entry["error"] = run["error"][-1200:]
    for kind in [dataset, dataset + ".smoke"]:
        path = root / "receipts/preprocess" / f"{kind}.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            entry[kind] = {"status": receipt["status"], "subjects": len(receipt["subjects"]),
                "failed": [{"subject": r["subject"], "failures": r["failures"],
                            "error": r.get("error", "")[-1000:]}
                           for r in receipt["subjects"] if r["status"] == "fail"]}
    summary[dataset] = entry
print(json.dumps(summary, indent=2))
