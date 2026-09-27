"""Run every included subject in a Slurm CPU job; retain all audit failures."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.stage_a_common import config, failure_stop, participants, provenance, require_compute, save_json, scratch_root
from subject_context.stage_a_preprocess import preprocess_subject


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["PhysionetMI", "Dreyer2023", "Cho2017"])
    parser.add_argument("--subjects", type=int, nargs="+", help="Explicit implementation smoke check only")
    parser.add_argument("--complete-audit", action="store_true", help="User-authorized completion despite a recorded quality stop; never authorizes training")
    parser.add_argument("--resume", action="store_true", help="Reuse completed subject audits with exactly the same preprocessing configuration")
    args = parser.parse_args()
    require_compute()
    root = scratch_root()
    receipt = provenance()
    begin = time.monotonic()
    receipt.update(dataset=args.dataset, status="running", subjects=[])
    receipt["complete_audit_after_threshold_authorized"] = args.complete_audit
    subjects = args.subjects or participants(args.dataset)
    if not set(subjects).issubset(participants(args.dataset)):
        raise ValueError("Subject outside the frozen cohort")
    receipt_name = args.dataset + (".smoke" if args.subjects else "") + ".json"
    receipt_path = root / "receipts/preprocess" / receipt_name
    if args.resume and receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous["config_sha256"] != receipt["config_sha256"]:
            raise ValueError("Cannot reuse preprocessing from a different configuration")
        save_json(receipt_path.parent / f'{args.dataset}.attempt-{previous["slurm_job_id"]}.json', previous)
        receipt["subjects"] = previous["subjects"]
        receipt["resumed_from"] = {k: previous.get(k) for k in ["git_commit", "slurm_job_id", "utc", "status", "elapsed_seconds"]}
        receipt["reused_subjects"] = len(previous["subjects"])
        completed = {r["subject"] for r in previous["subjects"]}
        subjects = [s for s in subjects if s not in completed]
    save_json(receipt_path, receipt)
    for subject in subjects:
        started = time.monotonic()
        try:
            row = preprocess_subject(root, args.dataset, subject)
        except Exception:
            row = {"dataset": args.dataset, "subject": subject, "status": "fail",
                   "failures": ["reader_or_preprocessing_error"], "error": traceback.format_exc()}
            save_json(root / "processed" / args.dataset / f"sub-{subject:03d}" / "audit.json", row)
        row["seconds"] = time.monotonic() - started
        receipt["subjects"].append(row)
        print(f'{args.dataset} {subject:03d}: {row["status"]} {row["failures"]} ({row["seconds"]:.1f}s)', flush=True)
        save_json(root / "receipts/preprocess" / receipt_name, receipt)
        failed = sum(r["status"] == "fail" for r in receipt["subjects"])
        # The denominator is the entire predefined cohort, including not-yet-read
        # subjects. Once this bound exceeds 10%, later passes cannot reverse it.
        if not args.subjects and not args.complete_audit and failure_stop(failed, len(participants(args.dataset)), config()["audit"]["max_failed_subject_fraction"]):
            receipt["status"] = "stopped_failure_threshold"
            receipt["failed_subjects"] = failed
            receipt["cohort_denominator"] = len(participants(args.dataset))
            break
    receipt["elapsed_seconds"] = time.monotonic() - begin
    if receipt["status"] == "running":
        receipt["status"] = "complete"
    save_json(root / "receipts/preprocess" / receipt_name, receipt)


if __name__ == "__main__":
    main()
