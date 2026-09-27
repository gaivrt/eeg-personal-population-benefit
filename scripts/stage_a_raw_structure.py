"""Audit Dreyer source boundaries/counts even when strict preprocessing rejects rest."""
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.edf_header import parse_edf_header
from subject_context.real_data import dreyer_rest_interval
from subject_context.stage_a_common import participants, provenance, require_compute, save_json, scratch_root, sha256

require_compute()
root = scratch_root()
record = {"provenance": provenance(), "subjects": []}
for subject in participants("Dreyer2023"):
    folder = root / "raw/MNE-dreyer2023-data" / f"sub-{subject:02d}/eeg"
    row = {"subject": subject, "rest": {}, "runs": [], "left_trials": 0, "right_trials": 0}
    for eye in ["OE", "CE"]:
        base = folder / f"sub-{subject:02d}_task-{eye}baseline"
        edf, tsv = Path(str(base) + "_eeg.edf"), Path(str(base) + "_events.tsv")
        with edf.open("rb") as f:
            fixed = f.read(256)
            header = parse_edf_header(fixed + f.read(int(fixed[184:192]) - 256))
        events = pd.read_csv(tsv, sep="\t")
        codes = pd.to_numeric(events.trial_type)
        rest = {"edf": str(edf), "events": str(tsv), "events_sha256": sha256(tsv),
                "header": header, "file_size_matches_header": edf.stat().st_size == header["expected_file_bytes"],
                "start_markers_seconds": events.loc[codes == 32775, "onset"].tolist(),
                "end_markers_seconds": events.loc[codes == 32776, "onset"].tolist()}
        try:
            start, stop = dreyer_rest_interval(events, header["sfreqs"][0], header["sample_counts"][0])
            rest.update(boundary_status="pass", duration_seconds=(stop - start) / header["sfreqs"][0])
        except ValueError as error:
            rest.update(boundary_status="unresolved", reason=str(error), duration_seconds=None)
        row["rest"][eye] = rest
    for path in sorted(folder.glob(f"sub-{subject:02d}_task-R*_events.tsv")):
        events = pd.read_csv(path, sep="\t")
        codes = pd.to_numeric(events.trial_type)
        left, right = int((codes == 769).sum()), int((codes == 770).sum())
        row["runs"].append({"events": str(path), "sha256": sha256(path), "left": left, "right": right})
        row["left_trials"] += left
        row["right_trials"] += right
    record["subjects"].append(row)
save_json(root / "receipts/dreyer_raw_structure.json", record)
print(json.dumps({"audited": len(record["subjects"]), "ambiguous_subjects": [r["subject"] for r in record["subjects"] if any(s["boundary_status"] != "pass" for s in r["rest"].values())]}))
