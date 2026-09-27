"""Reproduce the bounded data audit, all EDF headers and associated tests.

One command: .venv/Scripts/python.exe scripts/audit_data_followup.py
Previously downloaded EEG samples and official Cho sequences are required.
The header step resumes its cache; it never requests complete EEG signals.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subject_context.real_data import read_dreyer_rest, read_physionet_mi_events, read_cho_imagery, cho_sequence_order


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()), encoding="utf-8")


def subject_summary(output):
    rows = json.loads((output / "physionet_headers.json").read_text())
    assert {(r["subject"], r["run"]) for r in rows} == {(s, r) for s in range(1, 110) for r in range(1, 15)}
    assert len(rows) == 1526
    summaries = []
    table = ["# PhysioNet 全部头信息偏离项", "",
             "2026-09-25：覆盖 109 人的 R01–R14，1,526 个 EDF。标记是结构描述，不是排除名单。",
             "各 run 众数：R01/R02 为 61 秒，R03–R14 为 123 秒。表内列出所有非众数时长、非 160 Hz 文件；未列 run 为 160 Hz 且时长等于对应众数。",
             "",
             "| 被试 | 非 160 Hz | 非众数时长（秒） |", "|---|---|---|"]
    for subject in range(1, 110):
        person = [r for r in rows if r["subject"] == subject]
        rate = ", ".join(f'R{r["run"]:02d}={r["sfreqs"][0]:g}' for r in person if "sampling_rate" in r["flags"])
        duration = ", ".join(f'R{r["run"]:02d}={r["duration_seconds"]:g}' for r in person
                             if "duration_differs_from_run_mode" in r["flags"])
        # This descriptive envelope is the observed three frequent task lengths,
        # not an inclusion threshold or a replacement for annotation inspection.
        unusual = [r["run"] for r in person if r["run"] >= 3 and r["duration_seconds"] not in (123, 124, 125)]
        summaries.append({"subject": subject, "files": len(person), "sampling_rate_deviations": rate,
                          "duration_deviations": duration, "task_runs_outside_123_124_125s": unusual,
                          "flags": sorted({f for r in person for f in r["flags"]})})
        if rate or duration:
            table.append(f'| S{subject:03d} | {rate or "—"} | {duration or "—"} |')
    pd.DataFrame(summaries).to_csv(output / "physionet_subjects.csv", index=False)
    (output / "physionet_subjects.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    return {"non_160_hz_subjects": [r["subject"] for r in summaries if r["sampling_rate_deviations"]],
            "unusual_task_duration_subjects": [r["subject"] for r in summaries if r["task_runs_outside_123_124_125s"]],
            "duration_histogram_seconds": dict(sorted(Counter(r["duration_seconds"] for r in rows).items())),
            "sampling_rate_histogram_hz": dict(Counter(str(r["sfreqs"]) for r in rows)),
            "header_requests_all_206": all(r["http_statuses"] == [206, 206] for r in rows),
            "channel_count_or_file_length_errors": sum(bool(set(r["flags"]) & {"channel_count", "file_size_mismatch"}) for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_audit.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    output = ROOT / cfg["output"]
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_physionet_headers.py"), "--config", args.config],
                   cwd=ROOT, check=True)
    receipts = []

    def receipt(path):
        path = Path(path)
        receipts.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                         "sha256": sha256(path)})

    subject = cfg["physionet"]["sample_subject"]
    events = []
    for run in cfg["physionet"]["target_runs"]:
        path = ROOT / cfg["physionet"]["sample_dir"] / f"S{subject:03d}R{run:02d}.edf"
        events.append(read_physionet_mi_events(path))
        receipt(path)
    events = pd.concat(events, ignore_index=True).sort_values(["original_run", "onset_seconds"])
    events.to_csv(output / "physionet_sample_events.csv", index=False)

    dreyer = []
    for state in ("eyes_open", "eyes_closed"):
        rest = read_dreyer_rest(ROOT / cfg["dreyer"]["sample_root"], cfg["dreyer"]["sample_subject"], state)
        dreyer.append({**rest["metadata"], "sfreq": rest["sfreq"], "channels": rest["channels"],
                       "shape": list(rest["data"].shape),
                       "data_sha256_float64_V": hashlib.sha256(rest["data"].tobytes()).hexdigest()})
        path = Path(rest["metadata"]["source_file"])
        for suffix in ["_eeg.edf", "_eeg.json", "_events.tsv"]:
            receipt(path.with_name(path.name.replace("_eeg.edf", suffix)))
    dump(output / "dreyer_rest.json", dreyer)

    sequence_dir = ROOT / cfg["cho"]["sequence_dir"]
    sequence_rows = []
    for path in sorted(sequence_dir.glob("s*_trial_sequence_v1.mat")):
        order = cho_sequence_order(loadmat(path, simplify_cells=True)["trial_sequence"])
        counts = order.label.value_counts().to_dict()
        sequence_rows.append({"file": path.name, "trials": len(order), **counts,
                              "first_original_sample_matlab": int(order.original_event_sample_matlab.iloc[0]),
                              "last_original_sample_matlab": int(order.original_event_sample_matlab.iloc[-1]),
                              "sha256": sha256(path)})
        receipt(path)
    assert len(sequence_rows) == 52
    pd.DataFrame(sequence_rows).to_csv(output / "cho_all_sequences.csv", index=False)
    cho_path = ROOT / cfg["cho"]["sample"]
    cho_subject = int(cho_path.stem[1:])
    cho = read_cho_imagery(cho_path, sequence_dir / f"s{cho_subject}_trial_sequence_v1.mat")
    cho["order"].to_csv(output / "cho_sample_chronological_trials.csv", index=False)
    receipt(cho_path)
    eeg = loadmat(cho_path, simplify_cells=True)["eeg"]
    cho_summary = {**cho["metadata"], "shape": list(cho["data"].shape), "sfreq": cho["sfreq"],
                   "channels": cho["channels"], "n_sequence_subjects": len(sequence_rows),
                   "total_sequence_trials": sum(row["trials"] for row in sequence_rows),
                   "trial_count_distribution": dict(Counter(row["trials"] for row in sequence_rows)),
                   "rest_field_shape": list(eeg["rest"].shape),
                   "rest_field_seconds": eeg["rest"].shape[-1] / float(eeg["srate"]),
                   "nominal_rest_seconds": 60, "rest_eye_state_from_paper": "eyes_open",
                   "rest_position_from_protocol": "before_movement_practice_and_MI",
                   "exact_60_second_rest_markers_available": False,
                   "packed_cue_offsets": sorted(set(map(int, cho["order"].event_offset_in_epoch))),
                   "full_epoch_data_sha256_float64_V": hashlib.sha256(cho["data"].tobytes()).hexdigest()}
    dump(output / "cho_sample.json", cho_summary)
    overview = {"utc": datetime.now(timezone.utc).isoformat(), "config": args.config,
                "physionet": subject_summary(output),
                "physionet_sample_label_counts": events.label.value_counts().to_dict(),
                "dreyer_sample_marked_seconds": {row["eye_state"]: row["duration_seconds"] for row in dreyer},
                "cho_sequence_subjects": len(sequence_rows), "cho_sample_trials": len(cho["order"]),
                "new_full_signal_downloads": 0, "gpu_jobs_submitted": 0}
    for path in [ROOT / args.config, ROOT / "src/subject_context/real_data.py",
                 ROOT / "src/subject_context/edf_header.py", Path(__file__),
                 ROOT / "scripts/audit_physionet_headers.py", ROOT / "tests/test_real_data.py"]:
        receipt(path)
    dump(output / "input_receipts.json", receipts)
    dump(output / "summary.json", overview)
    print(json.dumps(overview, indent=2), flush=True)
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", "--junitxml=" + str(output / "tests.xml")],
                          cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / "tests.log").write_text(test.stdout, encoding="utf-8")
    print(test.stdout, flush=True)
    test.check_returncode()


if __name__ == "__main__":
    main()
