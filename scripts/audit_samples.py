"""Read-only signal/header audit of the explicitly bounded subject-1 files."""
from collections import Counter
import gc
import json
import os
from pathlib import Path
import zipfile

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data/samples"
OUT = ROOT / "reports/stage0a/data_cards"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["MNE_DATA"] = str(SAMPLES)
os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")


def write(name, result):
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(name, "audited", flush=True)


def physionet():
    records = []
    for run in [1, 2, 4, 8, 12]:
        file = SAMPLES / f"PhysionetMI/S001R{run:02d}.edf"
        raw = mne.io.read_raw_edf(file, preload=False, verbose=False)
        checked = raw.get_data() if run in [1, 2] else raw.get_data(start=0, stop=160)
        assert np.isfinite(checked).all() and np.std(checked) > 0
        records.append({"run": run, "sfreq": raw.info["sfreq"], "seconds": raw.n_times / raw.info["sfreq"],
                        "channels": raw.ch_names, "absolute_start": str(raw.info["meas_date"]),
                        "events": dict(Counter(raw.annotations.description)),
                        "onsets_monotonic": bool(np.all(np.diff(raw.annotations.onset) >= 0)),
                        "first_event_seconds": float(raw.annotations.onset[0])})
    return {"status": "verified_subject_1", "subjects_downloaded": [1], "sessions_observed": [1], "runs": records,
            "rest_seconds": {"eyes_open": records[0]["seconds"], "eyes_closed": records[1]["seconds"]},
            "left_trials": sum(r["events"].get("T1", 0) for r in records[2:]),
            "right_trials": sum(r["events"].get("T2", 0) for r in records[2:]),
            "timestamp_caveat": "EDF absolute_start repeats across runs; use official run order plus within-run offsets"}


def stieger():
    records = []
    for session in [1, 2]:
        file = SAMPLES / f"Stieger2021/S1_Session_{session}.mat"
        c = loadmat(file, squeeze_me=True, struct_as_record=False)["BCI"]
        assert np.isfinite(c.data[0]).all()
        td = c.TrialData
        trial_rows = [{"index": i, "task": int(t.tasknumber), "run": int(t.runnumber), "trial": int(t.trialnumber),
                       "target": int(t.targetnumber), "triallength": float(t.triallength), "artifact": int(t.artifact),
                       "samples": int(c.data[i].shape[-1]), "relative_start_ms": float(c.time[i][0]),
                       "relative_end_ms": float(c.time[i][-1])} for i, t in enumerate(td)]
        pd.DataFrame(trial_rows).to_csv(OUT / f"Stieger2021_subject1_session{session}_trials.csv", index=False)
        counts = Counter(int(t.targetnumber) for t in td)
        filtered = Counter(int(t.targetnumber) for t in td if t.artifact == 0 and t.triallength >= 3)
        order = [(int(t.runnumber), int(t.trialnumber)) for t in td]
        channels = c.chaninfo.label.tolist()
        standard = mne.channels.make_standard_montage("standard_1005").ch_names
        kept = [s.replace("Z", "z").replace("FP", "Fp") for s in channels]
        records.append({"session": session, "sfreq": int(c.SRATE), "channels_raw": channels,
                        "channels_moabb_standard": [ch for ch in kept if ch in standard],
                        "n_trials": len(td), "all_target_counts": dict(counts),
                        "left_trials_raw": counts[2], "right_trials_raw": counts[1],
                        "left_trials_noartifact_at_least_3s": filtered[2],
                        "right_trials_noartifact_at_least_3s": filtered[1],
                        "recorded_seconds_sum": sum(a.shape[-1] for a in c.data) / int(c.SRATE),
                        "run_trial_order_monotonic": all(a < b for a, b in zip(order, order[1:])),
                        "relative_time_resets": True, "top_level_fields": c._fieldnames,
                        "metadata_fields": c.metadata._fieldnames,
                        "standalone_initial_rest_field": False})
        del c, td
        gc.collect()
    return {"status": "verified_subject_1_sessions_1_2", "subjects_downloaded": [1], "sessions": records,
            "rest_caveat": "target 4 is cued rest task; no separate opening resting baseline in inspected structures",
            "order_caveat": "time is trial-relative ms, not an absolute continuous session clock; preserve run/trial order and gaps"}


def lee():
    records = []
    for session in [1, 2]:
        file = SAMPLES / f"Lee2019_MI/sess{session:02d}_subj01_EEG_MI.mat"
        mat = loadmat(file, simplify_cells=True)
        runs = []
        for phase in ["train", "test"]:
            data = mat[f"EEG_MI_{phase}"]
            labels = np.asarray(data["y_dec"]).ravel()
            times = np.asarray(data["t"]).ravel()
            channels = [str(c) for c in data["chan"]]
            assert np.isfinite(data["x"][:1000]).all()
            for rest_key in ["pre_rest", "post_rest"]:
                assert np.isfinite(data[rest_key]).all() and np.std(data[rest_key]) > 0
            runs.append({"phase": phase, "sfreq": int(data["fs"]), "channels": channels,
                         "continuous_shape": list(data["x"].shape), "fields": list(data),
                         "class_map": np.asarray(data["class"]).tolist(), "class_counts": dict(Counter(map(int, labels))),
                         "left_trials": int((labels == 2).sum()), "right_trials": int((labels == 1).sum()),
                         "pre_rest_seconds": len(data["pre_rest"]) / int(data["fs"]),
                         "post_rest_seconds": len(data["post_rest"]) / int(data["fs"]),
                         "event_samples_first": int(times[0]), "event_samples_last": int(times[-1]),
                         "timestamps_monotonic": bool(np.all(np.diff(times) > 0))})
        records.append({"session": session, "runs": runs})
        del mat, data
        gc.collect()
    return {"status": "verified_subject_1_both_sessions", "subjects_downloaded": [1], "sessions": records,
            "order_caveat": "train/test onset sample indices are local to each run; pre/post rest is stored separately"}


def cho():
    file = SAMPLES / "Cho2017/s01.mat"
    data = loadmat(file, simplify_cells=True)["eeg"]
    sequence = loadmat(ROOT / "docs/data_access_audit_2026-09-25/sources/cho_trial_sequence/s1_trial_sequence_v1.mat", simplify_cells=True)["trial_sequence"]
    shape_fields = {k: list(v.shape) for k, v in data.items() if isinstance(v, np.ndarray)}
    fs = int(data["srate"])
    rest = {k: {"shape": list(v.shape), "seconds": v.shape[-1] / fs} for k, v in data.items()
            if isinstance(v, np.ndarray) and v.ndim == 2 and "rest" in k.lower()}
    events = np.asarray(data["imagery_event"]).ravel()
    onset = np.flatnonzero((events > 0) & np.r_[True, events[:-1] <= 0])
    left = np.asarray(sequence["imagery_left"]).ravel()
    right = np.asarray(sequence["imagery_right"]).ravel()
    assert len(left) == len(right) == len(onset)
    rows = sorted([{"class": side, "original_onset_sample": int(t), "class_local_trial_index": i}
                   for side, times in [("left", left), ("right", right)] for i, t in enumerate(times)], key=lambda r: r["original_onset_sample"])
    pd.DataFrame(rows).to_csv(OUT / "Cho2017_subject1_original_trial_order.csv", index=False)
    assert np.isfinite(data["imagery_left"][:, :1000]).all()
    assert np.isfinite(data["rest"]).all() and np.std(data["rest"]) > 0
    # MOABB's reader contains the documented 64-channel BioSemi order.
    from moabb.datasets import Cho2017
    dataset = Cho2017()
    dataset.data_path = lambda *args, **kwargs: str(file)
    raw = dataset._get_single_subject_data(1)["0"]["0"]
    channels = [name for name, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "eeg"]
    return {"status": "verified_subject_1", "subjects_downloaded": [1], "sessions_observed": [1],
            "sfreq": fs, "channels": channels, "fields_and_shapes": shape_fields, "rest_fields": rest,
            "left_trials": len(left), "right_trials": len(right), "class_event_onsets": len(onset),
            "original_sample_onsets_unique": len(set(left) | set(right)) == len(left) + len(right),
            "moabb_load": "passed", "order_caveat": "MOABB stacks all left then all right; use separate trial_sequence for chronology",
            "rest_caveat": "noise/rest array presence alone does not establish that it preceded the MI trials"}


def dreyer():
    folder = SAMPLES / "Dreyer2023"
    archive = folder / "sub-01.zip"
    with zipfile.ZipFile(archive) as z:
        assert sum(i.file_size for i in z.infolist()) < 2_000_000_000
        for member in z.infolist():
            assert (folder / member.filename).resolve().is_relative_to(folder.resolve())
            assert ((member.external_attr >> 16) & 0o170000) != 0o120000
        z.extractall(folder)
    files = []
    readers = {".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf,
               ".vhdr": mne.io.read_raw_brainvision, ".set": mne.io.read_raw_eeglab, ".gdf": mne.io.read_raw_gdf}
    for file in sorted(folder.rglob("*")):
        if file.suffix not in readers:
            continue
        raw = readers[file.suffix](file, preload=False, verbose=False)
        assert np.isfinite(raw.get_data(start=0, stop=min(512, raw.n_times))).all()
        events_path = file.with_name(file.stem.removesuffix("_eeg") + "_events.tsv")
        events = pd.read_csv(events_path, sep="\t") if events_path.exists() else pd.DataFrame()
        counts = {str(col): {str(k): int(v) for k, v in events[col].value_counts().items()}
                  for col in ["trial_type", "value"] if col in events}
        rest_interval = None
        if "baseline" in file.name:
            start = float(events.loc[events.trial_type.astype(str) == "32775", "onset"].iloc[0])
            end = float(events.loc[events.trial_type.astype(str) == "32776", "onset"].iloc[0])
            rest_interval = {"start_seconds": start, "end_seconds": end, "seconds": end - start}
            signal = raw.get_data(start=round(start * raw.info["sfreq"]), stop=round(end * raw.info["sfreq"]))
            assert np.isfinite(signal).all() and np.std(signal) > 0
        files.append({"file": str(file.relative_to(folder)), "sfreq": raw.info["sfreq"], "channels": raw.ch_names,
                      "rest_interval": rest_interval,
                      "seconds": raw.n_times / raw.info["sfreq"], "annotation_counts": dict(Counter(raw.annotations.description)),
                      "event_columns": list(events.columns), "event_counts": counts,
                      "onsets_monotonic": bool(np.all(np.diff(events.onset) >= 0)) if "onset" in events else None})
    assert files, "No readable EEG files in subject-1 BIDS archive"
    return {"status": "verified_subject_1", "subjects_downloaded": [1], "sessions_observed": [1], "files": files,
            "order_caveat": "Use task run order + within-file onsets; MOABB task loader excludes baseline/rest tasks"}


def main():
    failures = []
    for name, func in [("PhysionetMI", physionet), ("Stieger2021", stieger), ("Lee2019_MI", lee), ("Cho2017", cho), ("Dreyer2023", dreyer)]:
        try:
            write(name, func())
        except Exception as error:
            write(name, {"status": "error", "error": repr(error)})
            failures.append(name)
        gc.collect()
    if failures:
        raise SystemExit("Audit incomplete: " + ", ".join(failures))


if __name__ == "__main__":
    main()
