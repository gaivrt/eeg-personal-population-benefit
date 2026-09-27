"""File-wise preprocessing and participant audit; never downloads or trains."""
from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch

from .real_data import (
    DREYER_AUXILIARIES, cho_trial_order, read_dreyer_rest, read_physionet_mi_events,
)
from .stage_a_common import ROOT, config, save_json, sha256

# Exact ordering from the fixed official preprocessing_physio.py.
PHYSIO_CHANNELS = (
    "Fc5. Fc3. Fc1. Fcz. Fc2. Fc4. Fc6. C5.. C3.. C1.. Cz.. C2.. C4.. C6.. "
    "Cp5. Cp3. Cp1. Cpz. Cp2. Cp4. Cp6. Fp1. Fpz. Fp2. Af7. Af3. Afz. Af4. Af8. "
    "F7.. F5.. F3.. F1.. Fz.. F2.. F4.. F6.. F8.. Ft7. Ft8. T7.. T8.. T9.. T10. "
    "Tp7. Tp8. P7.. P5.. P3.. P1.. Pz.. P2.. P4.. P6.. P8.. Po7. Po3. Poz. Po4. "
    "Po8. O1.. Oz.. O2.. Iz.."
).split()


def canonical_channel(name):
    target = {c.lower(): c for c in mne.channels.make_standard_montage("standard_1005").ch_names}
    return target[name.strip(".").lower()]


def filter_raw(raw, dataset, dataset_config=None, preprocessing=None):
    """Follow official PhysioNet operation order, at pinned MNE defaults."""
    cfg = config()
    processing = cfg["preprocessing"] if preprocessing is None else preprocessing
    d = cfg["datasets"][dataset] if dataset_config is None else dataset_config
    if raw.info["sfreq"] != d["sfreq"] or len(raw.ch_names) != d["channels"]:
        raise ValueError("Unexpected raw sampling rate/channel count")
    raw.load_data()
    if raw.info["bads"]:
        raw.set_montage("standard_1005", match_case=False)
        raw.interpolate_bads()
    raw.set_eeg_reference("average", verbose=False)
    raw.filter(l_freq=processing["highpass_hz"], h_freq=processing.get("lowpass_hz"), verbose=False)
    notch = processing.get("notch_hz", d["notch_hz"])
    raw.notch_filter(notch, verbose=False)
    raw.resample(processing["sfreq"], verbose=False)
    return raw


def filtered_array(volts, sfreq, channels, dataset, dataset_config=None, preprocessing=None):
    raw = mne.io.RawArray(volts, mne.create_info(channels, sfreq, "eeg"), verbose=False)
    return filter_raw(raw, dataset, dataset_config, preprocessing).get_data() * 1e4  # V -> uV / 100


def segment_rest(data, seconds=4, sfreq=200):
    width = seconds * sfreq
    n = data.shape[-1] // width
    return data[:, :n * width].reshape(data.shape[0], n, width).transpose(1, 0, 2)


def good_windows(data):
    return np.isfinite(data).all(axis=(1, 2))


def alpha_comparison(open_data, closed_data, channels, open_good, closed_good):
    cfg = config()["audit"]
    picks = [channels.index(c) for c in cfg["alpha_channels"] if c in channels]
    if closed_data is None:
        return {"status": "not_applicable", "reason": "No published eyes-closed rest"}
    if not picks:
        return {"status": "not_applicable", "reason": "No occipital O1/Oz/O2 channels"}
    n = min(len(open_data), len(closed_data))
    mask = open_good[:n] & closed_good[:n]
    if not mask.any():
        return {"status": "fail", "reason": "No matched clean windows"}
    powers = []
    for data in [open_data[:n][mask], closed_data[:n][mask]]:
        f, psd = welch(data[:, picks].astype(np.float64) * 100, fs=200,
                       nperseg=cfg["alpha_welch_seconds"] * 200, noverlap=200,
                       detrend="constant", window="hann", axis=-1)
        band = (f >= 8) & (f <= 13)
        powers.append(float(np.trapezoid(psd[..., band], f[band], axis=-1).mean()))
    ratio = powers[1] / powers[0] if powers[0] > 0 else None
    return {"status": "pass" if ratio is not None and ratio > cfg["alpha_required_ratio"] else "fail",
            "channels": [channels[i] for i in picks], "matched_windows": int(mask.sum()),
            "matched_seconds": int(mask.sum()) * 4, "open_power_uv2": powers[0],
            "closed_power_uv2": powers[1], "closed_open_ratio": ratio}


def save_signal(path, data):
    if not np.isfinite(data).all() or np.abs(data).max(initial=0) > np.finfo(np.float16).max:
        raise ValueError("Nonfinite signal or float16 overflow")
    np.save(path, np.asarray(data, dtype=np.float16), allow_pickle=False)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def physio(root, subject, *, preprocessing=None):
    matches = list(root.rglob(f"S{subject:03d}R01.edf"))
    if len(matches) != 1:
        raise ValueError("Expected one PhysioNet original subject directory")
    folder = matches[0].parent
    channels = [canonical_channel(c) for c in PHYSIO_CHANNELS]
    rest, rest_meta, tasks, rows, sources = {}, {}, [], [], []
    for run in [1, 2, 4, 8, 12]:
        path = folder / f"S{subject:03d}R{run:02d}.edf"
        sources.append(path)
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raw.pick(PHYSIO_CHANNELS)
        if raw.ch_names != PHYSIO_CHANNELS:
            raise ValueError("PhysioNet channel order disagrees with official ordering")
        duration = raw.n_times / raw.info["sfreq"]
        filter_raw(raw, "PhysionetMI", preprocessing=preprocessing)
        if run < 3:
            state = "open" if run == 1 else "closed"
            rest[state] = raw.get_data() * 1e4
            rest_meta[state] = {"source_file": str(path), "start_seconds": 0,
                                "stop_seconds": duration, "duration_seconds": duration,
                                "nominal_seconds": 60, "protocol_order": run - 1}
        else:
            events = read_physionet_mi_events(path)
            for e in events.itertuples():
                start = int(round(e.onset_seconds * 200))
                if start + 800 > raw.n_times:
                    raise ValueError("PhysioNet cue window exceeds run boundary")
                tasks.append(raw.get_data(start=start, stop=start + 800) * 1e4)
                rows.append({"label": int(e.label == "right_hand"), "source_file": str(path),
                             "source_event_index": e.source_event_index, "run": run,
                             "start_seconds": e.onset_seconds, "stop_seconds": e.onset_seconds + 4,
                             "source_event_duration": e.duration_seconds,
                             "protocol_order": {4: 2, 8: 3, 12: 4}[run]})
        raw.close()
    return channels, rest, rest_meta, np.stack(tasks), pd.DataFrame(rows), sources, {}


def dreyer(root, subject, *, preprocessing=None):
    matches = list(root.rglob(f"sub-{subject:02d}/eeg/sub-{subject:02d}_task-OEbaseline_eeg.edf"))
    if len(matches) != 1:
        raise ValueError("Expected one extracted Dreyer subject")
    dataset_root = matches[0].parents[2]
    rest, rest_meta, tasks, rows, sources = {}, {}, [], [], []
    channels = None
    for state, eye in [("open", "eyes_open"), ("closed", "eyes_closed")]:
        result = read_dreyer_rest(dataset_root, subject, eye, use_available_interval=True)
        channels = result["channels"]
        rest[state] = filtered_array(result["data"], result["sfreq"], channels, "Dreyer2023", preprocessing=preprocessing)
        rest_meta[state] = dict(result["metadata"], nominal_seconds=180,
                                 protocol_order=0 if state == "open" else 1)
        sources.append(Path(result["metadata"]["source_file"]))
    files = sorted(matches[0].parent.glob(f"sub-{subject:02d}_task-R*_eeg.edf"))
    if len(files) != (4 if subject == 59 else 6):
        raise ValueError("Unexpected number of Dreyer MI runs")
    for path in files:
        run = int(re.search(r"_task-R(\d)", path.name)[1])
        events_file = Path(str(path).replace("_eeg.edf", "_events.tsv"))
        events = pd.read_csv(events_file, sep="\t")
        codes = pd.to_numeric(events.trial_type, errors="raise")
        cues = events[codes.isin([769, 770])]
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raw.pick([c for c in raw.ch_names if c not in DREYER_AUXILIARIES])
        if raw.ch_names != channels:
            raise ValueError("Task/rest channel order mismatch")
        if not len(cues):
            raise ValueError(f"Dreyer run {run}: no class cues")
        if not np.allclose(cues["sample"] / raw.info["sfreq"], cues.onset, atol=0.5 / 512, rtol=0):
            raise ValueError("Dreyer cue samples disagree with onset")
        # Cross-check BIDS cues against original EDF annotations, including A01 recovery.
        original = [(float(o), str(d)) for o, d in zip(raw.annotations.onset, raw.annotations.description)
                    if str(d) in ("769", "770")]
        expected = [(float(r.onset), str(int(r.trial_type))) for r in cues.itertuples()]
        if len(original) != len(expected) or any(d1 != d2 or abs(o1-o2) > 1/512
               for (o1, d1), (o2, d2) in zip(original, expected)):
            raise ValueError("Dreyer EDF annotations and BIDS class cues disagree")
        filter_raw(raw, "Dreyer2023", preprocessing=preprocessing)
        for e in cues.itertuples():
            start = int(round(e.onset * 200))
            if start + 800 > raw.n_times:
                raise ValueError("Dreyer cue window exceeds source recording")
            tasks.append(raw.get_data(start=start, stop=start + 800) * 1e4)
            rows.append({"label": int(e.trial_type == 770), "run": run, "source_file": str(path),
                         "source_event_index": e.Index, "start_seconds": e.onset,
                         "stop_seconds": e.onset + 4, "protocol_order": run + 1})
        raw.close()
        sources.extend([path, events_file])
    return channels, rest, rest_meta, np.stack(tasks), pd.DataFrame(rows), sources, {
        "edf_bids_cue_roundtrip": "pass", "a01_recovered_runs": subject == 1}


def cho(root, subject, *, preprocessing=None, write_order=True):
    matches = list(root.rglob(f"s{subject:02d}.mat"))
    if len(matches) != 1:
        raise ValueError("Expected one original Cho MAT file")
    path = matches[0]
    sequence_file = ROOT / f"docs/data_access_audit_2026-09-25/sources/cho_trial_sequence/s{subject}_trial_sequence_v1.mat"
    eeg = loadmat(path, simplify_cells=True)["eeg"]
    if re.fullmatch(r"subject\s+0*" + str(subject), str(eeg["subject"]), re.IGNORECASE) is None:
        raise ValueError("Cho MAT identity mismatch")
    sequence = loadmat(sequence_file, simplify_cells=True)["trial_sequence"]
    order = cho_trial_order(eeg, sequence)
    channels = mne.channels.make_standard_montage("biosemi64").ch_names
    sfreq = float(eeg["srate"])
    rest_raw = np.asarray(eeg["rest"])
    if rest_raw.shape[0] != 68 or sfreq != 512:
        raise ValueError("Cho rest schema or sampling rate mismatch")
    rest = {"open": filtered_array(rest_raw[:64].astype(np.float64) * 1e-6, sfreq, channels, "Cho2017", preprocessing=preprocessing)}
    seconds = rest_raw.shape[-1] / sfreq
    rest_meta = {"open": {"source_file": str(path), "source_field": "eeg.rest", "start_seconds": 0,
                           "stop_seconds": seconds, "duration_seconds": seconds, "nominal_seconds": 60,
                           "protocol_order": 0, "boundary_basis": "complete_published_rest_field"}}
    tasks, rows = [], []
    for e in order.itertuples():
        signal = np.asarray(eeg["imagery_" + e.label.split("_")[0]])
        original = signal[:64, e.packed_epoch_start:e.packed_epoch_stop_exclusive]
        # Process the actual isolated seven-second released trial; never concatenate classes.
        processed = filtered_array(original.astype(np.float64) * 1e-6, sfreq, channels, "Cho2017", preprocessing=preprocessing)
        start = int(round(e.event_offset_in_epoch * 200 / sfreq))
        task = processed[:, start:start + 600]
        if task.shape != (64, 600):
            raise ValueError("Cho imagery window exceeds packed source trial")
        tasks.append(task)
        rows.append({"label": int(e.label == "right_hand"), "source_file": str(path),
                     "source_field": "imagery_" + e.label.split("_")[0],
                     "class_local_trial_index": e.class_local_trial_index,
                     "packed_epoch_start": e.packed_epoch_start,
                     "packed_epoch_stop_exclusive": e.packed_epoch_stop_exclusive,
                     "packed_cue_sample": e.packed_event_sample,
                     "original_event_sample_matlab": e.original_event_sample_matlab,
                     "start_seconds": e.original_event_sample_zero / sfreq,
                     "stop_seconds": e.original_event_sample_zero / sfreq + 3,
                     "protocol_order": 1})
        # Exact full-epoch mapping roundtrip to the published packed class is built into slicing;
        # additionally assert cue identity and complete 7-s width for every source trial.
        if eeg["imagery_event"][e.packed_event_sample] != 1 or original.shape != (64, 3584):
            raise ValueError("Cho full-epoch/cue roundtrip failed")
    if write_order:
        order.to_csv(path.parent / f"s{subject:02d}_stage_a_order.csv", index=False)
    return channels, rest, rest_meta, np.stack(tasks), pd.DataFrame(rows), [path, sequence_file], {
        "cho_order_mapping_and_signal_roundtrip": "pass", "roundtrip_trials": len(order),
        "independent_continuous_event_timeline": False,
        "rest_to_mi_absolute_gap_available": False,
        "published_bad_trial_indices": np.asarray(eeg["bad_trial_indices"]).tolist()}


READERS = {"PhysionetMI": physio, "Dreyer2023": dreyer, "Cho2017": cho}


def preprocess_subject(root, dataset, subject):
    cfg = config()
    out = root / "processed" / dataset / f"sub-{subject:03d}"
    out.mkdir(parents=True, exist_ok=True)
    channels, rest, meta, tasks, trials, sources, checks = READERS[dataset](root / "raw", subject)
    failures = []
    arrays, masks = {}, {}
    for state, signal in rest.items():
        arrays[state] = save_signal(out / f"rest_{state}.npy", segment_rest(signal))
        masks[state] = good_windows(arrays[state])
        np.save(out / f"rest_{state}_valid.npy", masks[state], allow_pickle=False)
        m = meta[state]
        m["complete_segment_seconds"] = len(arrays[state]) * 4
        m["usable_seconds"] = int(masks[state].sum()) * 4
        m["incomplete_tail_seconds"] = m["duration_seconds"] - len(arrays[state]) * 4
        m["rejected_segment_seconds"] = int((~masks[state]).sum()) * 4
        lo, hi = cfg["datasets"][dataset]["rest_seconds_range"]
        if not lo <= m["duration_seconds"] <= hi:
            failures.append(f"{state}_rest_duration")
        if m["usable_seconds"] < cfg["audit"]["min_usable_eyes_open_seconds"]:
            failures.append(f"{state}_rest_usable_under_30s")
    if "open" not in rest or (dataset != "Cho2017" and "closed" not in rest):
        failures.append("missing_expected_rest")
    stored_tasks = save_signal(out / "tasks.npy", tasks)
    task_valid = good_windows(stored_tasks)
    labels = trials.label.to_numpy(dtype=np.int64)
    np.save(out / "labels.npy", labels, allow_pickle=False)
    np.save(out / "task_valid.npy", task_valid, allow_pickle=False)
    trials.insert(0, "trial_id", [f"{dataset}:{subject:03d}:{i:04d}" for i in range(len(trials))])
    trials["quality_valid"] = task_valid
    trials.to_csv(out / "trials.csv", index=False)
    counts = np.bincount(labels, minlength=2)
    usable = np.bincount(labels[task_valid], minlength=2)
    imbalance = float(abs(int(usable[0]) - int(usable[1])) / max(1, usable.sum()))
    if usable.min() == 0:
        failures.append("class_missing")
    if dataset == "Dreyer2023" and counts.tolist() != ([80, 80] if subject == 59 else [120, 120]):
        failures.append("unexpected_trial_counts")
    if dataset == "Cho2017" and (counts[0] != counts[1] or counts[0] not in (100, 120)):
        failures.append("unexpected_trial_counts")
    # No fabricated absolute clock: order is proved at protocol/recording level;
    # within each source recording use measured event positions.
    if max(m["protocol_order"] for m in meta.values()) >= trials.protocol_order.min():
        failures.append("context_query_protocol_order")
    for _, group in trials.groupby(["source_file", "protocol_order"]):
        if not group.start_seconds.is_monotonic_increasing:
            failures.append("query_order")
    alpha = alpha_comparison(arrays["open"], arrays.get("closed"), channels,
                             masks["open"], masks.get("closed"))
    if alpha["status"] == "fail":
        failures.append("alpha_direction")
    metadata = {"dataset": dataset, "subject": subject, "session": 0, "channels": channels,
                "sfreq": 200, "units": "uV/100", "dtype": "float16", "rest": meta,
                "source_files": [{"path": str(p), "sha256": sha256(p)} for p in sources],
                "absolute_rest_query_gap_available": False, "checks": checks,
                "preprocessing_config_sha256": sha256(ROOT / "configs/stage_a.yaml")}
    save_json(out / "metadata.json", metadata)
    report = {"dataset": dataset, "subject": subject, "status": "fail" if failures else "pass",
              "failures": failures, "alpha": alpha, "left_trials": int(counts[0]),
              "right_trials": int(counts[1]), "usable_left": int(usable[0]),
              "usable_right": int(usable[1]), "imbalance": imbalance,
              "rest": meta, "checks": checks,
              "processed_signal_bytes": sum(p.stat().st_size for p in out.glob("*.npy"))}
    save_json(out / "audit.json", report)
    return report
