"""Dataset-specific readers preserving source run IDs, events and rest boundaries.

These functions consume the original files downloaded by MOABB. They do not
download data, filter artefacts, resample, or choose an experimental exclusion.
"""
import json
from numbers import Integral
from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat


PHYSIONET_MI_RUNS = (4, 8, 12)
DREYER_AUXILIARIES = frozenset(("EOG1", "EOG2", "EOG3", "EMGg", "EMGd"))


def physionet_label(original_run, annotation):
    """Map original R04/R08/R12 annotations; T0 is not a query label.

    Other runs are deliberately rejected, including executed left/right runs
    and bilateral imagery. MOABB's renumbered run keys are not valid inputs.
    """
    if (isinstance(original_run, bool) or not isinstance(original_run, Integral)
            or original_run not in PHYSIONET_MI_RUNS):
        raise ValueError("Left/right MI requires original EDF run 4, 8 or 12")
    if annotation not in ("T0", "T1", "T2"):
        raise ValueError(f"Unknown PhysioNet annotation: {annotation!r}")
    return {"T0": None, "T1": "left_hand", "T2": "right_hand"}[annotation]


def read_physionet_mi_events(edf_file):
    """Read MI annotations with source identity, without loading EEG samples."""
    path = Path(edf_file)
    match = re.fullmatch(r"S(\d{3})R(\d{2})\.edf", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Expected an original SxxxRyy.edf filename")
    subject, run = map(int, match.groups())
    if not 1 <= subject <= 109:
        raise ValueError("PhysioNet subject must be in 1..109")
    physionet_label(run, "T0")  # Validate before opening the file.
    with mne.io.read_raw_edf(path, preload=False, verbose=False) as raw:
        sfreq, seconds = float(raw.info["sfreq"]), raw.n_times / raw.info["sfreq"]
        rows = []
        previous = -np.inf
        for i, (onset, duration, annotation) in enumerate(zip(
                raw.annotations.onset, raw.annotations.duration, raw.annotations.description)):
            if onset < previous or onset < 0 or duration <= 0 or onset + duration > seconds + 1/sfreq:
                raise ValueError("Invalid/nonchronological PhysioNet event boundary")
            previous = onset
            label = physionet_label(run, annotation)
            if label is not None:
                rows.append({"subject": subject, "original_run": run, "source_event_index": i,
                             "label": label, "annotation": str(annotation),
                             "onset_seconds": float(onset), "duration_seconds": float(duration),
                             "onset_sample": int(round(onset * sfreq)), "sfreq": sfreq})
    return pd.DataFrame(rows)


def dreyer_rest_interval(events, sfreq, n_times):
    """Validate the baseline start/end trigger pair; return half-open samples.

    BIDS trial_type preserves GDF codes; value is a remapped event index.
    The 786 code also occurs in OE files, so it cannot determine eye state.
    """
    codes = pd.to_numeric(events["trial_type"], errors="raise").to_numpy()
    if np.any(np.isin(codes, [769, 770])):
        raise ValueError("Motor-imagery cues found in a baseline file")
    positions = []
    for code in (32775, 32776):
        selected = events.loc[codes == code]
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one baseline marker {code}")
        sample, onset = float(selected.iloc[0]["sample"]), float(selected.iloc[0]["onset"])
        if not np.isfinite([sample, onset]).all() or sample != round(sample):
            raise ValueError("Nonintegral/nonfinite baseline marker")
        if abs(sample / sfreq - onset) > 0.5 / sfreq:
            raise ValueError("Baseline sample and onset disagree")
        positions.append(int(sample))
    start, stop = positions
    if not 0 <= start < stop <= n_times:
        raise ValueError("Baseline markers outside recording or reversed")
    return start, stop


def read_dreyer_rest(root, subject, eye_state="eyes_open", *, use_available_interval=False):
    """Read marked rest from extracted BIDS sub-XX/eeg files, in volts.

    root is the directory containing sub-XX, including MOABB's extracted ZIP
    directory. Only the marked interval is returned, not EDF padding. Auxiliary
    channels are removed by name because the published TSV labels them EEG.
    """
    tasks = {"eyes_open": "OEbaseline", "eyes_closed": "CEbaseline"}
    if eye_state not in tasks:
        raise ValueError("eye_state must be eyes_open or eyes_closed")
    if isinstance(subject, bool) or not isinstance(subject, Integral) or not 1 <= subject <= 87:
        raise ValueError("Dreyer subject must be in 1..87")
    task = tasks[eye_state]
    base = Path(root) / f"sub-{subject:02d}" / "eeg" / f"sub-{subject:02d}_task-{task}"
    path = Path(str(base) + "_eeg.edf")
    description = json.loads(Path(str(base) + "_eeg.json").read_text(encoding="utf-8"))
    if description["TaskName"] != task:
        raise ValueError("TaskName disagrees with baseline filename")
    events = pd.read_csv(str(base) + "_events.tsv", sep="\t")
    with mne.io.read_raw_edf(path, preload=False, verbose=False) as raw:
        sfreq = float(raw.info["sfreq"])
        if sfreq != float(description["SamplingFrequency"]):
            raise ValueError("EDF and JSON sampling rates disagree")
        boundary_basis = "unique_start_end_markers"
        try:
            start, stop = dreyer_rest_interval(events, sfreq, raw.n_times)
        except ValueError:
            if not use_available_interval:
                raise
            codes = pd.to_numeric(events["trial_type"], errors="raise")
            starts = events.loc[codes == 32775, "sample"].to_numpy(dtype=int)
            stops = events.loc[codes == 32776, "sample"].to_numpy(dtype=int)
            if not len(starts) or codes.isin([769, 770]).any():
                raise ValueError("No identifiable rest-only interval")
            start = int(starts.max())
            if len(stops):
                stop = int(stops.min())
                boundary_basis = "overlap_of_duplicate_marker_intervals"
            else:
                stop = min(raw.n_times, int(float(description["RecordingDuration"]) * sfreq))
                boundary_basis = "start_marker_to_published_recording_duration"
            if not 0 <= start < stop <= raw.n_times:
                raise ValueError("No available rest interval")
        if not DREYER_AUXILIARIES.issubset(raw.ch_names):
            raise ValueError("Missing Dreyer auxiliary channels; inspect channel schema")
        channels = [name for name in raw.ch_names if name not in DREYER_AUXILIARIES]
        if len(channels) != 27 or len(set(channels)) != 27:
            raise ValueError("Expected 27 distinct EEG channels after auxiliary exclusion")
        data = raw.get_data(picks=channels, start=start, stop=stop)
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite baseline EEG")
        metadata = {"subject": int(subject), "eye_state": eye_state, "source_file": str(path),
                    "unit": "V", "start_sample": start, "stop_sample_exclusive": stop,
                    "boundary_basis": boundary_basis,
                    "start_seconds": start/sfreq, "stop_seconds": stop/sfreq,
                    "duration_seconds": (stop-start)/sfreq,
                    "edf_duration_seconds": raw.n_times/sfreq,
                    "json_recording_duration_seconds": description.get("RecordingDuration"),
                    "protocol_order": ["OEbaseline", "CEbaseline", "R1", "R2", "R3", "R4", "R5", "R6"],
                    "absolute_session_timestamps_available": False,
                    "excluded_channels": [c for c in raw.ch_names if c in DREYER_AUXILIARIES]}
    return {"data": data, "sfreq": sfreq, "channels": channels, "metadata": metadata}


def cho_sequence_order(sequence):
    """Restore class-local trial IDs in the order of official MATLAB positions.

    Keep the published 1-based positions as the authoritative column. Their
    zero-based conversion is explicit; neither column is a synthetic clock
    obtained by concatenating the packed left/right arrays.
    """
    rows = []
    for side in ("left", "right"):
        times = np.asarray(sequence[f"imagery_{side}"]).reshape(-1)
        if (not len(times) or not np.isfinite(times).all() or np.any(times != np.floor(times))
                or np.any(times < 1) or np.any(np.diff(times) <= 0)):
            raise ValueError("Official class sequence must contain increasing positive integer positions")
        rows.extend({"label": f"{side}_hand", "class_local_trial_index": i,
                     "original_event_sample_matlab": int(t), "original_event_sample_zero": int(t)-1}
                    for i, t in enumerate(times))
    rows.sort(key=lambda row: row["original_event_sample_matlab"])
    if len({r["original_event_sample_matlab"] for r in rows}) != len(rows):
        raise ValueError("Duplicated original event positions across classes")
    return pd.DataFrame(rows).assign(chronological_trial_index=np.arange(len(rows)))


def cho_trial_order(eeg, sequence):
    """Validate official order against packed MAT events and signal boundaries.

    imagery_event is shared by both packed classes, not a second original
    continuous event record. This checks mapping consistency, not independent
    verification of the official continuous timestamps.
    """
    rows = cho_sequence_order(sequence)
    sfreq = float(eeg["srate"])
    frame = np.asarray(eeg["frame"], dtype=float).reshape(-1)
    if not np.isfinite(sfreq) or sfreq <= 0 or frame.shape != (2,) or not frame[0] < 0 < frame[1]:
        raise ValueError("Invalid Cho sampling rate/epoch frame")
    width_exact = (frame[1] - frame[0]) * sfreq / 1000
    if not np.isclose(width_exact, round(width_exact), rtol=0, atol=1e-7):
        raise ValueError("Epoch frame is not an integral number of samples")
    width = int(round(width_exact))
    events = np.asarray(eeg["imagery_event"]).reshape(-1)
    if not np.isin(events, (0, 1)).all():
        raise ValueError("Expected binary packed imagery_event")
    onsets = np.flatnonzero((events == 1) & np.r_[True, events[:-1] == 0])
    n_trials = len(onsets)
    if n_trials == 0 or len(events) != n_trials * width:
        raise ValueError("Packed event length/count does not match epoch frame")
    if int(eeg["n_imagery_trials"]) != n_trials:
        raise ValueError("n_imagery_trials disagrees with packed event count")
    if not np.array_equal(onsets // width, np.arange(n_trials)):
        raise ValueError("Expected exactly one cue in each packed trial")
    offsets = onsets % width
    # The released S01 MAT cue is at zero-based 1023, not 1024. Preserve it.
    expected = -frame[0] * sfreq / 1000
    if len(set(offsets.tolist())) != 1 or abs(offsets[0] - expected) > 1:
        raise ValueError("Packed cue is inconsistent with the published epoch frame")
    for side in ("left", "right"):
        signal = np.asarray(eeg[f"imagery_{side}"])
        if signal.ndim != 2 or signal.shape != (68, len(events)):
            raise ValueError("Expected 64 EEG + 4 EMG in the packed class signal")
        if (rows.label == f"{side}_hand").sum() != n_trials:
            raise ValueError("Official sequence count disagrees with packed class trials")
    indices = rows.class_local_trial_index.to_numpy()
    return rows.assign(packed_event_sample=onsets[indices], packed_epoch_start=indices * width,
                       packed_epoch_stop_exclusive=(indices + 1) * width,
                       event_offset_in_epoch=offsets[indices])


def read_cho_imagery(mat_file, sequence_file):
    """Return chronologically ordered full 7-second EEG epochs in volts.

    Epochs retain the release's packed boundaries and measured cue offsets.
    They are NOT a reconstructed continuous recording. No trial rejection is
    applied here; published bad_trial_indices remain available in metadata.
    """
    mat_file, sequence_file = Path(mat_file), Path(sequence_file)
    subject = re.fullmatch(r"s(\d+)\.mat", mat_file.name)
    order_subject = re.fullmatch(r"s(\d+)_trial_sequence_v1\.mat", sequence_file.name)
    if not subject or not order_subject or int(subject[1]) != int(order_subject[1]):
        raise ValueError("MAT and official sequence must identify the same subject")
    eeg = loadmat(mat_file, simplify_cells=True)["eeg"]
    sequence = loadmat(sequence_file, simplify_cells=True)["trial_sequence"]
    subject_field = re.fullmatch(r"subject\s+(\d+)", str(eeg["subject"]), flags=re.IGNORECASE)
    if not subject_field or int(subject_field[1]) != int(subject[1]):
        raise ValueError("MAT subject field disagrees with filename")
    order = cho_trial_order(eeg, sequence)
    epochs = np.stack([eeg["imagery_" + row.label.split("_")[0]][:64,
                       row.packed_epoch_start:row.packed_epoch_stop_exclusive]
                       for row in order.itertuples()]).astype(np.float64) * 1e-6
    if not np.isfinite(epochs).all():
        raise ValueError("Nonfinite imagery EEG")
    channels = mne.channels.make_standard_montage("biosemi64").ch_names
    return {"data": epochs, "order": order, "sfreq": float(eeg["srate"]), "channels": channels,
            "metadata": {"subject": int(subject[1]), "unit": "V", "source_file": str(mat_file),
                         "sequence_file": str(sequence_file), "continuous_recording": False,
                         "bad_trial_indices": eeg["bad_trial_indices"],
                         "independent_original_event_record_available": False}}


def read_cho_rest(mat_file):
    """Read the entire published eyes-open rest field, without a guessed crop.

    The paper's nominal minute and the released field length are distinct.
    Source samples refer to the rest field; no absolute session clock is given.
    Artefact screening remains a separate preprocessing step.
    """
    path = Path(mat_file)
    subject = re.fullmatch(r"s(\d+)\.mat", path.name)
    if not subject:
        raise ValueError("Expected a Cho sXX.mat filename")
    eeg = loadmat(path, simplify_cells=True)["eeg"]
    identity = re.fullmatch(r"subject\s+(\d+)", str(eeg["subject"]), flags=re.IGNORECASE)
    if not identity or int(identity[1]) != int(subject[1]):
        raise ValueError("MAT subject field disagrees with filename")
    signal, sfreq = np.asarray(eeg["rest"]), float(eeg["srate"])
    if (signal.ndim != 2 or signal.shape[0] != 68 or signal.shape[1] == 0
            or not np.isfinite(sfreq) or sfreq <= 0):
        raise ValueError("Invalid Cho rest schema or sampling rate")
    data = signal[:64].astype(np.float64) * 1e-6
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite rest EEG")
    return {"data": data, "sfreq": sfreq,
            "channels": mne.channels.make_standard_montage("biosemi64").ch_names,
            "metadata": {"subject": int(subject[1]), "unit": "V", "source_file": str(path),
                         "source_field": "eeg.rest", "eye_state": "eyes_open",
                         "nominal_duration_seconds": 60, "duration_seconds": data.shape[1]/sfreq,
                         "start_sample": 0, "stop_sample_exclusive": data.shape[1],
                         "boundary_basis": "published_rest_field",
                         "position_in_protocol": "before_movement_practice_and_imagery",
                         "absolute_session_timestamps_available": False,
                         "quality_screened": False}}
