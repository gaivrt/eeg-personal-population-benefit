from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat

from subject_context.edf_header import parse_edf_header
from subject_context.real_data import (
    physionet_label, read_physionet_mi_events, dreyer_rest_interval,
    read_dreyer_rest, cho_sequence_order, cho_trial_order, read_cho_imagery,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data/samples"
SEQUENCES = ROOT / "docs/data_access_audit_2026-09-25/sources/cho_trial_sequence"


@pytest.mark.parametrize("run", [4, 8, 12])
def test_physionet_left_right_and_no_rest_query(run):
    assert physionet_label(run, "T1") == "left_hand"
    assert physionet_label(run, "T2") == "right_hand"
    assert physionet_label(run, "T0") is None
    with pytest.raises(ValueError, match="Unknown"):
        physionet_label(run, "left")


@pytest.mark.parametrize("run", [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, True, 4.0, "4"])
def test_physionet_rejects_nonquery_and_renumbered_runs(run):
    with pytest.raises(ValueError, match="original EDF run"):
        physionet_label(run, "T1")


def test_physionet_reader_checks_source_run_before_io():
    for name in ["S001R03.edf", "S001R06.edf", "S001R01.edf", "0.edf", "S110R04.edf"]:
        with pytest.raises(ValueError):
            read_physionet_mi_events(name)


def rest_events():
    # The `value` column deliberately disagrees with original GDF codes.
    return pd.DataFrame({"trial_type": [32775, 786, 32776], "value": [3, 6, 4],
                         "sample": [2560, 2560, 94720], "onset": [5., 5., 185.]})


def test_dreyer_uses_original_codes_and_exclusive_end():
    assert dreyer_rest_interval(rest_events(), 512, 188 * 512) == (2560, 94720)
    assert (94720 - 2560) / 512 == 180


@pytest.mark.parametrize("case", ["missing", "duplicate", "reversed", "outside", "onset", "mi", "fractional"])
def test_dreyer_refuses_ambiguous_boundaries(case):
    events = rest_events()
    if case == "missing": events = events.iloc[:2]
    if case == "duplicate": events = pd.concat([events, events.iloc[:1]])
    if case == "reversed": events.loc[[0, 2], "trial_type"] = [32776, 32775]
    if case == "outside": events.loc[2, ["sample", "onset"]] = [512*190, 190]
    if case == "onset": events.loc[0, "onset"] = 4
    if case == "mi": events.loc[1, "trial_type"] = 769
    if case == "fractional": events = events.astype({"sample": float}); events.loc[0, "sample"] = 2560.5
    with pytest.raises(ValueError):
        dreyer_rest_interval(events, 512, 188*512)


def cho_fixture():
    # Two 7-s trials per class; each has a distinct signal signature.
    eeg = {"srate": 1, "frame": [-2000, 5000], "n_imagery_trials": 2,
           "imagery_event": np.array([0, 1, 0, 0, 0, 0, 0] * 2),
           "imagery_left": np.tile(np.repeat([10, 11], 7), (68, 1)),
           "imagery_right": np.tile(np.repeat([20, 21], 7), (68, 1))}
    return eeg, {"imagery_left": np.array([20, 100]), "imagery_right": np.array([5, 60])}


def test_cho_order_maps_back_to_class_events_and_signal():
    eeg, sequence = cho_fixture()
    order = cho_trial_order(eeg, sequence)
    assert order.original_event_sample_matlab.tolist() == [5, 20, 60, 100]
    assert order.original_event_sample_zero.tolist() == [4, 19, 59, 99]
    assert order.label.tolist() == ["right_hand", "left_hand", "right_hand", "left_hand"]
    assert order.packed_event_sample.tolist() == [1, 1, 8, 8]
    observed = [eeg["imagery_" + r.label.split("_")[0]][0, r.packed_epoch_start]
                for r in order.itertuples()]
    assert observed == [20, 10, 21, 11]


@pytest.mark.parametrize("case", ["duplicate", "noninteger", "reversed", "missing_sequence", "missing_cue",
                                       "shifted_cue", "signal_length", "trial_count", "frame"])
def test_cho_refuses_inconsistent_order_or_events(case):
    eeg, sequence = cho_fixture()
    if case == "duplicate": sequence["imagery_right"][0] = 20
    if case == "noninteger": sequence["imagery_left"] = [20.5, 100]
    if case == "reversed": sequence["imagery_left"] = [100, 20]
    if case == "missing_sequence": sequence["imagery_left"] = [20]
    if case == "missing_cue": eeg["imagery_event"][8] = 0
    if case == "shifted_cue": eeg["imagery_event"][8:11] = [0, 0, 1]
    if case == "signal_length": eeg["imagery_right"] = eeg["imagery_right"][:, :-1]
    if case == "trial_count": eeg["n_imagery_trials"] = 3
    if case == "frame": eeg["frame"] = [-2000, 4500]
    with pytest.raises(ValueError):
        cho_trial_order(eeg, sequence)


def test_cho_refuses_sequence_from_different_subject_before_io():
    with pytest.raises(ValueError, match="same subject"):
        read_cho_imagery("s01.mat", "s2_trial_sequence_v1.mat")


def make_edf_header():
    header = bytearray(b" " * 1024)
    for start, end, value in [(0, 8, "0"), (184, 192, "1024"), (236, 244, "10"),
                               (244, 252, "0.5"), (252, 256, "3")]:
        header[start:end] = value.encode().ljust(end-start)
    for i, (name, samples) in enumerate([("C3", 64), ("C4", 64), ("EDF Annotations", 40)]):
        header[256+16*i:256+16*(i+1)] = name.encode().ljust(16)
        start = 256+216*3+8*i
        header[start:start+8] = str(samples).encode().ljust(8)
    return header


def test_edf_header_fractional_records_and_annotation_channel():
    parsed = parse_edf_header(make_edf_header())
    assert parsed["sfreqs"] == [128]
    assert parsed["sample_counts"] == [640]
    assert parsed["duration_seconds"] == 5
    assert parsed["eeg_labels"] == ["C3", "C4"]
    assert parsed["expected_file_bytes"] == 1024+2*10*(64+64+40)


def test_edf_header_rejects_unknown_duration_or_truncation():
    header = make_edf_header()
    with pytest.raises(ValueError): parse_edf_header(header[:-1])
    header[236:244] = b"-1      "
    with pytest.raises(ValueError): parse_edf_header(header)


@pytest.mark.skipif(not (SAMPLES / "PhysionetMI/S001R04.edf").exists(), reason="Local MOABB sample required")
def test_real_physionet_events_and_header_match_mne():
    results = []
    for run in [4, 8, 12]:
        path = SAMPLES / f"PhysionetMI/S001R{run:02d}.edf"
        result = read_physionet_mi_events(path)
        with mne.io.read_raw_edf(path, preload=False, verbose=False) as raw:
            with path.open("rb") as f:
                first = f.read(256)
                parsed = parse_edf_header(first + f.read(int(first[184:192])-256))
            assert parsed["sfreqs"] == [raw.info["sfreq"]] == [160]
            assert parsed["duration_seconds"] == raw.n_times/raw.info["sfreq"] == 125
            assert parsed["expected_file_bytes"] == path.stat().st_size
            for row in result.itertuples():
                assert row.annotation == raw.annotations.description[row.source_event_index]
                assert row.onset_seconds == raw.annotations.onset[row.source_event_index]
                assert row.original_run == run
        results.append(result)
    assert pd.concat(results).label.value_counts().to_dict() == {"left_hand": 23, "right_hand": 22}


@pytest.mark.skipif(not (SAMPLES / "Dreyer2023/sub-01").exists(), reason="Local MOABB sample required")
def test_real_dreyer_baselines_cut_padding_and_auxiliary_channels():
    for state in ["eyes_open", "eyes_closed"]:
        result = read_dreyer_rest(SAMPLES / "Dreyer2023", 1, state)
        assert result["data"].shape == (27, 92160)
        assert result["metadata"]["duration_seconds"] == 180
        assert result["metadata"]["edf_duration_seconds"] == 188
        with mne.io.read_raw_edf(result["metadata"]["source_file"], preload=False, verbose=False) as raw:
            np.testing.assert_array_equal(result["data"], raw.get_data(
                picks=result["channels"], start=2560, stop=94720))
        assert not any(c.startswith(("EOG", "EMG")) for c in result["channels"])


@pytest.mark.skipif(not (SAMPLES / "Cho2017/s01.mat").exists(), reason="Local MOABB sample required")
def test_real_cho_official_order_events_and_epoch_roundtrip():
    result = read_cho_imagery(SAMPLES / "Cho2017/s01.mat", SEQUENCES / "s1_trial_sequence_v1.mat")
    order = result["order"]
    assert result["data"].shape == (200, 64, 3584)
    assert order.original_event_sample_matlab.head(6).tolist() == [1537, 6017, 9601, 13953, 18177, 22401]
    assert order.label.head(6).tolist() == ["left_hand", "right_hand", "right_hand", "left_hand", "right_hand", "right_hand"]
    assert set(order.event_offset_in_epoch) == {1023}  # Measured cue, no one-sample shift.
    eeg = loadmat(SAMPLES / "Cho2017/s01.mat", simplify_cells=True)["eeg"]
    for row, epoch in zip(order.itertuples(), result["data"]):
        assert eeg["imagery_event"][row.packed_event_sample] == 1
        expected = eeg["imagery_" + row.label.split("_")[0]][:64,
                       row.packed_epoch_start:row.packed_epoch_stop_exclusive].astype(np.float64) * 1e-6
        np.testing.assert_array_equal(epoch, expected)
    assert order.original_event_sample_matlab.is_monotonic_increasing
    assert result["metadata"]["independent_original_event_record_available"] is False


@pytest.mark.skipif(not SEQUENCES.exists(), reason="Official sequence archive required")
def test_all_52_official_sequences_have_unique_order_and_complete_class_ids():
    paths = list(SEQUENCES.glob("s*_trial_sequence_v1.mat"))
    assert len(paths) == 52
    for path in paths:
        order = cho_sequence_order(loadmat(path, simplify_cells=True)["trial_sequence"])
        assert len(order) in (200, 240)
        for _, group in order.groupby("label"):
            assert group.class_local_trial_index.tolist() == list(range(len(group)))
