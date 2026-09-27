"""Reproduce the bounded rest-file checks from existing local evidence.

No network, training, task-label analysis, or quality-based subject selection.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mne
import numpy as np
from scipy.io import loadmat

from subject_context.real_data import read_cho_rest

OUT = ROOT / "reports/data_audit"
SOURCES = OUT / "followup_sources"
BNCI = ROOT / "data/external-test/BNCI2014_001"


def audit_bnci():
    """Compare unlabeled A01T MAT calibrations to their original GDF signals."""
    mat = BNCI / "MNE-bnci-data/~bci/database/001-2014/A01T.mat"
    blocks = loadmat(mat, simplify_cells=True)["data"]
    result = []
    with mne.io.read_raw_gdf(BNCI / "original_gdf/A01T.gdf", preload=False, verbose=False) as raw:
        sfreq = float(raw.info["sfreq"])
        starts = np.rint(raw.annotations.onset[raw.annotations.description == "32766"] * sfreq).astype(int)
        for i, (code, state) in enumerate([("276", "eyes_open"), ("277", "eyes_closed"), ("1072", "eye_movements")]):
            selected = np.flatnonzero(raw.annotations.description == code)
            assert len(selected) == 1
            a = selected[0]
            assert round(raw.annotations.onset[a] * sfreq) == starts[i]
            assert np.asarray(blocks[i]["y"]).size == np.asarray(blocks[i]["trial"]).size == 0
            data = np.asarray(blocks[i]["X"], dtype=np.float64).T * 1e-6
            assert data.shape[0] == 25 and blocks[i]["fs"] == sfreq == 250
            padding = starts[i+1] - starts[i] - data.shape[1]
            assert padding == (0 if i == 0 else 100)
            source_start = int(starts[i]+padding)
            original = raw.get_data(start=source_start, stop=int(starts[i+1]))
            # Exact comparison checks all 25 channels, including the three EOG.
            np.testing.assert_array_equal(data, original)
            if padding:
                separator = raw.get_data(start=int(starts[i]), stop=source_start)
                # GDF int16 missing-value sentinel becomes a finite physical
                # value in MNE; np.isfinite alone cannot remove this separator.
                np.testing.assert_array_equal(separator[:22], np.full((22, 100), -1600 * 1e-6))
            result.append({"file": "A01T", "mat_block_zero": i, "condition": state,
                           "original_gdf_code": int(code), "sfreq": sfreq,
                           "gdf_marker_onset_seconds": float(raw.annotations.onset[a]),
                           "gdf_annotation_duration_seconds": float(raw.annotations.duration[a]),
                           "gdf_signal_start_sample": source_start,
                           "gdf_signal_stop_exclusive": int(starts[i+1]),
                           "leading_separator_samples": int(padding),
                           "mat_samples": data.shape[1], "mat_duration_seconds": data.shape[1]/sfreq,
                           "all_25_channels_match_exactly": True, "quality_screened": False})
    with mne.io.read_raw_gdf(BNCI / "original_gdf/A04T.gdf", preload=False, verbose=False) as raw:
        assert not set(raw.annotations.description) & {"276", "277"}
        assert "1072" in raw.annotations.description
    return {"A01T_calibrations": result, "A04T_eyes_open_closed_absent": True,
            "scope": "A01T MAT/GDF calibration signals; A04T GDF annotation absence"}


def audit_yang_events():
    rows = []
    for session in (1, 2, 3):
        folder = SOURCES / "yang_s001" / f"ses-{session:02d}"
        # This is an annotation-only BDF. Reading it as raw EEG crops later
        # annotations to the dummy channel's 206 seconds and loses MI cues.
        events = mne.read_annotations(folder / "evt.bdf")
        header = (folder / "data.header").read_bytes()
        assert header[:8] == b"\xffBIOSEMI"
        nchan, nrecords = int(header[252:256]), int(header[236:244])
        record_seconds = float(header[244:252])
        assert len(header) == int(header[184:192]) == 256*(nchan+1)
        samples = [int(header[256+216*nchan+i*8:256+216*nchan+(i+1)*8]) for i in range(nchan)]
        duration = nrecords * record_seconds
        assert set(n / record_seconds for n in samples) == {1000}
        codes, counts = np.unique(events.description, return_counts=True)
        counts = dict(zip(codes.tolist(), counts.tolist()))
        assert counts == {"1": 100, "2": 100, "7": 1, "8": 1, "9": 4}
        assert np.all(np.diff(events.onset) >= 0) and np.all(events.duration == 0)
        assert events.onset[0] >= 0 and events.onset[-1] < duration
        rows.append({"subject": 1, "session": session, "data_duration_seconds": duration,
                     "signal_sfreq": 1000, "signal_channels_in_header": nchan,
                     "annotation_count": len(events), "annotation_counts": counts,
                     "code_7_seconds": float(events.onset[events.description == "7"][0]),
                     "code_8_seconds": float(events.onset[events.description == "8"][0]),
                     "first_mi_seconds": float(events.onset[np.isin(events.description, ["1", "2"])][0]),
                     "last_annotation_seconds": float(events.onset[-1]),
                     "rest_code_semantics_author_verified": False,
                     "rest_stop_markers_available": False})
    return rows


def main():
    rest = read_cho_rest(ROOT / "data/samples/Cho2017/s01.mat")
    summary = {"cho": {**rest["metadata"], "shape": list(rest["data"].shape), "sfreq": rest["sfreq"]},
               "bnci": audit_bnci(), "yang": audit_yang_events(),
               "lee": {"main_pretrain_rest": "60s eyes_open before MI; after ERP",
                       "source": "https://doi.org/10.3389/fnhum.2020.00321",
                       "other_three_rest_fields_eye_state": "not established by this source"},
               "stieger": {"independent_session_opening_rest_in_public_release": False,
                           "release": "10.6084/m9.figshare.13123148.v1",
                           "schema": "598 trial-based MAT files plus README; no separate rest field/file",
                           "scope": "Published manifest/schema and S1 sessions 1/2; not every signal file"}}
    (OUT / "rest_followup.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
