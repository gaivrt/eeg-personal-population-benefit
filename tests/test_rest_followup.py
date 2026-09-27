from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from subject_context.real_data import read_cho_rest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_rest_followup import audit_bnci, audit_yang_events


@pytest.mark.skipif(not (ROOT / "data/samples/Cho2017/s01.mat").exists(), reason="Cho source sample required")
def test_real_cho_rest_preserves_published_boundaries_and_units():
    path = ROOT / "data/samples/Cho2017/s01.mat"
    result = read_cho_rest(path)
    source = loadmat(path, simplify_cells=True)["eeg"]["rest"]
    np.testing.assert_array_equal(result["data"], source[:64].astype(np.float64)*1e-6)
    assert result["data"].shape == (64, 34048)
    assert result["metadata"]["duration_seconds"] == 66.5
    assert result["metadata"]["nominal_duration_seconds"] == 60
    assert result["metadata"]["absolute_session_timestamps_available"] is False
    assert result["metadata"]["quality_screened"] is False


@pytest.mark.parametrize("case", ["identity", "channels", "nonfinite", "sampling"])
def test_cho_rest_rejects_bad_source(tmp_path, case):
    eeg = {"subject": "subject 1", "rest": np.zeros((68, 512)), "srate": 512}
    if case == "identity": eeg["subject"] = "subject 2"
    if case == "channels": eeg["rest"] = np.zeros((64, 512))
    if case == "nonfinite": eeg["rest"][0, 0] = np.nan
    if case == "sampling": eeg["srate"] = 0
    path = tmp_path / "s01.mat"
    savemat(path, {"eeg": eeg})
    with pytest.raises(ValueError): read_cho_rest(path)


@pytest.mark.skipif(not (ROOT / "data/external-test/BNCI2014_001/original_gdf/A01T.gdf").exists(), reason="BNCI source samples required")
def test_real_bnci_calibration_roundtrip_and_missing_a04_rest():
    result = audit_bnci()
    rows = result["A01T_calibrations"]
    assert [r["mat_samples"] for r in rows] == [29683, 20172, 41463]
    assert [r["mat_duration_seconds"] for r in rows[:2]] == [118.732, 80.688]
    assert result["A04T_eyes_open_closed_absent"]


@pytest.mark.skipif(not (ROOT / "reports/data_audit/followup_sources/yang_s001/ses-01/evt.bdf").exists(), reason="Yang event files required")
def test_real_yang_event_only_bdf_keeps_late_annotations():
    rows = audit_yang_events()
    assert len(rows) == 3
    assert [r["code_7_seconds"] for r in rows] == [93.889, 35.241, 35.404]
    assert [r["code_8_seconds"] for r in rows] == [158.703, 100.289, 98.451]
    for row in rows:
        assert row["annotation_count"] == 206
        assert row["last_annotation_seconds"] > 2000
        assert row["rest_code_semantics_author_verified"] is False
