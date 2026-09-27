"""Stage A numerical and leakage contracts, independent of real test subjects."""
import numpy as np
import pytest

from subject_context.stage_a_common import participants
from subject_context.stage_a_preprocess import (
    PHYSIO_CHANNELS, alpha_comparison, canonical_channel, good_windows, segment_rest,
)
from subject_context.data import make_splits, assert_split
from subject_context.stage_a_common import failure_stop, save_json
from subject_context.stage_a_features import covariances, ea_references
import json


def test_fixed_cohort_and_dataset_stratification():
    counts = {d: len(participants(d)) for d in ["PhysionetMI", "Dreyer2023", "Cho2017"]}
    assert counts == {"PhysionetMI": 103, "Dreyer2023": 80, "Cho2017": 52}
    people = [(d, s) for d in counts for s in participants(d)]
    splits = make_splits(people, 5, .1, 20260925)
    seen = []
    for train, val, test in splits:
        assert_split(train, val, test)
        seen.extend(test)
        for d, n in counts.items():
            nt = sum(p[0] == d for p in test)
            nv = sum(p[0] == d for p in val)
            assert nt in (n // 5, n // 5 + 1)
            assert nv == round((n - nt) * .1)
    assert len(seen) == len(set(seen)) == 235
    assert set(seen) == set(people)


def test_leakage_deliberate_failures():
    train = [("PhysionetMI", 1)]
    val = [("Cho2017", 1)]
    test = [("Dreyer2023", 1)]
    assert_split(train, val, test)  # Numeric IDs from different datasets differ.
    for a, b, c in [(train, train, test), (train, val, train), (train, test, test)]:
        with pytest.raises(AssertionError, match="subject leakage"):
            assert_split(a, b, c)


def test_alpha_matched_windows_and_not_applicable():
    t = np.arange(800) / 200
    signal = np.stack([np.sin(2 * np.pi * 10 * t)] * 3)
    opened = np.stack([signal] * 4)
    closed = opened * 2
    valid = np.ones(4, dtype=bool)
    result = alpha_comparison(opened, closed, ["O1", "Oz", "O2"], valid, valid)
    assert result["status"] == "pass"
    assert result["closed_open_ratio"] == pytest.approx(4)
    result = alpha_comparison(closed, opened, ["O1", "Oz", "O2"], valid, valid)
    assert result["status"] == "fail"
    assert alpha_comparison(opened, None, ["O1", "Oz", "O2"], valid, None)["status"] == "not_applicable"
    assert alpha_comparison(opened, closed, ["C3", "Cz", "C4"], valid, valid)["status"] == "not_applicable"
    assert alpha_comparison(opened, closed, ["O1", "Oz", "O2"], ~valid, valid)["status"] == "fail"


def test_segmentation_never_pads_or_crosses_boundaries():
    raw = np.tile(np.arange(13300), (3, 1))  # 66.5 seconds at 200 Hz.
    segments = segment_rest(raw)
    assert segments.shape == (16, 3, 800)
    np.testing.assert_array_equal(segments[-1, :, -1], [12799] * 3)
    assert np.max(segments) < 12800


def test_channel_order_and_float16_quality():
    assert len(PHYSIO_CHANNELS) == len(set(PHYSIO_CHANNELS)) == 64
    assert [canonical_channel(c) for c in PHYSIO_CHANNELS[-4:]] == ["O1", "Oz", "O2", "Iz"]
    t = np.arange(800) / 200
    x = np.tile(np.sin(2 * np.pi * 10 * t), (3, 3, 1)).astype(np.float16)
    x[1] *= 100
    x[2, 0] = 0
    assert good_windows(x).tolist() == [True, True, True]
    x[2, 0, 0] = np.nan
    assert good_windows(x).tolist() == [True, True, False]


def test_cho_nested_mat_metadata_serialization(tmp_path):
    path = tmp_path / "metadata.json"
    save_json(path, {"bad_trial_indices": {"left": np.array([2, 5]),
                                          "right": np.array([], dtype=int)},
                     "count": np.int64(2)})
    assert json.loads(path.read_text()) == {"bad_trial_indices": {"left": [2, 5], "right": []}, "count": 2}


@pytest.mark.parametrize("cohort,last_allowed,first_stop", [(103, 10, 11), (80, 8, 9), (52, 5, 6)])
def test_stop_uses_full_cohort_and_strictly_greater_than_ten_percent(cohort, last_allowed, first_stop):
    assert failure_stop(last_allowed, cohort, .10) is False
    assert failure_stop(first_stop, cohort, .10) is True
    assert failure_stop(0, cohort, .10) is False


def test_ea_eyes_open_reference_does_not_use_task_or_closed_rest():
    rng = np.random.default_rng(12)
    signals = {k: rng.normal(size=(8, 4, 200)) for k in ["tasks", "open", "closed"]}
    before = ea_references(signals)["eo"]
    signals["tasks"] *= 100
    signals["closed"] *= .01
    np.testing.assert_array_equal(before, ea_references(signals)["eo"])
    reference = covariances(signals["open"]).mean(0)
    np.testing.assert_allclose(before @ reference @ before.T, np.eye(4), atol=1e-10)
