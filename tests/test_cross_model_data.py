"""Historical signal provenance must preserve the frozen A-v2 trial policy."""
import pytest

from subject_context.cross_model_data import validate_reve_preprocessing


@pytest.mark.parametrize("signal_hash", [
    "b42e8010b32fc8590dea9634cdc4d9180f6696e969ac6730d4d1755df2b88b62",
    "47cbc35dd68020a7f87eab628fd557b8af647a13da2656a46b729fb179607fea",
])
def test_original_signal_versions_with_restored_finite_trial_policy(signal_hash):
    validate_reve_preprocessing({"preprocessing_config_sha256": signal_hash,
        "sfreq": 200, "units": "uV/100", "dtype": "float16",
        "active_trial_policy": "all_finite_no_amplitude_or_balance_exclusion"})


@pytest.mark.parametrize("field,value", [
    ("preprocessing_config_sha256", "unknown"), ("sfreq", 250),
    ("units", "V"), ("active_trial_policy", "amplitude_exclusion"),
])
def test_reuse_rejects_changed_signal_or_trial_policy(field, value):
    metadata = {"preprocessing_config_sha256":
        "b42e8010b32fc8590dea9634cdc4d9180f6696e969ac6730d4d1755df2b88b62",
        "sfreq": 200, "units": "uV/100", "dtype": "float16",
        "active_trial_policy": "all_finite_no_amplitude_or_balance_exclusion"}
    metadata[field] = value
    with pytest.raises(ValueError):
        validate_reve_preprocessing(metadata)
