"""Reuse the frozen trial roster; re-filter LaBraM from raw without changing trial IDs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cross_model import config_x
from .stage_a_common import ROOT, save_json, sha256
from .stage_a_preprocess import READERS, save_signal
from .stage_b import temporal_halves


# A-v2 restored all finite trials without rewriting the signal-generation hash.
# Both audited configurations have identical filters, resampling, units and channels.
_A_V1_SHA256 = "b42e8010b32fc8590dea9634cdc4d9180f6696e969ac6730d4d1755df2b88b62"
_A_V2_SHA256 = "47cbc35dd68020a7f87eab628fd557b8af647a13da2656a46b729fb179607fea"


def validate_reve_preprocessing(metadata):
    if sha256(ROOT / "configs/stage_a.yaml") != _A_V2_SHA256:
        raise ValueError("Audited A-v2 preprocessing configuration changed")
    if metadata.get("preprocessing_config_sha256") not in {_A_V1_SHA256, _A_V2_SHA256}:
        raise ValueError("REVE source has an unaudited preprocessing configuration")
    expected = {"sfreq": 200, "units": "uV/100", "dtype": "float16",
                "active_trial_policy": "all_finite_no_amplitude_or_balance_exclusion"}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("REVE source does not implement the frozen A-v2 trial/signal policy")


def timing_people():
    fold = json.loads((ROOT / "configs/splits/stage_a_v1.json").read_text())["folds"][0]
    datasets = list(config_x()["pretraining_overlap"]["REVE"])[:3]
    people = {role: [next((d, s) for d, s in fold[role] if d == dataset) for dataset in datasets]
              for role in ("train", "validation")}
    assert not ({tuple(p) for p in fold["test"]} & set(people["train"] + people["validation"]))
    return people


def starter_people():
    """One frozen roster, shared by preprocessing/cache jobs and all 25 runs."""
    records = json.loads((ROOT / "configs/splits/stage_b_halves_v1.json").read_text())["subjects"]
    return [(r["dataset"], r["subject"]) for r in records]


def original_contract(stage_a_root, dataset, subject):
    folder = Path(stage_a_root) / "processed" / dataset / f"sub-{subject:03d}"
    records = json.loads((ROOT / "configs/splits/stage_b_halves_v1.json").read_text())["subjects"]
    record = next(r for r in records if r["dataset"] == dataset and r["subject"] == subject)
    if sha256(folder / "trials.csv") != record["trial_table_sha256"]:
        raise ValueError("Original trial table changed")
    table = pd.read_csv(folder / "trials.csv")
    fit, query, counts = temporal_halves(table)
    if fit.tolist() != record["fit_indices"] or query.tolist() != record["query_indices"]:
        raise ValueError("Original time halves changed")
    labels = np.load(folder / "labels.npy", allow_pickle=False)
    np.testing.assert_array_equal(labels, table.label)
    metadata = json.loads((folder / "metadata.json").read_text())
    return folder, table, labels, metadata, fit, query, counts


def prepare_labram(stage_a_root, output_root, dataset, subject):
    folder, table, labels, metadata, *_ = original_contract(stage_a_root, dataset, subject)
    cfg = config_x()["preprocessing"]
    profile = {"sfreq": cfg["sfreq"], "highpass_hz": cfg["LaBraM"]["highpass_hz"],
               "lowpass_hz": cfg["LaBraM"]["lowpass_hz"], "notch_hz": cfg["LaBraM"]["notch_hz_all_datasets"]}
    out = Path(output_root) / "LaBraM/processed" / dataset / f"sub-{subject:03d}"
    if (out / "done.json").exists():
        receipt = json.loads((out / "done.json").read_text())
        if receipt["profile"] != profile or receipt["trial_table_sha256"] != sha256(folder / "trials.csv"):
            raise ValueError("Existing LaBraM cache has different preprocessing or trials")
        if sha256(out / "tasks.npy") != receipt["tasks_sha256"]:
            raise ValueError("Existing LaBraM signal cache changed")
        return receipt
    extra = {"write_order": False} if dataset == "Cho2017" else {}
    channels, _, _, tasks, rows, sources, _ = READERS[dataset](Path(stage_a_root) / "raw", subject, preprocessing=profile, **extra)
    if channels != metadata["channels"] or len(tasks) != len(table):
        raise ValueError("LaBraM reread changed channels/trial count")
    for name in ("label", "protocol_order", "start_seconds", "stop_seconds", "source_event_index", "class_local_trial_index"):
        if name in rows and name in table:
            np.testing.assert_allclose(rows[name], table[name], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(rows.label, labels)
    if not np.isfinite(tasks).all():
        raise ValueError("Re-filtering produced invalid values; cannot silently drop trials")
    # Assert the reread source files match the old raw-data provenance before caching.
    old_sources = {Path(r["path"]).name: r["sha256"] for r in metadata["source_files"]}
    for path in sources:
        if path.name not in old_sources or sha256(path) != old_sources[path.name]:
            raise ValueError(f"Raw source changed: {path.name}")
    out.mkdir(parents=True, exist_ok=True)
    save_signal(out / "tasks.npy", tasks)
    receipt = {"dataset": dataset, "subject": subject, "profile": profile, "channels": channels,
               "units": "uV/100", "trial_table_sha256": sha256(folder / "trials.csv"),
               "tasks_sha256": sha256(out / "tasks.npy"), "shape": list(tasks.shape),
               "from_raw": True, "test_results_used": False}
    save_json(out / "done.json", receipt)
    return receipt


def load_subject(stage_a_root, output_root, model, dataset, subject, role, device):
    folder, table, labels, metadata, fit, query, counts = original_contract(stage_a_root, dataset, subject)
    if model == "LaBraM":
        signal_dir = Path(output_root) / "LaBraM/processed" / dataset / f"sub-{subject:03d}"
        receipt = json.loads((signal_dir / "done.json").read_text())
        if receipt["trial_table_sha256"] != sha256(folder / "trials.csv") or sha256(signal_dir / "tasks.npy") != receipt["tasks_sha256"]:
            raise ValueError("LaBraM cache identity mismatch")
    elif model == "REVE":
        signal_dir = folder
        validate_reve_preprocessing(metadata)
    else:
        raise ValueError(model)
    common = json.loads((ROOT / "configs/stage_a_channels.json").read_text())["readout_common"]
    channels = metadata["channels"]
    signals = np.load(signal_dir / "tasks.npy", allow_pickle=False)
    if signals.shape[0] != len(table) or not np.isfinite(signals).all():
        raise ValueError("Invalid signal roster")
    return {"key": (dataset, subject), "role": role, "eeg": torch.as_tensor(signals, device=device),
            "picks": {"channels": channels, "readout": [channels.index(c) for c in common]},
            "labels": torch.as_tensor(labels, dtype=torch.long, device=device),
            "fit": torch.as_tensor(fit, device=device), "query": torch.as_tensor(query, device=device),
            "query_labels": labels[query], "trial_ids": table.trial_id.tolist(), "counts": counts}
