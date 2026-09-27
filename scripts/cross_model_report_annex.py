"""CPU report annexes and independent selection/log checks; never refit a model."""
import json
import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

model = sys.argv[1]
root = Path("reports/stage_x") / model
out = root / "report"
folds = json.loads(Path("configs/splits/stage_a_v1.json").read_text(encoding="utf-8"))["folds"]
halves = {(r["dataset"], r["subject"]): r for r in json.loads(
    Path("configs/splits/stage_b_halves_v1.json").read_text(encoding="utf-8"))["subjects"]}
curves, choices, personal_choices, subjects, few, resources = [], [], [], [], [], []
manifest = {r["path"]: r for r in json.loads((root / "retained_manifest.json").read_text(encoding="utf-8"))}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

config_hash = digest("configs/cross_model_x.yaml")
split_hash = digest("configs/splits/stage_a_v1.json")
halves_hash = digest("configs/splits/stage_b_halves_v1.json")
local_code_hashes = {}

def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

for folder in sorted((root / "runs").iterdir()):
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete"
    fold, seed = run["fold"], run["seed"]
    assert run["config_sha256"] == config_hash
    assert run["split_sha256"] == split_hash and run["halves_sha256"] == halves_hash
    assert run["backbone_unchanged"] and run["population_unchanged"]
    for name, value in run["code_sha256"].items():
        if name not in local_code_hashes:
            local_code_hashes[name] = digest(name)
        assert local_code_hashes[name] == value, name
    locked = json.loads((folder / "frozen_before_test.json").read_text(encoding="utf-8"))
    assert locked["config_sha256"] == config_hash and locked["test_seen"] is False
    for name, value in locked["files"].items():
        relative = (folder / name).relative_to(root).as_posix()
        assert manifest[relative]["sha256"] == value
        if not name.endswith(".pt"):
            assert digest(folder / name) == value
    baseline = pd.read_csv(folder / "diagnostic/consistency.csv")
    expected_test = {tuple(s) for s in folds[fold]["test"]}
    assert set(map(tuple, baseline[["dataset", "subject"]].values)) == expected_test
    assert len(baseline) == len(expected_test) == run["test_subjects"]
    assert (baseline.zero_personal_max_logit_error == 0).all()
    assert (baseline.zero_personal_prediction_agreement == 1).all()
    validation = {tuple(s) for s in folds[fold]["validation"]}
    train = {tuple(s) for s in folds[fold]["train"]}
    ntrain = len(train)
    b0 = json.loads((folder / "B0/selection.json").read_text(encoding="utf-8"))
    best = None
    best_ba = -math.inf
    for head in ("linear", "mlp"):
        for lr in (0.001, 0.0003):
            rows = records(folder / f"B0/{head}-lr-{lr:g}/validation.jsonl")
            assert [r["epoch"] for r in rows] == list(range(51))
            for row in rows[1:]:
                value = row["metrics"]["pooled"]["BA"]
                if value > best_ba:
                    best_ba, best = value, (head, lr, row["epoch"])
    assert best == (b0["kind"], b0["learning_rate"], b0["epoch"])

    initial = json.loads((folder / "G/initial_selection.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(folder / "G/initial_selection_candidates.csv", float_precision="round_trip")
    rank = candidates.groupby(["learning_rate", "episodes"]).gain.agg(["median", "mean"]).reset_index()
    winner = rank.sort_values(["median", "mean", "episodes", "learning_rate"],
                             ascending=[False, False, True, True], kind="stable").iloc[0]
    assert (winner.learning_rate, winner.episodes) == (initial["learning_rate"], initial["episodes"])
    convergence = json.loads((folder / "G/convergence.json").read_text(encoding="utf-8"))
    recorded_allocated = [run["peak_allocated_bytes"], convergence["peak_allocated_bytes"]]
    recorded_reserved = [run["peak_reserved_bytes"], convergence["peak_reserved_bytes"]]
    for path in (folder / "personal/validation_fit_resources.csv", folder / "diagnostic/fit_resources.csv"):
        recorded_allocated.extend(pd.read_csv(path).peak_gpu_bytes.tolist())
    for lr in (0.0003, 0.001):
        rows = records(folder / f"G/initial-lr-{lr:g}/validation.jsonl")
        assert set(range(0, 3001, ntrain)) | {1000, 3000} <= {r["step"] for r in rows}
    continuation = records(folder / "G/continuation/validation.jsonl")
    stop_step = convergence["additional_steps_run"]
    assert set(range(0, stop_step + 1, ntrain)) | {stop_step} <= {r["step"] for r in continuation}
    checks = [r for r in continuation if r["W_check"]]
    selected = min(checks, key=lambda r: r["metrics"]["pooled"]["CE"])
    assert selected["step"] == convergence["selected_additional_step"]
    assert abs(selected["metrics"]["pooled"]["CE"] - convergence["selected_validation_CE"]) < 1e-12
    reference, bad = math.inf, 0
    for row in checks:
        value = row["metrics"]["pooled"]["CE"]
        if value < reference - 1e-4:
            reference, bad = value, 0
        else:
            bad += 1
    assert bad == convergence["bad_checks"]
    if convergence["stopped_by"] == "CE_patience":
        assert bad >= 8

    search = pd.read_csv(folder / "personal/validation_search.csv", float_precision="round_trip")
    selections = json.loads((folder / "personal/selection.json").read_text(encoding="utf-8"))
    for n, group in search.groupby("n", dropna=False):
        key = "full" if pd.isna(n) else str(int(n))
        expected = {s for s in validation if key != "40" or s[0] != "PhysionetMI"}
        ranked = group.groupby(["learning_rate", "weight_decay", "steps"]).gain.agg(["median", "mean"]).reset_index()
        picked = ranked.sort_values(["median", "mean", "steps", "learning_rate", "weight_decay"],
                                    ascending=[False, False, True, True, True], kind="stable").iloc[0]
        saved = selections[key]
        assert tuple(picked[k] for k in ("learning_rate", "weight_decay", "steps")) == tuple(
            saved[k] for k in ("learning_rate", "weight_decay", "steps"))
        for _, rows in group.groupby(["learning_rate", "weight_decay", "steps"]):
            assert set(map(tuple, rows[["dataset", "subject"]].values)) == expected
            assert len(rows) == len(expected)
        personal_choices.append({"fold": fold, "seed": seed, "n": key, **saved})

    for path in sorted(folder.glob("B0/*/validation.jsonl")) + sorted(folder.glob("G/*/*.jsonl")):
        if path.name == "sampling.jsonl":
            continue
        split = "training_all" if path.name == "train_all_trials.jsonl" else "validation"
        for row in records(path):
            recorded_allocated.append(row.get("peak_allocated_bytes", 0))
            recorded_reserved.append(row.get("peak_reserved_bytes", 0))
            expected = train if split == "training_all" else validation
            assert {(p["dataset"], p["subject"]) for p in row["subjects"]} == expected
            for person in row["subjects"]:
                h = halves[person["dataset"], person["subject"]]
                indices = (list(range(len(h["fit_indices"]) + len(h["query_indices"])))
                           if row["phase"] == "B0" or split == "training_all" else h["query_indices"])
                assert person["indices"] == indices
            for dataset, metric in row["metrics"].items():
                assert all(np.isfinite(metric[k]) for k in ("CE", "accuracy", "BA"))
                curves.append({"fold": fold, "seed": seed, "file": str(path.relative_to(folder)),
                    "split": split, "dataset": dataset, **{k: row.get(k) for k in
                    ("phase", "head", "learning_rate", "epoch", "step", "global_step", "epoch_record",
                     "partial_epoch", "W_check", "initial_selection_candidate", "seconds")}, **metric})
    choices.append({"fold": fold, "seed": seed, "B0_head": b0["kind"], "B0_lr": b0["learning_rate"],
        "B0_epoch": b0["epoch"], "G_lr": initial["learning_rate"], "initial_selected_step": initial["episodes"],
        **convergence})
    resources.append({"fold": fold, "seed": seed, "run_seconds": run["seconds"],
        "final_window_peak_allocated_bytes": run["peak_allocated_bytes"],
        "final_window_peak_reserved_bytes": run["peak_reserved_bytes"],
        "peak_allocated_bytes_recorded_max": max(recorded_allocated),
        "peak_reserved_bytes_recorded_max": max(recorded_reserved)})
    frame = pd.read_csv(folder / "diagnostic/subjects.csv", float_precision="round_trip")
    base = pd.read_csv(folder / "diagnostic/consistency.csv", float_precision="round_trip")
    frame = frame.merge(base[["dataset", "subject", "B0_BA"]], on=["dataset", "subject"], validate="one_to_one")
    subjects.append(frame)
    few.append(pd.read_csv(folder / "diagnostic/few_shot.csv", float_precision="round_trip"))

assert len(resources) == 25
people, small = pd.concat(subjects, ignore_index=True), pd.concat(few, ignore_index=True)
people.to_csv(out / "subject_seed_details.csv", index=False)
small.to_csv(out / "few_shot_seed_details.csv", index=False)
pd.DataFrame(curves).to_csv(out / "epoch_curves.csv", index=False)
pd.DataFrame(choices).drop(columns="initial_selection").to_csv(out / "training_selection.csv", index=False)
pd.DataFrame(personal_choices).to_csv(out / "personal_selection.csv", index=False)
pd.DataFrame(resources).to_csv(out / "run_resources.csv", index=False)
people.groupby(["dataset", "seed"])[["B0_BA", "G_BA", "own_ba", "swap_ba", "own_gain", "upper_bound"]].mean().to_csv(out / "seed_level_core.csv")
seed_curve = small.groupby(["dataset", "shots", "seed"])[["ba", "gain"]].mean()
seed_curve.to_csv(out / "seed_level_curves.csv")
assert (small.groupby(["dataset", "subject", "shots"]).one_class_only.nunique() == 1).all()
prefix = small.drop_duplicates(["dataset", "subject", "shots"]).groupby(["dataset", "shots"]).one_class_only.agg(["sum", "count"])
prefix.columns = ["single_class_people", "people"]
prefix.to_csv(out / "single_class_prefixes.csv")
receipt = {"model": model, "runs": 25, "B0_all_epochs_and_selection": "pass",
    "config_split_halves_and_deployed_code_hashes": "pass",
    "frozen_before_test_selections_and_checkpoint_manifest": "pass",
    "test_rosters_frozen_parameters_zero_personal_exact": "pass",
    "G_epoch_coverage_initial_selection_min_CE_and_patience": "pass", "personal_validation_only_choices": "pass",
    "validation_and_training_trial_subsets": "pass", "curve_rows": len(curves),
    "subject_seed_rows": len(people), "few_shot_seed_rows": len(small),
    "parameter_refits": 0, "new_hyperparameter_selection": False}
(out / "log_and_selection_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
