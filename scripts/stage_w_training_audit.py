"""Read-only audit of completed W training receipts, curves and split membership."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("runs", type=Path)
p.add_argument("split", type=Path)
p.add_argument("output", type=Path)
args = p.parse_args()
folds = json.loads(args.split.read_text(encoding="utf-8"))["folds"]
directories = sorted(args.runs.glob("fold-*_seed-*"))
assert len(directories) == 25
seen, rows = set(), []
for directory in directories:
    receipt = json.loads((directory / "run.json").read_text())
    assert receipt["status"] == "complete"
    fold, seed = receipt["fold"], receipt["seed"]
    assert (fold, seed) not in seen
    seen.add((fold, seed))
    curves = pd.read_csv(directory / "curves.csv")
    losses = pd.read_csv(directory / "subject_losses.csv")
    traces = pd.read_csv(directory / "training_steps.csv")
    members = {r: {tuple(k) for k in folds[fold][r]} for r in ["train", "validation", "test"]}
    for method, status in receipt["families"].items():
        c = curves[curves.method == method]
        person = losses[losses.method == method]
        assert status["additional_steps_cap"] == 3 * status["original_steps"]
        assert c.additional_step.max() == status["additional_steps_run"] <= status["additional_steps_cap"]
        val = c[(c.role == "validation") & (c.dataset == "pooled")].sort_values("additional_step")
        assert val.additional_step.iloc[0] == 0
        chosen = val.loc[val.subject_mean_ce.idxmin()]
        assert int(chosen.additional_step) == status["selected_additional_step"]
        np.testing.assert_allclose(chosen.subject_mean_ce, status["selected_validation_ce"], atol=1e-12, rtol=0)
        assert status["test_used_for_selection"] is False
        trace = traces[traces.method == method]
        if "phase" in trace:
            trace = trace[trace.phase == "continuation"]
        assert trace.additional_step.tolist() == list(range(1, status["additional_steps_run"] + 1))
        assert set(zip(trace.dataset, trace.subject)) <= members["train"]
        for role in ["train", "validation"]:
            for step, g in person[person.role == role].groupby("additional_step"):
                keys = list(zip(g.dataset, g.subject))
                assert len(keys) == len(set(keys))
                assert set(keys) == members[role]
                assert not (set(keys) & members["test"])
                saved = c[(c.role == role) & (c.additional_step == step)].set_index("dataset").subject_mean_ce
                np.testing.assert_allclose(g.loss.mean(), saved.loc["pooled"], atol=1e-12, rtol=0)
                for dataset, d in g.groupby("dataset"):
                    np.testing.assert_allclose(d.loss.mean(), saved.loc[dataset], atol=1e-12, rtol=0)
        for (dataset, role), g in c.groupby(["dataset", "role"]):
            g = g.sort_values("additional_step")
            tail = g.subject_mean_ce.to_numpy()[-8:]
            flat = len(tail) == 8 and np.ptp(tail) <= max(1e-4, .01 * abs(tail.mean()))
            if dataset == "pooled":
                key = "training_curve_flat" if role == "train" else "validation_curve_flat"
                assert bool(flat) == status[key]
            rows.append({"fold": fold, "seed": seed, "method": method, "dataset": dataset,
                         "role": role, "curve_flat": bool(flat), "last_ce": g.subject_mean_ce.iloc[-1],
                         "initial_ce": g.subject_mean_ce.iloc[0], "tail_range": float(np.ptp(tail)),
                         "selected_additional_step": status["selected_additional_step"],
                         "additional_steps_run": status["additional_steps_run"],
                         "additional_steps_cap": status["additional_steps_cap"],
                         "early_stopped": status["early_stopped"]})
assert seen == {(f, s) for f in range(5) for s in [11, 23, 37, 53, 71]}
args.output.mkdir(parents=True, exist_ok=True)
result = pd.DataFrame(rows)
result.to_csv(args.output / "training_audit_by_dataset.csv", index=False)
pooled = result[(result.dataset == "pooled") & (result.role == "validation")]
summary = {}
for method, g in pooled.groupby("method"):
    summary[method] = {"runs": len(g), "selected_start": int((g.selected_additional_step == 0).sum()),
                       "early_stopped": int(g.early_stopped.sum()),
                       "validation_flat": int(g.curve_flat.sum()),
                       "additional_steps_min": int(g.additional_steps_run.min()),
                       "additional_steps_max": int(g.additional_steps_run.max())}
receipt = {"status": "complete", "runs": 25, "models": 50, "summary": summary,
           "checks": ["validation_loss_minimum", "per_model_3x_cap", "complete_fold_seed_grid",
                      "subject_equal_weight_curves", "no_test_subjects_in_loss_records",
                      "training_trace_uses_train_subjects_only", "saved_plateau_flags"]}
(args.output / "training_audit.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt))
