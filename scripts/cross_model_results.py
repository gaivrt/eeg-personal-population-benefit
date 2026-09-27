"""Audit one completed model without waiting for the other; retain partial-run status."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.cross_model import config_x
from subject_context.cross_model_data import starter_people
from subject_context.stage_a_common import ROOT, save_json, sha256
from subject_context.statistics import holm


def distribution(values):
    # Same one-sided normal approximation, 12-digit differences and zero handling as W.
    x = np.round(np.asarray(values), 12)
    p = float(wilcoxon(x, alternative="greater", method="approx", zero_method="wilcox").pvalue) if np.any(x) else 1.
    return {"N": len(x), "mean_pp": float(100*x.mean()), "median_pp": float(100*np.median(x)),
            "SD_pp": float(100*x.std(ddof=1)), "improved_fraction": float((x > 0).mean()),
            "drop_gt_2pp_fraction": float((x < -.02).mean()), "p_raw_descriptive": p}


def ba(group):
    if set(group.label) != {0, 1}:
        raise ValueError("Incomplete query predictions")
    return np.mean([(group.prediction[group.label == c] == c).mean() for c in (0, 1)])


def audit_run(directory, stage_a_root):
    locked = json.loads((directory / "frozen_before_test.json").read_text())
    for path, digest in locked["files"].items():
        if sha256(directory / path) != digest:
            raise ValueError("Selection artifact changed after test evaluation")
    prediction = pd.read_csv(directory / "diagnostic/predictions.csv.gz")
    keys = ["dataset", "subject"]
    halves = {(r["dataset"], r["subject"]): r for r in json.loads((ROOT / config_x()["halves_file"]).read_text())["subjects"]}
    for key, rows in prediction.groupby(keys):
        table = pd.read_csv(stage_a_root / "processed" / key[0] / f"sub-{key[1]:03d}" / "trials.csv")
        expected = set(table.iloc[halves[key]["query_indices"]].trial_id)
        truth = table.set_index("trial_id").label
        np.testing.assert_array_equal(rows.label, truth.loc[rows.trial_id])
        if not rows.prediction.isin([0, 1]).all():
            raise ValueError("Nonbinary prediction")
        for _, subgroup in rows.groupby(["condition", "n", "donor"], dropna=False):
            if set(subgroup.trial_id) != expected or subgroup.trial_id.duplicated().any():
                raise ValueError("Incomplete or duplicated query roster")
    absolute = prediction[prediction.condition.isin(["B0", "G", "own"])].groupby(keys+["condition"]).apply(ba, include_groups=False).unstack("condition")
    subjects = pd.read_csv(directory / "diagnostic/subjects.csv").set_index(keys)
    consistency = pd.read_csv(directory / "diagnostic/consistency.csv").set_index(keys)
    for a, b in [(absolute.G, subjects.G_BA), (absolute.own, subjects.own_ba), (absolute.B0, consistency.B0_BA)]:
        np.testing.assert_allclose(a.loc[b.index], b, rtol=0, atol=1e-12)
    matrix = prediction[prediction.condition == "swap_matrix"].groupby(keys+["donor"]).apply(ba, include_groups=False)
    saved = pd.read_csv(directory / "diagnostic/swap_matrix.csv").set_index(keys+["donor"]).BA
    np.testing.assert_allclose(matrix.loc[saved.index], saved, rtol=0, atol=1e-12)
    for row in json.loads((directory / "diagnostic/swap_donors.json").read_text()):
        target = (row["dataset"], row["subject"])
        expected = np.mean([matrix.loc[(*target, donor)] for donor in row["donors"]])
        np.testing.assert_allclose(expected, subjects.loc[target].swap_ba, rtol=0, atol=1e-12)
    few = pd.read_csv(directory / "diagnostic/few_shot.csv").set_index(keys+["shots"])
    observed = prediction[prediction.condition == "few_shot"].groupby(keys+["n"]).apply(ba, include_groups=False)
    np.testing.assert_allclose(observed.loc[few.index], few.ba, rtol=0, atol=1e-12)
    subjects["B0_BA"] = absolute.B0
    return subjects.reset_index(), few.reset_index(), len(prediction)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["REVE", "LaBraM"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--stage-a-root", required=True, type=Path)
    args = p.parse_args()
    cfg = config_x()
    runs = args.out / args.model / "runs"
    report = args.out / args.model / "report"
    report.mkdir(parents=True, exist_ok=True)
    entries, ready = [], []
    for fold in range(5):
        for seed in cfg["seeds"]:
            folder = runs / f"fold-{fold}_seed-{seed}"
            receipt = json.loads((folder / "run.json").read_text()) if (folder / "run.json").exists() else {"status":"not_started"}
            entries.append({"fold":fold, "seed":seed, "status":receipt["status"]})
            if receipt["status"] == "complete":
                if receipt["config_sha256"] != sha256(ROOT / "configs/cross_model_x.yaml"):
                    raise ValueError("Cannot combine runs with changed configuration")
                ready.append(folder)
    save_json(report / "progress.json", {"model":args.model, "complete_runs":len(ready), "expected_runs":25, "runs":entries})
    if len(ready) != 25:
        print(f"{args.model}: {len(ready)}/25 complete; no incomplete-cohort scientific summary")
        return
    audited = [audit_run(folder, args.stage_a_root) for folder in ready]
    people, few = (pd.concat([r[i] for r in audited], ignore_index=True) for i in (0,1))
    if people.duplicated(["dataset","subject","seed"]).any() or set(map(tuple, people[["dataset","subject"]].values)) != set(starter_people()):
        raise ValueError("Wrong pooled subject roster")
    if not (people.groupby(["dataset","subject"]).seed.nunique() == 5).all():
        raise ValueError("Five independent seed results required per subject")
    expected_few = {(d,s,n,seed) for d,s in starter_people() for seed in cfg["seeds"]
                    for n in [*cfg["personal_and_few_shot"]["shots"],
                              *([40] if d in cfg["personal_and_few_shot"]["additional_40"] else [])]}
    keys_few = ["dataset","subject","shots","seed"]
    if few.duplicated(keys_few).any() or set(map(tuple,few[keys_few].values)) != expected_few:
        raise ValueError("Incomplete few-shot cohort or seed roster")
    numeric = ["B0_BA","G_BA","own_ba","swap_ba","own_gain","upper_bound"]
    means = people.groupby(["dataset","subject"])[numeric].mean()
    curves = few.groupby(["dataset","subject","shots"])[["ba","G_BA","gain","full_gain"]].mean()
    rows, curve_rows = [], []
    for dataset in sorted(means.index.get_level_values(0).unique()):
        group = means.loc[dataset]
        common = {"model":args.model,"dataset":dataset,"pretraining_source":cfg["pretraining_overlap"][args.model][dataset]}
        row = dict(common)
        for name in numeric[:4]:
            row[name+"_mean_percent"] = 100*group[name].mean()
            row[name+"_SD_percent"] = 100*group[name].std(ddof=1)
        for name, values in [("G_minus_B0",group.G_BA-group.B0_BA),("own_minus_G",group.own_gain),("own_minus_swap",group.upper_bound)]:
            row.update({name+"_"+k:v for k,v in distribution(values).items()})
        rows.append(row)
        for shots, small in curves.loc[dataset].groupby(level="shots"):
            denominator = float(small.full_gain.mean())
            curve_rows.append({**common,"shots":int(shots),**distribution(small.gain),
                               "full_gain_mean":denominator,"recovery":float(small.gain.mean()/denominator) if denominator > 0 else None})
    pooled = distribution(means.upper_bound)
    corrected = float(holm([1.,1.,pooled["p_raw_descriptive"]])[2])
    gate = {"model":args.model,"scope":"pooled_235_original_three_slots","unrun_slots":["FiLM","mixture"],
            **pooled,"p_Holm":corrected,"pass":pooled["median_pp"] >= 2 and corrected < .05,
            "audited_prediction_rows":sum(r[2] for r in audited)}
    means.to_csv(report / "subject_means.csv")
    curves.to_csv(report / "few_shot_subject_means.csv")
    pd.DataFrame(rows).to_csv(report / "core_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(report / "few_shot_summary.csv", index=False)
    save_json(report / "diagnostic1_gate.json", gate)
    print(json.dumps(gate))


if __name__ == "__main__":
    main()
