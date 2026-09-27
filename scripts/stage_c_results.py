"""Section-12 Stage C tables and the preregistered v1.2 success criteria."""
import argparse
import functools
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.stage_a_common import ROOT, config, save_json, scratch_root
from subject_context.stage_b import ba
from subject_context.stage_c import config_c
from subject_context.statistics import equivalent_trials, holm

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path)
args = parser.parse_args()
root = args.root if args.root else scratch_root()
cfg = config_c(); out = root / "stage_c/report"
out.mkdir(parents=True, exist_ok=True)
runs = sorted((root / "stage_c/runs").glob("fold-*_seed-*"))
assert len(runs) == 25
receipts = [json.loads((d / "run.json").read_text()) for d in runs]
assert all(r["status"] == "complete" and r["backbone_unchanged"] and r["base_head_unchanged"]
           and r["m_film_initial_equals_b0"] and r["m_lora_initial_equals_b0"] for r in receipts)
data = pd.concat([pd.read_csv(d / "subjects.csv") for d in runs], ignore_index=True)
assert not data.duplicated(["dataset", "subject", "seed", "condition"]).any()
assert (data[data.condition == "B0"].groupby(["dataset", "subject"]).size() == 5).all()
assert data[data.condition == "B0"].shape[0] == 235 * 5
b0 = data[data.condition == "B0"].set_index(["dataset", "subject", "seed"])
halves = {(r["dataset"], r["subject"]): r for r in json.loads((ROOT / cfg["halves_file"]).read_text())["subjects"]}


@functools.cache
def trial_ids(dataset, subject):
    return pd.read_csv(root / "processed" / dataset / f"sub-{subject:03d}" / "trials.csv").trial_id


def halves_query(dataset, subject):
    return halves[(dataset, subject)]["query_indices"]


checked = 0
for d in runs:  # every reported BA is recomputed from the trial-level predictions
    s = pd.read_csv(d / "subjects.csv").set_index(["dataset", "subject", "condition"])
    for (dataset, subject, condition), g in pd.read_csv(d / "predictions.csv").groupby(["dataset", "subject", "condition"]):
        r = s.loc[(dataset, subject, condition)]
        q = g[g["query"]]
        assert set(q.trial_id) == set(trial_ids(dataset, subject)[halves_query(dataset, subject)])
        assert len(q) == len(halves_query(dataset, subject)) and not g.trial_id.duplicated().any()
        assert abs(ba(q.label, q.prediction) - r.ba) < 1e-12
        if len(g) > len(q):
            assert abs(ba(g.label, g.prediction) - r.ba_all) < 1e-12
        checked += 1
assert checked == len(data)

# B6 from B2 and B1a/B1c from saved trial predictions, all against the same B0 second half.
b2 = pd.concat([pd.read_csv(d / "subjects.csv") for d in sorted((root / "stage_b2/runs").glob("fold-*_seed-*"))])
b2 = b2[b2.variant.isin(["film_first2", "film_all", "lora4", "lora8"])]
assert np.allclose(b2.b0_ba, b0.loc[list(zip(b2.dataset, b2.subject, b2.seed))].b0_ba, atol=1e-12)
extra = [b2.assign(condition="B6_" + b2.variant)[["dataset", "subject", "fold", "seed", "base_head", "condition",
                                                   "b0_ba", "ba", "gain"]]]
for name in ["B1a", "B1c"]:
    rows = []
    for (fold, seed, kind), g in b0.reset_index().groupby(["fold", "seed", "base_head"]):
        pred = pd.read_csv(root / "baselines_v2" / f"{name}_{kind}" / f"fold-{fold}_seed-{seed}" / "predictions.csv")
        for r in g.itertuples():
            p = pred.set_index("trial_id").loc[trial_ids(r.dataset, r.subject)]
            q = halves_query(r.dataset, r.subject)
            y, yhat = p.label.to_numpy(), p.prediction.to_numpy()
            rows.append({"dataset": r.dataset, "subject": r.subject, "fold": fold, "seed": seed, "base_head": kind,
                         "condition": name, "b0_ba": r.b0_ba, "ba": ba(y[q], yhat[q]), "gain": ba(y[q], yhat[q]) - r.b0_ba,
                         "b0_ba_all": r.b0_ba_all, "ba_all": ba(y, yhat), "gain_all": ba(y, yhat) - r.b0_ba_all})
    extra.append(pd.DataFrame(rows))
data = pd.concat([data, *extra], ignore_index=True)
data.to_csv(out / "subject_seed_results.csv", index=False)
metrics = ["b0_ba", "ba", "gain", "b0_ba_all", "ba_all", "gain_all"]
person = data.groupby(["condition", "dataset", "subject"], as_index=False)[metrics].mean()
person.to_csv(out / "subject_results.csv", index=False)
groups = [*config()["datasets"], "Dreyer+Cho", "pooled"]


def group(df, name):
    if name == "pooled":
        return df
    return df[df.dataset.isin(["Dreyer2023", "Cho2017"])] if name == "Dreyer+Cho" else df[df.dataset == name]


def stats(values, alternative="two-sided"):
    d = np.round(np.asarray(values, dtype=float), 12)
    test = wilcoxon(d, zero_method="wilcox", method="approx", alternative=alternative) if np.any(d) else None
    return {"n": len(d), "median_pp": 100*np.median(d), "mean_pp": 100*np.mean(d), "sd_pp": 100*np.std(d, ddof=1),
            "improved_fraction": float(np.mean(d > 0)), "drop_gt_2pp_fraction": float(np.mean(d < -.02)),
            "statistic": float(test.statistic) if test else 0., "p_raw": float(test.pvalue) if test else 1.}


def paired(a, b, column="ba"):
    x = person[person.condition == a].set_index(["dataset", "subject"])
    y = person[person.condition == b].set_index(["dataset", "subject"]).loc[x.index]
    return (x[column] - y[column]).reset_index(name="difference")


summary = []
for condition in sorted(person.condition.unique()):
    for name in groups:
        g = group(person[person.condition == condition], name)
        if g.empty:  # e.g. PhysioNet has no B5 n>=40
            continue
        row = {"condition": condition, "dataset": name, "b0_mean": g.b0_ba.mean(), "ba_mean": g.ba.mean(),
               "ba_sd": g.ba.std(), **stats(g.gain)}
        if g.ba_all.notna().all():
            row.update(ba_all_mean=g.ba_all.mean(), ba_all_sd=g.ba_all.std(),
                       **{k + "_all": v for k, v in stats(g.gain_all).items() if k != "n"})
        summary.append(row)
for row, corrected in zip(summary, holm([r["p_raw"] for r in summary])):
    row["p_holm_diagnostic"] = float(corrected)
summary = pd.DataFrame(summary); summary.to_csv(out / "summary.csv", index=False)

# Preregistered criteria (spec section 10, v1.2).
k = cfg["criteria"]; tests = []
for method in cfg["methods"]:
    m = "M_" + method[2:]
    for control in k["controls"]:
        d = paired(m, "B0" if control == "B0" else "B4_" + method[2:]).difference
        tests.append({"method": m, "control": control, **stats(d, "greater")})
for row, corrected in zip(tests, holm([r["p_raw"] for r in tests])):
    row["p_holm"] = float(corrected)
pd.DataFrame(tests).to_csv(out / "criteria_tests.csv", index=False)
criteria = []
for method in cfg["methods"]:
    m = "M_" + method[2:]
    gains = person[person.condition == m].gain
    upper = person[person.condition == "B6_" + cfg["b6_reference"][method]].gain.mean()
    significant = all(t["p_holm"] < k["alpha"] for t in tests if t["method"] == m)
    recovery = gains.mean() / upper
    drop = float(np.mean(np.round(gains, 12) < -.02))
    criteria.append({"method": m, "significant_vs_B4_and_B0": significant, "mean_gain_pp": 100*gains.mean(),
                     "b6_reference": cfg["b6_reference"][method], "b6_mean_gain_pp": 100*upper,
                     "recovery_fraction": recovery, "recovery_passed": bool(recovery >= k["recovery_fraction"]),
                     "drop_gt_2pp_fraction": drop, "drop_passed": drop <= k["max_drop_gt_2pp_fraction"],
                     "passed": bool(significant and recovery >= k["recovery_fraction"]
                                    and drop <= k["max_drop_gt_2pp_fraction"])})
save_json(out / "criteria.json", {"rule": "spec section 10 v1.2", "methods": criteria, "tests": tests,
                                  "stage_d_started": False})

# Diagnostics: real vs shuffled context and lambda=0, per dataset; equivalent labeled trials.
diagnostics = []
for method in ["film", "lora"]:
    for control in ["B4_" + method, "M_" + method + "_lambda0", "B3_" + method, "B2_T3A", "B1a", "B1c"]:
        d = paired("M_" + method, control)
        for name in groups:
            diagnostics.append({"method": "M_" + method, "versus": control, "dataset": name,
                                **stats(group(d, name).difference)})
for row, corrected in zip(diagnostics, holm([r["p_raw"] for r in diagnostics])):
    row["p_holm"] = float(corrected)
pd.DataFrame(diagnostics).to_csv(out / "diagnostic_tests.csv", index=False)
equivalent = []
shots = cfg["b5"]["shots"]
for name in config()["datasets"]:
    curve = [(n, group(person[person.condition == f"B5_n{n}"], name).ba.mean()) for n in shots]
    curve = [(n, a) for n, a in curve if np.isfinite(a)]
    for method in ["M_film", "M_lora"]:
        score = group(person[person.condition == method], name).ba.mean()
        equivalent.append({"dataset": name, "method": method, "ba_mean": score, "b5_curve": curve,
                           "equivalent_labeled_trials": equivalent_trials(score, *zip(*curve)) if curve else None})
save_json(out / "equivalent_trials.json", equivalent)
resources = pd.concat([pd.read_csv(d / "training_resources.csv") for d in runs], ignore_index=True)
resources.to_csv(out / "training_resources.csv", index=False)
save_json(out / "resources.json", {
    "job_seconds": [r["seconds"] for r in receipts], "total_job_seconds": sum(r["seconds"] for r in receipts),
    "training_seconds_by_method": resources.groupby("method").seconds.agg(["mean", "max", "sum"]).to_dict(),
    "peak_gpu_bytes": int(max(r["peak_gpu_bytes"] for r in receipts)),
    "trainable_parameters": resources.groupby("method").trainable_parameters.max().to_dict()})
save_json(out / "run_manifest.json", receipts)
print(json.dumps({"criteria": criteria}, indent=2))
