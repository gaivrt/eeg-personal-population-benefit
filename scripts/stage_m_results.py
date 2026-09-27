"""Stage M tables, strongest control (validation), the preregistered n=10 endpoint, secondary endpoints, label saving."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.stage_a_common import config, save_json, scratch_root
from subject_context.stage_m import config_m
from subject_context.statistics import equivalent_trials, holm

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path)
args = parser.parse_args()
root = args.root if args.root else scratch_root()
cfg = config_m(); out = root / "stage_m/report"
out.mkdir(parents=True, exist_ok=True)
runs = sorted((root / "stage_m/runs").glob("fold-*_seed-*"))
assert len(runs) == 25
receipts = [json.loads((d / "run.json").read_text()) for d in runs]
assert all(r["status"] == "complete" and r["backbone_unchanged"] and r["stage_c_model_unchanged"] for r in receipts)
methods, controls = cfg["variants"], list(cfg["controls"])
data = pd.concat([pd.read_csv(d / "results.csv") for d in runs], ignore_index=True)
valid = pd.concat([pd.read_csv(d / "validation_results.csv") for d in runs], ignore_index=True)
for m in methods + controls:
    for n in cfg["shots"]:
        want = 235 * 5 if n <= 20 else 132 * 5  # n=40: Dreyer 80 + Cho 52
        assert len(data[(data.method == m) & (data.shots == n)]) == want, (m, n)
data.to_csv(out / "subject_seed_results.csv", index=False)
keys = ["method", "shots", "dataset", "subject"]
person = data.groupby(keys, as_index=False)[["b3_ba", "ba", "gain"]].mean()
zero = person[person.shots == 0].set_index(["method", "dataset", "subject"]).ba
person["adapt_gain"] = person.ba - zero.loc[list(zip(person.method, person.dataset, person.subject))].to_numpy()
person.to_csv(out / "subject_results.csv", index=False)
groups = [*config()["datasets"], "Dreyer+Cho", "pooled"]


def group(df, name):
    if name == "pooled":
        return df
    return df[df.dataset.isin(["Dreyer2023", "Cho2017"])] if name == "Dreyer+Cho" else df[df.dataset == name]


def describe(d):
    d = np.round(np.asarray(d, dtype=float), 12)
    p = float(wilcoxon(d, zero_method="wilcox", method="approx", alternative="greater").pvalue) if np.any(d) else 1.
    return {"n": len(d), "median_pp": 100 * np.median(d), "mean_pp": 100 * d.mean(), "improved_fraction": float(np.mean(d > 0)),
            "drop_gt_2pp_fraction": float(np.mean(d < -.02)), "p_greater": p}


# Strongest control: pooled validation median BA at n=10 (each validation subject averaged over its five seeds).
vp = valid.groupby(keys, as_index=False).ba.mean()
v10 = vp[vp.shots == cfg["primary"]["shots"]]
ranking = v10[v10.method.isin(controls)].groupby("method").ba.agg(["median", "mean"]).sort_values(
    ["median", "mean"], ascending=False, kind="stable")
strongest = ranking.index[0]
v_methods = v10[v10.method.isin(methods)].groupby("method").ba.agg(["median", "mean"])

table = []
for (m, n), g in person.groupby(["method", "shots"]):
    for name in groups:
        h = group(g, name)
        if len(h):
            table.append({"method": m, "shots": n, "dataset": name, "n_subjects": len(h), "ba_mean": h.ba.mean(),
                          "ba_sd": h.ba.std(), "ba_median": h.ba.median(), "gain_vs_b3_median_pp": 100 * h.gain.median(),
                          "gain_vs_b3_mean_pp": 100 * h.gain.mean(), "adapt_gain_median_pp": 100 * h.adapt_gain.median(),
                          "adapt_gain_mean_pp": 100 * h.adapt_gain.mean()})
table = pd.DataFrame(table)
table.to_csv(out / "method_by_shots.csv", index=False)

wide = person.pivot_table(index=["shots", "dataset", "subject"], columns="method", values="ba").reset_index()
wide_adapt = person.pivot_table(index=["shots", "dataset", "subject"], columns="method", values="adapt_gain").reset_index()
comparisons = []
for n, g in wide.groupby("shots"):
    for m in methods:
        for ref in controls + ([m2 for m2 in methods if m2 != m] if m == "M2" else []):
            for name in groups:
                h = group(g, name)
                if len(h):
                    comparisons.append({"shots": n, "comparison": f"{m}-{ref}", "measure": "ba", "dataset": name,
                                        **describe(h[m] - h[ref])})
for n, g in wide_adapt[wide_adapt.shots > 0].groupby("shots"):
    for m in methods:
        for name in groups:
            h = group(g, name)
            if len(h):
                comparisons.append({"shots": n, "comparison": f"{m}-{strongest}", "measure": "adapt_gain", "dataset": name,
                                    **describe(h[m] - h[strongest])})
comparisons = pd.DataFrame(comparisons)
comparisons.to_csv(out / "comparisons.csv", index=False)


def vs_strongest(n, measure="ba"):
    rows = [comparisons[(comparisons.shots == n) & (comparisons.comparison == f"{m}-{strongest}") &
                        (comparisons.measure == measure) & (comparisons.dataset == "pooled")].iloc[0].to_dict() for m in methods]
    for r, p in zip(rows, holm([r["p_greater"] for r in rows])):
        r["p_holm"] = float(p)
    return dict(zip(methods, rows))


pr = cfg["primary"]
primary = vs_strongest(pr["shots"])
for m, r in primary.items():
    r["passed"] = bool(r["p_holm"] < pr["alpha"] and r["median_pp"] >= pr["min_median_pp"]
                       and r["drop_gt_2pp_fraction"] <= pr["max_drop_gt_2pp_fraction"])
passed = [m for m in methods if primary[m]["passed"]]
final = max(passed, key=lambda m: (v_methods.loc[m, "median"], v_methods.loc[m, "mean"])) if passed else None


def saving(dataset):
    h = group(person, dataset)
    curve = h[h.method == strongest].groupby("shots").ba.mean()
    if dataset in ("pooled", "PhysionetMI"):
        curve = curve[curve.index <= 20]
    result = {"strongest_curve": {int(k): float(v) for k, v in curve.items()}}
    for m in methods:
        target = float(h[(h.method == m) & (h.shots == pr["shots"])].ba.mean())
        n_star = equivalent_trials(target, curve.index.to_numpy(), curve.to_numpy())
        result[m] = {"ba_at_n10": target, "equivalent_labels": n_star,
                     "note": "interpolated" if n_star is not None else
                     "above the strongest control at the largest observed n" if target > curve.max() else
                     "below the strongest control at n=0" if target < curve.iloc[0] else "curve not monotone"}
    return result


selections = [json.loads((d / "selection.json").read_text()) for d in runs]
chosen = pd.DataFrame([{"run": d.name, "variant": m, "k": s[m]["k"], "steps": s[m]["steps"]}
                       for d, s in zip(runs, selections) for m in methods])
chosen.groupby(["variant", "k", "steps"]).size().rename("count").reset_index().to_csv(out / "m_selection_distribution.csv", index=False)
save_json(out / "endpoints.json", {
    "rule": "spec Stage M v1.15: n=10, M vs strongest control; Holm over M1/M2 p<.05, median >= 1 pp, drop>2pp <= 10%",
    "strongest_control": strongest, "validation_ranking_n10": ranking.reset_index().to_dict("records"),
    "validation_m_n10": v_methods.reset_index().to_dict("records"),
    "primary": primary, "passed_variants": passed, "primary_passed": bool(passed), "final_variant": final,
    "secondary": {int(n): vs_strongest(n) for n in cfg["secondary_shots"]},
    "secondary_adapt_gain_n10": vs_strongest(pr["shots"], "adapt_gain"),
    "label_saving": {name: saving(name) for name in groups if name != "Dreyer+Cho"}})
training = pd.concat([pd.read_csv(d / "training_resources.csv").assign(run=d.name) for d in runs], ignore_index=True)
training.to_csv(out / "training_resources.csv", index=False)
save_json(out / "resources.json", {
    "run_seconds": [r["seconds"] for r in receipts], "peak_gpu_bytes": int(max(r["peak_gpu_bytes"] for r in receipts)),
    "outer_training_seconds": training.groupby(["method", "k"]).seconds.agg(["mean", "max", "sum"]).reset_index().to_dict("records"),
    "outer_peak_gpu_bytes": training.groupby("method").peak_gpu_bytes.max().to_dict(),
    "b3_vs_c4_mismatches": int(sum(r["b3_vs_c4_mismatches"] for r in receipts)),
    "b3_vs_c4_max_abs_diff": float(max(r["b3_vs_c4_max_abs_diff"] for r in receipts))})
save_json(out / "run_manifest.json", receipts)
print(json.dumps({"strongest": strongest, "primary": primary, "passed": passed, "final": final}, indent=2, default=float))
