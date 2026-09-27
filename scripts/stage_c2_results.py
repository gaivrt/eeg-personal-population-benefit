"""Stage C2 tables, the diagnostic-1 gate and (only if it passes) the diagnostic-2 nearest-neighbour test."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.stage_a_common import config, save_json, scratch_root
from subject_context.stage_c2 import config_c2
from subject_context.statistics import holm

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path)
args = parser.parse_args()
root = args.root if args.root else scratch_root()
cfg = config_c2(); out = root / "stage_c2/report"
out.mkdir(parents=True, exist_ok=True)
runs = sorted((root / "stage_c2/runs").glob("fold-*_seed-*"))
assert len(runs) == 25
receipts = [json.loads((d / "run.json").read_text()) for d in runs]
assert all(r["status"] == "complete" and r["b3_reproduces_stage_c"] and r["backbone_unchanged"]
           and r["stage_c_models_unchanged"] for r in receipts)
data = pd.concat([pd.read_csv(d / "subjects.csv") for d in runs], ignore_index=True)
assert len(data) == 235 * 5 * len(cfg["variants"])
assert not data.duplicated(["variant", "dataset", "subject", "seed"]).any()
data.to_csv(out / "subject_seed_results.csv", index=False)
cols = ["b3_ba", "own_ba", "swap_ba", "upper_bound", "own_gain", "swap_gain"]
person = data.groupby(["variant", "dataset", "subject"], as_index=False)[cols].mean()
person.to_csv(out / "subject_results.csv", index=False)
groups = [*config()["datasets"], "Dreyer+Cho", "pooled"]


def group(df, name):
    if name == "pooled":
        return df
    return df[df.dataset.isin(["Dreyer2023", "Cho2017"])] if name == "Dreyer+Cho" else df[df.dataset == name]


def test(values, alternative):
    d = np.round(np.asarray(values, dtype=float), 12)
    r = wilcoxon(d, zero_method="wilcox", method="approx", alternative=alternative) if np.any(d) else None
    return (float(r.statistic), float(r.pvalue)) if r else (0., 1.)


summary = []
for variant in cfg["variants"]:
    for name in groups:
        g = group(person[person.variant == variant], name)
        u = np.round(g.upper_bound, 12)
        stat, p = test(u, "two-sided")
        summary.append({"variant": variant, "dataset": name, "n": len(g),
            **{f"{k}_mean": g[k].mean() for k in ["b3_ba", "own_ba", "swap_ba"]},
            **{f"{k}_sd": g[k].std() for k in ["b3_ba", "own_ba", "swap_ba"]},
            "upper_median_pp": 100*np.median(u), "upper_mean_pp": 100*u.mean(), "upper_sd_pp": 100*u.std(ddof=1),
            **{f"upper_q{q}_pp": 100*np.quantile(u, q/100) for q in [10, 25, 75, 90]},
            "upper_positive_fraction": float(np.mean(u > 0)), "own_gain_mean_pp": 100*g.own_gain.mean(),
            "swap_gain_mean_pp": 100*g.swap_gain.mean(),
            "generic_fraction": g.swap_gain.mean() / g.own_gain.mean(), "statistic": stat, "p_raw_two_sided": p})
for row, corrected in zip(summary, holm([r["p_raw_two_sided"] for r in summary])):
    row["p_holm_diagnostic"] = float(corrected)
pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)

gate = []
for variant in cfg["variants"]:
    u = person[person.variant == variant].upper_bound
    stat, p = test(u, "greater")
    gate.append({"variant": variant, "median_pp": 100*float(np.median(np.round(u, 12))), "statistic": stat, "p_raw": p})
for row, corrected in zip(gate, holm([r["p_raw"] for r in gate])):
    row["p_holm"] = float(corrected)
    row["passed"] = bool(row["median_pp"] >= 100*cfg["gate"]["median_upper_bound"] and corrected < cfg["gate"]["alpha"])
passing = [r["variant"] for r in gate if r["passed"]]
result = {"rule": "spec Stage C2 v1.6: any variant with pooled median U>=2pp and Holm one-sided p<.05",
          "variants": gate, "diagnostic1_passed": bool(passing), "diagnostic2": "not run (diagnostic 1 failed)"}

if passing:  # Diagnostic 2 is analysed only now; rest features are not read otherwise.
    rows = []
    for d in runs:
        feats = np.load(d / "rest_features.npz")
        matrix = pd.read_csv(d / "swap_matrix.csv")
        seed_rows = pd.read_csv(d / "subjects.csv")
        for variant in passing:
            m = matrix[matrix.variant == variant].set_index(["dataset", "subject", "donor"]).ba
            for r in seed_rows[seed_rows.variant == variant].itertuples():
                others = [s for (ds, s) in {(k[0], k[1]) for k in m.index} if ds == r.dataset and s != r.subject]
                dist = {s: np.linalg.norm(feats[f"{r.dataset}:{r.subject}"] - feats[f"{r.dataset}:{s}"]) for s in others}
                nearest = min(sorted(dist), key=dist.get)
                rows.append({"variant": variant, "dataset": r.dataset, "subject": r.subject, "seed": r.seed,
                             "nearest": nearest, "nearest_ba": m[(r.dataset, r.subject, nearest)], "random_ba": r.swap_ba})
    nn = pd.DataFrame(rows)
    nn.to_csv(out / "nearest_seed_results.csv", index=False)
    nn_person = nn.groupby(["variant", "dataset", "subject"], as_index=False)[["nearest_ba", "random_ba"]].mean()
    nn_person["difference"] = nn_person.nearest_ba - nn_person.random_ba
    tests = []
    for variant in passing:
        for name in groups:
            g = group(nn_person[nn_person.variant == variant], name)
            stat, p = test(g.difference, "greater")
            tests.append({"variant": variant, "dataset": name, "n": len(g), "median_pp": 100*g.difference.median(),
                          "mean_pp": 100*g.difference.mean(), "statistic": stat, "p_raw": p})
    pooled = [t for t in tests if t["dataset"] == "pooled"]
    for row, corrected in zip(pooled, holm([t["p_raw"] for t in pooled])):
        row["p_holm"] = float(corrected)
        row["rest_informative"] = bool(corrected < cfg["gate"]["alpha"])
    pd.DataFrame(tests).to_csv(out / "diagnostic2_tests.csv", index=False)
    result["diagnostic2"] = pooled
save_json(out / "gate.json", result)
fits = pd.concat([pd.read_csv(d / "fit_resources.csv") for d in runs], ignore_index=True)
save_json(out / "resources.json", {
    "run_seconds": [r["seconds"] for r in receipts], "peak_gpu_bytes": int(max(r["peak_gpu_bytes"] for r in receipts)),
    "fit_seconds_by_variant": fits[fits.phase == "test"].groupby("variant").fit_seconds.agg(["mean", "max"]).to_dict()})
save_json(out / "run_manifest.json", receipts)
print(json.dumps(result, indent=2))
