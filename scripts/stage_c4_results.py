"""Stage C4 tables, the preregistered n=10 endpoint, secondary endpoints and equivalent label saving."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.stage_a_common import config, save_json, scratch_root
from subject_context.stage_c4 import config_c4
from subject_context.statistics import equivalent_trials, holm

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path)
args = parser.parse_args()
root = args.root if args.root else scratch_root()
cfg = config_c4(); out = root / "stage_c4/report"
out.mkdir(parents=True, exist_ok=True)
runs = sorted((root / "stage_c4/runs").glob("fold-*_seed-*"))
assert len(runs) == 25
receipts = [json.loads((d / "run.json").read_text()) for d in runs]
assert all(r["status"] == "complete" and r["backbone_unchanged"] and r["stage_c_models_unchanged"] for r in receipts)
data = pd.concat([pd.read_csv(d / "results.csv") for d in runs], ignore_index=True)
for variant, spec in cfg["variants"].items():
    for prior in cfg["priors"]:
        for n in spec["shots"]:
            got = data[(data.variant == variant) & (data.prior == prior) & (data.shots == n)]
            assert len(got) == 235 * 5, (variant, prior, n, len(got))
data.to_csv(out / "subject_seed_results.csv", index=False)
person = data.groupby(["variant", "prior", "shots", "dataset", "subject"], as_index=False)[["b3_ba", "ba", "gain"]].mean()
person.to_csv(out / "subject_results.csv", index=False)
groups = [*config()["datasets"], "Dreyer+Cho", "pooled"]


def group(df, name):
    if name == "pooled":
        return df
    return df[df.dataset.isin(["Dreyer2023", "Cho2017"])] if name == "Dreyer+Cho" else df[df.dataset == name]


def describe(d, alternative="greater"):
    d = np.round(np.asarray(d, dtype=float), 12)
    p = float(wilcoxon(d, zero_method="wilcox", method="approx", alternative=alternative).pvalue) if np.any(d) else 1.
    return {"n": len(d), "median_pp": 100*np.median(d), "mean_pp": 100*d.mean(), "improved_fraction": float(np.mean(d > 0)),
            "drop_gt_2pp_fraction": float(np.mean(d < -.02)), "p_greater": p}


table = []
for (variant, prior, n), g in person.groupby(["variant", "prior", "shots"]):
    for name in groups:
        h = group(g, name)
        table.append({"variant": variant, "prior": prior, "shots": n, "dataset": name, "ba_mean": h.ba.mean(),
                      "ba_sd": h.ba.std(), "gain_median_pp": 100*h.gain.median(), "gain_mean_pp": 100*h.gain.mean(),
                      "improved_fraction": float(np.mean(np.round(h.gain, 12) > 0)),
                      "drop_gt_2pp_fraction": float(np.mean(np.round(h.gain, 12) < -.02))})
pd.DataFrame(table).to_csv(out / "prior_by_shots.csv", index=False)

wide = person.pivot_table(index=["variant", "shots", "dataset", "subject"], columns="prior", values="ba").reset_index()
comparisons = []
for (variant, n), g in wide.groupby(["variant", "shots"]):
    for a, b in [("P2", "P1"), ("P2", "P0"), ("P1", "P0")]:
        for name in groups:
            h = group(g, name)
            comparisons.append({"variant": variant, "shots": n, "comparison": f"{a}-{b}", "dataset": name,
                                **describe(h[a] - h[b])})
comparisons = pd.DataFrame(comparisons)
comparisons.to_csv(out / "comparisons.csv", index=False)


def pooled(variant, n, comparison):
    return comparisons[(comparisons.variant == variant) & (comparisons.shots == n) & (comparisons.comparison == comparison)
                       & (comparisons.dataset == "pooled")].iloc[0].to_dict()


pr = cfg["primary"]
primary = {c: pooled(pr["variant"], pr["shots"], c) for c in ["P2-P1", "P2-P0"]}
passed = all(primary[c]["p_greater"] < pr["alpha"] for c in primary)
curve = person[(person.variant == pr["variant"]) & (person.prior == "P0")].groupby("shots").ba.mean()
target = person[(person.variant == pr["variant"]) & (person.prior == "P2") & (person.shots == pr["shots"])].ba.mean()
saving = equivalent_trials(target, curve.index.to_numpy(), curve.to_numpy())
secondary = [pooled(pr["variant"], n, "P2-P1") for n in cfg["secondary"]["shots"]]
for row, corrected in zip(secondary, holm([r["p_greater"] for r in secondary])):
    row["p_holm"] = float(corrected)
save_json(out / "endpoints.json", {
    "rule": "spec Stage C4 v1.13: LoRA n=10, P2>P1 and P2>P0 (one-sided Wilcoxon p<.05)",
    "primary": primary, "passed": passed,
    "conclusion": "无标签任务数据提供的先验能减少所需标签" if passed else "无标签任务数据的信息不足以在少样本校准中带来可测收益",
    "p0_curve_pooled": curve.to_dict(), "p2_at_primary_n_pooled": target,
    "equivalent_p0_labels": saving, "label_saving": None if saving is None else saving - pr["shots"],
    "saving_note": "interpolated" if saving is not None else
                   ("above P0 at the largest observed n" if target > curve.max() else
                    "below P0 at n=0" if target < curve.iloc[0] else "P0 curve not monotone"),
    "secondary_P2_vs_P1": secondary,
    "secondary_P2_vs_P0": [pooled(pr["variant"], n, "P2-P0") for n in cfg["secondary"]["shots"]]})
ka = pd.concat([pd.read_csv(d / "k_alpha.csv").assign(seed=d.name) for d in runs], ignore_index=True)
ka.groupby(["variant", "prior", "k", "alpha"]).size().rename("count").reset_index().to_csv(out / "k_alpha_distribution.csv", index=False)
fits = pd.concat([pd.read_csv(d / "fit_resources.csv") for d in runs], ignore_index=True)
save_json(out / "resources.json", {"run_seconds": [r["seconds"] for r in receipts],
    "peak_gpu_bytes": int(max(r["peak_gpu_bytes"] for r in receipts)),
    "fit_seconds_mean": float(fits.fit_seconds.mean()), "fits": len(fits)})
save_json(out / "run_manifest.json", receipts)
print(json.dumps({"primary": primary, "passed": passed, "label_saving": None if saving is None else saving - pr["shots"]},
                 indent=2, default=float))
