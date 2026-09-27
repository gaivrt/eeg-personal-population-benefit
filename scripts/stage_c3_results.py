"""Stage C3 tables: diagnostic-3 gate, rest vs unlabeled-task comparison, few-shot curves."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.stage_a_common import config, save_json, scratch_root
from subject_context.stage_c3 import config_c3
from subject_context.statistics import holm

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path)
args = parser.parse_args()
root = args.root if args.root else scratch_root()
cfg = config_c3(); out = root / "stage_c3/report"
out.mkdir(parents=True, exist_ok=True)
runs = sorted((root / "stage_c3/runs").glob("fold-*_seed-*"))
assert len(runs) == 25
receipts = [json.loads((d / "run.json").read_text()) for d in runs]
assert all(r["status"] == "complete" and r["backbone_unchanged"] and r["stage_c_models_unchanged"] for r in receipts)
# Runs before spec v1.10 asserted exact agreement with C2; later runs record it (consistency_checks.csv).
consistency = [pd.read_csv(d / "consistency_checks.csv").assign(run=d.name) for d in runs
               if (d / "consistency_checks.csv").exists()]
if consistency:
    consistency = pd.concat(consistency, ignore_index=True)
    consistency.to_csv(out / "consistency_checks.csv", index=False)
    save_json(out / "consistency_summary.json", {
        "runs_recording": sorted(consistency.run.unique()),
        "runs_asserting_exact": [d.name for d in runs if not (d / "consistency_checks.csv").exists()],
        "by_check": {k: {"n": len(g), "mismatches": int((g.difference.abs() > 1e-12).sum()),
                         "max_abs_difference": float(g.difference.abs().max())}
                     for k, g in consistency.groupby("check")}})
groups = [*config()["datasets"], "Dreyer+Cho", "pooled"]


def group(df, name):
    if name == "pooled":
        return df
    return df[df.dataset.isin(["Dreyer2023", "Cho2017"])] if name == "Dreyer+Cho" else df[df.dataset == name]


def greater(values):
    d = np.round(np.asarray(values, dtype=float), 12)
    return float(wilcoxon(d, zero_method="wilcox", method="approx", alternative="greater").pvalue) if np.any(d) else 1.


nb = pd.concat([pd.read_csv(d / "neighbours.csv") for d in runs], ignore_index=True)
assert len(nb) == 235 * 5 * 2 * 2 * 2  # people x seeds x variants x sources x k
nb.to_csv(out / "neighbour_seed_results.csv", index=False)
keys = ["variant", "source", "k", "dataset", "subject"]
nbp = nb.groupby(keys, as_index=False)[["nearest_ba", "random_ba", "difference"]].mean()
nbp.to_csv(out / "neighbour_subject_results.csv", index=False)
comparison = []
for (variant, source, k), g in nbp.groupby(["variant", "source", "k"]):
    for name in groups:
        h = group(g, name)
        comparison.append({"variant": variant, "source": source, "k": int(k), "dataset": name, "n": len(h),
                           "nearest_mean": h.nearest_ba.mean(), "random_mean": h.random_ba.mean(),
                           "median_pp": 100*h.difference.median(), "mean_pp": 100*h.difference.mean(),
                           "improved_fraction": float(np.mean(np.round(h.difference, 12) > 0)),
                           "p_raw_greater": greater(h.difference)})
pd.DataFrame(comparison).to_csv(out / "rest_vs_task.csv", index=False)
gate = [r for r in comparison if r["source"] == "task" and r["k"] == cfg["diagnostic3"]["primary_k"]
        and r["dataset"] == "pooled"]
for row, corrected in zip(gate, holm([r["p_raw_greater"] for r in gate])):
    row["p_holm"] = float(corrected)
    row["passed"] = bool(corrected < cfg["diagnostic3"]["alpha"])
informative = any(r["passed"] for r in gate)
save_json(out / "gate.json", {"rule": "spec Stage C3 v1.9: task features, k=1, one-sided Wilcoxon, Holm over two variants",
    "variants": gate, "unlabeled_task_informative": informative,
    "conclusion": "无标签任务数据含有可用信息" if informative else
                  "静息与无标签任务数据的这些特征都不指示个人适配方向，停止无标签个性化路线"})

fs = pd.concat([pd.read_csv(d / "few_shot.csv") for d in runs], ignore_index=True)
fs.to_csv(out / "few_shot_seed_results.csv", index=False)
fsp = fs.groupby(["variant", "shots", "dataset", "subject"], as_index=False).agg(
    gain=("gain", "mean"), full_gain=("full_gain", "mean"), upper_bound=("upper_bound", "mean"),
    one_class_only=("one_class_only", "any"))
fsp.to_csv(out / "few_shot_subject_results.csv", index=False)
counts = fsp.groupby(["variant", "shots"]).size()
assert all(counts[(v, n)] == (235 if n <= 20 else 132) for v in cfg["variants"] for n in cfg["few_shot"]["shots"])
assert (fs.groupby(["variant", "shots", "dataset", "subject"]).size() == 5).all()  # every seed present
curve = []
for (variant, n), g in fsp.groupby(["variant", "shots"]):
    for name in groups:
        h = group(g, name)
        if h.empty:
            continue
        d = np.round(h.gain, 12)
        curve.append({"variant": variant, "shots": n, "dataset": name, "n_subjects": len(h),
                      "median_pp": 100*np.median(d), "mean_pp": 100*d.mean(),
                      "improved_fraction": float(np.mean(d > 0)), "drop_gt_2pp_fraction": float(np.mean(d < -.02)),
                      "full_mean_pp": 100*h.full_gain.mean(), "recovery": h.gain.mean() / h.full_gain.mean(),
                      "recovery_vs_upper_bound": h.gain.mean() / h.upper_bound.mean(),
                      "one_class_subjects": int(h.one_class_only.sum())})
pd.DataFrame(curve).to_csv(out / "few_shot_curve.csv", index=False)
zero = [pd.read_csv(d / "lora_zero_vs_b3.csv").assign(run=d.name) for d in runs if (d / "lora_zero_vs_b3.csv").exists()]
if zero:  # recorded only by runs executed after the B3-path fix (spec v1.10)
    zero = pd.concat(zero, ignore_index=True)
    zero.to_csv(out / "lora_zero_vs_b3.csv", index=False)
    save_json(out / "lora_zero_vs_b3.json", {"runs": sorted(zero.run.unique()), "max_logit_difference": zero.max_logit_difference.max(),
        "min_prediction_agreement": zero.prediction_agreement.min(), "subjects_not_identical": int((zero.prediction_agreement < 1).sum())})
fits = pd.concat([pd.read_csv(d / "fit_resources.csv") for d in runs], ignore_index=True)
save_json(out / "resources.json", {"run_seconds": [r["seconds"] for r in receipts],
    "peak_gpu_bytes": int(max(r["peak_gpu_bytes"] for r in receipts)),
    "fit_seconds": {f"{v}_n{n}": s for (v, n), s in fits.groupby(["variant", "shots"]).fit_seconds.mean().items()}})
save_json(out / "run_manifest.json", receipts)
print(json.dumps({"gate": gate, "unlabeled_task_informative": informative}, indent=2))
