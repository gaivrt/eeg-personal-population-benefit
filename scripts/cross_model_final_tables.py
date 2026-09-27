"""Derive final descriptive tables from audited outputs; no fitting or selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cross_model_results import distribution

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/stage_x"
DATASETS = ["PhysionetMI", "Dreyer2023", "Cho2017"]


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


accounting = pd.read_csv(OUT / "slurm_accounting.psv", sep="|", keep_default_na=False)
assert not accounting.JobID.duplicated().any()
accounting["GPUs"] = accounting.AllocTRES.map(
    lambda s: next((int(v.split("=")[1]) for v in s.split(",") if v.startswith("gres/gpu=")), 0))
accounting["GPU_card_hours"] = accounting.GPUs * accounting.ElapsedRaw / 3600
accounting.to_csv(OUT / "resource_allocations.csv", index=False)
gpu = accounting[accounting.GPUs > 0]
events = []
for row in gpu.itertuples():
    events.extend([(pd.Timestamp(row.Start), row.GPUs), (pd.Timestamp(row.End), -row.GPUs)])
active, peak = 0, 0
for _, delta in sorted(events):
    active += delta
    peak = max(peak, active)
assert active == 0 and peak <= 16
core = accounting[accounting.JobID.str.startswith("1910839_")]
assert len(core) == 50 and (core.State == "COMPLETED").all() and (core.ExitCode == "0:0").all()
summary = {"effective_core_runs": 50, "peak_GPUs": peak,
           "all_GPU_card_hours_including_failed_cache": float(gpu.GPU_card_hours.sum()),
           "core_GPU_card_hours": float(core.GPU_card_hours.sum()),
           "GPU_wall_hours": (pd.Timestamp(gpu.End.max()) - pd.Timestamp(gpu.Start.min())).total_seconds()/3600,
           "core_wall_hours": (pd.Timestamp(core.End.max()) - pd.Timestamp(core.Start.min())).total_seconds()/3600,
           "first_GPU_start_local": gpu.Start.min(), "last_GPU_end_local": gpu.End.max(),
           "shared_preparation_CPU_allocation_seconds": int(accounting.loc[accounting.JobID.str.startswith("1910586_"), "ElapsedRaw"].sum()),
           "failed_GPU_cache_seconds": 22, "cancelled_core_runs_before_start": 50,
           "models": {}}
comparison, curve_comparison = [], []
for model, offset, cache, report_job, export_job in [
        ("REVE", 0, "1910838_0", "1910840", "1910981"),
        ("LaBraM", 25, "1910587_1", "1910841", "1910982")]:
    out = OUT / model / "report"
    rows = core[core.JobID.isin([f"1910839_{i}" for i in range(offset, offset+25)])]
    rows.to_csv(out / "slurm_resources.csv", index=False)
    resources = read(out / "run_resources.csv")
    choices = read(out / "training_selection.csv")
    failed = 22/3600 if model == "REVE" else 0
    r = {"model": model, "core_jobs_completed": 25,
         "core_GPU_card_hours": float(rows.GPU_card_hours.sum()),
         "successful_cache_GPU_card_hours": float(accounting.set_index("JobID").loc[cache, "GPU_card_hours"]),
         "failed_cache_GPU_card_hours": failed,
         "core_wall_hours": (pd.Timestamp(rows.End.max())-pd.Timestamp(rows.Start.min())).total_seconds()/3600,
         "peak_allocated_GiB": float(resources.peak_allocated_bytes_recorded_max.max()/2**30),
         "peak_reserved_GiB": float(resources.peak_reserved_bytes_recorded_max.max()/2**30),
         "G_stops": choices.stopped_by.value_counts().to_dict(),
         "G_selected_initial_step_counts": choices.initial_selected_step.value_counts().to_dict(),
         "G_actual_additional_steps_range": [int(choices.additional_steps_run.min()), int(choices.additional_steps_run.max())],
         "G_selected_additional_steps_range": [int(choices.selected_additional_step.min()), int(choices.selected_additional_step.max())],
         "G_plateau_observed": int(choices.plateau_observed.sum()),
         "selected_G_in_final_plateau_window": int(choices.selected_in_final_plateau_window.sum()),
         "B0_head_counts": choices.B0_head.value_counts().to_dict(),
         "B0_selected_epoch_range": [int(choices.B0_epoch.min()), int(choices.B0_epoch.max())],
         "report_CPU_seconds": int(accounting.set_index("JobID").loc[report_job, "ElapsedRaw"]),
         "retention_CPU_seconds": int(accounting.set_index("JobID").loc[export_job, "ElapsedRaw"]),
         "peak_memory_basis": "Maximum over recorded B0/G segments and all personal fit allocated-memory records. Per-fit CUDA resets mean final run.json is only the final window. Reserved memory is logged only in B0/G and final window, not continuously via NVML."}
    r["total_model_GPU_card_hours"] = r["core_GPU_card_hours"]+r["successful_cache_GPU_card_hours"]+failed
    save_json(out / "resource_summary.json", r)
    summary["models"][model] = r
    people = read(out / "subject_means.csv")
    core_table = read(out / "core_summary.csv").set_index("dataset")
    source = core_table.pretraining_source.str.startswith("seen").to_dict()
    people["source_stratum"] = people.dataset.map(lambda d: "seen" if source[d] else "unlisted")
    strata = []
    for label, group in people.groupby("source_stratum"):
        row = {"model": model, "source_stratum": label, "datasets": ";".join(sorted(group.dataset.unique()))}
        for name in ["B0_BA", "G_BA", "own_ba", "swap_ba"]:
            row[name+"_mean_percent"] = 100*group[name].mean()
            row[name+"_SD_percent"] = 100*group[name].std(ddof=1)
        for name, values in [("G_minus_B0", group.G_BA-group.B0_BA), ("own_minus_G", group.own_gain), ("own_minus_swap", group.upper_bound)]:
            row.update({name+"_"+k:v for k,v in distribution(values).items()})
        strata.append(row)
    pd.DataFrame(strata).to_csv(out / "pretraining_strata_descriptive.csv", index=False)
    few = read(out / "few_shot_subject_means.csv")
    absolute = few.groupby(["dataset", "shots"]).ba.agg(["mean", "std"])*100
    absolute.columns = ["BA_mean_percent", "BA_subject_SD_percent"]
    seeds = read(out / "seed_level_curves.csv").groupby(["dataset", "shots"]).gain.std()*100
    seeds.name = "dataset_mean_gain_seed_SD_pp"
    absolute.join(seeds).join(read(out / "single_class_prefixes.csv").set_index(["dataset", "shots"])).to_csv(out / "few_shot_report_annex.csv")
    curves = read(out / "few_shot_summary.csv")
    for dataset in DATASETS:
        c = core_table.loc[dataset]
        ns = curves[curves.dataset == dataset].sort_values("shots")
        gains = np.round(ns.mean_pp.to_numpy()/100, 12)
        valid = ns.full_gain_mean.gt(0).all()
        stable = bool((gains >= 0).all() and (np.diff(gains) >= 0).all()) if valid else None
        comparison.append({"model": model, "dataset": dataset, "pretraining_source": c.pretraining_source,
                           "G_minus_B0_mean_pp": c.G_minus_B0_mean_pp, "own_minus_G_mean_pp": c.own_minus_G_mean_pp,
                           "own_minus_swap_mean_pp": c.own_minus_swap_mean_pp,
                           "own_minus_swap_median_pp": c.own_minus_swap_median_pp,
                           "own_minus_swap_SD_pp": c.own_minus_swap_SD_pp,
                           "own_minus_swap_p_raw_descriptive": c.own_minus_swap_p_raw_descriptive,
                           "few_shot_stable": stable})
        for row in ns.itertuples():
            curve_comparison.append({"model": model, "dataset": dataset, "shots": row.shots,
                                     "mean_pp": row.mean_pp, "full_gain_mean_pp": row.full_gain_mean*100})

cb_pop = read(ROOT / "reports/stage_w/report/population_absolute_mean_sd.csv").pivot(index="dataset", columns="condition", values="mean")
cb_own = read(ROOT / "reports/stage_w_followup/report/population_strength_by_dataset.csv")
cb_own = cb_own[(cb_own.start == "G") & (cb_own.variant == "lora8")].set_index("dataset")
cb_few = read(ROOT / "reports/stage_w/report/few_shot_side_by_side.csv")
cb_few = cb_few[cb_few.start == "G"]
for dataset in DATASETS:
    rows = cb_few[cb_few.dataset == dataset].sort_values("shots")
    gains = np.round(rows.mean_pp.to_numpy()/100, 12)
    assert rows.full_gain_mean_pp.gt(0).all()
    comparison.append({"model": "CBraMod", "dataset": dataset, "pretraining_source": "source_unlisted",
                       "G_minus_B0_mean_pp": cb_pop.loc[dataset, "G_lora"]-cb_pop.loc[dataset, "B0"],
                       "own_minus_G_mean_pp": cb_own.loc[dataset, "own_minus_population_mean_pp"],
                       "own_minus_swap_mean_pp": cb_own.loc[dataset, "own_minus_swap_mean_pp"],
                       "own_minus_swap_median_pp": cb_own.loc[dataset, "own_minus_swap_median_pp"],
                       "own_minus_swap_SD_pp": cb_own.loc[dataset, "own_minus_swap_sd_pp"],
                       "own_minus_swap_p_raw_descriptive": cb_own.loc[dataset, "own_minus_swap_p_one_sided_raw"],
                       "few_shot_stable": bool((gains >= 0).all() and (np.diff(gains) >= 0).all())})
    for row in rows.itertuples():
        curve_comparison.append({"model": "CBraMod", "dataset": dataset, "shots": row.shots,
                                 "mean_pp": row.mean_pp, "full_gain_mean_pp": row.full_gain_mean_pp})
comparison = pd.DataFrame(comparison)
curve_comparison = pd.DataFrame(curve_comparison)
comparison.to_csv(OUT / "three_model_effects.csv", index=False)
curve_comparison.to_csv(OUT / "three_model_few_shot.csv", index=False)
consistency = []
for dataset, rows in comparison.groupby("dataset"):
    assert set(rows.model) == {"CBraMod", "REVE", "LaBraM"}
    result = {"dataset": dataset}
    for name in ["G_minus_B0", "own_minus_G", "own_minus_swap"]:
        signs = np.sign(rows[name+"_mean_pp"])
        result[name] = "consistent_positive" if (signs == 1).all() else "consistent_negative" if (signs == -1).all() else "consistent_zero" if (signs == 0).all() else "inconsistent"
    result["few_shot_stability"] = "undetermined" if rows.few_shot_stable.isna().any() else "consistent_stable" if rows.few_shot_stable.all() else "consistent_unstable" if (~rows.few_shot_stable).all() else "inconsistent"
    for shots, ns in curve_comparison[curve_comparison.dataset == dataset].groupby("shots"):
        assert len(ns) == 3
        result[f"n{shots}_direction"] = "consistent_positive" if (ns.mean_pp > 0).all() else "consistent_negative" if (ns.mean_pp < 0).all() else "consistent_zero" if (ns.mean_pp == 0).all() else "inconsistent"
    consistency.append(result)
pd.DataFrame(consistency).to_csv(OUT / "three_model_consistency.csv", index=False)
save_json(OUT / "resource_summary.json", summary)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), sharey=True)
colors = {"CBraMod": "#5B6472", "REVE": "#007F86", "LaBraM": "#C25D24"}
for ax, dataset in zip(axes, DATASETS):
    for model in ["CBraMod", "REVE", "LaBraM"]:
        data = curve_comparison[(curve_comparison.model == model) & (curve_comparison.dataset == dataset)].sort_values("shots")
        seen = (model == "REVE" and dataset != "PhysionetMI") or (model == "LaBraM" and dataset == "PhysionetMI")
        ax.plot(data.shots, data.mean_pp, marker="o", color=colors[model],
                linestyle="--" if model == "CBraMod" else "-", label=model+(" (seen)" if seen else ""))
    ax.axhline(0, color="#979797", linewidth=.8)
    ax.set_title("PhysioNet MI" if dataset == "PhysionetMI" else dataset)
    ax.set_xticks([5, 10, 20] if dataset == "PhysionetMI" else [5, 10, 20, 40])
    ax.set_xlabel("Calibration trials (total)")
    ax.grid(axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
axes[0].set_ylabel("Mean BA gain over G (percentage points)")
fig.tight_layout()
for suffix in ["png", "pdf"]:
    fig.savefig(OUT / f"three_model_few_shot.{suffix}", dpi=180, bbox_inches="tight")
plt.close(fig)
print(json.dumps({"resources": summary, "consistency": consistency}))
