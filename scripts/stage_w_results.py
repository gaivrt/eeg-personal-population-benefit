"""Audit predictions and assemble per-dataset B3/G comparisons; refuse incomplete W."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.stage_a_common import ROOT, save_json, scratch_root, sha256
from subject_context.statistics import holm


def score(labels, predictions):
    """Independent recomputation, without importing the experiment BA function."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    assert set(labels) == {0, 1}
    return sum(float((predictions[labels == label] == label).mean()) for label in [0, 1])/2


def distribution(values):
    x = np.round(np.asarray(values), 12)
    p = float(wilcoxon(x, alternative="greater", method="approx", zero_method="wilcox").pvalue) if np.any(x) else 1.
    return {"subjects":len(x),"median_pp":100*np.median(x),"mean_pp":100*x.mean(),
            "sd_pp":100*x.std(ddof=1),"improved_fraction":float((x>0).mean()),
            "drop_gt_2pp_fraction":float((x<-.02).mean()),"p_one_sided_raw":p}


def load_runs(root, directory):
    runs = sorted((root / directory).glob("fold-*_seed-*"))
    assert len(runs) == 25, (directory, len(runs))
    receipts = [json.loads((d/"run.json").read_text()) for d in runs]
    assert all(r["status"] == "complete" for r in receipts)
    assert {(r["fold"],r["seed"]) for r in receipts} == {(f,s) for f in range(5) for s in [11,23,37,53,71]}
    assert all(r.get("w_config_sha256",r.get("stage_w_config_sha256")) == sha256(ROOT/"configs/stage_w.yaml") for r in receipts)
    return runs,receipts


def concat(runs, filename):
    parts = []
    for directory in runs:
        part = pd.read_csv(directory/filename)
        fold, seed = directory.name.split("_")
        part["fold"],part["seed"] = int(fold.split("-")[1]),int(seed.split("-")[1])
        parts.append(part)
    return pd.concat(parts,ignore_index=True)


def audit_truth(root, predictions):
    assert not predictions.empty
    for (dataset,subject),group in predictions.groupby(["dataset","subject"]):
        trials = pd.read_csv(root/"processed"/dataset/f"sub-{subject:03d}"/"trials.csv").set_index("trial_id")
        assert set(group.trial_id) <= set(trials.index)
        np.testing.assert_array_equal(group.label.to_numpy(),trials.loc[group.trial_id,"label"].to_numpy())
        assert group.prediction.isin([0,1]).all()


def audit_all(root, population, d1, d3, lee):
    checked = {"population_rows":0,"swap_rows":0,"few_shot_rows":0,"lee_rows":0}
    for directory in population:
        p = pd.read_csv(directory/"predictions.csv")
        audit_truth(root,p)
        observed = p[p.is_query].groupby(["method","dataset","subject"]).apply(
            lambda g:score(g.label,g.prediction),include_groups=False)
        saved = pd.read_csv(directory/"subjects.csv").set_index(["method","dataset","subject"]).ba
        np.testing.assert_allclose(observed.loc[saved.index],saved,atol=1e-12,rtol=0)
        checked["population_rows"] += len(p)
    for directory in d1:
        p = pd.read_csv(directory/"predictions.csv.gz")
        audit_truth(root,p)
        keys = ["variant","dataset","subject","donor"]
        observed = p.groupby(keys).apply(lambda g:score(g.label,g.prediction),include_groups=False)
        saved = pd.read_csv(directory/"swap_matrix.csv").set_index(keys).ba
        np.testing.assert_allclose(observed.loc[saved.index],saved,atol=1e-12,rtol=0)
        people = pd.read_csv(directory/"subjects.csv")
        for r in people.itertuples():
            own = observed[(r.variant,r.dataset,r.subject,r.subject)]
            swap = np.mean([observed[(r.variant,r.dataset,r.subject,int(s))] for s in str(r.swap_donors).split()])
            np.testing.assert_allclose([own,swap,own-swap],[r.own_ba,r.swap_ba,r.upper_bound],atol=1e-12,rtol=0)
        checked["swap_rows"] += len(p)
    for directory in d3:
        p = pd.read_csv(directory/"predictions.csv.gz")
        audit_truth(root,p)
        keys = ["variant","dataset","subject","shots"]
        observed = p[p.condition=="few_shot"].groupby(keys).apply(lambda g:score(g.label,g.prediction),include_groups=False)
        saved = pd.read_csv(directory/"few_shot.csv").set_index(keys).ba
        np.testing.assert_allclose(observed.loc[saved.index],saved,atol=1e-12,rtol=0)
        zero = p[p.condition=="G"].groupby(["variant","dataset","subject"]).apply(
            lambda g:score(g.label,g.prediction),include_groups=False)
        for r in pd.read_csv(directory/"few_shot.csv").itertuples():
            np.testing.assert_allclose(zero[(r.variant,r.dataset,r.subject)],r.b3_ba,atol=1e-12,rtol=0)
            np.testing.assert_allclose(r.ba-r.b3_ba,r.gain,atol=1e-12,rtol=0)
        checked["few_shot_rows"] += len(p)
    for directory in lee:
        p = pd.read_csv(directory/"predictions.csv.gz")
        audit_truth(root,p)
        saved = pd.read_csv(directory/"baseline.csv")
        for r in saved.itertuples():
            g = p[(p.subject==r.subject)&(p.condition==r.condition)]
            np.testing.assert_allclose(score(g.label,g.prediction),r.ba,atol=1e-12,rtol=0)
        saved = pd.read_csv(directory/"subjects.csv")
        for r in saved.itertuples():
            scores = {}
            for donor,g in p[(p.subject==r.subject)&(p.condition=="swap_"+r.variant)].groupby("donor"):
                scores[int(donor)] = score(g.label,g.prediction)
            np.testing.assert_allclose([scores[r.subject],np.mean([scores[int(s)] for s in str(r.swap_donors).split()])],
                                      [r.own_ba,r.swap_ba],atol=1e-12,rtol=0)
        for r in pd.read_csv(directory/"few_shot.csv").itertuples():
            g = p[(p.subject==r.subject)&(p.condition=="few_lora8")&(p.shots==r.shots)]
            np.testing.assert_allclose(score(g.label,g.prediction),r.ba,atol=1e-12,rtol=0)
            zero = p[(p.subject==r.subject)&(p.condition=="few_G")&(p.shots==r.shots)]
            np.testing.assert_allclose(score(zero.label,zero.prediction),r.b3_ba,atol=1e-12,rtol=0)
        checked["lee_rows"] += len(p)
    return checked


def person(data, keys, columns):
    assert not data.duplicated([*keys,"seed"]).any()
    assert (data.groupby(keys).seed.nunique()==5).all()
    return data.groupby(keys,as_index=False)[columns].mean()


def md(frame):
    result = ["| " + " | ".join(frame.columns) + " |", "|"+"|".join(["---"]*len(frame.columns))+"|"]
    for row in frame.itertuples(index=False,name=None):
        result.append("| " + " | ".join(f"{x:.3f}" if isinstance(x,float) else str(x) for x in row) + " |")
    return "\n".join(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path)
    args = parser.parse_args()
    root = args.root or scratch_root()
    population,rp = load_runs(root,"stage_w/population/runs")
    d1,r1 = load_runs(root,"stage_w/diagnostic1/runs")
    d3,r3 = load_runs(root,"stage_w/diagnostic3/runs")
    lee,rl = load_runs(root,"stage_w/lee/runs")
    out = root/"stage_w/report"
    out.mkdir(parents=True,exist_ok=True)
    verified = audit_all(root,population,d1,d3,lee)
    # Aggregate seeds before statistical comparisons.
    pop = person(concat(population,"subjects.csv"),["method","dataset","subject"],["ba"])
    old = pd.read_csv(root/"stage_c/report/subject_results.csv")
    r2 = pd.read_csv(root/"stage_m/report/subject_results.csv").query("method == 'R2' and shots == 0")
    comparisons = []
    for method,group in pop.groupby("method"):
        for control,ref in [("B0",old[old.condition=="B0"]),("B3",old[old.condition=="B3_"+method[2:]]),
                            ("R2",r2 if method=="m_lora" else pd.DataFrame())]:
            if ref.empty:
                continue
            merged = group.merge(ref[["dataset","subject","ba"]],on=["dataset","subject"],validate="one_to_one",suffixes=("_G","_control"))
            assert len(merged)==235
            for dataset,g in merged.groupby("dataset"):
                comparisons.append({"method":method,"dataset":dataset,"control":control,
                    "G_ba_percent":100*g.ba_G.mean(),"control_ba_percent":100*g.ba_control.mean(),
                    **distribution(g.ba_G-g.ba_control)})
    pd.DataFrame(comparisons).to_csv(out/"population_comparisons.csv",index=False)
    dc = ["b3_ba","own_ba","swap_ba","upper_bound","own_gain","swap_gain"]
    new_d = person(concat(d1,"subjects.csv"),["variant","dataset","subject"],dc).assign(start="G")
    old_d = pd.read_csv(root/"stage_c2/report/subject_results.csv").query("variant in ['film_offset','lora8']").assign(start="B3")
    lee_d = person(concat(lee,"subjects.csv"),["variant","dataset","subject"],dc).assign(start="G")
    all_d = pd.concat([old_d,new_d,lee_d],ignore_index=True)
    all_d.to_csv(out/"diagnostic1_subject_results.csv",index=False)
    d_summary = []
    for (start,variant,dataset),g in all_d.groupby(["start","variant","dataset"]):
        d_summary.append({"start":start,"variant":variant,"dataset":dataset,
            "start_ba_percent":100*g.b3_ba.mean(),"own_ba_percent":100*g.own_ba.mean(),"swap_ba_percent":100*g.swap_ba.mean(),
            "own_gain_median_pp":100*g.own_gain.median(),"own_gain_mean_pp":100*g.own_gain.mean(),
            **distribution(g.upper_bound)})
    pd.DataFrame(d_summary).to_csv(out/"diagnostic1_side_by_side.csv",index=False)
    nc = ["nearest_ba","random_ba","difference"]
    ng = ["variant","source","k","dataset","subject"]
    new_n = person(concat(d3,"neighbours.csv"),ng,nc).assign(start="G")
    old_n = pd.read_csv(root/"stage_c3/report/neighbour_subject_results.csv").assign(start="B3")
    lee_n = person(concat(lee,"neighbours.csv"),ng,nc).assign(start="G")
    all_n = pd.concat([old_n,new_n,lee_n],ignore_index=True)
    all_n.to_csv(out/"neighbour_subject_results.csv",index=False)
    nn_summary = []
    for (start,variant,source,k,dataset),g in all_n.groupby(["start","variant","source","k","dataset"]):
        nn_summary.append({"start":start,"variant":variant,"source":source,"k":k,"dataset":dataset,
            "nearest_ba_percent":100*g.nearest_ba.mean(),"random_ba_percent":100*g.random_ba.mean(),**distribution(g.difference)})
    pd.DataFrame(nn_summary).to_csv(out/"neighbours_side_by_side.csv",index=False)
    fc = ["ba","b3_ba","gain","full_gain"]
    fk = ["variant","shots","dataset","subject"]
    new_f = person(concat(d3,"few_shot.csv"),fk,fc).assign(start="G")
    old_f = person(pd.read_csv(root/"stage_c3/report/few_shot_seed_results.csv").query("variant == 'lora8'"),fk,fc).assign(start="B3")
    lee_f = person(concat(lee,"few_shot.csv"),fk,fc).assign(start="G")
    all_f = pd.concat([old_f,new_f,lee_f],ignore_index=True)
    all_f.to_csv(out/"few_shot_subject_results.csv",index=False)
    few_summary = []
    for (start,n,dataset),g in all_f.groupby(["start","shots","dataset"]):
        few_summary.append({"start":start,"shots":n,"dataset":dataset,"ba_percent":100*g.ba.mean(),
            "n0_ba_percent":100*g.b3_ba.mean(),"full_gain_mean_pp":100*g.full_gain.mean(),
            "recovery_fraction":g.gain.mean()/g.full_gain.mean() if abs(g.full_gain.mean())>1e-12 else np.nan,
            **distribution(g.gain)})
    pd.DataFrame(few_summary).to_csv(out/"few_shot_side_by_side.csv",index=False)
    person(concat(lee,"baseline.csv"),["condition","dataset","subject"],["ba"]).to_csv(out/"lee_baseline_subjects.csv",index=False)
    # Preserve original pooled diagnostic families; per-dataset tables above are descriptive.
    gates = []
    for variant,g in new_d.groupby("variant"):
        test = distribution(g.upper_bound)
        gates.append({"variant":variant,**test})
    adjusted = holm([r["p_one_sided_raw"] for r in gates]+[1.])[:len(gates)]
    for row,p in zip(gates,adjusted):
        row.update(p_holm=float(p),passed=bool(row["median_pp"]>=2 and p<.05),family_size=3)
    passing = {r["variant"] for r in gates if r["passed"]}
    nn_tests = []
    for source in ["rest","task"]:
        rows = []
        for variant,g in new_n[(new_n.source==source)&(new_n.k==1)].groupby("variant"):
            rows.append({"variant":variant,"source":source,**distribution(g.difference),
                         "eligible":source=="task" or variant in passing})
        eligible = [r for r in rows if r["eligible"]]
        pvalues = [r["p_one_sided_raw"] for r in eligible]
        # The omitted C2 mixture-offset variant occupies one conservative p=1 slot.
        corrected = holm(pvalues+([1.] if source=="rest" else []))[:len(eligible)]
        for row,p in zip(eligible,corrected):
            row.update(p_holm=float(p),passed=bool(p<.05))
        nn_tests.extend(rows)
    lee_gates,lee_nn = [],[]
    for variant,g in lee_d.groupby("variant"):
        lee_gates.append({"variant":variant,**distribution(g.upper_bound)})
    for row,p in zip(lee_gates,holm([r["p_one_sided_raw"] for r in lee_gates]+[1.])[:len(lee_gates)]):
        row.update(p_holm=float(p),passed=bool(row["median_pp"]>=2 and p<.05),family_size=3)
    for variant,g in lee_n.groupby("variant"):
        lee_nn.append({"variant":variant,**distribution(g.difference),
                       "eligible":any(r["variant"]==variant and r["passed"] for r in lee_gates)})
    eligible = [r for r in lee_nn if r["eligible"]]
    for row,p in zip(eligible,holm([r["p_one_sided_raw"] for r in eligible]+[1.])[:len(eligible)]):
        row.update(p_holm=float(p),passed=bool(p<.05))
    save_json(out/"diagnostic_tests.json",{"diagnostic1":gates,"diagnostic2_3":nn_tests,
        "Lee_diagnostic1":lee_gates,"Lee_diagnostic2":lee_nn,
        "rest_gate_note":"Only passing diagnostic-1 variants eligible; omitted mixture offset is p=1 in Holm family"})
    # Every individual trajectory remains visible; no smoothing or extrapolation after early stopping.
    convergence = []
    for label,runs,receipts in [("starter",population,rp),("Lee2019_MI",lee,rl)]:
        c = concat(runs,"curves.csv")
        c.to_csv(out/f"{label}_curves.csv",index=False)
        datasets = sorted(c.dataset.unique())
        fig,axes = plt.subplots(2,len(datasets),figsize=(5*len(datasets),7),squeeze=False)
        for i,method in enumerate(["m_film","m_lora"]):
            for j,dataset in enumerate(datasets):
                ax = axes[i,j]
                for (_,_,role),g in c[(c.method==method)&(c.dataset==dataset)].groupby(["fold","seed","role"]):
                    ax.plot(g.additional_step,g.subject_mean_ce,color={"train":"#2563eb","validation":"#d97706"}[role],alpha=.2,lw=.8)
                ax.set_title(method+" / "+dataset)
                ax.set_xlabel("Additional steps"); ax.set_ylabel("Subject mean CE")
        fig.suptitle(label+": blue=train; orange=validation; each line is one fold/seed")
        fig.tight_layout(); fig.savefig(out/f"{label}_training_validation.png",dpi=170); plt.close(fig)
        for r in receipts:
            convergence.extend({"cohort":label,"fold":r["fold"],"seed":r["seed"],"method":method,**status}
                               for method,status in r["families"].items())
    pd.DataFrame(convergence).to_csv(out/"convergence.csv",index=False)
    save_json(out/"verification.json",{"status":"complete",**verified,"completed_runs":100})
    save_json(out/"run_manifest.json",{"population":rp,"diagnostic1":r1,"diagnostic3":r3,"lee":rl})
    text = ["# Stage W results", "", "All results are reported by dataset. Five seeds are averaged within each subject before inference.",
        "Lee uses the explicitly approved 26-channel readout intersection; no FCz is synthesized. No Lee hyperparameter search was run.",
        "", "## Population model",md(pd.DataFrame(comparisons)),"", "## Personal and swapped parameters",md(pd.DataFrame(d_summary)),
        "", "## Rest/task neighbours",md(pd.DataFrame(nn_summary)),"", "## Few-shot LoRA",md(pd.DataFrame(few_summary)),
        "", "## Training/validation curves", "![](starter_training_validation.png)","![](Lee2019_MI_training_validation.png)",
        "", "See convergence.csv for each run's step cap, stopping reason and plateau flags. Early stopping alone is not evidence of convergence.",
        "R2 exists only for the LoRA family and is the historical n=0 comparator. Per-dataset one-sided p values above are descriptive; pooled diagnostic families are in diagnostic_tests.json.",
        "", "All required runs and prediction checks completed. Stop after W; do not start another method or experiment."]
    (out/"report.md").write_text("\n\n".join(text),encoding="utf-8")
    print(json.dumps({"status":"complete",**verified}),flush=True)


if __name__ == "__main__":
    main()
