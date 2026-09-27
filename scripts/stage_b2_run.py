"""One fixed fold/seed: validation search, then five full-forward personal adapters."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.data import assert_split
from subject_context.model import load_backbone, state_hash
from subject_context.stage_a_common import ROOT, config, provenance, require_compute, save_json, scratch_root, sha256
from subject_context.stage_b import temporal_halves, Readout, ba, select_candidate
from subject_context.stage_b2 import config_b2, FullAdapter, fit_path

p = argparse.ArgumentParser()
p.add_argument("--job", type=int, required=True)
p.add_argument("--benchmark", action="store_true")
args = p.parse_args()
require_compute()
assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name()
torch.set_num_threads(4)
cfg = config_b2(); root = scratch_root()
fold_index = args.job // 5; seed = cfg["seeds"][args.job % 5]
fold = json.loads((ROOT / cfg["split_file"]).read_text())["folds"][fold_index]
assert_split(*[[tuple(x) for x in fold[k]] for k in ["train", "validation", "test"]])
halves_path = ROOT / cfg["halves_file"]
halves = {(r["dataset"], r["subject"]): r for r in json.loads(halves_path.read_text())["subjects"]}
out = root / "stage_b2" / ("benchmark" if args.benchmark else "runs") / f"fold-{fold_index}_seed-{seed}"
out.mkdir(parents=True, exist_ok=True)
assert not (out / "run.json").exists(), "Refuse to overwrite completed experiment"
receipt = provenance()
receipt.update(stage_b2_config_sha256=sha256(ROOT / "configs/stage_b2.yaml"),
    split_sha256=sha256(ROOT / cfg["split_file"]), halves_sha256=sha256(halves_path),
    gpu=torch.cuda.get_device_name(), fold=fold_index, seed=seed, activation_cache_used=False)
start = time.monotonic()
stage_b = root / "stage_b/runs" / f"fold-{fold_index}_seed-{seed}"
prior_selection = json.loads((stage_b / "selection.json").read_text())
kind = prior_selection["base_head"]
base = root / "baselines_v2" / f"B0_{kind}" / f"fold-{fold_index}_seed-{seed}"
checkpoint = base / "head.pt"
readout = Readout(torch.load(checkpoint, map_location="cpu", weights_only=False), kind).cuda()
backbone = load_backbone(ROOT / config()["model"]["weights"], k=4).cuda().eval()
assert sha256(ROOT / config()["model"]["weights"]) == config()["model"]["weights_sha256"]
backbone_before = state_hash(backbone); head_before = state_hash(readout)
common = json.loads((ROOT / "configs/stage_a_channels.json").read_text())["readout_common"]
original_predictions = pd.read_csv(base / "predictions.csv").set_index("trial_id")
prior_results = pd.read_csv(stage_b / "subjects.csv")
head_results = prior_results[prior_results.variant == "head"].set_index(["dataset", "subject"])
receipt.update(base_head=kind, base_checkpoint_sha256=sha256(checkpoint),
               stage_b_head_results_sha256=sha256(stage_b / "subjects.csv"))
zero_checks = []; resources = []; rows = []; paths = 0; updates = 0


def load_subject(dataset, subject):
    src = root / "processed" / dataset / f"sub-{subject:03d}"
    record = halves[(dataset, subject)]
    assert sha256(src / "trials.csv") == record["trial_table_sha256"]
    trials = pd.read_csv(src / "trials.csv")
    fit, query, counts = temporal_halves(trials)
    assert fit.tolist() == record["fit_indices"] and query.tolist() == record["query_indices"]
    y = np.load(src / "labels.npy")
    np.testing.assert_array_equal(y, trials.label)
    meta = json.loads((src / "metadata.json").read_text())
    picks = [meta["channels"].index(c) for c in common]
    signal = np.load(src / "tasks.npy")
    assert signal.dtype == np.float16 and np.isfinite(signal).all()
    eeg = torch.as_tensor(signal.astype(np.float32), device="cuda")
    with torch.no_grad():
        baseline_model = FullAdapter(backbone, readout, picks, "b0").cuda()
        # Preserve A's original batch membership as well as its prefix precision.
        baseline = torch.cat([baseline_model(chunk) for chunk in eeg.split(32)])
        for variant in cfg["gate_variants"]:
            adapter = FullAdapter(backbone, readout, picks, variant).cuda()
            zero = torch.cat([adapter(chunk) for chunk in eeg.split(32)])
            torch.testing.assert_close(zero, baseline, rtol=1e-5, atol=1e-5)
            assert torch.equal(zero.argmax(1), baseline.argmax(1))
            zero_checks.append({"dataset": dataset, "subject": subject, "variant": variant,
                                "max_logit_error": float((zero-baseline).abs().max())})
            del adapter
        pred = baseline.argmax(1).cpu().numpy()
        if (dataset, subject) in set(map(tuple, fold["test"])):
            np.testing.assert_array_equal(pred, original_predictions.loc[trials.trial_id].prediction)
            assert abs(ba(y[query], pred[query]) - head_results.loc[(dataset, subject), "b0_ba"]) < 1e-12
    return {"eeg": eeg, "picks": picks, "fit": torch.tensor(fit, device="cuda"),
            "query": torch.tensor(query, device="cuda"), "labels": torch.tensor(y, device="cuda"),
            "query_labels": y[query], "b0_ba": ba(y[query], pred[query]), "b0_pred": pred[query],
            "trial_ids": trials.trial_id.iloc[query].tolist(), "counts": counts}


validation = fold["validation"]
if args.benchmark:
    validation = [next(pair for pair in validation if pair[0] == d) for d in config()["datasets"]]
for dataset, subject in validation:
    s = load_subject(dataset, subject)
    for variant in cfg["variants"]:
        lrs = cfg["learning_rates"][variant]
        for lr in (lrs[:1] if args.benchmark else lrs):
            for wd in (cfg["weight_decays"][:1] if args.benchmark else cfg["weight_decays"]):
                steps = [20] if args.benchmark else cfg["steps"]
                result, _, resource = fit_path(backbone, readout, s, variant, lr, wd, steps, seed+subject)
                paths += 1; updates += max(steps)
                resources.append({"phase": "validation", "dataset": dataset, "subject": subject,
                    "variant": variant, "learning_rate": lr, "weight_decay": wd, "steps": max(steps), **resource})
                for step, r in result.items():
                    rows.append({"dataset": dataset, "subject": subject, "variant": variant,
                        "learning_rate": lr, "weight_decay": wd, "steps": step,
                        "b0_ba": s["b0_ba"], "ba": r["ba"], "gain": r["ba"]-s["b0_ba"]})
    pd.DataFrame(rows).to_csv(out / "validation_search.csv", index=False)
    pd.DataFrame(resources).to_csv(out / "fit_resources.csv", index=False)
    print(json.dumps({"phase": "validation", "dataset": dataset, "subject": subject,
                      "seconds": time.monotonic()-start}), flush=True)
    del s
if not args.benchmark:
    selected = {v: select_candidate([r for r in rows if r["variant"] == v]) for v in cfg["variants"]}
    save_json(out / "selection.json", {"base_head": kind, "parameters": selected})
    subjects = []; predictions = []
    for dataset, subject in fold["test"]:
        s = load_subject(dataset, subject); params = {}
        for variant in cfg["variants"]:
            hp = selected[variant]
            result, raw, resource = fit_path(backbone, readout, s, variant,
                hp["learning_rate"], hp["weight_decay"], [hp["steps"]], seed+subject)
            paths += 1; updates += hp["steps"]; r = result[hp["steps"]]
            resources.append({"phase": "test", "dataset": dataset, "subject": subject,
                "variant": variant, "learning_rate": hp["learning_rate"],
                "weight_decay": hp["weight_decay"], "steps": hp["steps"], **resource})
            head_ba = float(head_results.loc[(dataset, subject), "ba"])
            subjects.append({"dataset": dataset, "subject": subject, "fold": fold_index, "seed": seed,
                "base_head": kind, "variant": variant, "b0_ba": s["b0_ba"], "ba": r["ba"],
                "gain": r["ba"]-s["b0_ba"], "head_ba": head_ba, "gain_vs_head": r["ba"]-head_ba, **s["counts"]})
            for tid, y, b, pred in zip(s["trial_ids"], s["query_labels"], s["b0_pred"], r["predictions"]):
                predictions.append({"dataset": dataset, "subject": subject, "fold": fold_index, "seed": seed,
                    "variant": variant, "trial_id": tid, "label": int(y), "b0_prediction": int(b), "prediction": int(pred)})
            params[variant] = raw
        torch.save(params, out / f"adapter-{dataset}-{subject:03d}.pt")
        pd.DataFrame(subjects).to_csv(out / "subjects.csv", index=False)
        pd.DataFrame(resources).to_csv(out / "fit_resources.csv", index=False)
        print(json.dumps({"phase": "test", "dataset": dataset, "subject": subject,
                          "seconds": time.monotonic()-start}), flush=True)
        del s
    pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
assert state_hash(backbone) == backbone_before and state_hash(readout) == head_before
assert all(p.grad is None for p in backbone.parameters())
save_json(out / "zero_checks.json", zero_checks)
receipt.update(status="complete", seconds=time.monotonic()-start,
    peak_gpu_bytes=max(r["peak_gpu_bytes"] for r in resources), optimization_paths=paths,
    optimization_steps=updates, backbone_unchanged=True, base_head_unchanged=True)
save_json(out / "run.json", receipt)
print(json.dumps(receipt), flush=True)
