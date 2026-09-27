"""One fold/seed of Stage C3: unlabeled-task neighbours over C2 personal parameters, and few-shot curves."""
import argparse
import itertools
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
from subject_context.stage_b import Readout, ba, select_candidate, temporal_halves
from subject_context.stage_c import ContextModel, backbone_features, config_c
from subject_context.stage_c_context import TangentFeatures, band_covariances
from subject_context.stage_c_train import load_snapshot
from subject_context.stage_c2 import PersonalModel, fit, predict
from subject_context.stage_c3 import average_personal, config_c3, nearest, random_sets

p = argparse.ArgumentParser()
p.add_argument("--job", type=int, required=True)
p.add_argument("--benchmark", action="store_true")
p.add_argument("--stage-w", action="store_true", help="Same diagnostics with G, and LoRA-only few-shot curves")
args = p.parse_args()
require_compute()
assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name()
torch.set_num_threads(4)
cfg, c_cfg = config_c3(), config_c(); root = scratch_root()
d3, fs = cfg["diagnostic3"], cfg["few_shot"]
fold_index = args.job // 5; seed = c_cfg["seeds"][args.job % 5]
fold = json.loads((ROOT / c_cfg["split_file"]).read_text())["folds"][fold_index]
split = {k: [tuple(x) for x in fold[k]] for k in ["train", "validation", "test"]}
assert_split(split["train"], split["validation"], split["test"])
halves = {(r["dataset"], r["subject"]): r for r in json.loads((ROOT / c_cfg["halves_file"]).read_text())["subjects"]}
name = f"fold-{fold_index}_seed-{seed}"
out = root / ("stage_w/diagnostic3" if args.stage_w else "stage_c3") / ("benchmark" if args.benchmark else "runs") / name
out.mkdir(parents=True, exist_ok=True)
assert not (out / "run.json").exists(), "Refuse to overwrite completed experiment"
c2 = root / ("stage_w/diagnostic1/runs" if args.stage_w else "stage_c2/runs") / name
assert json.loads((c2 / "run.json").read_text())["status"] == "complete"
receipt = provenance()
receipt.update(stage_c3_config_sha256=sha256(ROOT / "configs/stage_c3.yaml"), fold=fold_index, seed=seed,
               gpu=torch.cuda.get_device_name(),
               **{f"c2_{v}_sha256": sha256(c2 / f"personal-{v}.pt") for v in cfg["variants"]})
start = time.monotonic()
if args.stage_w:
    receipt.update(start_label="G", stage_w_config_sha256=sha256(ROOT / "configs/stage_w.yaml"))
kind = json.loads((root / "stage_b/runs" / name / "selection.json").read_text())["base_head"]
readout = Readout(torch.load(root / "baselines_v2" / f"B0_{kind}" / name / "head.pt", map_location="cpu",
                             weights_only=False), kind).cuda()
backbone = load_backbone(ROOT / config()["model"]["weights"], k=4).cuda().eval()
common = json.loads((ROOT / "configs/stage_a_channels.json").read_text())["readout_common"]
c = c_cfg["context"]
bases = {}
for method in ["m_film", "m_lora"]:
    bases[method] = ContextModel(backbone, readout, method, 27 * 200, 2 * 27 * 28 // 2, c_cfg).cuda()
    source = root / ("stage_w/population/runs" if args.stage_w else "stage_c/runs") / name
    load_snapshot(bases[method], torch.load(source / f"model-{method}-main.pt", map_location="cuda"))
frozen = state_hash(backbone), {m: state_hash(b) for m, b in bases.items()}
c2_rows = pd.read_csv(c2 / "subjects.csv").set_index(["variant", "dataset", "subject"])
c2_matrix = pd.read_csv(c2 / "swap_matrix.csv").set_index(["variant", "dataset", "subject", "donor"]).ba
rest_features = dict(np.load(c2 / "rest_features.npz"))


def load_subject(dataset, subject, role):
    src = root / "processed" / dataset / f"sub-{subject:03d}"
    record = halves[(dataset, subject)]
    trials = pd.read_csv(src / "trials.csv")
    assert sha256(src / "trials.csv") == record["trial_table_sha256"]
    fit_rows, query, counts = temporal_halves(trials)
    assert fit_rows.tolist() == record["fit_indices"] and query.tolist() == record["query_indices"]
    meta = json.loads((src / "metadata.json").read_text())
    picks = [meta["channels"].index(ch) for ch in common]
    assert trials.quality_valid.iloc[fit_rows].all()  # all trials valid under Stage A's finite-only policy
    tasks = np.load(src / "tasks.npy", mmap_mode="r" if role == "train" else None)
    # Unlabeled first-half signals only; labels are never read for these features.
    bands = band_covariances(tasks[fit_rows][:, picks].astype(np.float64), c["bands_hz"], c["filter_order"])
    if role == "train":
        return {"bands": bands}
    y = np.load(src / "labels.npy"); np.testing.assert_array_equal(y, trials.label)
    s = {"key": (dataset, subject), "eeg": torch.as_tensor(tasks, device="cuda"), "picks": picks,
         "labels": torch.tensor(y, device="cuda"), "fit": torch.tensor(fit_rows, device="cuda"),
         "query": torch.tensor(query, device="cuda"), "query_labels": y[query], "bands": bands, "counts": counts,
         "fit_labels": y[fit_rows], "trial_ids": trials.trial_id.tolist()}
    with torch.no_grad():
        first = s["eeg"][s["fit"]]
        pooled = torch.cat([backbone_features(backbone.model, first[i:i+32].float(), picks) for i in range(0, len(first), 32)])
        s["emb_mean"] = readout.normalize(pooled).mean(0).cpu().numpy()
    return s


roles = {p: r for r in ["train", "validation", "test"] for p in split[r]}
if args.benchmark:  # k=3 needs at least four test subjects per dataset
    keep = [p for d in config()["datasets"] for r, n in [("validation", 1), ("test", 4)]
            for p in [q for q in split[r] if q[0] == d][:n]]
    roles = {p: r for p, r in roles.items() if r == "train" or p in keep}
subjects = {p: load_subject(*p, r) for p, r in roles.items()}
tangent = TangentFeatures().fit([subjects[p]["bands"] for p in split["train"]])
groups = {r: [subjects[p] for p in subjects if roles[p] == r] for r in ["validation", "test"]}
task_features = {}
for s in groups["validation"] + groups["test"]:
    tan_mean = tangent(s.pop("bands")).mean(0)
    task_features[f"{s['key'][0]}:{s['key'][1]}"] = np.concatenate(
        [s.pop("emb_mean") / np.sqrt(27 * 200), tan_mean / np.sqrt(len(tan_mean))])
np.savez(out / "task_features.npz", **task_features)
receipt["load_seconds"] = time.monotonic() - start
print(json.dumps({"loaded": {r: len(v) for r, v in groups.items()}, "seconds": receipt["load_seconds"]}), flush=True)

models = {"film_offset": PersonalModel(bases["m_film"], "film_offset"),
          "lora8": PersonalModel(bases["m_lora"], "lora8", 8)}
merged_lora = PersonalModel(bases["m_lora"], "lora8", 8 * d3["secondary_k"])
checks = []
prediction_rows = []


def record_predictions(s, variant, condition, pred, shots=0, donors=""):
    if args.stage_w:
        prediction_rows.extend({"variant": variant, "dataset": s["key"][0], "subject": s["key"][1],
            "condition": condition, "shots": shots, "donors": donors, "trial_id": s["trial_ids"][i],
            "label": int(s["query_labels"][j]), "prediction": int(pred[j])}
            for j, i in enumerate(s["query"].cpu().tolist()))


def check(name, variant, key, expected, got):
    """Record agreement with C2 instead of aborting: rounding-level GPU differences must not kill a run."""
    checks.append({"check": name, "variant": variant, "dataset": key[0], "subject": key[1],
                   "expected": float(expected), "got": float(got), "difference": float(got - expected)})


# B3 through C2's own path (zero FiLM offset / zero mixture offset), compared with C2's record.
b3, lora_zero = {}, []
reference = {"film_offset": models["film_offset"], "lora8": PersonalModel(bases["m_lora"], "mix_offset")}
for variant, model in reference.items():
    model.reset()
    for s in groups["validation"] + groups["test"]:
        pred = predict(model, s, s["query"])
        if roles[s["key"]] == "test":
            record_predictions(s, variant, "G", pred)
        b3[(variant, s["key"])] = ba(s["query_labels"], pred)
        if roles[s["key"]] == "test":
            check("b3_vs_c2", variant, s["key"], c2_rows.loc[(variant, *s["key"]), "b3_ba"], b3[(variant, s["key"])])
        if variant == "lora8":  # zero personal LoRA takes a different GPU path; record, don't assert
            models["lora8"].reset()
            with torch.no_grad():
                a = torch.cat([model(s["eeg"][t].float(), s["picks"]) for t in s["query"].split(32)])
                b = torch.cat([models["lora8"](s["eeg"][t].float(), s["picks"]) for t in s["query"].split(32)])
            lora_zero.append({"dataset": s["key"][0], "subject": s["key"][1], "max_logit_difference": float((a - b).abs().max()),
                              "prediction_agreement": float((a.argmax(1) == b.argmax(1)).float().mean())})
pd.DataFrame(lora_zero).to_csv(out / "lora_zero_vs_b3.csv", index=False)

# Diagnostic 3 (task features) and the matching rest-feature supplement.
test_keys = [s["key"] for s in groups["test"]]
params = {v: {tuple([k.split(":")[0], int(k.split(":")[1])]): val
              for k, val in torch.load(c2 / f"personal-{v}.pt", map_location="cuda").items()} for v in cfg["variants"]}
rng = np.random.default_rng(seed + 1000)
random_k3 = {key: random_sets(rng, [t for t in test_keys if t[0] == key[0] and t != key], d3["secondary_k"],
                              d3["random_draws"]) for key in test_keys}
assert all(isinstance(d, tuple) and isinstance(d[1], int) for sets in random_k3.values() for draw in sets for d in draw)


def score_with(variant, s, donors):
    model = models[variant] if len(donors) == 1 or variant == "film_offset" else merged_lora
    model.load_personal(average_personal(variant, [params[variant][d] for d in donors]))
    pred = predict(model, s, s["query"])
    record_predictions(s, variant, "transfer", pred, donors=" ".join(str(d[1]) for d in donors))
    return ba(s["query_labels"], pred)


neighbours = []
for variant in cfg["variants"]:
    for s in groups["test"]:
        own = score_with(variant, s, [s["key"]])
        check("own_vs_c2", variant, s["key"], c2_rows.loc[(variant, *s["key"]), "own_ba"], own)
        others = [t for t in test_keys if t[0] == s["key"][0] and t != s["key"]]
        k3_random = float(np.mean([score_with(variant, s, draw) for draw in random_k3[s["key"]]]))
        for source, feats in [("task", task_features), ("rest", rest_features)]:
            by_key = {(d, n): feats[f"{d}:{n}"] for d, n in [s["key"], *others]}  # numeric tie-break on subject id
            for k in [d3["primary_k"], d3["secondary_k"]]:
                chosen = nearest(by_key, s["key"], others, k)
                nn_ba = score_with(variant, s, chosen)
                if k == 1:
                    check(f"nearest_{source}_vs_c2_matrix", variant, s["key"],
                          c2_matrix.loc[(variant, *s["key"], chosen[0][1])], nn_ba)
                random_ba = c2_rows.loc[(variant, *s["key"]), "swap_ba"] if k == 1 else k3_random
                neighbours.append({"variant": variant, "source": source, "k": k, "dataset": s["key"][0],
                                   "subject": s["key"][1], "fold": fold_index, "seed": seed,
                                   "neighbours": " ".join(str(n) for _, n in chosen), "nearest_ba": nn_ba,
                                   "random_ba": random_ba, "difference": nn_ba - random_ba})
    pd.DataFrame(neighbours).to_csv(out / "neighbours.csv", index=False)
    print(json.dumps({"diagnostic3": variant, "seconds": time.monotonic() - start}), flush=True)

# Few-shot curves from B3: earliest n first-half labels.
shots = fs["shots"][:1] if args.benchmark else fs["shots"]
steps = [20] if args.benchmark else fs["steps"]
curve, search, resources, selection = [], [], [], {}
for variant, n in itertools.product(["lora8"] if args.stage_w else cfg["variants"], shots):
    model = models[variant]
    view = lambda s: {**s, "fit": s["fit"][:n]}
    rows = []
    for s in [s for s in groups["validation"] if len(s["fit"]) >= n]:
        for lr, wd in itertools.product(fs["learning_rates"][variant], fs["weight_decays"]):
            result, _, resource = fit(model, view(s), lr, wd, steps, seed + s["key"][1])
            resources.append({"phase": "validation", "variant": variant, "shots": n, **resource})
            for k, r in result.items():
                rows.append({"learning_rate": lr, "weight_decay": wd, "steps": k, "gain": r["ba"] - b3[(variant, s["key"])]})
    search += [{"variant": variant, "shots": n, **r} for r in rows]
    if not rows:
        continue
    hp = selection[f"{variant}_n{n}"] = select_candidate(rows)
    for s in [s for s in groups["test"] if len(s["fit"]) >= n]:
        result, _, resource = fit(model, view(s), hp["learning_rate"], hp["weight_decay"], [hp["steps"]], seed + s["key"][1])
        resources.append({"phase": "test", "variant": variant, "shots": n, **resource})
        r = result[hp["steps"]]
        record_predictions(s, variant, "few_shot", r["predictions"], shots=n)
        full = c2_rows.loc[(variant, *s["key"])]
        curve.append({"variant": variant, "shots": n, "dataset": s["key"][0], "subject": s["key"][1],
                      "fold": fold_index, "seed": seed, "b3_ba": b3[(variant, s["key"])], "ba": r["ba"],
                      "gain": r["ba"] - b3[(variant, s["key"])], "full_gain": full.own_gain,
                      "upper_bound": full.upper_bound, "one_class_only": len(set(s["fit_labels"][:n])) < 2})
    pd.DataFrame(curve).to_csv(out / "few_shot.csv", index=False)
    pd.DataFrame(search).to_csv(out / "few_shot_validation.csv", index=False)
    pd.DataFrame(resources).to_csv(out / "fit_resources.csv", index=False)
    print(json.dumps({"variant": variant, "shots": n, "selected": hp, "seconds": time.monotonic() - start}), flush=True)
save_json(out / "selection.json", selection)
if args.stage_w:
    pd.DataFrame(prediction_rows).to_csv(out / "predictions.csv.gz", index=False)
checks = pd.DataFrame(checks); checks.to_csv(out / "consistency_checks.csv", index=False)
mismatches = int((checks.difference.abs() > 1e-12).sum())
assert state_hash(backbone) == frozen[0] and {m: state_hash(b) for m, b in bases.items()} == frozen[1]
receipt.update(status="complete", seconds=time.monotonic() - start, backbone_unchanged=True,
               stage_c_models_unchanged=True, c2_consistency_checks=len(checks), c2_consistency_mismatches=mismatches,
               peak_gpu_bytes=max(r["peak_gpu_bytes"] for r in resources))
save_json(out / "run.json", receipt)
print(json.dumps({"status": "complete", "seconds": receipt["seconds"]}), flush=True)
