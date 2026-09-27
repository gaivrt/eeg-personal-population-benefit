"""One fold/seed of Stage C: train M-FiLM/M-LoRA on training subjects, select on validation, score test."""
import argparse
import copy
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
from subject_context.stage_b2 import fit_path
from subject_context.stage_c import ContextModel, backbone_features, config_c
from subject_context.stage_c_context import TangentFeatures, band_covariances, derangement, t3a_predict
from subject_context.stage_c_train import load_snapshot, score, select_config, snapshot, train

p = argparse.ArgumentParser()
p.add_argument("--job", type=int, required=True)
p.add_argument("--benchmark", action="store_true")
p.add_argument("--tag", default="", help="benchmark only: separate outputs of co-located processes")
args = p.parse_args()
assert args.benchmark or not args.tag
require_compute()
assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name()
torch.set_num_threads(4)
cfg = config_c(); root = scratch_root()
if args.benchmark:
    cfg["training"]["episodes"] = [100]
fold_index = args.job // 5; seed = cfg["seeds"][args.job % 5]
fold = json.loads((ROOT / cfg["split_file"]).read_text())["folds"][fold_index]
split = {k: [tuple(x) for x in fold[k]] for k in ["train", "validation", "test"]}
assert_split(split["train"], split["validation"], split["test"])
halves = {(r["dataset"], r["subject"]): r for r in json.loads((ROOT / cfg["halves_file"]).read_text())["subjects"]}
out = root / "stage_c" / (f"benchmark/{args.tag}" if args.benchmark else "runs") / f"fold-{fold_index}_seed-{seed}"
out.mkdir(parents=True, exist_ok=True)
assert not (out / "run.json").exists(), "Refuse to overwrite completed experiment"
receipt = provenance()
receipt.update(stage_c_config_sha256=sha256(ROOT / "configs/stage_c.yaml"),
    split_sha256=sha256(ROOT / cfg["split_file"]), halves_sha256=sha256(ROOT / cfg["halves_file"]),
    gpu=torch.cuda.get_device_name(), fold=fold_index, seed=seed, activation_cache_used=False)
start = time.monotonic()
kind = json.loads((root / "stage_b/runs" / f"fold-{fold_index}_seed-{seed}" / "selection.json").read_text())["base_head"]
base = root / "baselines_v2" / f"B0_{kind}" / f"fold-{fold_index}_seed-{seed}"
readout = Readout(torch.load(base / "head.pt", map_location="cpu", weights_only=False), kind).cuda()
backbone = load_backbone(ROOT / config()["model"]["weights"], k=4).cuda().eval()
assert sha256(ROOT / config()["model"]["weights"]) == config()["model"]["weights_sha256"]
frozen = state_hash(backbone), state_hash(readout)
common = json.loads((ROOT / "configs/stage_a_channels.json").read_text())["readout_common"]
original_predictions = pd.read_csv(base / "predictions.csv").set_index("trial_id")
receipt.update(base_head=kind, base_checkpoint_sha256=sha256(base / "head.pt"))
c = cfg["context"]; minimum = int(np.ceil(c["min_seconds"] / c["segment_seconds"]))


@torch.no_grad()
def pooled(x, picks):
    return torch.cat([backbone_features(backbone.model, x[i:i+32].float(), picks) for i in range(0, len(x), 32)])


def load_subject(dataset, subject, role):
    src = root / "processed" / dataset / f"sub-{subject:03d}"
    record = halves[(dataset, subject)]
    assert sha256(src / "trials.csv") == record["trial_table_sha256"]
    trials = pd.read_csv(src / "trials.csv")
    fit, query, counts = temporal_halves(trials)
    assert fit.tolist() == record["fit_indices"] and query.tolist() == record["query_indices"]
    meta = json.loads((src / "metadata.json").read_text())
    rest_meta = meta["rest"][c["eye_state"]]
    # Context precedes every labeled trial in the recorded protocol and never shares a trial.
    assert rest_meta["protocol_order"] < trials.protocol_order.min()
    y = np.load(src / "labels.npy"); np.testing.assert_array_equal(y, trials.label)
    picks = [meta["channels"].index(ch) for ch in common]
    eeg = torch.as_tensor(np.load(src / "tasks.npy"), device="cuda")
    rest = np.load(src / "rest_open.npy")[np.load(src / "rest_open_valid.npy")]
    assert eeg.dtype == torch.float16 and rest.dtype == np.float16 and np.isfinite(rest).all()
    assert len(rest) * c["segment_seconds"] == rest_meta["usable_seconds"] and len(rest) >= minimum
    features = pooled(eeg, picks)
    b0 = readout(features).argmax(1).cpu().numpy()
    if role == "test":
        np.testing.assert_array_equal(b0, original_predictions.loc[trials.trial_id].prediction)
    rest_pooled = pooled(torch.as_tensor(rest, device="cuda"), picks)
    return {"key": (dataset, subject), "eeg": eeg, "picks": picks, "labels": torch.tensor(y, device="cuda"),
            "fit": torch.tensor(fit, device="cuda"), "query": torch.tensor(query, device="cuda"),
            "query_labels": y[query], "features": features, "rest_pooled": rest_pooled,
            "emb": readout.normalize(rest_pooled), "bands": band_covariances(rest[:, picks].astype(np.float64),
            c["bands_hz"], c["filter_order"]), "b0": b0, "b0_ba": ba(y[query], b0[query]),
            "b0_ba_all": ba(y, b0), "trial_ids": trials.trial_id.tolist(), "counts": counts}


roles = {p: r for r in ["train", "validation", "test"] for p in split[r]}
if args.benchmark:
    # One validation and two test subjects per dataset: B4 needs a same-dataset donor.
    keep = [p for d in config()["datasets"] for r, n in [("validation", 1), ("test", 2)]
            for p in [q for q in split[r] if q[0] == d][:n]]
    roles = {p: r for p, r in roles.items() if r == "train" or p in keep}
subjects = {p: load_subject(*p, r) for p, r in roles.items()}
tangent = TangentFeatures().fit([subjects[p]["bands"] for p in split["train"]])
for s in subjects.values():
    s["tan"] = torch.as_tensor(tangent(s.pop("bands")), device="cuda")
group = {r: [subjects[p] for p in subjects if roles[p] == r] for r in ["train", "validation", "test"]}
receipt["load_seconds"] = time.monotonic() - start
print(json.dumps({"loaded": {r: len(v) for r, v in group.items()}, "seconds": receipt["load_seconds"]}), flush=True)
t = cfg["training"]
grid = [{"learning_rate": lr, "lambda": lam, "margin": m}
        for lr, lam, m in itertools.product(t["learning_rates"], t["lambdas"], t["margins"])]
grid += [{"learning_rate": lr, "lambda": t["ablation_lambda"], "margin": 0.} for lr in t["learning_rates"]]
if args.benchmark:
    grid = [grid[0], grid[-1]]
search, resources, subjects_out, predictions, selection = [], [], [], [], {"base_head": kind}
width = group["train"][0]["emb"].shape[1], group["train"][0]["tan"].shape[1]


def record(s, condition, pred, extra=None):
    """pred covers all trials, or only the second half for label-using B5."""
    y = s["labels"].cpu().numpy(); q = s["query"].cpu().numpy()
    everything = len(pred) == len(y)
    assert everything or len(pred) == len(q)
    half = pred[q] if everything else pred
    row = {"dataset": s["key"][0], "subject": s["key"][1], "fold": fold_index, "seed": seed, "base_head": kind,
           "condition": condition, "b0_ba": s["b0_ba"], "ba": ba(y[q], half), "gain": ba(y[q], half) - s["b0_ba"],
           **({"b0_ba_all": s["b0_ba_all"], "ba_all": ba(y, pred), "gain_all": ba(y, pred) - s["b0_ba_all"]}
              if everything else {}), **(extra or {}), **s["counts"]}
    subjects_out.append(row)
    rows = np.arange(len(y)) if everything else q
    query_rows = set(q.tolist())
    predictions.extend({"dataset": s["key"][0], "subject": s["key"][1], "fold": fold_index, "seed": seed,
                        "condition": condition, "trial_id": s["trial_ids"][i], "label": int(y[i]),
                        "query": int(i) in query_rows, "prediction": int(pr)} for i, pr in zip(rows, pred))


for s in group["test"]:
    record(s, "B0", s["b0"])
donors = derangement([s["key"] for s in group["test"]], np.random.default_rng(seed))
for method in cfg["methods"]:
    states = {}
    for hp in grid:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); began = time.monotonic()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = ContextModel(backbone, readout, method, *width, cfg).cuda()
        if hp is grid[0]:
            # Untrained generator: own, default and donor contexts all reproduce B0 exactly.
            for s in group["test"]:
                for kwargs in [{}, {"default": True}, {"context": subjects[donors[s["key"]]]}]:
                    np.testing.assert_array_equal(score(model, s, **kwargs)["predictions"], s["b0"])
            receipt[f"{method}_initial_equals_b0"] = True

        def checkpoint(episode):
            states[(tuple(hp.values()), episode)] = snapshot(model)
            for s in group["validation"]:
                r = score(model, s)
                search.append({"method": method, **hp, "episodes": episode, "dataset": s["key"][0],
                               "subject": s["key"][1], "b0_ba": s["b0_ba"], "ba": r["ba"], "gain": r["ba"] - s["b0_ba"]})
        train(model, group["train"], hp, cfg, seed, checkpoint)
        torch.cuda.synchronize()
        resources.append({"method": method, **hp, "episodes": max(t["episodes"]), "seconds": time.monotonic() - began,
                          "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                          "trainable_parameters": sum(q.numel() for q in model.trainable())})
        pd.DataFrame(search).to_csv(out / "validation_search.csv", index=False)
        pd.DataFrame(resources).to_csv(out / "training_resources.csv", index=False)
        print(json.dumps(resources[-1]), flush=True)
    rows = [r for r in search if r["method"] == method]
    for name, keep in [("main", lambda r: r["lambda"] > 0), ("lambda0", lambda r: r["lambda"] == 0)]:
        chosen = select_config([r for r in rows if keep(r)])
        selection[f"{method}_{name}"] = chosen
        load_snapshot(model, states[((chosen["learning_rate"], chosen["lambda"], chosen["margin"]), chosen["episodes"])])
        torch.save(snapshot(model), out / f"model-{method}-{name}.pt")
        label = "M_" + method[2:] + ("" if name == "main" else "_lambda0")
        for s in group["test"]:
            record(s, label, score(model, s)["predictions"])
            if name == "main":
                record(s, "B3_" + method[2:], score(model, s, default=True)["predictions"])
                donor = subjects[donors[s["key"]]]
                record(s, "B4_" + method[2:], score(model, s, context=donor)["predictions"],
                       {"donor_subject": donor["key"][1]})
    del model, states
    torch.cuda.empty_cache()

# B2: T3A with the same rest segments as supports.
t3a_rows = [{"filter": f, "gain": ba(s["query_labels"], t3a_predict(readout, s["rest_pooled"],
             s["features"], f)[s["query"].cpu().numpy()]) - s["b0_ba"]}
            for f in cfg["t3a_filter_sizes"] for s in group["validation"]]
ranked = pd.DataFrame(t3a_rows).groupby("filter", sort=False).gain.agg(["median", "mean"])
selection["t3a_filter"] = ranked.sort_values(["median", "mean"], ascending=False, kind="stable").index[0]
for s in group["test"]:
    record(s, "B2_T3A", t3a_predict(readout, s["rest_pooled"], s["features"], selection["t3a_filter"]))

# B5: full-layer FiLM from the first n labeled trials of the first half, selected per n on validation.
b5 = cfg["b5"]; selection["b5"] = {}
for n in b5["shots"][:1] if args.benchmark else b5["shots"]:
    view = lambda s: {"eeg": s["eeg"].float(), "picks": s["picks"], "fit": s["fit"][:n], "query": s["query"],
                      "labels": s["labels"], "query_labels": s["query_labels"]}
    rows = []
    for s in [s for s in group["validation"] if len(s["fit"]) >= n]:
        for lr, wd in itertools.product(b5["learning_rates"], b5["weight_decays"]):
            result, _, _ = fit_path(backbone, readout, view(s), "film_all", lr, wd, b5["steps"], seed + s["key"][1])
            rows += [{"learning_rate": lr, "weight_decay": wd, "steps": k, "gain": r["ba"] - s["b0_ba"]}
                     for k, r in result.items()]
    if not rows:
        continue
    hp = selection["b5"][n] = select_candidate(rows)
    for s in [s for s in group["test"] if len(s["fit"]) >= n]:
        result, _, _ = fit_path(backbone, readout, view(s), "film_all", hp["learning_rate"],
                                hp["weight_decay"], [hp["steps"]], seed + s["key"][1])
        record(s, f"B5_n{n}", result[hp["steps"]]["predictions"])
    print(json.dumps({"b5": n, "seconds": time.monotonic() - start}), flush=True)

pd.DataFrame(subjects_out).to_csv(out / "subjects.csv", index=False)
pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
save_json(out / "selection.json", selection)
assert (state_hash(backbone), state_hash(readout)) == frozen
assert all(q.grad is None for q in backbone.parameters())
receipt.update(status="complete", seconds=time.monotonic() - start, donors={f"{d}:{s}": v[1] for (d, s), v in donors.items()},
               peak_gpu_bytes=max(r["peak_gpu_bytes"] for r in resources), backbone_unchanged=True, base_head_unchanged=True)
save_json(out / "run.json", receipt)
print(json.dumps({"status": "complete", "seconds": receipt["seconds"]}), flush=True)
