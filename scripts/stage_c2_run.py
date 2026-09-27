"""One fold/seed of Stage C2: personal parameters on B3, own vs swapped, full same-dataset swap matrix."""
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
from subject_context.stage_c2 import PersonalModel, config_c2, fit, predict

p = argparse.ArgumentParser()
p.add_argument("--job", type=int, required=True)
p.add_argument("--benchmark", action="store_true")
p.add_argument("--stage-w", action="store_true", help="Same diagnostics, with the corresponding G checkpoint")
args = p.parse_args()
require_compute()
assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name()
torch.set_num_threads(4)
cfg, c_cfg = config_c2(), config_c(); root = scratch_root()
if args.stage_w:
    cfg["variants"] = ["film_offset", "lora8"]
fold_index = args.job // 5; seed = c_cfg["seeds"][args.job % 5]
fold = json.loads((ROOT / c_cfg["split_file"]).read_text())["folds"][fold_index]
split = {k: [tuple(x) for x in fold[k]] for k in ["train", "validation", "test"]}
assert_split(split["train"], split["validation"], split["test"])
halves = {(r["dataset"], r["subject"]): r for r in json.loads((ROOT / c_cfg["halves_file"]).read_text())["subjects"]}
name = f"fold-{fold_index}_seed-{seed}"
out = root / ("stage_w/diagnostic1" if args.stage_w else "stage_c2") / ("benchmark" if args.benchmark else "runs") / name
out.mkdir(parents=True, exist_ok=True)
assert not (out / "run.json").exists(), "Refuse to overwrite completed experiment"
stage_c = root / ("stage_w/population/runs" if args.stage_w else "stage_c/runs") / name
assert json.loads((stage_c / "run.json").read_text())["status"] == "complete"
receipt = provenance()
receipt.update(stage_c2_config_sha256=sha256(ROOT / "configs/stage_c2.yaml"), fold=fold_index, seed=seed,
               gpu=torch.cuda.get_device_name(), stage_c_run=json.loads((stage_c / "run.json").read_text())["git_commit"])
start = time.monotonic()
if args.stage_w:
    receipt.update(start_label="G", stage_w_config_sha256=sha256(ROOT / "configs/stage_w.yaml"))
kind = json.loads((root / "stage_b/runs" / name / "selection.json").read_text())["base_head"]
base_dir = root / "baselines_v2" / f"B0_{kind}" / name
readout = Readout(torch.load(base_dir / "head.pt", map_location="cpu", weights_only=False), kind).cuda()
backbone = load_backbone(ROOT / config()["model"]["weights"], k=4).cuda().eval()
common = json.loads((ROOT / "configs/stage_a_channels.json").read_text())["readout_common"]
c = c_cfg["context"]
bases = {}
for method in c_cfg["methods"]:
    model = ContextModel(backbone, readout, method, 27 * 200, 2 * 27 * 28 // 2, c_cfg).cuda()
    state = torch.load(stage_c / f"model-{method}-main.pt", map_location="cuda")
    receipt[f"{method}_start_sha256"] = sha256(stage_c / f"model-{method}-main.pt")
    load_snapshot(model, state)
    bases[method] = model
frozen = state_hash(backbone), {m: state_hash(b) for m, b in bases.items()}
c_predictions = pd.read_csv(stage_c / "predictions.csv")
if args.stage_w:
    c_predictions["condition"] = c_predictions.method.map({"m_film": "B3_film", "m_lora": "B3_lora"})
else:
    c_predictions = c_predictions.query("condition in ['B3_film', 'B3_lora']")


def load_subject(dataset, subject, role):
    src = root / "processed" / dataset / f"sub-{subject:03d}"
    record = halves[(dataset, subject)]
    trials = pd.read_csv(src / "trials.csv")
    assert sha256(src / "trials.csv") == record["trial_table_sha256"]
    fit_rows, query, counts = temporal_halves(trials)
    assert fit_rows.tolist() == record["fit_indices"] and query.tolist() == record["query_indices"]
    meta = json.loads((src / "metadata.json").read_text())
    picks = [meta["channels"].index(ch) for ch in common]
    rest = np.load(src / "rest_open.npy")[np.load(src / "rest_open_valid.npy")]
    bands = band_covariances(rest[:, picks].astype(np.float64), c["bands_hz"], c["filter_order"])
    if role == "train":
        return {"bands": bands}
    y = np.load(src / "labels.npy"); np.testing.assert_array_equal(y, trials.label)
    s = {"key": (dataset, subject), "eeg": torch.as_tensor(np.load(src / "tasks.npy"), device="cuda"),
         "picks": picks, "labels": torch.tensor(y, device="cuda"), "fit": torch.tensor(fit_rows, device="cuda"),
         "query": torch.tensor(query, device="cuda"), "query_labels": y[query], "bands": bands,
         "trial_ids": trials.trial_id.tolist(), "counts": counts}
    with torch.no_grad():
        rest_t = torch.as_tensor(rest, device="cuda")
        pooled = torch.cat([backbone_features(backbone.model, rest_t[i:i+32].float(), picks) for i in range(0, len(rest_t), 32)])
        s["emb_mean"] = readout.normalize(pooled).mean(0).cpu().numpy()
    return s


roles = {p: r for r in ["train", "validation", "test"] for p in split[r]}
if args.benchmark:
    keep = [p for d in config()["datasets"] for r, n in [("validation", 1), ("test", 2)]
            for p in [q for q in split[r] if q[0] == d][:n]]
    roles = {p: r for p, r in roles.items() if r == "train" or p in keep}
subjects = {p: load_subject(*p, r) for p, r in roles.items()}
tangent = TangentFeatures().fit([subjects[p]["bands"] for p in split["train"]])
groups = {r: [subjects[p] for p in subjects if roles[p] == r] for r in ["validation", "test"]}
features = {}
for s in groups["validation"] + groups["test"]:
    tan_mean = tangent(s.pop("bands")).mean(0)
    features[f"{s['key'][0]}:{s['key'][1]}"] = np.concatenate(
        [s.pop("emb_mean") / np.sqrt(27 * 200), tan_mean / np.sqrt(len(tan_mean))])
np.savez(out / "rest_features.npz", **features)
receipt["load_seconds"] = time.monotonic() - start
print(json.dumps({"loaded": {r: len(v) for r, v in groups.items()}, "seconds": receipt["load_seconds"]}), flush=True)

# (i) B3 must reproduce Stage C exactly.
b3 = {}
for method, suffix in [("m_film", "film"), ("m_lora", "lora")]:
    zero = PersonalModel(bases[method], "film_offset" if method == "m_film" else "mix_offset")
    for s in groups["validation"] + groups["test"]:
        everything = torch.arange(len(s["labels"]), device="cuda")
        pred = predict(zero, s, everything)
        if roles[s["key"]] == "test":
            saved = c_predictions[(c_predictions.condition == "B3_" + suffix) & (c_predictions.dataset == s["key"][0])
                                  & (c_predictions.subject == s["key"][1])].set_index("trial_id").prediction
            np.testing.assert_array_equal(pred, saved.loc[s["trial_ids"]].to_numpy())
        q = s["query"].cpu().numpy()
        b3[(method, s["key"])] = ba(s["labels"].cpu().numpy()[q], pred[q])
    del zero
receipt["start_reproduces_source" if args.stage_w else "b3_reproduces_stage_c"] = True

draw_rng = np.random.default_rng(seed)
draws = {}
for s in groups["test"]:
    others = [t["key"] for t in groups["test"] if t["key"][0] == s["key"][0] and t["key"] != s["key"]]
    draws[s["key"]] = [others[i] for i in draw_rng.integers(len(others), size=cfg["swap_draws"])]
steps = [20] if args.benchmark else cfg["steps"]
rows, search, resources, cross, selection = [], [], [], [], {}
prediction_rows = []
for variant in cfg["variants"]:
    method = "m_film" if variant == "film_offset" else "m_lora"
    model = PersonalModel(bases[method], variant, cfg["lora_rank"])
    candidates = []
    for s in groups["validation"]:
        for lr, wd in itertools.product(cfg["learning_rates"][variant], cfg["weight_decays"]):
            result, _, resource = fit(model, s, lr, wd, steps, seed + s["key"][1])
            resources.append({"phase": "validation", "variant": variant, "dataset": s["key"][0],
                              "subject": s["key"][1], "learning_rate": lr, "weight_decay": wd, **resource})
            for k, r in result.items():
                row = {"variant": variant, "dataset": s["key"][0], "subject": s["key"][1], "learning_rate": lr,
                       "weight_decay": wd, "steps": k, "b3_ba": b3[(method, s["key"])], "ba": r["ba"],
                       "gain": r["ba"] - b3[(method, s["key"])]}
                candidates.append(row); search.append(row)
    hp = selection[variant] = select_candidate(candidates)
    own, params = {}, {}
    for s in groups["test"]:
        result, params[s["key"]], resource = fit(model, s, hp["learning_rate"], hp["weight_decay"], [hp["steps"]],
                                                 seed + s["key"][1])
        own[s["key"]] = result[hp["steps"]]["ba"]
        resources.append({"phase": "test", "variant": variant, "dataset": s["key"][0], "subject": s["key"][1],
                          "learning_rate": hp["learning_rate"], "weight_decay": hp["weight_decay"], **resource})
    torch.save({f"{d}:{n}": v for (d, n), v in params.items()}, out / f"personal-{variant}.pt")
    matrix = {}
    for s in groups["test"]:
        for donor in [t for t in groups["test"] if t["key"][0] == s["key"][0]]:
            model.load_personal(params[donor["key"]])
            pred = predict(model, s, s["query"])
            if args.stage_w:
                prediction_rows.extend({"variant": variant, "dataset": s["key"][0], "subject": s["key"][1],
                    "donor": donor["key"][1], "trial_id": s["trial_ids"][i], "label": int(s["query_labels"][j]),
                    "prediction": int(pred[j])} for j, i in enumerate(s["query"].cpu().tolist()))
            matrix[(s["key"], donor["key"])] = ba(s["query_labels"], pred)
            cross.append({"variant": variant, "dataset": s["key"][0], "subject": s["key"][1],
                          "donor": donor["key"][1], "ba": matrix[(s["key"], donor["key"])]})
    for s in groups["test"]:
        assert abs(matrix[(s["key"], s["key"])] - own[s["key"]]) < 1e-12  # reloading own params reproduces (ii)
        swap = float(np.mean([matrix[(s["key"], d)] for d in draws[s["key"]]]))
        rows.append({"variant": variant, "dataset": s["key"][0], "subject": s["key"][1], "fold": fold_index,
                     "seed": seed, "b3_ba": b3[(method, s["key"])], "own_ba": own[s["key"]], "swap_ba": swap,
                     "upper_bound": own[s["key"]] - swap, "own_gain": own[s["key"]] - b3[(method, s["key"])],
                     "swap_gain": swap - b3[(method, s["key"])],
                     "swap_donors": " ".join(str(d[1]) for d in draws[s["key"]]), **s["counts"]})
    pd.DataFrame(rows).to_csv(out / "subjects.csv", index=False)
    pd.DataFrame(cross).to_csv(out / "swap_matrix.csv", index=False)
    pd.DataFrame(search).to_csv(out / "validation_search.csv", index=False)
    pd.DataFrame(resources).to_csv(out / "fit_resources.csv", index=False)
    print(json.dumps({"variant": variant, "selected": hp, "seconds": time.monotonic() - start}), flush=True)
    del model
save_json(out / "selection.json", selection)
if args.stage_w:
    pd.DataFrame(prediction_rows).to_csv(out / "predictions.csv.gz", index=False)
assert state_hash(backbone) == frozen[0] and {m: state_hash(b) for m, b in bases.items()} == frozen[1]
receipt.update(status="complete", seconds=time.monotonic() - start, backbone_unchanged=True, stage_c_models_unchanged=True,
               peak_gpu_bytes=max(r["peak_gpu_bytes"] for r in resources))
save_json(out / "run.json", receipt)
print(json.dumps({"status": "complete", "seconds": receipt["seconds"]}), flush=True)
