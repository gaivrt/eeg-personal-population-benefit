"""One fold/seed of Stage C4: priors P0/P1/P2 from other users' C2 parameters, then few-shot calibration."""
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
from subject_context.stage_b import Readout, ba, temporal_halves
from subject_context.stage_c import ContextModel, config_c
from subject_context.stage_c_train import load_snapshot
from subject_context.stage_c2 import predict
from subject_context.stage_c3 import nearest, random_sets
from subject_context.stage_c4 import CalibModel, calibrate, config_c4, make_prior, select_hp, select_k_alpha

p = argparse.ArgumentParser()
p.add_argument("--job", type=int, required=True)
p.add_argument("--benchmark", action="store_true")
args = p.parse_args()
require_compute()
assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name()
torch.set_num_threads(4)
cfg, c_cfg = config_c4(), config_c(); root = scratch_root()
fold_index = args.job // 5; seed = c_cfg["seeds"][args.job % 5]
fold = json.loads((ROOT / c_cfg["split_file"]).read_text())["folds"][fold_index]
split = {k: [tuple(x) for x in fold[k]] for k in ["train", "validation", "test"]}
assert_split(split["train"], split["validation"], split["test"])
halves = {(r["dataset"], r["subject"]): r for r in json.loads((ROOT / c_cfg["halves_file"]).read_text())["subjects"]}
name = f"fold-{fold_index}_seed-{seed}"
out = root / "stage_c4" / ("benchmark" if args.benchmark else "runs") / name
out.mkdir(parents=True, exist_ok=True)
assert not (out / "run.json").exists(), "Refuse to overwrite completed experiment"
c2, c3 = root / "stage_c2/runs" / name, root / "stage_c3/runs" / name
receipt = provenance()
receipt.update(stage_c4_config_sha256=sha256(ROOT / "configs/stage_c4.yaml"), fold=fold_index, seed=seed,
               gpu=torch.cuda.get_device_name(), c3_task_features_sha256=sha256(c3 / "task_features.npz"),
               **{f"c2_{v}_sha256": sha256(c2 / f"personal-{v}.pt") for v in cfg["variants"]})
start = time.monotonic()
kind = json.loads((root / "stage_b/runs" / name / "selection.json").read_text())["base_head"]
readout = Readout(torch.load(root / "baselines_v2" / f"B0_{kind}" / name / "head.pt", map_location="cpu",
                             weights_only=False), kind).cuda()
backbone = load_backbone(ROOT / config()["model"]["weights"], k=4).cuda().eval()
common = json.loads((ROOT / "configs/stage_a_channels.json").read_text())["readout_common"]
bases = {}
for method in ["m_film", "m_lora"]:
    bases[method] = ContextModel(backbone, readout, method, 27 * 200, 2 * 27 * 28 // 2, c_cfg).cuda()
    load_snapshot(bases[method], torch.load(root / "stage_c/runs" / name / f"model-{method}-main.pt", map_location="cuda"))
frozen = state_hash(backbone), {m: state_hash(b) for m, b in bases.items()}
features = {tuple([k.split(":")[0], int(k.split(":")[1])]): v for k, v in np.load(c3 / "task_features.npz").items()}
params = {v: {tuple([k.split(":")[0], int(k.split(":")[1])]): val
              for k, val in torch.load(c2 / f"personal-{v}.pt", map_location="cuda").items()} for v in cfg["variants"]}


def load_subject(dataset, subject):
    src = root / "processed" / dataset / f"sub-{subject:03d}"
    record = halves[(dataset, subject)]
    trials = pd.read_csv(src / "trials.csv")
    fit_rows, query, _ = temporal_halves(trials)
    assert fit_rows.tolist() == record["fit_indices"] and query.tolist() == record["query_indices"]
    meta = json.loads((src / "metadata.json").read_text())
    y = np.load(src / "labels.npy")
    return {"key": (dataset, subject), "eeg": torch.as_tensor(np.load(src / "tasks.npy"), device="cuda"),
            "picks": [meta["channels"].index(ch) for ch in common], "labels": torch.tensor(y, device="cuda"),
            "fit": torch.tensor(fit_rows, device="cuda"), "query": torch.tensor(query, device="cuda"),
            "query_labels": y[query]}


validation, test = split["validation"], split["test"]
if args.benchmark:  # 1 validation + 7 test per dataset: k=5 leave-one-out needs >=5 donors after removing i and j
    validation = [next(p for p in validation if p[0] == d) for d in config()["datasets"]]
    test = [p for d in config()["datasets"] for p in [q for q in test if q[0] == d][:7]]
subjects = {p: load_subject(*p) for p in validation + test}
models = {v: CalibModel(bases["m_film" if v == "film_offset" else "m_lora"], v) for v in cfg["variants"]}
receipt["load_seconds"] = time.monotonic() - start
print(json.dumps({"loaded": {"validation": len(validation), "test": len(test)}, "seconds": receipt["load_seconds"]}), flush=True)

b3 = {}
for variant, model in models.items():
    model.set_prior(None); model.reset()
    for key, s in subjects.items():
        b3[(variant, key)] = ba(s["query_labels"], predict(model, s, s["query"]))
cache = {}


def prior_score(variant, target, donors, alpha):
    """n=0 BA of `target` under alpha x mean parameters of `donors` (cached)."""
    key = (variant, target, tuple(sorted(donors)), alpha)
    if key not in cache:
        model = models[variant]
        model.set_prior(make_prior(variant, [params[variant][d] for d in donors], alpha)); model.reset()
        cache[key] = ba(subjects[target]["query_labels"], predict(model, subjects[target], subjects[target]["query"]))
    return cache[key]


def donor_sets(kind, target, pool, k, draws, rng):
    return [nearest(features, target, pool, k)] if kind == "P2" else random_sets(rng, pool, k, draws)


def choose(variant, kind, pool):
    """Leave-one-subject-out over the pool at n=0; returns (k, alpha)."""
    candidates = {}
    for k, alpha in itertools.product(cfg["ks"], cfg["alphas"]):
        gains = []
        for j in pool:
            rest = [q for q in pool if q != j]
            rng = np.random.default_rng([seed, j[1], k, 1])
            sets = donor_sets(kind, j, rest, k, cfg["random_draws"], rng)
            gains.append(np.mean([prior_score(variant, j, d, alpha) for d in sets]) - b3[(variant, j)])
        candidates[(k, alpha)] = gains
    return select_k_alpha(candidates)


def plan(variant, target, pool, draws):
    """Priors for one subject: {prior: (k, alpha, [donor sets])}."""
    result = {"P0": (0, 0.0, [None])}
    for kind in ["P1", "P2"]:
        k, alpha = choose(variant, kind, pool)
        rng = np.random.default_rng([seed, target[1], k, 2])
        result[kind] = (k, alpha, donor_sets(kind, target, pool, k, draws, rng))
    return result


def run_prior(variant, s, donors, alpha, n, hp):
    model = models[variant]
    model.set_prior(None if donors is None else make_prior(variant, [params[variant][d] for d in donors], alpha))
    if n == 0:
        model.reset()
        return ba(s["query_labels"], predict(model, s, s["query"])), None
    result, resource = calibrate(model, s, n, hp["learning_rate"], hp["mu"], hp["steps"], seed + s["key"][1])
    return result, resource


rows, search, resources, chosen, selection = [], [], [], [], {}
for variant, spec in cfg["variants"].items():
    shots = spec["shots"][:2] if args.benchmark else spec["shots"]
    steps = [20] if args.benchmark else cfg["steps"]
    pools = {d: [p for p in test if p[0] == d] for d in config()["datasets"]}
    plans = {}
    for key in validation:  # validation subjects: the whole same-dataset test pool; one random draw for P1
        plans[key] = plan(variant, key, pools[key[0]], cfg["validation_random_draws"])
    for key in test:
        plans[key] = plan(variant, key, [p for p in pools[key[0]] if p != key], cfg["random_draws"])
    for key in test:
        for kind in ["P1", "P2"]:
            k, alpha, sets = plans[key][kind]
            chosen.append({"variant": variant, "prior": kind, "dataset": key[0], "subject": key[1], "k": k, "alpha": alpha,
                           "donors": ";".join(" ".join(str(d[1]) for d in s) for s in sets)})
    print(json.dumps({"variant": variant, "priors_planned": True, "seconds": time.monotonic() - start}), flush=True)
    for kind, n in itertools.product(["P0", "P1", "P2"], shots):
        hp = None
        if n > 0:
            candidates = []
            for key in validation:
                k, alpha, sets = plans[key][kind]
                for lr, mu in itertools.product(spec["learning_rates"], spec["mu"]):
                    result, resource = run_prior(variant, subjects[key], sets[0], alpha, n,
                                                 {"learning_rate": lr, "mu": mu, "steps": steps})
                    resources.append({"phase": "validation", "variant": variant, "prior": kind, "shots": n, **resource})
                    candidates += [{"learning_rate": lr, "mu": mu, "steps": st, "gain": r - b3[(variant, key)]}
                                   for st, r in result.items()]
            search += [{"variant": variant, "prior": kind, "shots": n, **c} for c in candidates]
            hp = selection[f"{variant}_{kind}_n{n}"] = select_hp(candidates)
        for key in test:
            k, alpha, sets = plans[key][kind]
            scores = []
            for donors in sets:
                if n == 0:
                    scores.append(run_prior(variant, subjects[key], donors, alpha, 0, None)[0])
                else:
                    result, resource = run_prior(variant, subjects[key], donors, alpha, n,
                                                 {**hp, "steps": [hp["steps"]]})
                    resources.append({"phase": "test", "variant": variant, "prior": kind, "shots": n, **resource})
                    scores.append(result[hp["steps"]])
            ba_mean = float(np.mean(scores))
            rows.append({"variant": variant, "prior": kind, "shots": n, "dataset": key[0], "subject": key[1],
                         "fold": fold_index, "seed": seed, "k": k, "alpha": alpha, "draws": len(sets),
                         "b3_ba": b3[(variant, key)], "ba": ba_mean, "gain": ba_mean - b3[(variant, key)]})
        pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
        pd.DataFrame(search).to_csv(out / "validation_search.csv", index=False)
        pd.DataFrame(resources).to_csv(out / "fit_resources.csv", index=False)
        pd.DataFrame(chosen).to_csv(out / "k_alpha.csv", index=False)
        print(json.dumps({"variant": variant, "prior": kind, "shots": n, "hp": hp, "seconds": time.monotonic() - start}), flush=True)
save_json(out / "selection.json", selection)
assert state_hash(backbone) == frozen[0] and {m: state_hash(b) for m, b in bases.items()} == frozen[1]
peak = [r["peak_gpu_bytes"] for r in resources]
receipt.update(status="complete", seconds=time.monotonic() - start, backbone_unchanged=True,
               stage_c_models_unchanged=True, peak_gpu_bytes=max(peak) if peak else None, prior_cache_entries=len(cache))
save_json(out / "run.json", receipt)
print(json.dumps({"status": "complete", "seconds": receipt["seconds"]}), flush=True)
