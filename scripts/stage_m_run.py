"""One fold/seed of Stage M: FOMAML starts (M1/M2) vs B3 (R0), C4's P1 prior (R1) and continued B3 (R2)."""
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
from subject_context.stage_c3 import random_sets
from subject_context.stage_c4 import CalibModel, config_c4, make_prior, select_k_alpha
from subject_context import stage_m
from subject_context.stage_m import (adapt, config_m, continue_train, load_population, meta_train, pilot_grid,
                                     population_state, select_inner, select_meta, timed)

p = argparse.ArgumentParser()
p.add_argument("--job", type=int, required=True)
p.add_argument("--pilot", action="store_true", help="fold-0/seed-11 validation lr scan and step timing only")
p.add_argument("--benchmark", action="store_true", help="whole path on a subject subset with 5/10 outer steps")
args = p.parse_args()
assert not (args.pilot and args.benchmark)
require_compute()
assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name()
torch.set_num_threads(4)
cfg, c_cfg, c4_cfg = config_m(), config_c(), config_c4(); root = scratch_root()
fold_index = args.job // 5; seed = c_cfg["seeds"][args.job % 5]
if args.pilot:
    assert (fold_index, seed) == (cfg["inner"]["pilot"]["fold"], cfg["inner"]["pilot"]["seed"])
else:
    assert cfg["inner"]["learning_rates"], "inner learning-rate grid must be fixed from the pilot first"
fold = json.loads((ROOT / c_cfg["split_file"]).read_text())["folds"][fold_index]
split = {k: [tuple(x) for x in fold[k]] for k in ["train", "validation", "test"]}
assert_split(split["train"], split["validation"], split["test"])
halves = {(r["dataset"], r["subject"]): r for r in json.loads((ROOT / c_cfg["halves_file"]).read_text())["subjects"]}
name = f"fold-{fold_index}_seed-{seed}"
out = root / "stage_m" / ("pilot" if args.pilot else "benchmark" if args.benchmark else "runs") / name
if args.benchmark:
    cfg["outer"]["steps"] = [5, 10]
out.mkdir(parents=True, exist_ok=True)
assert not (out / "run.json").exists(), "Refuse to overwrite completed experiment"
c2, c4 = root / "stage_c2/runs" / name, root / "stage_c4/runs" / name
receipt = provenance()
receipt.update(stage_m_config_sha256=sha256(ROOT / "configs/stage_m.yaml"), fold=fold_index, seed=seed,
               gpu=torch.cuda.get_device_name(), c2_lora8_sha256=sha256(c2 / "personal-lora8.pt"),
               c4_k_alpha_sha256=sha256(c4 / "k_alpha.csv"), c4_results_sha256=sha256(c4 / "results.csv"))
start = time.monotonic()
kind = json.loads((root / "stage_b/runs" / name / "selection.json").read_text())["base_head"]
readout = Readout(torch.load(root / "baselines_v2" / f"B0_{kind}" / name / "head.pt", map_location="cpu",
                             weights_only=False), kind).cuda()
backbone = load_backbone(ROOT / config()["model"]["weights"], k=4).cuda().eval()
common = json.loads((ROOT / "configs/stage_a_channels.json").read_text())["readout_common"]
base = ContextModel(backbone, readout, "m_lora", 27 * 200, 2 * 27 * 28 // 2, c_cfg).cuda()
load_snapshot(base, torch.load(root / "stage_c/runs" / name / "model-m_lora-main.pt", map_location="cuda"))
frozen = state_hash(backbone), state_hash(base)
model = CalibModel(base, "lora8")
b3_state = population_state(model)
params = {tuple([k.split(":")[0], int(k.split(":")[1])]): v
          for k, v in torch.load(c2 / "personal-lora8.pt", map_location="cuda").items()}


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


roles = ["validation"] if args.pilot else ["train", "validation", "test"]
subjects = {p: load_subject(*p) for r in roles for p in split[r]}
train = [subjects[p] for p in split["train"]] if not args.pilot else []
validation = split["validation"]; test = [] if args.pilot else split["test"]
if args.benchmark:  # 1 validation + 7 test per dataset: C4's k=5 leave-one-out needs >=6 others
    validation = [next(p for p in validation if p[0] == d) for d in config()["datasets"]]
    test = [p for d in config()["datasets"] for p in [q for q in test if q[0] == d][:7]]
receipt["load_seconds"] = time.monotonic() - start
print(json.dumps({"loaded": {r: len(split[r]) for r in roles}, "seconds": receipt["load_seconds"]}), flush=True)
ks = cfg["inner"]["ks"]


def start_from(state, prior=None):
    load_population(model, state)
    model.set_prior(prior)


def zero_shot(key):
    return adapt(model, subjects[key], 0, [0], 0.0, seed + key[1])[0]


start_from(b3_state)
b3 = {key: zero_shot(key) for key in validation + test}
if not args.pilot:  # C4 scored B3 through the same CalibModel path and query batches
    c4_rows = pd.read_csv(c4 / "results.csv").query("variant == 'lora8' and prior == 'P0' and shots == 0")
    c4_b3 = {(r.dataset, r.subject): r.b3_ba for r in c4_rows.itertuples()}
    receipt["b3_vs_c4_mismatches"] = sum(abs(b3[k] - c4_b3[k]) > 1e-12 for k in test)
    receipt["b3_vs_c4_max_abs_diff"] = max(abs(b3[k] - c4_b3[k]) for k in test)

if args.pilot:
    pilot = cfg["inner"]["pilot"]; rows = []
    for key, lr in itertools.product(validation, pilot["scan"]):
        before = len(stage_m.divergences)
        result = adapt(model, subjects[key], pilot["shots"], ks, lr, seed + key[1])
        diverged = len(stage_m.divergences) > before
        rows += [{"dataset": key[0], "subject": key[1], "k": k, "learning_rate": lr, "ba": v, "gain": v - b3[key],
                  "diverged": diverged} for k, v in result.items()]
    pd.DataFrame(rows).to_csv(out / "pilot_scan.csv", index=False)
    grid = pilot_grid(rows, pilot["scan"])
    summary = pd.DataFrame(rows).groupby("learning_rate").gain.agg(["median", "mean"]).reset_index()
    print(summary.to_string(), flush=True)
    # Step timing on the real training pool (loaded only now), 30 outer steps per K, M2 path, then discarded.
    train = [load_subject(*p) for p in split["train"]]
    timing = {}
    for k in ks:
        short = {**cfg, "outer": {**cfg["outer"], "steps": [30]}}
        log_lr = torch.full((12,), float(np.log(grid[len(grid) // 2])), device="cuda", requires_grad=True)
        _, seconds, peak = timed(lambda: meta_train(model, train, k, None, short, seed, log_lr=log_lr))
        timing[k] = {"seconds_per_outer_step": seconds / 30, "peak_gpu_bytes": peak}
        start_from(b3_state)
    save_json(out / "pilot.json", {"grid": grid, "timing": timing, "summary": summary.to_dict("records")})
    receipt.update(status="complete", seconds=time.monotonic() - start, inner_divergences=len(stage_m.divergences))
    save_json(out / "run.json", receipt)
    print(json.dumps({"grid": grid, "timing": timing}), flush=True)
    raise SystemExit

grid = cfg["inner"]["learning_rates"]
valid_rows, test_rows, search, training, selection = [], [], [], [], {}


def shots_for(key):
    return [n for n in cfg["shots"] if n <= len(subjects[key]["fit"]) and (n != 40 or key[0] != "PhysionetMI")]


def row(method, key, n, ba_value, **extra):
    return {"method": method, "shots": n, "dataset": key[0], "subject": key[1], "fold": fold_index, "seed": seed,
            "b3_ba": b3[key], "ba": ba_value, "gain": ba_value - b3[key], **extra}


def control_search(method, setup):
    """Per n: grid over lr (one K=20 path scores all K) on validation; returns {n: hp} and validation rows."""
    chosen = {}
    for n in [n for n in cfg["shots"] if n > 0]:
        rows = []
        for key in [k for k in validation if n in shots_for(k)]:
            for lr in grid:
                setup(key, 0)
                result = adapt(model, subjects[key], n, ks, lr, seed + key[1])
                rows += [{"method": method, "shots": n, "dataset": key[0], "subject": key[1], "k": k,
                          "learning_rate": lr, "ba": v, "gain": v - b3[key]} for k, v in result.items()]
        if not rows:
            continue
        search.extend(rows)
        hp = chosen[n] = select_inner(rows)
        valid_rows.extend(row(method, (r["dataset"], r["subject"]), n, r["ba"], k=hp["k"], learning_rate=hp["learning_rate"])
                          for r in rows if r["k"] == hp["k"] and r["learning_rate"] == hp["learning_rate"])
    for key in validation:
        setup(key, 0)
        valid_rows.append(row(method, key, 0, zero_shot(key)))
    return chosen


def evaluate_test(method, setup, draws, hp_for):
    """hp_for(n) -> (k, lr); setup(key, draw) sets the start. Scores are averaged over draws."""
    for key in test:
        for n in shots_for(key):
            k, lr = hp_for(n) if n else (0, 0.0)
            scores = []
            for d in range(draws(key)):
                setup(key, d)
                scores.append(adapt(model, subjects[key], n, [k] if n else [0], lr, seed + key[1])[k if n else 0])
            test_rows.append(row(method, key, n, float(np.mean(scores)), k=k,
                                 learning_rate=float(lr) if not torch.is_tensor(lr) else float("nan"), draws=len(scores)))
        pd.DataFrame(test_rows).to_csv(out / "results.csv", index=False)


# ---- R0: B3; its n=10 search also fixes M1's inner lr per K.
r0 = lambda key, d: start_from(b3_state)
selection["R0"] = control_search("R0", r0)
r0_n10 = pd.DataFrame([r for r in search if r["method"] == "R0" and r["shots"] == 10])
m1_lr = {k: select_inner(r0_n10[r0_n10.k == k].to_dict("records"))["learning_rate"] for k in ks}
selection["m1_inner_lr_by_k"] = m1_lr
print(json.dumps({"R0": selection["R0"], "m1_lr": m1_lr, "seconds": time.monotonic() - start}), flush=True)

# ---- M1/M2: FOMAML from B3, one training per K, checkpoints at the configured outer steps.
states, meta_rows, logs = {}, [], []
for variant, k in itertools.product(cfg["variants"], ks):
    start_from(b3_state)
    log_lr = (torch.full((12,), float(np.log(m1_lr[k])), device="cuda", requires_grad=True) if variant == "M2" else None)

    def checkpoint(step, variant=variant, k=k, log_lr=log_lr):
        rates = m1_lr[k] if log_lr is None else log_lr.detach().exp().clone()
        states[(variant, k, step)] = (population_state(model), rates)
        for key in validation:
            if 10 in shots_for(key):
                v = adapt(model, subjects[key], 10, [k], rates, seed + key[1])[k]
                meta_rows.append({"variant": variant, "k": k, "steps": step, "dataset": key[0], "subject": key[1],
                                  "ba": v, "gain": v - b3[key]})
    log, seconds, peak = timed(lambda: meta_train(model, train, k, m1_lr[k], cfg, seed, log_lr=log_lr,
                                                  on_checkpoint=checkpoint))
    logs += [{"variant": variant, "k": k, **r} for r in log]
    training.append({"method": variant, "k": k, "outer_steps": max(cfg["outer"]["steps"]), "seconds": seconds,
                     "peak_gpu_bytes": peak, **({"final_log_lr": log_lr.detach().cpu().tolist()} if log_lr is not None else {})})
    pd.DataFrame(meta_rows).to_csv(out / "meta_validation.csv", index=False)
    pd.DataFrame(training).to_csv(out / "training_resources.csv", index=False)
    print(json.dumps({**training[-1], "seconds_elapsed": time.monotonic() - start}), flush=True)
pd.DataFrame(logs).to_csv(out / "meta_training_log.csv", index=False)
for variant in cfg["variants"]:
    chosen = selection[variant] = select_meta([r for r in meta_rows if r["variant"] == variant])
    state, rates = states[(variant, chosen["k"], chosen["steps"])]
    chosen["inner_lr"] = rates.tolist() if torch.is_tensor(rates) else rates
    torch.save({"population": state, "inner_lr": rates}, out / f"start-{variant}.pt")
    setup = lambda key, d, state=state: start_from(state)
    hp = lambda n, chosen=chosen, rates=rates: (chosen["k"], rates)
    for key in validation:  # validation scores at every n for the strongest-control rule and reporting
        setup(key, 0)
        for n in shots_for(key):
            v = adapt(model, subjects[key], n, [chosen["k"]] if n else [0], rates, seed + key[1])[chosen["k"] if n else 0]
            valid_rows.append(row(variant, key, n, v, k=chosen["k"] if n else 0))
    evaluate_test(variant, setup, lambda key: 1, hp)
    print(json.dumps({variant: {k: v for k, v in chosen.items() if k != "inner_lr"}, "seconds": time.monotonic() - start}), flush=True)
del states

# ---- R2: B3 continued with its original objective for the same number of outer (gradient) steps.
start_from(b3_state)
r2_steps = [max(cfg["outer"]["steps"])]
log, seconds, peak = timed(lambda: continue_train(model, train, r2_steps, cfg, seed))
training.append({"method": "R2", "k": 0, "outer_steps": r2_steps[0], "seconds": seconds, "peak_gpu_bytes": peak})
r2_state = population_state(model)
torch.save({"population": r2_state}, out / "start-R2.pt")
pd.DataFrame(log).to_csv(out / "r2_training_log.csv", index=False)
r2 = lambda key, d: start_from(r2_state)
selection["R2"] = control_search("R2", r2)

# ---- R1: C4's P1 prior (random k others' averaged personal LoRA x alpha, leave-one-out k/alpha).
c4_plan = pd.read_csv(c4 / "k_alpha.csv").query("variant == 'lora8' and prior == 'P1'")
plans = {}
for r in c4_plan.itertuples():
    sets = [[(r.dataset, int(x)) for x in s.split()] for s in r.donors.split(";")]
    plans[(r.dataset, r.subject)] = (int(r.k), float(r.alpha), sets)
assert set(test) <= set(plans)
pools = {d: [q for q in test if q[0] == d] for d in config()["datasets"]}
cache = {}


def prior_score(target, donors, alpha):
    key = (target, tuple(sorted(donors)), alpha)
    if key not in cache:
        start_from(b3_state, make_prior("lora8", [params[d] for d in donors], alpha))
        cache[key] = zero_shot(target)
    return cache[key]


for dataset, pool in pools.items():  # validation subjects: C4's rule on the whole same-dataset test pool
    candidates = {}
    for kk, alpha in itertools.product(c4_cfg["ks"], c4_cfg["alphas"]):
        gains = []
        for j in pool:
            rest = [q for q in pool if q != j]
            sets = random_sets(np.random.default_rng([seed, j[1], kk, 1]), rest, kk, c4_cfg["random_draws"])
            gains.append(np.mean([prior_score(j, d, alpha) for d in sets]) - b3[j])
        candidates[(kk, alpha)] = gains
    kk, alpha = select_k_alpha(candidates)
    for key in [v for v in validation if v[0] == dataset]:
        rng = np.random.default_rng([seed, key[1], kk, 2])
        plans[key] = (kk, alpha, random_sets(rng, pool, kk, c4_cfg["validation_random_draws"]))
del cache


def r1(key, draw):
    kk, alpha, sets = plans[key]
    start_from(b3_state, make_prior("lora8", [params[d] for d in sets[draw]], alpha))


selection["R1"] = control_search("R1", r1)
selection["R1_validation_k_alpha"] = {f"{k[0]}:{k[1]}": plans[k][:2] for k in validation}

# ---- Test scoring for the controls (hp per n from validation).
for method, setup, draws in [("R0", r0, lambda key: 1), ("R2", r2, lambda key: 1), ("R1", r1, lambda key: len(plans[key][2]))]:
    hp = lambda n, method=method: (selection[method][n]["k"], selection[method][n]["learning_rate"])
    evaluate_test(method, setup, draws, hp)
    print(json.dumps({"tested": method, "seconds": time.monotonic() - start}), flush=True)
model.set_prior(None)

pd.DataFrame(valid_rows).to_csv(out / "validation_results.csv", index=False)
pd.DataFrame(search).to_csv(out / "validation_search.csv", index=False)
pd.DataFrame(training).to_csv(out / "training_resources.csv", index=False)
save_json(out / "selection.json", {m: ({str(n): v for n, v in s.items()} if m in ("R0", "R1", "R2") else s)
                                   for m, s in selection.items()})
assert state_hash(backbone) == frozen[0] and state_hash(base) == frozen[1]
receipt.update(inner_divergences=len(stage_m.divergences),
               inner_divergences_by_shots=pd.Series([d[0] for d in stage_m.divergences], dtype=int).value_counts().to_dict())
receipt.update(status="complete", seconds=time.monotonic() - start, backbone_unchanged=True, stage_c_model_unchanged=True,
               peak_gpu_bytes=int(max(t["peak_gpu_bytes"] for t in training)))
save_json(out / "run.json", receipt)
print(json.dumps({"status": "complete", "seconds": receipt["seconds"]}), flush=True)
