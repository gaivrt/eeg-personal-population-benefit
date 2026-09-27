"""Fixed-budget continuation and the existing full-half personal diagnostic only."""
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .cross_model_run import plateau_report
from .cross_model_training import (EarlyStop, append_validation, evaluate, summarize,
                                   save_training_state, update)
from .stage_a_common import save_json
from .stage_b import ba, select_candidate
from .stage_c_context import context_window
from .stage_c_train import context_z
from .stage_c2 import fit, predict


class DefaultContext(nn.Module):
    """Expose the historical CBraMod default-z predictor to the common evaluator."""
    def __init__(self, population):
        super().__init__()
        self.population = population

    def forward(self, eeg, picks):
        return self.population(eeg, picks, self.population.z())


def endpoint(added, baseline):
    if baseline <= 0 or not 0 <= added <= 3 * baseline:
        raise ValueError("Update outside the frozen 4x budget")
    return {0: "1x", baseline: "2x", 3 * baseline: "4x"}.get(added)


def cbramod_update(model, subjects, hp, cfg, optimizer, rng, generator):
    """One unchanged C/W update, with persistent optimizer and sampler state."""
    t = cfg["training"]
    s = subjects[int(rng.integers(len(subjects)))]
    take = torch.randperm(len(s["labels"]), generator=generator, device=s["labels"].device)[:t["query_batch"]]
    minimum = int(np.ceil(cfg["context"]["min_seconds"] / cfg["context"]["segment_seconds"]))
    dropout = rng.random() < t["context_dropout"]
    z = model.z() if dropout else context_z(model, s, context_window(len(s["emb"]), rng, minimum))
    own = F.cross_entropy(model(s["eeg"][take].float(), s["picks"], z), s["labels"][take])
    loss = own
    foreign = None
    other = None
    if not dropout and hp["lambda"] > 0:
        pool = [other for other in subjects if other["key"][0] == s["key"][0] and other["key"] != s["key"]]
        if not pool:
            raise ValueError("No same-dataset training donor")
        other = pool[int(rng.integers(len(pool)))]
        z_other = context_z(model, other, context_window(len(other["emb"]), rng, minimum))
        foreign = F.cross_entropy(model(s["eeg"][take].float(), s["picks"], z_other), s["labels"][take])
        loss = own + hp["lambda"] * F.relu(own - foreign + hp["margin"])
    if not torch.isfinite(loss):
        trace = getattr(model, "budget_retry_failure_trace", None)
        if trace:
            save_json(trace, {"dataset": s["key"][0], "subject": s["key"][1],
                              "indices": take.cpu().tolist(), "context_dropout": bool(dropout),
                              "donor": other["key"] if other is not None else None,
                              "own_CE": str(float(own.detach())),
                              "foreign_CE": str(float(foreign.detach())) if foreign is not None else None,
                              "objective": str(float(loss.detach())), "phase": "before_backward"})
        raise ValueError("Nonfinite CBraMod training objective")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.trainable(), t["grad_clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    row = {"dataset": s["key"][0], "subject": s["key"][1], "indices": take.cpu().tolist(),
           "CE": float(own.detach()), "objective": float(loss.detach()), "context_dropout": bool(dropout)}
    if getattr(getattr(model, "core", None), "budget_retry_fp32", False):
        row["grad_norm_before_existing_clip"] = float(norm)
    return row


def continue_population(population, predictor, groups, optimizer, rng, generator,
                        source, cfg_x, cfg_c, out):
    """No CE early stop, no best-checkpoint selection, no test access."""
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    baseline = source["baseline_total_steps"]
    cap = source["maximum_added_steps"]
    if cap != 3 * baseline:
        raise ValueError("Inventory budget mismatch")
    training, validation = groups["train"], groups["validation"]
    conv = cfg_x["convergence"]
    tracker = EarlyStop(conv["CE_early_stop_patience_checks"], conv["CE_min_delta"])
    checks, endpoints, sampled = [], [], []
    start = time.monotonic()
    plateau = {"plateau_observed": False}
    with (out / "sampling.jsonl").open("w", encoding="utf-8") as sampling:
        for added in range(cap + 1):
            if added:
                if source["model"] == "CBraMod":
                    sample = cbramod_update(population, training, source["population_hyperparameters"],
                                           cfg_c, optimizer, rng, generator)
                else:
                    s = training[int(rng.integers(len(training)))]
                    take = torch.randperm(len(s["labels"]), generator=generator, device=s["labels"].device)[:32]
                    ce = update(population, s, take, optimizer, microbatch=8, clip=1.)
                    sample = {"dataset": s["key"][0], "subject": s["key"][1],
                              "indices": take.cpu().tolist(), "CE": ce, "objective": ce}
                sampling.write(json.dumps({"added_steps": added, "total_steps": baseline + added, **sample}) + "\n")
                sampled.append(sample["CE"])
            point = endpoint(added, baseline)
            regular = added % conv["G_check_every_steps"] == 0
            epoch = added % len(training) == 0
            if not (point or regular or epoch):
                continue
            val_rows = evaluate(predictor, validation)
            context = {"added_steps": added, "total_steps": baseline + added, "multiplier": 1 + added / baseline,
                       "epoch": added / len(training), "epoch_record": epoch or added == cap,
                       "partial_epoch": added == cap and not epoch, "regular_check": regular,
                       "point": point, "sampled_CE": float(np.mean(sampled)) if sampled else None,
                       "seconds": time.monotonic() - start}
            if regular:
                train_rows = evaluate(predictor, training, subset="all")
                append_validation(out / "train_all_trials.jsonl", context, train_rows)
                tracker.update(added, summarize(val_rows)["pooled"]["CE"])
                checks.append({"W_check": True, "step": added, "train": summarize(train_rows)["pooled"],
                               "validation": summarize(val_rows)["pooled"]})
                raw = plateau_report(checks, tracker, conv)
                plateau = {k: raw[k] for k in ("train_loss_plateau", "validation_loss_plateau",
                                              "validation_scores_plateau", "plateau_observed")}
            stop = plateau["plateau_observed"] or added == cap
            if stop:
                context.update(epoch_record=True, partial_epoch=not epoch)
            append_validation(out / "validation.jsonl", context, val_rows)
            metadata = {**context, "baseline_steps": baseline, "source_optimizer_restored": source["optimizer_resume"],
                        "plateau": plateau, "stop_reason": "joint_plateau" if plateau["plateau_observed"] else "4x_cap" if added == cap else None}
            if getattr(getattr(population, "core", None), "budget_retry_fp32", False):
                metadata["numerical_policy"] = "fp32_layer8_no_rounding"
            if stop and point is None:
                point = f"plateau-{baseline + added}"
            if point:
                metadata["point"] = point
                save_training_state(out / f"{point}.pt", population, optimizer, rng, generator, added, metadata)
                endpoints.append({**metadata, "validation": summarize(val_rows), "checkpoint": f"{point}.pt"})
                save_json(out / "endpoints.json", endpoints)
            save_training_state(out / "last.pt", population, optimizer, rng, generator, added, metadata)
            sampling.flush()
            if epoch:
                sampled.clear()
            if stop:
                break
    status = {"baseline_steps": baseline, "added_steps": added, "total_steps": baseline + added,
              "cap_added_steps": cap, "stop_reason": metadata["stop_reason"], "plateau": plateau,
              "platform_checks": checks, "test_used": False, "seconds": time.monotonic() - start,
              "peak_gpu_bytes": torch.cuda.max_memory_allocated() if next(population.parameters()).is_cuda else 0}
    save_json(out / "status.json", status)
    return endpoints, status


def select_personal(predictor, factory, validation, seed, microbatch, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    if any(s["role"] != "validation" for s in validation):
        raise ValueError("Personal selection must use validation subjects only")
    base = {(r["dataset"], r["subject"]): r["BA"] for r in evaluate(predictor, validation)}
    model = factory()
    rows, resources = [], []
    for s in validation:
        for lr, wd in itertools.product([1e-4, 1e-3], [0., .01]):
            result, _, resource = fit(model, s, lr, wd, [20, 60], seed + s["key"][1], microbatch=microbatch)
            resources.append({"dataset": s["key"][0], "subject": s["key"][1], "lr": lr, "wd": wd, **resource})
            rows.extend({"dataset": s["key"][0], "subject": s["key"][1], "learning_rate": lr,
                         "weight_decay": wd, "steps": steps, "gain": r["ba"] - base[s["key"]], "BA": r["ba"]}
                        for steps, r in result.items())
    choice = select_candidate(rows)
    save_json(out / "selection.json", choice)
    pd.DataFrame(rows).to_csv(out / "validation_search.csv", index=False)
    pd.DataFrame(resources).to_csv(out / "validation_fit_resources.csv", index=False)
    return choice


def source_donors(source, test):
    path = Path(source["source_diagnostic"])
    if source["model"] == "CBraMod":
        table = pd.read_csv(path / "subjects.csv").query("variant == 'lora8'")
        rows = [{"dataset": r.dataset, "subject": int(r.subject),
                 "donors": [int(v) for v in r.swap_donors.split()]} for r in table.itertuples()]
    else:
        rows = json.loads((path / "swap_donors.json").read_text())
    result = {(r["dataset"], r["subject"]): r["donors"] for r in rows}
    keys = {s["key"] for s in test}
    if set(result) != keys or len(rows) != len(keys):
        raise ValueError("Source donor roster mismatch")
    for (dataset, subject), donors in result.items():
        if len(donors) != 10 or any(d == subject or (dataset, d) not in keys for d in donors):
            raise ValueError("Invalid original swap donors")
    return result


def diagnose(predictor, factory, test, choice, source, point, out, microbatch):
    """Same fixed donor draws at every point. Fit only new points, reuse 1x parameters."""
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    donors = source_donors(source, test)
    model = factory()
    population_rows = evaluate(predictor, test)
    base = {(r["dataset"], r["subject"]): r for r in population_rows}
    append_validation(out / "G_test.jsonl", {"point": point, "choices_frozen": True}, population_rows)
    parameters = {}
    if point == "1x":
        loaded = torch.load(source["files"]["personal_parameters"]["path"], map_location="cpu", weights_only=True)
        parameters = {(d, int(s)): v for k, v in loaded.items() for d, s in [k.rsplit(":", 1)]}
        if set(parameters) != {s["key"] for s in test}:
            raise ValueError("Historical personal parameter roster mismatch")
    own, checks, resources, predictions = {}, [], [], []

    def record(s, condition, pred, donor=None):
        predictions.extend({"dataset": s["key"][0], "subject": s["key"][1], "fold": source["fold"],
                            "seed": source["seed"], "point": point, "condition": condition, "donor": donor,
                            "trial_id": s["trial_ids"][index], "label": int(s["query_labels"][j]), "prediction": int(pred[j])}
                           for j, index in enumerate(s["query"].cpu().tolist()))

    for s in test:
        model.reset()
        with torch.no_grad():
            g = torch.cat([predictor(s["eeg"][take].float(), s["picks"]) for take in s["query"].split(32)])
            zero = torch.cat([model(s["eeg"][take].float(), s["picks"]) for take in s["query"].split(32)])
        equal = bool(torch.equal(g.argmax(1), zero.argmax(1)))
        error = float((g - zero).abs().max())
        checks.append({"dataset": s["key"][0], "subject": s["key"][1],
                       "zero_personal_max_logit_error": error, "zero_personal_predictions_equal": equal})
        if not equal or not torch.isfinite(zero).all():
            raise ValueError("Zero personal update differs from corresponding G")
        record(s, "G", base[s["key"]]["predictions"])
        if point == "1x":
            model.load_personal(parameters[s["key"]])
            pred = predict(model, s, s["query"])
        else:
            result, values, resource = fit(model, s, choice["learning_rate"], choice["weight_decay"],
                                            [choice["steps"]], source["seed"] + s["key"][1], microbatch=microbatch)
            pred = result[choice["steps"]]["predictions"]
            parameters[s["key"]] = [p.cpu().clone() for p in values]
            resources.append({"dataset": s["key"][0], "subject": s["key"][1], **resource})
        own[s["key"]] = ba(s["query_labels"], pred)
        record(s, "own", pred)
    if point != "1x":
        torch.save({f"{d}:{s}": v for (d, s), v in parameters.items()}, out / "personal-full.pt")
    subjects, matrix = [], []
    for s in test:
        scores = {}
        for donor in [key for key in parameters if key[0] == s["key"][0]]:
            model.load_personal(parameters[donor])
            pred = predict(model, s, s["query"])
            scores[donor[1]] = ba(s["query_labels"], pred)
            matrix.append({"dataset": s["key"][0], "subject": s["key"][1], "donor": donor[1], "BA": scores[donor[1]]})
            record(s, "swap_matrix", pred, donor[1])
        if abs(scores[s["key"][1]] - own[s["key"]]) > 1e-12:
            raise ValueError("Personal reload changed own score")
        swap = float(np.mean([scores[d] for d in donors[s["key"]]]))
        subjects.append({"dataset": s["key"][0], "subject": s["key"][1], "fold": source["fold"], "seed": source["seed"],
                         "point": point, "G_BA": base[s["key"]]["BA"], "own_ba": own[s["key"]], "swap_ba": swap,
                         "own_gain": own[s["key"]] - base[s["key"]]["BA"], "upper_bound": own[s["key"]] - swap, **s["counts"]})
    for filename, rows in [("subjects.csv", subjects), ("swap_matrix.csv", matrix), ("consistency.csv", checks),
                           ("fit_resources.csv", resources), ("predictions.csv.gz", predictions)]:
        pd.DataFrame(rows).to_csv(out / filename, index=False)
    save_json(out / "swap_donors.json", [{"dataset": d, "subject": s, "donors": v} for (d, s), v in donors.items()])
    return max([r["peak_gpu_bytes"] for r in resources] + [0])


def classify(medians_pp, corrected_p):
    """Frozen user rule, including explicit precedence and missing-endpoint policy."""
    if any(medians_pp.get(k) is None for k in ("1x", "2x", "4x")):
        return "undetermined_missing_endpoints"
    m1, m2, m4 = [int(round(round(medians_pp[k] / 100, 12) * 10**12)) for k in ("1x", "2x", "4x")]
    threshold = 5_000_000_000  # 0.5 pp on the frozen 12-decimal BA grid.
    if m1 >= m2 >= m4 and m1 > m4 and m4 < threshold:
        return "primarily_population_undertraining"
    if abs(m4 - m2) < threshold and min(m2, m4) > 0 and all(corrected_p.get(k, 1) < .05 for k in ("2x", "4x")):
        return "stable_positive_after_continuation"
    return "descriptive_only"
