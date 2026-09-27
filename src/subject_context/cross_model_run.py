"""Formal X1-core stages. All selection consumes training/validation subjects only."""
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .cross_model import PersonalModel, PopulationModel
from .cross_model_training import (EarlyStop, append_validation, evaluate, restore_training_state,
                                   save_training_state, select_population, summarize, update)
from .stage_a_common import save_json
from .stage_b import Readout, ba, select_candidate
from .stage_c2 import fit, predict


def device_seed(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def resource_clock(start):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.monotonic() - start
    return {"seconds": seconds, "single_GPU_hours_this_stage": seconds / 3600 if torch.cuda.is_available() else 0.,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0}


@torch.no_grad()
def feature_metrics(readout, subjects):
    """B0 uses all validation trials; counts/logits follow the original head path."""
    rows = []
    for s in subjects:
        logits = readout(s["features"])
        truth, pred = s["labels"].cpu().numpy(), logits.argmax(1).cpu().numpy()
        ce = float(nn.functional.cross_entropy(logits, s["labels"], reduction="sum"))
        if not np.isfinite(ce):
            raise ValueError("Nonfinite B0 validation")
        confusion = np.bincount(truth * 2 + pred, minlength=4).reshape(2, 2)
        rows.append({"dataset": s["key"][0], "subject": s["key"][1], "subset": "all",
                     "n": len(truth), "CE_sum": ce, "CE": ce / len(truth),
                     "accuracy": float((truth == pred).mean()), "BA": ba(truth, pred),
                     "confusion": confusion.tolist(), "predictions": pred.tolist(),
                     "indices": list(range(len(truth)))})
    return rows


def train_b0(groups, cfg, seed, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    device = groups["train"][0]["labels"].device
    # Float64 statistics over training trials, population SD, same as original A.
    x = torch.cat([s["features"] for s in groups["train"]])
    mean = x.double().mean(0)
    std = x.double().std(0, correction=0).clamp_min(1e-6)
    y = torch.cat([s["labels"] for s in groups["train"]])
    normalized = ((x.double() - mean) / std).float()
    del x
    chosen, best, trajectories = None, -float("inf"), {}
    start = time.monotonic()
    for kind in ("linear", "mlp"):
        for lr in cfg["learning_rates"]:
            device_seed(seed, device)
            width = normalized.shape[1]
            head = nn.Linear(width, 2) if kind == "linear" else nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 2))
            readout = Readout({"head": head.state_dict(), "feature_mean": mean.cpu(), "feature_std": std.cpu()}, kind).to(device)
            readout.head.requires_grad_(True)
            optimizer = torch.optim.AdamW(readout.head.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
            candidate = out / f"{kind}-lr-{lr:g}"
            candidate.mkdir(exist_ok=True)
            rng = np.random.default_rng(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
            trace = []
            for epoch in range(cfg["max_epochs"] + 1):
                train_loss, examples = 0., 0
                if epoch:
                    readout.head.train()
                    for take in torch.randperm(len(y), device=device).split(cfg["batch_size"]):
                        optimizer.zero_grad(set_to_none=True)
                        loss = nn.functional.cross_entropy(readout.head(normalized[take]), y[take])
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite B0 training loss")
                        loss.backward(); optimizer.step()
                        train_loss += float(loss.detach()) * len(take); examples += len(take)
                readout.eval()
                metrics = feature_metrics(readout, groups["validation"])
                context = {"phase": "B0", "head": kind, "learning_rate": lr, "epoch": epoch,
                           "train_CE": train_loss / examples if examples else None, **resource_clock(start)}
                append_validation(candidate / "validation.jsonl", context, metrics)
                trace.append({"epoch": epoch, "train_sampled_CE": context["train_CE"], **summarize(metrics)["pooled"]})
                value = summarize(metrics)["pooled"]["BA"]
                # epoch0 is a required log, never an additional selection candidate.
                if epoch and value > best:
                    best = value
                    chosen = {"head": {k: v.detach().cpu().clone() for k, v in readout.head.state_dict().items()},
                              "feature_mean": mean.cpu(), "feature_std": std.cpu(),
                              "kind": kind, "learning_rate": lr, "epoch": epoch, "validation_BA": value}
                    torch.save(chosen, out / "selected.pt")
                    save_training_state(out / "selected-resumable.pt", readout, optimizer, rng, generator, epoch, context)
                save_training_state(candidate / "last.pt", readout, optimizer, rng, generator, epoch, context)
            trajectories[f"{kind}-lr-{lr:g}"] = trace
    save_json(out / "selection.json", {k: chosen[k] for k in ("kind", "learning_rate", "epoch", "validation_BA")})
    convergence = {}
    for name, trace in trajectories.items():
        epochs = trace[1:]
        tail = epochs[-8:]
        checks = {}
        for metric in ("train_sampled_CE", "CE", "accuracy", "BA"):
            values = np.array([r[metric] for r in tail])
            threshold = max(1e-4, .01*abs(values.mean())) if metric.endswith("CE") else .005
            stable = len(tail) == 8 and np.ptp(values) <= threshold
            if metric in ("accuracy", "BA"):
                bests = np.maximum.accumulate([r[metric] for r in epochs])
                stable = stable and bests[-1]-bests[max(0,len(epochs)-9)] <= .001
            checks[metric+"_plateau"] = bool(stable)
        convergence[name] = {**checks, "last_epoch": len(epochs), "plateau_observed": all(checks.values()),
                             "train_CE_definition": "sample-weighted training-update loss within each epoch",
                             "selected_epoch_in_final_window": (name == f"{chosen['kind']}-lr-{chosen['learning_rate']:g}"
                                and all(checks.values()) and chosen["epoch"] >= tail[0]["epoch"])}
    save_json(out / "convergence.json", convergence)
    return Readout(chosen, chosen["kind"]).to(device)


def plateau_report(curve, stop, cfg):
    checks = [r for r in curve if r["W_check"]]
    window = cfg["window_checks"]
    tail = checks[-window:]
    full = len(tail) == window
    def flat(values):
        return full and np.ptp(values) <= max(cfg["loss_absolute_range_floor"], cfg["loss_relative_range"] * abs(np.mean(values)))
    train_flat = flat([r["train"]["CE"] for r in tail])
    val_flat = flat([r["validation"]["CE"] for r in tail])
    score_flat = full
    for metric in ("accuracy", "BA"):
        values = [r["validation"][metric] for r in tail]
        all_values = [r["validation"][metric] for r in checks]
        bests = np.maximum.accumulate(all_values)
        anchor = max(0, len(checks) - window - 1)
        score_flat = score_flat and np.ptp(values) <= cfg["accuracy_and_BA_absolute_range"] and bests[-1] - bests[anchor] <= cfg["accuracy_and_BA_min_delta"]
    observed = bool(train_flat and val_flat and score_flat)
    return {"train_loss_plateau": bool(train_flat), "validation_loss_plateau": bool(val_flat),
            "validation_scores_plateau": bool(score_flat), "plateau_observed": observed,
            "selected_in_final_plateau_window": observed and tail[0]["step"] <= stop.best_step <= tail[-1]["step"],
            "selected_additional_step": stop.best_step, "selected_validation_CE": stop.best,
            "bad_checks": stop.bad_checks, "stopped_by": "CE_patience" if stop.bad_checks >= stop.patience else "step_budget"}


def train_population(backbone, readout, groups, cfg, seed, out, microbatch):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    train_cfg, conv = cfg["population"]["G"], cfg["convergence"]
    training, validation = groups["train"], groups["validation"]
    device = next(backbone.parameters()).device
    # The B0 comparison uses the identical raw EEG query batches used by G.
    class Baseline(nn.Module):
        def forward(self, eeg, picks):
            return readout(backbone(eeg, picks))
    baseline = {(r["dataset"], r["subject"]): r["BA"] for r in evaluate(Baseline(), validation)}
    search = []
    start = time.monotonic()

    def initialize(lr):
        device_seed(seed, device)
        model = PopulationModel(backbone, readout).to(device)
        optimizer = torch.optim.AdamW(model.trainable(), lr=lr, weight_decay=train_cfg["weight_decay"])
        return model, optimizer, np.random.default_rng(seed), torch.Generator(device=device).manual_seed(seed)

    def trajectory(model, optimizer, rng, generator, budget, phase, folder, initial_steps=0):
        folder.mkdir(exist_ok=True)
        stop = EarlyStop(conv["CE_early_stop_patience_checks"], conv["CE_min_delta"])
        curve, exposure, losses = [], [], []
        lr = optimizer.param_groups[0]["lr"]
        for step in range(budget + 1):
            if step:
                s = training[int(rng.integers(len(training)))]
                take = torch.randperm(len(s["labels"]), generator=generator, device=device)[:train_cfg["batch_size"]]
                ce = update(model, s, take, optimizer, microbatch, train_cfg["gradient_clip"])
                exposure.append({"step": step, "dataset": s["key"][0], "subject": s["key"][1],
                                 "indices": take.cpu().tolist(), "CE": ce})
                losses.append(ce)
            epoch_record = step == 0 or step % len(training) == 0 or step == budget
            check = phase == "continuation" and (step % conv["G_check_every_steps"] == 0 or step == budget)
            candidate = phase == "initial" and step in train_cfg["initial_checkpoint_candidates"]
            if not (epoch_record or check or candidate):
                continue
            rows = evaluate(model, validation)
            context = {"phase": phase, "learning_rate": lr, "step": step, "global_step": initial_steps + step,
                       "epoch": step / len(training), "epoch_record": epoch_record,
                       "partial_epoch": step == budget and step % len(training) != 0,
                       "W_check": check, "initial_selection_candidate": candidate,
                       "train_sampled_CE": float(np.mean(losses)) if losses else None, **resource_clock(start)}
            done, improved = False, False
            if check:
                train_rows = evaluate(model, training, subset="all")
                append_validation(folder / "train_all_trials.jsonl", context, train_rows)
                improved, done = stop.update(step, summarize(rows)["pooled"]["CE"])
                curve.append({"W_check": True, "step": step, "train": summarize(train_rows)["pooled"],
                              "validation": summarize(rows)["pooled"]})
            context.update(best_additional_step=stop.best_step if check else None, bad_checks=stop.bad_checks if check else None)
            if done and not epoch_record:
                context.update(epoch_record=True, partial_epoch=step % len(training) != 0)
            append_validation(folder / "validation.jsonl", context, rows)
            if exposure:
                with (folder / "sampling.jsonl").open("a", encoding="utf-8") as stream:
                    for r in exposure:
                        stream.write(json.dumps(r) + "\n")
                exposure.clear()
            metadata = {**context, "early_stop": vars(stop) if check else None}
            save_training_state(folder / "last.pt", model, optimizer, rng, generator, step, metadata)
            if candidate:
                save_training_state(folder / f"step-{step}.pt", model, optimizer, rng, generator, step, metadata)
                search.extend({"learning_rate": lr, "episodes": step, "dataset": r["dataset"], "subject": r["subject"],
                               "gain": r["BA"] - baseline[(r["dataset"], r["subject"])], "BA": r["BA"]} for r in rows)
            if improved:
                save_training_state(folder / "selected.pt", model, optimizer, rng, generator, step, metadata)
            if epoch_record:
                losses.clear()
            if done:
                break
        return curve, stop, step

    for lr in train_cfg["learning_rates"]:
        model, optimizer, rng, generator = initialize(lr)
        trajectory(model, optimizer, rng, generator, train_cfg["initial_max_steps_per_lr"], "initial", out / f"initial-lr-{lr:g}")
        del model, optimizer
    hp = select_population(search)
    pd.DataFrame(search).to_csv(out / "initial_selection_candidates.csv", index=False)
    save_json(out / "initial_selection.json", hp)
    model, optimizer, rng, generator = initialize(hp["learning_rate"])
    source = out / f"initial-lr-{hp['learning_rate']:g}" / f"step-{hp['episodes']}.pt"
    restore_training_state(source, model, optimizer, rng, generator)
    budget = hp["episodes"] * train_cfg["continuation_additional_steps_multiplier"]
    curve, stop, last_step = trajectory(model, optimizer, rng, generator, budget, "continuation", out / "continuation", hp["episodes"])
    status = {**plateau_report(curve, stop, conv), "initial_selection": hp, "additional_steps_run": last_step,
              "additional_steps_cap": budget, "optimizer_and_RNG_restored": True,
              "test_used_for_selection": False, **resource_clock(start)}
    save_json(out / "convergence.json", status)
    restore_training_state(out / "continuation/selected.pt", model, optimizer, rng, generator)
    return model, status


def personal_selection(population, validation, cfg, seed, out, microbatch):
    """Freeze all personal choices before any test-subject evaluation."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    baseline = {(r["dataset"], r["subject"]): r["BA"] for r in evaluate(population, validation)}
    model = PersonalModel(population)
    selections, rows, resources = {}, [], []
    extra = [40] if any(s["key"][0] in cfg["additional_40"] for s in validation) else []
    for n in [None, *cfg["shots"], *extra]:
        candidates = []
        for s in validation:
            if n == 40 and s["key"][0] not in cfg["additional_40"]:
                continue
            if n is not None and len(s["fit"]) < n:
                raise ValueError("Frozen roster cannot supply requested n")
            view = dict(s, fit=s["fit"] if n is None else s["fit"][:n])
            for lr, wd in itertools.product(cfg["learning_rates"]["lora8"], cfg["weight_decays"]):
                result, _, resource = fit(model, view, lr, wd, cfg["steps"], seed + s["key"][1], microbatch=microbatch)
                resources.append({"n": n, "dataset": s["key"][0], "subject": s["key"][1], "lr": lr, "wd": wd, **resource})
                for steps, value in result.items():
                    candidates.append({"n": n, "dataset": s["key"][0], "subject": s["key"][1],
                                       "learning_rate": lr, "weight_decay": wd, "steps": steps,
                                       "gain": value["ba"] - baseline[s["key"]], "BA": value["ba"]})
        selections["full" if n is None else str(n)] = select_candidate(candidates)
        rows.extend(candidates)
        save_json(out / "selection.json", selections)
        pd.DataFrame(rows).to_csv(out / "validation_search.csv", index=False)
        pd.DataFrame(resources).to_csv(out / "validation_fit_resources.csv", index=False)
    return selections


def diagnose(population, backbone, readout, test, selections, cfg, seed, fold, out, microbatch):
    """Full own/swap matrix reuses fitted full-half parameters; few-shot fits are separate."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model = PersonalModel(population)
    base_rows = evaluate(population, test)
    base = {(r["dataset"], r["subject"]): r for r in base_rows}
    append_validation(out / "G_test.jsonl", {"selected_before_test": True}, base_rows)
    own, parameters, curve, resource_rows, checks, predictions = {}, {}, [], [], [], []

    def record(s, condition, pred, n=None, donor=None):
        predictions.extend({"dataset": s["key"][0], "subject": s["key"][1], "fold": fold, "seed": seed,
                            "condition": condition, "n": n, "donor": donor, "trial_id": s["trial_ids"][i],
                            "label": int(s["query_labels"][j]), "prediction": int(pred[j])}
                           for j, i in enumerate(s["query"].cpu().tolist()))
    for s in test:
        model.reset()
        with torch.no_grad():
            g, zero, b0 = [], [], []
            for take in s["query"].split(32):
                signal = s["eeg"][take].float()
                g.append(population(signal, s["picks"]))
                zero.append(model(signal, s["picks"]))
                b0.append(readout(backbone(signal, s["picks"])))
            g, zero, b0 = map(torch.cat, (g, zero, b0))
        checks.append({"dataset": s["key"][0], "subject": s["key"][1],
                       "zero_personal_max_logit_error": float((g-zero).abs().max()),
                       "zero_personal_prediction_agreement": float((g.argmax(1)==zero.argmax(1)).float().mean()),
                       "B0_BA": ba(s["query_labels"], b0.argmax(1).cpu().numpy()), "G_BA": base[s["key"]]["BA"]})
        record(s, "B0", b0.argmax(1).cpu().numpy()); record(s, "G", g.argmax(1).cpu().numpy())
        for n in [None, *cfg["shots"], *([40] if s["key"][0] in cfg["additional_40"] else [])]:
            hp = selections["full" if n is None else str(n)]
            view = dict(s, fit=s["fit"] if n is None else s["fit"][:n])
            result, parameter, resources = fit(model, view, hp["learning_rate"], hp["weight_decay"], [hp["steps"]], seed+s["key"][1], microbatch=microbatch)
            value = result[hp["steps"]]
            record(s, "own" if n is None else "few_shot", value["predictions"], n=n)
            resource_rows.append({"dataset": s["key"][0], "subject": s["key"][1], "n": n, **resources})
            if n is None:
                own[s["key"]] = value["ba"]
                parameters[s["key"]] = [p.cpu().clone() for p in parameter]
            else:
                curve.append({"dataset": s["key"][0], "subject": s["key"][1], "fold": fold, "seed": seed,
                              "shots": n, "ba": value["ba"], "G_BA": base[s["key"]]["BA"],
                              "gain": value["ba"]-base[s["key"]]["BA"], "full_gain": own[s["key"]]-base[s["key"]]["BA"],
                              "one_class_only": len(torch.unique(s["labels"][view["fit"]])) < 2})
    torch.save({f"{d}:{s}": v for (d,s), v in parameters.items()}, out / "personal-full.pt")
    draw_rng = np.random.default_rng(seed)
    subjects, matrix, donors = [], [], []
    for s in test:
        pool = [t["key"] for t in test if t["key"][0] == s["key"][0]]
        others = [key for key in pool if key != s["key"]]
        draws = [others[i] for i in draw_rng.integers(len(others), size=10)]
        scores = {}
        for donor in pool:
            model.load_personal(parameters[donor])
            pred = predict(model, s, s["query"])
            scores[donor] = ba(s["query_labels"], pred)
            record(s, "swap_matrix", pred, donor=donor[1])
            matrix.append({"dataset": s["key"][0], "subject": s["key"][1], "donor": donor[1], "BA": scores[donor]})
        if abs(scores[s["key"]]-own[s["key"]]) > 1e-12:
            raise ValueError("Personal checkpoint reload changed own predictions")
        swap = float(np.mean([scores[d] for d in draws]))
        donors.append({"dataset": s["key"][0], "subject": s["key"][1], "donors": [d[1] for d in draws]})
        subjects.append({"dataset": s["key"][0], "subject": s["key"][1], "fold": fold, "seed": seed,
                         "G_BA": base[s["key"]]["BA"], "own_ba": own[s["key"]], "swap_ba": swap,
                         "own_gain": own[s["key"]]-base[s["key"]]["BA"], "upper_bound": own[s["key"]]-swap, **s["counts"]})
    for filename, rows in [("subjects.csv", subjects), ("few_shot.csv", curve), ("swap_matrix.csv", matrix),
                           ("consistency.csv", checks), ("fit_resources.csv", resource_rows), ("predictions.csv.gz", predictions)]:
        pd.DataFrame(rows).to_csv(out / filename, index=False)
    save_json(out / "swap_donors.json", donors)
