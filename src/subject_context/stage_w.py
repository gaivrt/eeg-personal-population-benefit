"""Validation-loss continuation of the existing Stage C training procedure."""
import copy
import numpy as np
import torch
import yaml
from .stage_a_common import ROOT
from .stage_c_train import load_snapshot, snapshot, train


def config_w():
    return yaml.safe_load((ROOT / "configs/stage_w.yaml").read_text(encoding="utf-8"))


class EarlyStop:
    def __init__(self, patience, min_delta):
        if patience < 1 or min_delta < 0:
            raise ValueError("Invalid early stopping rule")
        self.patience, self.min_delta = patience, min_delta
        self.best = self.reference = float("inf")
        self.best_step, self.bad_checks = 0, 0

    def update(self, step, value):
        if not np.isfinite(value):
            raise ValueError("Non-finite validation loss; do not score as chance")
        improved = value < self.best
        if improved:
            self.best, self.best_step = value, step
        if value < self.reference - self.min_delta:
            self.reference, self.bad_checks = value, 0
        else:
            self.bad_checks += 1
        return improved, self.bad_checks >= self.patience


@torch.no_grad()
def subject_losses(model, subjects, validation):
    rows = []
    for s in subjects:
        indices = s["query"] if validation else torch.arange(len(s["labels"]), device=s["labels"].device)
        z = model.z()
        loss_sum, correct, predicted, observed = 0., 0, [], []
        for take in indices.split(32):
            logits = model(s["eeg"][take].float(), s["picks"], z)
            labels = s["labels"][take]
            loss_sum += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum"))
            predictions = logits.argmax(1)
            correct += int((predictions == labels).sum())
            predicted.extend(predictions.cpu().tolist()); observed.extend(labels.cpu().tolist())
        loss = loss_sum / len(indices)
        if not np.isfinite(loss):
            raise ValueError(f"Non-finite loss: {s['key']}")
        truth, pred = np.asarray(observed), np.asarray(predicted)
        balanced = float(np.mean([(pred[truth == c] == c).mean() for c in np.unique(truth)]))
        rows.append({"dataset": s["key"][0], "subject": s["key"][1], "loss": loss,
                     "accuracy": correct / len(indices), "ba": balanced})
    return rows


def continuation(model, training, validation, hp, c_cfg, w_cfg, seed, save_checkpoint):
    """Same C sampler/loss/parameters/lr; only budget and checkpoint selection change.

    The caller supplies no test subjects. C saved no optimizer state, so AdamW
    and the sampler restart with the same recorded seed; this is not an exact
    resume of C's optimizer trajectory. Step zero can remain the selected G.
    """
    p = w_cfg["population"]
    budget = int(hp["episodes"]) * p["additional_steps_multiplier"]
    cfg = copy.deepcopy(c_cfg)
    cfg["training"]["episodes"] = sorted(set(range(p["check_every_steps"], budget + 1,
                                                    p["check_every_steps"])) | {budget})
    stop = EarlyStop(p["early_stop_patience_checks"], p["early_stop_min_delta"])
    curve, losses, steps = [], [], []
    best_state, last_step = None, 0

    def checkpoint(step):
        nonlocal best_state, last_step
        last_step = step
        for role, subjects, is_validation in [("train", training, False), ("validation", validation, True)]:
            rows = subject_losses(model, subjects, is_validation)
            losses.extend({"additional_step": step, "role": role, **r} for r in rows)
            for dataset in [*sorted({r["dataset"] for r in rows}), "pooled"]:
                values = [r["loss"] for r in rows if dataset == "pooled" or r["dataset"] == dataset]
                curve.append({"additional_step": step, "total_step": int(hp["episodes"]) + step,
                              "role": role, "dataset": dataset, "subject_mean_ce": float(np.mean(values)),
                              "subject_mean_accuracy": float(np.mean([r["accuracy"] for r in rows if dataset == "pooled" or r["dataset"] == dataset])),
                              "subject_mean_ba": float(np.mean([r["ba"] for r in rows if dataset == "pooled" or r["dataset"] == dataset]))})
        value = curve[-1]["subject_mean_ce"]
        improved, done = stop.update(step, value)
        if improved:
            best_state = snapshot(model)
            save_checkpoint(best_state, step)
        return not done

    checkpoint(0)
    train(model, training, hp, cfg, seed, checkpoint,
          on_step=lambda step, key, ce, loss: steps.append({"additional_step": step, "dataset": key[0],
                "subject": key[1], "training_ce": ce, "training_objective": loss}))
    load_snapshot(model, best_state)
    # Descriptive flatness; early stopping can also be caused by overfitting.
    flat = {}
    for role in ["train", "validation"]:
        values = [r["subject_mean_ce"] for r in curve if r["role"] == role and r["dataset"] == "pooled"]
        tail = values[-p["plateau_window_checks"]:]
        flat[role] = len(tail) == p["plateau_window_checks"] and np.ptp(tail) <= max(
            p["early_stop_min_delta"], p["plateau_relative_range"] * abs(np.mean(tail)))
    status = {"original_steps": int(hp["episodes"]), "additional_steps_cap": budget,
              "additional_steps_run": last_step, "selected_additional_step": stop.best_step,
              "selected_validation_ce": stop.best, "early_stopped": stop.bad_checks >= stop.patience,
              "training_curve_flat": bool(flat["train"]), "validation_curve_flat": bool(flat["validation"]),
              "plateau_observed": bool(all(flat.values())), "optimizer_restarted": True,
              "test_used_for_selection": False}
    return curve, losses, steps, status
