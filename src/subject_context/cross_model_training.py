"""Validation records and numerical steps; original C/W selection rules remain separate."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .stage_c_train import select_config
from .stage_w import EarlyStop  # Same CE patience semantics as the completed W runs.


def record_events(step, total_steps, train_subjects, phase):
    if train_subjects < 1 or not 0 <= step <= total_steps:
        raise ValueError("Invalid logging interval")
    return {"epoch_record": step == 0 or step % train_subjects == 0 or step == total_steps,
            "partial_epoch": step == total_steps and step % train_subjects != 0,
            "epoch": step / train_subjects,
            "W_check": phase == "continuation" and (step % 250 == 0 or step == total_steps),
            "initial_selection_candidate": phase == "initial" and step in (1000, 3000)}


def select_population(rows):
    # Extra epoch/check logs must not enlarge the original 1000/3000 search grid.
    candidates = [dict(r, **{"lambda": 0., "margin": 0.}) for r in rows if r["episodes"] in (1000, 3000)]
    selected = select_config(candidates)
    return {k: v for k, v in selected.items() if k not in ("lambda", "margin")}


def update(model, subject, indices, optimizer, microbatch=32, clip=None):
    """One optimizer update with the same effective batch, optionally split for memory."""
    if len(indices) == 0 or microbatch < 1:
        raise ValueError("Empty update or invalid microbatch")
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    for take in indices.split(microbatch):
        loss = F.cross_entropy(model(subject["eeg"][take].float(), subject["picks"]),
                               subject["labels"][take], reduction="sum") / len(indices)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        total += float(loss.detach())
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.trainable(), clip, error_if_nonfinite=True)
    optimizer.step()
    return total


@torch.no_grad()
def evaluate(model, subjects, subset="query", batch_size=32):
    """Per-subject CE/accuracy/BA plus sufficient counts to recompute every metric."""
    rows = []
    for subject in subjects:
        ids = subject["query"] if subset == "query" else torch.arange(len(subject["labels"]), device=subject["labels"].device)
        if subset not in ("query", "all"):
            raise ValueError(subset)
        predictions, truth, ce = [], [], 0.
        for take in ids.split(batch_size):
            logits = model(subject["eeg"][take].float(), subject["picks"])
            target = subject["labels"][take]
            loss = F.cross_entropy(logits, target, reduction="sum")
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise ValueError("Nonfinite validation output")
            ce += float(loss)
            predictions.extend(logits.argmax(1).cpu().tolist())
            truth.extend(target.cpu().tolist())
        confusion = np.bincount(np.asarray(truth) * 2 + predictions, minlength=4).reshape(2, 2)
        if not np.all(confusion.sum(1) > 0):
            raise ValueError("Query/evaluation half lacks a class; do not silently change BA definition")
        rows.append({"dataset": subject["key"][0], "subject": subject["key"][1], "subset": subset,
                     "n": len(truth), "CE_sum": ce, "CE": ce / len(truth),
                     "accuracy": float(np.trace(confusion) / len(truth)),
                     "BA": float(np.mean(np.diag(confusion) / confusion.sum(1))),
                     "confusion": confusion.tolist(), "indices": ids.cpu().tolist(), "predictions": predictions})
    return rows


def summarize(rows):
    if not rows:
        raise ValueError("No validation subjects")
    return {name: {m: float(np.mean([r[m] for r in rows if name == "pooled" or r["dataset"] == name]))
                   for m in ("CE", "accuracy", "BA")}
            for name in [*sorted({r["dataset"] for r in rows}), "pooled"]}


def append_validation(path, context, people):
    """One append-only JSONL record; never overwrite past curves with the best checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**context, "metrics": summarize(people), "subjects": people}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_training_state(path, model, optimizer, numpy_rng, torch_generator, step, metadata):
    """Compact adapter/head state plus the exact optimizer/sampler continuation state."""
    state = {"parameters": {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
             "optimizer": copy.deepcopy(optimizer.state_dict()), "numpy_rng": copy.deepcopy(numpy_rng.bit_generator.state),
             "torch_generator": torch_generator.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
             "step": step, "metadata": metadata}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_training_state(path, model, optimizer, numpy_rng, torch_generator):
    state = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(params) != set(state["parameters"]):
        raise ValueError("Checkpoint trainable parameter set changed")
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(state["parameters"][n])
    optimizer.load_state_dict(state["optimizer"])
    numpy_rng.bit_generator.state = state["numpy_rng"]
    torch_generator.set_state(state["torch_generator"].cpu())
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    return state["step"], state["metadata"]
