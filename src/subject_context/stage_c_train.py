"""Episode training for the Stage C context model; subjects are dicts built by the run script."""
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .stage_b import ba
from .stage_c_context import context_window


KEYS = ["learning_rate", "lambda", "margin", "episodes"]


def select_config(rows):
    """Validation median gain, then mean; ties go to fewer episodes, smaller lr, lambda, margin."""
    grouped = pd.DataFrame(rows).groupby(KEYS, sort=True).gain.agg(["median", "mean"]).reset_index()
    best = grouped.sort_values(["median", "mean", "episodes", "learning_rate", "lambda", "margin"],
                               ascending=[False, False, True, True, True, True], kind="stable").iloc[0]
    return {"learning_rate": float(best.learning_rate), "lambda": float(best["lambda"]),
            "margin": float(best.margin), "episodes": int(best.episodes),
            "validation_median_gain": float(best["median"]), "validation_mean_gain": float(best["mean"])}


def snapshot(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def load_snapshot(model, state):
    params = dict(model.named_parameters())
    assert set(state) == {n for n, p in params.items() if p.requires_grad}
    for name, value in state.items():
        params[name].copy_(value)


def context_z(model, subject, window=None):
    start, n = window if window else (0, len(subject["emb"]))
    return model.z(subject["emb"][start:start+n], subject["tan"][start:start+n])


@torch.no_grad()
def predict(model, subject, indices, z):
    return torch.cat([model(subject["eeg"][take].float(), subject["picks"], z).argmax(1)
                      for take in indices.split(32)]).cpu().numpy()


@torch.no_grad()
def score(model, subject, context=None, default=False):
    """Second-half and all-trial predictions with one z from the full usable context."""
    z = model.z() if default else context_z(model, context or subject)
    everything = torch.arange(len(subject["labels"]), device=subject["labels"].device)
    pred = predict(model, subject, everything, z)
    query = subject["query"].cpu().numpy()
    labels = subject["labels"].cpu().numpy()
    return {"ba": ba(labels[query], pred[query]), "ba_all": ba(labels, pred), "predictions": pred}


def train(model, subjects, hp, cfg, seed, on_checkpoint, on_step=None):
    """subjects: training subjects; the margin donor always comes from the same dataset."""
    t = cfg["training"]
    rng = np.random.default_rng(seed)
    device = subjects[0]["labels"].device
    generator = torch.Generator(device=device).manual_seed(seed)
    by_dataset = {}
    for index, s in enumerate(subjects):
        by_dataset.setdefault(s["key"][0], []).append(index)
    minimum = int(np.ceil(cfg["context"]["min_seconds"] / cfg["context"]["segment_seconds"]))
    optimizer = torch.optim.AdamW(model.trainable(), lr=hp["learning_rate"], weight_decay=t["weight_decay"])
    for episode in range(1, max(t["episodes"]) + 1):
        s = subjects[int(rng.integers(len(subjects)))]
        take = torch.randperm(len(s["labels"]), generator=generator, device=device)[:t["query_batch"]]
        x, y = s["eeg"][take].float(), s["labels"][take]
        dropout = rng.random() < t["context_dropout"]
        z = model.z() if dropout else context_z(model, s, context_window(len(s["emb"]), rng, minimum))
        own = F.cross_entropy(model(x, s["picks"], z), y)
        loss = own
        if not dropout and hp["lambda"] > 0:
            pool = [i for i in by_dataset[s["key"][0]] if subjects[i]["key"] != s["key"]]
            assert pool, f"{s['key']}: no other training subject in the same dataset"
            other = subjects[pool[int(rng.integers(len(pool)))]]
            z_other = context_z(model, other, context_window(len(other["emb"]), rng, minimum))
            foreign = F.cross_entropy(model(x, s["picks"], z_other), y)
            loss = own + hp["lambda"] * F.relu(own - foreign + hp["margin"])
        assert torch.isfinite(loss), (hp, episode)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable(), t["grad_clip_norm"])
        optimizer.step()
        if on_step is not None:
            on_step(episode, s["key"], float(own.detach()), float(loss.detach()))
        if episode in t["episodes"]:
            # Existing C callbacks return None. Only W explicitly returns False
            # after a validation-loss early stop; no historical behavior changes.
            if on_checkpoint(episode) is False:
                break
