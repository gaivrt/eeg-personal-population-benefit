"""Stage B temporal split and direct, bounded per-person adaptation."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml
from .stage_a_common import ROOT


def config_b():
    return yaml.safe_load((ROOT / "configs/stage_b.yaml").read_text())


def temporal_halves(trials):
    assert not trials.trial_id.duplicated().any()
    assert set(trials.label) == {0, 1}
    ordered = trials.assign(array_index=np.arange(len(trials))).sort_values(
        ["protocol_order", "start_seconds"], kind="stable")
    assert not ordered.duplicated(["protocol_order", "start_seconds"]).any()
    n = len(ordered) // 2
    fit, query = ordered.iloc[:n], ordered.iloc[n:]
    assert len(fit) and len(query)
    a, b = fit.iloc[-1], query.iloc[0]
    assert a.protocol_order < b.protocol_order or a.stop_seconds < b.start_seconds, "temporal overlap"
    counts = {}
    for name, half in [("fit", fit), ("query", query)]:
        left, right = int((half.label == 0).sum()), int((half.label == 1).sum())
        counts.update({name+"_left": left, name+"_right": right,
                       name+"_balanced": min(left,right)/len(half) >= .4})
        assert left and right, f"{name}: one class absent"
    return fit.array_index.to_numpy(), query.array_index.to_numpy(), counts


def ba(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    assert set(y) == {0, 1}
    return float(np.mean([np.mean(pred[y == k] == k) for k in [0, 1]]))


def select_candidate(rows):
    grouped = pd.DataFrame(rows).groupby(["learning_rate", "weight_decay", "steps"], sort=True).gain.agg(["median", "mean"]).reset_index()
    best = grouped.sort_values(["median", "mean", "steps", "learning_rate", "weight_decay"],
                              ascending=[False, False, True, True, True], kind="stable").iloc[0]
    return {"learning_rate":float(best.learning_rate), "weight_decay":float(best.weight_decay),
            "steps":int(best.steps), "validation_median_gain":float(best["median"]),
            "validation_mean_gain":float(best["mean"])}


class Readout(nn.Module):
    def __init__(self, checkpoint, kind):
        super().__init__()
        width = len(checkpoint["feature_mean"])
        self.head = (nn.Linear(width, 2) if kind == "linear" else
                     nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 2)))
        self.head.load_state_dict(checkpoint["head"])
        self.register_buffer("mean", torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float64))
        self.register_buffer("std", torch.as_tensor(checkpoint["feature_std"], dtype=torch.float64))
        self.requires_grad_(False)
        self.eval()

    def normalize(self, features):
        # Match A's float64 train-only normalization followed by float32 head input.
        return ((features.double() - self.mean) / self.std).float()

    def forward(self, features):
        return self.head(self.normalize(features))


class DirectAdapter(nn.Module):
    def __init__(self, backbone, readout, picks, variant):
        super().__init__()
        self.variant, self.picks = variant, picks
        self.readout = copy.deepcopy(readout) if variant == "head" else readout
        if variant == "head":
            self.readout.head.requires_grad_(True)
            self.layers = nn.ModuleList()
        else:
            k = 2 if variant == "film2" else 4
            self.layers = nn.ModuleList(list(backbone.model.encoder.layers[-k:]))
            self.raw = nn.Parameter(torch.zeros(k, 1 if variant == "beta4" else 2, 200))
        self.eval()

    def trainable(self):
        return list(self.readout.head.parameters()) if self.variant == "head" else [self.raw]

    def forward(self, x):
        if self.variant == "head":
            return self.readout(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.variant == "beta4":
                x = x + .1 * torch.tanh(self.raw[i, 0])
            else:
                x = x * (1 + .1 * torch.tanh(self.raw[i, 0])) + .1 * torch.tanh(self.raw[i, 1])
        return self.readout(x[:, self.picks].mean(2).flatten(1))


@torch.no_grad()
def predict(model, x, indices):
    return torch.cat([model(x[take]).argmax(1) for take in indices.split(32)]).cpu().numpy()


def fit_path(backbone, readout, subject, variant, lr, wd, steps, seed):
    model = DirectAdapter(backbone, readout, subject["picks"], variant).cuda()
    x = subject["b0"] if variant == "head" else subject["p10" if variant == "film2" else "p8"]
    fit, query = subject["fit"], subject["query"]
    optimizer = torch.optim.AdamW(model.trainable(), lr=lr, weight_decay=wd)
    rng = torch.Generator(device="cuda").manual_seed(seed)
    result = {}
    for step in range(1, max(steps)+1):
        take = fit[torch.randperm(len(fit), generator=rng, device="cuda")[:32]]
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x[take]), subject["labels"][take])
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        if step in steps:
            pred = predict(model, x, query)
            result[step] = {"ba": ba(subject["query_labels"], pred), "predictions":pred}
    params = model.raw.detach().cpu() if variant != "head" else None
    return result, params
