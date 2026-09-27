"""Stage C4: a frozen prior from other users' personal parameters, then few-shot calibration toward it."""
import copy
import math
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import parametrize
import yaml
from .stage_a_common import ROOT
from .stage_b import ba
from .stage_b2 import LowRankWeight
from .stage_c import backbone_features
from .stage_c2 import predict
from .stage_c3 import average_personal


def config_c4():
    return yaml.safe_load((ROOT / "configs/stage_c4.yaml").read_text())


def select_hp(rows):
    """Validation median gain, then mean; ties go to fewer steps, smaller lr, smaller mu."""
    g = pd.DataFrame(rows).groupby(["learning_rate", "mu", "steps"]).gain.agg(["median", "mean"]).reset_index()
    best = g.sort_values(["median", "mean", "steps", "learning_rate", "mu"],
                         ascending=[False, False, True, True, True], kind="stable").iloc[0]
    return {"learning_rate": float(best.learning_rate), "mu": float(best.mu), "steps": int(best.steps),
            "validation_median_gain": float(best["median"]), "validation_mean_gain": float(best["mean"])}


def select_k_alpha(candidates):
    """candidates: {(k, alpha): [gain per pool member]}; median, then mean; ties -> smaller alpha, then k."""
    return min(candidates, key=lambda c: (-np.median(candidates[c]), -np.mean(candidates[c]), c[1], c[0]))


def make_prior(variant, params, alpha):
    """alpha x mean personal parameters of the chosen users (LoRA: mean of B·A, B scaled by alpha)."""
    merged = average_personal(variant, params)
    if variant == "film_offset":
        return [merged[0] * alpha]
    return [p * alpha if i % 2 else p for i, p in enumerate(merged)]


class CalibModel(nn.Module):
    """B3 + frozen prior + trainable personal part (zero at reset, so the start equals the prior)."""
    def __init__(self, base, variant, prior_rank=40, rank=8):
        super().__init__()
        assert (variant == "film_offset") == (base.method == "m_film")
        self.variant = variant
        if variant == "lora8":
            base.mixture.w = base.mixture.w.detach()
            base = copy.deepcopy(base)
        self.base = base
        self.base.requires_grad_(False)
        with torch.no_grad():
            self.register_buffer("default", self.base.generator(self.base.default_z).detach().clone())
        device = self.default.device
        if variant == "film_offset":
            self.register_buffer("prior", torch.zeros_like(self.default))
            self.delta = nn.Parameter(torch.zeros_like(self.default))
            return
        self.priors, self.personal = nn.ModuleList(), nn.ModuleList()
        for layer in self.base.core.encoder.layers:
            for name in ["self_attn_s", "self_attn_t"]:
                attention = getattr(layer, name)
                for owner, tensor, blocks in [(attention, "in_proj_weight", 3), (attention.out_proj, "weight", 1)]:
                    prior = LowRankWeight(attention.embed_dim, prior_rank, blocks).to(device).requires_grad_(False)
                    with torch.no_grad():
                        prior.b.zero_()
                    own = LowRankWeight(attention.embed_dim, rank, blocks).to(device)
                    parametrize.register_parametrization(owner, tensor, prior)
                    parametrize.register_parametrization(owner, tensor, own)
                    self.priors.append(prior); self.personal.append(own)
        self.eval()

    def trainable(self):
        return [self.delta] if self.variant == "film_offset" else list(self.personal.parameters())

    @torch.no_grad()
    def set_prior(self, prior=None):
        """prior: output of make_prior, or None for P0."""
        if self.variant == "film_offset":
            self.prior.copy_(prior[0] if prior is not None else torch.zeros_like(self.prior))
            return
        for index, module in enumerate(self.priors):
            module.b.zero_()
            if prior is not None:
                a, b = prior[2 * index], prior[2 * index + 1]
                r = a.shape[1]
                assert r <= module.a.shape[1], "prior rank exceeds the reserved rank"
                module.a[:, :r].copy_(a)
                module.b[:, :, :r].copy_(b)

    @torch.no_grad()
    def reset(self):
        if self.variant == "film_offset":
            self.delta.zero_()
            return
        for module in self.personal:
            for block in module.a:
                nn.init.kaiming_uniform_(block, a=math.sqrt(5))
            module.b.zero_()

    def penalty(self):
        """Squared distance of the calibrated model from the prior, in adapted-parameter space."""
        if self.variant == "film_offset":
            return self.delta.pow(2).sum()
        return sum(torch.bmm(m.b, m.a).pow(2).sum() for m in self.personal)

    def forward(self, eeg, picks):
        base = self.base
        if self.variant == "film_offset":
            film = (self.default + self.prior + self.delta).reshape(base.shape)
            return base.readout(backbone_features(base.core, eeg, picks, film))
        base.mixture.w = self.default
        return base.readout(backbone_features(base.core, eeg, picks))


def calibrate(model, subject, n, lr, mu, steps, seed):
    """Fit the personal part on the earliest n first-half trials; score the second half per step."""
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        model.reset()
    optimizer = torch.optim.AdamW(model.trainable(), lr=lr, weight_decay=0.0)
    rng = torch.Generator(device="cuda").manual_seed(seed)
    rows, result = subject["fit"][:n], {}
    for step in range(1, max(steps) + 1):
        take = rows[torch.randperm(len(rows), generator=rng, device="cuda")[:32]]
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(subject["eeg"][take].float(), subject["picks"]), subject["labels"][take])
        loss = loss + mu * model.penalty()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        if step in steps:
            result[step] = ba(subject["query_labels"], predict(model, subject, subject["query"]))
    torch.cuda.synchronize()
    return result, {"fit_seconds": time.monotonic() - start, "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
