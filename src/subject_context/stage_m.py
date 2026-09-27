"""Stage M: first-order MAML over B3's population LoRA and head, with a zero-started personal LoRA r=8."""
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml
from .stage_a_common import ROOT
from .stage_b import ba
from .stage_c import MixedLowRank
from .stage_c2 import predict

LAYERS = 12
PARAMS_PER_LAYER = 8  # CalibModel.personal: (spatial, temporal) x (in_proj, out_proj) modules, each (a, b)
CHANCE = 0.5
divergences = []  # (n, steps done) of inner loops stopped by a non-finite loss; read by the run script


def config_m():
    return yaml.safe_load((ROOT / "configs/stage_m.yaml").read_text())


def population(model):
    """{name: parameter} of the population part inside a CalibModel: B3's LoRA bases and the task head."""
    base = model.base
    bases = {id(p) for m in base.core.modules() if isinstance(m, MixedLowRank) for p in m.parameters()}
    head = {id(p) for p in base.readout.head.parameters()}
    return {n: p for n, p in base.named_parameters() if id(p) in bases | head}


def population_state(model):
    return {n: p.detach().cpu().clone() for n, p in population(model).items()}


@torch.no_grad()
def load_population(model, state):
    params = population(model)
    assert set(state) == set(params)
    for name, value in state.items():
        params[name].copy_(value)


def layer_lr(lr, count):
    """Per-parameter learning rates: a float, or one value per encoder layer."""
    if not torch.is_tensor(lr):
        return [float(lr)] * count
    assert lr.shape == (LAYERS,) and count == LAYERS * PARAMS_PER_LAYER
    return [lr[j // PARAMS_PER_LAYER] for j in range(count)]


def inner_loop(model, eeg, picks, labels, steps, lr, on_step=None):
    """Plain SGD on the personal LoRA with the full support batch; returns the summed inner gradients,
    or None if the loss became non-finite (the loop stops there)."""
    personal = model.trainable()
    rates = layer_lr(lr, len(personal))
    total = [torch.zeros_like(p) for p in personal]
    for step in range(1, steps + 1):
        loss = nn.functional.cross_entropy(model(eeg, picks), labels)
        if not torch.isfinite(loss):
            divergences.append((len(labels), step - 1))
            return None
        grads = torch.autograd.grad(loss, personal)
        with torch.no_grad():
            for p, g, rate, t in zip(personal, grads, rates, total):
                p.sub_(rate * g)
                t.add_(g)
        if on_step:
            on_step(step)
    return total


@torch.no_grad()
def log_lr_grad(personal, total, rates):
    """First order: d theta_K / d lr_l = -sum_k g_k on layer l, so dL/d log lr_l = -lr_l <grad L, sum_k g_k>_l."""
    dots = torch.stack([(p.grad * t).sum() for p, t in zip(personal, total)])
    return -rates * dots.view(LAYERS, PARAMS_PER_LAYER).sum(1)


def reset_personal(model, seed):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
        torch.manual_seed(seed)
        model.reset()


def adapt(model, subject, n, ks, lr, seed):
    """Few-shot calibration from the current start: earliest n first-half trials, K SGD steps; BA per K.
    A diverged loop scores chance (0.5) at every K it did not reach."""
    reset_personal(model, seed)
    if n == 0:
        return {0: ba(subject["query_labels"], predict(model, subject, subject["query"]))}
    rows = subject["fit"][:n]
    result = {}

    def score(step):
        if step in ks:
            result[step] = ba(subject["query_labels"], predict(model, subject, subject["query"]))
    inner_loop(model, subject["eeg"][rows].float(), subject["picks"], subject["labels"][rows], max(ks), lr, score)
    return {k: result.get(k, CHANCE) for k in ks}


def meta_train(model, subjects, k, lr, cfg, seed, log_lr=None, on_checkpoint=None):
    """First-order MAML. log_lr (12 values) given -> M2 learns per-layer inner learning rates."""
    o = cfg["outer"]
    params = population(model)
    for p in params.values():
        p.requires_grad_(True)
    outer = torch.optim.AdamW(params.values(), lr=o["learning_rate"], weight_decay=o["weight_decay"])
    rate_opt = None if log_lr is None else torch.optim.AdamW([log_lr], lr=o["m2_log_lr_learning_rate"], weight_decay=0.0)
    rng = np.random.default_rng(seed)
    device = subjects[0]["labels"].device
    generator = torch.Generator(device=device).manual_seed(seed)
    log, personal = [], model.trainable()
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        for step in range(1, max(o["steps"]) + 1):
            s = subjects[int(rng.integers(len(subjects)))]
            n = int(rng.choice(o["episode_shots"]))
            support = s["fit"][:n]
            query = s["query"][torch.randperm(len(s["query"]), generator=generator, device=device)[:32]]
            model.reset()
            rates = lr if log_lr is None else log_lr.detach().exp()
            total = inner_loop(model, s["eeg"][support].float(), s["picks"], s["labels"][support], k, rates)
            if total is None:  # diverged inner loop: no outer update for this episode
                log.append({"step": step, "shots": n, "dataset": s["key"][0], "query_loss": float("nan"), "skipped": True})
                if on_checkpoint and step in o["steps"]:
                    on_checkpoint(step)
                continue
            outer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(s["eeg"][query].float(), s["picks"]), s["labels"][query])
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(list(params.values()), o["grad_clip_norm"])
            outer.step()
            if rate_opt is not None:
                log_lr.grad = log_lr_grad(personal, total, rates)
                rate_opt.step(); rate_opt.zero_grad(set_to_none=True)
            for p in personal:
                p.grad = None
            log.append({"step": step, "shots": n, "dataset": s["key"][0], "query_loss": float(loss), "skipped": False})
            if on_checkpoint and step in o["steps"]:
                on_checkpoint(step)
    for p in params.values():
        p.requires_grad_(False)
        p.grad = None
    model.reset()
    return log


def continue_train(model, subjects, steps, cfg, seed, on_checkpoint=None):
    """R2: B3's original default-z objective (CE on 32 random trials of a random training subject)."""
    o = cfg["outer"]
    params = population(model)
    for p in params.values():
        p.requires_grad_(True)
    for p in model.trainable():
        p.requires_grad_(False)
    model.reset()  # personal B = 0: the forward equals the population model
    optimizer = torch.optim.AdamW(params.values(), lr=o["learning_rate"], weight_decay=o["weight_decay"])
    rng = np.random.default_rng(seed)
    device = subjects[0]["labels"].device
    generator = torch.Generator(device=device).manual_seed(seed)
    log = []
    for step in range(1, max(steps) + 1):
        s = subjects[int(rng.integers(len(subjects)))]
        take = torch.randperm(len(s["labels"]), generator=generator, device=device)[:32]
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(s["eeg"][take].float(), s["picks"]), s["labels"][take])
        assert torch.isfinite(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(list(params.values()), o["grad_clip_norm"])
        optimizer.step()
        log.append({"step": step, "dataset": s["key"][0], "loss": float(loss)})
        if on_checkpoint and step in steps:
            on_checkpoint(step)
    for p in params.values():
        p.requires_grad_(False)
        p.grad = None
    for p in model.trainable():
        p.requires_grad_(True)
    return log


def select_inner(rows):
    """rows: {k, learning_rate, gain}; validation median gain, then mean; ties -> smaller k, smaller lr."""
    g = pd.DataFrame(rows).groupby(["k", "learning_rate"]).gain.agg(["median", "mean"]).reset_index()
    best = g.sort_values(["median", "mean", "k", "learning_rate"], ascending=[False, False, True, True],
                         kind="stable").iloc[0]
    return {"k": int(best.k), "learning_rate": float(best.learning_rate),
            "validation_median_gain": float(best["median"]), "validation_mean_gain": float(best["mean"])}


def select_meta(rows):
    """rows: {k, steps, gain} at n=10; median, then mean; ties -> smaller k, fewer steps."""
    g = pd.DataFrame(rows).groupby(["k", "steps"]).gain.agg(["median", "mean"]).reset_index()
    best = g.sort_values(["median", "mean", "k", "steps"], ascending=[False, False, True, True], kind="stable").iloc[0]
    return {"k": int(best.k), "steps": int(best.steps),
            "validation_median_gain": float(best["median"]), "validation_mean_gain": float(best["mean"])}


def pilot_grid(rows, scan):
    """Best scanned lr (median gain over ks, then mean) and its two scan neighbours."""
    g = pd.DataFrame(rows).groupby("learning_rate").gain.agg(["median", "mean"]).reset_index()
    best = float(g.sort_values(["median", "mean", "learning_rate"], ascending=[False, False, True],
                               kind="stable").iloc[0].learning_rate)
    i = scan.index(best)
    return scan[max(0, i - 1):i + 2]


def timed(fn):
    """Run fn and return (value, seconds, peak GPU bytes)."""
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
    value = fn()
    torch.cuda.synchronize()
    return value, time.monotonic() - start, torch.cuda.max_memory_allocated()
