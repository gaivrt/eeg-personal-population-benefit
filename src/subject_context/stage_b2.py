"""Full-forward personal adapters; no stored activations enter Stage B2."""
import copy
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
import yaml
from .stage_a_common import ROOT
from .stage_b import ba


def config_b2():
    return yaml.safe_load((ROOT / "configs/stage_b2.yaml").read_text())


class LowRankWeight(nn.Module):
    """Independent low-rank updates for each packed Q/K/V matrix."""
    def __init__(self, width, rank, blocks=1):
        super().__init__()
        self.a = nn.Parameter(torch.empty(blocks, rank, width))
        self.b = nn.Parameter(torch.zeros(blocks, width, rank))
        for block in self.a:
            nn.init.kaiming_uniform_(block, a=math.sqrt(5))

    def forward(self, weight):
        return weight + torch.bmm(self.b, self.a).reshape_as(weight)


class FullAdapter(nn.Module):
    def __init__(self, backbone, readout, picks, variant):
        super().__init__()
        self.variant, self.picks = variant, picks
        self.core = copy.deepcopy(backbone.model) if variant.startswith("lora") else backbone.model
        self.readout = copy.deepcopy(readout) if variant == "scratch_head" else readout
        self.film_layers = ([0, 1] if variant == "film_first2" else
                            list(range(len(self.core.encoder.layers))) if variant == "film_all" else [])
        if self.film_layers:
            self.raw = nn.Parameter(torch.zeros(len(self.film_layers), 2, 200))
        if variant.startswith("lora"):
            rank = int(variant[4:])
            for layer in self.core.encoder.layers:
                for name in ["self_attn_s", "self_attn_t"]:
                    attention = getattr(layer, name)
                    width = attention.embed_dim
                    parametrize.register_parametrization(attention, "in_proj_weight",
                                                         LowRankWeight(width, rank, blocks=3).to(attention.in_proj_weight.device))
                    parametrize.register_parametrization(attention.out_proj, "weight",
                                                         LowRankWeight(width, rank).to(attention.out_proj.weight.device))
        if variant == "scratch_head":
            for module in self.readout.head.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
            self.readout.head.requires_grad_(True)
        self.eval()

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, eeg):
        b, c, samples = eeg.shape
        x = self.core.patch_embedding(eeg.reshape(b, c, samples // 200, 200))
        for index, layer in enumerate(self.core.encoder.layers):
            x = layer(x)
            if index in self.film_layers:
                raw = self.raw[self.film_layers.index(index)]
                x = x * (1 + .1 * torch.tanh(raw[0])) + .1 * torch.tanh(raw[1])
            if index == 7:
                # Reproduce B0's prefix precision without reading/writing any activation cache.
                # PyTorch's dtype cast propagates the gradient to the full preceding graph.
                x = x.half().float()
        return self.readout(x[:, self.picks].mean(2).flatten(1))

    def adaptation_state(self):
        return {name: p.detach().cpu() for name, p in self.named_parameters() if p.requires_grad}


@torch.no_grad()
def predict(model, eeg, indices):
    return torch.cat([model(eeg[take]).argmax(1) for take in indices.split(32)]).cpu().numpy()


def fit_path(backbone, readout, subject, variant, lr, wd, steps, seed):
    torch.cuda.synchronize()
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    # A/B minibatch stream is preserved; initialization has a separate, repeatable stream.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = FullAdapter(backbone, readout, subject["picks"], variant).cuda()
    params = model.trainable()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    rng = torch.Generator(device="cuda").manual_seed(seed)
    fit, query = subject["fit"], subject["query"]
    result = {}
    for step in range(1, max(steps) + 1):
        take = fit[torch.randperm(len(fit), generator=rng, device="cuda")[:32]]
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(subject["eeg"][take]), subject["labels"][take])
        assert torch.isfinite(loss), (variant, lr, wd, step)
        loss.backward()
        optimizer.step()
        if step in steps:
            pred = predict(model, subject["eeg"], query)
            result[step] = {"ba": ba(subject["query_labels"], pred), "predictions": pred}
    assert torch.stack([torch.isfinite(p).all() for p in params]).all()
    torch.cuda.synchronize()
    resource = {"fit_seconds": time.monotonic() - start,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                "trainable_parameters": sum(p.numel() for p in params)}
    return result, model.adaptation_state(), resource
