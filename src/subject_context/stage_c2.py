"""Stage C2: per-person parameters on top of Stage C's default-z model (B3)."""
import copy
import math
import time
import torch
from torch import nn
from torch.nn.utils import parametrize
import yaml
from .stage_a_common import ROOT
from .stage_b import ba
from .stage_b2 import LowRankWeight
from .stage_c import backbone_features


def config_c2():
    return yaml.safe_load((ROOT / "configs/stage_c2.yaml").read_text())


class PersonalModel(nn.Module):
    """B3 plus one zero-initialized personal parameter set; everything else frozen."""
    def __init__(self, base, variant, rank=8, *, allow_film_on_lora=False):
        super().__init__()
        self.film_on_lora = bool(allow_film_on_lora and variant == "film_offset" and base.method == "m_lora")
        assert (variant == "film_offset") == (base.method == "m_film") or self.film_on_lora
        self.variant = variant
        if variant == "lora8":
            base.mixture.w = base.mixture.w.detach()  # a non-leaf from a grad forward cannot be deep-copied
        self.base = copy.deepcopy(base) if variant == "lora8" else base
        self.base.requires_grad_(False)
        with torch.no_grad():
            self.register_buffer("default", self.base.generator(self.base.default_z).detach().clone())
        device = self.default.device
        if variant == "lora8":
            self.lora = []
            for layer in self.base.core.encoder.layers:
                for name in ["self_attn_s", "self_attn_t"]:
                    attention = getattr(layer, name)
                    for owner, tensor, blocks in [(attention, "in_proj_weight", 3), (attention.out_proj, "weight", 1)]:
                        module = LowRankWeight(attention.embed_dim, rank, blocks).to(device)
                        parametrize.register_parametrization(owner, tensor, module)
                        self.lora.append(module)
            self.lora = nn.ModuleList(self.lora)
        else:
            self.delta = nn.Parameter(torch.zeros(math.prod(base.shape), device=device) if self.film_on_lora
                                      else torch.zeros_like(self.default))
        self.eval()

    def trainable(self):
        return list(self.lora.parameters()) if self.variant == "lora8" else [self.delta]

    @torch.no_grad()
    def reset(self):
        """Zero personal adaptation (= B3); LoRA A redrawn from the current RNG, B zero."""
        if self.variant == "lora8":
            for module in self.lora:
                for block in module.a:
                    nn.init.kaiming_uniform_(block, a=math.sqrt(5))
                module.b.zero_()
        else:
            self.delta.zero_()

    def personal(self):
        return [p.detach().clone() for p in self.trainable()]

    @torch.no_grad()
    def load_personal(self, values):
        for p, v in zip(self.trainable(), values, strict=True):
            p.copy_(v)

    def forward(self, eeg, picks):
        base = self.base
        if self.variant == "film_offset":
            if self.film_on_lora:
                base.mixture.w = self.default
                film = self.delta.reshape(base.shape)
            else:
                film = (self.default + self.delta).reshape(base.shape)
            return base.readout(backbone_features(base.core, eeg, picks, film))
        base.mixture.w = self.default + self.delta if self.variant == "mix_offset" else self.default
        return base.readout(backbone_features(base.core, eeg, picks))


@torch.no_grad()
def predict(model, subject, indices):
    return torch.cat([model(subject["eeg"][t].float(), subject["picks"]).argmax(1)
                      for t in indices.split(32)]).cpu().numpy()


def fit(model, subject, lr, wd, steps, seed, microbatch=32):
    """Fit on the first half; score the second half at each requested step."""
    device = subject["labels"].device
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        model.reset()
    optimizer = torch.optim.AdamW(model.trainable(), lr=lr, weight_decay=wd)
    rng = torch.Generator(device=device).manual_seed(seed)
    fit_rows, result = subject["fit"], {}
    for step in range(1, max(steps) + 1):
        take = fit_rows[torch.randperm(len(fit_rows), generator=rng, device=device)[:32]]
        optimizer.zero_grad(set_to_none=True)
        if microbatch < 1:
            raise ValueError("microbatch must be positive")
        if len(take) <= microbatch:
            loss = nn.functional.cross_entropy(model(subject["eeg"][take].float(), subject["picks"]), subject["labels"][take])
            assert torch.isfinite(loss)
            loss.backward()
        else:
            for chunk in take.split(microbatch):
                loss = nn.functional.cross_entropy(model(subject["eeg"][chunk].float(), subject["picks"]),
                                                    subject["labels"][chunk], reduction="sum") / len(take)
                assert torch.isfinite(loss)
                loss.backward()
        optimizer.step()
        if step in steps:
            pred = predict(model, subject, subject["query"])
            result[step] = {"ba": ba(subject["query_labels"], pred), "predictions": pred}
    assert all(torch.isfinite(p).all() for p in model.trainable())
    if device.type == "cuda":
        torch.cuda.synchronize()
    resource ={"fit_seconds": time.monotonic() - start,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}
    return result, model.personal(), resource
