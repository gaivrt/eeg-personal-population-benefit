"""Stage C: a set encoder turns unlabeled rest into full-layer FiLM or a LoRA-basis mixture."""
import copy
import math
import torch
from torch import nn
from torch.nn.utils import parametrize
import yaml
from .stage_a_common import ROOT


def config_c():
    return yaml.safe_load((ROOT / "configs/stage_c.yaml").read_text())


def backbone_features(core, eeg, picks, film=None):
    """B2's full-forward path; film is (layers, 2, 200) raw values or None."""
    b, c, samples = eeg.shape
    x = core.patch_embedding(eeg.reshape(b, c, samples // 200, 200))
    for index, layer in enumerate(core.encoder.layers):
        x = layer(x)
        if film is not None:
            x = x * (1 + .1 * torch.tanh(film[index, 0])) + .1 * torch.tanh(film[index, 1])
        if index == 7 and not getattr(core, "budget_retry_fp32", False):
            x = x.half().float()
    return x[:, picks].mean(2).flatten(1)


class ContextEncoder(nn.Module):
    """DeepSets over rest segments: project both views, phi per segment, mean, rho."""
    def __init__(self, embedding_dim, tangent_dim, projection=64, hidden=128, z_dim=16):
        super().__init__()
        self.embedding = nn.Linear(embedding_dim, projection)
        self.tangent = nn.Linear(tangent_dim, projection)
        self.phi = nn.Sequential(nn.Linear(2 * projection, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, z_dim))

    def forward(self, embedding, tangent):
        segments = torch.cat([self.embedding(embedding), self.tangent(tangent)], 1)
        return self.rho(self.phi(segments).mean(0))


class Mixture:
    """Holds the current subject's basis weights; deliberately not an nn.Module."""
    w = None


class MixedLowRank(nn.Module):
    """W + sum_i w_i B_i A_i for each packed projection block."""
    def __init__(self, width, rank, bases, blocks, mixture, b_std):
        super().__init__()
        self.a = nn.Parameter(torch.empty(bases, blocks, rank, width))
        self.b = nn.Parameter(torch.randn(bases, blocks, width, rank) * b_std)
        for block in self.a.view(-1, rank, width):
            nn.init.kaiming_uniform_(block, a=math.sqrt(5))
        self.mixture = mixture

    def forward(self, weight):
        delta = torch.einsum("k,kbwr,kbrv->bwv", self.mixture.w, self.b, self.a)
        return weight + delta.reshape_as(weight)


class ContextModel(nn.Module):
    def __init__(self, backbone, readout, method, embedding_dim, tangent_dim, cfg):
        super().__init__()
        self.method = method
        self.core = copy.deepcopy(backbone.model) if method == "m_lora" else backbone.model
        self.readout = copy.deepcopy(readout)
        self.readout.head.requires_grad_(True)
        e = cfg["encoder"]
        self.encoder = ContextEncoder(embedding_dim, tangent_dim, e["projection"], e["hidden"], e["z_dim"])
        self.default_z = nn.Parameter(torch.zeros(e["z_dim"]))
        layers, width = len(self.core.encoder.layers), 200  # layer output width, as in B2
        out = layers * 2 * width if method == "m_film" else cfg["lora"]["bases"]
        self.generator = nn.Sequential(nn.Linear(e["z_dim"], cfg["generator_hidden"]), nn.GELU(),
                                       nn.Linear(cfg["generator_hidden"], out))
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)
        self.shape = (layers, 2, width)
        if method == "m_lora":
            lora = cfg["lora"]
            device = next(self.core.parameters()).device
            self.mixture = Mixture()
            self.mixture.w = torch.zeros(lora["bases"], device=device)  # registration evaluates once
            for layer in self.core.encoder.layers:
                for name in ["self_attn_s", "self_attn_t"]:
                    attention = getattr(layer, name)
                    dim = attention.embed_dim
                    parametrize.register_parametrization(attention, "in_proj_weight", MixedLowRank(
                        dim, lora["rank"], lora["bases"], 3, self.mixture, lora["b_init_std"]).to(device))
                    parametrize.register_parametrization(attention.out_proj, "weight", MixedLowRank(
                        dim, lora["rank"], lora["bases"], 1, self.mixture, lora["b_init_std"]).to(device))
        self.eval()

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def z(self, embedding=None, tangent=None):
        return self.default_z if embedding is None else self.encoder(embedding, tangent)

    def forward(self, eeg, picks, z):
        adaptation = self.generator(z)
        if self.method == "m_film":
            return self.readout(backbone_features(self.core, eeg, picks, adaptation.reshape(self.shape)))
        self.mixture.w = adaptation
        return self.readout(backbone_features(self.core, eeg, picks))
