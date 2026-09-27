from pathlib import Path
import sys
import hashlib
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


def load_backbone(weights, k=2):
    # MHA's inference fast path differs at rounding level from its grad path.
    # Keep one kernel path so zero FiLM is bit-identical even with autograd on.
    torch.backends.mha.set_fastpath_enabled(False)
    sys.path.insert(0, str(ROOT / "vendor/CBraMod"))
    from models.cbramod import CBraMod
    backbone = CBraMod()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    backbone.proj_out = nn.Identity()
    return FrozenBackbone(backbone, k)


class FrozenBackbone(nn.Module):
    def __init__(self, model, k):
        super().__init__()
        self.model = model
        self.k = k
        assert 2 <= k <= 4
        self.model.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def prefix(self, eeg):
        b, c, t = eeg.shape
        assert t % 200 == 0
        x = self.model.patch_embedding(eeg.reshape(b, c, t // 200, 200))
        for layer in self.model.encoder.layers[:-self.k]:
            x = layer(x)
        return x.detach()

    def suffix(self, x, film=None):
        for i, layer in enumerate(self.model.encoder.layers[-self.k:]):
            x = layer(x)
            if film is not None:
                dg, db = film[i, 0], film[i, 1]
                x = x * (1 + 0.1 * torch.tanh(dg)) + 0.1 * torch.tanh(db)
        return x

    def forward(self, eeg, film=None):
        return self.suffix(self.prefix(eeg), film)


class Adapter(nn.Module):
    def __init__(self, context_dim, feature_dim=200, k=2, z_dim=16, hidden=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(context_dim, hidden), nn.GELU(), nn.Linear(hidden, z_dim))
        self.aggregate = nn.Sequential(nn.Linear(z_dim, z_dim), nn.Tanh())
        self.default_z = nn.Parameter(torch.zeros(z_dim))
        self.generator = nn.Sequential(nn.Linear(z_dim, hidden), nn.GELU(), nn.Linear(hidden, k * 2 * feature_dim))
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)
        self.head = nn.Linear(feature_dim, 2)
        self.k, self.feature_dim = k, feature_dim

    def film(self, context=None):
        z = self.default_z if context is None else self.aggregate(self.encoder(context).mean(dim=0))
        return self.generator(z).reshape(self.k, 2, self.feature_dim)

    def forward(self, backbone, prefix, context=None):
        x = backbone.suffix(prefix, self.film(context)).mean(dim=(1, 2))
        return self.head(x)


def state_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
