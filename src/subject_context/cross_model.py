"""Pinned REVE/LaBraM interfaces for the existing frozen-backbone LoRA diagnosis."""
import ast
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
import yaml

from .stage_a_common import ROOT, sha256
from .stage_b2 import LowRankWeight

AUDIT = ROOT / "docs/cross_model_audit_2026-09-27"


def config_x():
    return yaml.safe_load((ROOT / "configs/cross_model_x.yaml").read_text(encoding="utf-8"))


def source(name):
    rows = json.loads((AUDIT / "source_manifest.json").read_text(encoding="utf-8"))["files"]
    row = next(r for r in rows if r["file"] == f"sources/{name}")
    path = AUDIT / row["file"]
    if sha256(path) != row["sha256"]:
        raise ValueError(f"Pinned upstream source changed: {path}")
    return path


def import_source(module_name, filename):
    path = source(filename)
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def official_classes():
    # Import the audited files verbatim, including HF's relative configuration import.
    package = "_eeg_x_reve"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(AUDIT / "sources")]
        sys.modules[package] = module
    config = import_source(f"{package}.configuration_reve", "reve_hf_configuration_reve.py")
    reve = import_source(f"{package}.modeling_reve", "reve_hf_modeling_reve.py")
    labram = import_source("_eeg_x_labram", "labram_modeling_finetune.py")
    return config.ReveConfig, reve.Reve, labram


def channel_tables():
    names = json.loads(source("reve_positions_config.json").read_text())["position_names"]
    tree = ast.parse(source("labram_utils.py").read_text(encoding="utf-8"))
    standard = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "standard_1020" for t in n.targets))
    return names, standard


class CrossBackbone(nn.Module):
    def __init__(self, name, core, positions=None):
        super().__init__()
        if name not in ("REVE", "LaBraM"):
            raise ValueError(name)
        self.name, self.core = name, core
        self.width = core.embed_dim
        self.reve_names, self.labram_names = channel_tables()
        self.labram_indices = {c.lower(): i + 1 for i, c in enumerate(self.labram_names)}
        self.register_buffer("positions", positions)
        self.requires_grad_(False)
        self.eval()

    def forward(self, eeg, picks):
        channels, readout = picks["channels"], picks["readout"]
        if eeg.ndim != 3 or eeg.shape[1] != len(channels) or len(set(channels)) != len(channels):
            raise ValueError("Input channel count/order is not the declared montage")
        if len(set(readout)) != len(readout) or any(i < 0 or i >= len(channels) for i in readout):
            raise ValueError("Invalid readout channel indices")
        if self.name == "REVE":
            if self.positions is None or any(c not in self.reve_names for c in channels):
                raise ValueError("Missing exact REVE channel coordinate")
            idx = [self.reve_names.index(c) for c in channels]
            pos = self.positions[idx].to(eeg).unsqueeze(0).expand(len(eeg), -1, -1)
            tokens = self.core(eeg.float(), pos)
        else:
            if eeg.shape[-1] % 200 or any(c.lower() not in self.labram_indices for c in channels):
                raise ValueError("LaBraM needs full 200-sample patches and known channels")
            input_chans = [0] + [self.labram_indices[c.lower()] for c in channels]
            patches = eeg.float().reshape(len(eeg), len(channels), -1, 200)
            tokens = self.core.forward_features(patches, input_chans=input_chans, return_patch_tokens=True)
            tokens = tokens.reshape(len(eeg), len(channels), -1, self.width)
        return tokens[:, readout].mean(2).flatten(1)


def load_backbone(name, weights_root):
    """Strictly load a verified PT checkpoint, never an MI-finetuned checkpoint."""
    from safetensors.torch import load_file
    cfg = config_x()["models"][name]
    weights_root = Path(weights_root)
    manifest = json.loads((weights_root / "manifest.json").read_text())
    by_name = {r["path"].replace("\\", "/").rsplit("/", 1)[-1]: r for r in manifest["files"]}
    filename = "model.safetensors" if name == "REVE" else "labram-base.pth"
    path = weights_root / name / filename
    if path.stat().st_size != cfg["bytes"] or sha256(path) != by_name[filename]["sha256"]:
        raise ValueError("Weight bytes do not match the download receipt")
    if cfg.get("published_sha256") and sha256(path) != cfg["published_sha256"]:
        raise ValueError("Weight bytes do not match the pinned upstream hash")
    if cfg.get("locally_verified_sha256") and sha256(path) != cfg["locally_verified_sha256"]:
        raise ValueError("Weight bytes do not match the independently verified Git blob")
    ReveConfig, Reve, labram = official_classes()
    if name == "REVE":
        config = json.loads(source("reve_hf_config.json").read_text())
        model = Reve(ReveConfig(**config))
        model.load_state_dict(load_file(path), strict=True)
        # Fixed FP32 PyTorch SDPA path, independent of whether flash-attn happens to be installed.
        module = import_source("_eeg_x_reve.modeling_reve", "reve_hf_modeling_reve.py")
        for attention, _ in model.transformer.layers:
            attention.attend = module.ClassicalAttention(attention.heads, use_sdpa=True)
            attention.use_flash = False
        position_path = weights_root / name / "positions.safetensors"
        if sha256(position_path) != by_name["positions.safetensors"]["sha256"]:
            raise ValueError("Position weights changed")
        positions = load_file(position_path)["embedding"]
        if positions.shape != (len(channel_tables()[0]), 3) or not torch.isfinite(positions).all():
            raise ValueError("Invalid position table")
    else:
        # The PT encoder owns norm, not the newly initialized downstream fc_norm.
        # Keep the official patch-token path and load every encoder tensor strictly.
        model = labram.labram_base_patch200_200(num_classes=0, use_mean_pooling=False,
                                              qkv_bias=False, init_values=0.1,
                                              use_abs_pos_emb=True, use_rel_pos_bias=False)
        allowed = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
                   np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Int64DType, argparse.Namespace]
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint)
        # Official pretraining checkpoints namespace the downstream encoder as student.
        state = {k.removeprefix("student."): v for k, v in state.items() if k.startswith("student.")}
        if not state:
            raise ValueError("Expected official LaBraM student encoder")
        ignored = [k for k in state if k.startswith("lm_head.") or k == "mask_token"]
        state = {k: v for k, v in state.items() if k not in ignored}
        model.load_state_dict(state, strict=True)
        positions = None
    model.eval().requires_grad_(False)
    result = CrossBackbone(name, model, positions)
    layers = model.transformer.layers if name == "REVE" else model.blocks
    if len(layers) != cfg["depth"] or result.width != cfg["width"]:
        raise ValueError("Wrong model variant")
    return result


def add_lora(backbone, rank=8):
    modules = []
    core = backbone.core
    layers = core.transformer.layers if backbone.name == "REVE" else core.blocks
    for layer in layers:
        attention = layer[0] if backbone.name == "REVE" else layer.attn
        qkv = attention.to_qkv if backbone.name == "REVE" else attention.qkv
        output = attention.to_out if backbone.name == "REVE" else attention.proj
        for owner, blocks in [(qkv, 3), (output, 1)]:
            if owner.weight.shape != (blocks * backbone.width, backbone.width):
                raise ValueError("Unexpected projection shape; independent Q/K/V factors required")
            update = LowRankWeight(backbone.width, rank, blocks).to(owner.weight)
            parametrize.register_parametrization(owner, "weight", update)
            modules.append(update)
    return modules  # Already owned/registered by the backbone projections.


class PopulationModel(nn.Module):
    def __init__(self, backbone, readout, rank=8):
        super().__init__()
        self.backbone = copy.deepcopy(backbone).requires_grad_(False)
        self.readout = copy.deepcopy(readout).requires_grad_(False)
        self.readout.head.requires_grad_(True)
        self.lora = add_lora(self.backbone, rank)
        self.eval()

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, eeg, picks):
        return self.readout(self.backbone(eeg, picks))


class PersonalModel(nn.Module):
    """Duck-types the original C2 fit/predict API; zero personal update equals G."""
    def __init__(self, population, rank=8):
        super().__init__()
        self.base = copy.deepcopy(population).requires_grad_(False)
        self.lora = add_lora(self.base.backbone, rank)
        self.eval()

    def trainable(self):
        return [p for module in self.lora for p in module.parameters()]

    @torch.no_grad()
    def reset(self):
        for module in self.lora:
            for block in module.a:
                nn.init.kaiming_uniform_(block, a=math.sqrt(5))
            module.b.zero_()

    def personal(self):
        return [p.detach().clone() for p in self.trainable()]

    @torch.no_grad()
    def load_personal(self, values):
        for parameter, value in zip(self.trainable(), values, strict=True):
            parameter.copy_(value)

    def forward(self, eeg, picks):
        return self.base(eeg, picks)
