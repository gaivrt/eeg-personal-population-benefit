"""CPU-only engineering checks with pinned pretrained weights and synthetic inputs."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.cross_model import config_x, load_backbone, PersonalModel, PopulationModel
from subject_context.cross_model_training import update
from subject_context.model import state_hash
from subject_context.stage_a_common import ROOT, save_json, sha256
from subject_context.stage_b import Readout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/cross_model")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/cross_model_engineering_2026-09-27/cpu_check.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    checks = []
    for name in ("REVE", "LaBraM"):
        start = time.monotonic()
        torch.manual_seed(11)
        backbone = load_backbone(name, args.weights)
        frozen = state_hash(backbone)
        channels = ["C3", "Cz", "C4"]
        picks = {"channels": channels, "readout": [0, 1, 2]}
        # Both real input lengths; this is not data from any evaluation subject.
        with torch.no_grad():
            shapes = {str(t): list(backbone(torch.randn(2, 3, t), picks).shape) for t in (600, 800)}
            native = json.loads((ROOT / "configs/stage_a_channels.json").read_text())
            native_shapes = {}
            for dataset, names in native["native"].items():
                n = 600 if dataset == "Cho2017" else 800
                p = {"channels": names, "readout": [names.index(c) for c in native["readout_common"]]}
                result = backbone(torch.randn(1, len(names), n), p)
                assert result.shape == (1, 27 * backbone.width) and torch.isfinite(result).all()
                native_shapes[dataset] = {"input": [1, len(names), n], "output": list(result.shape)}
        width = 3 * backbone.width
        head = nn.Linear(width, 2)
        readout = Readout({"head": head.state_dict(), "feature_mean": np.zeros(width), "feature_std": np.ones(width)}, "linear")
        subject = {"key": ("synthetic", 0), "eeg": torch.randn(2, 3, 400), "labels": torch.tensor([0, 1]), "picks": picks}
        with torch.no_grad():
            baseline = readout(backbone(subject["eeg"], picks))
        population = PopulationModel(backbone, readout)
        with torch.no_grad():
            zero_error = float((population(subject["eeg"], picks) - baseline).abs().max())
        assert zero_error == 0
        count = sum(p.numel() for m in population.lora for p in m.parameters())
        assert count == config_x()["models"][name]["lora8_parameters"]
        optimizer = torch.optim.AdamW(population.trainable(), lr=.001)
        update(population, subject, torch.arange(2), optimizer)
        gradients = [m.b.grad.flatten(1).norm(dim=1).tolist() for m in population.lora]
        assert all(all(v > 0 for v in row) for row in gradients)
        population_hash = state_hash(population)
        personal = PersonalModel(population)
        with torch.no_grad():
            personal_error = float((personal(subject["eeg"], picks) - population(subject["eeg"], picks)).abs().max())
        assert personal_error == 0 and state_hash(population) == population_hash
        assert state_hash(backbone) == frozen
        checks.append({"model": name, "loaded_all_encoder_keys": True,
                       "pretrained_parameters": sum(p.numel() for p in backbone.parameters()),
                       "lora8_parameters": count, "zero_population_max_logit_error": zero_error,
                       "zero_personal_max_logit_error": personal_error, "QKV_and_output_B_gradient_norms": gradients,
                       "input_length_output_shapes": shapes, "original_backbone_unchanged": True,
                       "starter_native_montage_checks": native_shapes,
                       "seconds": time.monotonic() - start})
        del backbone, readout, population, personal, optimizer
        gc.collect()
    receipt = {"scope": "CPU engineering check; synthetic EEG and labels only", "real_EEG_subjects": 0,
               "GPU_jobs": 0, "scientific_results": False, "checks": checks,
               "config_sha256": sha256(ROOT / "configs/cross_model_x.yaml"),
               "packages": {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "transformers", "timm", "safetensors"]}}
    save_json(args.out, receipt)
    print({"status": "pass", "models": [r["model"] for r in checks], "output": str(args.out)})


if __name__ == "__main__":
    main()
