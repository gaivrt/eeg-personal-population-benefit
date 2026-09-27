import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subject_context.model import load_backbone, state_hash
from subject_context.data import preprocess_synthetic

cfg = yaml.safe_load((ROOT / "configs/stage0a.yaml").read_text())
torch.set_num_threads(cfg["threads"])
torch.manual_seed(cfg["data_seed"])
torch.use_deterministic_algorithms(True)
model = load_backbone(ROOT / cfg["model"]["weights"], cfg["model"]["film_layers"])
x = torch.from_numpy(preprocess_synthetic(np.random.default_rng(cfg["data_seed"]).normal(0, 20, (1, 22, 800))))
start = time.perf_counter()
with torch.no_grad():
    features = model(x)
elapsed = time.perf_counter() - start
assert features.shape == (1, 22, 4, 200) and torch.isfinite(features).all()
assert all(not p.requires_grad for p in model.parameters())
receipt = {"device": str(features.device), "torch": torch.__version__, "strict_load": True,
           "weights_sha256": hashlib.file_digest(open(ROOT / cfg["model"]["weights"], "rb"), "sha256").hexdigest(),
           "source_commit": cfg["model"]["source_commit"], "input_shape_patched": [1, 22, 4, 200],
           "output_shape": list(features.shape), "finite": True, "seconds": elapsed,
           "frozen_parameters": sum(p.numel() for p in model.parameters()), "state_hash": state_hash(model)}
(ROOT / "reports/stage0a/backbone_smoke.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2))
