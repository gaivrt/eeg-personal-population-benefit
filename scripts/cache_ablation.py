"""Remove only prefix caching, compare identical computations and CPU time."""
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subject_context.model import load_backbone, Adapter

torch.set_num_threads(2)
torch.manual_seed(20260925)
torch.use_deterministic_algorithms(True)
backbone = load_backbone(ROOT / "weights/cbramod.pth")
model = Adapter(212)
torch.nn.init.normal_(model.generator[-1].weight, std=.01)
x, context = torch.randn(8, 22, 800), torch.randn(30, 212)
prefix = backbone.prefix(x)


def step(cached):
    model.zero_grad(set_to_none=True)
    logits = model(backbone, prefix if cached else backbone.prefix(x), context)
    logits.square().mean().backward()
    return logits.detach(), [p.grad.clone() if p.grad is not None else None for p in model.parameters()]


cached, cg = step(True)
uncached, ug = step(False)
assert torch.equal(cached, uncached)
assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(cg, ug))
times = {}
for name, use_cache in [("cached", True), ("uncached", False)]:
    start = time.perf_counter()
    for _ in range(20):
        step(use_cache)
    times[name] = (time.perf_counter() - start) / 20
result = {"device": "cpu", "shape": list(x.shape), "repeats": 20, "logits_bit_identical": True,
          "gradients_bit_identical": True, "seconds_per_step": times,
          "uncached_over_cached": times["uncached"] / times["cached"],
          "decision": "retain prefix cache: same outputs/gradients, avoids redundant frozen-prefix computation"}
(ROOT / "reports/stage0a/cache_ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
