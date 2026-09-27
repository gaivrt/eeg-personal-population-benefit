"""One bounded RTX3090 timing job; no test-subject access or scientific model selection."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.cross_model import PersonalModel, PopulationModel, config_x, load_backbone
from subject_context.cross_model_data import load_subject, timing_people
from subject_context.cross_model_training import append_validation, evaluate, update
from subject_context.stage_a_common import ROOT, require_compute, save_json, sha256
from subject_context.stage_b import Readout
from subject_context.stage_c2 import fit, predict


def timed(operation):
    torch.cuda.synchronize()
    start = time.monotonic()
    result = operation()
    torch.cuda.synchronize()
    return result, time.monotonic() - start


def profile(name, args):
    started = time.monotonic()
    torch.manual_seed(11); torch.cuda.manual_seed_all(11)
    out = args.out / "timing" / name
    out.mkdir(parents=True, exist_ok=False)  # A rerun must have its own reviewable directory.
    backbone, load_seconds = timed(lambda: load_backbone(name, args.weights).cuda())
    people = timing_people()
    groups = {role: [load_subject(args.stage_a_root, args.out, name, d, s, role, "cuda") for d, s in ids]
              for role, ids in people.items()}
    receipt = {"scope": "engineering timing only; temporary three-subject head, not B0/G results", "model": name,
               "gpu": torch.cuda.get_device_name(), "microbatch": args.microbatch, "people": people,
               "config_sha256": sha256(ROOT / "configs/cross_model_x.yaml"), "load_seconds": load_seconds,
               "test_subjects_used": False, "feature_seconds": [], "group_updates": [], "personal_fits": []}
    receipt["code_sha256"] = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
                              for p in sorted((ROOT / "src/subject_context").glob("*.py"))}
    receipt["code_sha256"]["scripts/cross_model_profile.py"] = sha256(Path(__file__))
    receipt["status"] = "running"
    save_json(out / "profile.json", receipt)
    # Feature extraction covers the original channel/window shapes; normalization uses train only.
    with torch.no_grad():
        for role, subjects in groups.items():
            for s in subjects:
                def features():
                    return torch.cat([backbone(x.float(), s["picks"]) for x in s["eeg"].split(32)])
                s["features"], seconds = timed(features)
                receipt["feature_seconds"].append({"dataset": s["key"][0], "role": role,
                                                   "trials": len(s["eeg"]), "seconds": seconds})
                save_json(out / "profile.json", receipt)
    train = torch.cat([s["features"] for s in groups["train"]])
    train_labels = torch.cat([s["labels"] for s in groups["train"]])
    mean, std = train.double().mean(0), train.double().std(0, correction=0).clamp_min(1e-6)
    head = nn.Linear(train.shape[1], 2).cuda()
    readout = Readout({"head": head.state_dict(), "feature_mean": mean.cpu(), "feature_std": std.cpu()}, "linear").cuda()
    readout.head.requires_grad_(True)
    head_optimizer = torch.optim.AdamW(readout.head.parameters(), lr=.001, weight_decay=.01)
    def head_steps():
        for step in range(20):
            ids = torch.randperm(len(train), device="cuda")[:128]
            head_optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(readout(train[ids]), train_labels[ids])
            loss.backward(); head_optimizer.step()
    _, receipt["head_20_updates_seconds"] = timed(head_steps)
    population = PopulationModel(backbone, readout).cuda()
    del backbone, head, head_optimizer, train, train_labels
    optimizer = torch.optim.AdamW(population.trainable(), lr=.001, weight_decay=.01)
    metrics = evaluate(population, groups["validation"])
    append_validation(out / "validation.jsonl", {"phase": "timing", "step": 0, "partial_epoch": True}, metrics)
    for s in groups["train"]:
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for i in range(12):
            take = torch.randperm(len(s["eeg"]), device="cuda")[:32]
            _, seconds = timed(lambda: update(population, s, take, optimizer, args.microbatch, 1.))
            if i >= 2:
                samples.append(seconds)
        receipt["group_updates"].append({"dataset": s["key"][0], "effective_batch": len(take),
            "warmup_updates": 2, "measured_updates": 10, "seconds": samples,
            "median_seconds": float(np.median(samples)), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        save_json(out / "profile.json", receipt)
    metrics, seconds = timed(lambda: evaluate(population, groups["validation"]))
    receipt["validation_seconds"] = seconds
    append_validation(out / "validation.jsonl", {"phase": "timing", "step": 36, "partial_epoch": True}, metrics)
    personal = PersonalModel(population).cuda()
    for s in groups["validation"]:
        shots = [5, 10, 20] + ([40] if s["key"][0] in config_x()["personal_and_few_shot"]["additional_40"] else [])
        for n in [None, *shots]:
            view = dict(s, fit=s["fit"] if n is None else s["fit"][:n])
            _, _, resource = fit(personal, view, .001, 0., [20, 60], 11 + s["key"][1], microbatch=args.microbatch)
            receipt["personal_fits"].append({"dataset": s["key"][0], "n": n, "steps": 60, **resource})
            save_json(out / "profile.json", receipt)
        _, seconds = timed(lambda: predict(personal, s, s["query"]))
        receipt.setdefault("swap_inference_seconds", []).append({"dataset": s["key"][0], "query_trials": len(s["query"]), "seconds": seconds})
    receipt["status"] = "complete"
    receipt["total_seconds"] = time.monotonic() - started
    save_json(out / "profile.json", receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=8)
    args = parser.parse_args()
    require_compute()
    if not torch.cuda.is_available() or "3090" not in torch.cuda.get_device_name() or torch.cuda.device_count() != 1:
        raise RuntimeError("Timing requires exactly one allocated RTX3090")
    torch.set_num_threads(4)
    # Sequential models: total simultaneous GPUs remains one.
    for name in ("LaBraM", "REVE"):
        try:
            profile(name, args)
        except Exception as error:
            save_json(args.out / "timing" / name / "failure.json", {"type": type(error).__name__, "error": str(error)})
            raise  # No automatic resubmission, enlarged budget, or changed model.
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
