"""Formal B0 feature cache, extracted once per backbone and shared by all folds/seeds."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.cross_model import load_backbone
from subject_context.cross_model_data import load_subject, starter_people
from subject_context.stage_a_common import ROOT, require_compute, save_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["REVE", "LaBraM"])
    p.add_argument("--stage-a-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    args = p.parse_args()
    require_compute()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "3090" not in torch.cuda.get_device_name():
        raise RuntimeError("Exactly one allocated RTX3090 required")
    torch.set_num_threads(4)
    model = load_backbone(args.model, args.weights).cuda()
    started = time.monotonic()
    identity = {"model": args.model, "config_sha256": sha256(ROOT / "configs/cross_model_x.yaml"),
                "interface_sha256": sha256(ROOT / "src/subject_context/cross_model.py")}
    for dataset, person in starter_people():
        out = args.out / args.model / "features" / dataset / f"sub-{person:03d}"
        if (out / "done.json").exists():
            import json
            receipt = json.loads((out / "done.json").read_text())
            if any(receipt.get(k) != v for k, v in identity.items()) or sha256(out / "features.npy") != receipt["features_sha256"]:
                raise ValueError("Existing feature cache identity mismatch")
            continue
        start = time.monotonic()
        s = load_subject(args.stage_a_root, args.out, args.model, dataset, person, "cache", "cuda")
        with torch.no_grad():
            features = torch.cat([model(chunk.float(), s["picks"]) for chunk in s["eeg"].split(32)]).cpu().numpy()
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite B0 cache")
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "features.npy", features, allow_pickle=False)
        save_json(out / "done.json", {**identity, "features_sha256": sha256(out / "features.npy"),
                  "shape": list(features.shape), "seconds": time.monotonic() - start,
                  "trial_table_sha256": sha256(args.stage_a_root / "processed" / dataset / f"sub-{person:03d}" / "trials.csv"),
                  "supervised_selection": False})
        print(f"{args.model} {dataset} {person}: cached", flush=True)
        del s, features
    save_json(args.out / args.model / "features/complete.json", {**identity, "subjects": len(starter_people()),
              "seconds": time.monotonic() - started, "gpu": torch.cuda.get_device_name(), "status": "complete"})


if __name__ == "__main__":
    main()
