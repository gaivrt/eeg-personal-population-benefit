"""One complete, formal model x fold x seed run of the X1 core."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.cross_model import config_x, load_backbone
from subject_context.cross_model_data import load_subject
from subject_context.cross_model_run import diagnose, personal_selection, resource_clock, train_b0, train_population
from subject_context.data import assert_split
from subject_context.model import state_hash
from subject_context.stage_a_common import ROOT, provenance, require_compute, save_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["REVE", "LaBraM"])
    p.add_argument("--job", required=True, type=int, choices=range(25))
    p.add_argument("--stage-a-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    args = p.parse_args()
    require_compute()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "3090" not in torch.cuda.get_device_name():
        raise RuntimeError("Exactly one allocated RTX3090 required")
    torch.set_num_threads(4)
    cfg = config_x()
    fold_index, seed = args.job // 5, cfg["seeds"][args.job % 5]
    fold = json.loads((ROOT / cfg["split_file"]).read_text())["folds"][fold_index]
    assert_split(*[[tuple(row) for row in fold[role]] for role in ("train", "validation", "test")])
    out = args.out / args.model / "runs" / f"fold-{fold_index}_seed-{seed}"
    out.mkdir(parents=True, exist_ok=False)  # Do not silently overwrite or rerun a partial experiment.
    started = time.monotonic()
    receipt = {**provenance(), "model": args.model, "fold": fold_index, "seed": seed,
               "scope": "X1_core_formal", "status": "running", "gpu": torch.cuda.get_device_name(),
               "config_sha256": sha256(ROOT / "configs/cross_model_x.yaml"),
               "split_sha256": sha256(ROOT / cfg["split_file"]), "halves_sha256": sha256(ROOT / cfg["halves_file"]),
               "code_sha256": {f.relative_to(ROOT).as_posix(): sha256(f) for f in sorted((ROOT / "src/subject_context").glob("*.py"))}}
    receipt["code_sha256"]["scripts/cross_model_run.py"] = sha256(Path(__file__))
    save_json(out / "run.json", receipt)

    def people(role, features=False):
        result = []
        for dataset, subject in fold[role]:
            s = load_subject(args.stage_a_root, args.out, args.model, dataset, subject, role, "cuda")
            if features:
                folder = args.out / args.model / "features" / dataset / f"sub-{subject:03d}"
                saved = json.loads((folder / "done.json").read_text())
                identity = {"model": args.model, "config_sha256": receipt["config_sha256"],
                            "interface_sha256": sha256(ROOT / "src/subject_context/cross_model.py"),
                            "trial_table_sha256": sha256(args.stage_a_root / "processed" / dataset / f"sub-{subject:03d}" / "trials.csv")}
                if any(saved.get(k) != v for k, v in identity.items()) or sha256(folder / "features.npy") != saved["features_sha256"]:
                    raise ValueError("B0 cache identity mismatch")
                s["features"] = torch.as_tensor(np.load(folder / "features.npy", allow_pickle=False), device="cuda")
                if len(s["features"]) != len(s["labels"]):
                    raise ValueError("B0 cache trial count mismatch")
            result.append(s)
        return result

    try:
        backbone = load_backbone(args.model, args.weights).cuda()
        frozen = state_hash(backbone)
        groups = {role: people(role, features=True) for role in ("train", "validation")}
        readout = train_b0(groups, cfg["population"]["B0"], seed, out / "B0")
        for subjects in groups.values():
            for s in subjects:
                del s["features"]
        population, convergence = train_population(backbone, readout, groups, cfg, seed, out / "G", microbatch=8)
        del groups["train"]
        selected_hash = state_hash(population)
        selections = personal_selection(population, groups["validation"], cfg["personal_and_few_shot"], seed, out / "personal", microbatch=8)
        del groups
        # No test EEG/labels has been loaded by this process before every choice is written and hashed.
        selection_files = [out / "B0/selection.json", out / "G/initial_selection.json", out / "G/convergence.json",
                           out / "G/continuation/selected.pt", out / "B0/selected.pt", out / "personal/selection.json"]
        save_json(out / "frozen_before_test.json", {"files": {p.relative_to(out).as_posix(): sha256(p) for p in selection_files},
                  "population_state_hash": selected_hash, "config_sha256": receipt["config_sha256"], "test_seen": False})
        test = people("test")
        diagnose(population, backbone, readout, test, selections, cfg["personal_and_few_shot"], seed, fold_index, out / "diagnostic", microbatch=8)
        if state_hash(backbone) != frozen or state_hash(population) != selected_hash:
            raise ValueError("Frozen source model changed during diagnosis")
        receipt.update(status="complete", backbone_unchanged=True, population_unchanged=True,
                       convergence=convergence, test_subjects=len(test), **resource_clock(started))
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error), **resource_clock(started))
        raise
    finally:
        save_json(out / "run.json", receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
