"""One existing fold/seed: continue both B3 families and save G plus loss curves."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.data import assert_split
from subject_context.model import load_backbone, state_hash
from subject_context.stage_a_common import ROOT, config, provenance, require_compute, save_json, scratch_root, sha256
from subject_context.stage_b import Readout, ba, temporal_halves
from subject_context.stage_c import ContextModel, backbone_features, config_c
from subject_context.stage_c_context import TangentFeatures, band_covariances
from subject_context.stage_c_train import load_snapshot, score
from subject_context.stage_w import config_w, continuation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=int, required=True, choices=range(25))
    args = parser.parse_args()
    require_compute()
    assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name()
    torch.set_num_threads(4)
    root, c_cfg, w_cfg = scratch_root(), config_c(), config_w()
    fold_index, seed = args.job // 5, w_cfg["seeds"][args.job % 5]
    name = f"fold-{fold_index}_seed-{seed}"
    out = root / "stage_w/population/runs" / name
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.glob("model-*.pt")) and not (out / "run.json").exists(), "Use an explicit new attempt directory"
    fold = json.loads((ROOT / w_cfg["split_file"]).read_text())["folds"][fold_index]
    split = {r: [tuple(k) for k in fold[r]] for r in ["train", "validation", "test"]}
    assert_split(*split.values())
    halves = {(r["dataset"], r["subject"]): r for r in json.loads((ROOT / w_cfg["halves_file"]).read_text())["subjects"]}
    source = root / "stage_c/runs" / name
    selection = json.loads((source / "selection.json").read_text())
    kind = selection["base_head"]
    base_dir = root / "baselines_v2" / f"B0_{kind}" / name
    readout = Readout(torch.load(base_dir / "head.pt", map_location="cpu", weights_only=False), kind).cuda()
    backbone = load_backbone(ROOT / config()["model"]["weights"], k=4).cuda().eval()
    frozen = state_hash(backbone), state_hash(readout)
    common = json.loads((ROOT / "configs/stage_a_channels.json").read_text())["readout_common"]
    receipt = provenance()
    receipt.update(fold=fold_index, seed=seed, gpu=torch.cuda.get_device_name(),
                   w_config_sha256=sha256(ROOT / "configs/stage_w.yaml"),
                   split_sha256=sha256(ROOT / w_cfg["split_file"]), halves_sha256=sha256(ROOT / w_cfg["halves_file"]),
                   source_selection_sha256=sha256(source / "selection.json"), families={})
    start = time.monotonic()

    @torch.no_grad()
    def load_subject(key, role):
        dataset, subject = key
        src = root / "processed" / dataset / f"sub-{subject:03d}"
        record = halves[key]
        assert sha256(src / "trials.csv") == record["trial_table_sha256"]
        trials = pd.read_csv(src / "trials.csv")
        fit_rows, query, counts = temporal_halves(trials)
        assert fit_rows.tolist() == record["fit_indices"] and query.tolist() == record["query_indices"]
        meta = json.loads((src / "metadata.json").read_text())
        picks = [meta["channels"].index(ch) for ch in common]
        y = np.load(src / "labels.npy")
        np.testing.assert_array_equal(y, trials.label)
        s = {"key": key, "eeg": torch.as_tensor(np.load(src / "tasks.npy"), device="cuda"), "picks": picks,
             "labels": torch.tensor(y, device="cuda"), "query": torch.tensor(query, device="cuda"),
             "query_labels": y[query], "trial_ids": trials.trial_id.tolist()}
        if role != "test":
            rest = np.load(src / "rest_open.npy")[np.load(src / "rest_open_valid.npy")]
            assert meta["rest"]["open"]["protocol_order"] < trials.protocol_order.min()
            x = torch.as_tensor(rest, device="cuda")
            pooled = torch.cat([backbone_features(backbone.model, x[t:t+32].float(), picks) for t in range(0, len(x), 32)])
            s["emb"] = readout.normalize(pooled)
            s["bands"] = band_covariances(rest[:, picks].astype(np.float64), c_cfg["context"]["bands_hz"],
                                         c_cfg["context"]["filter_order"])
        return s

    # No test signals or labels are loaded until BOTH G checkpoints are fixed.
    groups = {r: [load_subject(k, r) for k in split[r]] for r in ["train", "validation"]}
    tangent = TangentFeatures().fit([s["bands"] for s in groups["train"]])
    for role in groups.values():
        for s in role:
            s["tan"] = torch.as_tensor(tangent(s.pop("bands")), device="cuda")
    models, curve, losses, steps = {}, [], [], []
    for method in w_cfg["population"]["methods"]:
        torch.manual_seed(seed)
        model = ContextModel(backbone, readout, method, 5400, 756, c_cfg).cuda()
        src = source / f"model-{method}-main.pt"
        load_snapshot(model, torch.load(src, map_location="cuda", weights_only=True))
        destination = out / f"model-{method}-main.pt"
        curves, people, episodes, status = continuation(model, groups["train"], groups["validation"],
            selection[f"{method}_main"], c_cfg, w_cfg, seed,
            lambda state, step: torch.save(state, destination))
        for target, rows in [(curve, curves), (losses, people), (steps, episodes)]:
            target.extend({"method": method, **row} for row in rows)
        receipt["families"][method] = {**status, "source_sha256": sha256(src), "G_sha256": sha256(destination),
                                        "hyperparameters": selection[f"{method}_main"]}
        for filename, rows in [("curves.csv", curve), ("subject_losses.csv", losses), ("training_steps.csv", steps)]:
            pd.DataFrame(rows).to_csv(out / filename, index=False)
        models[method] = model
        print(json.dumps({"method": method, **status}), flush=True)
    save_json(out / "selection.json", receipt["families"])
    tests = [load_subject(k, "test") for k in split["test"]]
    rows, predictions = [], []
    for method, model in models.items():
        for s in tests:
            result = score(model, s, default=True)  # C2's all-trials batch path
            labels = s["labels"].cpu().numpy()
            query = set(s["query"].cpu().tolist())
            rows.append({"dataset": s["key"][0], "subject": s["key"][1], "fold": fold_index, "seed": seed,
                         "method": method, "ba": result["ba"], "ba_all": result["ba_all"]})
            predictions.extend({"dataset": s["key"][0], "subject": s["key"][1], "method": method,
                "trial_id": tid, "label": int(labels[i]), "prediction": int(result["predictions"][i]),
                "is_query": i in query} for i, tid in enumerate(s["trial_ids"]))
    pd.DataFrame(rows).to_csv(out / "subjects.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
    assert (state_hash(backbone), state_hash(readout)) == frozen
    receipt.update(status="complete", seconds=time.monotonic()-start, frozen_backbone_unchanged=True,
                   frozen_b0_head_unchanged=True, peak_gpu_bytes=torch.cuda.max_memory_allocated())
    save_json(out / "run.json", receipt)


if __name__ == "__main__":
    main()
