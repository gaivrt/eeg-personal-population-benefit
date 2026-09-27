"""CPU synthetic training/evaluation. This entry point cannot train real data."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from .data import synthetic_sessions, make_splits, assert_split, assert_episode, other_subject
from .features import band_covariances, TangentSpace
from .model import load_backbone, Adapter, state_hash, ROOT
from .statistics import paired_statistics, holm


def seed_everything(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def cache_features(cfg, backbone, sessions, folder):
    provenance = {"synthetic": cfg["synthetic"], "data_seed": cfg["data_seed"],
                  "weights_sha256": hashlib.file_digest(open(ROOT / cfg["model"]["weights"], "rb"), "sha256").hexdigest(),
                  "commit": cfg["model"]["source_commit"], "k": cfg["model"]["film_layers"],
                  "vendor_source": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (ROOT / "vendor/CBraMod/models").glob("*.py")},
                  "torch": torch.__version__, "numpy": np.__version__, "threads": cfg["threads"],
                  "code": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    folder = folder / key[:16]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    cached = {}
    with torch.no_grad():
        for key, session in sessions.items():
            filename = folder / ("_".join(map(str, key)) + ".pt")
            if filename.exists():
                cached[key] = torch.load(filename, map_location="cpu", weights_only=True)
                continue
            prefix = backbone.prefix(torch.from_numpy(session["query"]))
            context_embedding = []
            for batch in np.array_split(session["context"], max(1, len(session["context"]) // 16)):
                context_embedding.append(backbone(torch.from_numpy(batch)).mean(dim=(1, 2)))
            cached[key] = {"prefix": prefix, "embedding": backbone.suffix(prefix).mean(dim=(1, 2)),
                           "context_embedding": torch.cat(context_embedding),
                           "covariance": torch.from_numpy(band_covariances(session["context"])),
                           "labels": torch.tensor(session["labels"], dtype=torch.long)}
            torch.save(cached[key], filename)
            print("cached", key, flush=True)
    return cached, folder


def contexts_for_fold(cached, train):
    trainkeys = [key for key in cached if key[:2] in train]
    tangent = TangentSpace().fit(np.concatenate([cached[k]["covariance"].numpy() for k in trainkeys]))
    features = {k: torch.cat([v["context_embedding"], torch.from_numpy(tangent.transform(v["covariance"].numpy()))], dim=1)
                for k, v in cached.items()}
    training = torch.cat([features[k] for k in trainkeys])
    mean, std = training.mean(0), training.std(0).clamp_min(1e-5)
    return {k: (v - mean) / std for k, v in features.items()}, mean, std, tangent.whitener


def balanced_accuracy(labels, predictions):
    return float(np.mean([np.mean(predictions[labels == c] == c) for c in [0, 1]]))


def run(cfg):
    assert cfg["stage"] == "0a" and cfg["device"] == "cpu"
    assert len(cfg["seeds"]) >= 5 and cfg["folds"] == 5
    seed_everything(cfg["data_seed"], cfg["threads"])
    started = time.perf_counter()
    output = ROOT / cfg["output"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    backbone = load_backbone(ROOT / cfg["model"]["weights"], cfg["model"]["film_layers"])
    original_hash = state_hash(backbone)
    sessions = synthetic_sessions(cfg)
    people = sorted({key[:2] for key in sessions})
    manifest = [dict(asdict(w), role=role) for session in sessions.values()
                for role in ["context", "query"] for w in session[role + "_windows"]]
    pd.DataFrame(manifest).to_csv(output / "windows.csv", index=False)
    cache_start = time.perf_counter()
    cached, cache_path = cache_features(cfg, backbone, sessions, ROOT / "artifacts/stage0a/cache")
    cache_seconds = time.perf_counter() - cache_start
    all_splits, predictions, validation, training_log = [], [], [], []
    for seed in cfg["seeds"]:
        for fold, (train, val, test) in enumerate(make_splits(people, cfg["folds"], cfg["validation_fraction"], seed)):
            seed_everything(seed * 100 + fold, cfg["threads"])
            rng = np.random.default_rng(seed * 100 + fold)
            assert_split(train, val, test)
            all_splits.append({"seed": seed, "fold": fold, "train": train, "validation": val, "test": test})
            contexts, mean, std, whitener = contexts_for_fold(cached, train)
            model = Adapter(len(mean), k=cfg["model"]["film_layers"], z_dim=cfg["model"]["z_dim"], hidden=cfg["model"]["hidden"])
            baseline = torch.nn.Linear(200, 2)
            baseline.load_state_dict(model.head.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
            base_optimizer = torch.optim.Adam(baseline.parameters(), lr=cfg["training"]["learning_rate"])
            trainkeys = sorted(k for k in sessions if k[:2] in train)
            for step in range(cfg["training"]["steps"]):
                key = trainkeys[int(rng.integers(len(trainkeys)))]
                record = cached[key]
                seconds = rng.uniform(cfg["training"]["context_min_seconds"], cfg["training"]["context_max_seconds"])
                count = int(seconds // cfg["synthetic"]["segment_seconds"])
                assert_episode(sessions[key]["context_windows"][:count], sessions[key]["query_windows"])
                donor = other_subject(key[:2], train, rng) + (key[2],)
                assert donor[:2] in train and donor[0] == key[0] and donor[1] != key[1]
                dropped = rng.random() < cfg["model"]["context_dropout"]
                context = None if dropped else contexts[key][:count]
                optimizer.zero_grad(set_to_none=True)
                logits = model(backbone, record["prefix"], context)
                ce = F.cross_entropy(logits, record["labels"])
                if dropped:
                    margin_loss = ce * 0  # no subject-matching contrast for default-z episodes
                else:
                    wrong = model(backbone, record["prefix"], contexts[donor][:count])
                    margin_loss = torch.relu(ce - F.cross_entropy(wrong, record["labels"]) + cfg["training"]["margin"])
                loss = ce + cfg["training"]["lambda"] * margin_loss
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is None for p in backbone.parameters())
                optimizer.step()
                base_optimizer.zero_grad(set_to_none=True)
                base_loss = F.cross_entropy(baseline(record["embedding"]), record["labels"])
                base_loss.backward()
                base_optimizer.step()
                training_log.append({"seed": seed, "fold": fold, "step": step, "dataset": key[0], "subject": key[1],
                                     "session": key[2], "context_seconds": count * cfg["synthetic"]["segment_seconds"],
                                     "donor_subject": donor[1], "dropout": dropped,
                                     "loss_M": loss.item(), "loss_B0": base_loss.item()})
            model.eval()
            assert state_hash(backbone) == original_hash, "backbone changed"
            torch.save({"adapter": model.state_dict(), "baseline": baseline.state_dict(), "mean": mean,
                        "std": std, "whitener": torch.tensor(whitener), "backbone_hash": original_hash},
                       output / f"seed{seed}_fold{fold}.pt")
            count = cfg["evaluation"]["context_seconds"] // cfg["synthetic"]["segment_seconds"]
            with torch.no_grad():
                for key in sorted(k for k in sessions if k[:2] in val):
                    assert_episode(sessions[key]["context_windows"][:count], sessions[key]["query_windows"])
                    logits = model(backbone, cached[key]["prefix"], contexts[key][:count])
                    validation.append({"seed": seed, "fold": fold, "dataset": key[0], "subject": key[1], "session": key[2],
                                       "ce": F.cross_entropy(logits, cached[key]["labels"]).item()})
                for setting in cfg["evaluation"]["settings"]:
                    keys = sorted(k for k in sessions if k[:2] in test and (setting == "same_session" or k[2] == 2))
                    for key in keys:
                        ctxkey = key if setting == "same_session" else key[:2] + (1,)
                        assert_episode(sessions[ctxkey]["context_windows"][:count], sessions[key]["query_windows"],
                                       cross_session=setting == "cross_session")
                        donor = other_subject(key[:2], test, rng) + (ctxkey[2],)
                        assert donor[:2] in test
                        scores = {"B0": baseline(cached[key]["embedding"]),
                                  "B3": model(backbone, cached[key]["prefix"], None),
                                  "B4": model(backbone, cached[key]["prefix"], contexts[donor][:count]),
                                  "M": model(backbone, cached[key]["prefix"], contexts[ctxkey][:count])}
                        for condition in cfg["evaluation"]["conditions"]:
                            probability = scores[condition].softmax(-1)[:, 1].numpy()
                            for i, (label, p) in enumerate(zip(sessions[key]["labels"], probability)):
                                predictions.append({"seed": seed, "fold": fold, "dataset": key[0], "subject": key[1],
                                                    "session": key[2], "setting": setting, "condition": condition,
                                                    "trial_id": sessions[key]["query_windows"][i].trial_id,
                                                    "context_subject": donor[1] if condition == "B4" else (key[1] if condition == "M" else ""),
                                                    "context_session": ctxkey[2] if condition in ["M", "B4"] else "",
                                                    "y_true": int(label), "p_right": float(p), "y_pred": int(p >= .5)})
            print(f"finished seed={seed} fold={fold}", flush=True)
    frame = pd.DataFrame(predictions)
    frame.to_csv(output / "predictions.csv", index=False)
    (output / "splits.json").write_text(json.dumps(all_splits, indent=2), encoding="utf-8")
    pd.DataFrame(validation).to_csv(output / "validation.csv", index=False)
    pd.DataFrame(training_log).to_csv(output / "training.csv", index=False)
    rows = []
    for key, group in frame.groupby(["seed", "dataset", "subject", "setting", "condition"]):
        rows.append(dict(zip(["seed", "dataset", "subject", "setting", "condition"], key),
                         balanced_accuracy=balanced_accuracy(group.y_true.to_numpy(), group.y_pred.to_numpy())))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics_per_subject_seed.csv", index=False)
    subject = metrics.groupby(["dataset", "subject", "setting", "condition"]).balanced_accuracy.mean().reset_index()
    subject.to_csv(output / "metrics_per_subject.csv", index=False)
    summary = subject.groupby(["setting", "condition"]).balanced_accuracy.agg(["mean", "std", "count"])
    summary.to_csv(output / "summary.csv")
    stats = []
    for setting, group in subject.groupby("setting"):
        wide = group.pivot(index=["dataset", "subject"], columns="condition", values="balanced_accuracy")
        assert not wide.isna().any().any()
        for baseline_name in ["B0", "B3", "B4"]:
            stats.append({"setting": setting, "comparison": f"M-{baseline_name}",
                          **paired_statistics(wide.M.to_numpy(), wide[baseline_name].to_numpy())})
    adjusted = holm([s["p_raw"] for s in stats])
    for row, p in zip(stats, adjusted):
        row["p_holm"] = p
    pd.DataFrame(stats).to_csv(output / "statistics.csv", index=False)
    receipt = {"status": "completed", "device": "cpu", "synthetic_only": True, "python": platform.python_version(),
               "torch": torch.__version__, "seconds": time.perf_counter() - started, "cache_seconds": cache_seconds,
               "cache_path": str(cache_path.relative_to(ROOT)), "frozen_before": original_hash, "frozen_after": state_hash(backbone),
               "n_seeds": len(cfg["seeds"]), "folds": cfg["folds"], "n_subjects": len(people),
               "prediction_rows": len(frame), "conditions": cfg["evaluation"]["conditions"],
               "statistical_unit": "dataset-subject, seeds averaged before testing", "holm_family": "all six smoke comparisons",
               "budget": "20 matched subject-session episodes per condition per fold; not equal FLOPs"}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(summary.to_string())
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage0a.yaml")
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
