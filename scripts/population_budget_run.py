"""One of 75 authorized group-budget curves, followed by fixed personal diagnostics."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.cross_model import config_x, load_backbone as load_cross, PopulationModel, PersonalModel
from subject_context.cross_model_data import load_subject, original_contract
from subject_context.cross_model_run import device_seed, resource_clock
from subject_context.cross_model_training import restore_training_state
from subject_context.data import assert_split
from subject_context.model import load_backbone as load_cb, state_hash
from subject_context.population_budget import DefaultContext, continue_population, select_personal, diagnose
from subject_context.stage_a_common import ROOT, provenance, require_compute, save_json, sha256
from subject_context.stage_b import Readout
from subject_context.stage_c import ContextModel, backbone_features, config_c
from subject_context.stage_c2 import PersonalModel as CBPersonal
from subject_context.stage_c_context import TangentFeatures, band_covariances
from subject_context.stage_c_train import load_snapshot


def audit_baseline(source, diagnostic):
    """Keep historical and matched-query scores/predictions; do not rewrite X1/W."""
    old_dir = Path(source["source_diagnostic"])
    old = pd.read_csv(old_dir / "subjects.csv")
    if source["model"] == "CBraMod":
        old = old.query("variant == 'lora8'").rename(columns={"b3_ba": "G_BA"})
    new = pd.read_csv(diagnostic / "subjects.csv")
    keys = ["dataset", "subject", "fold", "seed"]
    joined = new.merge(old[keys + ["G_BA", "own_ba", "swap_ba"]], on=keys, suffixes=("_matched", "_historical"), validate="one_to_one")
    if len(joined) != len(new):
        raise ValueError("Historical score roster mismatch")
    for metric in ["G_BA", "own_ba", "swap_ba"]:
        joined[metric + "_difference"] = joined[metric + "_matched"] - joined[metric + "_historical"]
    joined.to_csv(diagnostic / "historical_score_comparison.csv", index=False)
    previous = pd.read_csv(old_dir / "predictions.csv.gz")
    current = pd.read_csv(diagnostic / "predictions.csv.gz")
    match_keys = ["dataset", "subject", "condition", "donor", "trial_id"]
    if source["model"] == "CBraMod":
        previous = previous.query("variant == 'lora8'").copy()
        previous["condition"] = "swap_matrix"
        population = pd.read_csv(Path(source["source_population"]) / "predictions.csv")
        population = population.query("method == 'm_lora' and is_query == True").copy()
        population["condition"], population["donor"] = "G", np.nan
        previous = pd.concat([previous, population], ignore_index=True)
        current = current[current.condition != "own"]  # Own is present as the matrix diagonal.
    else:
        previous = previous[previous.condition.isin(["G", "own", "swap_matrix"])]
    compared = current.merge(previous[match_keys + ["label", "prediction"]], on=match_keys,
                             suffixes=("_matched", "_historical"), validate="one_to_one", how="outer", indicator=True)
    if not (compared._merge == "both").all() or not (compared.label_matched == compared.label_historical).all():
        raise ValueError("Historical prediction trial identity mismatch")
    different = compared.prediction_matched != compared.prediction_historical
    compared[different].to_csv(diagnostic / "historical_prediction_differences.csv.gz", index=False)
    save_json(diagnostic / "historical_comparison.json", {"matched_trials": len(compared),
              "changed_predictions": int(different.sum()), "old_report_preserved": True,
              "maximum_absolute_BA_difference": {m: float(joined[m + "_difference"].abs().max()) for m in ["G_BA", "own_ba", "swap_ba"]}})


def retry_authorized(authorization, index):
    """Restrict the one additional authorization to the failed fold/seed."""
    if not authorization.get("user_confirmed"):
        raise ValueError("Recorded user authorization required")
    if authorization.get("scope") == "population_budget_75_runs":
        return False
    if (authorization.get("scope") != "population_budget_CB_retry_once"
            or index != 57 or authorization.get("array_index") != 57
            or authorization.get("maximum_attempts") != 1
            or authorization.get("numerical_policy") != "fp32_layer8_no_rounding"):
        raise ValueError("Only CBraMod fold1/seed37 has one retry authorization")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int, required=True, choices=range(75))
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--stage-a-root", type=Path, required=True)
    p.add_argument("--stage-x-root", type=Path, required=True)
    p.add_argument("--cross-weights", type=Path, required=True)
    p.add_argument("--cb-weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    require_compute()
    cfg = yaml.safe_load((ROOT / "configs/population_budget.yaml").read_text())
    authorization = json.loads(args.authorization.read_text())
    retry = retry_authorized(authorization, args.index)
    if authorization.get("inventory_sha256") != sha256(args.inventory) or authorization.get("config_sha256") != sha256(ROOT / "configs/population_budget.yaml"):
        raise ValueError("Authorization does not match this exact inventory/config")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "3090" not in torch.cuda.get_device_name():
        raise RuntimeError("Exactly one allocated RTX3090 required")
    torch.set_num_threads(4)
    inventory = json.loads(args.inventory.read_text())
    source = inventory["runs"][args.index]
    assert source["array_index"] == args.index
    if retry:
        assert (source["model"], source["fold"], source["seed"]) == ("CBraMod", 1, 37)
        previous = Path(authorization["failed_run"]["path"])
        assert sha256(previous) == authorization["failed_run"]["sha256"]
        failed = json.loads(previous.read_text())
        assert failed["status"] == "failed" and failed["error"] == "Nonfinite CBraMod training objective"
        assert failed["source"] == source
        assert sha256(ROOT / authorization["retry_protocol"]) == authorization["retry_protocol_sha256"]
        assert args.out.resolve() == Path(authorization["output_directory"]).resolve()
    for record in source["files"].values():
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Historical source checkpoint changed")
    model_name, seed, fold_index = source["model"], source["seed"], source["fold"]
    fold = json.loads((ROOT / cfg["split_file"]).read_text())["folds"][fold_index]
    assert_split(*[[tuple(row) for row in fold[r]] for r in ["train", "validation", "test"]])
    out = args.out / model_name / "runs" / f"fold-{fold_index}_seed-{seed}"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {**provenance(), "model": model_name, "fold": fold_index, "seed": seed, "status": "running",
               "scope": "population_budget", "gpu": torch.cuda.get_device_name(), "source": source,
               "inventory_sha256": sha256(args.inventory), "authorization_sha256": sha256(args.authorization),
               "config_sha256": sha256(ROOT / "configs/population_budget.yaml"),
               "protocol_sha256": sha256(ROOT / cfg["protocol"]),
               "split_sha256": sha256(ROOT / cfg["split_file"]), "halves_sha256": sha256(ROOT / cfg["halves_file"]),
               "code_sha256": {f.relative_to(ROOT).as_posix(): sha256(f) for f in sorted((ROOT / "src/subject_context").glob("*.py"))}}
    receipt["code_sha256"]["scripts/population_budget_run.py"] = sha256(Path(__file__))
    if retry:
        receipt.update(retry_attempt=1, replaces_failed_run=authorization["failed_run"],
                       numerical_policy=authorization["numerical_policy"],
                       retry_protocol_sha256=authorization["retry_protocol_sha256"])
    references = [Path(source["source_diagnostic"]) / name for name in ["subjects.csv", "predictions.csv.gz"]]
    references.append(Path(source["source_population"]) / "predictions.csv" if model_name == "CBraMod"
                      else Path(source["source_diagnostic"]) / "swap_donors.json")
    receipt["historical_diagnostic_sha256"] = {str(p): sha256(p) for p in references}
    save_json(out / "run.json", receipt)
    try:
        cfg_x, cfg_c = config_x(), config_c()
        device_seed(seed, torch.device("cuda"))
        readout = Readout(torch.load(source["files"]["B0_head"]["path"], map_location="cpu", weights_only=False), source["head_kind"]).cuda()
        backbone = load_cb(args.cb_weights, k=4).cuda().eval() if model_name == "CBraMod" else load_cross(model_name, args.cross_weights).cuda()
        frozen = state_hash(backbone), state_hash(readout)
        common = json.loads((ROOT / "configs/stage_a_channels.json").read_text())["readout_common"]

        @torch.no_grad()
        def people(role):
            result = []
            for dataset, person in fold[role]:
                if model_name != "CBraMod":
                    result.append(load_subject(args.stage_a_root, args.stage_x_root, model_name, dataset, person, role, "cuda"))
                    continue
                folder, table, labels, meta, fit, query, counts = original_contract(args.stage_a_root, dataset, person)
                picks = [meta["channels"].index(ch) for ch in common]
                s = {"key": (dataset, person), "role": role, "eeg": torch.as_tensor(np.load(folder / "tasks.npy"), device="cuda"),
                     "labels": torch.as_tensor(labels, dtype=torch.long, device="cuda"), "picks": picks,
                     "fit": torch.as_tensor(fit, device="cuda"), "query": torch.as_tensor(query, device="cuda"),
                     "query_labels": labels[query], "trial_ids": table.trial_id.tolist(), "counts": counts}
                if role == "train":
                    rest = np.load(folder / "rest_open.npy")[np.load(folder / "rest_open_valid.npy")]
                    assert meta["rest"]["open"]["protocol_order"] < table.protocol_order.min()
                    signal = torch.as_tensor(rest, device="cuda")
                    pooled = torch.cat([backbone_features(backbone.model, x.float(), picks) for x in signal.split(32)])
                    s["emb"] = readout.normalize(pooled)
                    s["bands"] = band_covariances(rest[:, picks].astype(np.float64), cfg_c["context"]["bands_hz"], cfg_c["context"]["filter_order"])
                result.append(s)
            return result

        groups = {r: people(r) for r in ["train", "validation"]}
        if model_name == "CBraMod":
            tangent = TangentFeatures().fit([s["bands"] for s in groups["train"]])
            for s in groups["train"]:
                s["tan"] = torch.as_tensor(tangent(s.pop("bands")), device="cuda")
        # Match C/W's seed placement after deterministic rest/context preparation.
        device_seed(seed, torch.device("cuda"))
        population = (ContextModel(backbone, readout, "m_lora", 5400, 756, cfg_c) if model_name == "CBraMod"
                      else PopulationModel(backbone, readout)).cuda()
        predictor = DefaultContext(population) if model_name == "CBraMod" else population
        factory = (lambda: CBPersonal(population, "lora8", 8)) if model_name == "CBraMod" else (lambda: PersonalModel(population))
        microbatch = cfg["personal"]["microbatch"][model_name]
        optimizer = torch.optim.AdamW(population.trainable(), lr=source["population_hyperparameters"]["learning_rate"], weight_decay=.01)
        rng, generator = np.random.default_rng(seed), torch.Generator(device="cuda").manual_seed(seed)
        if model_name == "CBraMod":
            load_snapshot(population, torch.load(source["files"]["checkpoint"]["path"], map_location="cuda", weights_only=True))
        else:
            old_added, _ = restore_training_state(source["files"]["checkpoint"]["path"], population, optimizer, rng, generator)
            assert old_added + source["population_hyperparameters"]["episodes"] == source["baseline_total_steps"]
        if retry:
            # Rest features have already been prepared through the historical path.
            # Apply the precision exception only to this adapted population and its
            # deep-copied personal models; input files and normalization stay fixed.
            population.core.budget_retry_fp32 = True
            population.budget_retry_failure_trace = str(out / "numerical_failure.json")
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            assert all(p.dtype == torch.float32 for p in population.parameters())
            assert cfg_c["training"]["grad_clip_norm"] == 1.
            receipt["numerical_settings"] = {"layer8_roundtrip_fp16": False, "population_parameters": "float32",
                "matmul_precision": torch.get_float32_matmul_precision(), "matmul_allow_tf32": False,
                "cudnn_allow_tf32": False, "group_grad_clip_norm_unchanged": 1., "rest_features_historical_path": True}
            save_json(out / "run.json", receipt)
        endpoints, training_status = continue_population(population, predictor, groups, optimizer, rng, generator, source, cfg_x, cfg_c, out / "G")
        peak = training_status["peak_gpu_bytes"]
        del groups["train"]
        selections = {}
        for entry in endpoints:
            point = entry["point"]
            if point == "1x":
                selections[point] = source["personal_hyperparameters_1x"]
            else:
                restore_training_state(out / "G" / entry["checkpoint"], population, optimizer, rng, generator)
                before = state_hash(population)
                selections[point] = select_personal(predictor, factory, groups["validation"], seed, microbatch, out / "personal" / point)
                assert state_hash(population) == before
                resource = pd.read_csv(out / "personal" / point / "validation_fit_resources.csv")
                peak = max(peak, int(resource.peak_gpu_bytes.max()))
        save_json(out / "personal_selections.json", selections)
        choices = [out / "personal_selections.json", out / "G/endpoints.json", out / "G/status.json"]
        choices.extend(out / "G" / e["checkpoint"] for e in endpoints)
        save_json(out / "frozen_before_test.json", {"test_seen": False, "files": {p.relative_to(out).as_posix(): sha256(p) for p in choices}})
        del groups
        test = people("test")
        for entry in endpoints:
            point = entry["point"]
            restore_training_state(out / "G" / entry["checkpoint"], population, optimizer, rng, generator)
            before = state_hash(population)
            peak = max(peak, diagnose(predictor, factory, test, selections[point], source, point, out / "diagnostic" / point, microbatch))
            assert state_hash(population) == before
            if point == "1x":
                audit_baseline(source, out / "diagnostic/1x")
        assert (state_hash(backbone), state_hash(readout)) == frozen
        assert all(sha256(Path(p)) == h for p, h in receipt["historical_diagnostic_sha256"].items())
        receipt.update(status="complete", endpoints=[e["point"] for e in endpoints], test_subjects=len(test),
                       frozen_sources_unchanged=True, maximum_segment_peak_gpu_bytes=peak, **resource_clock(started))
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error), **resource_clock(started))
        raise
    finally:
        save_json(out / "run.json", receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
