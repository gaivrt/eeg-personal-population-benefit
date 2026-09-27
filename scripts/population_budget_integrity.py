"""CPU-only provenance, epoch-log and selection audit of terminal budget runs.

This reads existing artifacts; it never fits a model or changes scientific rules.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def window_flags(checks):
    tail = checks[-8:]
    full = len(tail) == 8
    result = {}
    for name in ("train", "validation"):
        values = [r[name]["CE"] for r in tail]
        result[name + "_loss_plateau"] = full and max(values) - min(values) <= max(1e-4, abs(statistics.mean(values)) * .01)
    score = full
    for name in ("accuracy", "BA"):
        values = [r["validation"][name] for r in checks]
        recent = values[-8:]
        anchor = max(0, len(checks) - 9)
        score = score and max(recent) - min(recent) <= .005 and max(values) - max(values[:anchor + 1]) <= .001
    result["validation_scores_plateau"] = score
    result["plateau_observed"] = all(result.values())
    return result


def audit_logs(directory, source, fold):
    group = directory / "G"
    validation, training = lines(group / "validation.jsonl"), lines(group / "train_all_trials.jsonl")
    baseline, cap = source["baseline_total_steps"], source["maximum_added_steps"]
    status = read(group / "status.json") if (group / "status.json").exists() else None
    final_step = validation[-1]["added_steps"]
    require(cap == 3 * baseline and final_step <= cap, "Budget exceeded")
    ntrain = len(fold["train"])
    expected = set(range(0, final_step + 1, ntrain)) | set(range(0, final_step + 1, 250))
    expected |= {s for s in (0, baseline, cap) if s <= final_step}
    if status:
        expected.add(final_step)
    steps = [r["added_steps"] for r in validation]
    require(steps == sorted(expected), "Missing, duplicate or unexpected validation/epoch record")
    require([r["added_steps"] for r in training] == list(range(0, final_step + 1, 250)), "Training plateau checks missing")
    epochs = {r["added_steps"] for r in validation if r["epoch_record"]}
    required_epochs = set(range(0, final_step + 1, ntrain)) | ({final_step} if status else set())
    require(epochs == required_epochs, "Epoch flags differ from required complete/final epochs")
    for role, records in (("validation", validation), ("train", training)):
        roster = {tuple(p) for p in fold[role]}
        for record in records:
            step = record["added_steps"]
            require(record["total_steps"] == baseline + step, "Wrong cumulative step")
            require(record["regular_check"] == (step % 250 == 0), "Incorrect plateau-window membership")
            people = record["subjects"]
            require(len(people) == len(roster) and {(p["dataset"], p["subject"]) for p in people} == roster, "Wrong training/validation roster")
            for metric in ("CE", "accuracy", "BA"):
                require(all(math.isfinite(p[metric]) for p in people), "Nonfinite saved metric")
                for dataset, values in record["metrics"].items():
                    subset = [p[metric] for p in people if dataset == "pooled" or p["dataset"] == dataset]
                    require(math.isclose(statistics.mean(subset), values[metric], abs_tol=1e-12, rel_tol=0), "Incorrect subject-equal metric")
    by_step = {r["added_steps"]: r for r in validation}
    checks = [{"step": r["added_steps"], "train": r["metrics"]["pooled"], "validation": by_step[r["added_steps"]]["metrics"]["pooled"]} for r in training]
    first_platform = next((r["step"] for i, r in enumerate(checks) if window_flags(checks[:i + 1])["plateau_observed"]), None)
    require(first_platform is None or first_platform == final_step, "Training continued after first platform")
    endpoints = read(group / "endpoints.json")
    for e in endpoints:
        require(e["total_steps"] == baseline + e["added_steps"] <= 4 * baseline, "Endpoint step error")
        if e["point"] in ("1x", "2x", "4x"):
            require(e["total_steps"] == source["target_total_steps"][e["point"]], "Mislabelled endpoint")
        require(e["validation"] == by_step[e["added_steps"]]["metrics"], "Endpoint validation changed")
    expected_points = [name for name, step in (("1x", 0), ("2x", baseline), ("4x", cap)) if step <= final_step]
    if first_platform is not None and first_platform not in (0, baseline, cap):
        expected_points.append(f"plateau-{baseline + first_platform}")
    require([e["point"] for e in endpoints] == expected_points, "Missing or fabricated group endpoint")
    samples = 0
    with (group / "sampling.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            samples += 1
            require(row["added_steps"] == samples and row["total_steps"] == baseline + samples, "Sampling steps not contiguous")
            require((row["dataset"], row["subject"]) in {tuple(p) for p in fold["train"]}, "Sample outside training fold")
            require(math.isfinite(row["CE"]) and math.isfinite(row["objective"]), "Nonfinite recorded training sample")
    require(final_step <= samples <= cap, "Sampled update budget invalid")
    if status:
        require(samples == status["added_steps"] == final_step, "Final sampled/logged steps disagree")
        require(status["plateau"] == window_flags(checks), "Recorded platform flags disagree")
        require(len(status["platform_checks"]) == len(checks), "Platform check count changed")
        for saved, computed in zip(status["platform_checks"], checks):
            require(all(saved[k] == computed[k] for k in ("step", "train", "validation")), "Platform trace changed")
        require(status["stop_reason"] == ("joint_plateau" if first_platform is not None else "4x_cap"), "Stop reason mismatch")
        require(first_platform is not None or final_step == cap, "Premature nonplatform stop")
    return {"validation_records": len(validation), "epoch_records": len(epochs), "regular_checks": len(checks),
            "last_logged_added_step": final_step, "sampled_updates": samples, "first_platform_added_step": first_platform,
            "saved_points": expected_points, "group_terminal": status is not None}, endpoints, by_step


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    require(bool(os.environ.get("SLURM_JOB_ID")), "Run this audit in a CPU Slurm allocation")
    require(not args.report.exists(), "Preserve earlier integrity reports")
    import pandas as pd
    import torch
    from subject_context.stage_b import select_candidate
    torch.set_num_threads(4)
    require(not torch.cuda.is_available(), "This audit must not use GPUs")
    digest_cache = {}

    def digest(path):
        path = Path(path)
        if path not in digest_cache:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            digest_cache[path] = h.hexdigest()
        return digest_cache[path]

    inventory = read(args.inventory)["runs"]
    splits = read(ROOT / "configs/splits/stage_a_v1.json")["folds"]
    auth = read(ROOT / "authorization.json")
    results, errors, endpoint_rows = [], [], []
    for source in inventory:
        directory = args.out / source["model"] / "runs" / f"fold-{source['fold']}_seed-{source['seed']}"
        if not (directory / "run.json").exists():
            continue
        run = read(directory / "run.json")
        if run["status"] not in ("complete", "failed"):
            continue
        identity = {k: source[k] for k in ("model", "fold", "seed", "array_index")}
        result = {**identity, "run_status": run["status"]}
        try:
            require(run["source"] == source, "Source inventory entry changed")
            for key, path in (("inventory_sha256", args.inventory), ("config_sha256", ROOT / "configs/population_budget.yaml"),
                              ("protocol_sha256", ROOT / "docs/population_budget_protocol.md"), ("authorization_sha256", ROOT / "authorization.json"),
                              ("split_sha256", ROOT / "configs/splits/stage_a_v1.json"), ("halves_sha256", ROOT / "configs/splits/stage_b_halves_v1.json")):
                require(run[key] == digest(path), "Run provenance mismatch: " + key)
                if key in auth:
                    require(auth[key] == run[key], "Authorization mismatch: " + key)
            for name, expected in run["code_sha256"].items():
                require(digest(ROOT / name) == expected, "Training code changed: " + name)
            for entry in source["files"].values():
                require(digest(entry["path"]) == entry["sha256"], "Historical source changed")
            for name, expected in run["historical_diagnostic_sha256"].items():
                require(digest(name) == expected, "Historical diagnostic changed")
            if auth.get("scope") == "population_budget_CB_retry_once":
                require((source["model"], source["fold"], source["seed"], source["array_index"]) == ("CBraMod", 1, 37, 57), "Unauthorized retry identity")
                require(run.get("retry_attempt") == auth["maximum_attempts"] == 1, "Retry count changed")
                require(run.get("numerical_policy") == auth["numerical_policy"] == "fp32_layer8_no_rounding", "Precision exception changed")
                require(run["retry_protocol_sha256"] == auth["retry_protocol_sha256"] == digest(ROOT / auth["retry_protocol"]), "Retry protocol changed")
                require(run["replaces_failed_run"] == auth["failed_run"] and digest(auth["failed_run"]["path"]) == auth["failed_run"]["sha256"], "Original failure changed")
            log_audit, endpoints, by_step = audit_logs(directory, source, splits[source["fold"]])
            result.update(log_audit)
            for endpoint in endpoints:
                if auth.get("scope") == "population_budget_CB_retry_once":
                    require(endpoint.get("numerical_policy") == auth["numerical_policy"], "Checkpoint precision metadata missing")
                for dataset, metric in endpoint["validation"].items():
                    endpoint_rows.append({**identity, "run_status": run["status"], "point": endpoint["point"],
                                          "total_steps": endpoint["total_steps"], "dataset": dataset, **metric})
                ckpt = torch.load(directory / "G" / endpoint["checkpoint"], map_location="cpu", weights_only=True)
                require(ckpt["step"] == endpoint["added_steps"], "Checkpoint step mismatch")
                require(all(torch.isfinite(t).all() for t in ckpt["parameters"].values()), "Nonfinite saved group parameters")
                require(all(k in ckpt for k in ("optimizer", "numpy_rng", "torch_generator", "torch_rng", "cuda_rng")), "Missing continuation state")
                if endpoint["point"] == "1x":
                    old = torch.load(source["files"]["checkpoint"]["path"], map_location="cpu", weights_only=True)
                    old_params = old if source["model"] == "CBraMod" else old["parameters"]
                    require(set(old_params) == set(ckpt["parameters"]) and all(torch.equal(t, ckpt["parameters"][k]) for k, t in old_params.items()), "1x parameters differ from source G")
                    del old, old_params
                del ckpt
            if run["status"] == "complete":
                require(run["frozen_sources_unchanged"], "Missing frozen-source completion check")
                locked = read(directory / "frozen_before_test.json")
                require(locked["test_seen"] is False, "Selection was not frozen before test")
                for name, expected in locked["files"].items():
                    require(digest(directory / name) == expected, "Frozen artifact changed")
                choices = read(directory / "personal_selections.json")
                require(choices["1x"] == source["personal_hyperparameters_1x"], "1x selection changed")
                old_diag = Path(source["source_diagnostic"])
                if source["model"] == "CBraMod":
                    old_rows = pd.read_csv(old_diag / "subjects.csv").query("variant == 'lora8'")
                    original_donors = {(r.dataset, int(r.subject)): [int(x) for x in r.swap_donors.split()] for r in old_rows.itertuples()}
                else:
                    original_donors = {(r["dataset"], r["subject"]): r["donors"] for r in read(old_diag / "swap_donors.json")}
                for endpoint in endpoints:
                    point = endpoint["point"]
                    donor_rows = read(directory / "diagnostic" / point / "swap_donors.json")
                    donors = {(r["dataset"], r["subject"]): r["donors"] for r in donor_rows}
                    require(len(donor_rows) == len(original_donors) and donors == original_donors, "Historical donor identity/order changed")
                    if point == "1x":
                        require(not (directory / "personal/1x").exists(), "Unexpected 1x personal search")
                        continue
                    search = pd.read_csv(directory / "personal" / point / "validation_search.csv", float_precision="round_trip")
                    expected = {(d, s, lr, wd, step) for d, s in splits[source["fold"]]["validation"] for lr in (1e-4, 1e-3) for wd in (0., .01) for step in (20, 60)}
                    actual = set(search[["dataset", "subject", "learning_rate", "weight_decay", "steps"]].itertuples(index=False, name=None))
                    require(actual == expected and len(search) == len(expected), "Personal validation grid/cohort changed")
                    base = {(r["dataset"], r["subject"]): r["BA"] for r in by_step[endpoint["added_steps"]]["subjects"]}
                    require(all(math.isclose(r.gain, r.BA - base[(r.dataset, r.subject)], abs_tol=1e-12, rel_tol=0) for r in search.itertuples()), "Personal validation gain mismatch")
                    selected = select_candidate(search.to_dict("records"))
                    require(all(math.isclose(selected[k], choices[point][k], abs_tol=1e-12, rel_tol=0) for k in selected), "Personal choice differs from frozen validation ranking")
                result["personal_selection_and_historical_donors_verified"] = True
            result["audit_pass"] = True
        except Exception as error:
            result.update(audit_pass=False, audit_error=f"{type(error).__name__}: {error}")
            errors.append({**identity, "error": result["audit_error"]})
        results.append(result)
    payload = {"scope": "terminal runs only; no training or test-based selection", "slurm_job_id": os.environ["SLURM_JOB_ID"],
               "runs": results, "counts": dict(Counter(r["run_status"] for r in results)), "errors": errors,
               "file_sha256": {str(k): v for k, v in digest_cache.items()}, "endpoint_validation_including_failed_runs": endpoint_rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps({"audited": len(results), "counts": payload["counts"], "errors": errors}))
    require(not errors, "Integrity audit found discrepancies; preserve report and inspect")


if __name__ == "__main__":
    main()
