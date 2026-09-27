"""Audit curve predictions and report every endpoint; never shrink missing cohorts."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.population_budget import classify
from subject_context.stage_a_common import ROOT, save_json, sha256
from subject_context.statistics import holm
from cross_model_results import ba, distribution

MODELS = ["REVE", "LaBraM", "CBraMod"]
SEEDS = [11, 23, 37, 53, 71]
METRICS = ["G_BA", "own_ba", "swap_ba", "own_gain", "upper_bound"]


def audit_point(directory, stage_a_root, source, halves, expected_people):
    prediction = pd.read_csv(directory / "predictions.csv.gz")
    scores = pd.read_csv(directory / "subjects.csv")
    keys = ["dataset", "subject"]
    expected = set(map(tuple, expected_people))
    if scores.duplicated(keys).any() or set(map(tuple, scores[keys].values)) != expected:
        raise ValueError("Test cohort changed")
    if not (scores.fold == source["fold"]).all() or not (scores.seed == source["seed"]).all():
        raise ValueError("Fold/seed identity changed")
    for key, rows in prediction.groupby(keys):
        if key not in expected:
            raise ValueError("Unexpected test subject")
        table = pd.read_csv(stage_a_root / "processed" / key[0] / f"sub-{key[1]:03d}" / "trials.csv")
        required = set(table.iloc[halves[key]["query_indices"]].trial_id)
        np.testing.assert_array_equal(rows.label, table.set_index("trial_id").label.loc[rows.trial_id])
        if not rows.prediction.isin([0, 1]).all():
            raise ValueError("Nonbinary prediction")
        for _, part in rows.groupby(["condition", "donor"], dropna=False):
            if set(part.trial_id) != required or part.trial_id.duplicated().any():
                raise ValueError("Query trial count/identity changed")
    scores = scores.set_index(keys)
    absolute = prediction[prediction.condition.isin(["G", "own"])].groupby(keys + ["condition"]).apply(ba, include_groups=False).unstack("condition")
    for a, b in [(absolute.G, scores.G_BA), (absolute.own, scores.own_ba)]:
        np.testing.assert_allclose(a.loc[b.index], b, rtol=0, atol=1e-12)
    matrix = prediction[prediction.condition == "swap_matrix"].groupby(keys + ["donor"]).apply(ba, include_groups=False)
    saved = pd.read_csv(directory / "swap_matrix.csv").set_index(keys + ["donor"]).BA
    expected_matrix = {(d, s, other) for d, s in expected for d2, other in expected if d2 == d}
    if set(matrix.index) != expected_matrix or set(saved.index) != expected_matrix:
        raise ValueError("Incomplete same-dataset donor matrix")
    np.testing.assert_allclose(matrix.loc[saved.index], saved, rtol=0, atol=1e-12)
    for row in json.loads((directory / "swap_donors.json").read_text()):
        key = row["dataset"], row["subject"]
        assert len(row["donors"]) == 10 and row["subject"] not in row["donors"]
        np.testing.assert_allclose(np.mean([matrix.loc[(*key, d)] for d in row["donors"]]), scores.loc[key].swap_ba, rtol=0, atol=1e-12)
    np.testing.assert_allclose(scores.own_ba - scores.G_BA, scores.own_gain, rtol=0, atol=1e-12)
    np.testing.assert_allclose(scores.own_ba - scores.swap_ba, scores.upper_bound, rtol=0, atol=1e-12)
    return scores.reset_index(), len(prediction)


def aggregate(people, roster):
    summaries, means, availability = [], [], []
    for model in MODELS:
        for point in ["1x", "2x", "4x"]:
            rows = people[(people.model == model) & (people.point == point)]
            keys = ["dataset", "subject", "seed"]
            wanted = {(d, s, seed) for d, s in roster for seed in SEEDS}
            complete = not rows.duplicated(keys).any() and set(map(tuple, rows[keys].values)) == wanted
            availability.append({"model": model, "point": point, "rows": len(rows), "complete_235_times_5": complete})
            if not complete:
                continue
            average = rows.groupby(["dataset", "subject"])[METRICS].mean().reset_index()
            average["model"], average["point"] = model, point
            means.append(average)
            for dataset in [*sorted(average.dataset.unique()), "pooled"]:
                part = average if dataset == "pooled" else average[average.dataset == dataset]
                row = {"model": model, "point": point, "dataset": dataset, "N": len(part)}
                for metric in ["G_BA", "own_ba", "swap_ba"]:
                    row[metric + "_mean_percent"] = float(100 * part[metric].mean())
                    row[metric + "_median_percent"] = float(100 * part[metric].median())
                    row[metric + "_SD_percent"] = float(100 * part[metric].std(ddof=1))
                for name, metric in [("own_minus_G", "own_gain"), ("own_minus_swap", "upper_bound")]:
                    result = distribution(part[metric])
                    result["improve_gt_2pp_fraction"] = float((np.round(part[metric], 12) > .02).mean())
                    row.update({name + "_" + k: v for k, v in result.items()})
                if dataset == "pooled":
                    corrected = float(holm([1., 1., row["own_minus_swap_p_raw_descriptive"]])[2])
                    row.update(swap_original_three_slot_Holm=corrected,
                               swap_diagnostic1_pass=corrected < .05 and row["own_minus_swap_median_pp"] >= 2)
                summaries.append(row)
    slots = [(m, p) for m in MODELS for p in ["2x", "4x"]]
    pooled = {(r["model"], r["point"]): r for r in summaries if r["dataset"] == "pooled"}
    raw = [pooled.get(key, {}).get("own_minus_G_p_raw_descriptive", 1.) for key in slots]
    correction = dict(zip(slots, map(float, holm(raw)), strict=True))
    decisions = []
    for model in MODELS:
        medians = {p: pooled.get((model, p), {}).get("own_minus_G_median_pp") for p in ["1x", "2x", "4x"]}
        corrected = {p: correction[(model, p)] for p in ["2x", "4x"]}
        decisions.append({"model": model, "median_own_minus_G_pp": medians, "six_slot_Holm": corrected,
                          "curve_interpretation": classify(medians, corrected),
                          "personal_matching_diagnostic_pass": {p: pooled.get((model, p), {}).get("swap_diagnostic1_pass") for p in ["1x", "2x", "4x"]}})
    return summaries, pd.concat(means, ignore_index=True) if means else pd.DataFrame(), availability, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--stage-a-root", type=Path, required=True)
    args = parser.parse_args()
    report = args.out / "report"; report.mkdir(exist_ok=True, parents=True)
    inventory = json.loads(args.inventory.read_text())["runs"]
    splits = json.loads((ROOT / "configs/splits/stage_a_v1.json").read_text())["folds"]
    halves = {(r["dataset"], r["subject"]): r for r in json.loads((ROOT / "configs/splits/stage_b_halves_v1.json").read_text())["subjects"]}
    statuses, people, endpoints, validation = [], [], [], []
    predictions = 0
    for source in inventory:
        directory = args.out / source["model"] / "runs" / f"fold-{source['fold']}_seed-{source['seed']}"
        receipt = json.loads((directory / "run.json").read_text()) if (directory / "run.json").exists() else {"status": "not_started"}
        identity = {k: source[k] for k in ["model", "fold", "seed"]}
        statuses.append({**identity, "status": receipt["status"]})
        if receipt["status"] != "complete":
            continue
        if receipt["inventory_sha256"] != sha256(args.inventory):
            raise ValueError("Inventory changed")
        if receipt["config_sha256"] != sha256(ROOT / "configs/population_budget.yaml"):
            raise ValueError("Curve configuration changed")
        locked = json.loads((directory / "frozen_before_test.json").read_text())
        assert not locked["test_seen"]
        for path, digest in locked["files"].items():
            if sha256(directory / path) != digest:
                raise ValueError("Frozen choices/checkpoints changed")
        status = json.loads((directory / "G/status.json").read_text())
        assert status["added_steps"] <= source["maximum_added_steps"] == 3 * source["baseline_total_steps"]
        for entry in json.loads((directory / "G/endpoints.json").read_text()):
            point = entry["point"]
            if point in ["1x", "2x", "4x"]:
                assert entry["total_steps"] == source["target_total_steps"][point]
            path = directory / "diagnostic" / point
            rows, count = audit_point(path, args.stage_a_root, source, halves, splits[source["fold"]]["test"])
            predictions += count
            rows["model"] = source["model"]
            rows["total_steps"], rows["multiplier"] = entry["total_steps"], entry["multiplier"]
            people.append(rows)
            endpoints.append({**identity, **{k: v for k, v in entry.items() if k not in ["validation", "plateau"]}})
            for dataset, values in entry["validation"].items():
                validation.append({**identity, "point": point, "total_steps": entry["total_steps"], "dataset": dataset, **values})
        donor_files = list((directory / "diagnostic").glob("*/swap_donors.json"))
        assert len({sha256(p) for p in donor_files}) == 1
    save_json(report / "progress.json", {"runs": statuses, "completed": sum(r["status"] == "complete" for r in statuses), "audited_prediction_rows": predictions})
    if not people:
        return
    combined = pd.concat(people, ignore_index=True)
    summary, means, availability, decisions = aggregate(combined, set(halves))
    combined.to_csv(report / "all_subjects_and_actual_plateaus.csv", index=False)
    means.to_csv(report / "subject_seed_means.csv", index=False)
    pd.DataFrame(summary).to_csv(report / "curve_summary.csv", index=False)
    pd.DataFrame(endpoints).to_csv(report / "endpoints.csv", index=False)
    pd.DataFrame(validation).to_csv(report / "endpoint_validation.csv", index=False)
    save_json(report / "availability.json", availability)
    save_json(report / "interpretation.json", decisions)
    print(json.dumps(decisions))


if __name__ == "__main__":
    main()
