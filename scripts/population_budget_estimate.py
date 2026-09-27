"""Read historical timing only; no model loading, fitting or evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def estimate(inventory):
    records = []
    for source in inventory["runs"]:
        population = Path(source["source_population"])
        diagnostic = Path(source["source_diagnostic"])
        if source["model"] == "CBraMod":
            history = json.loads((population / "run.json").read_text())
            updates = sum(v["additional_steps_run"] for v in history["families"].values())
            rate = history["seconds"] / updates
            # Historical timers cover both parameterizations, including context setup/evaluation.
            # LoRA-only rate was not recorded: disclose this proxy and reserve 50% for it.
            group_low = source["maximum_added_steps"] * rate
            group_high = 1.5 * group_low
            fits = pd.read_csv(diagnostic / "fit_resources.csv")
            lo = fits[fits.variant == "lora8"]
            search = lo[lo.phase == "validation"].fit_seconds.sum()
            own = lo[lo.phase == "test"].fit_seconds.sum()
            diagnostic_receipt = json.loads((diagnostic / "run.json").read_text())
            other = max(0., diagnostic_receipt["seconds"] - fits.fit_seconds.sum())
            rate_source = "W both-family whole-run seconds/update proxy; high includes 50% allowance"
        else:
            logs = [json.loads(line) for line in (population / "continuation/validation.jsonl").read_text().splitlines()]
            rate = (logs[-1]["seconds"] - logs[0]["seconds"]) / logs[-1]["step"]
            group_low = source["maximum_added_steps"] * rate
            group_high = 1.15 * group_low
            fits = pd.read_csv(population.parent / "personal/validation_fit_resources.csv")
            search = fits[fits.n.isna()].fit_seconds.sum()
            test_fits = pd.read_csv(diagnostic / "fit_resources.csv")
            own = test_fits[test_fits.n.isna()].fit_seconds.sum()
            run = json.loads((population.parent / "run.json").read_text())
            old_g = json.loads((population / "convergence.json").read_text())["seconds"]
            # Residual includes all old B0 work, load, diagnostics and matrix inference.
            # Reusing it for each point is conservative; B0 is not actually retrained.
            other = max(0., run["seconds"] - old_g - fits.fit_seconds.sum() - test_fits.fit_seconds.sum())
            rate_source = "measured same-model continuation slope incl validation; high includes 15% allowance"
        new_own = own * 60 / source["personal_hyperparameters_1x"]["steps"]
        low = group_low + 2 * (search + own) + 1.5 * other
        high = group_high + 2 * (search + new_own) + 3 * other
        files = source["files"]
        # Four group files (1x,2x,4x,last), two new full-personal sets, logging reserve separately.
        group_bytes = files["checkpoint"]["bytes"] * (3 if source["model"] == "CBraMod" else 1)
        records.append({"index": source["array_index"], "model": source["model"], "baseline_steps": source["baseline_total_steps"],
                        "maximum_added_steps": source["maximum_added_steps"], "historical_seconds_per_group_update": rate,
                        "rate_source": rate_source, "group_low_seconds": group_low, "group_high_seconds": group_high,
                        "one_point_validation_full_grid_seconds": search, "historical_test_fit_seconds": own,
                        "test_fit_60_step_estimate_seconds": new_own, "historical_residual_seconds": other,
                        "low_seconds": low, "high_seconds": high,
                        "checkpoint_bytes_estimate": 4 * group_bytes + 2 * files["personal_parameters"]["bytes"]})
    frame = pd.DataFrame(records)
    summary = []
    for model, rows in frame.groupby("model", sort=False):
        summary.append({"model": model, "runs": len(rows), "max_new_updates": int(rows.maximum_added_steps.sum()),
                        "card_hours_low": float(rows.low_seconds.sum() / 3600), "card_hours_high": float(rows.high_seconds.sum() / 3600),
                        "max_single_job_hours_high": float(rows.high_seconds.max() / 3600),
                        "checkpoint_GB_estimate": float(rows.checkpoint_bytes_estimate.sum() / 1e9)})
    # Simple FIFO 16-slot scheduling, no claim that the scheduler guarantees this order.
    makespans = {}
    for bound in ["low", "high"]:
        slots = np.zeros(16)
        finishes = []
        for row in records:
            slot = int(slots.argmin())
            slots[slot] += row[bound + "_seconds"]
            finishes.append({"model": row["model"], "seconds": float(slots[slot])})
        makespans[bound] = {model: max(r["seconds"] for r in finishes if r["model"] == model) / 3600 for model in frame.model.unique()}
    return {"assumption": "all trajectories reach 4x; no new experiment or throughput pilot", "per_model": summary,
            "total_card_hours": [float(frame.low_seconds.sum()/3600), float(frame.high_seconds.sum()/3600)],
            "FIFO_16_GPU_model_finish_hours": makespans, "runs": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = estimate(json.loads(args.inventory.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}))
