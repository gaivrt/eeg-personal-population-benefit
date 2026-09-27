"""Ensure the independent log audit rejects missing or fabricated evidence."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from population_budget_integrity import audit_logs, window_flags


def make_logs(tmp_path, failed=False):
    group = tmp_path / "G"
    group.mkdir()
    fold = {"train": [["d", 1], ["d", 2]], "validation": [["d", 3]]}
    source = {"baseline_total_steps": 10, "maximum_added_steps": 30,
              "target_total_steps": {"1x": 10, "2x": 20, "4x": 40}}
    final = 14 if failed else 30
    metric = {"CE": .5, "accuracy": .6, "BA": .6}

    def row(step, role):
        return {"added_steps": step, "total_steps": 10 + step,
                "regular_check": step % 250 == 0, "epoch_record": step % 2 == 0,
                "metrics": {"d": metric, "pooled": metric},
                "subjects": [{"dataset": d, "subject": s, **metric} for d, s in fold[role]]}

    validation = [row(s, "validation") for s in range(0, final + 1, 2)]
    (group / "validation.jsonl").write_text("\n".join(map(json.dumps, validation)))
    (group / "train_all_trials.jsonl").write_text(json.dumps(row(0, "train")))
    samples = final + 1 if failed else final
    (group / "sampling.jsonl").write_text("\n".join(json.dumps({"added_steps": s, "total_steps": 10 + s,
        "dataset": "d", "subject": 1, "CE": .5, "objective": .5}) for s in range(1, samples + 1)))
    points = [{"point": name, "added_steps": step, "total_steps": 10 + step,
               "validation": {"d": metric, "pooled": metric}}
              for name, step in (("1x", 0), ("2x", 10), ("4x", 30)) if step <= final]
    (group / "endpoints.json").write_text(json.dumps(points))
    if not failed:
        checks = [{"step": 0, "train": metric, "validation": metric}]
        status = {"added_steps": final, "plateau": window_flags(checks), "platform_checks": checks, "stop_reason": "4x_cap"}
        (group / "status.json").write_text(json.dumps(status))
    return source, fold, validation


@pytest.mark.parametrize("failed", [False, True])
def test_complete_and_numerical_failure_logs_are_distinguished(tmp_path, failed):
    source, fold, _ = make_logs(tmp_path, failed)
    report, _, _ = audit_logs(tmp_path, source, fold)
    assert report["group_terminal"] is not failed
    assert report["saved_points"] == (["1x", "2x"] if failed else ["1x", "2x", "4x"])
    assert report["sampled_updates"] == (15 if failed else 30)


@pytest.mark.parametrize("corruption", ["missing_epoch", "test_subject", "fabricated_endpoint", "over_budget"])
def test_audit_rejects_corrupted_evidence(tmp_path, corruption):
    source, fold, records = make_logs(tmp_path)
    if corruption == "missing_epoch":
        records.pop(2)
    elif corruption == "test_subject":
        records[2]["subjects"][0]["subject"] = 99
    elif corruption == "fabricated_endpoint":
        p = tmp_path / "G/endpoints.json"
        points = json.loads(p.read_text())
        points[-1]["total_steps"] = 39
        p.write_text(json.dumps(points))
    else:
        source["maximum_added_steps"] = 29
    (tmp_path / "G/validation.jsonl").write_text("\n".join(map(json.dumps, records)))
    with pytest.raises(ValueError):
        audit_logs(tmp_path, source, fold)


def test_loss_flatness_does_not_mask_score_drift():
    checks = [{"step": i * 250, "train": {"CE": .5},
               "validation": {"CE": .5, "accuracy": .6, "BA": .6}} for i in range(8)]
    assert window_flags(checks)["plateau_observed"]
    changed = copy.deepcopy(checks)
    changed[-1]["validation"]["accuracy"] = .61
    assert not window_flags(changed)["plateau_observed"]
    changed = copy.deepcopy(checks)
    changed[-1]["validation"]["BA"] = .602
    assert not window_flags(changed)["plateau_observed"]  # best improves by > 0.1 pp
