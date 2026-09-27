"""Formal orchestration contracts on tiny synthetic data, no scientific EEG results."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from subject_context.cross_model import PersonalModel, PopulationModel, config_x
from subject_context.cross_model_data import starter_people
from subject_context.cross_model_run import diagnose, personal_selection, plateau_report, train_b0, train_population
from subject_context.cross_model_training import EarlyStop, select_population
from subject_context.stage_b import select_candidate
from test_cross_model import parts


def complete_subject(s, number):
    return {**s, "key": ("Dreyer2023", number), "query_labels": s["labels"][s["query"]].numpy(),
            "trial_ids": [f"person-{number}-trial-{i}" for i in range(len(s["labels"]))],
            "counts": {"fit_left": 1, "fit_right": 1, "query_left": 1, "query_right": 1}}


def test_formal_roster_matches_235_people_and_50_jobs():
    roster = starter_people()
    assert len(roster) == len(set(roster)) == 235
    assert pd.Series([p[0] for p in roster]).value_counts().to_dict() == {"PhysionetMI":103, "Dreyer2023":80, "Cho2017":52}
    cfg = config_x()
    runs = [(model, index//5, cfg["seeds"][index % 5]) for model in cfg["models"] for index in range(25)]
    assert len(set(runs)) == 50


def test_b0_all_trial_selection_preserves_original_ties(tmp_path):
    # Constant features force every candidate BA=0.5: earliest lr/epoch and linear must win.
    _, _, source = parts("REVE")
    s = complete_subject(source, 1)
    s["features"] = torch.zeros(4, 5)
    cfg = {"learning_rates": [.001, .0003], "batch_size": 4, "max_epochs": 2, "weight_decay": .01}
    train_b0({"train": [s], "validation": [complete_subject(s, 2)]}, cfg, 11, tmp_path)
    chosen = json.loads((tmp_path / "selection.json").read_text())
    assert (chosen["kind"], chosen["learning_rate"], chosen["epoch"]) == ("linear", .001, 1)
    logs = [json.loads(line) for line in (tmp_path / "linear-lr-0.001/validation.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in logs] == [0, 1, 2]
    assert all(r["subjects"][0]["n"] == 4 and r["subjects"][0]["subset"] == "all" for r in logs)


def test_population_dense_logging_does_not_select_extra_checkpoints(tmp_path, monkeypatch):
    # Scale only iteration counts for an end-to-end CPU wiring test; select using real C ordering.
    backbone, readout, source = parts("REVE")
    cfg = copy.deepcopy(config_x())
    cfg["population"]["G"].update(initial_max_steps_per_lr=3, initial_checkpoint_candidates=[1,3])
    cfg["convergence"].update(G_check_every_steps=1, CE_early_stop_patience_checks=2)
    def small_candidates(rows):
        hp = select_population([dict(r, episodes=r["episodes"]*1000) for r in rows])
        hp["episodes"] //= 1000
        return hp
    monkeypatch.setattr("subject_context.cross_model_run.select_population", small_candidates)
    model, status = train_population(backbone, readout,
        {"train":[complete_subject(source,1)], "validation":[complete_subject(source,2)]}, cfg, 11, tmp_path, 2)
    selected = pd.read_csv(tmp_path / "initial_selection_candidates.csv")
    assert set(selected.episodes) == {1,3}
    assert len(selected) == 4
    assert status["optimizer_and_RNG_restored"] and not status["test_used_for_selection"]
    assert status["additional_steps_run"] <= status["additional_steps_cap"]
    assert model.training is False


def test_personal_selection_frozen_then_own_swap_and_few_shot(tmp_path, monkeypatch):
    backbone, readout, source = parts("REVE")
    population = PopulationModel(backbone, readout)
    cfg = {"shots": [1,2], "additional_40": [], "learning_rates": {"lora8":[.001]},
           "weight_decays":[0.], "steps":[1,2]}
    selections = personal_selection(population, [complete_subject(source,1)], cfg, 11, tmp_path / "selection", 1)
    saved_selection = (tmp_path / "selection/selection.json").read_bytes()
    diagnose(population, backbone, readout, [complete_subject(source,2), complete_subject(source,3)],
             selections, cfg, 11, 0, tmp_path / "test", 1)
    assert (tmp_path / "selection/selection.json").read_bytes() == saved_selection
    subjects = pd.read_csv(tmp_path / "test/subjects.csv")
    matrix = pd.read_csv(tmp_path / "test/swap_matrix.csv")
    predictions = pd.read_csv(tmp_path / "test/predictions.csv.gz")
    assert len(subjects) == 2 and len(matrix) == 4
    for row in subjects.itertuples():
        own = predictions.query("subject == @row.subject and condition == 'own'")
        assert row.own_ba == np.mean([(own.prediction[own.label==c] == c).mean() for c in [0,1]])
        assert row.own_ba == matrix.query("subject == @row.subject and donor == @row.subject").BA.item()
    curve = pd.read_csv(tmp_path / "test/few_shot.csv")
    assert set(curve.shots) == {1,2} and len(curve) == 4
    checks = pd.read_csv(tmp_path / "test/consistency.csv")
    assert (checks.zero_personal_max_logit_error == 0).all()
    # The independent reporter must reproduce these real synthetic predictions and reject changed truth.
    script = Path(__file__).resolve().parents[1] / "scripts/cross_model_results.py"
    spec = importlib.util.spec_from_file_location("cross_model_result_audit_test", script)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "config_x", lambda: {"halves_file": "halves.json"})
    half_rows = []
    for person in (2,3):
        s = complete_subject(source, person)
        folder = tmp_path / "raw_contract/processed/Dreyer2023" / f"sub-{person:03d}"
        folder.mkdir(parents=True)
        pd.DataFrame({"trial_id":s["trial_ids"],"label":s["labels"].numpy()}).to_csv(folder / "trials.csv",index=False)
        half_rows.append({"dataset":"Dreyer2023","subject":person,"query_indices":[2,3]})
    (tmp_path / "halves.json").write_text(json.dumps({"subjects":half_rows}))
    shutil.copytree(tmp_path / "test", tmp_path / "audited/diagnostic")
    (tmp_path / "audited/frozen_before_test.json").write_text(json.dumps({"files":{}}))
    audited, audited_few, _ = audit.audit_run(tmp_path / "audited", tmp_path / "raw_contract")
    assert len(audited) == 2 and len(audited_few) == 4
    target = tmp_path / "raw_contract/processed/Dreyer2023/sub-002/trials.csv"
    table = pd.read_csv(target); table.loc[2,"label"] = 1 - table.loc[2,"label"]
    table.to_csv(target,index=False)
    with pytest.raises(AssertionError):
        audit.audit_run(tmp_path / "audited", tmp_path / "raw_contract")


def test_early_selected_checkpoint_cannot_borrow_later_plateau():
    cfg = config_x()["convergence"]
    stop = EarlyStop(8, .0001)
    curve = []
    for i in range(10):
        value = .5 if i == 0 else .6
        stop.update(i*250, value)
        curve.append({"W_check": True, "step": i*250, "train": {"CE": .6},
                      "validation": {"CE":value,"accuracy":.7,"BA":.7}})
    result = plateau_report(curve, stop, cfg)
    assert result["plateau_observed"]
    assert not result["selected_in_final_plateau_window"]
