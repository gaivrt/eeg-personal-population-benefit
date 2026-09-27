"""Scientific invariants: hard cap, plateau, historical update and matched donors."""
import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from subject_context.cross_model import config_x
from subject_context.cross_model_training import save_training_state, restore_training_state
from subject_context.population_budget import cbramod_update, classify, continue_population, diagnose, endpoint, source_donors
from subject_context.stage_c_train import train


class TinyContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[.1, -.2], [.3, -.1]]))
        self.offset = nn.Parameter(torch.tensor([.2, -.2]))

    def z(self, emb=None, tan=None):
        return self.offset if emb is None else self.offset + emb.mean(0) + tan.mean(0)

    def forward(self, eeg, picks, z=None):
        return eeg @ self.weight + (self.offset if z is None else z)

    def trainable(self):
        return list(self.parameters())


def subjects():
    gen = torch.Generator().manual_seed(7)
    return [{"key": ("data", s), "role": "train", "eeg": torch.randn(8, 2, generator=gen),
             "picks": [], "labels": torch.tensor([0, 1] * 4), "query": torch.arange(8),
             "emb": torch.randn(12, 2, generator=gen), "tan": torch.randn(12, 2, generator=gen)} for s in range(3)]


def test_historical_cbramod_step_and_exact_resume(tmp_path):
    cfg = {"training": {"weight_decay": .01, "query_batch": 4, "context_dropout": .2,
                         "grad_clip_norm": 1., "episodes": [12]}, "context": {"min_seconds": 30, "segment_seconds": 4}}
    hp = {"learning_rate": .001, "lambda": 1., "margin": .1}
    old, new = TinyContext(), TinyContext()
    data, old_steps, new_steps = subjects(), [], []
    train(old, data, hp, cfg, 11, lambda _: None, lambda i, key, ce, loss: old_steps.append((key, ce, loss)))
    opt = torch.optim.AdamW(new.trainable(), lr=.001, weight_decay=.01)
    rng, generator = np.random.default_rng(11), torch.Generator().manual_seed(11)
    for i in range(12):
        if i == 5:
            save_training_state(tmp_path / "state.pt", new, opt, rng, generator, i, {})
            new = TinyContext()
            opt = torch.optim.AdamW(new.trainable(), lr=.9)
            rng, generator = np.random.default_rng(999), torch.Generator().manual_seed(999)
            assert restore_training_state(tmp_path / "state.pt", new, opt, rng, generator)[0] == 5
        row = cbramod_update(new, data, hp, cfg, opt, rng, generator)
        new_steps.append(((row["dataset"], row["subject"]), row["CE"], row["objective"]))
    assert old_steps == new_steps
    for a, b in zip(old.parameters(), new.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("flat,expected_added,expected_points", [(False, 12, ["1x", "2x", "4x"]), (True, 7, ["1x", "2x", "plateau-11"])])
def test_real_training_loop_cap_and_plateau(tmp_path, flat, expected_added, expected_points):
    population = TinyContext()
    optimizer = torch.optim.AdamW(population.trainable(), lr=0 if flat else .1)
    cfg = copy.deepcopy(config_x())
    cfg["convergence"]["G_check_every_steps"] = 1
    # Force a nonplateau loss test in the nonflat path; it must run beyond CE patience.
    if not flat:
        cfg["convergence"].update(loss_absolute_range_floor=0., loss_relative_range=0., CE_early_stop_patience_checks=1)
    data = subjects()
    source = {"model": "REVE", "baseline_total_steps": 4, "maximum_added_steps": 12, "optimizer_resume": True}
    points, status = continue_population(population, population, {"train": data, "validation": data}, optimizer,
                                         np.random.default_rng(11), torch.Generator().manual_seed(11), source, cfg, {}, tmp_path / "G")
    assert status["added_steps"] == expected_added
    assert [p["point"] for p in points] == expected_points
    lines = [json.loads(line) for line in (tmp_path / "G/validation.jsonl").read_text().splitlines()]
    assert lines[-1]["epoch_record"]
    assert all({"CE", "accuracy", "BA"} <= set(line["metrics"]["pooled"]) for line in lines)
    saved = torch.load(tmp_path / "G/last.pt", weights_only=True)
    assert saved["step"] == expected_added
    with pytest.raises(ValueError):
        endpoint(13, 4)


def test_fixed_donors_and_invalid_self_swap(tmp_path):
    path = tmp_path / "swap_donors.json"
    rows = [{"dataset": "d", "subject": 1, "donors": [2] * 10}, {"dataset": "d", "subject": 2, "donors": [1] * 10}]
    path.write_text(json.dumps(rows))
    source = {"source_diagnostic": str(tmp_path), "model": "REVE"}
    test = [{"key": ("d", 1)}, {"key": ("d", 2)}]
    assert source_donors(source, test)[("d", 1)] == [2] * 10
    rows[0]["donors"][0] = 1
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        source_donors(source, test)


def test_curve_rules_strict_boundary_precedence_missingness():
    p = {"2x": .01, "4x": .01}
    assert classify({"1x": 1., "2x": .4, "4x": .3}, p) == "primarily_population_undertraining"
    assert classify({"1x": 1., "2x": 1.1, "4x": 1.}, p) == "stable_positive_after_continuation"
    assert classify({"1x": 1., "2x": 1.1, "4x": .6}, p) == "descriptive_only"  # exactly 0.5 pp
    assert classify({"1x": 1., "2x": .7, "4x": .5}, {"2x": .05, "4x": .01}) == "descriptive_only"
    assert classify({"1x": 1., "2x": 1.1}, p) == "undetermined_missing_endpoints"
    assert classify({"1x": .2, "2x": .2, "4x": .2}, {}) == "descriptive_only"


def test_extra_epoch_and_endpoint_logs_do_not_compress_plateau(tmp_path):
    population = TinyContext()
    cfg = copy.deepcopy(config_x())
    cfg["convergence"]["G_check_every_steps"] = 4
    source = {"model": "REVE", "baseline_total_steps": 10, "maximum_added_steps": 30, "optimizer_resume": True}
    data = subjects()
    points, status = continue_population(population, population, {"train": data, "validation": data},
        torch.optim.AdamW(population.trainable(), lr=0), np.random.default_rng(11), torch.Generator().manual_seed(11),
        source, cfg, {}, tmp_path / "G")
    assert status["added_steps"] == 28
    assert [r["step"] for r in status["platform_checks"]] == list(range(0, 29, 4))
    assert [r["point"] for r in points] == ["1x", "2x", "plateau-38"]


def test_baseline_reuses_personal_and_computes_own_swap(tmp_path, monkeypatch):
    class Population(nn.Module):
        def forward(self, eeg, picks):
            return torch.zeros(len(eeg), 2)

    class Personal(nn.Module):
        def __init__(self):
            super().__init__()
            self.delta = nn.Parameter(torch.zeros(()))

        @torch.no_grad()
        def reset(self):
            self.delta.zero_()

        @torch.no_grad()
        def load_personal(self, values):
            self.delta.copy_(values[0])

        def forward(self, eeg, picks):
            return torch.stack([-eeg[:, 0] * self.delta, eeg[:, 0] * self.delta], dim=1)

    def forbidden(*a, **kw):
        raise AssertionError("1x must not refit")

    monkeypatch.setattr("subject_context.population_budget.fit", forbidden)
    historical = tmp_path / "history"; historical.mkdir()
    donor_rows = [{"dataset": "d", "subject": 1, "donors": [2]*10}, {"dataset": "d", "subject": 2, "donors": [1]*10}]
    (historical / "swap_donors.json").write_text(json.dumps(donor_rows))
    torch.save({"d:1": [torch.tensor(1.)], "d:2": [torch.tensor(-1.)]}, historical / "personal.pt")
    data = [{"key": ("d", s), "eeg": torch.tensor([[-1.], [1.]]), "picks": [], "query": torch.arange(2),
             "labels": torch.tensor(labels), "query_labels": np.array(labels), "trial_ids": ["a", "b"], "counts": {}}
            for s, labels in [(1, [0, 1]), (2, [1, 0])]]
    source = {"model": "REVE", "fold": 0, "seed": 11, "source_diagnostic": str(historical),
              "files": {"personal_parameters": {"path": str(historical / "personal.pt")}}}
    diagnose(Population(), Personal, data, {}, source, "1x", tmp_path / "diagnostic", 8)
    import pandas as pd
    rows = pd.read_csv(tmp_path / "diagnostic/subjects.csv")
    np.testing.assert_array_equal(rows.G_BA, [.5, .5])
    np.testing.assert_array_equal(rows.own_gain, [.5, .5])
    np.testing.assert_array_equal(rows.upper_bound, [1., 1.])


def test_report_requires_every_seed_and_keeps_missing_Holm_slots():
    import sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from population_budget_results import aggregate
    rows = [{"model": "REVE", "point": p, "dataset": "d", "subject": s, "seed": seed,
             "G_BA": .6, "own_ba": .62, "swap_ba": .61, "own_gain": .02, "upper_bound": .01}
            for p in ["1x", "2x", "4x"] for s in range(1, 21) for seed in [11, 23, 37, 53, 71]]
    full = pd.DataFrame(rows)
    summary, means, availability, decisions = aggregate(full, {("d", s) for s in range(1, 21)})
    assert decisions[0]["curve_interpretation"] == "stable_positive_after_continuation"
    assert decisions[1]["curve_interpretation"] == "undetermined_missing_endpoints"
    assert decisions[1]["six_slot_Holm"] == {"2x": 1., "4x": 1.}
    raw = next(r for r in summary if r["model"] == "REVE" and r["point"] == "2x" and r["dataset"] == "pooled")["own_minus_G_p_raw_descriptive"]
    assert decisions[0]["six_slot_Holm"]["2x"] == pytest.approx(6 * raw)
    _, _, availability, decisions = aggregate(full.iloc[:-1], {("d", s) for s in range(1, 21)})
    assert decisions[0]["curve_interpretation"] == "undetermined_missing_endpoints"
    assert next(r for r in availability if r["model"] == "REVE" and r["point"] == "4x")["complete_235_times_5"] is False
