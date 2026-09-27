"""CPU contracts for real upstream layers, frozen parameters, logging and continuation."""
from functools import partial
import json

import numpy as np
import pytest
import torch
from torch import nn

from subject_context.cross_model import (
    CrossBackbone, PersonalModel, PopulationModel, channel_tables, official_classes,
)
from subject_context.cross_model_training import (
    EarlyStop, append_validation, evaluate, record_events, restore_training_state,
    save_training_state, select_population, summarize, update,
)
from subject_context.model import state_hash
from subject_context.stage_b import Readout


def parts(name):
    torch.manual_seed(11)
    Config, Reve, labram = official_classes()
    if name == "REVE":
        core = Reve(Config(embed_dim=32, depth=2, heads=2, head_dim=16, mlp_dim_ratio=2))
        positions = torch.randn(len(channel_tables()[0]), 3)
    else:
        core = labram.NeuralTransformer(embed_dim=200, depth=2, num_heads=10, num_classes=0,
                qk_norm=partial(nn.LayerNorm, eps=1e-6), norm_layer=partial(nn.LayerNorm, eps=1e-6),
                use_mean_pooling=False, init_values=.1)
        positions = None
    backbone = CrossBackbone(name, core, positions)
    width = 3 * backbone.width
    head = nn.Linear(width, 2)
    readout = Readout({"head": head.state_dict(), "feature_mean": np.zeros(width), "feature_std": np.ones(width)}, "linear")
    s = {"key": ("synthetic", 1), "picks": {"channels": ["C3", "Cz", "C4"], "readout": [0, 1, 2]},
         "eeg": torch.randn(4, 3, 400), "labels": torch.tensor([0, 1, 0, 1]),
         "fit": torch.tensor([0, 1]), "query": torch.tensor([2, 3])}
    return backbone, readout, s


@pytest.mark.parametrize("name", ["REVE", "LaBraM"])
def test_real_upstream_zero_lora_all_projection_gradients_and_personal_reload(name):
    backbone, readout, s = parts(name)
    frozen = state_hash(backbone), state_hash(readout)
    baseline = readout(backbone(s["eeg"], s["picks"]))
    population = PopulationModel(backbone, readout)
    torch.testing.assert_close(population(s["eeg"], s["picks"]), baseline, atol=0, rtol=0)
    assert sum(p.numel() for m in population.lora for p in m.parameters()) == 2 * 4 * 8 * 2 * backbone.width
    immutable = {n: p.detach().clone() for n, p in population.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW(population.trainable(), lr=.001)
    update(population, s, torch.arange(4), opt)
    for module in population.lora:
        assert module.b.grad is not None
        assert (module.b.grad.flatten(1).norm(dim=1) > 0).all()  # Each Q/K/V slice, including F.linear access.
    update(population, s, torch.arange(4), opt)
    for module in population.lora:
        assert (module.a.grad.flatten(1).norm(dim=1) > 0).all()
    for n, p in population.named_parameters():
        if n in immutable:
            assert p.grad is None
            torch.testing.assert_close(p, immutable[n], atol=0, rtol=0)
    assert (state_hash(backbone), state_hash(readout)) == frozen
    population_hash = state_hash(population)
    expected = population(s["eeg"], s["picks"]).detach()
    personal = PersonalModel(population)
    torch.testing.assert_close(personal(s["eeg"], s["picks"]), expected, atol=0, rtol=0)
    opt = torch.optim.AdamW(personal.trainable(), lr=.001)
    update(personal, s, s["fit"], opt)
    parameters = personal.personal()
    own = personal(s["eeg"], s["picks"]).detach()
    personal.reset()
    torch.testing.assert_close(personal(s["eeg"], s["picks"]), expected, atol=0, rtol=0)
    personal.load_personal(parameters)
    torch.testing.assert_close(personal(s["eeg"], s["picks"]), own, atol=0, rtol=0)
    assert state_hash(population) == population_hash


@pytest.mark.parametrize("name", ["REVE", "LaBraM"])
def test_microbatch_preserves_effective_update(name):
    backbone, readout, s = parts(name)
    torch.manual_seed(19); a = PopulationModel(backbone, readout)
    torch.manual_seed(19); b = PopulationModel(backbone, readout)
    oa = torch.optim.SGD(a.trainable(), lr=.01)
    ob = torch.optim.SGD(b.trainable(), lr=.01)
    update(a, s, torch.arange(4), oa, microbatch=4)
    update(b, s, torch.arange(4), ob, microbatch=1)
    for pa, pb in zip(a.trainable(), b.trainable(), strict=True):
        torch.testing.assert_close(pa, pb, atol=2e-7, rtol=2e-5)


def test_channel_failure_and_reve_unsupported_tail_do_not_silently_change_inputs():
    backbone, _, s = parts("REVE")
    bad = {"channels": ["C3", "MISSING", "C4"], "readout": [0, 1, 2]}
    with pytest.raises(ValueError, match="coordinate"):
        backbone(s["eeg"], bad)
    a = torch.randn(1, 3, 800); b = a.clone(); b[:, :, 740:] = 100
    torch.testing.assert_close(backbone(a, s["picks"]), backbone(b, s["picks"]), atol=0, rtol=0)


def test_epoch_logging_does_not_expand_original_selection_or_control_early_stop(tmp_path):
    events = [record_events(s, 3000, 169, "initial") for s in range(3001)]
    assert [i for i, e in enumerate(events) if e["initial_selection_candidate"]] == [1000, 3000]
    assert events[0]["epoch_record"] and events[3000]["partial_epoch"]
    assert sum(e["epoch_record"] for e in events) == 19
    rows = [{"learning_rate": .001, "episodes": step, "gain": gain} for step, gain in [(169, 1.), (1000, .1), (3000, .2)]]
    assert select_population(rows)["episodes"] == 3000  # Extra epoch has the best gain, but isn't a candidate.
    stop = EarlyStop(8, .0001)
    assert not stop.update(0, 1.)[1]
    for i in range(1, 9):
        _, done = stop.update(250 * i, 1. + i / 10)
        assert done == (i == 8)  # Rising loss permits early stop, never a forced flatness condition.
    _, readout, s = parts("LaBraM")
    class Scores(nn.Module):
        def forward(self, x, picks):
            return torch.stack((x[:, 0, 0], -x[:, 0, 0]), dim=1)
    people = evaluate(Scores(), [s])
    row = people[0]
    assert row["n"] == 2 and sum(map(sum, row["confusion"])) == 2
    assert summarize(people)["pooled"]["CE"] == row["CE_sum"] / 2
    path = tmp_path / "curve.jsonl"
    append_validation(path, {"step": 0}, people)
    append_validation(path, {"step": 169}, people)
    stored = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["step"] for r in stored] == [0, 169]
    assert all(set(r["metrics"]["pooled"]) == {"CE", "accuracy", "BA"} for r in stored)


def test_resume_restores_optimizer_rng_and_update_exactly(tmp_path):
    backbone, readout, s = parts("LaBraM")
    model = PopulationModel(backbone, readout)
    optimizer = torch.optim.AdamW(model.trainable(), lr=.001)
    rng = np.random.default_rng(17); generator = torch.Generator().manual_seed(17)
    update(model, s, torch.randperm(4, generator=generator)[:2], optimizer)
    checkpoint = tmp_path / "state.pt"
    save_training_state(checkpoint, model, optimizer, rng, generator, 1, {"phase": "initial"})
    expected_draw = rng.integers(100)
    indices = torch.randperm(4, generator=generator)[:2]
    update(model, s, indices, optimizer)
    expected = state_hash(model)
    step, metadata = restore_training_state(checkpoint, model, optimizer, rng, generator)
    assert step == 1 and metadata["phase"] == "initial" and rng.integers(100) == expected_draw
    torch.testing.assert_close(torch.randperm(4, generator=generator)[:2], indices)
    update(model, s, indices, optimizer)
    assert state_hash(model) == expected
