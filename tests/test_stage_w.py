import copy
import numpy as np
import pytest
import torch
from subject_context.model import state_hash
from subject_context.stage_c import ContextModel, config_c
from subject_context.stage_c_train import snapshot
from subject_context.stage_w import EarlyStop, config_w, continuation, subject_losses
from test_stage_c import parts, subject


def test_loss_checkpoint_keeps_global_minimum_even_below_patience_delta():
    s = EarlyStop(2, .01)
    assert s.update(0, 1.0) == (True, False)
    assert s.update(250, .995) == (True, False)
    assert s.update(500, .993) == (True, True)
    assert s.best_step == 500 and s.best == .993
    with pytest.raises(ValueError):
        s.update(750, float("nan"))


def test_validation_uses_subject_equal_weight_and_only_query_labels():
    class Uniform:
        def z(self):
            return None
        def __call__(self, x, picks, z):
            return torch.zeros(len(x), 2)
    people = [subject("A", 1, trials=6), subject("B", 2, trials=12)]
    people[0]["labels"][:3] = 999  # inaccessible for the validation calculation
    rows = subject_losses(Uniform(), people, True)
    assert len(rows) == 2
    np.testing.assert_allclose([r["loss"] for r in rows], np.log(2), rtol=1e-6)
    for row, person in zip(rows, people):
        labels = person['labels'][person['query']].numpy()
        assert row['accuracy'] == float((labels == 0).mean())
        assert row['ba'] == .5


@pytest.mark.parametrize("method", ["m_film", "m_lora"])
def test_continuation_is_bounded_and_keeps_pretrained_weights(method):
    torch.set_num_threads(2)
    backbone, head = parts()
    before = state_hash(backbone), state_hash(head)
    cfg = copy.deepcopy(config_c())
    cfg["context"]["min_seconds"] = 8
    cfg["training"]["query_batch"] = 4
    model = ContextModel(backbone, head, method, 600, 12, cfg)
    w = config_w()
    w["population"].update(check_every_steps=1, early_stop_patience_checks=1,
                            early_stop_min_delta=100., plateau_window_checks=2)
    train = [subject("A", 1), subject("A", 2)]
    validation = [subject("A", 3)]
    saved = []
    _, _, steps, status = continuation(model, train, validation,
        {"episodes": 2, "learning_rate": .001, "lambda": .1, "margin": .1}, cfg, w, 11,
        lambda state, step: saved.append((state, step)))
    assert status["additional_steps_cap"] == 6
    assert len(steps) == status["additional_steps_run"] == 1
    assert status["early_stopped"] and status["test_used_for_selection"] is False
    assert saved[0][1] == 0
    assert (state_hash(backbone), state_hash(head)) == before
    final = snapshot(model)
    assert all(torch.equal(final[k], v) for k, v in saved[-1][0].items())
