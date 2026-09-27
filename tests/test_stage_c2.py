import numpy as np
import pytest
import torch
from subject_context.model import state_hash
from subject_context.stage_c import ContextModel, config_c
from subject_context.stage_c2 import PersonalModel
from test_stage_c import EMB, TAN, parts, subject

VARIANTS = [('m_film', 'film_offset', 4800), ('m_lora', 'mix_offset', 8), ('m_lora', 'lora8', 153600)]


def trained_base(method):
    backbone, readout = parts()
    base = ContextModel(backbone, readout, method, EMB, TAN, config_c())
    torch.manual_seed(3)
    with torch.no_grad():  # a non-trivial "Stage C" model whose default z is not B0
        for p in base.trainable():
            p.add_(0.02 * torch.randn_like(p))
    return base, backbone


@pytest.mark.parametrize('method,variant,count', VARIANTS)
def test_zero_personal_equals_b3_and_training_moves_only_personal(method, variant, count):
    base, backbone = trained_base(method)
    s = subject('A', 1)
    eeg = s['eeg'].float()
    with torch.no_grad():
        b3 = base(eeg, s['picks'], base.z())
    frozen = state_hash(base), state_hash(backbone)
    model = PersonalModel(base, variant)
    with torch.no_grad():
        torch.testing.assert_close(model(eeg, s['picks']), b3, atol=0, rtol=0)
    assert sum(p.numel() for p in model.trainable()) == count
    before = model.personal()
    opt = torch.optim.AdamW(model.trainable(), lr=.01)
    torch.nn.functional.cross_entropy(model(eeg, s['picks']), s['labels']).backward()
    opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.personal()))
    assert (state_hash(base), state_hash(backbone)) == frozen
    assert all(p.grad is None for p in backbone.parameters())
    assert all(p.grad is None for p in base.parameters())


@pytest.mark.parametrize('method,variant,count', VARIANTS)
def test_swapping_parameters_reproduces_owner_predictions(method, variant, count):
    base, _ = trained_base(method)
    model = PersonalModel(base, variant)
    s = subject('A', 1)
    eeg = s['eeg'].float()
    torch.manual_seed(1)
    with torch.no_grad():
        for p in model.trainable():
            p.add_(torch.randn_like(p) * (0.5 if variant != 'lora8' else 0.05))
    mine = model.personal()
    with torch.no_grad():
        expected = model(eeg, s['picks'])
    model.reset()
    with torch.no_grad():
        assert not torch.equal(model(eeg, s['picks']), expected)
    model.load_personal(mine)
    with torch.no_grad():
        torch.testing.assert_close(model(eeg, s['picks']), expected, atol=0, rtol=0)
