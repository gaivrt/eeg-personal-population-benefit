import numpy as np
import pytest
import torch
from subject_context.model import state_hash
from subject_context.stage_c2 import PersonalModel
from subject_context.stage_c3 import average_personal
from subject_context.stage_c4 import CalibModel, make_prior, select_hp, select_k_alpha
from test_stage_c import subject
from test_stage_c2 import trained_base
from test_stage_c3 import random_personal


def people(base, variant, count, scale):
    model = PersonalModel(base, variant, 8)
    return [random_personal(model, s, scale) for s in range(count)], model


@pytest.mark.parametrize('method,variant,scale', [('m_film', 'film_offset', .5), ('m_lora', 'lora8', .05)])
def test_prior_equals_averaged_personal_model_and_starts_calibration(method, variant, scale):
    base, backbone = trained_base(method)
    s = subject('A', 1)
    eeg = s['eeg'].float()
    params, single = people(base, variant, 3, scale)
    calib = CalibModel(base, variant)
    calib.reset()
    with torch.no_grad():
        b3 = base(eeg, s['picks'], base.z())
        calib.set_prior(None)
        torch.testing.assert_close(calib(eeg, s['picks']), b3, atol=0, rtol=0)  # P0 = B3
        calib.set_prior(make_prior(variant, params[:1], 1.0))
        single.load_personal(params[0])
        torch.testing.assert_close(calib(eeg, s['picks']), single(eeg, s['picks']), atol=1e-5, rtol=1e-5)
    if variant == 'lora8':  # k=3 padded into the rank-40 prior = mean of B·A, scaled by alpha
        calib.set_prior(make_prior(variant, params, .5))
        for i, m in enumerate(calib.priors):
            want = torch.stack([p[2 * i + 1] @ p[2 * i] for p in params]).mean(0) * .5
            torch.testing.assert_close(torch.bmm(m.b, m.a), want, atol=1e-7, rtol=1e-5)
    assert float(calib.penalty()) == 0.0


@pytest.mark.parametrize('method,variant', [('m_film', 'film_offset'), ('m_lora', 'lora8')])
def test_calibration_moves_only_personal_part(method, variant):
    base, backbone = trained_base(method)
    s = subject('A', 1)
    params, _ = people(base, variant, 2, .05)
    calib = CalibModel(base, variant)
    calib.set_prior(make_prior(variant, params, 1.0))
    frozen = state_hash(backbone), state_hash(base)
    prior_state = [p.clone() for m in getattr(calib, 'priors', []) for p in m.parameters()]
    calib.reset()
    opt = torch.optim.AdamW(calib.trainable(), lr=.01)
    loss = torch.nn.functional.cross_entropy(calib(s['eeg'].float(), s['picks']), s['labels']) + calib.penalty()
    loss.backward(); opt.step()
    assert float(calib.penalty()) > 0
    assert (state_hash(backbone), state_hash(base)) == frozen
    after = [p for m in getattr(calib, 'priors', []) for p in m.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(prior_state, after))
    assert all(p.grad is None for p in backbone.parameters())


def test_selection_rules():
    cands = {(3, 1.0): [.02, .01, .0], (5, .5): [.02, .01, .0], (3, .25): [.0, .0, .09]}  # last: lower median
    assert select_k_alpha(cands) == (5, .5)  # equal median/mean -> smaller alpha first
    rows = [{'learning_rate': lr, 'mu': mu, 'steps': st, 'gain': g}
            for lr, mu, st, g in [(1e-3, 0., 20, .02), (1e-4, 1., 20, .02), (1e-4, .1, 60, .03), (1e-4, .1, 60, -.5)]]
    assert select_hp(rows)['mu'] == 1.0  # (1e-4, .1, 60) median -.235 loses; tie at .02 -> smaller lr
