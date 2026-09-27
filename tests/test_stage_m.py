import copy
import numpy as np
import pytest
import torch
from subject_context.model import state_hash
from subject_context.stage_c4 import CalibModel
from subject_context.stage_m import (adapt, config_m, continue_train, inner_loop, load_population, log_lr_grad,
                                     meta_train, pilot_grid, population, population_state, select_inner, select_meta)
from test_stage_c import subject
from test_stage_c2 import trained_base


def person(dataset, number, trials=12):
    s = subject(dataset, number, trials=trials)
    s['fit'], s['query'] = torch.arange(trials // 2), torch.arange(trials // 2, trials)
    s['query_labels'] = s['labels'][s['query']].numpy()
    return s


def frozen_parts(model):
    base = model.base
    return state_hash(base.encoder), state_hash(base.generator), base.default_z.detach().clone(), \
        {n: p.detach().clone() for n, p in base.core.named_parameters() if 'original' in n or 'parametrizations' not in n}


def small_cfg(steps=(1, 2)):
    cfg = copy.deepcopy(config_m())
    cfg['outer'].update(steps=list(steps), episode_shots=[2, 4])
    return cfg


def test_population_is_b3_bases_and_head_and_start_equals_b3():
    base, _ = trained_base('m_lora')
    model = CalibModel(base, 'lora8')
    model.reset()
    s = person('A', 1)
    with torch.no_grad():
        torch.testing.assert_close(model(s['eeg'].float(), s['picks']), base(s['eeg'].float(), s['picks'], base.z()),
                                   atol=0, rtol=0)
    names = set(population(model))
    assert names and all('head' in n or n.endswith(('.a', '.b')) and 'parametrizations' in n for n in names)
    assert sum(p.numel() for p in population(model).values()) == \
        sum(p.numel() for n, p in base.named_parameters() if 'parametrizations' in n and 'original' not in n) + \
        sum(p.numel() for p in base.readout.head.parameters())
    state = population_state(model)
    with torch.no_grad():
        for p in population(model).values():
            p.add_(1.)
    load_population(model, state)
    assert all(torch.equal(population(model)[n], v) for n, v in state.items())


def test_adapt_is_a_prefix_path_and_moves_only_personal():
    base, backbone = trained_base('m_lora')
    model = CalibModel(base, 'lora8')
    s = person('A', 1)
    before = population_state(model), state_hash(backbone)
    long = adapt(model, s, 4, [1, 3], 0.5, seed=7)
    short = adapt(model, s, 4, [1], 0.5, seed=7)
    assert long[1] == short[1] and set(long) == {1, 3}
    assert float(sum(p.abs().sum() for p in model.trainable()[1::2])) > 0  # personal B moved
    after = population_state(model), state_hash(backbone)
    assert all(torch.equal(before[0][n], after[0][n]) for n in before[0]) and before[1] == after[1]
    assert adapt(model, s, 0, [5], 0.5, seed=7)[0] == adapt(model, s, 4, [1], 0.0, seed=7)[1]  # lr 0 = start


def test_meta_train_updates_population_only_and_resets_personal():
    base, backbone = trained_base('m_lora')
    model = CalibModel(base, 'lora8')
    subjects = [person('A', 1), person('A', 2), person('B', 3)]
    frozen, backbone_hash = frozen_parts(model), state_hash(backbone)
    start = population_state(model)
    seen = []
    log = meta_train(model, subjects, 2, 0.5, small_cfg(), seed=3, on_checkpoint=seen.append)
    assert seen == [1, 2] and len(log) == 2
    now = population_state(model)
    assert any(not torch.equal(start[n], now[n]) for n in start)
    after = frozen_parts(model)
    assert frozen[:2] == after[:2] and torch.equal(frozen[2], after[2])
    assert all(torch.equal(frozen[3][n], after[3][n]) for n in frozen[3]) and state_hash(backbone) == backbone_hash
    assert all(not p.requires_grad for p in population(model).values())
    assert all(float(p.abs().sum()) == 0 for p in model.trainable()[1::2])  # returned at personal zero


def test_meta_train_m2_moves_layer_rates():
    base, _ = trained_base('m_lora')
    model = CalibModel(base, 'lora8')
    log_lr = torch.full((12,), float(np.log(0.5)), requires_grad=True)
    meta_train(model, [person('A', 1), person('A', 2)], 2, 0.5, small_cfg(), seed=3, log_lr=log_lr)
    assert not torch.allclose(log_lr.detach(), torch.full((12,), float(np.log(0.5))))


def test_first_order_layer_rate_gradient_matches_finite_difference_for_one_step():
    base, _ = trained_base('m_lora')
    model = CalibModel(base, 'lora8')
    s, q = person('A', 1), person('A', 2)
    eeg, y = s['eeg'].float(), s['labels']
    rates = torch.full((12,), 2.0)

    def query_loss(r, backward=False):
        torch.manual_seed(0); model.reset()
        with torch.no_grad():  # a non-zero personal start so every factor has a gradient
            for p in model.trainable()[1::2]:
                p.normal_(0, 0.05)
        total = inner_loop(model, eeg, s['picks'], y, 1, r)
        loss = torch.nn.functional.cross_entropy(model(q['eeg'].float(), q['picks']), q['labels'])
        if backward:
            loss.backward()
            return log_lr_grad(model.trainable(), total, r)
        return float(loss)

    grad = query_loss(rates, backward=True)
    layer = int(grad.abs().argmax())
    eps = 0.02
    up, down = rates.clone(), rates.clone()
    up[layer] *= np.exp(eps); down[layer] *= np.exp(-eps)
    numeric = (query_loss(up) - query_loss(down)) / (2 * eps)
    assert abs(numeric - float(grad[layer])) <= 0.05 * abs(float(grad[layer])) + 1e-5


def test_continue_train_moves_population_and_keeps_personal_zero():
    base, backbone = trained_base('m_lora')
    model = CalibModel(base, 'lora8')
    start, backbone_hash = population_state(model), state_hash(backbone)
    seen = []
    continue_train(model, [person('A', 1), person('B', 2)], [1, 2], small_cfg(), seed=3, on_checkpoint=seen.append)
    now = population_state(model)
    assert seen == [1, 2] and any(not torch.equal(start[n], now[n]) for n in start)
    assert state_hash(backbone) == backbone_hash
    assert all(float(p.abs().sum()) == 0 for p in model.trainable()[1::2])
    assert all(p.requires_grad for p in model.trainable())


def test_selection_rules():
    rows = [{'k': k, 'learning_rate': lr, 'gain': g} for k, lr, g in
            [(10, .1, .02), (5, .3, .02), (20, .1, .01), (5, .1, .02)]]
    assert select_inner(rows) == {'k': 5, 'learning_rate': .1, 'validation_median_gain': .02,
                                  'validation_mean_gain': .02}
    meta = [{'k': k, 'steps': st, 'gain': g} for k, st, g in [(10, 1000, .03), (10, 2000, .03), (20, 1000, .01)]]
    assert select_meta(meta)['steps'] == 1000
    scan = [.001, .003, .01, .03]
    pilot = [{'learning_rate': lr, 'gain': g} for lr, g in [(.001, 0), (.003, .01), (.01, .02), (.03, -.1)]]
    assert pilot_grid(pilot, scan) == [.003, .01, .03]
    assert pilot_grid([{'learning_rate': .001, 'gain': .1}, {'learning_rate': .003, 'gain': 0}], scan) == [.001, .003]


def test_diverged_inner_loop_scores_chance_at_unreached_steps():
    from subject_context import stage_m
    base, _ = trained_base('m_lora')
    model = CalibModel(base, 'lora8')
    before = len(stage_m.divergences)
    result = adapt(model, person('A', 1), 4, [1, 5], 1e12, seed=7)
    assert len(stage_m.divergences) == before + 1 and result[5] == stage_m.CHANCE
