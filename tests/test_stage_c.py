import copy
import numpy as np
import pytest
import torch
from subject_context.model import load_backbone, state_hash
from subject_context.stage_a_common import ROOT
from subject_context.stage_b import Readout
from subject_context.stage_b2 import FullAdapter
from subject_context.stage_c import ContextModel, config_c
from subject_context.stage_c_context import (TangentFeatures, band_covariances, context_window,
                                             derangement, t3a_predict)
from subject_context.stage_c_train import load_snapshot, score, select_config, snapshot, train

EMB, TAN = 600, 12


def parts(kind='linear', seed=5):
    torch.manual_seed(seed)
    backbone = load_backbone(ROOT / 'weights/cbramod.pth', 4)
    head = torch.nn.Linear(EMB, 2) if kind == 'linear' else torch.nn.Sequential(
        torch.nn.Linear(EMB, 128), torch.nn.GELU(), torch.nn.Linear(128, 2))
    readout = Readout({'head': head.state_dict(), 'feature_mean': np.zeros(EMB),
                       'feature_std': np.ones(EMB)}, kind)
    return backbone, readout


def subject(dataset, number, trials=6, segments=10, channels=3):
    g = torch.Generator().manual_seed(number)
    labels = torch.tensor([0, 1] * (trials // 2))
    return {'key': (dataset, number), 'eeg': torch.randn(trials, channels, 400, generator=g).half(),
            'labels': labels, 'picks': [0, 1, 2], 'query': torch.arange(trials // 2, trials),
            'emb': torch.randn(segments, EMB, generator=g), 'tan': torch.randn(segments, TAN, generator=g)}


@pytest.mark.parametrize('method', ['m_film', 'm_lora'])
def test_initial_model_is_exactly_b0_with_context_and_default_z(method):
    torch.set_num_threads(2)
    backbone, readout = parts()
    s = subject('A', 1)
    eeg = s['eeg'].float()
    expected = FullAdapter(backbone, readout, s['picks'], 'b0')(eeg)
    model = ContextModel(backbone, readout, method, EMB, TAN, config_c())
    for z in [model.z(), model.z(s['emb'], s['tan'])]:
        with torch.no_grad():  # scoring path: bit-identical
            torch.testing.assert_close(model(eeg, s['picks'], z), expected, atol=0, rtol=0)
        # Grad-mode attention kernels may differ at rounding level, as in B2.
        torch.testing.assert_close(model(eeg, s['picks'], z), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('method', ['m_film', 'm_lora'])
def test_training_moves_only_context_parts_and_keeps_backbone_frozen(method):
    backbone, readout = parts('mlp')
    frozen = state_hash(backbone), state_hash(readout)
    model = ContextModel(backbone, readout, method, EMB, TAN, config_c())
    cfg = copy.deepcopy(config_c())
    cfg['training'].update(episodes=[1, 2], query_batch=4, context_dropout=0.0)
    cfg['context'].update(min_seconds=8)
    subjects = [subject('A', 1), subject('A', 2), subject('B', 3, channels=4), subject('B', 4, channels=4)]
    before = snapshot(model)
    seen = []
    train(model, subjects, {'learning_rate': 1e-3, 'lambda': 1.0, 'margin': .5}, cfg, 0, seen.append)
    after = snapshot(model)
    assert seen == [1, 2]
    assert not torch.equal(before['generator.2.weight'], after['generator.2.weight'])
    assert (state_hash(backbone), state_hash(readout)) == frozen
    assert all(p.grad is None for p in backbone.parameters())
    if method == 'm_lora':
        bases = [n for n in after if 'parametrizations' in n]
        assert len(bases) == 12 * 2 * 2 * 2
        assert sum(after[n].numel() for n in bases) == 8 * 153600
    else:
        assert not any('core' in n for n in after)
    load_snapshot(model, before)
    assert all(torch.equal(before[n], v) for n, v in snapshot(model).items())


def test_lora_bases_follow_backbone_device():
    backbone, readout = parts()
    backbone.to('meta')
    model = ContextModel(backbone, readout, 'm_lora', EMB, TAN, config_c())
    bases = [p for n, p in model.named_parameters() if 'parametrizations' in n and p.requires_grad]
    assert bases and all(p.device.type == 'meta' for p in bases)
    assert model.mixture.w.device.type == 'meta'


def test_score_uses_second_half_and_all_trials():
    backbone, readout = parts()
    model = ContextModel(backbone, readout, 'm_film', EMB, TAN, config_c())
    s = subject('A', 1)
    r = score(model, s)
    assert len(r['predictions']) == 6 and 0 <= r['ba'] <= 1 and 0 <= r['ba_all'] <= 1
    donor = score(model, s, context=subject('A', 2))
    np.testing.assert_array_equal(donor['predictions'], r['predictions'])  # zero generator at init


def test_set_encoder_is_permutation_invariant():
    backbone, readout = parts()
    model = ContextModel(backbone, readout, 'm_film', EMB, TAN, config_c())
    torch.nn.init.normal_(model.generator[-1].weight)
    s = subject('A', 1)
    order = torch.randperm(10)
    torch.testing.assert_close(model.z(s['emb'], s['tan']), model.z(s['emb'][order], s['tan'][order]))


def test_selection_uses_median_then_mean_then_cheaper_settings():
    def rows(lr, lam, margin, episodes, gains):
        return [{'learning_rate': lr, 'lambda': lam, 'margin': margin, 'episodes': episodes, 'gain': g}
                for g in gains]
    table = (rows(1e-3, 1., .5, 3000, [0, .1, .2]) + rows(3e-4, 1., .5, 3000, [0, .1, .3]) +
             rows(3e-4, .1, .1, 1000, [-.5, .1, .3]) + rows(3e-4, .1, .1, 3000, [-.5, .1, .3]))
    assert select_config(table) == {'learning_rate': 3e-4, 'lambda': 1., 'margin': .5, 'episodes': 3000,
                                    'validation_median_gain': .1, 'validation_mean_gain': pytest.approx(.4/3)}
    tied = rows(1e-3, .1, .1, 1000, [.2]) + rows(3e-4, 1., .1, 1000, [.2]) + rows(3e-4, .1, .5, 1000, [.2])
    chosen = select_config(tied)
    assert (chosen['learning_rate'], chosen['lambda'], chosen['margin']) == (3e-4, .1, .5)


def test_context_window_and_derangement():
    rng = np.random.default_rng(0)
    for _ in range(200):
        start, n = context_window(15, rng, 8)
        assert 8 <= n <= 15 and 0 <= start and start + n <= 15
    with pytest.raises(AssertionError):
        context_window(7, rng, 8)
    people = [('A', i) for i in range(5)] + [('B', i) for i in range(2)]
    donors = derangement(people, rng)
    assert set(donors) == set(people)
    assert all(d[0] == p[0] and d != p for p, d in donors.items())


@pytest.mark.parametrize('kind', ['linear', 'mlp'])
def test_t3a_without_supports_is_cosine_to_classifier_weights(kind):
    _, readout = parts(kind)
    query = torch.randn(7, EMB)
    features = readout.normalize(query) if kind == 'linear' else readout.head[1](readout.head[0](readout.normalize(query)))
    classifier = readout.head if kind == 'linear' else readout.head[2]
    raw = classifier.weight
    labels = classifier(raw).argmax(1)
    prototypes = torch.stack([torch.nn.functional.normalize(raw, dim=1)[labels == k].sum(0) for k in [0, 1]], 1)
    expected = (features @ torch.nn.functional.normalize(prototypes, dim=0)).argmax(1).numpy()
    np.testing.assert_array_equal(t3a_predict(readout, torch.empty(0, EMB), query, 'all'), expected)
    assert len(t3a_predict(readout, torch.randn(9, EMB), query, 1)) == 7


def test_tangent_features_fit_on_training_rest_only():
    rng = np.random.default_rng(1)
    train_sets = [band_covariances(rng.standard_normal((6, 4, 800)), [[8, 13], [13, 30]]) for _ in range(3)]
    tangent = TangentFeatures().fit(train_sets)
    stacked = np.concatenate([tangent(b) for b in train_sets])
    assert stacked.shape == (18, 2 * 4 * 5 // 2)
    np.testing.assert_allclose(stacked.mean(0), 0, atol=1e-5)
    assert all(np.all(np.linalg.eigvalsh(r) > 0) for r in tangent.references)
