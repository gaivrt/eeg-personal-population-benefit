from itertools import product
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from subject_context.data import Window, assert_split, assert_episode, make_splits, other_subject
from subject_context.features import TangentSpace, band_covariances
from subject_context.model import Adapter, load_backbone, state_hash, ROOT
from subject_context.statistics import holm, paired_statistics, equivalent_trials


@pytest.fixture(scope="module")
def backbone():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    return load_backbone(ROOT / "weights/cbramod.pth", 2)


def test_subject_disjoint_and_complete():
    people = list(product(["A", "B"], range(10)))
    for seed in [11, 22, 33, 44, 55]:
        splits = make_splits(people, 5, .1, seed)
        tests = []
        for train, val, test in splits:
            assert_split(train, val, test)
            assert set(train + val + test) == set(people)
            tests.extend(test)
        assert sorted(tests) == sorted(people)
    with pytest.raises(AssertionError):
        assert_split([("A", 1)], [("A", 2)], [("A", 1)])


def test_context_temporal_and_trial_leakage():
    c = Window("A", 1, 1, 0, 30, "rest")
    q = Window("A", 1, 1, 32, 36, "trial")
    assert_episode([c], [q])
    for bad in [Window("A", 1, 1, 29, 33, "trial"), Window("A", 1, 1, 32, 36, "rest"),
                Window("A", 2, 1, 32, 36, "trial")]:
        with pytest.raises(AssertionError):
            assert_episode([c], [bad])
    assert_episode([c], [Window("A", 1, 2, 86400, 86404, "s2trial")], cross_session=True)
    with pytest.raises(AssertionError):
        assert_episode([c], [Window("A", 1, 2, 0, 4, "s2trial")], cross_session=True)


def test_shuffle_same_dataset_different_person_and_split():
    rng = np.random.default_rng(1)
    pool = [("A", 1), ("A", 2), ("B", 1), ("B", 2)]
    for _ in range(100):
        assert other_subject(("A", 1), pool, rng) == ("A", 2)
    with pytest.raises(ValueError):
        other_subject(("A", 1), [("A", 1), ("B", 2)], rng)


def test_zero_film_exact_identity_and_cache(backbone):
    torch.manual_seed(13)
    model = Adapter(212)
    x = torch.randn(2, 3, 800)
    context = torch.randn(15, 212)
    with torch.no_grad():
        prefix = backbone.prefix(x)
        baseline = model.head(backbone.suffix(prefix).mean((1, 2)))
        assert torch.equal(model(backbone, prefix, context), baseline)
        assert torch.equal(model(backbone, prefix, None), baseline)
        assert torch.equal(backbone(x), backbone.model(x.reshape(2, 3, 4, 200)))
        assert torch.equal(model.film(context), torch.zeros(2, 2, 200))
    assert not backbone.training and not backbone.model.training
    # Also enforce exact equality when FiLM introduces gradients into frozen layers.
    baseline = model.head(backbone.suffix(prefix).mean((1, 2)))
    assert torch.equal(model(backbone, prefix, context), baseline)
    assert torch.equal(model(backbone, prefix, None), baseline)


def test_set_permutation_invariance():
    torch.manual_seed(44)
    model = Adapter(12)
    torch.nn.init.normal_(model.generator[-1].weight, std=.01)
    context = torch.randn(30, 12)
    torch.testing.assert_close(model.film(context), model.film(context[torch.randperm(30)]), atol=1e-7, rtol=1e-6)


def test_small_batch_overfits_with_frozen_backbone(backbone):
    torch.manual_seed(7)
    original = state_hash(backbone)
    x = torch.randn(4, 3, 400)
    x[:2, 0] *= .1
    x[2:, 2] *= 3
    labels = torch.tensor([0, 0, 1, 1])
    prefix = backbone.prefix(x)
    context = torch.randn(15, 12)
    model = Adapter(12)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        logits = model(backbone, prefix, context)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(backbone, prefix, context)
        loss = F.cross_entropy(logits, labels)
    assert torch.equal(logits.argmax(-1), labels)
    assert loss.item() < .03, loss.item()
    assert state_hash(backbone) == original
    assert all(p.grad is None for p in backbone.parameters())
    assert model.generator[-1].weight.abs().max().item() > 0


def test_statistics_hand_calculation_and_ties():
    # Four positive unequal ranks: only all+ and all- are equally extreme;
    # exact two-sided p = 2 / 2**4 = 0.125, min rank sum = 0.
    result = paired_statistics([.6, .7, .8, .9], [.5, .5, .5, .5])
    assert result["statistic"] == 0 and result["p_raw"] == .125
    assert result["median_difference"] == pytest.approx(.25)
    np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
    assert paired_statistics([.5, .5], [.5, .5])["p_raw"] == 1
    # Tied ranks: enumerate all 16 sign flips independently.
    observed = [1, 1, 2, 2]
    ranks = np.array([1.5, 1.5, 3.5, 3.5])
    extreme = sum(min(sum(ranks[np.array(signs) > 0]), sum(ranks[np.array(signs) < 0])) <= 0
                  for signs in product([-1, 1], repeat=4)) / 16
    tied = paired_statistics(np.array(observed) / 10 + .5, [.5] * 4)
    assert tied["p_raw"] == extreme


def test_equivalent_trials_no_extrapolation():
    assert equivalent_trials(.65, [10, 20, 40], [.5, .6, .7]) == pytest.approx(30)
    assert equivalent_trials(.8, [10, 20], [.5, .6]) is None
    assert equivalent_trials(.5, [10, 20], [.6, .5]) is None


def test_covariance_spd_and_train_reference():
    rng = np.random.default_rng(0)
    cov = band_covariances(rng.normal(size=(5, 3, 400)))
    assert np.linalg.eigvalsh(cov).min() > 0
    tangent = TangentSpace().fit(cov[:3])
    before = tangent.whitener.copy()
    result = tangent.transform(cov[3:])
    np.testing.assert_array_equal(tangent.whitener, before)
    assert result.shape == (2, 12) and np.isfinite(result).all()
