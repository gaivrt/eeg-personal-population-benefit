import numpy as np
import torch
from subject_context.stage_c2 import PersonalModel
from subject_context.stage_c3 import average_personal, nearest, random_sets
from test_stage_c2 import trained_base
from test_stage_c import subject


def random_personal(model, seed, scale):
    torch.manual_seed(seed)
    model.reset()
    with torch.no_grad():
        for p in model.trainable():
            p.add_(torch.randn_like(p) * scale)
    return model.personal()


def test_lora_average_is_mean_weight_update_as_rank_24():
    base, _ = trained_base('m_lora')
    single = PersonalModel(base, 'lora8', 8)
    people = [random_personal(single, s, .05) for s in range(3)]
    updates = []
    for p in people:
        single.load_personal(p)
        updates.append([m.b @ m.a for m in single.lora])  # per-module, per-block B·A
    merged = PersonalModel(base, 'lora8', 24)
    merged.load_personal(average_personal('lora8', people))
    for i, m in enumerate(merged.lora):
        torch.testing.assert_close(m.b @ m.a, torch.stack([u[i] for u in updates]).mean(0), atol=1e-7, rtol=1e-5)
    single.load_personal(people[0])
    s = subject('A', 1)
    with torch.no_grad():
        expected = single(s['eeg'].float(), s['picks'])
    one = PersonalModel(base, 'lora8', 8)
    one.load_personal(average_personal('lora8', people[:1]))
    with torch.no_grad():
        torch.testing.assert_close(one(s['eeg'].float(), s['picks']), expected, atol=0, rtol=0)


def test_film_average_and_neighbour_helpers():
    base, _ = trained_base('m_film')
    model = PersonalModel(base, 'film_offset')
    people = [random_personal(model, s, .5) for s in range(3)]
    torch.testing.assert_close(average_personal('film_offset', people)[0], torch.stack([p[0] for p in people]).mean(0))
    features = {1: np.array([0., 0]), 2: np.array([1., 0]), 3: np.array([0., 1]), 4: np.array([5., 5])}
    assert nearest(features, 1, [2, 3, 4], 1) == [2]  # tie between 2 and 3 goes to the smaller id
    assert nearest(features, 1, [2, 3, 4], 3) == [2, 3, 4]
    sets = random_sets(np.random.default_rng(0), [2, 3, 4, 5], 3, 10)
    assert len(sets) == 10 and all(len(set(s)) == 3 and 1 not in s for s in sets)
    people = [('Cho2017', 9), ('Cho2017', 10), ('Cho2017', 11), ('Cho2017', 12)]
    tuples = random_sets(np.random.default_rng(0), people, 3, 5)
    assert all(isinstance(p, tuple) and isinstance(p[1], int) and p in people for s in tuples for p in s)
    tied = {p: np.zeros(2) for p in people}
    assert nearest(tied, people[3], people[:3], 2) == [('Cho2017', 9), ('Cho2017', 10)]  # numeric, not "10" < "9"
