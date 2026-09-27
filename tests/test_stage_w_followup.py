"""R2 sensitivity retains the population prediction and updates only personal parameters."""
from types import SimpleNamespace
import pytest
import torch
from subject_context.model import state_hash
from subject_context.stage_c2 import PersonalModel
from subject_context.stage_m import load_population, population_state
from test_stage_c import subject
from test_stage_c2 import trained_base


def test_existing_R2_population_load_and_identity_film_personal_fit():
    base, backbone = trained_base('m_lora')
    wrapper = SimpleNamespace(base=base)
    state = population_state(wrapper)
    with torch.no_grad():
        for value in state.values(): value.add_(.001)
    load_population(wrapper,state)
    loaded = population_state(wrapper)
    assert loaded.keys() == state.keys()
    assert all(torch.equal(loaded[k],v) for k,v in state.items())
    s=subject('A',1)
    with torch.no_grad(): expected=base(s['eeg'].float(),s['picks'],base.z())
    frozen=state_hash(base),state_hash(backbone)
    with pytest.raises(AssertionError): PersonalModel(base,'film_offset')
    model=PersonalModel(base,'film_offset',allow_film_on_lora=True)
    assert sum(p.numel() for p in model.trainable())==4800
    torch.testing.assert_close(model(s['eeg'].float(),s['picks']),expected,atol=0,rtol=0)
    optimizer=torch.optim.AdamW(model.trainable(),lr=.1)
    loss=torch.nn.functional.cross_entropy(model(s['eeg'].float(),s['picks']),s['labels'])
    loss.backward();optimizer.step()
    assert model.delta.abs().sum()>0
    fitted=model.personal()
    with torch.no_grad(): fitted_prediction=model(s['eeg'].float(),s['picks'])
    model.reset()
    torch.testing.assert_close(model(s['eeg'].float(),s['picks']),expected,atol=0,rtol=0)
    model.load_personal(fitted)
    torch.testing.assert_close(model(s['eeg'].float(),s['picks']),fitted_prediction,atol=0,rtol=0)
    assert (state_hash(base),state_hash(backbone))==frozen
    assert all(p.grad is None for p in base.parameters())
