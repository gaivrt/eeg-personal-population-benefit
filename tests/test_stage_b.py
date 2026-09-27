import json
import numpy as np
import pandas as pd
import pytest
import torch
from subject_context.model import load_backbone, state_hash
from subject_context.stage_a_common import ROOT
from subject_context.stage_b import temporal_halves, select_candidate, DirectAdapter, Readout


def test_half_uses_chronology_not_labels_or_input_row_order():
    t=pd.DataFrame({"trial_id":list("abcdefgh"),"label":[0,0,0,1,1,1,0,1],
                    "protocol_order":[0]*4+[1]*4,"start_seconds":[0,6,12,18]*2,
                    "stop_seconds":[4,10,16,22]*2})
    t=t.iloc[[5,0,7,1,3,2,6,4]].reset_index(drop=True)
    fit,query,c=temporal_halves(t)
    assert t.iloc[fit].trial_id.tolist()==list("abcd")
    assert t.iloc[query].trial_id.tolist()==list("efgh")
    assert not c["fit_balanced"] and not c["query_balanced"]
    assert not set(fit)&set(query)
    bad=t.copy();bad.loc[bad.trial_id=='e','protocol_order']=0;bad.loc[bad.trial_id=='e','start_seconds']=20
    with pytest.raises(AssertionError,match="temporal overlap"):temporal_halves(bad)


def test_validation_selection_tie_prefers_smaller_budget():
    rows=[{"learning_rate":lr,"weight_decay":wd,"steps":s,"gain":g}
          for lr in [.01,.1] for wd in [0,.01] for s in [20,60] for g in [.02,.04]]
    best=select_candidate(rows)
    assert (best["learning_rate"],best["weight_decay"],best["steps"])==(.01,0,20)


def test_direct_film_zero_and_only_requested_parameters_update():
    torch.set_num_threads(2);torch.manual_seed(7)
    backbone=load_backbone(ROOT/'weights/cbramod.pth',4)
    p8=torch.randn(3,3,2,200);picks=[0,1,2]
    head=torch.nn.Linear(600,2)
    readout=Readout({"head":head.state_dict(),"feature_mean":np.zeros(600),"feature_std":np.ones(600)},'linear')
    b0=backbone.suffix(p8)[:,picks].mean(2).flatten(1)
    p10=p8
    for layer in backbone.model.encoder.layers[-4:-2]:p10=layer(p10)
    before=(state_hash(backbone),state_hash(readout))
    for variant in ['film2','film4','beta4','head']:
        m=DirectAdapter(backbone,readout,picks,variant)
        x=b0 if variant=='head' else (p10 if variant=='film2' else p8)
        assert torch.equal(m(x),readout(b0))
        opt=torch.optim.AdamW(m.trainable(),lr=.01)
        opt.zero_grad();torch.nn.functional.cross_entropy(m(x),torch.tensor([0,1,0])).backward();opt.step()
        assert (state_hash(backbone),state_hash(readout))==before
        assert all(p.grad is None for p in backbone.parameters())
        if variant!='head':
            assert m.raw.abs().max()>0
            assert (.1*torch.tanh(m.raw)).abs().max()<=.1
            if variant=='beta4':assert m.raw.shape==(4,1,200)


def test_frozen_halves_cover_each_subject_once():
    doc=json.loads((ROOT/'configs/splits/stage_b_halves_v1.json').read_text())
    assert len(doc['subjects'])==235
    for r in doc['subjects']:
        a,b=r['fit_indices'],r['query_indices']
        assert not set(a)&set(b)
        assert sorted(a+b)==list(range(len(a)+len(b)))
        assert len(b)-len(a) in [0,1]
