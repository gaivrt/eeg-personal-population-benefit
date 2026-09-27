"""Original diagnostic-1 fit/grid/swap procedure on the saved R2 population."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from subject_context.data import assert_split
from subject_context.model import load_backbone,state_hash
from subject_context.stage_a_common import ROOT,config,provenance,require_compute,save_json,scratch_root,sha256
from subject_context.stage_b import Readout,ba,select_candidate,temporal_halves
from subject_context.stage_c import ContextModel,config_c
from subject_context.stage_c_train import load_snapshot
from subject_context.stage_c2 import PersonalModel,config_c2,fit,predict
from subject_context.stage_m import load_population

parser=argparse.ArgumentParser()
parser.add_argument('--job',type=int,required=True,choices=range(25))
args=parser.parse_args()
require_compute()
assert torch.cuda.is_available() and '3090' in torch.cuda.get_device_name()
torch.set_num_threads(4)
root,c_cfg,cfg=scratch_root(),config_c(),config_c2()
fold,seed=args.job//5,c_cfg['seeds'][args.job%5]
name=f'fold-{fold}_seed-{seed}'
out=root/'stage_w_followup/r2/runs'/name
out.mkdir(parents=True,exist_ok=True)
assert not any(out.iterdir()),'Do not overwrite an existing attempt'
split=json.loads((ROOT/c_cfg['split_file']).read_text())['folds'][fold]
split={r:[tuple(k) for k in split[r]] for r in ['train','validation','test']}
assert_split(*split.values())
halves={(r['dataset'],r['subject']):r for r in json.loads((ROOT/c_cfg['halves_file']).read_text())['subjects']}
source=root/'stage_c/runs'/name
source_r2=root/'stage_m/runs'/name
assert json.loads((source_r2/'run.json').read_text())['status']=='complete'
kind=json.loads((source/'selection.json').read_text())['base_head']
head=root/'baselines_v2'/f'B0_{kind}'/name/'head.pt'
readout=Readout(torch.load(head,map_location='cpu',weights_only=False),kind).cuda()
backbone=load_backbone(ROOT/config()['model']['weights'],k=4).cuda().eval()
base=ContextModel(backbone,readout,'m_lora',5400,756,c_cfg).cuda()
load_snapshot(base,torch.load(source/'model-m_lora-main.pt',map_location='cuda',weights_only=True))
load_population(SimpleNamespace(base=base),torch.load(source_r2/'start-R2.pt',map_location='cuda',weights_only=True)['population'])
frozen=state_hash(base),state_hash(backbone)
common=json.loads((ROOT/'configs/stage_a_channels.json').read_text())['readout_common']
receipt=provenance()
receipt.update(fold=fold,seed=seed,start_label='R2',population_family='m_lora',primary_reference='G',
               sensitivity_only=True,config_sha256=sha256(ROOT/'configs/stage_w_followup.yaml'),
               c2_config_sha256=sha256(ROOT/'configs/stage_c2.yaml'),
               source_R2_sha256=sha256(source_r2/'start-R2.pt'),source_B3_sha256=sha256(source/'model-m_lora-main.pt'))
start=time.monotonic()

def load_subject(key):
    dataset,subject=key
    directory=root/'processed'/dataset/f'sub-{subject:03d}'
    table=pd.read_csv(directory/'trials.csv')
    assert sha256(directory/'trials.csv')==halves[key]['trial_table_sha256']
    support,query,counts=temporal_halves(table)
    assert support.tolist()==halves[key]['fit_indices'] and query.tolist()==halves[key]['query_indices']
    labels=np.load(directory/'labels.npy');np.testing.assert_array_equal(labels,table.label)
    channels=json.loads((directory/'metadata.json').read_text())['channels']
    return {'key':key,'eeg':torch.as_tensor(np.load(directory/'tasks.npy'),device='cuda'),
            'picks':[channels.index(c) for c in common],'labels':torch.as_tensor(labels,device='cuda'),
            'fit':torch.as_tensor(support,device='cuda'),'query':torch.as_tensor(query,device='cuda'),
            'query_labels':labels[query],'trial_ids':table.trial_id.tolist(),'counts':counts}

groups={r:[load_subject(k) for k in split[r]] for r in ['validation','test']}
old=pd.read_csv(source_r2/'results.csv').query("method=='R2' and shots==0").set_index(['dataset','subject']).ba
zero=PersonalModel(base,'mix_offset')
baseline,base_rows,predictions={},[],[]

def record(s,variant,condition,pred,donor=None):
    predictions.extend({'variant':variant,'condition':condition,'dataset':s['key'][0],'subject':s['key'][1],
                        'donor':donor,'trial_id':s['trial_ids'][i],'label':int(s['query_labels'][j]),
                        'prediction':int(pred[j])} for j,i in enumerate(s['query'].cpu().tolist()))

for role in ['validation','test']:
    for s in groups[role]:
        all_pred=predict(zero,s,torch.arange(len(s['labels']),device='cuda'))[s['query'].cpu().numpy()]
        query_pred=predict(zero,s,s['query'])
        all_ba,query_ba=ba(s['query_labels'],all_pred),ba(s['query_labels'],query_pred)
        baseline[s['key']]=all_ba
        if role=='test':
            np.testing.assert_allclose(query_ba,old.loc[s['key']],atol=1e-12,rtol=0)
            record(s,'population','R2_all_batch',all_pred)
            record(s,'population','R2_query_batch',query_pred)
        base_rows.append({'role':role,'dataset':s['key'][0],'subject':s['key'][1],
                          'all_batch_ba':all_ba,'query_batch_ba':query_ba,
                          'trial_flips':int(np.count_nonzero(all_pred!=query_pred))})
del zero
pd.DataFrame(base_rows).to_csv(out/'baseline.csv',index=False)
rng=np.random.default_rng(seed)
draws={}
for s in groups['test']:
    others=[t['key'] for t in groups['test'] if t['key'][0]==s['key'][0] and t['key']!=s['key']]
    draws[s['key']]=[others[i] for i in rng.integers(len(others),size=cfg['swap_draws'])]
rows,search,resources,cross,selection=[],[],[],[],{}
for variant in ['film_offset','lora8']:
    model=PersonalModel(base,variant,cfg['lora_rank'],allow_film_on_lora=True)
    model.reset()
    for s in groups['test']:
        # Personal zero must equal the matched population through the same query path.
        zero_pred=predict(model,s,s['query'])
        np.testing.assert_allclose(ba(s['query_labels'],zero_pred),old.loc[s['key']],atol=1e-12,rtol=0)
        record(s,variant,'zero_personal',zero_pred)
    candidates=[]
    for s in groups['validation']:
        for lr,wd in itertools.product(cfg['learning_rates'][variant],cfg['weight_decays']):
            result,_,resource=fit(model,s,lr,wd,cfg['steps'],seed+s['key'][1])
            resources.append({'phase':'validation','variant':variant,'dataset':s['key'][0],'subject':s['key'][1],**resource})
            for k,r in result.items():
                row={'variant':variant,'dataset':s['key'][0],'subject':s['key'][1],'learning_rate':lr,'weight_decay':wd,
                     'steps':k,'b3_ba':baseline[s['key']],'ba':r['ba'],'gain':r['ba']-baseline[s['key']]}
                candidates.append(row);search.append(row)
    hp=selection[variant]=select_candidate(candidates)
    own,params,matrix={},{},{}
    for s in groups['test']:
        result,params[s['key']],resource=fit(model,s,hp['learning_rate'],hp['weight_decay'],[hp['steps']],seed+s['key'][1])
        own[s['key']]=result[hp['steps']]['ba']
        resources.append({'phase':'test','variant':variant,'dataset':s['key'][0],'subject':s['key'][1],**resource})
    torch.save({f'{d}:{s}':[v.cpu() for v in values] for (d,s),values in params.items()},out/f'personal-{variant}.pt')
    for s in groups['test']:
        for donor in [t for t in groups['test'] if t['key'][0]==s['key'][0]]:
            model.load_personal(params[donor['key']])
            pred=predict(model,s,s['query'])
            matrix[(s['key'],donor['key'])]=ba(s['query_labels'],pred)
            record(s,variant,'swap',pred,donor['key'][1])
            cross.append({'variant':variant,'dataset':s['key'][0],'subject':s['key'][1],
                          'donor':donor['key'][1],'ba':matrix[(s['key'],donor['key'])]})
        assert abs(matrix[(s['key'],s['key'])]-own[s['key']])<1e-12
        swap=float(np.mean([matrix[(s['key'],d)] for d in draws[s['key']]]))
        rows.append({'variant':variant,'dataset':s['key'][0],'subject':s['key'][1],'fold':fold,'seed':seed,
                     'b3_ba':baseline[s['key']],'own_ba':own[s['key']],'swap_ba':swap,
                     'upper_bound':own[s['key']]-swap,'own_gain':own[s['key']]-baseline[s['key']],
                     'swap_gain':swap-baseline[s['key']],'swap_donors':' '.join(str(d[1]) for d in draws[s['key']]),**s['counts']})
    for filename,data in [('subjects.csv',rows),('swap_matrix.csv',cross),('validation_search.csv',search),('fit_resources.csv',resources)]:
        pd.DataFrame(data).to_csv(out/filename,index=False)
    print(json.dumps({'variant':variant,'selected':hp,'seconds':time.monotonic()-start}),flush=True)
    del model
save_json(out/'selection.json',selection)
pd.DataFrame(predictions).to_csv(out/'predictions.csv.gz',index=False)
assert (state_hash(base),state_hash(backbone))==frozen
receipt.update(status='complete',seconds=time.monotonic()-start,population_unchanged=True,backbone_unchanged=True,
               R2_query_matches_stage_M=True,peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in resources))
save_json(out/'run.json',receipt)
