"""Audit saved R2 personal diagnostics locally, without reading model weights."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from subject_context.stage_a_common import ROOT,sha256
from stage_w_results import audit_truth,score

p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('truth',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
split=json.loads((ROOT/'configs/splits/stage_a_v1.json').read_text(encoding='utf-8'))['folds']
halves={(v['dataset'],v['subject']):v for v in json.loads((ROOT/'configs/splits/stage_b_halves_v1.json').read_text(encoding='utf-8'))['subjects']}
cfg=yaml.safe_load((ROOT/'configs/stage_c2.yaml').read_text(encoding='utf-8'))
runs=sorted((args.evidence/'stage_w_followup/r2/runs').glob('fold-*_seed-*'));assert len(runs)==25
count=0;flips=[];subject_rows=[]
for directory in runs:
    receipt=json.loads((directory/'run.json').read_text(encoding='utf-8'))
    assert receipt['status']=='complete' and receipt['population_unchanged'] and receipt['backbone_unchanged']
    assert receipt['git_commit']=='ea764cd2d2e6cc67faf295ef3086f8b52e87e30d'
    assert receipt['config_sha256']==sha256(ROOT/'configs/stage_w_followup.yaml')
    fold,seed=receipt['fold'],receipt['seed'];members={r:{tuple(k) for k in split[fold][r]} for r in ['train','validation','test']}
    validation=pd.read_csv(directory/'validation_search.csv',float_precision='round_trip')
    selected=json.loads((directory/'selection.json').read_text(encoding='utf-8'))
    for variant,g in validation.groupby('variant'):
        assert set(zip(g.dataset,g.subject))==members['validation']
        grid=set(itertools.product(cfg['learning_rates'][variant],cfg['weight_decays'],cfg['steps']))
        for _,s in g.groupby(['dataset','subject']):
            assert set(zip(s.learning_rate,s.weight_decay,s.steps))==grid and len(s)==len(grid)
        ranked=g.groupby(['learning_rate','weight_decay','steps'],sort=True).gain.agg(['median','mean']).reset_index()
        best=ranked.sort_values(['median','mean','steps','learning_rate','weight_decay'],ascending=[False,False,True,True,True],kind='stable').iloc[0]
        hp=selected[variant]
        np.testing.assert_array_equal([best.learning_rate,best.weight_decay,best.steps],[hp['learning_rate'],hp['weight_decay'],hp['steps']])
        np.testing.assert_allclose([best['median'],best['mean']],[hp['validation_median_gain'],hp['validation_mean_gain']],atol=1e-14,rtol=0)
    predictions=pd.read_csv(directory/'predictions.csv.gz');audit_truth(args.truth,predictions);count+=len(predictions)
    assert set(zip(predictions.dataset,predictions.subject))==members['test']
    for key,g in predictions.groupby(['dataset','subject']):
        path=args.truth/'processed'/key[0]/f'sub-{key[1]:03d}'/'trials.csv'
        assert sha256(path)==halves[key]['trial_table_sha256']
        expected=set(pd.read_csv(path).iloc[halves[key]['query_indices']].trial_id)
        for _,values in g.groupby(['variant','condition','donor'],dropna=False):
            assert len(values)==len(expected) and set(values.trial_id)==expected
        base=g[g.condition=='R2_query_batch'].set_index('trial_id').prediction
        for variant in ['film_offset','lora8']:
            zero=g[(g.variant==variant)&(g.condition=='zero_personal')].set_index('trial_id').prediction
            np.testing.assert_array_equal(base.loc[zero.index],zero)
    subjects=pd.read_csv(directory/'subjects.csv')
    for row in subjects.itertuples():
        data=predictions[(predictions.dataset==row.dataset)&(predictions.subject==row.subject)&(predictions.variant==row.variant)&(predictions.condition=='swap')]
        matrix={int(d):score(g.label,g.prediction) for d,g in data.groupby('donor')}
        assert set(matrix)=={s for d,s in members['test'] if d==row.dataset}
        donors=[int(v) for v in row.swap_donors.split()]
        assert len(donors)==10 and row.subject not in donors
        np.testing.assert_allclose([matrix[row.subject],np.mean([matrix[d] for d in donors])],[row.own_ba,row.swap_ba],atol=1e-12,rtol=0)
    subject_rows.append(subjects.assign(fold=fold,seed=seed))
    flips.append(pd.read_csv(directory/'baseline.csv').assign(fold=fold,seed=seed))
args.output.mkdir(parents=True,exist_ok=True)
pd.concat(subject_rows,ignore_index=True).to_csv(args.output/'R2_subject_seed_results.csv',index=False)
baselines=pd.concat(flips,ignore_index=True);baselines.to_csv(args.output/'R2_batch_path_checks.csv',index=False)
result={'status':'complete','runs':25,'prediction_rows':count,'validation_hyperparameter_choices':50,
        'query_trial_sets_match':True,'zero_personal_predictions_equal_population':True,
        'all_batch_vs_query_batch_trial_flips':int(baselines.trial_flips.sum()),
        'affected_baseline_checks':int((baselines.trial_flips>0).sum()),'baseline_checks':len(baselines),
        'G_remains_primary':True,'R2_sensitivity_only':True}
(args.output/'R2_independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
