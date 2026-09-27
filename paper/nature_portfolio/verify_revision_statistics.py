"""Independently verify reporting arithmetic and seed-first aggregation."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
R=P.parent.parent
checks=[]

def seed_check(raw_path,summary_path,keys,cols):
    raw=pd.read_csv(R/raw_path);summary=pd.read_csv(R/summary_path)
    assert not raw.duplicated(keys+['seed']).any()
    assert (raw.groupby(keys).seed.nunique()==5).all()
    means=raw.groupby(keys)[cols].mean().sort_index()
    saved=summary.set_index(keys)[cols].sort_index()
    assert means.index.equals(saved.index)
    np.testing.assert_allclose(means,saved,atol=1e-10,rtol=0)
    checks.append({'raw':raw_path,'summary':summary_path,'subject_conditions':len(saved),'seeds_per_subject':5,'verified':True})

for model in ['REVE','LaBraM']:
    base=f'reports/stage_x/{model}/report/'
    seed_check(base+'subject_seed_details.csv',base+'subject_means.csv',['dataset','subject'],['G_BA','own_ba','swap_ba','own_gain','upper_bound','B0_BA'])
    seed_check(base+'few_shot_seed_details.csv',base+'few_shot_subject_means.csv',['dataset','subject','shots'],['ba','G_BA','gain','full_gain'])
seed_check('reports/stage_c/report/subject_seed_results.csv','reports/stage_c/report/subject_results.csv',['dataset','subject','condition'],['ba','b0_ba','gain'])
seed_check('reports/stage_m/report/subject_seed_results.csv','reports/stage_m/report/subject_results.csv',['dataset','subject','method','shots'],['ba','b3_ba','gain'])
base='reports/population_budget/retry1/retained/report/'
seed_check(base+'all_subjects_and_actual_plateaus.csv',base+'subject_seed_means.csv',['dataset','subject','model','point'],['G_BA','own_ba','swap_ba','own_gain','upper_bound'])
# Historical CBraMod source aggregation is independently documented in its
# preserved prediction audit and scripts/stage_w_results.py:person(). We do not
# pretend unavailable original per-seed run files were re-read in this revision.
historical={'summary':'reports/stage_w/report/diagnostic1_subject_results.csv','evidence':'reports/stage_w/report/verification.json','aggregation_code':'scripts/stage_w_results.py:person','scope':'saved seed-averaged historical scores; original per-seed runs not present in this local report directory'}

summary=pd.read_csv(P/'tables/bootstrap_summary.csv');vectors=pd.read_csv(P/'tables/bootstrap_subject_vectors.csv')
la=pd.read_csv(R/'reports/stage_x/LaBraM/report/subject_means.csv')
la=la[la.dataset=='Cho2017'].sort_values(['dataset','subject'])
transfer=vectors[vectors.key=='core.LaBraM.Cho.swap_gain'].sort_values(['dataset','subject'])
np.testing.assert_allclose(transfer.value_pp,(la.swap_ba-la.G_BA)*100,atol=1e-10,rtol=0)
assert len(transfer)==52
for row in summary.itertuples():
    v=vectors[vectors.key==row.key].sort_values(['dataset','subject'])
    assert len(v)==row.n and not v.duplicated(['dataset','subject']).any()
    x=v.value_pp.to_numpy(); y=v.paired_value_pp.to_numpy()
    rng=np.random.default_rng(20260927+int(hashlib.sha256(row.key.encode()).hexdigest()[:8],16))
    indices=rng.integers(0,len(x),size=(20000,len(x)))
    for label,fn in [('mean',np.mean),('median',np.median)]:
        estimate=fn(x);draws=fn(x[indices],axis=1)
        if np.isfinite(y).all():estimate-=fn(y);draws-=fn(y[indices],axis=1)
        lo,hi=np.quantile(draws,[.025,.975])
        np.testing.assert_allclose([estimate,lo,hi],[getattr(row,label),getattr(row,label+'_lo'),getattr(row,label+'_hi')],atol=1e-10,rtol=0)

numbers=pd.read_csv(P/'numbers.csv').set_index('key')
for row in summary.itertuples():
    for stat in ['mean','median']:
        for suffix in ['', '_lo','_hi']:
            assert np.isclose(numbers.loc['bootstrap.'+row.key+'.'+stat+suffix,'value'],getattr(row,stat+suffix))

effects=pd.read_csv(R/'reports/stage_x/three_model_effects.csv')
assert len(effects)==9 and effects.model.nunique()==3 and effects.dataset.nunique()==3
assert (effects.own_minus_G_mean_pp>0).all() and (effects.own_minus_swap_mean_pp>0).all()
assert f'{effects.own_minus_G_mean_pp.min():.1f}'=='1.5' and f'{effects.own_minus_G_mean_pp.max():.1f}'=='5.4'
assert f'{effects.own_minus_swap_mean_pp.min():.1f}'=='2.3' and f'{effects.own_minus_swap_mean_pp.max():.1f}'=='7.3'
budget=pd.read_csv(R/(base+'subject_seed_means.csv'))
endpoints=budget[budget.point=='4x'].groupby('model').own_gain.median()*100
assert np.isclose(endpoints.min(),1) and np.isclose(endpoints.max(),2)
assert budget[['dataset','subject']].drop_duplicates().shape[0]==235
few=pd.read_csv(R/'reports/stage_x/three_model_few_shot.csv')
positive=few.groupby('dataset').mean_pp.apply(lambda x:bool((x>=0).all()))
assert positive.to_dict()=={'Cho2017':False,'Dreyer2023':False,'PhysionetMI':True}
unlisted=effects[~effects.pretraining_source.str.startswith('seen')]
assert len(unlisted)==6 and (unlisted[['own_minus_G_mean_pp','own_minus_swap_mean_pp']]>0).all().all()

report={'status':'passed','new_training':False,'new_inference':False,'new_hypothesis_tests':False,
        'bootstrap_contrasts_verified':len(summary),'bootstrap_ledger_values_verified':len(summary)*6,'bootstrap_resamples':20000,
        'seed_averaging_checks':checks,'historical_CBraMod_scope':historical,
        'abstract_numeric_checks':{'subjects':235,'models':3,'datasets':3,'combinations':9,'mean_personal_range':[1.5,5.4],'mean_specificity_range':[2.3,7.3],'fourfold_median_range':[1,2],'all_fourfold_medians_lower':True,'few_label_nonnegative_dataset':'PhysioNet'},
        'source_unlisted_positive_combinations':unlisted[['model','dataset']].to_dict('records'),
        'dependence_adjusted':False}
(P/'revision_statistics_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['seed_averaging_checks','source_unlisted_positive_combinations']},indent=2))
