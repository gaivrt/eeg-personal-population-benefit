"""Local CPU audit/figures for completed follow-up receipts; no model fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('root',type=Path,help='Extracted evidence root containing stage_w_followup')
parser.add_argument('output',type=Path)
args=parser.parse_args()
root=args.root/'stage_w_followup';out=args.output;out.mkdir(parents=True,exist_ok=True)
prep=json.loads((root/'bnci/splits.json').read_text(encoding='utf-8'))['folds']
runs=sorted((root/'bnci/runs').glob('fold-*_seed-*'));assert len(runs)==25
rows=[];endpoints=[];curves=[];batch_checks=[];few_flags=[]
for directory in runs:
    receipt=json.loads((directory/'run.json').read_text(encoding='utf-8'))
    assert receipt['git_commit']=='ea764cd2d2e6cc67faf295ef3086f8b52e87e30d'
    assert not receipt['external_hyperparameter_search'] and receipt['backbone_unchanged']
    assert len(receipt['readout_channels'])==19
    for relative,digest in receipt['source_config_hashes'].items():
        assert hashlib.sha256((args.root/relative).read_bytes()).hexdigest()==digest
    for field,relative,subkey in [
        ('population_hyperparameters',f'stage_c/runs/{directory.name}/selection.json',None),
        ('personal_hyperparameters',f'stage_w/diagnostic1/runs/{directory.name}/selection.json',None),
        ('few_shot_hyperparameters',f'stage_w/diagnostic3/runs/{directory.name}/selection.json',None),
        ('head_hyperparameters',f'baselines_v2/B0_{receipt["head_kind"]}/{directory.name}/run.json','selection'),
        ('linear_probe_hyperparameters',f'baselines_v2/B0_linear/{directory.name}/run.json','selection')]:
        original=json.loads((args.root/relative).read_text(encoding='utf-8'))
        assert receipt[field]==(original[subkey] if subkey else original)
    fold,seed=receipt['fold'],receipt['seed']
    predictions=pd.read_csv(directory/'predictions.csv.gz')
    for subject,g in predictions.groupby('subject'):
        full=g[g.condition=='G_lora8'].set_index('trial_id').prediction
        first=None
        for n,z in g[g.condition=='few_G'].groupby('shots'):
            query=z.set_index('trial_id').prediction
            if first is not None:np.testing.assert_array_equal(query.loc[first.index],first)
            first=query
            batch_checks.append({'fold':fold,'seed':seed,'subject':subject,'shots':n,
                                 'trial_flips':int(np.count_nonzero(full.loc[query.index].to_numpy()!=query.to_numpy()))})
    few_flags.append(pd.read_csv(directory/'few_shot.csv').assign(fold=fold,seed=seed))
    curve=pd.read_csv(directory/'curves.csv')
    person=pd.read_csv(directory/'subject_losses.csv')
    assert set(person.subject)<=set(s for role in ['train','validation'] for _,s in prep[fold][role])
    assert not set(person.subject)&{s for _,s in prep[fold]['test']}
    assert person.accuracy.between(0,1).all() and person.ba.between(0,1).all()
    for (method,role,step),g in person.groupby(['method','role','additional_step']):
        c=curve[(curve.method==method)&(curve.role==role)&(curve.additional_step==step)]
        for dataset in ['BNCI2014_001','pooled']:
            r=c[c.dataset==dataset].iloc[0]
            np.testing.assert_allclose([g.loss.mean(),g.accuracy.mean(),g.ba.mean()],
                                      [r.subject_mean_ce,r.subject_mean_accuracy,r.subject_mean_ba],atol=1e-12,rtol=0)
    for method,status in receipt['families'].items():
        rows.append({'fold':fold,'seed':seed,'method':method,**status})
        for role in ['train','validation']:
            c=curve[(curve.method==method)&(curve.role==role)&(curve.dataset=='pooled')].sort_values('additional_step')
            selected=c[c.additional_step==status['selected_additional_step']].iloc[0]
            last,previous=c.iloc[-1],c.iloc[-2]
            endpoints.append({'fold':fold,'seed':seed,'method':method,'role':role,
                'initial_accuracy':c.iloc[0].subject_mean_accuracy,'selected_accuracy':selected.subject_mean_accuracy,
                'stop_accuracy':last.subject_mean_accuracy,'last_accuracy_change':last.subject_mean_accuracy-previous.subject_mean_accuracy,
                'last_ce_change':last.subject_mean_ce-previous.subject_mean_ce,
                'last_interval_accuracy_rising':bool(last.subject_mean_accuracy>previous.subject_mean_accuracy)})
    curves.append(curve.assign(fold=fold,seed=seed))
status=pd.DataFrame(rows);status.to_csv(out/'bnci_convergence.csv',index=False)
pd.DataFrame(batch_checks).to_csv(out/'bnci_batch_path_checks.csv',index=False)
pd.concat(few_flags,ignore_index=True).to_csv(out/'bnci_few_shot_seed_checks.csv',index=False)
pd.DataFrame(endpoints).to_csv(out/'bnci_validation_accuracy_endpoints.csv',index=False)
all_curves=pd.concat(curves,ignore_index=True)
for method in ['m_film','m_lora']:
    fig,axes=plt.subplots(4,5,figsize=(17,10),squeeze=False)
    panels=[('train','subject_mean_ce','Train CE'),('validation','subject_mean_ce','Validation CE'),
            ('train','subject_mean_accuracy','Train accuracy'),('validation','subject_mean_accuracy','Validation accuracy')]
    for fold in range(5):
        for i,(role,metric,label) in enumerate(panels):
            ax=axes[i,fold]
            part=all_curves[(all_curves.method==method)&(all_curves.fold==fold)&(all_curves.role==role)&(all_curves.dataset=='pooled')]
            for seed,c in part.groupby('seed'):
                c=c.sort_values('additional_step')
                chosen=status[(status.method==method)&(status.fold==fold)&(status.seed==seed)].iloc[0].selected_additional_step
                line,=ax.plot(c.additional_step,c[metric],lw=1,label=str(seed))
                selected=c[c.additional_step==chosen].iloc[0]
                ax.scatter([chosen],[selected[metric]],marker='*',s=40,color=line.get_color())
                ax.scatter([c.additional_step.iloc[-1]],[c[metric].iloc[-1]],marker='x',s=20,color=line.get_color())
            ax.set_title(f'Fold {fold}: {label}',fontsize=9);ax.set_xlabel('Additional steps');ax.grid(alpha=.15)
            if metric.endswith('accuracy'):ax.set_ylim(0,1)
            if fold==0 and i==0:ax.legend(ncol=2,fontsize=7,title='Seed')
    fig.suptitle(f'BNCI session 2 / {method}\nStar = CE-selected checkpoint; x = actual stop. Accuracy does not select checkpoints.',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/f'bnci_{method}_training_validation_by_fold.png',dpi=150);plt.close(fig)
verification={'status':'complete','models':len(status),'no_training_performed':True,
              'new_accuracy_records_match_per_subject_means':True,'test_subjects_absent_from_selection_curves':True,
              'historical_G_accuracy_trajectory_recovered':False,'checkpoint_selection':'validation_CE_only',
              'all_starter_hyperparameter_values_and_source_hashes_verified':True,
              'batch_path_comparisons':len(batch_checks),'batch_path_trial_flips':sum(r['trial_flips'] for r in batch_checks),
              'one_class_support_rows':int(pd.concat(few_flags).one_class_only.sum()),
              'families':{m:{'models':len(g),'selected_start':int(g.selected_additional_step.eq(0).sum()),
                            'early_stopped':int(g.early_stopped.sum()),'plateau_observed':int(g.plateau_observed.sum()),
                            'additional_steps_min':int(g.additional_steps_run.min()),'additional_steps_max':int(g.additional_steps_run.max())}
                           for m,g in status.groupby('method')}}
(out/'postprocess_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
report=root/'report'
tables=['# 补充结果表（每人先平均五种子）','']
strength=pd.read_csv(report/'population_strength_by_dataset.csv')
tables.extend(['## B3 / G / R2 按库并列','',
    '| 数据集 | 个人参数 | 起点 | 群体族 | 群体 BA/%（均值±被试SD） | 本人−群体：均值 / 中位pp | 本人−交换：均值 / 中位pp |',
    '|---|---|---|---|---:|---:|---:|'])
for dataset in ['PhysionetMI','Dreyer2023','Cho2017']:
    for variant in ['film_offset','lora8']:
        for start in ['B3','G','R2']:
            r=strength[(strength.dataset==dataset)&(strength.variant==variant)&(strength.start==start)].iloc[0]
            tables.append(f'| {dataset} | {variant} | {start} | {r.population_family} | {r.population_mean_BA:.3f} ± {r.population_sd_BA:.3f} | {r.own_minus_population_mean_pp:+.3f} / {r.own_minus_population_median_pp:+.3f} | {r.own_minus_swap_mean_pp:+.3f} / {r.own_minus_swap_median_pp:+.3f} |')
tables.extend(['','G为主参照，R2为敏感性分析；FiLM的R2行属于LoRA群体族，与历史B3/G-FiLM行并非只差强度。','',
    '## BNCI 群体与个人成绩','',
    '| 条件 | 有效人数 | BA/%（均值±被试SD） | 本人−G均值pp | 本人−交换均值pp |',
    '|---|---:|---:|---:|---:|'])
base=pd.read_csv(report/'bnci_baselines.csv')
for condition,g in base.groupby('condition'):
    tables.append(f'| {condition} | {len(g)} | {100*g.ba.mean():.3f} ± {100*g.ba.std(ddof=1):.3f} | — | — |')
d1=pd.read_csv(report/'bnci_diagnostic1_subjects.csv')
for variant,g in d1.groupby('variant'):
    tables.append(f'| 本人 {variant} | {len(g)} | {100*g.own_ba.mean():.3f} ± {100*g.own_ba.std(ddof=1):.3f} | {100*g.own_gain.mean():+.3f} | {100*g.upper_bound.mean():+.3f}（N={g.upper_bound.notna().sum()}） |')
    valid=g[g.swap_ba.notna()]
    tables.append(f'| 交换 {variant} | {len(valid)} | {100*valid.swap_ba.mean():.3f} ± {100*valid.swap_ba.std(ddof=1):.3f} | — | — |')
tables.extend(['','本人−G基于9人，本人−交换只基于8人；不能用9人本人均值减8人交换均值代替配对差。','',
    '## BNCI 少样本','',
    '| n | N | BA/%（均值±被试SD） | 相对G平均增益pp | 增益中位pp |',
    '|---:|---:|---:|---:|---:|'])
few=pd.read_csv(report/'bnci_few_shot_subjects.csv')
for n,g in few.groupby('shots'):
    tables.append(f'| {n} | {len(g)} | {100*g.ba.mean():.3f} ± {100*g.ba.std(ddof=1):.3f} | {100*g.gain.mean():+.3f} | {100*g.gain.median():+.3f} |')
(out/'tables.md').write_text('\n'.join(tables)+'\n',encoding='utf-8')
print(json.dumps(verification))
