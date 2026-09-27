"""Audit all saved predictions and report R2 sensitivity and BNCI replication separately."""
import argparse,json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from subject_context.stage_a_common import ROOT,scratch_root,sha256,save_json
from subject_context.statistics import holm
from stage_w_results import audit_truth,concat,person,distribution,score

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path);args=parser.parse_args()
    root=args.root or scratch_root();out=root/'stage_w_followup/report';out.mkdir(parents=True,exist_ok=True)
    config_hash=sha256(ROOT/'configs/stage_w_followup.yaml')
    runs={};counts={};receipts={}
    for label in ['r2','bnci']:
        directories=sorted((root/f'stage_w_followup/{label}/runs').glob('fold-*_seed-*'))
        assert len(directories)==25
        seen=set();checked=0;records=[]
        for directory in directories:
            r=json.loads((directory/'run.json').read_text());assert r['status']=='complete'
            assert r.get('w_config_sha256',r['config_sha256'])==config_hash
            seen.add((r['fold'],r['seed']));records.append(r)
            p=pd.read_csv(directory/'predictions.csv.gz');audit_truth(root,p);checked+=len(p)
            half_path=(ROOT/'configs/splits/stage_b_halves_v1.json' if label=='r2' else
                       root/'stage_w_followup/bnci/halves.json')
            halves={(v['dataset'],v['subject']):v for v in json.loads(half_path.read_text())['subjects']}
            for key,g in p.groupby(['dataset','subject']):
                trials_path=root/'processed'/key[0]/f'sub-{key[1]:03d}'/'trials.csv'
                assert sha256(trials_path)==halves[key]['trial_table_sha256']
                expected=set(pd.read_csv(trials_path).iloc[halves[key]['query_indices']].trial_id)
                grouping=[c for c in ['variant','condition','shots','donor'] if c in g]
                for _,prediction_set in g.groupby(grouping,dropna=False):
                    assert not prediction_set.trial_id.duplicated().any()
                    assert set(prediction_set.trial_id)==expected
            subjects=pd.read_csv(directory/'subjects.csv')
            assert not subjects.duplicated(['variant','dataset','subject']).any()
            bases=pd.read_csv(directory/'baseline.csv')
            if label=='bnci':
                for b in bases.itertuples():
                    g=p[(p.subject==b.subject)&(p.condition==b.condition)]
                    assert len(g)==72
                    np.testing.assert_allclose(score(g.label,g.prediction),b.ba,atol=1e-12,rtol=0)
                frozen=json.loads((directory/'frozen_hyperparameters.json').read_text())
                assert not r['external_hyperparameter_search']
                assert r['readout_channels']==json.loads((root/'stage_w_followup/bnci/prepared.json').read_text())['readout_channels']
                for relative,digest in r['source_config_hashes'].items():
                    assert sha256(root/relative)==digest
                for field in ['head_hyperparameters','population_hyperparameters','personal_hyperparameters',
                              'few_shot_hyperparameters','linear_probe_hyperparameters']:
                    assert r[field]==frozen[field]
            else:
                assert r['primary_reference']=='G' and r['sensitivity_only'] and r['population_unchanged']
                for b in bases[bases.role=='test'].itertuples():
                    for condition,value in [('R2_all_batch',b.all_batch_ba),('R2_query_batch',b.query_batch_ba)]:
                        g=p[(p.dataset==b.dataset)&(p.subject==b.subject)&(p.condition==condition)]
                        np.testing.assert_allclose(score(g.label,g.prediction),value,atol=1e-12,rtol=0)
            for row in subjects.itertuples():
                selection=p[(p.dataset==row.dataset)&(p.subject==row.subject)]
                if label=='r2':selection=selection[(selection.variant==row.variant)&(selection.condition=='swap')]
                else:selection=selection[selection.condition=='swap_'+row.variant]
                matrix={int(d):score(g.label,g.prediction) for d,g in selection.groupby('donor')}
                np.testing.assert_allclose(matrix[row.subject],row.own_ba,atol=1e-12,rtol=0)
                np.testing.assert_allclose(row.own_ba-row.b3_ba,row.own_gain,atol=1e-12,rtol=0)
                if pd.isna(row.swap_ba):
                    assert label=='bnci' and len(matrix)==1 and not row.swap_available
                    assert pd.isna(row.upper_bound) and pd.isna(row.swap_gain)
                else:
                    donors=[int(s) for s in str(row.swap_donors).split()];assert len(donors)==10 and row.subject not in donors
                    np.testing.assert_allclose([np.mean([matrix[d] for d in donors]),row.own_ba-row.swap_ba,row.swap_ba-row.b3_ba],
                                              [row.swap_ba,row.upper_bound,row.swap_gain],atol=1e-12,rtol=0)
            if label=='bnci':
                for row in pd.read_csv(directory/'few_shot.csv').itertuples():
                    q=p[(p.subject==row.subject)&(p.condition=='few_lora8')&(p.shots==row.shots)]
                    z=p[(p.subject==row.subject)&(p.condition=='few_G')&(p.shots==row.shots)]
                    np.testing.assert_allclose([score(q.label,q.prediction),score(z.label,z.prediction),row.ba-row.b3_ba],
                                              [row.ba,row.b3_ba,row.gain],atol=1e-12,rtol=0)
                nn=pd.read_csv(directory/'neighbours.csv');features=dict(np.load(directory/'rest_features.npz'))
                assert all(np.isfinite(v).all() for v in features.values())
                for row in nn.itertuples():
                    if not row.available:
                        assert len(features)==1 and pd.isna(row.difference)
                        continue
                    target=f'{row.dataset}:{row.subject}';others=[s for s in features if s!=target];assert len(others)==1
                    donor=int(others[0].split(':')[1]);assert int(row.neighbours)==donor
                    g=p[(p.subject==row.subject)&(p.condition=='swap_'+row.variant)&(p.donor==donor)]
                    actual=score(g.label,g.prediction)
                    np.testing.assert_allclose([row.nearest_ba,row.random_ba,row.difference],[actual,actual,0],atol=1e-12,rtol=0)
        assert seen=={(f,s) for f in range(5) for s in [11,23,37,53,71]}
        runs[label]=directories;counts[label]=checked;receipts[label]=records

    columns=['b3_ba','own_ba','swap_ba','upper_bound','own_gain','swap_gain']
    historical=[]
    for start,directory in [('B3','stage_c2/report'),('G','stage_w/report')]:
        table=pd.read_csv(root/directory/('subject_results.csv' if start=='B3' else 'diagnostic1_subject_results.csv'))
        table=table[table.variant.isin(['film_offset','lora8'])]
        if start=='G':table=table[(table.start=='G')&(table.dataset!='Lee2019_MI')]
        historical.append(table[['variant','dataset','subject',*columns]].assign(start=start))
    r2=person(concat(runs['r2'],'subjects.csv'),['variant','dataset','subject'],columns).assign(start='R2')
    combined=pd.concat([*historical,r2],ignore_index=True);assert len(combined)==1410
    combined.to_csv(out/'population_strength_subjects.csv',index=False)
    summary=[]
    for (start,variant,dataset),g in combined.groupby(['start','variant','dataset']):
        row={'start':start,'variant':variant,'dataset':dataset,'subjects':len(g),
             'population_family':'m_film' if variant=='film_offset' and start!='R2' else 'm_lora',
             'reference_status':'primary' if start=='G' else 'sensitivity' if start=='R2' else 'historical'}
        for label,column in [('population','b3_ba'),('own','own_ba'),('swap','swap_ba')]:
            row[label+'_mean_BA']=100*g[column].mean();row[label+'_sd_BA']=100*g[column].std(ddof=1)
        for label,column in [('own_minus_population','own_gain'),('own_minus_swap','upper_bound')]:
            row.update({label+'_'+k:v for k,v in distribution(g[column]).items()})
        summary.append(row)
    pd.DataFrame(summary).to_csv(out/'population_strength_by_dataset.csv',index=False)

    def gate(frame):
        result=[]
        for variant,g in frame.groupby('variant'):
            valid=g.upper_bound.dropna()
            result.append({'variant':variant,'own_subjects':len(g),'swap_subjects':len(valid),**distribution(valid)})
        for row,p in zip(result,holm([r['p_one_sided_raw'] for r in result]+[1.])[:len(result)]):
            row.update(p_holm=float(p),passed=bool(row['median_pp']>=2 and p<.05),family_size=3)
        return result
    bnci=person(concat(runs['bnci'],'subjects.csv'),['variant','dataset','subject'],columns)
    assert len(bnci)==18 and bnci.upper_bound.isna().sum()==2
    bnci.to_csv(out/'bnci_diagnostic1_subjects.csv',index=False)
    base=person(concat(runs['bnci'],'baseline.csv'),['condition','dataset','subject'],['ba'])
    assert len(base)==36
    base.to_csv(out/'bnci_baselines.csv',index=False)
    few=person(concat(runs['bnci'],'few_shot.csv'),['variant','shots','dataset','subject'],['ba','b3_ba','gain','full_gain'])
    assert len(few)==27;few.to_csv(out/'bnci_few_shot_subjects.csv',index=False)
    nearest=person(concat(runs['bnci'],'neighbours.csv'),['variant','source','k','dataset','subject'],['nearest_ba','random_ba','difference'])
    nearest.to_csv(out/'bnci_neighbour_subjects.csv',index=False)
    summaries=[]
    for label,frame,groupcols,cols in [('baseline',base,['condition'],['ba']),('diagnostic1',bnci,['variant'],columns),
                                    ('few_shot',few,['shots'],['ba','b3_ba','gain','full_gain'])]:
        for group,g in frame.groupby(groupcols):
            for col in cols:
                values=g[col].dropna()
                summaries.append({'analysis':label,'group':str(group),'metric':col,'n':len(values),
                                  'mean_percent_or_pp':100*values.mean(),'sd_percent_or_pp':100*values.std(ddof=1),
                                  'median_percent_or_pp':100*values.median()})
    pd.DataFrame(summaries).to_csv(out/'bnci_summary.csv',index=False)
    paired=base.pivot(index='subject',columns='condition',values='ba');comparisons=[]
    for method in ['G_film_offset','G_lora8']:
        for control in ['B0','B0_linear']:
            comparisons.append({'method':method,'control':control,**distribution(paired[method]-paired[control])})
    pd.DataFrame(comparisons).to_csv(out/'bnci_population_comparisons.csv',index=False)
    save_json(out/'diagnostic_tests.json',{'R2_sensitivity_diagnostic1':gate(r2),'BNCI_diagnostic1_available_subset':gate(bnci),
                 'BNCI_rest':'not_identifiable: singleton has no donor; every other test fold has exactly one donor per target',
                 'primary_population_reference':'G','R2_changes_primary_reference':False})
    curve=concat(runs['bnci'],'curves.csv');curve.to_csv(out/'bnci_training_curves.csv',index=False)
    save_json(out/'run_manifest.json',receipts)
    save_json(out/'verification.json',{'status':'complete','completed_runs':50,'prediction_rows':counts,
                 'BNCI_subjects':9,'BNCI_swap_subjects':8,'BNCI_missing_swap_subjects':bnci[bnci.upper_bound.isna()].subject.unique().tolist(),
                 'no_population_retraining_R2':True,'historical_G_remains_primary':True})
    print(json.dumps({'status':'complete','prediction_rows':counts}),flush=True)

if __name__=='__main__':main()
