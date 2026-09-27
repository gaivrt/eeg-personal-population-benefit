"""Read saved W losses and receipts only; never reconstruct or replay training."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'reports/stage_w/report'
out = ROOT/'reports/stage_w_followup/early_stop'
out.mkdir(parents=True, exist_ok=True)
convergence = pd.read_csv(source/'convergence.csv')
rows = []
figures = []
for cohort in ['starter','Lee2019_MI']:
    curve = pd.read_csv(source/f'{cohort}_curves.csv')
    assert 'subject_mean_ce' in curve and not any('accuracy' in c or c=='ba' for c in curve.columns)
    curve = curve[curve.role == 'validation'].copy()
    curve.to_csv(out/f'{cohort}_validation_losses.csv',index=False)
    datasets = sorted(curve.dataset.unique())
    if cohort == 'Lee2019_MI':
        datasets = ['Lee2019_MI']  # pooled is identical for this one-dataset cohort.
    for method in ['m_film','m_lora']:
        fig,axes = plt.subplots(len(datasets)+1,5,figsize=(17,2.6*(len(datasets)+1)),squeeze=False)
        for fold in range(5):
            for i,dataset in enumerate(datasets):
                ax = axes[i,fold]
                data = curve[(curve.method==method)&(curve.fold==fold)&(curve.dataset==dataset)]
                for seed,g in data.groupby('seed'):
                    g=g.sort_values('additional_step')
                    state=convergence[(convergence.cohort==cohort)&(convergence.method==method)&
                                      (convergence.fold==fold)&(convergence.seed==seed)].iloc[0]
                    assert g.additional_step.max()==state.additional_steps_run
                    line,=ax.plot(g.additional_step,g.subject_mean_ce,lw=1,label=str(seed))
                    selected=g[g.additional_step==state.selected_additional_step].iloc[0]
                    ax.scatter([selected.additional_step],[selected.subject_mean_ce],marker='*',s=45,color=line.get_color())
                    ax.scatter([g.additional_step.iloc[-1]],[g.subject_mean_ce.iloc[-1]],marker='x',s=22,color=line.get_color())
                    rows.append({'cohort':cohort,'method':method,'dataset':dataset,'fold':fold,'seed':int(seed),
                                 'initial_validation_ce':g.subject_mean_ce.iloc[0],
                                 'selected_validation_ce':selected.subject_mean_ce,
                                 'stop_validation_ce':g.subject_mean_ce.iloc[-1],
                                 'last_check_ce_change':g.subject_mean_ce.iloc[-1]-g.subject_mean_ce.iloc[-2],
                                 'accuracy_recorded':False,'accuracy_rising_at_stop':'not_identifiable'})
                ax.set_title(f'Fold {fold} / {dataset}',fontsize=9)
                ax.set_ylabel('Validation CE');ax.set_xlabel('Additional steps');ax.grid(alpha=.15)
                if fold==0 and i==0: ax.legend(title='Seed',fontsize=7,ncol=2)
            ax=axes[-1,fold]
            ax.text(.5,.5,'Validation accuracy\nnot recorded\ntrajectory unavailable',ha='center',va='center',transform=ax.transAxes,color='#64748b')
            ax.set_title(f'Fold {fold} / accuracy',fontsize=9)
            ax.set_xticks([]);ax.set_yticks([]);ax.set_facecolor('#f1f5f9')
        fig.suptitle(f'{cohort} / {method}: validation losses; star = selected step, x = actual stop\nAccuracy panels mark missing observations; no retraining or interpolation',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.955))
        name=f'{cohort}_{method}_validation_by_fold.png'
        fig.savefig(out/name,dpi=150);plt.close(fig);figures.append(name)
pd.DataFrame(rows).to_csv(out/'validation_endpoints.csv',index=False)

steps=convergence[['cohort','method','fold','seed','original_steps','additional_steps_cap',
                   'additional_steps_run','selected_additional_step','early_stopped']].copy()
steps['G_selected_total_steps']=steps.original_steps+steps.selected_additional_step
steps['G_executed_total_steps']=steps.original_steps+steps.additional_steps_run
resources=pd.read_csv(ROOT/'reports/stage_m/report/training_resources.csv')
r2_resources=resources[resources.method=='R2']
assert len(r2_resources)==25 and r2_resources.run.nunique()==25
assert r2_resources.outer_steps.eq(2000).all()
steps['R2_additional_steps']=np.where((steps.cohort=='starter')&(steps.method=='m_lora'),2000,np.nan)
steps['R2_total_steps']=np.where(steps.R2_additional_steps.notna(),steps.original_steps+steps.R2_additional_steps,np.nan)
steps['G_selected_minus_R2_total']=steps.G_selected_total_steps-steps.R2_total_steps
steps.to_csv(out/'G_R2_training_steps.csv',index=False)
summary=[]
for (cohort,method),g in steps.groupby(['cohort','method']):
    summary.append({'cohort':cohort,'method':method,'runs':len(g),
                    'original_min':int(g.original_steps.min()),'original_max':int(g.original_steps.max()),
                    'G_run_extra_min':int(g.additional_steps_run.min()),'G_run_extra_max':int(g.additional_steps_run.max()),
                    'G_selected_extra_min':int(g.selected_additional_step.min()),'G_selected_extra_max':int(g.selected_additional_step.max()),
                    'selected_start_count':int((g.selected_additional_step==0).sum())})
pd.DataFrame(summary).to_csv(out/'step_summary.csv',index=False)
lora=steps[(steps.cohort=='starter')&(steps.method=='m_lora')]
audit={'status':'complete','no_retraining':True,'models':len(steps),'validation_accuracy_trajectory_available':False,
       'accuracy_rising_at_early_stop':'not_identifiable_from_saved_losses',
       'reason':'W saved CE only and overwrote best checkpoints; no per-checkpoint predictions or complete weights.',
       'G_remains_primary':True,'R2_is_sensitivity_only':True,
       'starter_LoRA_G_selected_fewer_equal_more_steps_than_R2':{
           'fewer':int((lora.G_selected_minus_R2_total<0).sum()),'equal':int((lora.G_selected_minus_R2_total==0).sum()),
           'more':int((lora.G_selected_minus_R2_total>0).sum())},'figures':figures}
(out/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps({'audit':audit,'step_summary':summary},indent=2))
