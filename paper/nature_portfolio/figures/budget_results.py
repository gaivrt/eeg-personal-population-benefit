"""Report the completed, independently audited budget curves; no new tests."""
from common import *
from tables import catalog_csv,catalog_json

BASE='reports/population_budget/retry1/retained/'
ORIGINAL='reports/population_budget/final/'
RETRY='reports/population_budget/retry1/'
def build():
    d=catalog_csv(BASE+'report/curve_summary.csv','Figure 3; Supplementary budget results','C1/C2')
    interpretation=catalog_json(BASE+'report/interpretation.json','Supplementary budget decisions','C2')
    available=catalog_json(BASE+'report/availability.json','Figure 3: complete original cohort','C2')
    cb_idx=next(i for i,x in enumerate(interpretation) if x['model']=='CBraMod')
    cb=interpretation[cb_idx]
    add('budget.CBraMod.change_2x_4x',abs(cb['median_own_minus_G_pp']['4x']-cb['median_own_minus_G_pp']['2x']),'pp',BASE+'report/interpretation.json',f'/{cb_idx}/median_own_minus_G_pp','2x,4x','abs(saved 4x median - saved 2x median); budget_results.py','Results: CBraMod stability criterion','C2')
    fact('BudgetCBChange','budget.CBraMod.change_2x_4x')
    for point in ['2x','4x']:
        key='budget.CBraMod.'+point+'.six_slot_Holm'
        add(key,cb['six_slot_Holm'][point],'p',BASE+'report/interpretation.json',f'/{cb_idx}/six_slot_Holm/{point}',point,'direct JSON pointer','Results: CBraMod stable-positive-gain rule','C2')
        fact('BudgetCB'+point[0]+'PH',key)
    source('docs/population_budget_protocol.md')
    source(ORIGINAL+'integrity-audit-final.json')
    source(RETRY+'final_verification.json')
    source(RETRY+'local_verification.json')
    source('docs/population_budget_retry_protocol.md')
    retry=catalog_json(RETRY+'final_decisions.json','Methods; Supplementary: retry and precision exception','C2')
    freeze=catalog_json(RETRY+'experiment_freeze.json','Methods: final recorded compute','C2')
    add('budget.retry.swap_difference',100*retry['precision_comparison']['maximum_absolute_BA_difference']['swap_ba'],'pp',RETRY+'final_decisions.json',
        '/precision_comparison/maximum_absolute_BA_difference/swap_ba','swap_ba','100 * direct JSON pointer','Supplementary: retry precision comparison','C2')
    fact('BudgetRetrySwapDifference','budget.retry.swap_difference')
    verification=catalog_json(ORIGINAL+'local_verification.json','Supplementary: historical prediction comparisons','C2')
    cb_difference=verification['historical_1x_comparisons'][0]['max_BA_difference']
    add('budget.CBraMod.path_difference',100*cb_difference,'pp',ORIGINAL+'local_verification.json',
        '/historical_1x_comparisons/0/max_BA_difference','max_BA_difference','100 * direct JSON pointer',
        'Supplementary: historical prediction discrepancy','C2')
    fact('BudgetCBPathDifference','budget.CBraMod.path_difference')
    subjects=read(BASE+'report/subject_seed_means.csv')
    assert set(d.model)=={'CBraMod','REVE','LaBraM'}
    for model in ['CBraMod','REVE','LaBraM']:
        assert all(x['complete_235_times_5'] for x in available if x['model']==model)
    pooled=d[d.dataset=='pooled']
    for model in ['CBraMod','REVE','LaBraM']:
        for point in ['1x','2x','4x']:
            x=pooled[(pooled.model==model)&(pooled.point==point)]
            assert len(x)==1 and int(x.iloc[0].N)==235
            for alias,col in [('G','G_BA_mean_percent'),('Delta','own_minus_G_median_pp'),('U','own_minus_swap_median_pp')]:
                key=f'budget.{model}.{point}.{alias}'
                direct(x,x.index[0],col,key,'Results: population-training budgets','C1/C2',unit='BA_percent' if alias=='G' else 'pp')
                fact('Budget'+model+point[0]+alias,key)
    direct(pooled,pooled.index[0],'N','budget.cohort.N','Figure 3; Methods: budget cohort','C2',unit='count')
    fact('BudgetN','budget.cohort.N')

    # Main figure: identical subject-level median estimands in all three panels.
    # One CBraMod trajectory uses the separately registered precision exception.
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.1),sharey=True)
    for j,model in enumerate(['CBraMod','REVE','LaBraM']):
        ax=axes[j]
        s=pooled[pooled.model==model].set_index('point').loc[['1x','2x','4x']]
        for col,color,marker,label in [('own_minus_G_median_pp',BLUE,'o','Own − population'),
                                        ('own_minus_swap_median_pp',COPPER,'D','Own − swapped')]:
            ax.plot([1,2,4],s[col],color=color,marker=marker,label=label,lw=1.6,ms=4.2)
        ax.set(title=f'{"ABC"[j]}  {model}',xlim=(.7,4.3),ylim=(-.2,4.05),
               xticks=[1,2,4],xticklabels=['1×','2×','4×'],xlabel='Population update budget')
        ax.axhline(0,color='#83909C',lw=.7)
        ax.grid(axis='y',alpha=.8)
    axes[0].set_ylabel('Pooled median difference (pp)')
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1),ncol=2)
    fig.subplots_adjust(left=.09,right=.985,bottom=.20,top=.78,wspace=.22)
    save(fig,'fig3_budget_personal_medians')

    # Retained as a supplementary figure: G and dataset-specific context.
    fig,axes=plt.subplots(3,3,figsize=(7.2,6.8))
    for row,model in enumerate(['CBraMod','REVE','LaBraM']):
        for col,(metric,title) in enumerate([('G_BA_mean_percent','Population BA (%)'),('own_minus_G_median_pp','Median own − G (pp)'),('own_minus_swap_median_pp','Median own − swapped (pp)')]):
            ax=axes[row,col]
            for j,ds in enumerate(DATASETS[:3]+['pooled']):
                s=d[(d.model==model)&(d.dataset==ds)].set_index('point').loc[['1x','2x','4x']]
                ax.plot([1,2,4],s[metric],marker=['o','s','^','D'][j],ms=3.6,color=[BLUE,COPPER,TEAL,INK][j],ls='--' if ds=='pooled' else '-',label='Pooled' if ds=='pooled' else NAMES[ds],lw=1.65 if ds=='pooled' else 1.2)
            ax.set(title=f'{"ABCDEFGHI"[row*3+col]}  {model}',xticks=[1,2,4],xticklabels=['1×','2×','4×'],
                   xlabel='Population update budget' if row==2 else '',
                   ylabel=['Mean population BA (%)','Median own − G (pp)','Median own − swapped (pp)'][col])
            if col:ax.axhline(0,color='#83909C',lw=.7)
            ax.margins(x=.10,y=.14);ax.grid(axis='y',alpha=.8)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1),ncol=4)
    fig.subplots_adjust(left=.09,right=.985,bottom=.08,top=.90,wspace=.57,hspace=.62)
    save(fig,'fig3_population_budget')
    for model in ['CBraMod','REVE','LaBraM']:
        rows=[]
        for ds in DATASETS[:3]+['pooled']:
            for point in ['1x','2x','4x']:
                r=d[(d.model==model)&(d.dataset==ds)&(d.point==point)].iloc[0]
                fmt=lambda mean,sd:f'{r[mean]:.2f} ({r[sd]:.2f})'
                rows.append(['Pooled' if ds=='pooled' else NAMES[ds],point,int(r.N),fmt('G_BA_mean_percent','G_BA_SD_percent'),fmt('own_minus_G_mean_pp','own_minus_G_SD_pp'),f'{r.own_minus_G_median_pp:.3f}',fmt('own_minus_swap_mean_pp','own_minus_swap_SD_pp'),f'{r.own_minus_swap_median_pp:.3f}'])
        table_file('budget_'+model,['Dataset','Budget','N','G mean (SD)','Delta mean (SD)','Median','U mean (SD)','Median'],rows,widths='p{.8in}rrp{.9in}p{1.0in}rp{1.0in}r',long=True)
    rows=[]
    for obj in interpretation:
        for point in ['2x','4x']:
            rows.append([obj['model'],point,f'{obj["six_slot_Holm"][point]:.3g}',
                         'Operational criterion met' if obj['model']=='CBraMod' else 'Descriptive only'])
    table_file('budget_decisions',['Model','Budget','Six-slot Holm p','Curve interpretation'],rows,long=True)
    initial=catalog_json(ORIGINAL+'report/interpretation.json','Supplementary: initial budget decision record','C2')
    catalog_csv(ORIGINAL+'report/curve_summary.csv','Supplementary: initial result archive','C2')
    rows=[]
    for obj in initial:
        for point in ['2x','4x']:
            missing=obj['model']=='CBraMod'
            rows.append([obj['model'],point,'NA' if missing else f'{obj["six_slot_Holm"][point]:.3g}',
                         'Incomplete cohort' if missing else 'Descriptive only'])
    table_file('budget_initial_decisions',['Model','Budget','Initial six-slot Holm p','Initial interpretation'],rows,long=True)
    rows=[]
    for _,r in d.iterrows():
        rows.append([r.model,'Pooled' if r.dataset=='pooled' else NAMES[r.dataset],r.point,f'{r.own_minus_G_p_raw_descriptive:.3g}',f'{r.own_minus_swap_p_raw_descriptive:.3g}','--' if pd.isna(r.swap_original_three_slot_Holm) else f'{r.swap_original_three_slot_Holm:.3g}',f'{r.own_minus_G_drop_gt_2pp_fraction:.3f}',f'{r.own_minus_swap_drop_gt_2pp_fraction:.3f}'])
    table_file('budget_all_tests',['Model','Dataset','Budget','Delta raw p','U raw p','U budget Holm','Delta decline','U decline'],rows,long=True)
    validation=read(RETRY+'endpoint_validation_final.csv');rows=[]
    for (model,point,ds),s in validation.groupby(['model','point','dataset']):
        assert len(s)==25
        vals=[]
        for col,scale in [('CE',1),('accuracy',100),('BA',100)]:
            key=f'budget.validation.{model}.{point}.{ds}.{col}'
            mean=stat(s,col,key,'Supplementary budget validation','C1/C2',scale=scale,unit='CE' if col=='CE' else 'percent')
            sd=stat(s,col,key+'.sd','Supplementary budget validation','C1/C2',op='sd',scale=scale,unit='CE' if col=='CE' else 'percent')
            vals.append(f'{mean:.3f} ({sd:.3f})')
        rows.append([model,'Pooled' if ds=='pooled' else NAMES[ds],point]+vals)
    table_file('budget_validation',['Model','Dataset','Budget','CE mean (SD)',r'Accuracy \% (SD)',r'BA \% (SD)'],rows,long=True)
    resources=catalog_json(ORIGINAL+'all_model_resources.json','Methods: original budget compute','C1/C2')
    for i,r in enumerate(resources['models']):
        fact('Budget'+r['model']+'Hours','json:'+ORIGINAL+'all_model_resources.json:/models/'+str(i)+'/card_hours')
    fact('BudgetTotalHours','json:'+RETRY+'experiment_freeze.json:/combined_card_hours')
    fact('BudgetRetryHours','json:'+RETRY+'experiment_freeze.json:/retry_card_hours')
    progress=catalog_json(ORIGINAL+'report/progress.json','Supplementary: initial budget execution','C2')
    for name,value in [('BudgetRuns',len(progress['runs'])),
                       ('BudgetCBSuccess',sum(r['model']=='CBraMod' and r['status']=='complete' for r in progress['runs'])),
                       ('BudgetFailed',sum(r['status']=='failed' for r in progress['runs']))]:
        key='budget.execution.'+name
        add(key,value,'count',ORIGINAL+'report/progress.json','/runs','model,status',
            'count saved runs with the indicated model/status; budget_results.py','Results; Figure 3; Supplementary budget','C2')
        fact(name,key)
    roster=read(ORIGINAL+'report/CBraMod_full_roster_with_missing.csv')
    missing_rows=[]
    for point in ['1x','2x','4x']:
        for ds in DATASETS[:3]+['pooled']:
            s=roster[(roster.point==point)&((roster.dataset==ds) if ds!='pooled' else True)]
            counts=s.groupby(['dataset','subject'])['observed'].sum()
            seeds=s.seed.nunique();n=len(counts);observed=int(s.observed.sum())
            values={'subjects':n,'seeds':seeds,'planned':len(s),'observed':observed,
                    'complete_subjects':int((counts==seeds).sum()),'incomplete_subjects':int((counts<seeds).sum())}
            for label,value in values.items():
                key=f'budget.CBraMod.coverage.{point}.{ds}.{label}'
                add(key,value,'count',roster.attrs['source'],','.join(s['_record'].astype(str)),
                    'dataset,subject,seed,point,observed','coverage count over full saved roster; budget_results.py',
                    'Figure 3; Supplementary budget missingness','C2')
                if point=='1x' and ds=='pooled':
                    alias={'seeds':'BudgetSeeds','observed':'BudgetCBRows','complete_subjects':'BudgetCBCompleteSubjects','incomplete_subjects':'BudgetCBMissingSubjects'}.get(label)
                    if alias:fact(alias,key)
            missing_rows.append([point,'Pooled' if ds=='pooled' else NAMES[ds],n,seeds,len(s),observed,values['complete_subjects'],values['incomplete_subjects']])
    table_file('budget_CB_missing',['Budget','Dataset','N','Seeds','Planned pairs','Observed pairs','Complete N','Incomplete N'],missing_rows,long=True)
    runs=catalog_csv(ORIGINAL+'report/CBraMod_per_run_descriptive.csv','Supplementary: initial CBraMod run-level descriptive results','C2')
    runrows=[]
    for _,r in runs[runs.dataset=='pooled'].sort_values(['fold','seed','point']).iterrows():
        val=lambda col:'NA' if pd.isna(r[col]) else f'{r[col]:.3f}'
        runrows.append([int(r.fold),int(r.seed),r.point,r.status,int(r.observed_subjects),
                        val('G_BA_mean_percent_or_pp'),val('own_gain_median_percent_or_pp'),val('upper_bound_median_percent_or_pp')])
    table_file('budget_CB_runs',['Fold','Seed','Budget','Status','N','G mean','Delta median','U median'],runrows,long=True)
    all_validation=catalog_csv(ORIGINAL+'report/endpoint_validation_including_failed_runs.csv','Supplementary: initial budget validation inventory','C1/C2')
    failed=all_validation[(all_validation.model=='CBraMod')&(all_validation.run_status=='failed')]
    rows=[[int(r.fold),int(r.seed),r.point,'Pooled' if r.dataset=='pooled' else NAMES[r.dataset],f'{r.CE:.4f}',f'{100*r.accuracy:.2f}',f'{100*r.BA:.2f}'] for _,r in failed.iterrows()]
    table_file('budget_CB_failed_validation',['Fold','Seed','Budget','Dataset','CE',r'Accuracy (\%)',r'BA (\%)'],rows,long=True)
    # Direct protocol constants are source-linked; they are not newly chosen thresholds.
    cfg='configs/population_budget.yaml';source(cfg)
    c=yaml.safe_load((ROOT/cfg).read_text(encoding='utf-8'))
    def walk(v,p=''):
        if isinstance(v,dict):
            for k,x in v.items():walk(x,p+'/'+str(k))
        elif isinstance(v,list):
            for i,x in enumerate(v):walk(x,p+'/'+str(i))
        elif isinstance(v,(int,float)) and not isinstance(v,bool):add('budget.protocol'+p,v,'protocol',cfg,p,p,'recorded budget configuration','Methods: budget protocol')
    walk(c)
    CHECKS.append({'check':'budget_final_three_models_original_cohort_after_registered_retry','models':['CBraMod','REVE','LaBraM'],
                   'CBraMod':'24 original trajectories plus one registered precision exception; initial failure retained','new_tests':False})
