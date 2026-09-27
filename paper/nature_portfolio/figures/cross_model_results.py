"""Present completed X1 results; no model execution or statistical tests."""
from common import *
from tables import catalog_csv, catalog_json

MODELS=['CBraMod','REVE','LaBraM']
PALETTE=MODEL_COLORS

def build():
    effects=catalog_csv('reports/stage_x/three_model_effects.csv','Fig2;Results;claims ledger','C1/C2')
    few=catalog_csv('reports/stage_x/three_model_few_shot.csv','Fig3;Results','C2')
    consistency=read('reports/stage_x/three_model_consistency.csv')
    for path in ['docs/cross_model_summary.md','docs/cross_model_protocol.md','docs/cross_model_tests.md','docs/population_budget_protocol.md']:
        source(path)
    check_unique(effects,['model','dataset']);check_unique(few,['model','dataset','shots'])
    assert len(effects)==9 and (effects.own_minus_G_mean_pp>0).all() and (effects.own_minus_swap_mean_pp>0).all()
    assert (effects.own_minus_swap_median_pp>0).all()
    for _,r in consistency.iterrows():
        s=effects[effects.dataset==r.dataset]
        for name in ['G_minus_B0','own_minus_G','own_minus_swap']:
            label='consistent_positive' if (s[name+'_mean_pp']>0).all() else 'inconsistent'
            assert r[name]==label
        assert r.few_shot_stability==('consistent_stable' if s.few_shot_stable.all() else 'inconsistent')
    for alias,value,col in [('CrossModels',effects.model.nunique(),'model'),('CrossDatasets',effects.dataset.nunique(),'dataset'),('CrossCells',len(effects),'model,dataset')]:
        key='x.count.'+alias
        add(key,value,'count',effects.attrs['source'],'all rows',col,'count distinct models/datasets or unique pairs; paper/figures/cross_model_results.py','Abstract;Results','C2');fact(alias,key)
    for alias,col,op in [('CrossDeltaMin','own_minus_G_mean_pp','min'),('CrossDeltaMax','own_minus_G_mean_pp','max'),('CrossUMin','own_minus_swap_mean_pp','min'),('CrossUMax','own_minus_swap_mean_pp','max')]:
        key='x.range.'+alias;stat(effects,col,key,'Abstract;Results','C2',op=op,scale=1);fact(alias,key)
    for _,r in effects.iterrows():
        for prefix,col in [('Delta','own_minus_G_mean_pp'),('U','own_minus_swap_mean_pp'),('Pop','G_minus_B0_mean_pp')]:
            fact('X'+r.model+NAMES[r.dataset]+prefix,f'source:{effects.attrs["source"]}:{int(r._record)}:{col}')
    rows=[]
    for ds in DATASETS[:3]:
        for model in MODELS:
            r=effects[(effects.model==model)&(effects.dataset==ds)].iloc[0]
            rows.append([NAMES[ds],model,'listed' if r.pretraining_source.startswith('seen') else 'unlisted']+[f'{r[c]:+.2f}' for c in ['G_minus_B0_mean_pp','own_minus_G_mean_pp','own_minus_swap_mean_pp','own_minus_swap_median_pp']])
    table_file('cross_model_effects',['Dataset','Model','Source','G-B0 mean','Own-G mean','Own-swap mean','Own-swap median'],rows,long=True)
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.75))
    for j,(col,title) in enumerate([('own_minus_G_mean_pp','A  Personal benefit'),('own_minus_swap_mean_pp','B  Personal specificity'),('G_minus_B0_mean_pp','C  Population adaptation')]):
        ax=axes[j]
        for m,model in enumerate(MODELS):
            s=effects[effects.model==model].set_index('dataset').loc[DATASETS[:3]]
            xs=np.arange(3)+(m-1)*.21
            ax.scatter(xs,s[col],color=PALETTE[m],marker=['o','s','^'][m],s=28,label=model,zorder=3,edgecolor='white',linewidth=.45)
            for x,(_,r) in zip(xs,s.iterrows()):
                if r.pretraining_source.startswith('seen'):ax.annotate('*',(x,r[col]),xytext=(0,7),textcoords='offset points',fontsize=8.5,color=INK,ha='center',va='bottom')
        ax.axhline(0,color='#83909C',lw=.75)
        ax.set(title=title,xticks=range(3),xticklabels=['PhysioNet','Dreyer','Cho'],
               ylabel=['Mean own − G (pp)','Mean own − swapped (pp)','Mean G − B0 (pp)'][j],xlim=(-.5,2.5))
        lo=min(0,effects[col].min());hi=effects[col].max();span=hi-lo
        ax.set_ylim(lo-.08*span,hi+.24*span);ax.grid(axis='y',alpha=.8)
        ax.tick_params(axis='x',labelsize=7.5)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1),ncol=3)
    fig.subplots_adjust(left=.085,right=.985,bottom=.19,top=.76,wspace=.55)
    save(fig,'fig2_cross_model_attribution')
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),sharey=True)
    stable_rows=[]
    for ax,ds in zip(axes,DATASETS[:3]):
        for m,model in enumerate(MODELS):
            s=few[(few.model==model)&(few.dataset==ds)].sort_values('shots')
            observed=bool((s.mean_pp>=0).all() and (np.diff(s.mean_pp)>=0).all())
            e=effects[(effects.model==model)&(effects.dataset==ds)].iloc[0]
            assert observed==bool(e.few_shot_stable)
            stable_rows.append([NAMES[ds],model,'Stable' if observed else 'Not stable'])
            ax.plot(s.shots,s.mean_pp,marker=['o','s','^'][m],ms=4,color=PALETTE[m],ls='--' if model=='CBraMod' else '-',label=model)
            if e.pretraining_source.startswith('seen'):
                ax.annotate('*',(s.shots.iloc[-1],s.mean_pp.iloc[-1]),xytext=(7,0),textcoords='offset points',fontsize=8.5,ha='left',va='center')
        ax.axhline(0,color='#83909C',lw=.75)
        ax.set(title=NAMES[ds],xlabel='Target labels (total)',xticks=sorted(few[few.dataset==ds].shots.unique()))
        ax.margins(x=.15);ax.grid(axis='y',alpha=.8)
    lo=min(0,few.mean_pp.min());hi=few.mean_pp.max();span=hi-lo
    axes[0].set(ylim=(lo-.12*span,hi+.17*span),ylabel='Mean own − G (pp)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1),ncol=3)
    fig.subplots_adjust(left=.09,right=.985,bottom=.22,top=.76,wspace=.24)
    save(fig,'fig3_cross_model_few_shot')
    table_file('cross_model_stability',['Dataset','Model','Descriptive stability'],stable_rows)
    # Two logically distinct contrasts: a saved subject-level illustration, no new test.
    w=read('reports/stage_w/report/diagnostic1_subject_results.csv')
    s=w[(w.start=='G')&(w.variant=='lora8')&(w.dataset=='Cho2017')&(w.subject==36)]
    assert len(s)==1
    for alias,col in [('ExampleG','b3_ba'),('ExampleOwn','own_ba'),('ExampleSwap','swap_ba'),('ExampleDelta','own_gain'),('ExampleU','upper_bound')]:
        key='x.example.'+alias;direct(s,s.index[0],col,key,'Protocol: individual illustration;claims ledger','C2/C6',unit='BA_percent' if col.endswith('_ba') else 'pp',scale=100);fact(alias,key)
    original=[];gates=[];convergence=[]
    for model in MODELS[1:]:
        base=f'reports/stage_x/{model}/report/'
        d=catalog_csv(base+'core_summary.csv','Appendix: X1 comparisons','C1/C2')
        subjects=read(base+'subject_means.csv');check_unique(subjects,['dataset','subject'])
        for _,r in d.iterrows():
            x=effects[(effects.model==model)&(effects.dataset==r.dataset)].iloc[0]
            for contrast in ['G_minus_B0','own_minus_G','own_minus_swap']:
                assert np.isclose(x[contrast+'_mean_pp'],r[contrast+'_mean_pp'])
                original.append([model,NAMES[r.dataset],contrast.replace('_',' '),int(r[contrast+'_N'])]+[f'{r[contrast+c]:.3g}' if 'p_raw' in c else f'{r[contrast+c]:.3f}' for c in ['_mean_pp','_median_pp','_SD_pp','_p_raw_descriptive']])
            s=subjects[subjects.dataset==r.dataset]
            assert np.isclose(100*s.own_gain.mean(),r.own_minus_G_mean_pp)
            assert np.isclose(100*s.upper_bound.mean(),r.own_minus_swap_mean_pp)
        fs=catalog_csv(base+'few_shot_summary.csv','Appendix: X1 few-shot','C2')
        rows=[]
        for _,r in fs.iterrows():
            rows.append([NAMES[r.dataset],int(r.shots),int(r.N)]+[f'{r[c]:.3g}' if c.startswith('p_') else f'{r[c]:.3f}' for c in ['mean_pp','median_pp','SD_pp','drop_gt_2pp_fraction','p_raw_descriptive']])
        table_file('cross_few_'+model,['Dataset','Labels','N','Mean','Median','SD','Decline fraction','Raw p'],rows,long=True)
        g=catalog_json(base+'diagnostic1_gate.json','Appendix: X1 original gate','C2')
        gates.append([model,g['N'],f'{g["median_pp"]:.2f}',f'{g["p_Holm"]:.3g}','Pass' if g['pass'] else 'Fail'])
        for alias,col in [('Median','median_pp'),('PH','p_Holm')]:fact('X'+model+alias,'json:'+base+'diagnostic1_gate.json:/'+col)
        d=catalog_csv(base+'training_selection.csv','Appendix: X1 convergence','C1/C2')
        assert len(d)==25 and not d.plateau_observed.any()
        for _,r in d.iterrows():convergence.append([model,int(r.fold),int(r.seed),int(r.initial_selected_step),int(r.selected_additional_step),int(r.additional_steps_run),r.stopped_by,'No'])
    table_file('cross_original_comparisons',['Model','Dataset','Contrast','N','Mean','Median','SD','Raw p'],original,widths='llp{1.5in}rrrrr',long=True)
    table_file('cross_original_gates',['Model','N','Median U (pp)','Original Holm p','Decision'],gates)
    table_file('cross_convergence',['Model','Fold','Seed','Initial','G selected','Extra run','Stop','Plateau'],convergence,long=True)
    CHECKS.append({'check':'X1_summary_against_saved_subject_means_and_core_tables','combinations':len(effects),'new_tests':False,'passed':True})
    CHECKS.append({'check':'X1_fixed_descriptive_stability','passed':True,'source':consistency.attrs['source']})
