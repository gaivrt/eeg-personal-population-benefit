"""Subject bootstrap of saved scores only; never loads models or runs inference.

All input tables already average seeds within (dataset, subject). We enforce
one row per subject/condition, preserve paired contrasts and archive each vector.
"""
from common import *

MODELS = ['CBraMod', 'REVE', 'LaBraM']
B = 20000
SEED = 20260927
SUMMARIES, VECTORS = [], []


def estimate(key, frame, column, *, group, model='', dataset='Pooled', condition='',
             paths=None, paired_column=None):
    """Bootstrap a subject statistic; for budget change bootstrap paired medians."""
    check_unique(frame, ['dataset', 'subject'])
    cols = [column] + ([paired_column] if paired_column else [])
    frame = frame.dropna(subset=cols).sort_values(['dataset', 'subject'])
    x = frame[column].to_numpy(float) * 100
    assert len(x) and np.isfinite(x).all()
    y = frame[paired_column].to_numpy(float) * 100 if paired_column else None
    # A stable seed per contrast makes rebuilds independent of call ordering.
    rng = np.random.default_rng(SEED + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    idx = rng.integers(0, len(x), size=(B, len(x)))
    row = dict(key=key, group=group, model=model, dataset=dataset,
               condition=condition, n=len(x), resamples=B)
    paths = paths or [frame.attrs['source']]
    for statistic, fn in [('mean', np.mean), ('median', np.median)]:
        value = fn(x) - (fn(y) if y is not None else 0)
        draws = fn(x[idx], axis=1)
        if y is not None:
            draws -= fn(y[idx], axis=1)
        low, high = np.quantile(draws, [.025, .975])
        for suffix, v in [('', value), ('_lo', low), ('_hi', high)]:
            name = statistic + suffix
            row[name] = float(v)
            unit = 'percent' if column == 'G_BA' and paired_column is None else 'pp'
            add('bootstrap.' + key + '.' + name, v, unit, paths,
                'eligible (dataset, subject); see bootstrap_subject_vectors.csv',
                ','.join(cols),
                f'{B} percentile subject bootstrap; seed {SEED}+sha256(key)[:8]; '
                f'seeds averaged within subject before paired contrast; {name}; '
                + ('paired difference of budget statistics' if y is not None else 'statistic of within-subject differences'),
                'Main figures and comparisons; Supplementary uncertainty tables')
    for j, (_, r) in enumerate(frame.iterrows()):
        VECTORS.append(dict(key=key, dataset=r.dataset, subject=r.subject,
                            value_pp=float(x[j]), paired_value_pp=float(y[j]) if y is not None else None))
    SUMMARIES.append(row)
    return row


def dot(ax, x, row, color, statistic='median', label=None, marker='o'):
    # Draw interval separately: percentile intervals need not contain the estimate.
    ax.vlines(x, row[statistic+'_lo'], row[statistic+'_hi'], color=color, lw=1.15)
    ax.plot(x, row[statistic], marker=marker, color=color, ms=4, label=label, ls='none')


def decorate(ax, title, ylabel=None):
    ax.set_title(title)
    if ylabel: ax.set_ylabel(ylabel)
    ax.axhline(0, color=GRAY, lw=.7)
    ax.grid(axis='y', alpha=.7)
    ax.margins(x=.12, y=.14)


def build():
    core, few = {}, {}
    effects = read('reports/stage_x/three_model_effects.csv')
    cbpath = 'reports/stage_w/report/diagnostic1_subject_results.csv'
    cb = read(cbpath)
    cb = cb[(cb.start=='G') & (cb.variant=='lora8') & cb.dataset.isin(DATASETS[:3])].copy()
    poppath = 'reports/stage_w/report/population_absolute_subjects.csv'
    pop = read(poppath).pivot(index=['dataset','subject'], columns='condition', values='ba')
    cb = cb.merge(pop[['B0','G_lora']], on=['dataset','subject'], validate='one_to_one')
    cb['population_gain'] = cb.G_lora - cb.B0
    core['CBraMod'] = cb
    fs = read('reports/stage_w/report/few_shot_subject_results.csv')
    few['CBraMod'] = fs[(fs.start=='G') & fs.dataset.isin(DATASETS[:3])].copy()
    for model in MODELS[1:]:
        d = read(f'reports/stage_x/{model}/report/subject_means.csv')
        d['population_gain'] = d.G_BA-d.B0_BA
        core[model] = d
        few[model] = read(f'reports/stage_x/{model}/report/few_shot_subject_means.csv')
    records = {}
    for model in MODELS:
        paths = [cbpath,poppath] if model=='CBraMod' else [core[model].attrs['source']]
        for ds in DATASETS[:3]:
            d = core[model][core[model].dataset==ds]
            e = effects[(effects.model==model)&(effects.dataset==ds)].iloc[0]
            for col, name, old in [('own_gain','Personal benefit','own_minus_G_mean_pp'),
                                   ('upper_bound','Personal specificity','own_minus_swap_mean_pp'),
                                   ('population_gain','Population gain','G_minus_B0_mean_pp')]:
                key = f'core.{model}.{NAMES[ds]}.{col}'
                row = estimate(key,d,col,group='Core',model=model,dataset=NAMES[ds],condition=name,paths=paths)
                assert np.isclose(row['mean'],e[old],atol=1e-7), (key,row['mean'],e[old])
                records[model,ds,col] = row
    transferred = core['LaBraM'][core['LaBraM'].dataset=='Cho2017'].copy()
    transferred['swap_gain'] = transferred.swap_ba - transferred.G_BA
    assert np.allclose(transferred.swap_gain, transferred.own_gain-transferred.upper_bound)
    estimate('core.LaBraM.Cho.swap_gain', transferred, 'swap_gain', group='Core',
             model='LaBraM', dataset='Cho', condition='Exchanged minus population',
             paths=['reports/stage_x/LaBraM/report/subject_means.csv'])
    for statistic in ['median','mean']:
        fig, axes = plt.subplots(1,3,figsize=(7.2,3.25))
        for j,(col,title) in enumerate([('own_gain','A  Personal benefit'),('upper_bound','B  Personal specificity'),('population_gain','C  Population gain')]):
            ax=axes[j]
            for m,model in enumerate(MODELS):
                for i,ds in enumerate(DATASETS[:3]):
                    x=i+(m-1)*.22; row=records[model,ds,col]
                    dot(ax,x,row,MODEL_COLORS[m],statistic,label=model if i==0 else None,marker=['o','s','^'][m])
                    listed=effects[(effects.model==model)&(effects.dataset==ds)].iloc[0].pretraining_source.startswith('seen')
                    if listed: ax.annotate('*',(x,row[statistic+'_hi']),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
            decorate(ax,title,statistic.title()+' difference (pp)')
            ax.set(xticks=range(3),xticklabels=['PhysioNet','Dreyer','Cho'],xlim=(-.5,2.5))
            ax.margins(y=.24)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3)
        fig.subplots_adjust(left=.085,right=.985,bottom=.18,top=.77,wspace=.52)
        save(fig,'fig2_attribution_'+statistic+'_ci')

    bud=read('reports/population_budget/retry1/retained/report/subject_seed_means.csv')
    br={}
    for model in MODELS:
        for point in ['1x','2x','4x']:
            s=bud[(bud.model==model)&(bud.point==point)]
            assert len(s)==235
            for col,title in [('own_gain','Personal benefit'),('upper_bound','Personal specificity'),('G_BA','Population BA')]:
                br[model,point,col]=estimate(f'budget.{model}.{point}.{col}',s,col,group='Population budget',model=model,condition=point+' '+title)
        wide=bud[bud.model==model].pivot(index=['dataset','subject'],columns='point',values='own_gain').reset_index()
        change=estimate(f'budget.{model}.4x_minus_1x',wide,'4x',paired_column='1x',group='Budget change',model=model,condition='4x minus 1x',paths=[bud.attrs['source']])
        assert change['median']<0
    for statistic in ['median','mean']:
        fig,axes=plt.subplots(1,3,figsize=(7.2,3.1),sharey=True)
        for m,model in enumerate(MODELS):
            ax=axes[m]
            for col,color,marker,label in [('own_gain',BLUE,'o','Own - population'),('upper_bound',COPPER,'D','Own - exchanged')]:
                rows=[br[model,p,col] for p in ['1x','2x','4x']]
                ax.plot([1,2,4],[r[statistic] for r in rows],color=color,lw=1.2)
                for x,r in zip([1,2,4],rows):dot(ax,x,r,color,statistic,label=label if x==1 else None,marker=marker)
            decorate(ax,model,statistic.title()+' difference (pp)' if m==0 else None)
            ax.set(xticks=[1,2,4],xticklabels=['1×','2×','4×'],xlabel='Population update budget')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2)
        fig.subplots_adjust(left=.09,right=.985,bottom=.21,top=.77,wspace=.22)
        save(fig,'fig3_budget_'+statistic+'_ci')

    fr={}
    for model in MODELS:
        for (ds,shots),s in few[model].groupby(['dataset','shots']):
            fr[model,ds,int(shots)]=estimate(f'few.{model}.{NAMES[ds]}.{int(shots)}',s,'gain',group='Few labels',model=model,dataset=NAMES[ds],condition=f'{int(shots)} labels')
    for statistic in ['median','mean']:
        fig,axes=plt.subplots(1,3,figsize=(7.2,3.2),sharey=True)
        for j,ds in enumerate(DATASETS[:3]):
            ax=axes[j]
            for m,model in enumerate(MODELS):
                shots=sorted(few[model][few[model].dataset==ds].shots.unique())
                xs=np.arange(len(shots))+(m-1)*.065
                rows=[fr[model,ds,int(n)] for n in shots]
                ax.plot(xs,[r[statistic] for r in rows],color=MODEL_COLORS[m],ls='--' if m==0 else '-',lw=1.1)
                for i,(x,r) in enumerate(zip(xs,rows)):dot(ax,x,r,MODEL_COLORS[m],statistic,label=model if i==0 else None,marker=['o','s','^'][m])
            decorate(ax,NAMES[ds],statistic.title()+' own - population (pp)' if j==0 else None)
            ax.set(xticks=range(len(shots)),xticklabels=shots,xlabel='Target labels (total)')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3)
        fig.subplots_adjust(left=.10,right=.985,bottom=.22,top=.77,wspace=.23)
        save(fig,'fig4_few_'+statistic+'_ci')

    # All contextual displays show paired differences, not unpaired score bars.
    context=read('reports/stage_c/report/subject_results.csv')
    cp=context.pivot(index=['dataset','subject'],columns='condition',values='ba').reset_index()
    cr={}
    for family in ['film','lora']:
        for control in ['B4','B3']:
            cp['difference']=cp['M_'+family]-cp[control+'_'+family]
            name='True - '+('shuffled' if control=='B4' else 'default')
            cr[family,control]=estimate(f'context.{family}.{control}',cp,'difference',group='Context',model='CBraMod',condition=family.upper()+' '+name,paths=[context.attrs['source']])
    neighbour=read('reports/stage_w/neighbour_audit/subject_results.csv')
    neighbour=neighbour[(neighbour.start=='G')&(neighbour.k==1)]
    nr={}
    for (var,kind),s in neighbour.groupby(['variant','source']):
        for col in ['nearest_gain','random_gain','difference']:
            nr[var,kind,col]=estimate(f'donor.{var}.{kind}.{col}',s,col,group='Donor selection',model='CBraMod',condition=var+' '+kind+' '+col)
    for statistic in ['median','mean']:
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.35))
        for i,family in enumerate(['film','lora']):
            for j,control in enumerate(['B4','B3']):dot(axes[0],i+(j-.5)*.22,cr[family,control],[BLUE,COPPER][j],statistic,label=['True - shuffled','True - default'][j] if i==0 else None)
        axes[0].set(xticks=[0,1],xticklabels=['FiLM','LoRA'])
        decorate(axes[0],'A  Rest-conditioned adaptation',statistic.title()+' paired difference (pp)')
        labels=[]
        for i,(var,kind) in enumerate([(v,k) for v in ['film_offset','lora8'] for k in ['rest','task']]):
            labels.append(('FiLM' if var=='film_offset' else 'LoRA')+'\n'+kind)
            for j,col in enumerate(['difference','nearest_gain']):dot(axes[1],i+(j-.5)*.22,nr[var,kind,col],[BLUE,COPPER][j],statistic,label=['Nearest - random','Nearest - population'][j] if i==0 else None)
        axes[1].set(xticks=range(4),xticklabels=labels)
        decorate(axes[1],'B  Single-donor selection',statistic.title()+' paired difference (pp)')
        if statistic=='median':
            # Sub-ulp score differences must not create a misleading 1e-14 axis.
            # Preserve all underlying values; use a readable pp scale around zero.
            axes[0].set_ylim(-.5,.5)
        for ax in axes:ax.legend(loc='upper center',bbox_to_anchor=(.5,1.39),fontsize=7.2,ncol=1)
        fig.subplots_adjust(left=.09,right=.985,bottom=.23,top=.69,wspace=.38)
        save(fig,'fig5_context_'+statistic+'_ci')

    meta=read('reports/stage_m/report/subject_results.csv')
    mp=meta.pivot(index=['dataset','subject'],columns=['method','shots'],values='ba')
    mr={}
    for method in ['M1','M2']:
        for name,n in [('Before',0),('After',10)]:
            s=(mp[method,n]-mp['R2',n]).rename('difference').reset_index()
            mr[method,name]=estimate(f'meta.{method}.{name}',s,'difference',group='Meta versus continuation',model='CBraMod',condition=method+' '+name,paths=[meta.attrs['source']])
        s=((mp[method,10]-mp[method,0])-(mp['R2',10]-mp['R2',0])).rename('difference').reset_index()
        mr[method,'Increment']=estimate(f'meta.{method}.Increment',s,'difference',group='Meta versus continuation',model='CBraMod',condition=method+' Calibration increment',paths=[meta.attrs['source']])
    for statistic in ['median','mean']:
        fig,axes=plt.subplots(1,3,figsize=(7.2,3.05),sharey=True)
        for ax,name,title in zip(axes,['Before','After','Increment'],['A  Before calibration','B  After ten labels','C  Calibration increment']):
            for i,method in enumerate(['M1','M2']):dot(ax,i,mr[method,name],[BLUE,TEAL][i],statistic)
            ax.set(xticks=[0,1],xticklabels=['First-order\nMAML','Learned\ninner rates'],xlim=(-.55,1.55))
            decorate(ax,title,statistic.title()+' difference from continuation (pp)' if name=='Before' else None)
        fig.subplots_adjust(left=.105,right=.985,bottom=.22,top=.84,wspace=.20)
        save(fig,'fig6_meta_'+statistic+'_ci')

    # External results and reference sensitivity remain available with intervals.
    ex=read(cbpath)
    for var in ['film_offset','lora8']:
        s=ex[(ex.start=='G')&(ex.dataset=='Lee2019_MI')&(ex.variant==var)]
        for col in ['own_gain','upper_bound']:
            estimate(f'extension.Lee.{var}.{col}',s,col,group='External cohort',model='CBraMod',dataset='Lee',condition=var+' '+col)
    bn=read('reports/stage_w_followup/report/bnci_diagnostic1_subjects.csv')
    for var,s in bn.groupby('variant'):
        for col in ['own_gain','upper_bound']:
            estimate(f'extension.BNCI.{var}.{col}',s,col,group='External cohort',model='CBraMod',dataset='BNCI',condition=var+' '+col)

    summary=pd.DataFrame(SUMMARIES)
    summary.to_csv(TABLES/'bootstrap_summary.csv',index=False)
    pd.DataFrame(VECTORS).to_csv(TABLES/'bootstrap_subject_vectors.csv',index=False)
    clean=lambda v: 0.0 if abs(v)<.0000001 else round(float(v), 10)
    macros=[]
    for r in SUMMARIES:
        for st in ['mean','median']:
            shown=f"{clean(r[st]):.2f} [{clean(r[st+'_lo']):.2f}, {clean(r[st+'_hi']):.2f}]"
            macros.append(r'\expandafter\def\csname ci'+r['key']+st+r'\endcsname{'+shown+'}')
    (PAPER/'bootstrap_facts.tex').write_text('\n'.join(macros)+'\n',encoding='utf-8')
    core_rows=[]
    for model in MODELS:
        for ds in DATASETS[:3]:
            vals=[]
            for col in ['own_gain','upper_bound']:
                r=records[model,ds,col]
                vals.append(f"{clean(r['mean']):.2f} [{clean(r['mean_lo']):.2f}, {clean(r['mean_hi']):.2f}]")
            core_rows.append([model,NAMES[ds]]+vals)
    table_file('core_mean_intervals',['Model','Dataset','Own - population','Own - exchanged'],core_rows,widths='llrr')
    rows=[]
    for r in SUMMARIES:
        cond=r['condition'].replace('film_offset','FiLM').replace('lora8','LoRA').replace('own_gain','Own - population').replace('upper_bound','Own - exchanged').replace('M1','First-order MAML').replace('M2','Learned inner rates')
        fmt=lambda st:f"{clean(r[st]):.2f} [{clean(r[st+'_lo']):.2f}, {clean(r[st+'_hi']):.2f}]"
        rows.append([r['model'],r['dataset'],cond,r['n'],fmt('median'),fmt('mean')])
    table_file('bootstrap_intervals',['Model','Dataset','Contrast / condition','N','Median [95\\% CI]','Mean [95\\% CI]'],rows,widths='p{.65in}p{.60in}p{1.9in}rp{1.20in}p{1.20in}',long=True)
    CHECKS.append({'check':'subject_bootstrap_saved_scores_only','resamples':B,'seed':SEED,'contrasts':len(SUMMARIES),'no_new_training_or_inference':True,'dependence_adjustment':False})
