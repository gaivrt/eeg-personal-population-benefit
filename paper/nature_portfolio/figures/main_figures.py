"""Main figures and descriptive supplements, reconstructed from saved results."""
from common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def protocol():
    c=read('reports/stage_c/report/subject_results.csv')
    assert {'M_film','M_lora','B3_film','B3_lora','B4_film','B4_lora'} <= set(c.condition)
    read_json('reports/stage_c/report/run_manifest.json')
    read_json('reports/stage_w/report/run_manifest.json')
    m=read_json('reports/stage_m/report/endpoints.json');assert m['strongest_control']=='R1'
    fig=plt.figure(figsize=(7.2,3.65));ax=fig.add_axes([.025,.025,.95,.95])
    ax.set(xlim=(0,10),ylim=(0,5));ax.axis('off');contained=[]
    for x,label in [(0.12,'EVALUATION SPLIT'),(3.5,'ATTRIBUTION CONTROLS'),(7.15,'TWO DISTINCT OUTCOMES')]:
        ax.text(x,4.78,label,fontsize=8,fontweight='bold',color=BLUE)
    def box(x,y,w,h,title,body,tint='#F3F6F9'):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.07',
                            facecolor=tint,edgecolor='#C9D4DE',lw=.7)
        ax.add_patch(patch)
        a=ax.text(x+.14,y+h-.19,title,ha='left',va='top',fontsize=9,fontweight='bold')
        b=ax.text(x+.14,y+h-.54,body,ha='left',va='top',fontsize=8,linespacing=1.55)
        contained.extend([(patch,a),(patch,b)])
    def arrow(x,y,u,v):
        ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=9,
                                    color='#7B8D9E',lw=.9,connectionstyle='arc3,rad=0'))
    box(.12,2.87,2.75,1.58,'Source subjects','Train / validation only\nFrozen EEG backbone\n+ shared population model')
    box(.12,.96,2.75,1.58,'Held-out subject','Earlier half: labels / context\nLater half: fixed query trials')
    box(3.50,3.34,3.03,1.11,'Capacity control','Real / shuffled / default context')
    box(3.50,2.01,3.03,1.11,'Information mismatch','Own / swapped parameters')
    box(3.50,.68,3.03,1.11,'Training-budget control','Population 1× / 2× / 4×\nMeta-learning / continuation')
    box(7.15,1.61,2.70,2.84,'Report both contrasts','Δ = own − population\n\nU = own − swapped\n\nU > 0 does not imply Δ > 0', '#EDF4F3')
    arrow(2.93,3.61,3.42,3.89);arrow(2.93,1.94,3.42,2.54)
    for y,endpoint in [(3.89,3.81),(2.54,3.02),(1.21,2.21)]:arrow(6.60,y,7.07,endpoint)
    ax.text(7.15,1.22,'Donor: another held-out subject\nin the same dataset and fold.\nFit uses earlier-half labels.',
            fontsize=7.6,ha='left',va='top',linespacing=1.4,color='#586B7C')
    ax.plot([.12,9.85],[.43,.43],color='#DCE3E9',lw=.7)
    ax.text(.12,.22,'Population reference: convergence unconfirmed.   Continuation: scheduled outer updates matched.',
            fontsize=7.5,ha='left',va='center',color='#586B7C')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for patch,label in contained:
        outer=patch.get_window_extent(renderer);inner=label.get_window_extent(renderer)
        assert outer.contains(inner.x0,inner.y0) and outer.contains(inner.x1,inner.y1),label.get_text()
    save(fig,'fig1_protocol')

def gains():
    d=read('reports/stage_w_followup/report/population_strength_subjects.csv')
    check_unique(d,['start','variant','dataset','subject'])
    assert np.allclose(d.own_ba-d.b3_ba,d.own_gain)
    assert np.allclose(d.own_ba-d.swap_ba,d.upper_bound)
    cohorts=[set(map(tuple,g[['dataset','subject']].to_numpy())) for _,g in d.groupby(['start','variant'])]
    assert all(c==cohorts[0] for c in cohorts)
    CHECKS.append({'check':'B3_G_R2_same_subjects_and_paired_arithmetic','passed':True})
    base=read('reports/stage_w/report/population_absolute_subjects.csv')
    fig=plt.figure(figsize=(7.2,5.7));gs=fig.add_gridspec(2,3,height_ratios=[1,1.04],hspace=.62,wspace=.55,
                                                       left=.09,right=.985,bottom=.19,top=.85)
    for j,ds in enumerate(DATASETS[:3]):
        ax=fig.add_subplot(gs[0,j]);b=base[(base.dataset==ds)&(base.condition=='B0')]
        bmean=stat(b,'ba',f'f2.{ds}.B0','Appendix: CBraMod gain decompositionA','C1',unit='BA_percent');fact('B0'+NAMES[ds],f'f2.{ds}.B0')
        ax.axhline(bmean,color='#83909C',ls=':',lw=1,label='Task head B0')
        for k,start in enumerate(['B3','G','R2']):
            s=d[(d.dataset==ds)&(d.variant=='lora8')&(d.start==start)]
            means=[stat(s,col,f'f2.{ds}.{start}.{col}','Appendix: CBraMod gain decompositionA','C1/C2',unit='BA_percent') for col in ['b3_ba','own_ba']]
            ax.plot([k-.17,k+.17],means,color=COLORS[start],lw=1.5)
            ax.scatter(k-.17,means[0],marker='o',facecolors='white',edgecolors=COLORS[start],zorder=3)
            ax.scatter(k+.17,means[1],marker='^',color=COLORS[start],zorder=3)
            if start=='G':fact('G'+NAMES[ds],f'f2.{ds}.{start}.b3_ba')
        ax.set(xticks=range(3),xticklabels=['B3','G\nprimary','R2'],title=f'{"ABC"[j]}  {NAMES[ds]} · LoRA',ylim=(58,88))
        ax.grid(axis='y',alpha=.8)
        if j==0:ax.set_ylabel('Mean BA (%)')
    ax=fig.add_subplot(gs[1,0])
    for j,variant in enumerate(['film_offset','lora8']):
        means=[]
        for start in ['B3','G','R2']:
            s=d[(d.start==start)&(d.variant==variant)]
            key=f'f2.pooled.{start}.{variant}.gain'
            means.append(stat(s,'own_gain',key,'Appendix: CBraMod gain decompositionD;Results;Abstract','C2'));fact(start+('Film' if j==0 else 'Lora'),key)
            stat(s,'own_gain',key+'.sd','App: distributions','C2',op='sd')
            stat(s,'own_gain',key+'.median','App: distributions','C2',op='median')
            stat(s,'upper_bound',key+'.Umean','App: distributions','C2')
            stat(s,'upper_bound',key+'.Umedian','App: distributions','C2',op='median')
            stat(s,'own_gain',key+'.n','Appendix: CBraMod gain decompositionD','C2',op='count',scale=1,unit='count')
        x=np.arange(3)+(j-.5)*.22
        ax.plot(x,means,['o-','s--'][j],label=['FiLM','LoRA'][j],color=[TEAL,BLUE][j],lw=1.4,ms=4)
        for kk,(xx,y) in enumerate(zip(x,means)):ax.annotate(f'{y:.2f}',(xx,y),xytext=(0,7 if j==0 or kk==2 else -12),textcoords='offset points',ha='center',fontsize=7)
    ax.axhline(0,color='#83909C',lw=.7);ax.set(xticks=range(3),xticklabels=['B3','G','R2*'],ylim=(-.1,4.4),ylabel='Mean personal gain (pp)',title='D  Reference sensitivity')
    ax.grid(axis='y',alpha=.8)
    family_handles,family_labels=ax.get_legend_handles_labels()
    lee=read('reports/stage_w/report/diagnostic1_subject_results.csv')
    lee=lee[(lee.dataset=='Lee2019_MI')&(lee.start=='G')]
    bnci=read('reports/stage_w_followup/report/bnci_diagnostic1_subjects.csv')
    for j,(ds,ext) in enumerate([('Lee2019_MI',lee),('BNCI2014_001',bnci)],1):
        ax=fig.add_subplot(gs[1,j])
        for vi,var in enumerate(['film_offset','lora8']):
            s=ext[ext.variant==var]
            for ci,col in enumerate(['own_gain','upper_bound']):
                key=f'f2.{ds}.{var}.{col}';mean=stat(s,col,key,'Appendix: CBraMod gain decompositionE/F;Results','C2')
                fact(('Lee' if j==1 else 'BNCI')+('Film' if vi==0 else 'Lora')+('Delta' if ci==0 else 'U'),key)
                n=stat(s,col,key+'.n','Appendix: CBraMod gain decompositionE/F','C2',op='count',scale=1,unit='count')
                x=vi+(ci-.5)*.26
                if j==2:
                    vals=[]
                    for idx in s.index[s[col].notna()]: vals.append(direct(s,idx,col,key+f'.subject.{idx}','Appendix: CBraMod gain decompositionF','C2',scale=100))
                    ax.scatter(x+np.linspace(-.05,.05,len(vals)),vals,s=8,alpha=.3,color=[BLUE,COPPER][ci])
                else:
                    q=[stat(s,col,key+'.'+op,'Appendix: CBraMod gain decompositionE','C2',op=op) for op in ['q25','q75']]
                    ax.plot([x,x],q,color=[BLUE,COPPER][ci],lw=3,alpha=.25)
                ax.scatter(x,mean,s=27,marker=['o','D'][ci],color=[BLUE,COPPER][ci],label=['Δ: own − G','U: own − swapped'][ci] if vi==0 else None,zorder=4)
        ax.axhline(0,color='#83909C',lw=.7);ax.set(xticks=[0,1],xticklabels=['FiLM','LoRA'],title=f'{"EF"[j-1]}  {NAMES[ds]}',ylabel='Paired difference (pp)')
        ax.margins(x=.2,y=.16);ax.grid(axis='y',alpha=.8)
    from matplotlib.lines import Line2D
    primary_handles=[Line2D([],[],marker='o',mfc='white',mec=BLUE,color=BLUE,ls='None',label='Population start'),
                     Line2D([],[],marker='^',color=BLUE,ls='None',label='Personal fit'),
                     Line2D([],[],ls=':',color=GRAY,label='Task head B0')]
    fig.legend(handles=primary_handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=3)
    fig.legend(family_handles,family_labels,loc='lower left',bbox_to_anchor=(.06,.02),ncol=2,fontsize=8,title='Panel D',title_fontsize=8)
    h,l=ax.get_legend_handles_labels()
    fig.legend(h,l,loc='lower right',bbox_to_anchor=(.985,.02),ncol=2,fontsize=8,title='Panels E–F',title_fontsize=8)
    fig.text(.5,.008,'BNCI: Δ uses all subjects; U uses the available donors.',ha='center',va='bottom',fontsize=7,color='#586B7C')
    save(fig,'fig2_gain_decomposition')
    # Full paired distributions, separately by dataset and start.
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.6),sharey=True)
    for i,var in enumerate(['film_offset','lora8']):
        for j,ds in enumerate(DATASETS[:3]):
            ax=axes[i,j]
            for k,start in enumerate(['B3','G','R2']):
                s=d[(d.start==start)&(d.variant==var)&(d.dataset==ds)];key=f's2.{ds}.{var}.{start}'
                val=[stat(s,'own_gain',key+'.'+op,'FigS1','C2',op=op) for op in ['mean','q25','q75']]
                ax.vlines(k,val[1],val[2],lw=5,color=COLORS[start],alpha=.32);ax.scatter(k,val[0],color=COLORS[start],s=23)
            ax.axhline(0,color='#83909C',lw=.7);ax.set(xticks=range(3),xticklabels=['B3','G','R2*'],title=f'{NAMES[ds]} · '+('FiLM' if i==0 else 'LoRA'))
            ax.margins(x=.22,y=.12);ax.grid(axis='y',alpha=.8)
            if j==0:ax.set_ylabel('Own − population (pp)')
    fig.subplots_adjust(left=.09,right=.985,bottom=.10,top=.91,wspace=.23,hspace=.43);save(fig,'figS1_gain_distributions')

def context():
    d=read('reports/stage_c/report/subject_results.csv');check_unique(d,['condition','dataset','subject'])
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.05),gridspec_kw={'width_ratios':[1,1.15,.8]})
    ax=axes[0]
    for j,family in enumerate(['film','lora']):
        values=[]
        for cond in ['B3_'+family,'B4_'+family,'M_'+family]:
            s=d[d.condition==cond];key='f3.context.'+cond;values.append(stat(s,'ba',key,'Fig4A','C3',unit='BA_percent'))
        ax.plot(range(3),values,['o-','s--'][j],label=['FiLM','LoRA'][j],color=[TEAL,BLUE][j],ms=4)
    ax.set(xticks=range(3),xticklabels=['Default','Shuffled','Real'],ylabel='Mean BA (%)',title='A  Context controls')
    ax.margins(x=.12,y=.12);ax.grid(axis='y',alpha=.8)
    p=read('reports/stage_w/neighbour_audit/pooled_descriptive.csv')
    ax=axes[1];labels=[]
    for i,(variant,kind) in enumerate([(v,k) for v in ['film_offset','lora8'] for k in ['rest','task']]):
        row=p[(p.start=='G')&(p.variant==variant)&(p.source==kind)&(p.k==1)];assert len(row)==1
        labels.append(('FiLM' if variant=='film_offset' else 'LoRA')+'\n'+kind)
        for j,col in enumerate(['nearest_minus_start_mean_pp','random_minus_start_mean_pp']):
            v=direct(row,row.index[0],col,f'f3.G.{variant}.{kind}.{col}','Fig4B','C3')
            ax.bar(i+(j-.5)*.31,v,width=.28,color=[TEAL,GRAY][j],label=['Nearest','Random'][j] if i==0 else None,edgecolor='white',lw=.3)
    ax.set(xticks=range(4),xticklabels=labels,ylabel='Mean gain over G (pp)',title='B  Donor selection')
    ax.axhline(0,color='#83909C',lw=.75);ax.set_ylim(ax.get_ylim()[0]*1.08,.25);ax.grid(axis='y',alpha=.8)
    lee=read('reports/stage_w/lee_result_audit/neighbour_subject_results.csv');ax=axes[2]
    for i,var in enumerate(['film_offset','lora8']):
        s=lee[lee.variant==var]
        for j,col in enumerate(['nearest_gain','random_gain']):
            value=stat(s,col,f'f3.Lee.{var}.{col}','Fig4C','C3');ax.bar(i+(j-.5)*.31,value,width=.28,color=[TEAL,GRAY][j],edgecolor='white',lw=.3)
    ax.set(xticks=[0,1],xticklabels=['FiLM','LoRA'],title='C  Lee rest',ylabel='Mean gain over G (pp)')
    ax.axhline(0,color='#83909C',lw=.75);ax.set_ylim(ax.get_ylim()[0]*1.08,.5);ax.grid(axis='y',alpha=.8)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper left',bbox_to_anchor=(.08,1),ncol=2,fontsize=8)
    h,l=axes[1].get_legend_handles_labels();fig.legend(h,l,loc='upper right',bbox_to_anchor=(.97,1),ncol=2,fontsize=8)
    fig.subplots_adjust(left=.09,right=.985,bottom=.23,top=.76,wspace=.62)
    save(fig,'fig3_unlabeled_context')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),sharey=True)
    for j,start in enumerate(['B3','G']):
        ax=axes[j]
        for i,(var,kind) in enumerate([(v,k) for v in ['film_offset','lora8'] for k in ['rest','task']]):
            s=p[(p.start==start)&(p.variant==var)&(p.source==kind)&(p.k==3)]
            for b,col in enumerate(['nearest_minus_start_mean_pp','random_minus_start_mean_pp']):
                value=direct(s,s.index[0],col,f's3.{start}.{var}.{kind}.{col}','FigS2','C3')
                ax.bar(i+(b-.5)*.3,value,width=.28,color=[TEAL,GRAY][b],label=['Nearest','Random'][b] if i==0 else None,edgecolor='white',lw=.3)
        ax.set(xticks=range(4),xticklabels=labels,title=start+' · three-donor average',ylabel='Mean gain over population (pp)' if j==0 else '')
        ax.axhline(0,color='#83909C',lw=.75);ax.margins(y=.16);ax.grid(axis='y',alpha=.8)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,1),ncol=2)
    fig.subplots_adjust(left=.10,right=.985,bottom=.23,top=.76,wspace=.18)
    save(fig,'figS2_neighbors_k3')

def fewshot():
    w=read('reports/stage_w/report/few_shot_subject_results.csv')
    b=read('reports/stage_w_followup/report/bnci_few_shot_subjects.csv');b['start']='G'
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.5),sharey=True)
    for ax,ds in zip(axes.flat,DATASETS):
        d=b if ds==DATASETS[-1] else w
        for start in ['B3','G']:
            s=d[(d.dataset==ds)&(d.start==start)]
            if s.empty:continue
            check_unique(s,['shots','dataset','subject'])
            xs=sorted(s.shots.unique());ys=[]
            for n in xs:
                t=s[s.shots==n];key=f'f4.{ds}.{start}.n{n}';ys.append(stat(t,'gain',key,'Appendix: CBraMod few-shot;Results','C2'))
                stat(t,'gain',key+'.sd','App: few-shot distributions','C2',op='sd')
                stat(t,'gain',key+'.n','Fig4','C2',op='count',scale=1,unit='count')
                if ds in DATASETS[3:]:fact(NAMES[ds]+'Shot'+str(n),key)
            # The zero reference is an identity, not a new empirical observation.
            ax.plot([0]+xs,[0]+ys,marker='o' if start=='G' else 's',ms=3,lw=1.4,ls='-' if start=='G' else '--',color=COLORS[start],label=start)
            if start=='G':
                t=s[s.shots==xs[0]];full=stat(t,'full_gain',f'f4.{ds}.full','Fig4','C2')
                ax.axhline(full,color=COLORS['G'],ls=':',lw=1,label='Full earlier half')
        ax.axhline(0,color='#83909C',lw=.7);ax.set(title=NAMES[ds],xlabel='Target labels (total)',xticks=[0,10,20,40] if ds in DATASETS[1:3] else [0,10,20]);ax.margins(x=.08,y=.12);ax.grid(axis='y',alpha=.8)
    for ax in axes[:,0]:ax.set_ylabel('Mean personal gain (pp)')
    h,l=axes[0,0].get_legend_handles_labels();axes[1,2].axis('off');axes[1,2].legend(h,l,loc='center',fontsize=8.5)
    fig.subplots_adjust(left=.09,right=.985,bottom=.12,top=.92,wspace=.24,hspace=.61);save(fig,'fig4_few_shot')
    c=read('reports/stage_c3/report/few_shot_subject_results.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),sharey=True)
    for ax,ds in zip(axes,DATASETS[:3]):
        for v,var in enumerate(['film_offset','lora8']):
            s=c[(c.dataset==ds)&(c.variant==var)];xs=sorted(s.shots.unique());values=[]
            for n in xs:values.append(stat(s[s.shots==n],'gain',f's4.{ds}.{var}.n{n}','FigS3','C2'))
            ax.plot([0]+xs,[0]+values,['o-','s--'][v],label=['FiLM','LoRA'][v],color=[TEAL,BLUE][v],ms=4)
        ax.axhline(0,color='#83909C',lw=.7);ax.set(title=NAMES[ds],xlabel='Target labels (total)');ax.grid(axis='y',alpha=.8);ax.margins(x=.08,y=.15)
    axes[0].set_ylabel('Mean own − B3 (pp)')
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,1),ncol=2)
    fig.subplots_adjust(left=.09,right=.985,bottom=.22,top=.76,wspace=.24);save(fig,'figS3_full_few_shot')

def meta():
    d=read('reports/stage_m/report/subject_results.csv');check_unique(d,['method','shots','dataset','subject'])
    methods=['R0','R1','R2','M1','M2'];fig,axes=plt.subplots(1,3,figsize=(7.2,2.95),gridspec_kw={'width_ratios':[1.1,1,.85]})
    ba={};ax=axes[0]
    for j,n in enumerate([0,10]):
        ys=[]
        for method in methods:
            s=d[(d.method==method)&(d.shots==n)];key=f'f5.{method}.n{n}.ba';v=stat(s,'ba',key,'Fig5A;Results','C4',unit='BA_percent');ys.append(v);ba[(method,n)]=v;fact(method+'N'+str(n),key)
        ax.plot(range(5),ys,['o-','s--'][j],color=[GRAY,BLUE][j],label=['No target labels','After calibration'][j],ms=4)
    ax.set(xticks=range(5),xticklabels=methods,ylabel='Mean BA (%)',title='A  Decoder performance')
    ax.margins(x=.09,y=.15);ax.grid(axis='y',alpha=.8)
    ax=axes[1]
    for i,method in enumerate(methods):
        s=d[(d.method==method)&(d.shots==10)];key=f'f5.{method}.adapt';v=stat(s,'adapt_gain',key,'Fig5B;Results','C4');fact(method+'Adapt',key)
        z=d[(d.method==method)&(d.shots==0)].set_index(['dataset','subject']);q=s.set_index(['dataset','subject']);assert np.allclose((q.ba-z.ba).sort_index(),q.adapt_gain.sort_index())
        ax.bar(i,v,width=.62,color=COPPER if method=='R2' else BLUE,edgecolor='white',lw=.3)
    ax.set(xticks=range(5),xticklabels=methods,ylabel='Mean adaptation gain (pp)',title='B  Calibration increment')
    ax.axhline(0,color='#83909C',lw=.75);ax.margins(y=.18);ax.grid(axis='y',alpha=.8)
    ax=axes[2]
    for i,method in enumerate(['M1','M2']):
        for j,(name,n) in enumerate([('Initial',0),('Final',10)]):
            m=d[(d.method==method)&(d.shots==n)].set_index(['dataset','subject']);r=d[(d.method=='R2')&(d.shots==n)].set_index(['dataset','subject']);assert m.index.equals(r.index)
            diff=(m.ba-r.ba)*100;key=f'f5.{method}.R2.{name}'
            value=add(key,diff.mean(),'pp',d.attrs['source'],','.join(pd.concat([m,r])['_record'].astype(str)),'ba',f'mean(100 * paired {method}(n={n}) - R2(n={n})); paper/figures/main_figures.py','Fig5C','C4')
            ax.bar(i+(j-.5)*.32,value,width=.29,color=[GRAY,BLUE][j],label=name if i==0 else None,edgecolor='white',lw=.3)
            fact(method+'VsR2'+name,key)
        inc=(ba[(method,10)]-ba[(method,0)])-(ba[('R2',10)]-ba[('R2',0)])
        assert np.isclose(ba[(method,10)]-ba[('R2',10)],ba[(method,0)]-ba[('R2',0)]+inc)
    CHECKS.append({'check':'meta_paired_n10_minus_n0_and_mean_decomposition','passed':True})
    ax.axhline(0,color='#83909C',lw=.75);ax.set(xticks=[0,1],xticklabels=['M1','M2'],ylabel='Mean gain over R2 (pp)',title='C  Relative to R2')
    ax.set_ylim(ax.get_ylim()[0]*1.12,.035);ax.grid(axis='y',alpha=.8)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper left',bbox_to_anchor=(.07,1),ncol=2,fontsize=8)
    h,l=axes[2].get_legend_handles_labels();fig.legend(h,l,loc='upper right',bbox_to_anchor=(.985,1),ncol=2,fontsize=8)
    fig.subplots_adjust(left=.09,right=.985,bottom=.18,top=.76,wspace=.64)
    save(fig,'fig5_meta_decomposition')

def trajectories():
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.6),sharex=True)
    for j,(name,path) in enumerate([('Initial cohort','reports/stage_w/report/starter_curves.csv'),('Lee','reports/stage_w/report/Lee2019_MI_curves.csv'),('BNCI','reports/stage_w_followup/report/bnci_training_curves.csv')]):
        d=read(path);d=d[d.dataset=='pooled']
        check_unique(d,['method','additional_step','role','fold','seed'])
        for i,method in enumerate(['m_film','m_lora']):
            ax=axes[i,j]
            for (role,fold,seed),s in d[d.method==method].groupby(['role','fold','seed']):
                s=s.sort_values('additional_step');ys=[]
                for idx in s.index:
                    key=f's5.{name}.{method}.{role}.{fold}.{seed}.{s.at[idx,"additional_step"]}'
                    ys.append(direct(s,idx,'subject_mean_ce',key,'FigS4;Appendix: convergence','C2',unit='cross_entropy'))
                    direct(s,idx,'additional_step',key+'.step','FigS4','C2',unit='count')
                ax.plot(s.additional_step,ys,color=BLUE if role=='train' else COPPER,lw=.65,alpha=.30)
            ax.set(title=name+' · '+('FiLM' if i==0 else 'LoRA'),ylabel='Subject-mean CE' if j==0 else '',xlabel='Additional updates' if i==1 else '')
            ax.grid(axis='y',alpha=.6);ax.margins(x=.05,y=.07)
    axes[0,0].plot([],[],color=BLUE,label='Train');axes[0,0].plot([],[],color=COPPER,label='Validation')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,1),ncol=2)
    fig.subplots_adjust(left=.09,right=.985,bottom=.13,top=.85,wspace=.28,hspace=.45);save(fig,'figS4_training_trajectories')

def build():
    protocol();gains();context();fewshot();meta();trajectories()
