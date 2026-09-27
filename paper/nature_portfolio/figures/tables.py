"""Render preserved comparisons and metadata without running statistical tests."""
from common import *
import shutil
import yaml

def catalog_csv(path, location, claim=''):
    d=read(path)
    for col in d.select_dtypes(include='number').columns:
        if col=='_record':continue
        for i in d.index[d[col].notna()]:
            unit='p' if col.startswith('p_') or col=='p' else 'pp' if 'pp' in col else 'source_unit'
            direct(d,i,col,f'source:{path}:{int(d.at[i,"_record"])}:{col}',location,claim,unit=unit)
    return d

def catalog_json(path,location,claim=''):
    d=read_json(path)
    def walk(x,pointer=''):
        if isinstance(x,dict):
            for k,v in x.items():walk(v,pointer+'/'+k)
        elif isinstance(x,list):
            for i,v in enumerate(x):walk(v,pointer+'/'+str(i))
        elif isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x):
            col=pointer.rsplit('/',1)[-1];unit='p' if col.startswith('p_') else 'pp' if col.endswith('_pp') else 'source_unit'
            add('json:'+path+':'+pointer,x,unit,path,pointer,col,'direct JSON pointer',location,claim)
    walk(d)
    return d

def datasets():
    cards=read('reports/stage_a/baselines/data_cards.csv');channels=read_json('reports/stage_a/baselines/channels.json')
    path='configs/splits/stage_b_halves_v1.json';halves=read_json(path)['subjects'];rows=[]
    for ds in DATASETS[:3]:
        s=cards[cards.dataset==ds];n=stat(s,'subject','table1.'+ds+'.N','Table1','C1/C2',op='count',scale=1,unit='count');fact('N'+NAMES[ds],'table1.'+ds+'.N')
        hs=[(i,r) for i,r in enumerate(halves) if r['dataset']==ds]
        counts=[]
        for kind in ['fit_indices','query_indices']:
            vals=[len(r[kind]) for _,r in hs]
            for op,value in [('min',min(vals)),('max',max(vals))]:
                add(f'table1.{ds}.{kind}.{op}',value,'count',path,','.join('/subjects/'+str(i)+'/'+kind for i,_ in hs),kind,op+' length of saved index lists','Table1')
            counts.append(str(min(vals)) if min(vals)==max(vals) else f'{min(vals)}--{max(vals)}')
        for name,value in [('native',len(channels['native'][ds])),('readout',len(channels['readout_common']))]:
            add(f'table1.{ds}.{name}',value,'count','reports/stage_a/baselines/channels.json','/native/'+ds if name=='native' else '/readout_common',name,'length of channel list','Table1')
        rows.append([NAMES[ds],int(n),'single',f'{len(channels["native"][ds])} / {len(channels["readout_common"])}',' / '.join(counts),'EO'])
    lp='reports/stage_w/lee_data_audit/subjects.csv';lee=read(lp)
    for col in ['task_trials','fit_trials','query_trials','rest_seconds','EEG_channels','readout_channels']:
        vals=lee[col].unique();assert len(vals)==1
        direct(lee,lee.index[0],col,'table1.Lee.'+col,'Table1;Appendix A','C2',unit='count')
    n=stat(lee,'subject','table1.Lee.N','Table1','C2',op='count',scale=1,unit='count');fact('NLee','table1.Lee.N')
    rows.append(['Lee',int(n),'1',f'{lee.EEG_channels.iloc[0]} / {lee.readout_channels.iloc[0]}',f'{lee.fit_trials.iloc[0]} / {lee.query_trials.iloc[0]}','EO'])
    bp='reports/stage_w_followup/preparation/preprocessing.json';bn=read_json(bp)['subjects']
    add('table1.BNCI.N',len(bn),'count',bp,'/subjects','subjects','length of subject list','Table1');fact('NBNCI','table1.BNCI.N')
    for field,value,pointer in [('session',bn[0]['session'],'/subjects/0/session'),('readout',len(bn[0]['readout_channels']),'/subjects/0/readout_channels'),('fit',len(bn[0]['fit_indices']),'/subjects/0/fit_indices'),('query',len(bn[0]['query_indices']),'/subjects/0/query_indices')]:
        add('table1.BNCI.'+field,value,'count',bp,pointer,field,'direct or length of saved index list','Table1')
    # Use the observed raw-data audit, not a nominal channel count.
    rawpath='reports/stage_w_followup/preparation/rest_gdf_audit.json';raw=read_json(rawpath)['subjects']
    assert all(r['EEG_channels']==raw[0]['EEG_channels'] for r in raw)
    add('table1.BNCI.native',raw[0]['EEG_channels'],'count',rawpath,'/subjects/0/EEG_channels','EEG_channels','direct raw-data audit field','Table1')
    rows.append(['BNCI',len(bn),bn[0]['session'],f'{raw[0]["EEG_channels"]} / {len(bn[0]["readout_channels"])}',f'{len(bn[0]["fit_indices"])} / {len(bn[0]["query_indices"])}','EO'])
    table_file('datasets',['Dataset','$N$','Session','EEG / readout','Fit / query trials','Rest'],rows)
    # Each fold preserves train / validation / test populations as actually stored.
    splitrows=[]
    for cohort,sp in [('Starter','configs/splits/stage_a_v1.json'),('Lee','reports/stage_w/lee_data_audit/splits.json'),('BNCI','reports/stage_w_followup/preparation/splits.json')]:
        split=read_json(sp)
        for i,f in enumerate(split['folds']):
            vals=[]
            for role in ['train','validation','test']:
                value=add(f'splits.{cohort}.{i}.{role}',len(f[role]),'count',sp,f'/folds/{i}/{role}',role,'length of subject list','Appendix A');vals.append(int(value))
            splitrows.append(['Core cohort' if cohort=='Starter' else cohort,i]+vals)
    table_file('splits',['Cohort','Fold','Train','Validation','Test'],splitrows)

def literature():
    source('docs/literature_review/papers.csv');source('docs/literature_review/references.bib')
    rows=[
      ['Identity Trap','lin2026identity','Representation / identity audit','Task- and dataset-dependent','Identity diagnostics; attribution package not established here'],
      ['EEG-FM-Compass','liu2026compass','Benchmark and adaptation comparison','LOSO and within-subject few-shot separately','Separate protocols; no joint own/population/swapped gain decomposition'],
      ['Stacked LoRA','sarhane2026stacked','Shared and personal low-rank adapters','Subjects seen in pooled trial split','Global adapter comparison; different capacity; mismatch not reported in read text'],
      ['Lopes et al. (Scientific Reports)','lopes2026published','Personal encoders and alignment','LOSO; target calibration','Low-rank, shared-adaptation, mismatch, capacity and normalization controls'],
      ['Nguyen et al.','nguyen2026stroke','Frozen-model population LoRA','Held-out subjects','Head versus LoRA; no target personal adapter in evaluated pipeline'],
      ['ResTL','an2024restl','Rest-based generative transfer','LOSO; target rest','Noise/rest control; updates decoder; donor-rest swap not reported in read text'],
      ['Sharma et al.','sharma2024fast','Transfer learning versus MAML','Held-out subjects; few-shot labels','Ordinary transfer baseline; optimizers differ; total training budget not matched'],
      ['This study','','Gain attribution on frozen backbone','Held-out subjects; same-session split','Default/shuffled; own/swapped; outer-update continuation; separate Delta and U']]
    table_file('related_work_full',['Work','Bib key','Question','Subject / target setting','Verified control scope'],rows,
               widths='p{.8in}p{1.05in}p{1.05in}p{1.15in}p{2.15in}',long=True)
    path=TABLES/'related_work_full.tex';rendered=path.read_text(encoding='utf-8').replace('Bib key','Reference')
    for row in rows:
        if row[1]:rendered=rendered.replace(row[1],r'\citep{'+row[1]+'}')
    path.write_text(rendered,encoding='utf-8')
    compact=[['Identity Trap','Representation and identity shortcuts','Identity diagnostics'],['Compass','LOSO; separate within-subject few-shot','Adaptation benchmarks'],['Stacked LoRA','Personal branches; subjects seen during training','Global adapter'],['Lopes et al.','Held-out-subject encoders and alignment','Shared-adaptation and representation controls'],['This study','Unseen-subject classification gains','Population, mismatch, continuation']]
    table_file('related_work',['Work','Evaluation target','Control / comparison'],compact,widths='p{1.0in}p{2.9in}p{2.5in}')

def gates():
    rows=[]
    def gate(identifier,scope,rule,obs,passed):rows.append([identifier,scope,rule,obs,'Pass' if passed else 'Fail'])
    p='reports/stage_b/report/gate.json';b=catalog_json(p,'Appendix: gates','C1');gate('B-G','pooled','Median vs B0 >= 2 pp',f"median {b['observed_median_gain_pp']:.3f}",b['passed'])
    p='reports/stage_b2/report/gate.json';b=catalog_json(p,'Appendix: gates','C1')
    for r in b['variants']:gate('B2-'+r['variant'],'pooled','Median vs B0 >= 2 pp; one-sided Holm(4) vs tuned head < .05',f"median {r['median_gain_vs_b0_pp']:.3f}; pH {r['vs_head_p_holm']:.5g}",r['passed'])
    p='reports/stage_c/report/criteria.json';b=catalog_json(p,'Appendix: gates','C3')
    for r in b['methods']:gate('C-'+r['method'],'pooled','One-sided Holm(4) vs B4 and B0 < .05; recovery >= .30; drop > 2 pp <= .10',f"both p pass {r['significant_vs_B4_and_B0']}; recovery {r['recovery_fraction']:.3f}; drop {r['drop_gt_2pp_fraction']:.3f}",r['passed'])
    p='reports/stage_c2/report/gate.json';b=catalog_json(p,'Appendix: gates','C2/C3')
    for section in ['variants','diagnostic2']:
        for r in b[section]:gate('C2-'+('D1-' if section=='variants' else 'D2-')+r['variant'],'pooled','U median >= 2 pp; Holm(3) < .05' if section=='variants' else 'After D1: nearest > random; Holm(3) < .05',f"median {r['median_pp']:.3f}; pH {r['p_holm']:.5g}",r.get('passed',r.get('rest_informative')))
    p='reports/stage_c3/report/gate.json';b=catalog_json(p,'Appendix: gates','C3')
    for r in b['variants']:gate('C3-D3-'+r['variant'],'pooled','Task nearest > random; k=1; Holm(2) < .05',f"median {r['median_pp']:.3f}; pH {r['p_holm']:.5g}",r['passed'])
    p='reports/stage_c4/report/endpoints.json';b=catalog_json(p,'Appendix: gates','C3')
    gate('C4-main','pooled','LoRA n=10; P2 > P1 and P0; both one-sided raw p < .05','; '.join(f'{k}: p {r["p_greater"]:.5g}' for k,r in b['primary'].items()),b['passed'])
    for r in b['secondary_P2_vs_P1']:gate('C4-secondary-n'+str(r['shots']),'pooled','P2 > P1; Holm(2) across label budgets < .05',f"median {r['median_pp']:.3f}; pH {r['p_holm']:.5g}",r['p_holm']<.05)
    p='reports/stage_m/report/endpoints.json';b=catalog_json(p,'Appendix: gates','C4')
    for label,items in [('main',b['primary'])]+[('secondary-n'+str(n),v) for n,v in b['secondary'].items()]:
        for method,r in items.items():
            passed=r.get('passed',r['p_holm']<.05 and r['median_pp']>=1 and r['drop_gt_2pp_fraction']<=.1)
            gate('M-'+label+'-'+method,'pooled','vs R1; Holm(2) < .05; median >= 1 pp; drop > 2 pp <= .10',f"median {r['median_pp']:.3f}; pH {r['p_holm']:.5g}; drop {r['drop_gt_2pp_fraction']:.3f}",passed)
            if label=='main':
                for col,alias in [('median_pp','Median'),('p_holm','PH'),('drop_gt_2pp_fraction','Drop')]:
                    key='json:'+p+':/primary/'+method+'/'+col;fact(method+alias,key)
    for method,r in b['secondary_adapt_gain_n10'].items():gate('M-adapt-'+method,'pooled','Adaptation increment vs R1; Holm(2) < .05',f"mean {r['mean_pp']:.3f}; pH {r['p_holm']:.5g}",r['p_holm']<.05)
    for prefix,path in [('W','reports/stage_w/report/diagnostic_tests.json'),('WF','reports/stage_w_followup/report/diagnostic_tests.json')]:
        b=catalog_json(path,'Appendix: gates','C2/C3')
        for section,items in b.items():
            if not isinstance(items,list):continue
            for r in items:
                rule='U median >= 2 pp; Holm(3) < .05' if 'diagnostic1' in section else 'Nearest > random; k=1; rest Holm(3), task Holm(2) < .05'
                scope='Lee' if 'Lee' in section else 'BNCI; swap subset' if 'BNCI' in section else 'pooled'
                gate(prefix+'-'+section+'-'+r['variant']+'-'+r.get('source',''),scope,rule,f"median {r['median_pp']:.3f}; pH {r['p_holm']:.5g}",r['passed'])
    assert len(rows)==40,len(rows)
    table_file('gates',['Gate ID','Cohort','Prespecified rule','Observed','Decision'],rows,widths='p{1.75in}p{.55in}p{2.15in}p{1.65in}p{.5in}',long=True)
    CHECKS.append({'check':'all_original_joint_gates','rows':len(rows)})

def tests():
    source('docs/all_tests.md');source('docs/claims_ledger.md');source('docs/cross_model_summary.md')
    cov=read_json('reports/stage_w_followup/test_inventory/source_coverage.json')
    preserved=TABLES/'source_tables';preserved.mkdir(exist_ok=True)
    for item in cov['source_tables']:
        path=item['source'].replace('\\','/');d=catalog_csv(path,'Appendix: complete source tables')
        assert len(d)==item['rows'] and SOURCES[path]==item['sha256'],path
        # Anonymous report tables only, with source name preserved in a mapping manifest.
        dest=preserved/(path.replace('/','__'))
        shutil.copyfile(ROOT/path,dest)
    invpath='reports/stage_w_followup/test_inventory/all_tests_rows.csv';inv=catalog_csv(invpath,'Appendix: all comparisons')
    assert len(inv)==904
    inv['print_id']=['T'+str(i).zfill(4) for i in inv['_record']]
    rules={'双侧 Holm<.05':'Two-sided; source Holm','单侧 Holm<.05':'One-sided; source Holm','双侧原始p<.05':'Two-sided; raw p','单侧原始p<.05':'One-sided; raw p','仅描述，无显著性门槛':'Descriptive only'}
    hypotheses={'B1a−B0':'Rest EA minus task head','各个人适配−B0':'Personal adaptation minus B0','selected_film−head':'Selected FiLM minus tuned head','固定FiLM−head':'Fixed FiLM minus tuned head','同一头的EA−B0':'EA minus B0 with the same head','个人适配−B0':'Personal adaptation minus B0','个人适配−B阶段微调头':'Personal adaptation minus the stage-B tuned head','真实上下文−对照':'Real context minus control','真实上下文−所列对照':'Real context minus listed control','各条件−B0':'Each condition minus B0 (later half)','各条件全部试次−B0':'Each condition minus B0 (all trials; all-trial scoring only)','本人参数−交换参数':'Own minus swapped parameters','最近邻参数−随机参数':'Nearest minus random parameters','近邻参数−同k随机参数':'Nearest minus random parameters at the same k','前n个标签适配−B3':'Earliest-n-label adaptation minus B3','所列先验左项−右项':'Listed prior contrast: left minus right','所列方法左项−右项':'Listed method contrast: left minus right','G−指定群体对照':'G minus named population control','本人−交换':'Own minus swapped','近邻−随机':'Nearest minus random','少样本适配−对应起点':'Few-shot adaptation minus matched start','G−冻结超参数的头对照':'G minus fixed-hyperparameter task head','本人−对应群体起点':'Own minus matched population start','少样本适配−G':'Few-shot adaptation minus G','本人−G':'Own minus G'}
    pieces=[]
    for group,(keys,items) in enumerate(inv.groupby(['stage','source','hypothesis','rule'],sort=False)):
        stage,path,hyp,rule=keys
        rows=[]
        for _,r in items.iterrows():
            values=[]
            for col in ['n','median_pp','mean_pp','p_raw','p_adjusted']:
                v=r[col];values.append('--' if pd.isna(v) else str(int(v)) if col=='n' else f'{v:.4g}' if col.startswith('p_') else f'{v:.3f}')
            cond=str(r.condition) if pd.notna(r.condition) else '(selected variant)'
            ds=str(r.dataset).replace('起步组合并','pooled')
            rows.append([r.print_id,cond,ds]+values)
        name=f'comparisons_{group:02d}'
        table_file(name,['ID','Condition / contrast','Dataset','$N$','Median','Mean','$p$','$p_H$'],rows,
                   widths='p{.45in}p{2.6in}p{.75in}rrrrr',long=True)
        access_note=' R1 comparisons use test-query-informed validation priors; these decisions describe the query-informed procedure and do not provide independent confirmation.' if 'stage_m/' in path.replace('\\','/') else ''
        pieces.append(r'\subsection*{'+tex_escape(stage+' / '+Path(path.replace('\\','/')).stem)+'}\n'+tex_escape(hypotheses[hyp])+'. '+tex_escape(rules[rule])+'. Differences in pp; original correction family only.'+access_note+' '+r'\input{tables/'+name+'}\n')
    (TABLES/'all_comparisons.tex').write_text('\n'.join(pieces),encoding='utf-8')
    inv.drop(columns='_record').to_csv(TABLES/'all_tests_rows.csv',index=False,encoding='utf-8')
    CHECKS.append({'check':'source_coverage_rows_and_sha256','tables':len(cov['source_tables']),'comparison_rows':len(inv),'passed':True})
    rows=[]
    for cohort,path in [('Core cohort','reports/stage_w/report/convergence.csv'),('BNCI','reports/stage_w_followup/analysis/bnci_convergence.csv')]:
        d=catalog_csv(path,'Appendix: convergence')
        for _,r in d.iterrows():rows.append([r.get('cohort',cohort),r['method'],r['fold'],r['seed'],r['additional_steps_run'],r['selected_additional_step'],str(r['training_curve_flat']),str(r['validation_curve_flat']),str(r['plateau_observed'])])
    assert len(rows)==150
    table_file('convergence',['Cohort','Family','Fold','Seed','Run steps','Selected','Train flat','Val. flat','Both flat'],rows,long=True)

def metadata():
    lookup={r['key']:r for r in NUMBERS};rows=[]
    for key,r in lookup.items():
        if re.fullmatch(r'f4\.[^.]+\.(?:B3|G)\.n\d+',key):
            _,ds,start,budget=key.split('.')
            rows.append([NAMES[ds],start,budget[1:],lookup[key+'.n']['display_value'],r['display_value'],lookup[key+'.sd']['display_value']])
    table_file('fewshot_distributions',['Dataset','Start','Labels','$N$','Mean gain (pp)','Subject SD (pp)'],rows,long=True)
    # Preserve exact frozen grids in English, with paths in the internal ledger only.
    rows=[]
    for name in ['stage_a','stage_b','stage_b2','stage_c','stage_c2','stage_c3','stage_c4','stage_m','stage_w','stage_w_followup','cross_model_x']:
        path='configs/'+name+'.yaml';source(path);data=yaml.safe_load((ROOT/path).read_text(encoding='utf-8'))
        if name == 'cross_model_x':
            data={k:data[k] for k in ['seeds','preprocessing','readout','adaptation','population','convergence','personal_and_few_shot']}
            # Only executed core settings; external-dataset and FiLM fields describe
            # unexecuted extensions and must not appear as completed Methods.
            data['preprocessing'].pop('task_seconds',None)
            for family in ['REVE','LaBraM']:
                data['preprocessing'][family]={k:v for k,v in data['preprocessing'][family].items() if not k.startswith('BNCI')}
            data['readout']={k:v for k,v in data['readout'].items() if not k.startswith(('Lee','BNCI'))}
            data['adaptation']={k:v for k,v in data['adaptation'].items() if not k.startswith('film')}
            data['personal_and_few_shot']['learning_rates'].pop('film_offset',None)
        def walk(x,pointer=''):
            if isinstance(x,dict):
                for k,v in x.items():walk(v,pointer+'/'+str(k))
            elif isinstance(x,list) and all(isinstance(v,(str,int,float)) for v in x):
                if any(isinstance(v,str) and ('/' in v or ':' in v) for v in x):return
                rows.append([name,pointer,', '.join(map(str,x))])
                for i,v in enumerate(x):
                    if isinstance(v,(int,float)) and not isinstance(v,bool):add('config:'+path+pointer+'/'+str(i),v,'protocol',path,pointer+'/'+str(i),pointer,'frozen configuration scalar','Appendix: hyperparameters')
            elif isinstance(x,list):
                for i,v in enumerate(x):walk(v,pointer+'/'+str(i))
            elif isinstance(x,(int,float)) and not isinstance(x,bool):
                add('config:'+path+pointer,x,'protocol',path,pointer,pointer,'frozen configuration scalar','Appendix: hyperparameters');rows.append([name,pointer,str(x)])
            elif isinstance(x,str) and not any(token in pointer for token in ['sha256','commit','split','halves','weights','minimum_free','source']) and len(x)<105:
                rows.append([name,pointer,x])
        walk(data)
    table_file('configuration_archive',['Stage','Configuration field','Frozen value / grid'],rows,widths='p{.65in}p{2.8in}p{3.2in}',long=True)
    from journal_grids import compact_grids
    compact_grids()
    grid=TABLES/'hyperparameters.tex'
    text=grid.read_text(encoding='utf-8')
    first=text.index('\n')
    text=text[:first+1]+r'\caption{Recorded architecture and optimization settings. Semicolons in each value cell follow the setting-column order; commas enumerate a search grid. Forty-label conditions require sufficient earlier support. New-model entries refer to REVE and LaBraM; personal optimization follows the common grid.}\label{tab:grids}\\'+'\n'+text[first+1:]
    grid.write_text(text,encoding='utf-8')
    # Recorded resources: no extrapolation of missing historical cost.
    w=catalog_json('reports/stage_w/report/resources.json','Appendix: compute')
    f=catalog_json('reports/stage_w_followup/report/resources.json','Appendix: compute')
    d=catalog_csv('reports/stage_m/report/training_resources.csv','Appendix: compute','C4');rows=[]
    for (method,k),s in d.groupby(['method','k'],dropna=False):
        seconds=stat(s,'seconds',f'resource.{method}.{k}.seconds','Appendix: compute','C4',scale=1,unit='seconds')
        rows.append([method,'--' if pd.isna(k) else int(k),int(s.outer_steps.iloc[0]),f'{seconds:.1f}',len(s)])
    table_file('compute',['Method','Inner steps','Outer updates','Mean seconds','Runs'],rows)
    source('scripts/stage_m_run.py');source('src/subject_context/stage_m.py')
    mp='reports/stage_m/report/run_manifest.json'
    manifests=catalog_json(mp,'Methods; Supplementary: recorded numerical events','C4')
    for alias,key,value,formula in [
        ('MetaFormalRuns','meta.formal_runs',len(manifests),'length of saved run manifest list'),
        ('MetaDivergences','meta.inner_divergences',sum(r['inner_divergences'] for r in manifests),'sum of inner_divergences over saved formal run manifests'),
        ('MetaAffectedRuns','meta.affected_runs',sum(r['inner_divergences']>0 for r in manifests),'count saved formal runs with inner_divergences > 0')]:
        add(key,value,'count',mp,'/'.join(['all','inner_divergences']),'inner_divergences',formula+'; figures/tables.py','Methods: numerical events','C4');fact(alias,key)
    rows=[[r['fold'],r['seed'],r['inner_divergences']]+[r['inner_divergences_by_shots'].get(str(n),0) for n in [5,10,20]] for r in manifests]
    table_file('meta_divergence_inventory',['Fold','Seed','All recorded events','Five labels','Ten labels','Twenty labels'],rows,long=True)
    for name,path in [('W','reports/stage_w/report/resources.json'),('Followup','reports/stage_w_followup/report/resources.json')]:fact(name+'CardHours','json:'+path+':/GPU_card_hours')
    # The reviewed draft's candidate ledger remains traceable; formal figures use their own IDs.
    for r in csv.DictReader((PAPER/'numbers_draft.csv').open(encoding='utf-8-sig')):
        add('draft:'+r['number_id'],r['value'],r['unit'],r['source_file'],r['source_record'],r['source_column'],r['computation'],r['location'],r['claim_id'],r['display_value'])
    bindings=[('reports/stage_c/report/criteria_tests.csv',{'method':'M_film','control':'B4'},'p_holm','ContextFilmPH'),
              ('reports/stage_c/report/criteria_tests.csv',{'method':'M_lora','control':'B4'},'p_holm','ContextLoraPH'),
              ('reports/stage_w_followup/report/bnci_population_comparisons.csv',{'method':'G_lora8','control':'B0_linear'},'mean_pp','BNCIPopGain'),
              ('reports/stage_w_followup/report/bnci_population_comparisons.csv',{'method':'G_lora8','control':'B0_linear'},'p_one_sided_raw','BNCIPopP'),
              ('reports/stage_b/report/ea_tests.csv',{'condition':'B1b','head':'mlp','dataset':'Dreyer2023'},'mean_gain_pp','EADreyerGain'),
              ('reports/stage_b/report/ea_tests.csv',{'condition':'B1b','head':'mlp','dataset':'Dreyer2023'},'p_holm','EADreyerPH')]
    for path,selection,col,alias in bindings:
        d=read(path)
        for k,v in selection.items():d=d[d[k]==v]
        assert len(d)==1
        key=f'source:{path}:{int(d.iloc[0]["_record"])}:{col}';fact(alias,key)

def build():
    datasets();literature();gates();tests();metadata()
