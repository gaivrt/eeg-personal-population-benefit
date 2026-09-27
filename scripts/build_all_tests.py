"""Compile the complete research-test appendix from preserved result tables.

This is a reporting inventory, not a new statistical analysis or a new test family.
Repeated/overlapping summaries remain visible and are never counted as replications.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/stage_w_followup/test_inventory'
OUT.mkdir(parents=True, exist_ok=True)
lines = []
inventory = []
sources = []


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def fmt(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return '—'
    if isinstance(value, float):
        return f'{value:.5g}'
    return str(value).replace('|', '/').replace('\n', ' ')


def table(headers, rows):
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    lines.extend('| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in rows)
    lines.append('')


lines.append('''# 全部预先定义检验与完整比较附录

本附录依据各阶段结果前固定的规格、配置、判定回执和完整比较表编制，保留未通过、反向、不可估与未执行项目。它不是在全部历史结果上重新定义的统一检验族，也不把连续探索追溯称为一次预注册研究。G 始终为 W 的主参照，R2 为补充敏感性分析。

每名被试先平均五种子；被试是统计单位。起步组为 PhysionetMI 103 人、Dreyer2023 80 人、Cho2017 52 人；`pooled` 为全部 235 人，`Dreyer+Cho` 为 132 人。重叠汇总、历史并列及主检验在描述表中的再次出现不构成独立重复。差值为左项减右项，单位 pp。所有显著性均使用当时的规则，不新增等效界或事后通过门槛。

“通过/未通过”用于预先指定的门槛；仅有描述性 p 值的行标为“无方法门槛”，并报告是否检出差异及其方向。双侧显著退化不标为有效性通过。p≥.05 不证明等效；单个 p<.05 不覆盖效应量、下降比例等联合条件。原报告中的 `passed` 只在对应原检验范围内使用。

## 1. 数据与工程检查

| ID | 阶段 / 数据集 | 预定问题与规则 | 结果 / 判定 | 证据 |
|---|---|---|---|---|
| E01 | 0a / 合成与固定真实样本 | 被试划分互斥；上下文早于查询且无交叠；同库打乱；零 FiLM 恒等；小批过拟合；统计实现校验 | 实现检查通过；不构成真实任务性能证据 | [原规格](experiment_spec.md)，`tests/` |
| E02 | 0a–0b / CBraMod 与 HPC4 | 官方权重及预训练来源；真实 CPU 前向；CUDA 小测；计算节点网络；服务器测试 | 执行记录通过；预训练数据重合边界单列，不能用本项目折隔离替代 | [阶段 0b](../reports/stage0b/report.md)；[预训练审计](pretraining_overlap.md) |
| E03 | A / 起步组三库 | 任一库检查失败人数>10%则停 | 初次 PhysioNet 11/103 触发；经用户另行授权恢复。旧实现筛选也拒绝全部 Cho，不能抹除这次失败或当作真实坏数据结论 | [A 历史记录](stage_a_report_2026-09-25.md) |
| E04 | A / 起步组三库 | 原 B0 按库 BA<55%停止规则；后经用户简化改为记录不删人 | 两种 B0 三库均≥55%；记录事实，不倒推原数据门槛通过 | [A 基线报告](../reports/stage_a/baselines/report.md)，[简化协议](stage_a_simple_protocol.md) |
| E05 | A / PhysioNet | 匹配窗口枕区 8–13 Hz 闭眼功率应高于睁眼；异常列示，不自动删人 | 最终 101/103 满足；S037/S048 方向不符；Dreyer 无枕区通道、Cho 无闭眼段，不适用 | [A 数据卡](../reports/stage_a/baselines/data_cards.csv) |
| E06 | A / 起步组三库 | 静息存在、眼状态、边界、时长、通道及左右手提示审计；有限值检查 | 235 人纳入；Dreyer 特殊边界及 Cho 66.5 秒字段差异保留，不把缺记录填成通过 | [A 报告](stage_a_report_2026-09-25.md)，[局限](findings_summary.md) |
| E07 | A / Cho2017 | 可脚本化则验证 52 人顺序/类内事件/完整信号；否则明确覆盖范围 | 52 人官方顺序元数据检查；s01 信号回查；缺独立连续时间线和 rest→MI 绝对间隔，完整时序验证不能判为通过 | [规格与更正](experiment_spec.md)，[局限](findings_summary.md) |
| E08 | B–W / 起步组与 Lee | 前后半类别与计数；双类 BA；少数类<40%仅诊断，不删人 | 固定半段、每人分母与异常保留；PhysioNet 前半22，不执行 n=40/80 | 各阶段 `halves` / 逐人计数与 [W 核验](../reports/stage_w/report/verification.json) |
| E09 | C3 / 起步组 | B3 与原诊断严格同分、零个人参数恒等 | 批次组成导致边界预测变化，原断言失败；按结果前明确的修订记录差异。374 项中4项不同，最大约 .042 BA；不是新模型效果 | [C3 执行偏离](stage_c3_report_2026-09-26.md)，规格 v1.10–1.11 |
| E10 | W / 起步组与 Lee | 逐试次源标签、BA、供体及冻结超参数来源一致；旧结果复核 | 6,816,775 行已核验；起步组2/2350基线检查不同，为2条边界试次；Lee810项无差异 | [W 核验](../reports/stage_w/report/verification.json)，[W 报告](stage_w_report_2026-09-26.md) |
| E11 | W 补充 / 历史 G | 不重训，检查验证准确率早停时是否仍上升 | 不可判定：历史只保存 CE、覆盖式最佳权重，无逐检查点准确率/预测/全套权重 | [早停记录报告](stage_w_followup_report_2026-09-27.md) |
| E12 | W 补充 / BNCI | 下载前空间；9 人 session 2 原始 GDF 睁眼276标记、MAT逐样本匹配；19读出/22 EEG；前后半隔离 | CPU 审计通过：9人均有任务前睁眼静息；每人144试次，前后各72且左右手各36。单人折无供体，其他折一个供体，近邻选择价值不可辨识 | [结果前协议](stage_w_followup_protocol.md)，[补充报告](stage_w_followup_report_2026-09-27.md) |
| E13 | W 补充 / R2与BNCI | 原规则的试次/标签/供体核验；零个人参数恒等；群体/骨干冻结；超参数来源不变 | 50运行、2,953,990行预测本地重算一致；R2零参数恒等，3/1625基线检查因批次路径各翻转1次；BNCI135项无翻转、全部起步组超参数值与哈希一致，无本库搜索 | [R2独立审计](../reports/stage_w_followup/r2_audit/R2_independent_audit.json)、[补充结果审计](../reports/stage_w_followup/report/verification.json)、[来源审计](../reports/stage_w_followup/analysis/postprocess_verification.json) |

## 2. 预定主门槛和指定次要终点

下表保留联合门槛，不用完整比较表中的描述性 p 值替代它。全部合并检验也在随后分阶段明细中给出其组成统计量。
''')

gates = []
def gate(identifier, stage, dataset, hypothesis, rule, observed, passed, source):
    gates.append([identifier,stage,dataset,hypothesis,rule,observed,
                  '通过' if passed is True else '未通过' if passed is False else passed,
                  f'[原记录](../{source})'])

b=read('reports/stage_b/report/gate.json')
gate('B-G','B','起步组合并','验证选定末层 FiLM 相对 B0 有足够增益','中位增益≥2pp',f"中位 {b['observed_median_gain_pp']:.3f}pp",b['passed'],'reports/stage_b/report/gate.json')
for r in read('reports/stage_b2/report/gate.json')['variants']:
    gate('B2-'+r['variant'],'B2','起步组合并',r['variant']+'：相对 B0 有增益且优于微调头','中位≥2pp 且 vs头单侧 Holm(4)<.05',f"中位={r['median_gain_vs_b0_pp']:.3f}; pH={r['vs_head_p_holm']:.5g}",r['passed'],'reports/stage_b2/report/gate.json')
for r in read('reports/stage_c/report/criteria.json')['methods']:
    gate('C-'+r['method'],'C','起步组合并',r['method']+'：上下文个性化满足原成功标准','vs B4与B0 单侧 Holm(4)<.05；均值恢复≥30%；下降>2pp≤10%',f"两对照显著={r['significant_vs_B4_and_B0']};恢复={r['recovery_fraction']:.3f};下降={r['drop_gt_2pp_fraction']:.3%}",r['passed'],'reports/stage_c/report/criteria.json')
for r in read('reports/stage_c2/report/gate.json')['variants']:
    gate('C2-D1-'+r['variant'],'C2诊断1','起步组合并',r['variant']+'：本人优于交换','U中位≥2pp 且单侧 Holm(3)<.05',f"U中位={r['median_pp']:.3f};pH={r['p_holm']:.5g}",r['passed'],'reports/stage_c2/report/gate.json')
for r in read('reports/stage_c2/report/gate.json')['diagnostic2']:
    gate('C2-D2-'+r['variant'],'C2诊断2','起步组合并',r['variant']+'：静息最近邻优于随机','D1通过后，单侧 Holm(3)<.05',f"差中位={r['median_pp']:.3f};pH={r['p_holm']:.5g}",r['rest_informative'],'reports/stage_c2/report/gate.json')
for r in read('reports/stage_c3/report/gate.json')['variants']:
    gate('C3-D3-'+r['variant'],'C3诊断3','起步组合并',r['variant']+'：任务k=1近邻优于随机','单侧 Holm(2)<.05',f"差中位={r['median_pp']:.3f};pH={r['p_holm']:.5g}",r['passed'],'reports/stage_c3/report/gate.json')
c4=read('reports/stage_c4/report/endpoints.json')
gate('C4-main','C4','起步组合并','LoRA n=10 无标签先验优于随机先验及 B3','P2−P1 与 P2−P0 单侧原始p均<.05', '; '.join(f'{k}:p={r["p_greater"]:.5g}' for k,r in c4['primary'].items()),c4['passed'],'reports/stage_c4/report/endpoints.json')
for r in c4['secondary_P2_vs_P1']:
    gate('C4-secondary-n'+str(r['shots']),'C4次要','起步组合并',f"LoRA n={r['shots']} P2>P1",'两标签量单侧 Holm(2)<.05',f"中位={r['median_pp']:.3f};pH={r['p_holm']:.5g}",r['p_holm']<.05,'reports/stage_c4/report/endpoints.json')
m=read('reports/stage_m/report/endpoints.json')
for name,r in m['primary'].items():
    gate('M-main-'+name,'M主终点','起步组合并',name+' n=10 优于验证选出的R1','单侧 Holm(2)<.05；中位≥1pp；下降>2pp≤10%',f"中位={r['median_pp']:.3f};pH={r['p_holm']:.5g};下降={r['drop_gt_2pp_fraction']:.3%}",r['passed'],'reports/stage_m/report/endpoints.json')
for n,values in m['secondary'].items():
    for name,r in values.items():
        success=r['p_holm']<.05 and r['median_pp']>=1 and r['drop_gt_2pp_fraction']<=.1
        gate('M-secondary-'+name+'-n'+n,'M次要','起步组合并',f'{name} n={n} vs R1','同样比较；各n Holm(2)，列原效应门槛',f"中位={r['median_pp']:.3f};pH={r['p_holm']:.5g};下降={r['drop_gt_2pp_fraction']:.3%}",success,'reports/stage_m/report/endpoints.json')
for name,r in m['secondary_adapt_gain_n10'].items():
    gate('M-adapt-'+name,'M次要适配增益','起步组合并',name+' 相对 R1 的 (n10−n0) 增益','指定单侧 Holm(2)<.05；不改变主判定',f"均值={r['mean_pp']:.3f};pH={r['p_holm']:.5g}",r['p_holm']<.05,'reports/stage_m/report/endpoints.json')
for name,values in read('reports/stage_w/report/diagnostic_tests.json').items():
    if not isinstance(values,list):continue
    for r in values:
        diagnostic1='diagnostic1' in name
        gate('W-'+name+'-'+r['variant']+'-'+r.get('source',''),'W','Lee2019_MI' if name.startswith('Lee') else '起步组合并',r['variant']+' '+('本人>交换' if diagnostic1 else r.get('source','rest')+' k=1近邻>随机'),'U中位≥2pp且单侧Holm(3)<.05' if diagnostic1 else '原近邻单侧Holm：静息3项/任务2项',f"中位={r['median_pp']:.3f};pH={r['p_holm']:.5g}",r['passed'],'reports/stage_w/report/diagnostic_tests.json')
followup=ROOT/'reports/stage_w_followup/report/diagnostic_tests.json'
if followup.exists():
    for name,values in read(str(followup.relative_to(ROOT))).items():
        if not isinstance(values,list):continue
        for r in values:
            gate('WF-'+name+'-'+r['variant'],'W补充','起步组合并' if name.startswith('R2') else 'BNCI可交换8人',r['variant']+' 本人>交换','U中位≥2pp 且单侧Holm(3)<.05',f"U中位={r['median_pp']:.3f};pH={r['p_holm']:.5g};交换N={r['swap_subjects']}",r['passed'],'reports/stage_w_followup/report/diagnostic_tests.json')
else:
    r2_source='reports/stage_w_followup/r2_audit/R2_subject_seed_results.csv'
    if (ROOT/r2_source).exists():
        from stage_w_results import distribution,holm
        r2=pd.read_csv(ROOT/r2_source).groupby(['variant','dataset','subject'],as_index=False).upper_bound.mean()
        r2_stats=[{'variant':v,**distribution(g.upper_bound)} for v,g in r2.groupby('variant')]
        for r,corrected in zip(r2_stats,holm([r['p_one_sided_raw'] for r in r2_stats]+[1.])[:len(r2_stats)]):
            gate('WF-R2_sensitivity_diagnostic1-'+r['variant'],'W补充','起步组合并',r['variant']+' 本人>交换',
                 'U中位≥2pp 且单侧Holm(3)<.05',f"U中位={r['median_pp']:.3f};pH={corrected:.5g};交换N={r['subjects']}",
                 bool(r['median_pp']>=2 and corrected<.05),r2_source)
    else:
        gate('WF-R2','W补充','起步组合并','R2起点FiLM与LoRA本人>交换','原诊断1门槛','待取回核验','待完成','configs/stage_w_followup.yaml')
    gate('WF-BNCI','W补充','BNCI9人/交换8人','本人−G、本人−交换、群体与少样本','原诊断1门槛；其他按冻结方案描述','已提交，待完整取回核验','待完成','configs/stage_w_followup.yaml')
table(['ID','阶段','数据集','假设','判定规则','结果','是否通过','出处'],gates)
lines.append('''C4 只有主终点通过才支持其预定“标签节省”结论；原终点文件保存的插值量不满足该前提，不能宣称已证明节省标签。M 的按库等效标签插值为预定描述，不能替代失败的主终点；非单调或超出观察范围时不外推。R2 匹配外循环更新次数，并非 FLOPs 或墙钟。W 的 U>0 仅检验个人特异性，个人收益另以本人−G衡量。

## 3. 全部统计比较与少样本描述

以下每张表逐行列出完整源表，包括反向及未显著结果。主门槛行在上一节解释；以下描述性显著性不被升级为方法通过。没有原定 p 值的少样本曲线不事后补检验。源文件中的全精度值保留，正文只缩短显示精度。
''')


def detail(stage, relative, title, hypothesis, keys, median='median_pp', mean='mean_pp', p='p_raw', adjusted=None, side='双侧', note='', frame=None, aggregation=None):
    path=ROOT/relative
    data=pd.read_csv(path) if frame is None else frame
    if not any(s['source']==relative for s in sources):
        sources.append({'stage':stage,'source':relative,'rows':len(pd.read_csv(path)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    lines.extend([f'### {stage} · {title}', '',f'{hypothesis}。{note} [完整源表](../{relative})。',''])
    records=[]
    for sequence,(index,r) in enumerate(data.iterrows(),start=1):
        identifier=f'{stage}-{Path(relative).stem}-{sequence:03d}'
        condition='; '.join(f'{k}={fmt(r[k])}' for k in keys if k in r and pd.notna(r[k]))
        dataset=r.get('dataset','起步组合并')
        raw=r.get(p) if p else None
        corrected=r.get(adjusted) if adjusted else None
        used=corrected if corrected is not None and pd.notna(corrected) else raw
        rule=(f'{side} Holm<.05' if corrected is not None and pd.notna(corrected) else f'{side}原始p<.05') if raw is not None and pd.notna(raw) else '仅描述，无显著性门槛'
        med=r.get(median);avg=r.get(mean)
        if used is None or pd.isna(used):status='不适用（描述）'
        elif used>=.05:status='未检出差异；无方法门槛'
        elif side=='单侧':status='检出正向差异；无方法门槛'
        else:
            metric_label='均值' if avg is not None and pd.notna(avg) else '中位'
            sign=avg if metric_label=='均值' else med
            status=f'检出双侧差异；{metric_label}'+('反向' if sign<0 else '正向')+'；无方法门槛'
        n=next((r[x] for x in ['subjects','n_subjects','n','vs_head_n'] if x in r),None)
        record={'id':identifier,'stage':stage,'hypothesis':hypothesis,'condition':condition,'dataset':dataset,'n':n,
                'median_pp':med,'mean_pp':avg,'p_raw':raw,'p_adjusted':corrected,'rule':rule,'status':status,'source':relative,'source_row':None if aggregation else index+1,'aggregation':aggregation}
        inventory.append(record)
        records.append([identifier,condition,dataset,n,med,avg,raw,corrected,status])
    table(['ID','条件','数据集','N','中位差pp','均值差pp','原始p','校正p','判定范围'],records)


detail('A','reports/stage_a/baselines/wilcoxon.csv','EA静息参照','B1a−B0',['head'],median='median_difference_pp',mean='',adjusted='p_holm',note='两头×三库共6项双侧Holm。')
for filename,title,hypothesis,keys in [
    ('summary','全部适配−B0','各个人适配−B0',['variant']),
    ('film_vs_head','验证选定FiLM−微调头','selected_film−head',[]),
    ('fixed_film_vs_head','固定FiLM−微调头','固定FiLM−head',['variant']),
    ('ea_tests','两种EA诊断','同一头的EA−B0',['condition','head'])]:
    detail('B',f'reports/stage_b/report/{filename}.csv',title,hypothesis,keys,median='median_gain_pp',mean='mean_gain_pp',adjusted='p_holm',note='原表内统一校正；主门槛仍仅为选定FiLM中位≥2pp。')
for filename,title,hypothesis in [('summary','全部适配−B0','个人适配−B0'),('vs_finetuned_head','全部适配−微调头','个人适配−B阶段微调头')]:
    detail('B2',f'reports/stage_b2/report/{filename}.csv',title,hypothesis,['variant'],median='median_gain_pp',mean='mean_gain_pp',adjusted='p_holm',note='每表25项双侧Holm；B2四项主门槛另见第2节。B2的EA表原样复用B，不计作新检验。')
detail('C','reports/stage_c/report/criteria_tests.csv','原成功标准的组成检验','真实上下文−对照',['method','control'],adjusted='p_holm',side='单侧',note='四项主比较；两个方法的整体判定均为未通过，见第2节。')
detail('C','reports/stage_c/report/diagnostic_tests.csv','全部诊断对照','真实上下文−所列对照',['method','versus'],adjusted='p_holm',note='60项双侧Holm，含默认z、打乱、去间隔、T3A及EA。')
detail('C','reports/stage_c/report/summary.csv','各条件相对B0（后半）','各条件−B0',['condition'],adjusted='p_holm_diagnostic',note='98行原报告诊断族，包含复用的基线/上界，不能当作98次独立实验。')
c_summary=pd.read_csv(ROOT/'reports/stage_c/report/summary.csv')
detail('C-all','reports/stage_c/report/summary.csv','兼容的全部试次结果','各条件全部试次−B0',['condition'],median='median_pp_all',mean='mean_pp_all',p='p_raw_all',frame=c_summary[c_summary.p_raw_all.notna()],note='仅原表已有的全部试次列；与主后半设定分开，原始p，无新校正。')
detail('C2','reports/stage_c2/report/summary.csv','本人−交换分组诊断','本人参数−交换参数',['variant'],median='upper_median_pp',mean='upper_mean_pp',p='p_raw_two_sided',adjusted='p_holm_diagnostic',note='15项双侧诊断；D1三项单侧联合门槛见第2节。')
detail('C2','reports/stage_c2/report/diagnostic2_tests.csv','静息近邻完整比较','最近邻参数−随机参数',['variant'],adjusted='p_holm',side='单侧',note='仅pooled三项为Holm主检验，其他分组为原始p描述。')
detail('C3','reports/stage_c3/report/rest_vs_task.csv','静息/任务、k=1/3全部比较','近邻参数−同k随机参数',['variant','source','k'],p='p_raw_greater',side='单侧',note='任务k=1合并主检验校正结果另见第2节；本表原始p。')
detail('C3-few','reports/stage_c3/report/few_shot_curve.csv','全部少样本曲线','前n个标签适配−B3',['variant','shots'],p=None,side='无',note='n40仅Dreyer/Cho；没有预定通过门槛，不新增检验。')
detail('C4','reports/stage_c4/report/comparisons.csv','全部先验与标签量','所列先验左项−右项',['variant','shots','comparison'],p='p_greater',side='单侧',note='90项完整描述；指定主/次终点及其校正见第2节。')
detail('M','reports/stage_m/report/comparisons.csv','全部方法、标签量及适配增益','所列方法左项−右项',['shots','comparison','measure'],p='p_greater',side='单侧',note='206项描述；ba为绝对BA差，adapt_gain为(n−0)增益之差；主/次终点见第2节。')
for filename,title,hypothesis,keys in [
    ('population_comparisons','群体模型比较','G−指定群体对照',['method','control']),
    ('diagnostic1_side_by_side','本人/交换逐库并列','本人−交换',['start','variant']),
    ('neighbours_side_by_side','近邻逐库并列','近邻−随机',['start','variant','source','k']),
    ('few_shot_side_by_side','少样本逐库并列','少样本适配−对应起点',['start','shots'])]:
    detail('W',f'reports/stage_w/report/{filename}.csv',title,hypothesis,keys,p='p_one_sided_raw',side='单侧',note='按库原始p描述；B3行是历史并列，不是独立新实验。所有主门槛以第2节为准。')
detail('W-pooled','reports/stage_w/neighbour_audit/pooled_descriptive.csv','合并近邻主/次描述','近邻−随机',['start','variant','source','k'],p='p_raw_greater',side='单侧',note='核验后生成的合并描述；k=3为次要，k=1校正门槛见第2节。')
bnci_comparisons=ROOT/'reports/stage_w_followup/report/bnci_population_comparisons.csv'
if bnci_comparisons.exists():
    detail('WF-BNCI',str(bnci_comparisons.relative_to(ROOT)),'群体适配与任务头/显式线性探针','G−冻结超参数的头对照',['method','control'],p='p_one_sided_raw',side='单侧',frame=pd.read_csv(bnci_comparisons).assign(dataset='BNCI2014_001'),note='BNCI9人，session2二分类；原始p描述，不与Compass四分类LOSO数值直接相减。')
    strength_relative='reports/stage_w_followup/report/population_strength_by_dataset.csv'
    strength=pd.read_csv(ROOT/strength_relative)
    for prefix,hypothesis in [('own_minus_population','本人−对应群体起点'),('own_minus_swap','本人−交换')]:
        renamed=strength.rename(columns={c:c.removeprefix(prefix+'_') for c in strength if c.startswith(prefix+'_')})
        # The source also has its own unprefixed subject count; retain it only once.
        renamed=renamed.loc[:,~renamed.columns.duplicated()]
        detail('WF-R2-'+prefix,strength_relative,'三个起点逐库敏感性描述',hypothesis,['start','variant','population_family'],p='p_one_sided_raw',side='单侧',frame=renamed,note='G主参照，R2敏感性；B3/G复用历史样本，描述p不构成独立验证或新通过标准。')
    for relative,group,hypothesis,column,stage in [
        ('reports/stage_w_followup/report/bnci_few_shot_subjects.csv','shots','少样本适配−G','gain','WF-few'),
        ('reports/stage_w_followup/report/bnci_diagnostic1_subjects.csv','variant','本人−G','own_gain','WF-own')]:
        raw=pd.read_csv(ROOT/relative)
        descriptive=raw.groupby(group)[column].agg(n='size',median_pp='median',mean_pp='mean').reset_index()
        descriptive[['median_pp','mean_pp']]*=100
        descriptive['dataset']='BNCI2014_001'
        detail(stage,relative,'冻结设置的逐人均值描述',hypothesis,[group],p=None,side='无',frame=descriptive,aggregation=f'groupby {group}: count, median and mean of {column}; convert to pp',note='9人，不新增显著性门槛；从完整个人明细按所列条件聚合，结构化清单记录聚合规则，不伪装成单条源行。')

lines.append('''## 4. 平台判据与未执行/不可估项目

W 平台是描述性检查，不是模型有效性门槛：最后8个检查点损失极差≤max(1e−4, |均值|×1%)，训练和验证均满足才记平台。实际停止步数与所选步数分开报告。历史W四组各25条均未满足平台；单条结果如下。准确率轨迹没有记录，不能据CE判定其趋势。
''')
convergence=pd.read_csv(ROOT/'reports/stage_w/report/convergence.csv')
table(['ID','数据群','模型','折','种子','训练平台','验证平台','共同平台'],
      [[f'W-plateau-{i+1:03d}',r.cohort,r.method,r.fold,r.seed,'是' if r.training_curve_flat else '否','是' if r.validation_curve_flat else '否','通过' if r.plateau_observed else '未通过'] for i,r in enumerate(convergence.itertuples())])
new_convergence=ROOT/'reports/stage_w_followup/analysis/bnci_convergence.csv'
if new_convergence.exists():
    bnci_convergence=pd.read_csv(new_convergence)
    lines.extend(['BNCI补充运行同样记录平台；新增准确率日志不参与选择检查点。',''])
    table(['ID','数据集','模型','折','种子','训练平台','验证平台','共同平台'],
          [[f'WF-plateau-{i+1:03d}','BNCI2014_001',r.method,r.fold,r.seed,'是' if r.training_curve_flat else '否','是' if r.validation_curve_flat else '否','通过' if r.plateau_observed else '未通过'] for i,r in enumerate(bnci_convergence.itertuples())])
lines.append('''| ID | 原计划 / 假设 | 数据集 | 原判定规则或执行前提 | 状态与原因 |
|---|---|---|---|---|
| N01 | 原成功标准相对B1而非B0 | 起步组 | 原v1.0；v1.2结果前经用户更正为B0 | 已被结果前修订替代，不按旧标准另判成功 |
| N02 | M确认性Lee检验 | Lee54人 | 仅M主终点通过后执行同变体、同最强对照 | 未执行，M主终点失败；W的Lee是另行授权、不同训练方案，不能补当M确认 |
| N03 | 跨session鲁棒性 | Lee/Yang/Stieger | 已冻结上下文预算，按指定session方向报告；未设独立显著门槛 | 未执行；当前结果均同session |
| N04 | D上下文长度30/60/120/180秒 | Dreyer及其他库真实可用范围 | 仅Dreyer完整范围；300秒另属睁闭合用，不接入睁眼曲线 | D未启动，未执行，不能判阴性 |
| N05 | D睁眼/闭眼/合用 | 共同可用被试/session | 总时长和被试匹配；未预设数值门槛 | 未执行 |
| N06 | D静息 vs 任务上下文生成器 | 起步/扩展组 | 原D机制分析；无固定成功门槛 | 未执行；C3近邻诊断不是该训练消融 |
| N07 | D协方差/骨干嵌入/两者消融 | 原D计划 | 比较输入；无固定成功门槛 | 未执行 |
| N08 | D训练被试数25/50/100/全部 | 原D计划 | 缩放曲线；无固定成功门槛 | 未执行 |
| N09 | D z探针：个人最优参数 vs 身份 | 原D计划 | 线性探针；无固定成功门槛 | 未执行 |
| N10 | D去间隔损失 | 起步组 | λ=0；无独立成功门槛 | D未启动；相同λ=0比较已经在C作为指定诊断执行，见C完整诊断表，勿重复计实验 |
| N11 | 高标签量超出可用前半 | PhysioNet；其他库按实际计数 | n不能超过原前半标签量，不重复样本冒充新标签 | 相应n不可用；C的40/80及C3/W的40按源表的实际库范围报告 |
| N12 | BNCI单人测试折交换 | BNCI单人折 | 供体仅同折同库其他测试被试，保留5折 | 不可估；本人−G仍覆盖该人，不能将缺供体补零 |
| N13 | BNCI静息近邻选择价值 | BNCI全部9人 | 原k=1近邻与随机供体比较 | 不可辨识：1人无供体，8人各唯一供体；差值0由结构决定，不是检验失败或无信息证据 |
| N14 | G早停时验证准确率是否仍上升 | 起步组与Lee | 不重训，只复核既有轨迹 | 未记录而不可判定；不补造曲线、不由损失推断 |

数据质量、执行异常和数值一致性是可审计性检查，不应与研究假设的显著性混算。每次阶段失败后的进一步工作均由用户另行授权，旧判定保留。C3重跑、M预实验发散规则及W历史记录缺失详见规格修改记录，不把程序错误的中止算作支持或否定科学假设。

## 5. 追溯与完整性

本文件由 `scripts/build_all_tests.py` 从保存的原始报告表逐行生成；结构化清单与源表SHA256见 [逐项清单](../reports/stage_w_followup/test_inventory/all_tests_rows.csv) 和 [源表覆盖](../reports/stage_w_followup/test_inventory/source_coverage.json)。完整精度、个人差值、改善比例和下降比例留在各源表。第2节的联合判定来源为原JSON，不由第3节的描述性标记反推。

所有范围变更见[规格修改记录](experiment_spec.md)、[W协议](stage_w_protocol.md)、[补充协议](stage_w_followup_protocol.md)。结果含义及文献冲突见[主张账本](claims_ledger.md)。
''')
pd.DataFrame(inventory).to_csv(OUT/'all_tests_rows.csv',index=False,encoding='utf-8')
covered={s['source'] for s in sources}
omissions={'reports/stage_b2/report/ea_tests.csv':'Exact historical reuse of Stage B EA, already enumerated.',
           'reports/stage_b2/report/gate_tests.csv':'All four gates enumerated from original gate.json in section 2.'}
discovered=[]
for directory in [ROOT/'reports/stage_a/baselines',*[ROOT/f'reports/stage_{s}/report' for s in ['b','b2','c','c2','c3','c4','m','w']]]:
    for path in sorted(directory.glob('*.csv')):
        columns=pd.read_csv(path,nrows=0).columns
        if any(c.startswith('p_') or c.endswith('_p_raw') or c.endswith('_p_holm') for c in columns):
            relative=path.relative_to(ROOT).as_posix();discovered.append(relative)
            assert relative in covered or relative in omissions,relative
assert not pd.DataFrame(inventory).id.duplicated().any()
(OUT/'source_coverage.json').write_text(json.dumps({'source_tables':sources,'comparison_rows':len(inventory),'gate_rows':len(gates),'historical_plateau_rows':len(convergence),'BNCI_plateau_rows':len(bnci_convergence) if new_convergence.exists() else 0,'new_inference_tests_performed':False,'historical_statistical_csvs_discovered':discovered,'represented_elsewhere':omissions,'unrepresented_statistical_csvs':[]},ensure_ascii=False,indent=2),encoding='utf-8')
x_tests = ROOT / 'docs/cross_model_tests.md'
if x_tests.exists():
    lines.append(x_tests.read_text(encoding='utf-8'))
budget_tests = ROOT / 'docs/population_budget_tests.md'
if budget_tests.exists():
    lines.append(budget_tests.read_text(encoding='utf-8'))
(ROOT/'docs/all_tests.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'comparison_rows':len(inventory),'gate_rows':len(gates),'source_tables':len(sources)},ensure_ascii=False))
