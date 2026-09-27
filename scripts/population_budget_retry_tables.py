"""Report the single authorized retry with the original complete-cohort statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from population_budget_tables import table, DATASETS, NAMES, POINTS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.dir.resolve()
    base = directory.relative_to(ROOT).as_posix()
    raw = directory/'retained'
    report = raw/'report'
    run = json.loads((raw/'CBraMod/runs/fold-1_seed-37/run.json').read_text())
    audit = json.loads((raw/'retry-integrity.json').read_text())
    assert run['status'] in ('complete','failed') and len(audit['runs']) == 1 and not audit['errors']
    retained = json.loads((directory/'retention.json').read_text())
    verification = json.loads((directory/'local_verification.json').read_text())
    assert verification['all_sha256_passed']
    summary = pd.read_csv(report/'curve_summary.csv')
    decisions = {r['model']:r for r in json.loads((report/'interpretation.json').read_text())}
    availability = json.loads((report/'availability.json').read_text())
    originals = pd.read_csv(ROOT/'reports/population_budget/final/report/curve_summary.csv')
    for model in ['REVE','LaBraM']:
        pd.testing.assert_frame_equal(originals.query('model == @model').reset_index(drop=True),
                                      summary.query('model == @model').reset_index(drop=True))
        assert decisions[model]['curve_interpretation'] == 'descriptive_only'
    main_rows, data_rows, stat_rows = [], [], []
    labels = {'descriptive_only':'如实描述，不作强结论',
              'primarily_population_undertraining':'个人收益主要反映群体模型训练不足',
              'stable_positive_after_continuation':'在群体模型持续训练后，个人收益稳定为正',
              'undetermined_missing_endpoints':'缺完整端点，未判定'}
    for model in ['CBraMod','REVE','LaBraM']:
        d = decisions[model]
        part = summary[(summary.model == model) & (summary.dataset == 'pooled')].set_index('point')
        curves = []
        for metric in ['G_BA_mean_percent','own_minus_G_median_pp','own_minus_swap_median_pp']:
            curves.append(' → '.join(f'{part.loc[p,metric]:.3f}' if p in part.index else 'NA' for p in POINTS))
        main_rows.append([model,*curves,labels[d['curve_interpretation']]])
        for dataset in DATASETS:
            for point in POINTS:
                rows = summary[(summary.model == model)&(summary.dataset == dataset)&(summary.point == point)]
                if rows.empty:
                    data_rows.append([model,NAMES[dataset],point,'NA','NA','NA','NA','NA','NA'])
                    continue
                r = rows.iloc[0]
                data_rows.append([model,NAMES[dataset],point,int(r.N),
                    f'{r.G_BA_mean_percent:.2f} ± {r.G_BA_SD_percent:.2f}',
                    f'{r.own_minus_G_mean_pp:+.2f} ± {r.own_minus_G_SD_pp:.2f}; 中位{r.own_minus_G_median_pp:+.3f}',
                    f'{r.own_minus_swap_mean_pp:+.2f} ± {r.own_minus_swap_SD_pp:.2f}; 中位{r.own_minus_swap_median_pp:+.3f}',
                    f'{100*r.own_minus_G_improve_gt_2pp_fraction:.1f}/{100*r.own_minus_G_drop_gt_2pp_fraction:.1f}',
                    f'{100*r.own_minus_swap_improve_gt_2pp_fraction:.1f}/{100*r.own_minus_swap_drop_gt_2pp_fraction:.1f}'])
        for point in POINTS:
            if point not in part.index:
                stat_rows.append([model,point,'NA','NA（缺失槽校正取1）','NA','NA','未判定'])
            else:
                r = part.loc[point]
                stat_rows.append([model,point,f'{r.own_minus_G_p_raw_descriptive:.6g}',
                    '描述性' if point == '1x' else f"{d['six_slot_Holm'][point]:.6g}",
                    f'{r.own_minus_swap_p_raw_descriptive:.6g}',f'{r.swap_original_three_slot_Holm:.6g}',
                    '通过' if r.swap_diagnostic1_pass else '未通过'])
    # Include the retry's real saved group endpoints even if its diagnostic failed.
    prior_val = pd.read_csv(ROOT/'reports/population_budget/final/report/endpoint_validation_including_failed_runs.csv')
    prior_val = prior_val[~((prior_val.model=='CBraMod')&(prior_val.fold==1)&(prior_val.seed==37))]
    val = pd.concat([prior_val,pd.DataFrame(audit['endpoint_validation_including_failed_runs'])],ignore_index=True)
    assert not val.duplicated(['model','fold','seed','point','dataset']).any()
    val.to_csv(directory/'endpoint_validation_final.csv',index=False)
    validation_rows=[]
    for model in ['CBraMod','REVE','LaBraM']:
        for dataset in DATASETS:
            for point in POINTS:
                v=val[(val.model==model)&(val.dataset==dataset)&(val.point==point)]
                validation_rows.append([model,NAMES[dataset],point,f'{len(v)}/25',
                    f'{v.CE.mean():.4f} ± {v.CE.std():.4f}',
                    f'{100*v.accuracy.mean():.2f} ± {100*v.accuracy.std():.2f}',
                    f'{100*v.BA.mean():.2f} ± {100*v.BA.std():.2f}'])
    accounting=pd.read_csv(raw/'retry_gpu_accounting.psv',sep='|')
    assert len(accounting)==1 and accounting.iloc[0].State in ['COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY']
    retry_hours=float(accounting.ElapsedRaw.sum()/3600)
    original_hours=json.loads((ROOT/'reports/population_budget/final/all_model_resources.json').read_text())['total_card_hours']
    progress=json.loads((report/'progress.json').read_text())
    groups=audit['runs'][0]
    frozen={'frozen_at_UTC':verification['verified_at_UTC'],'reason':'User requested all experiments frozen after this one retry',
        'GPU_retry_job':'1912202_57','retry_status':run['status'],'maximum_attempts':1,'no_further_GPU_runs':True,
        'original_batch_card_hours':original_hours,'retry_card_hours':retry_hours,'combined_card_hours':original_hours+retry_hours,
        'source_manifest':str((raw/'merge_manifest.json').relative_to(ROOT)),
        'classifications':{k:v['curve_interpretation'] for k,v in decisions.items()}}
    (directory/'experiment_freeze.json').write_text(json.dumps(frozen,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for model,color in [('CBraMod','#4d7c0f'),('REVE','#2563eb'),('LaBraM','#d97706')]:
        p=summary[(summary.model==model)&(summary.dataset=='pooled')].set_index('point')
        if set(p.index)!=set(POINTS):continue
        for axis,metric in zip(axes,['G_BA_mean_percent','own_minus_G_median_pp','own_minus_swap_median_pp']):
            axis.plot([1,2,4],[p.loc[x,metric] for x in POINTS],marker='o',label=model,color=color)
    for axis,title,ylabel in zip(axes,['Group performance','Own minus group','Own minus swapped'],['Mean BA (%)','Median difference (pp)','Median difference (pp)']):
        axis.set(title=title,xlabel='Total update budget',ylabel=ylabel,xticks=[1,2,4],xticklabels=['1x','2x','4x'])
        axis.spines[['top','right']].set_visible(False);axis.grid(alpha=.2)
    axes[1].axhline(.5,color='.5',linestyle=':',linewidth=1)
    axes[2].axhline(2,color='.5',linestyle=':',linewidth=1)
    axes[0].legend(frameon=False)
    fig.suptitle('Frozen final curves: 235 subjects, five-seed means per subject\nCBraMod: one retry with the recorded FP32 exception',fontsize=10)
    fig.savefig(directory/'three_model_curves.png',dpi=180);fig.savefig(directory/'three_model_curves.pdf');plt.close(fig)
    baseline_path=raw/'CBraMod/runs/fold-1_seed-37/diagnostic/1x/historical_comparison.json'
    comparison=json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    precision_comparison=('1×历史匹配对比'+f"{comparison['matched_trials']:,}条预测，改变{comparison['changed_predictions']}条；最大单被试×种子BA差（G/本人/交换）为"+
        '/'.join(f'{100*comparison["maximum_absolute_BA_difference"][k]:.6f}' for k in ['G_BA','own_ba','swap_ba'])+'pp。' if comparison else '此次未产生完整1×个人诊断；历史比较NA。')
    cb_matching=decisions['CBraMod']['personal_matching_diagnostic_pass']
    matching_text='，'.join(p.replace('x','×')+('通过' if cb_matching[p] is True else '未通过' if cb_matching[p] is False else '未判定') for p in POINTS)
    classification_consistency=('未判定（CBraMod缺完整端点）' if decisions['CBraMod']['curve_interpretation']=='undetermined_missing_endpoints' else
        '一致：均属描述性结果' if decisions['CBraMod']['curve_interpretation']=='descriptive_only' else '不一致：CBraMod与REVE/LaBraM分类不同')
    matching_consistency=('未判定（CBraMod有缺失）' if any(v is None for v in cb_matching.values()) else
        '一致：三个模型三点均通过' if all(cb_matching.values()) else '不一致：CBraMod存在未通过点')
    text=f'''# 三模型群体训练量曲线：唯一重跑后的冻结报告（2026-09-27）

**REVE与LaBraM按事先规则“如实描述，不作强结论”，不再追加训练。CBraMod唯一重跑状态：{'成功' if run['status']=='complete' else '失败'}；完整队列判读：{labels[decisions['CBraMod']['curve_interpretation']]}。本轮已结束，之后全部实验冻结。**

## 做了什么与数值精度例外

只重跑原失败的CBraMod fold1/seed37，作业1912202_57；原失败和另外74条保留。由同一G（S=3250）重新起步，目标总步数6500/13000，最多新增9750，沿用原平台与4×上限，未降低学习率或改变数据、目标、供体及个人网格。新运行实际成功新增{groups['sampled_updates']}步；{'未达到预设联合平台' if groups['first_platform_added_step'] is None else '首次达到联合平台的新增步数为'+str(groups['first_platform_added_step'])}。{('成功后仅替换原失败槽位，与另74条合并。' if run['status']=='complete' else '此次再次失败，保留缺失，不再重试；完整队列缺失不缩小分母。')}

预登记的数值措施仅作用于本运行适配模型：取消第8层激活的fp16往返，保留fp32，并关闭TF32；原群体梯度裁剪1.0保留。输入文件、归一化及静息上下文特征使用历史路径。其个人模型也采用相同精度，使G/本人/交换路径一致。完整CBraMod包含24条原精度运行与1条该例外，不能表述为完全相同的精度设置。此措施的成败不能单独证明原错误的数值根因。[重跑前规格](population_budget_retry_protocol.md)。

{precision_comparison}历史X1/W成绩不改，差异原表在该运行的`diagnostic/1x`保留。首批其他CBraMod运行已有一条G硬预测差异也继续披露于[首批合并报告](population_budget_initial_report_2026-09-27.md)。

## 统一统计量与合并结果

两项差值均为**同人先平均五种子，再对预定235人取中位数**，不是逐运行均值的范围。顺序为1×→2×→4×，差值单位pp。若缺完整端点，相应主统计NA。

{table(['模型','G平均BA（%）','本人−G中位（pp）','本人−交换中位（pp）','冻结规则判读'],main_rows)}

REVE的2×到4×本人−G中位变化绝对值0.833pp，LaBraM为0.606pp，均未满足<0.5pp；两者4×中位又不低于0.5pp，所以不触发任一强结论。本人的净收益与交换匹配价值分别解释；通过匹配诊断不等于所有收益都是个人独有，也不等于群体充分收敛。

CBraMod完整队列的2×与4×中位差为0.400pp，小于0.5pp；两点中位均正，六槽Holm校正p分别为5.69e-12和6.57e-13，因此按原规则判为“在群体模型持续训练后，个人收益稳定为正”。最终75条成功轨迹均未达到联合平台，这一判读不表示群体模型已充分收敛；三个模型均未触发“主要反映群体训练不足”，也不等于排除了训练不足。

![最终三模型训练量曲线](../{base}/three_model_curves.png)

图中G使用均值，两种差值使用中位数；无置信区间。缺完整队列的模型不绘线。REVE的4×群体平均BA退化，应与个人收益同时解释。

## 三模型×三库×预算点

N为被试数；所有均值±SD均跨每人五种子平均后的被试计算。最后两列为改善>2pp/下降>2pp的被试比例（%），分库为描述性结果。

{table(['模型','数据集','预算','N','G BA（%）','本人−G均值±SD及中位（pp）','本人−交换均值±SD及中位（pp）','本人−G改善/下降%','本人−交换改善/下降%'],data_rows)}

[完整逐被试×种子数据](../{base}/retained/report/all_subjects_and_actual_plateaus.csv) · [每人五种子平均](../{base}/retained/report/subject_seed_means.csv) · [精确统计及分库原始p](../{base}/retained/report/curve_summary.csv) · [235×5端点覆盖检查](../{base}/retained/report/availability.json)。

## 验证损失与准确率

以下为可用折/种子运行均值±SD，每次运行内被试等权；人数重复，不将25次验证当独立样本。缺失运行直接标注数量。[逐运行原值](../{base}/endpoint_validation_final.csv)。

{table(['模型','数据集','预算','可用运行','验证CE','验证accuracy（%）','验证BA（%）'],validation_rows)}

## 统计检验与限制

配对单侧Wilcoxon greater，差值12位、剔除零差、正态近似。本人−G的3模型×2×/4×仍为同一六槽Holm，替换失败槽后重算校正值；REVE/LaBraM原始数据、中位数与原始p保持不变。本人−交换每点仍是独立原三槽诊断（未执行变体p=1），不改历史X1/W家族。

{table(['模型','预算','本人−G原始p','本人−G六槽Holm','本人−交换原始p','本人−交换原三槽Holm','匹配诊断'],stat_rows)}

三模型的判读分别为CBraMod“{labels[decisions['CBraMod']['curve_interpretation']]}”、REVE/LaBraM“如实描述，不作强结论”。本人−交换诊断：CBraMod {matching_text}；REVE/LaBraM三点均通过。

| 跨模型一致性 | 冻结结果 |
|---|---|
| 群体平均BA随预算单调上升 | 不一致：CBraMod是、REVE否、LaBraM是；完整数值见主表 |
| 个人收益训练量分类 | {classification_consistency} |
| 三点本人−交换均通过 | {matching_consistency} |

仅按完整端点规则表述一致/不一致，不由某个正均值替代判据。原X1少样本结论未重算：三模型均稳定仅限PhysioNet；Dreyer、Cho低标签量结果仍随模型变化。

REVE的Dreyer/Cho为seen、PhysioNet为unlisted；LaBraM的PhysioNet为seen、另两库unlisted；CBraMod三库unlisted。三模型没有共同unlisted起步库，unlisted不等于逐记录排重。CBraMod原八基/上下文目标及优化器重启与其他骨干不同，此次另有单运行精度例外。测试均同session，供体共用，结果不推广到所有人或其他标签预算。

## 核验、资源与实验冻结

19项准备检查通过，原前向路径回归不变；CPU复核原{retained['prior_files_reverified']:,}份保留文件并审核此次终态。合并后重新复算{progress['audited_prediction_rows']:,}条预测；来源映射明确只替换原索引57。[来源清单](../{base}/retained/merge_manifest.json) · [重跑完整性审核](../{base}/retained/retry-integrity.json)。

本次GPU用时41分36秒，另耗{retry_hours:.3f}卡时；首批88.096卡时包含原失败，本检查累计{original_hours+retry_hours:.3f}卡时。此任务只加1张3090，其他项目GPU不混算。随后零GPU核验/归档1912210耗时4分36秒、0.307核时；[两项最终账目](../{base}/final_accounting.psv)。

新工件{retained['file_count']:,}份、{retained['retained_bytes']:,}字节已逐份核验保留home；本地{verification['verified_noncheckpoint_files']:,}份非检查点工件全部哈希通过。[保留清单](../{base}/retention.json) · [本地回执](../{base}/local_verification.json)。原失败与首批归档保持不变。

**本阶段到此停止，之后全部实验冻结**，见[冻结记录](../{base}/experiment_freeze.json)。不启动其他种子重跑、延长4×、新的稳定措施搜索或X1扩展。下一步仅可基于冻结结果写作；本文件是实验报告，未新增方法或实验。
'''
    old=ROOT/'docs/population_budget_report_2026-09-27.md'
    snapshot=ROOT/'reports/population_budget/initial_combined_report.md'
    if not snapshot.exists():snapshot.write_bytes(old.read_bytes())
    readable_snapshot=ROOT/'docs/population_budget_initial_report_2026-09-27.md'
    if not readable_snapshot.exists():readable_snapshot.write_bytes(snapshot.read_bytes())
    old.write_text(text,encoding='utf-8')
    (ROOT/'docs/population_budget_retry_report_2026-09-27.md').write_text(text,encoding='utf-8')
    # Machine-readable decisions are the source for the brief cross-model update.
    (directory/'final_decisions.json').write_text(json.dumps({'decisions':decisions,'main_rows':main_rows,
        'availability':availability,'retry_status':run['status'],'precision_comparison':comparison,
        'classification_consistency':classification_consistency,'matching_consistency':matching_consistency},indent=2)+'\n')
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if not link.startswith(('http','#')):
            assert (old.parent/link).exists(),link
    print(json.dumps(frozen,indent=2))


if __name__ == '__main__':
    main()
