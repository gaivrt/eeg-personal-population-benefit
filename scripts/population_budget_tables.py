"""Render already-audited complete-model budget curves; never aggregate partial models."""
import argparse
from datetime import datetime
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['PhysionetMI', 'Dreyer2023', 'Cho2017', 'pooled']
NAMES = {'PhysionetMI': 'PhysioNet MI', 'Dreyer2023': 'Dreyer2023', 'Cho2017': 'Cho2017', 'pooled': '合并235人'}
POINTS = ['1x', '2x', '4x']


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--final', action='store_true', help='Combine all terminal models, keeping incomplete cohorts NA')
    args = parser.parse_args()
    args.audit_dir = args.audit_dir.resolve()
    report = args.audit_dir / 'report'
    summary = pd.read_csv(report / 'curve_summary.csv')
    validation = pd.read_csv(report / 'endpoint_validation.csv')
    decisions = {r['model']: r for r in json.loads((report / 'interpretation.json').read_text())}
    resources = json.loads((args.audit_dir / 'resource_snapshot.json').read_text())
    accounting = pd.read_csv(io.StringIO(resources['accounting_psv']), sep='|', header=None,
                             names=['job','state','seconds','allocation','exit','start','end'])
    accounting = accounting[accounting.job.str.fullmatch(r'1911397_\d+')].copy()
    accounting['index'] = accounting.job.str.split('_').str[-1].astype(int)
    complete_models = list(summary.model.unique())
    base = args.audit_dir.relative_to(ROOT).as_posix()
    integrity_name = 'integrity-audit-final.json' if args.final else 'integrity-audit-1.json'
    notes = []
    for model in complete_models:
        part = summary[summary.model == model]
        if set(part[part.dataset == 'pooled'].point) != set(POINTS):
            raise ValueError('Incomplete model must not receive a complete report')
        run_rows = [r for r in resources['runs'] if r['model'] == model]
        if len(run_rows) != 25 or any(r['status'] != 'complete' for r in run_rows):
            raise ValueError('Resource records are not complete for this model')
        selected = accounting[accounting['index'].isin([r['array_index'] for r in run_rows])]
        if len(selected) != 25 or not (selected.state == 'COMPLETED').all():
            raise ValueError('Slurm completion missing')
        card_hours = selected.seconds.sum() / 3600
        wall_hours = (pd.to_datetime(selected.end).max() - pd.to_datetime(selected.start).min()).total_seconds()/3600
        peak = max(r['maximum_segment_peak_gpu_bytes'] for r in run_rows) / 1024**3
        decision = decisions[model]
        medians = decision['median_own_minus_G_pp']
        change = abs(medians['4x'] - medians['2x'])
        data_rows, validation_rows, test_rows = [], [], []
        for dataset in DATASETS:
            for point in POINTS:
                row = part[(part.dataset == dataset) & (part.point == point)].iloc[0]
                data_rows.append([NAMES[dataset], point.replace('x','×'), int(row.N),
                    f'{row.G_BA_mean_percent:.2f} ± {row.G_BA_SD_percent:.2f}',
                    f'{row.own_minus_G_mean_pp:+.2f} ± {row.own_minus_G_SD_pp:.2f}', f'{row.own_minus_G_median_pp:+.2f}',
                    f'{row.own_minus_swap_mean_pp:+.2f} ± {row.own_minus_swap_SD_pp:.2f}', f'{row.own_minus_swap_median_pp:+.2f}'])
                values = validation[(validation.model == model) & (validation.dataset == dataset) & (validation.point == point)]
                if len(values) != 25:
                    raise ValueError('Missing endpoint validation values')
                validation_rows.append([NAMES[dataset], point.replace('x','×'),
                    f'{values.CE.mean():.4f} ± {values.CE.std():.4f}',
                    f'{100*values.accuracy.mean():.2f} ± {100*values.accuracy.std():.2f}',
                    f'{100*values.BA.mean():.2f} ± {100*values.BA.std():.2f}'])
                test_rows.append([NAMES[dataset], point.replace('x','×'),
                    f'{100*row.own_minus_G_improve_gt_2pp_fraction:.1f} / {100*row.own_minus_G_drop_gt_2pp_fraction:.1f}',
                    f'{row.own_minus_G_p_raw_descriptive:.3g}',
                    f'{100*row.own_minus_swap_improve_gt_2pp_fraction:.1f} / {100*row.own_minus_swap_drop_gt_2pp_fraction:.1f}',
                    f'{row.own_minus_swap_p_raw_descriptive:.3g}'])
        pooled = part[part.dataset == 'pooled'].set_index('point')
        gain_curve = ' → '.join(f'{medians[p]:.3f}' for p in POINTS)
        matching = ' → '.join(f'{pooled.loc[p].own_minus_swap_median_pp:.3f}' for p in POINTS)
        g_curve = ' → '.join(f'{pooled.loc[p].G_BA_mean_percent:.2f}' for p in POINTS)
        classification = {'descriptive_only': '按冻结规则仅作描述，不作两类强结论',
                          'stable_positive_after_continuation': '在群体模型持续训练后，个人收益稳定为正',
                          'primarily_population_undertraining': '个人收益主要反映群体模型训练不足'}[decision['curve_interpretation']]
        caveat = ('REVE的群体平均BA在4×低于1×及2×，不能把该终点称为更强或已收敛的群体模型。'
                  if model == 'REVE' else 'LaBraM群体平均BA随预算增加上升；本人−G均值接近，但规则预先使用中位数，不改用均值判定稳定。')
        seen = ('Dreyer2023与Cho2017为seen，PhysioNet为unlisted。' if model == 'REVE' else
                'PhysioNet EEGMMIDB为seen，Dreyer2023与Cho2017为unlisted。')
        stat_rows = [[p.replace('x','×'), f'{pooled.loc[p].own_minus_G_p_raw_descriptive:.3g}',
                      '描述性' if p == '1x' else f"{decision['six_slot_Holm'][p]:.3g}",
                      f'{pooled.loc[p].swap_original_three_slot_Holm:.3g}', '通过' if pooled.loc[p].swap_diagnostic1_pass else '未通过'] for p in POINTS]
        content = f'''# {model} 群体训练量曲线报告（2026-09-27）

完成25/25条群体续训与三点个人诊断，覆盖原五折、五种子、起步三库235人。目标IJCAI 2027。结果已通过逐试次复算及来源/日志/供体完整性复核；本文件为已完成模型的分项报告，CBraMod和全批保留工作见总汇总。

## 做了什么与规格偏离

沿用现有G、原算法和超参数，1×为每次已报告G的实际累计更新步数S；仅续至2S与4S。本模型25条均在4×硬上限停止，无预设联合平台。未更改科学判读规则或重训B0。新增个人参数仅按原验证网格选择，1×复用原参数；同折同库原供体顺序及十次抽样全部保留。

三库联合训练，资源不重复乘三。训练与验证平台未满足，结果仅覆盖原学习率与有限预算。{seen}unlisted不是逐记录排重证明。CBraMod的群体参数化、目标和优化器重启不同，跨模型对照不构成只改变骨干的因果实验。

## 结果表

按人先平均五种子，再报告被试间均值±SD；G为BA百分比，差值为百分点（pp）。合并235人是预先固定的主判读集合，分库属于描述。

{table(['数据集','预算','N','G BA（%）','本人−G 均值±SD','本人−G 中位','本人−交换 均值±SD','本人−交换 中位'], data_rows)}

[每被试五种子平均](../{base}/report/subject_seed_means.csv) · [全部被试×种子×检查点](../{base}/report/all_subjects_and_actual_plateaus.csv) · [精确汇总及完整SD](../{base}/report/curve_summary.csv)。

## 验证损失与准确率

下表对25次折/种子运行等权报告均值±SD；每次运行内部为验证被试等权。仅为曲线描述，不将重复验证被试当作独立统计样本。完整逐运行端点见[验证值](../{base}/report/endpoint_validation.csv)，逐epoch和平台完整性见[审核](../{base}/{integrity_name})。

{table(['数据集','预算','验证CE','验证accuracy（%）','验证BA（%）'], validation_rows)}

## 统计检验及冻结判读

本人−G合并中位曲线：**{gain_curve} pp**；2×与4×差的绝对值为**{change:.3f} pp**。4×中位数为{medians['4x']:.3f}pp，不低于0.5pp；稳定性变化门槛为严格小于0.5pp。结论：**{classification}**。

本人−交换合并中位曲线：**{matching} pp**。所有三点仍通过原三槽位诊断1（中位≥2pp且Holm p<.05），支持个人参数匹配价值；不等于全部收益为个人特异。

{table(['预算','本人−G 原始p','本人−G 六槽Holm p','本人−交换 三槽Holm p','匹配诊断'], stat_rows)}

检验单位为同人五种子平均后的235人，配对单侧Wilcoxon greater，12位差值、剔除零差、正态近似。六槽家族固定为三模型×2×/4×，缺失CBraMod槽位保留p=1、实际结果NA。原本人−交换三槽位家族独立，不追溯修改X1或CBraMod原W检验。

{table(['数据集','预算','本人−G 改善/下降>2pp（%）','本人−G 原始p','本人−交换 改善/下降>2pp（%）','本人−交换 原始p'], test_rows)}

## 资源、问题与下一步

本模型实际分配**{card_hours:.3f}卡时**，模型开始至最后完成墙钟**{wall_hours:.3f}小时**（与其他模型重叠，不相加）；已记录最大分段分配显存**{peak:.2f}GiB**。单作业一卡，总批并发未超过16卡。资源来源为[原始Slurm快照](../{base}/resource_snapshot.json)。1×与历史预测比较覆盖1,454,675条记录，改变预测数为0；原X1保留不变。

群体合并平均BA为**{g_curve}%**。{caveat}无平台证据，不能把持续训练后的正个人收益改写为“群体充分收敛后仍稳定”。继续完成既有全批作业和保留核验；不增加预算或启动扩展。
'''
        if args.final:
            content = content.replace('继续完成既有全批作业和保留核验；不增加预算或启动扩展。',
                                      '全部GPU作业已结束；合并报告见[三模型总报告](population_budget_report_2026-09-27.md)，本阶段停止。')
        (ROOT / 'docs' / f'population_budget_{model}_report_2026-09-27.md').write_text(content, encoding='utf-8')
        notes.append({'model': model, 'card_hours': float(card_hours), 'wall_hours': wall_hours, 'peak_segment_GiB': peak,
                      'own_G_median_2x_to_4x_change_pp': change, 'classification': decision['curve_interpretation']})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
    for model, color in zip(complete_models, ['#2563eb', '#d97706', '#4d7c0f']):
        pooled = summary[(summary.model == model) & (summary.dataset == 'pooled')].set_index('point')
        for axis, metric in zip(axes, ['G_BA_mean_percent','own_minus_G_median_pp','own_minus_swap_median_pp']):
            axis.plot([1,2,4], [pooled.loc[p,metric] for p in POINTS], marker='o', color=color, label=model)
    for axis, title, unit in zip(axes, ['Group performance','Own minus group','Own minus swapped'], ['Mean BA (%)','Median difference (pp)','Median difference (pp)']):
        axis.set(title=title, xlabel='Total update budget', ylabel=unit, xticks=[1,2,4], xticklabels=['1x','2x','4x'])
        axis.spines[['top','right']].set_visible(False)
        axis.grid(alpha=.2)
    axes[1].axhline(.5, color='.5', linestyle=':', linewidth=1)
    axes[2].axhline(2, color='.5', linestyle=':', linewidth=1)
    axes[0].legend(frameon=False)
    fig.suptitle('Complete-model curves: 235 subjects, five-seed means per subject', fontsize=11)
    fig.savefig(args.audit_dir / 'complete_model_curves.png', dpi=180)
    fig.savefig(args.audit_dir / 'complete_model_curves.pdf')
    plt.close(fig)
    (args.audit_dir / 'complete_model_resources.json').write_text(json.dumps(notes, indent=2)+'\n')
    if args.final:
        final_report(args.audit_dir, resources, accounting, notes)
    print(json.dumps(notes, indent=2))


def final_report(directory, resources, accounting, complete_resources):
    """Combine existing complete-model reports and explicit missing-endpoint annexes."""
    base = directory.relative_to(ROOT).as_posix()
    report = directory / 'report'
    integrity = json.loads((directory / 'integrity-audit-final.json').read_text())
    assert len(integrity['runs']) == 75 and not integrity['errors']
    assert integrity['counts'] == {'complete': 74, 'failed': 1}
    assert len(accounting) == 75 and set(accounting.state) == {'COMPLETED', 'FAILED'}
    assert sum(accounting.state == 'FAILED') == 1
    people = pd.read_csv(report / 'all_subjects_and_actual_plateaus.csv')
    cb = people[people.model == 'CBraMod']
    assert set(cb.point) == set(POINTS)
    # Per-run descriptions expose every successful observation without pooling
    # 24 runs or averaging four seeds as though the preregistered cohort were complete.
    run_rows = []
    for record in [r for r in resources['runs'] if r['model'] == 'CBraMod']:
        for point in POINTS:
            for dataset in DATASETS:
                part = cb[(cb.fold == record['fold']) & (cb.seed == record['seed']) & (cb.point == point)]
                if dataset != 'pooled':
                    part = part[part.dataset == dataset]
                row = {k: record[k] for k in ['model','fold','seed','status']}
                row.update(point=point, dataset=dataset, observed_subjects=len(part))
                for key in ['G_BA','own_gain','upper_bound']:
                    row[key+'_mean_percent_or_pp'] = 100 * part[key].mean()
                    row[key+'_SD_percent_or_pp'] = 100 * part[key].std(ddof=1)
                    row[key+'_median_percent_or_pp'] = 100 * part[key].median()
                run_rows.append(row)
    run_table = pd.DataFrame(run_rows)
    run_table.to_csv(report / 'CBraMod_per_run_descriptive.csv', index=False)
    ranges = []
    for point in POINTS:
        part = run_table[(run_table.point == point) & (run_table.dataset == 'pooled') & (run_table.status == 'complete')]
        assert len(part) == 24
        ranges.append([point.replace('x','×'),'24 / 25',
            *[f'{part[k].min():.3f} ～ {part[k].max():.3f}' for k in
              ['G_BA_mean_percent_or_pp','own_gain_mean_percent_or_pp','upper_bound_mean_percent_or_pp']],
            f"{int((part.own_gain_mean_percent_or_pp > 0).sum())}/24；{int((part.upper_bound_mean_percent_or_pp > 0).sum())}/24"])
    roster = json.loads((ROOT / 'configs/splits/stage_b_halves_v1.json').read_text())['subjects']
    full_index = pd.MultiIndex.from_tuples([(r['dataset'],r['subject'],s,p) for r in roster
        for s in [11,23,37,53,71] for p in POINTS], names=['dataset','subject','seed','point'])
    full = cb.set_index(['dataset','subject','seed','point']).reindex(full_index)
    full['observed'] = full.G_BA.notna()
    full.reset_index().to_csv(report / 'CBraMod_full_roster_with_missing.csv', index=False)
    complete_subjects = full.groupby(level=['dataset','subject','point'])['observed'].sum()
    coverage = []
    for dataset in DATASETS:
        for point in POINTS:
            part = full.reset_index().query('point == @point')
            n = len(roster)
            counts = complete_subjects.xs(point, level='point')
            if dataset != 'pooled':
                part = part[part.dataset == dataset]
                n = sum(r['dataset'] == dataset for r in roster)
                counts = counts.xs(dataset, level='dataset')
            coverage.append([NAMES[dataset],point.replace('x','×'),f'{int(part.observed.sum())} / {5*n}',
                             f'{int((counts == 5).sum())} / {n}', 'NA'])
    val = pd.DataFrame(integrity['endpoint_validation_including_failed_runs'])
    val.to_csv(report / 'endpoint_validation_including_failed_runs.csv', index=False)
    val_rows = []
    for dataset in DATASETS:
        for point in POINTS:
            v = val[(val.model == 'CBraMod') & (val.dataset == dataset) & (val.point == point)]
            assert len(v) == (24 if point == '4x' else 25)
            val_rows.append([NAMES[dataset],point.replace('x','×'),len(v),
                f'{v.CE.mean():.4f} ± {v.CE.std():.4f}',
                f'{100*v.accuracy.mean():.2f} ± {100*v.accuracy.std():.2f}',
                f'{100*v.BA.mean():.2f} ± {100*v.BA.std():.2f}'])
    resource_rows, resource_records = [], []
    for model in ['CBraMod','REVE','LaBraM']:
        rr = [r for r in resources['runs'] if r['model'] == model]
        ac = accounting[accounting['index'].isin([r['array_index'] for r in rr])]
        hours = float(ac.seconds.sum()/3600)
        wall = float((pd.to_datetime(ac.end).max()-pd.to_datetime(ac.start).min()).total_seconds()/3600)
        peak = max(r['maximum_segment_peak_gpu_bytes'] or 0 for r in rr)/1024**3
        resource_rows.append([model,sum(r['status']=='complete' for r in rr),sum(r['status']=='failed' for r in rr),
                              f'{hours:.3f}',f'{wall:.3f}',f'{peak:.2f}'])
        resource_records.append(dict(model=model,card_hours=hours,wall_hours=wall,peak_segment_GiB=peak))
    events = sorted([(pd.Timestamp(r.start),1) for r in accounting.itertuples()]+
                    [(pd.Timestamp(r.end),-1) for r in accounting.itertuples()])
    live = peak_jobs = 0
    for _,delta in events:
        live += delta
        peak_jobs = max(peak_jobs,live)
    assert live == 0 and peak_jobs <= 16
    total_hours = float(accounting.seconds.sum()/3600)
    wall = (pd.to_datetime(accounting.end).max()-pd.to_datetime(accounting.start).min()).total_seconds()/3600
    new_updates = sum(r['sampled_updates'] for r in integrity['runs'])
    platform_count = sum(r['first_platform_added_step'] is not None for r in integrity['runs'])
    assert platform_count == 0 and new_updates <= 756750
    (directory / 'all_model_resources.json').write_text(json.dumps({'models':resource_records,
        'total_card_hours':total_hours,'batch_wall_hours':wall,'peak_concurrent_GPUs':peak_jobs,
        'successful_new_group_updates':new_updates,'platform_count':platform_count},indent=2)+'\n')
    count = json.loads((report / 'progress.json').read_text())['audited_prediction_rows']
    verification = json.loads((directory / 'local_verification.json').read_text())
    assert verification['all_sha256_passed']
    baseline_rows = [[r['model'],r['runs'],f"{r['matched_trials']:,}",r['changed_predictions'],
                      f"{100*r['max_BA_difference']:.6f}"] for r in verification['historical_1x_comparisons']]
    cpu = pd.read_csv(directory / 'cpu_accounting.psv', sep='|')
    assert (cpu.State == 'COMPLETED').all() and (cpu.ExitCode == '0:0').all()
    assert not cpu.AllocTRES.str.contains('gres/gpu').any()
    cpu_hours = (cpu.ElapsedRaw*cpu.AllocCPUS).sum()/3600
    observed_per_point = int(full.xs('1x', level='point')['observed'].sum())
    five_seed_people = int((complete_subjects.xs('1x', level='point') == 5).sum())
    intro = f'''# 三模型群体训练量曲线：合并报告（2026-09-27）

**全批结束：74条成功，1条CBraMod按预设数值失败停止。REVE、LaBraM续训后合并本人−G和本人−交换仍为正，但都未触发预设的“主要反映训练不足”或“稳定正”强结论；CBraMod缺完整端点，三模型一致性未判定。** 目标会议为用户确定的IJCAI 2027。

## 做了什么与规格偏离

三个基础模型各5折×5种子，起步三库联合训练，共75条轨迹。从现有报告G的累计步数S出发，保存1×、2×、4×，检查G成绩、本人−G及本人−交换；超参只在原验证被试上选择，原供体和个人r8设置不变。1×重用历史个人参数，2×/4×按同一网格重新拟合。原X1与本轮检查分列，避免把历史结果补作本轮缺失值。

科学规则和上限未改变。74条成功轨迹均到4×预算上限，75条均无联合平台证据。CBraMod fold1/seed37在下一次更新（累计10700步，约3.2923×）计算出非有限目标并停止；最后成功累计10699步，1×/2×群体检查点存在，4×及该条三点个人诊断缺失。这是预登记的失败停止，未自动重试或剔除种子。[协议](population_budget_protocol.md) · [失败现场](population_budget_numerical_failure_2026-09-27.md)。

REVE/LaBraM恢复原优化器和RNG，CBraMod历史只存参数而按原超参/种子重启；CBraMod沿用八个共享r8基和原上下文目标，新骨干沿用普通群体r8与CE。三模型不同预处理和预训练重合限制继续保留，不将差异归因于单一骨干。

REVE的Dreyer2023/Cho2017为seen、PhysioNet为unlisted；LaBraM的PhysioNet为seen、其余两库为unlisted；CBraMod三库均为unlisted。三模型没有共同的unlisted起步库，且unlisted仅指公开来源未列入，不是逐记录去重证明。结果分库报告，不将seen库当作独立未见来源复现。

## 三模型并列结论

下表差值均为完整235人先平均五种子后的中位数（pp），依次1×→2×→4×。CBraMod的NA指本轮完整队列不可得，并非所有已完成运行没有结果。

| 模型 | 成功/计划 | G平均BA（%） | 本人−G中位（pp） | 本人−交换中位（pp） | 冻结判读 |
|---|---|---|---|---|---|
| CBraMod | 24/25；1失败 | NA | NA | NA | 缺完整端点，未判定 |
| REVE | 25/25 | 81.13→81.80→79.81 | 2.424→1.833→1.000 | 3.500→3.100→2.240 | 仅描述 |
| LaBraM | 25/25 | 77.81→78.46→78.69 | 2.424→2.576→1.970 | 3.174→2.867→2.939 | 仅描述 |

REVE的个人收益随预算缩小，但4×中位仍≥0.5pp；LaBraM不单调缩小。2×到4×中位变化分别0.833pp与0.606pp，均不满足<0.5pp，因此即使本人−G显著为正，也不能判为“稳定正”。两模型三点均通过原本人−交换门槛，支持个人参数的匹配价值，不能据此认定全部收益都只属于个人。REVE的G在4×退化，尤其不能称其充分收敛或更强。

| 一致性问题 | 本轮结论 |
|---|---|
| 群体平均BA随训练量增加单调上升 | REVE否、LaBraM是，两者不一致；CBraMod完整队列NA |
| 两类强判读 | REVE/LaBraM一致落在描述性类别；三模型未判定 |
| 三点本人−交换均通过原门槛 | REVE/LaBraM一致通过；三模型未判定 |
| 群体充分收敛后的个人收益 | 三模型均无联合平台证据，未判定 |

历史X1（含原CBraMod W）三模型的本人−交换合并中位分别为CBraMod **3.533pp**、REVE **3.500pp**、LaBraM **3.174pp**，均通过各自原诊断1；这是已有的跨模型匹配证据。本轮CBraMod缺失不撤销原结论，也不由原结果填补新曲线。[原X1三模型汇总及少样本曲线](cross_model_summary.md)。

![完整队列的训练量曲线](../{base}/complete_model_curves.png)

图中G用均值，两种差值用中位数；未画置信区间。CBraMod完整队列NA不绘线。以下将三模型逐库结果、验证值、检验和全部观察值入口放在同一文件。

## CBraMod：完整队列缺失与全部可用观察

不将四种子均值冒充五种子，不删除缺失被试后进行主检验。每个端点成功诊断{observed_per_point}/1175个被试×种子，{five_seed_people}/235人齐五种子，其余{235-five_seed_people}人缺seed37。G/本人−G/本人−交换完整队列统计及六槽Holm实际结果均为NA（校正时两个缺失槽保留p=1）。

{table(['数据集','预算','已观察被试×种子 / 计划','齐五种子人数 / 计划','完整队列(a)(b)(c)'],coverage)}

24条成功轨迹全部保留。[逐运行×三库×三点G及差值均值±SD/中位](../{base}/report/CBraMod_per_run_descriptive.csv) · [235人×五种子×三点完整索引，缺失明确留空](../{base}/report/CBraMod_full_roster_with_missing.csv) · [所有实际被试值](../{base}/report/all_subjects_and_actual_plateaus.csv)。逐运行描述未合成为少一条轨迹的主判读，也不执行部分队列的显著性检验。

下面直接列出可用24条运行各自测试折的均值范围，单次测试人数46–48，不是235人五种子平均。4×时24条的本人−G及本人−交换逐运行均值均为正；这支持已观察运行中的正方向描述，不构成完整队列通过、统计显著或稳定性的判定。

{table(['预算','可用运行','逐运行G平均BA范围（%）','逐运行本人−G均值范围（pp）','逐运行本人−交换均值范围（pp）','均值>0运行数：本人−G；本人−交换'],ranges)}

### CBraMod验证损失与准确率

1×和2×包括失败轨迹已保存的G，共25条；4×仅24条，表中直接标注。以下是可用运行验证值的均值±SD，不是完整测试队列成绩；各预算集合不同，不用于个人收益强判读。每次运行内部为验证被试等权，重复验证集合不作独立样本推断。

{table(['数据集','预算','可用运行/25','验证CE','验证accuracy（%）','验证BA（%）'],val_rows)}

[全部模型逐运行验证端点（含失败现场）](../{base}/report/endpoint_validation_including_failed_runs.csv)。逐epoch日志均在保留目录内，审核无缺漏。CBraMod三库预训练来源均为unlisted；这不是逐记录排重证明。

## REVE与LaBraM：全部三库结果和检验

下述统计严格使用各自完整235人×五种子。分库描述不替代合并主判读，新增六槽Holm不混入历史本人−交换家族。

'''
    sections=[]
    for model in ['REVE','LaBraM']:
        text=(ROOT/'docs'/f'population_budget_{model}_report_2026-09-27.md').read_text(encoding='utf-8')
        sections.append('## '+model+'\n\n'+text[text.index('## 结果表'):].replace('\n## ','\n### ').removeprefix('## ').replace('结果表\n','### 结果表\n',1))
    ending=f'''

## 全批核验、资源与停止

最终零GPU复核1912160：{count:,}条预测重新计算通过；全部75条终态的来源/配置/代码哈希、逐epoch日志、平台窗口、确切端点、有限参数、历史供体与验证选择审核无差异。数值失败仍计作失败，不因现场审核通过改记训练成功。[完整性复核](../{base}/integrity-audit-final.json) · [预测复核](../{base}/report/progress.json) · [资源原始记录](../{base}/resource_snapshot.json)。

1×匹配批次路径与历史预测的比较如下；“复算通过”指本轮保存预测能复现本轮成绩，不等于所有历史硬预测完全相同。

{table(['模型','可比较运行','匹配预测记录','硬预测改变数','最大单被试×种子BA变化（pp）'],baseline_rows)}

CBraMod唯一差异在fold4/seed11、PhysioNet被试87的G预测，试次`PhysionetMI:087:0038`：历史预测0，本轮匹配路径预测1，真实标签1。该人该种子G的BA从60.6061%到64.7727%，本人−G相应减少4.1667pp；本人和交换成绩不变。1×来源参数逐张量核验相同，未把该差异归因为已证实的具体浮点原因；按结果前协议保留两套值和逐试次差异，历史W/X1不改。本轮使用匹配路径值，CBraMod完整曲线判读仍为NA。[该运行差异原表](../reports/population_budget/retained/CBraMod/runs/fold-4_seed-11/diagnostic/1x/historical_score_comparison.csv)。

{table(['模型','成功','失败','分配卡时','模型墙钟小时','已记录最大分段分配显存GiB'],resource_rows)}

合计**{total_hours:.3f}卡时**（含失败）、GPU墙钟**{wall:.3f}小时**，11:48:30开始、17:41:47最后结束；模型时间互相重叠不相加，三库联合运行不重复乘三。最大同时使用**{peak_jobs}张RTX3090**。实际成功新增群体更新{new_updates:,}步，计划上限756,750步；失败的下一次前向未完成更新。显存是记录的分段PyTorch分配峰值，不是设备总使用量。

完整检查点、必要1×个人参数、日志、预测、选择及哈希保留在`/path/to/results/eeg-cross-subject/population-budget-20260927`；本地保存非检查点可复算工件。[最终保留清单](../{base}/retention_manifest.json) · [本地逐文件核验](../{base}/local_verification.json)。CPU核验和归档不占GPU，单独列入[CPU作业账目](../{base}/cpu_accounting.psv)。

最终清单共{verification['retained_home_files']:,}份文件、{verification['retained_home_bytes']:,}字节（十进制{verification['retained_home_bytes']/1e9:.3f}GB）；本地{verification['verified_noncheckpoint_files']:,}份非检查点工件、{verification['verified_noncheckpoint_bytes']:,}字节，全部SHA256通过。五项CPU核验/归档作业共分配{cpu_hours:.3f}核时，GPU为零；上述GPU卡时不包含这些CPU核时。

本阶段按上限结束，所有已完成结果与失败现场合并交付，停止自动跟进。下一步写作可使用历史三模型匹配证据及本次两模型曲线，但不能声称三模型均在充分收敛G上稳定正；本轮不追加训练、不重试CBraMod、不启动X1扩展。没有基于已看测试集增加新的消融或方法选择；保留现有两个独立审核器用于逐试次复算与日志/状态核验，未另建重复训练流程。
'''
    (ROOT/'docs/population_budget_report_2026-09-27.md').write_text(intro+'\n\n'.join(sections)+ending,encoding='utf-8')


if __name__ == '__main__':
    main()
