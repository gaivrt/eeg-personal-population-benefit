"""Close the authorized one-run retry using audited data; never submits experiments."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from population_budget_tables import table, NAMES, POINTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/population_budget/retry1'
DOC = ROOT / 'docs'


def read(path):
    return path.read_text(encoding='utf-8')


def save(path, value):
    path.write_text(value, encoding='utf-8')


def snapshot(path):
    target = path.with_name(path.stem + '_before_retry' + path.suffix)
    if not target.exists():
        target.write_bytes(path.read_bytes())
    return target


def main():
    raw = OUT / 'retained/report'
    summary = pd.read_csv(raw / 'curve_summary.csv')
    people = pd.read_csv(raw / 'all_subjects_and_actual_plateaus.csv')
    decisions = json.loads(read(OUT / 'final_decisions.json'))['decisions']
    merge = json.loads(read(OUT / 'retained/merge_manifest.json'))['runs']
    assert len(merge) == 75 and all(r['status'] == 'complete' for r in merge)
    assert [r['array_index'] for r in merge if r['retry_precision_exception']] == [57]
    keys = ['model', 'point', 'dataset', 'subject']
    assert len(people) == 3 * 3 * 235 * 5
    assert not people.duplicated(keys + ['seed']).any()
    assert all(set(g.seed) == {11, 23, 37, 53, 71} for _, g in people.groupby(keys))
    means = people.groupby(keys)[['G_BA', 'own_ba', 'swap_ba', 'own_gain', 'upper_bound']].mean().reset_index()
    checks = []
    raw_p = {}
    for row in summary.itertuples():
        sub = means[(means.model == row.model) & (means.point == row.point)]
        if row.dataset != 'pooled':
            sub = sub[sub.dataset == row.dataset]
        assert len(sub) == row.N
        np.testing.assert_allclose(100 * sub.G_BA.mean(), row.G_BA_mean_percent, atol=1e-10, rtol=0)
        for name, col in [('own_minus_G', 'own_gain'), ('own_minus_swap', 'upper_bound')]:
            values = np.round(sub[col].to_numpy(), 12)
            np.testing.assert_allclose(100 * np.median(values), getattr(row, name + '_median_pp'), atol=1e-10, rtol=0)
            nz = values[values != 0]
            p = float(wilcoxon(nz, alternative='greater', method='approx').pvalue)
            np.testing.assert_allclose(p, getattr(row, name + '_p_raw_descriptive'), rtol=1e-9, atol=1e-45)
            if row.dataset == 'pooled' and name == 'own_minus_G' and row.point != '1x':
                raw_p[row.model, row.point] = p
        checks.append([row.model, row.point, row.dataset, row.N])
    # Independent sequential Holm calculation; same preregistered six slots.
    corrected = {}
    previous = 0.
    for i, (key, p) in enumerate(sorted(raw_p.items(), key=lambda kv: kv[1])):
        previous = min(1., max(previous, (6-i) * p))
        corrected[key] = previous
        np.testing.assert_allclose(previous, decisions[key[0]]['six_slot_Holm'][key[1]], rtol=1e-9, atol=1e-45)
    val = pd.read_csv(OUT / 'endpoint_validation_final.csv')
    assert (val.groupby(['model', 'dataset', 'point']).size() == 25).all()
    assert len(val.groupby(['model', 'dataset', 'point'])) == 36
    prefix = (DOC / 'all_tests.md').read_bytes()[:180205]
    prefix_hash = hashlib.sha256(prefix).hexdigest()
    assert prefix_hash == 'e77174c675fc03c70b358d54661d641d744aed499b26ea2c9e6c84be8219c184'
    original = pd.read_csv(ROOT / 'reports/population_budget/final/report/curve_summary.csv')
    for model in ['REVE', 'LaBraM']:
        pd.testing.assert_frame_equal(original.query('model == @model').reset_index(drop=True),
                                      summary.query('model == @model').reset_index(drop=True))

    pooled_rows, dataset_rows, stat_rows = [], [], []
    for model in ['CBraMod', 'REVE', 'LaBraM']:
        d = decisions[model]
        for dataset in ['pooled', 'PhysionetMI', 'Dreyer2023', 'Cho2017']:
            rows = summary[(summary.model == model) & (summary.dataset == dataset)].set_index('point')
            values = [' → '.join(f'{rows.loc[p,c]:.3f}' for p in POINTS)
                      for c in ['G_BA_mean_percent','own_minus_G_median_pp','own_minus_swap_median_pp']]
            if dataset == 'pooled':
                label = '持续训练后稳定为正' if model == 'CBraMod' else '如实描述，不作强结论'
                pooled_rows.append([model, *values, label])
                stat_rows.append([model, values[1], ' / '.join(f'{d["six_slot_Holm"][p]:.3g}' for p in ['2x','4x']),
                                  values[2], ' / '.join(f'{rows.loc[p,"swap_original_three_slot_Holm"]:.3g}' for p in POINTS), label])
            else:
                dataset_rows.append([model,NAMES[dataset],*values])
    pooled_table = table(['模型','G平均BA（%）','本人−G中位（pp）','本人−交换中位（pp）','冻结判读'],pooled_rows)
    stats_table = table(['模型','本人−G中位1×→2×→4×（pp）','2× / 4×六槽Holm p','本人−交换中位1×→2×→4×（pp）','本人−交换各点三槽Holm p','判读'],stat_rows)
    conclusion = '''CBraMod的2×到4×本人−G中位变化为0.400pp，小于0.5pp；两点中位均正，六槽Holm p分别为5.69e-12、6.57e-13，按原规则判为“在群体模型持续训练后，个人收益稳定为正”。REVE、LaBraM的变化分别为0.833pp、0.606pp，仍为“如实描述，不作强结论”。三个模型均未触发“个人收益主要反映群体模型训练不足”，这不等于排除训练不足。

三模型在三个预算点的本人−交换均通过原三槽位诊断门槛，支持既定供体程序下的个人参数匹配价值；不表示全部收益都是个人独有。最终75条成功群体轨迹均未达到联合平台，不能将CBraMod的预算稳定判读写成“已充分收敛后”的结论。'''
    precision = '''唯一重跑取消适配模型第8层激活的fp16往返、保留fp32，并关闭TF32；原群体梯度裁剪1.0不变，其余配置、数据、预算、选择和供体保持原协议。完整CBraMod队列包含24条原精度运行和1条上述例外。该运行1×历史对比56,706条预测中3条硬预测变化，G/本人BA最大差均为0，交换BA最大单人单种子差0.300pp。首批另外24条中已有1条G硬预测变化（单人单种子G BA差4.1667pp）继续保留；历史W/X1原报告不改。这些差异与数值措施均已披露，重跑成功不证明原故障根因。'''
    resources = '''本次GPU1912202_57于19:12:04完成，用时41分36秒、0.693卡时；CPU1912210于19:17:06完成，用时4分36秒、0.307核时、零GPU。群体新增9750步，总13000步，在4×上限停止。首批含原失败88.096卡时，本检查累计88.789卡时。原3,720份保留文件重新核验，新126份工件已保留home，本地120份非检查点工件全部SHA256通过；合并复算13,092,075条预测，只替换原索引57。'''
    budget_section = f'''## 用户新增的群体训练量检查（唯一重跑完成，全部实验冻结）

目标会议维持IJCAI 2027。每运行已报告G的实际累计更新数S为1×，原规则、平台和4×上限见[结果前规格](population_budget_protocol.md)。用户在首批结果后仅授权CBraMod fold1/seed37重跑一次，[数值稳定例外在重跑前登记](population_budget_retry_protocol.md)；此次成功，和另74条合并后，三个模型均为25/25完整轨迹，每个预算点均覆盖预定235人×五种子。原失败及首批结果保留于[首批报告](population_budget_initial_report_2026-09-27.md)，不覆盖、不另作成功次数。

两项差值均为**同人先平均五种子，再对235人取中位数**。下表按1×→2×→4×排列；pp为百分点。

{pooled_table}

{conclusion}

| 一致性问题 | 最终三模型结果 |
|---|---|
| 群体合并平均BA随预算单调上升 | **不一致**：CBraMod、LaBraM上升；REVE 4×退化 |
| 个人收益满足持续训练后稳定为正 | **不一致**：CBraMod通过；REVE、LaBraM仅描述 |
| 三个预算点均通过本人−交换诊断 | **一致**：三个模型均通过 |

### 模型×数据集

分库为描述性统计，主判定仍使用完整235人合并中位数。

{table(['模型','数据集','G平均BA（%）','本人−G中位（pp）','本人−交换中位（pp）'],dataset_rows)}

![最终三模型曲线](../reports/population_budget/retry1/three_model_curves.png)

图中G为均值、差值为中位数，无置信区间；点线不表示充分收敛。原X1少样本一致性结论保持不变，三模型均稳定仅限PhysioNet。预训练重合、同session及骨干间群体结构差异的解释限制继续保留。

### 精度例外、核验与停止

{precision}

{resources}

本人−G仍用原3模型×2端点六槽Holm；补齐CBraMod后REVE 4×校正p变为6.57e-13，原始数据、中位数和原始p未变。本人−交换沿用本检查各预算点的原三槽家族；历史W/X1校正值不追溯修改。

[完整冻结报告](population_budget_report_2026-09-27.md)包含均值±SD、逐人/种子、验证CE/accuracy/BA、完整检验和资源。[执行记录](population_budget_retry_2026-09-27/execution_history.md) · [最终判读](../reports/population_budget/retry1/final_decisions.json) · [实验冻结记录](../reports/population_budget/retry1/experiment_freeze.json)。**此次为唯一重跑，至此本项目全部实验冻结，不再追加训练或X1扩展。**
'''
    cross = DOC / 'cross_model_summary.md'
    snapshot(cross)
    before = read(cross).split('## 用户新增的群体训练量检查',1)[0]
    save(cross, before + budget_section)

    tests = DOC / 'population_budget_tests.md'
    snapshot(tests)
    test_text = f'''## 群体训练量曲线及唯一数值重跑：最终冻结检验表（2026-09-27）

原科学规则v1.31，唯一重跑例外v1.34，结果登记v1.35；[原协议](population_budget_protocol.md)、[重跑前协议](population_budget_retry_protocol.md)。1×为现有G实际步数S，2×/4×为总2S/4S；个人r8与原网格只在验证被试选择，测试在选择冻结后读取。历史CBraMod优化器重启与群体参数化差异继续披露。

| ID | 预定规则 | 最终状态 |
|---|---|---|
| GB-G | 三模型三库三预算点G成绩、逐epoch与端点验证损失/准确率 | 全75轨迹完整；逐模型/库/点均25次运行，均到4×且未达联合平台 |
| GB-own-G | 同人五种子均值，完整235人中位；2×/4×六槽Holm | 三模型三点中位正，2×/4×均显著为正 |
| GB-own-swap | 原供体及10次抽样，全矩阵；中位≥2pp且原三槽Holm<.05 | 三模型三个预算点合并均通过 |
| GB-undertraining | M1≥M2≥M4且至少一处下降，M4<0.5pp；优先 | 三模型均未触发，不等于排除训练不足 |
| GB-stable | abs(M4−M2)<0.5pp，两点中位正且六槽Holm<.05 | CBraMod通过（0.400pp）；REVE/LaBraM不通过变化门槛 |
| GB-other | 其余如实描述；缺完整队列不缩小分母 | REVE/LaBraM描述性；CBraMod唯一重跑成功，已无缺失槽 |

{stats_table}

{conclusion}

REVE/LaBraM原始成绩和原始p逐项保持不变；原失败CBraMod的两个p=1缺失槽替换为真实p后，按相同六槽家族重算，REVE 4×校正p由首批9.60e-13更新为6.57e-13。原三槽本人−交换检验按各预算点分别报告，未运行变体p=1；不改历史W/X1家族。

{precision}

{resources}

19项准备CPU检查全部通过。最终合并75条来源、原失败保留、端点、逐epoch日志、个人验证选择及供体均已核验；另从最终逐人×种子表独立复算中位数、Wilcoxon及六槽Holm，验证完整分母与36组验证端点覆盖。工程核验不增加科学检验家族。

首批74成功/1失败及缺失NA是当时状态，保存在[首批报告](population_budget_initial_report_2026-09-27.md)和[重跑前检验快照](population_budget_tests_before_retry.md)。[最终完整报告](population_budget_report_2026-09-27.md)提供模型×库均值±SD、全部个体及验证日志。[独立复核回执](../reports/population_budget/retry1/final_verification.json)。**本阶段停止，之后全部实验冻结。**
'''
    save(tests, test_text)
    (DOC / 'all_tests.md').write_bytes(prefix + (DOC / 'cross_model_tests.md').read_bytes() + tests.read_bytes())
    assert hashlib.sha256((DOC/'all_tests.md').read_bytes()[:180205]).hexdigest() == prefix_hash

    spec = DOC / 'experiment_spec.md'
    content = read(spec)
    row = '| 2026-09-27 / v1.35（唯一重跑结果登记，全部实验冻结） | CBraMod fold1/seed37唯一数值稳定重跑成功，总13000步在4×停止；原失败保留，合并75条完整轨迹。235人五种子均值后的本人−G中位2.600→1.600→2.000pp，本人−交换3.533→3.400→3.233pp。按原判据，CBraMod持续训练后稳定正，REVE/LaBraM均只描述；三模型三点本人−交换均通过。 | [最终报告](population_budget_report_2026-09-27.md)。沿用v1.34精度例外与v1.31统计，不改判据；2×/4×中位变化0.400pp，六槽Holm均显著。75条均未达联合平台，不声称充分收敛。唯一重跑0.693卡时、13,092,075条合并预测复核通过。本项目全部实验冻结，之后不追加GPU、重跑或扩展。 |\n'
    if '/ v1.35' not in content:
        content = content.replace('|---|---|---|\n', '|---|---|---|\n' + row, 1)
        assert row in content
    # Historical result entry must keep pointing to its own evidence snapshot.
    oldline = next(x for x in content.splitlines() if '/ v1.33' in x)
    content = content.replace(oldline, oldline.replace('(population_budget_report_2026-09-27.md)', '(population_budget_initial_report_2026-09-27.md)'))
    save(spec, content)
    final_note = f'''\n\n## 唯一重跑完成后的最终证据（2026-09-27；全部实验冻结）

此段更新本文件上方首批/写作快照中的CBraMod缺失状态；历史稿件和其证据截止范围不自动改写。完整证据已按精度例外、原235人五种子分母和原校正家族核验。

{pooled_table}

{conclusion}

{precision}

群体成绩随预算单调上升的跨模型结论不一致；个人预算稳定类别不一致；三个预算点均通过本人−交换诊断一致。C3/C4/C5及少样本结论不扩张。本次只重跑授权的一条，未追加方法或更改判据。[最终完整报告](population_budget_report_2026-09-27.md)、[最终精确值](../reports/population_budget/retry1/retained/report/curve_summary.csv)。本项目实验全部冻结。
'''
    for name in ['findings_summary.md', 'claims_ledger.md']:
        path = DOC / name
        content = read(path)
        if '## 唯一重跑完成后的最终证据' not in content:
            content = content.replace('## 群体训练量曲线完成（2026-09-27）', '## 群体训练量曲线首批快照（2026-09-27；重跑前）')
            save(path, content + final_note)
    fail = DOC/'population_budget_numerical_failure_2026-09-27.md'
    if '## 唯一授权重跑的结果' not in read(fail):
        save(fail, read(fail) + '\n\n## 唯一授权重跑的结果\n\n用户随后授权仅本条数值稳定重跑一次，规则在重跑前登记。1912202_57于2026-09-27 19:12:04成功完成，CPU核验于19:17:06完成；原失败文件及上述现场保持不变。精度例外、完整队列统计和冻结状态见[最终报告](population_budget_report_2026-09-27.md)。没有第二次重跑，之后全部实验冻结。\n')
    for model in ['REVE', 'LaBraM']:
        path = DOC/f'population_budget_{model}_report_2026-09-27.md'
        content = read(path)
        note = '> 本文件保留首批分项报告。CBraMod唯一重跑后，六槽Holm校正及最终三模型结论见[冻结合并报告](population_budget_report_2026-09-27.md)；本模型原始数据、中位数与原始p未改变。\n\n'
        if note not in content:
            save(path, note + content)
    history = DOC/'population_budget_retry_2026-09-27/execution_history.md'
    if '## 2026-09-27 19:17后：' not in read(history):
        save(history, read(history) + f'\n\n## 2026-09-27 19:17后：唯一重跑及复核完成，实验冻结\n\n{resources}\n\n{conclusion}\n\n{precision}\n\n更新主规格v1.35、跨模型汇总、主报告、全部检验和主张账本；保留首批快照及所有原失败工件。科学协议/配置/来源清单哈希与历史all_tests前180205字节不变。所有实验冻结，最终交付后删除仅跟进本次重跑的自动任务。\n')
    receipt = {'verified_at_UTC':datetime.now(timezone.utc).isoformat(), 'complete_trajectories':75,
               'replacement_array_indices':[57], 'subject_seed_rows':len(people),
               'model_dataset_point_statistics_checked':checks, 'six_slot_Holm':{str(k):v for k,v in corrected.items()},
               'validation_groups_with_25_runs':36, 'REVE_LaBraM_original_summary_unchanged':True,
               'all_tests_historical_prefix_sha256':prefix_hash, 'all_checks_passed':True}
    save(OUT/'final_verification.json',json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'verified':True,'rows':len(people),'statistics':len(checks),'prefix':prefix_hash}))


if __name__ == '__main__':
    main()
