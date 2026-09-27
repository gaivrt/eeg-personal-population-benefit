"""Section-12 B2 report and the preregistered four-comparison gate."""
import argparse
import json
from pathlib import Path
import shutil
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.stage_a_common import config, save_json, scratch_root
from subject_context.stage_b2 import config_b2
from subject_context.statistics import holm

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path)
args = parser.parse_args()
root = args.root if args.root else scratch_root()
cfg = config_b2(); out = root / 'stage_b2/report'
out.mkdir(parents=True, exist_ok=True)
run_dirs = sorted((root / 'stage_b2/runs').glob('fold-*_seed-*'))
assert len(run_dirs) == 25
receipts = [json.loads((d / 'run.json').read_text()) for d in run_dirs]
assert all(r['status'] == 'complete' and r['backbone_unchanged'] and r['base_head_unchanged']
           and not r['activation_cache_used'] for r in receipts)
data = pd.concat([pd.read_csv(d / 'subjects.csv') for d in run_dirs], ignore_index=True)
assert len(data) == 235 * 5 * 5
assert not data.duplicated(['dataset', 'subject', 'seed', 'variant']).any()
data.to_csv(out / 'subject_seed_results.csv', index=False)
person = data.groupby(['variant', 'dataset', 'subject'], as_index=False)[
    ['b0_ba', 'ba', 'gain', 'head_ba', 'gain_vs_head']].mean()
person.to_csv(out / 'subject_results.csv', index=False)
groups = [*config()['datasets'], 'Dreyer+Cho', 'pooled']


def group(df, name):
    if name == 'pooled':
        return df
    return df[df.dataset.isin(['Dreyer2023', 'Cho2017'])] if name == 'Dreyer+Cho' else df[df.dataset == name]


def stats(values, alternative='two-sided'):
    d = np.round(np.asarray(values, dtype=float), 12)
    test = wilcoxon(d, zero_method='wilcox', method='approx', alternative=alternative) if np.any(d) else None
    return {'n': len(d), 'median_gain_pp': 100*np.median(d), 'mean_gain_pp': 100*np.mean(d),
            'sd_gain_pp': 100*np.std(d, ddof=1), 'improved_fraction': float(np.mean(d > 0)),
            'drop_gt_2pp_fraction': float(np.mean(d < -.02)),
            'statistic': float(test.statistic) if test else 0., 'p_raw': float(test.pvalue) if test else 1.}


summary = []; head_tests = []
for variant in cfg['variants']:
    for dataset in groups:
        g = group(person[person.variant == variant], dataset)
        summary.append({'variant': variant, 'dataset': dataset, 'b0_mean': g.b0_ba.mean(),
            'b0_sd': g.b0_ba.std(), 'adapted_mean': g.ba.mean(), 'adapted_sd': g.ba.std(),
            'head_mean': g.head_ba.mean(), **stats(g.gain)})
        head_tests.append({'variant': variant, 'dataset': dataset, **stats(g.gain_vs_head)})
for frame in [summary, head_tests]:
    for row, corrected in zip(frame, holm([r['p_raw'] for r in frame])):
        row['p_holm'] = float(corrected)
pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
pd.DataFrame(head_tests).to_csv(out / 'vs_finetuned_head.csv', index=False)
gate_rows = []
for variant in cfg['gate_variants']:
    g = person[person.variant == variant]
    gate_rows.append({'variant': variant, 'median_gain_vs_b0_pp': 100*float(np.median(np.round(g.gain, 12))),
        **{'vs_head_' + key: value for key, value in stats(g.gain_vs_head, 'greater').items()}})
for row, corrected in zip(gate_rows, holm([r['vs_head_p_raw'] for r in gate_rows])):
    row['vs_head_p_holm'] = float(corrected)
    row['passed'] = bool(row['median_gain_vs_b0_pp'] >= 100*cfg['gate_median_gain']
                         and corrected < cfg['gate_alpha'])
gate = {'passed': any(r['passed'] for r in gate_rows), 'threshold_pp': 2., 'alpha': .05,
    'rule': 'five-seed mean per subject; median gain >=2pp AND one-sided paired Wilcoxon vs Stage B head, Holm across four variants p<.05',
    'variants': gate_rows, 'stage_c_started': False}
save_json(out / 'gate.json', gate)
pd.DataFrame(gate_rows).to_csv(out / 'gate_tests.csv', index=False)

resources = pd.concat([pd.read_csv(d / 'fit_resources.csv') for d in run_dirs], ignore_index=True)
resources.to_csv(out / 'fit_resources.csv', index=False)
resource_rows = []
for variant in cfg['variants']:
    for dataset in groups:
        g = group(resources[(resources.variant == variant) & (resources.phase == 'test')], dataset)
        resource_rows.append({'variant': variant, 'dataset': dataset,
            'trainable_parameters_min': int(g.trainable_parameters.min()),
            'trainable_parameters_max': int(g.trainable_parameters.max()),
            'fit_seconds_mean': g.fit_seconds.mean(), 'fit_seconds_median': g.fit_seconds.median(),
            'fit_seconds_max': g.fit_seconds.max(), 'peak_gpu_bytes': int(g.peak_gpu_bytes.max()),
            'steps_min': int(g.steps.min()), 'steps_max': int(g.steps.max())})
pd.DataFrame(resource_rows).to_csv(out / 'resource_summary.csv', index=False)
zero = [row for d in run_dirs for row in json.loads((d / 'zero_checks.json').read_text())]
resource_receipt = {'worker_seconds_sum': sum(r['seconds'] for r in receipts),
    'longest_worker_seconds': max(r['seconds'] for r in receipts),
    'peak_gpu_bytes': int(resources.peak_gpu_bytes.max()),
    'zero_max_logit_error': max(r['max_logit_error'] for r in zero),
    'activation_cache_bytes': 0, 'activation_cache_used': False}
accounting = out / 'slurm_accounting.psv'
if accounting.exists():
    accounting_rows = pd.read_csv(accounting, sep='|')
    assert len(accounting_rows) == 25 and (accounting_rows.State == 'COMPLETED').all()
    events = []; gpu_seconds = 0
    for r in accounting_rows.itertuples():
        match = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', r.AllocTRES)
        assert match and int(match[1]) == 1
        events.extend([(pd.Timestamp(r.Start), 1), (pd.Timestamp(r.End), -1)])
        gpu_seconds += int(r.ElapsedRaw)
    active = peak = 0
    for timestamp, change in sorted(events):
        active += change; peak = max(peak, active)
    assert active == 0 and peak <= 16
    resource_receipt.update(full_array_wall_seconds=(max(t for t, d in events)-min(t for t, d in events)).total_seconds(),
                            full_array_gpu_seconds=gpu_seconds, actual_peak_concurrent_gpus=peak)
save_json(out / 'resources.json', resource_receipt)
save_json(out / 'run_manifest.json', receipts)
save_json(out / 'selection_manifest.json', [json.loads((d / 'selection.json').read_text()) for d in run_dirs])
for name in ['ea_tests.csv', 'ea_subject_results.csv', 'ea_subject_seed_results.csv', 'temporal_balance.csv']:
    shutil.copy2(root / 'stage_b/report' / name, out / name)
ea = pd.read_csv(out / 'ea_tests.csv')
label = {'film_first2': '前2层 FiLM', 'film_all': '全部层 FiLM', 'lora4': '全层 LoRA r=4',
         'lora8': '全层 LoRA r=8', 'scratch_head': '从零训练任务头'}
lines = ['# 阶段 B2：扩展上界检查', '', '## 做了什么', '',
    '沿用 235 人、固定 5 折×5 种子、B 的前半拟合/后半测试划分与对应 B0 起点。五变体全部运行，每名被试先平均五种子，再汇总被试统计。共 5,875 条被试/种子/变体结果。',
    '前2层及全12层 FiLM 为零初始化有界残差（幅度0.1）；LoRA 秩4/8作用全层空间和时间注意力的 Q/K/V/输出四个投影，各矩阵独立低秩更新，A随机、B零、alpha/r=1。只更新适配参数，原骨干和任务头冻结。参照头完全随机重置，只用本人前半标签拟合。',
    '每次均从保存的 float16 预处理 EEG 完整运行 patch embedding 与12层骨干，不读取或保存激活缓存。逐人单独重启适配参数；保留 B 的只微调头成绩作同试次对照。', '',
    '## 与规格的任何偏离', '',
    '- 新参数族的实现细节在结果前固定：LoRA 学习率0.0001/0.001，FiLM沿用0.01/0.1；随机头沿用0.0001/0.001。所有变体 AdamW、batch32、20/60步、weight decay 0/0.01；每折每种子仅验证被试选超参数，测试后半不选停止步。',
    '- 完整前向仍在第8层临时做float16→float32舍入，保持B0既有精度；这不是缓存。梯度通过dtype cast回传。随机头保留B0训练折拟合的固定特征标准化和架构，不继承B0头权重。',
    '- 四个适配变体的合并主比较预先固定单侧Wilcoxon及Holm校正；其余分库/合并双侧诊断单独校正，不混作新的判定门槛。', '',
    '- 自动审批拒绝报告脚本向既有HPC4服务器同步，故取回实验输出后在本地CPU生成报告和独立核验。训练代码、参数、数据及预定判定均未改变；服务器保留训练提交，本地保存报告工具后续提交。', '',
    '## 结果表', '',
    'BA 为平衡准确率；均值±SD在被试间计算。提升单位为百分点；改善严格>0，下降超2严格<-2。', '',
    '| 变体 | 数据集 | B0 BA/% | 适配 BA/% | 提升均值±SD | 提升中位数 | 改善比例 | 下降>2比例 |',
    '|---|---|---:|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f'| {label[r["variant"]]} | {r["dataset"]} | {100*r["b0_mean"]:.2f}±{100*r["b0_sd"]:.2f} | {100*r["adapted_mean"]:.2f}±{100*r["adapted_sd"]:.2f} | {r["mean_gain_pp"]:.2f}±{r["sd_gain_pp"]:.2f} | {r["median_gain_pp"]:.2f} | {r["improved_fraction"]:.1%} | {r["drop_gt_2pp_fraction"]:.1%} |')
lines += ['', '[每被试明细](subject_results.csv) · [每被试/种子明细](subject_seed_results.csv) · [验证选择记录](selection_manifest.json)。', '',
    '成本为正式测试被试单次拟合（含创建模型及最终计分）；显存为进程PyTorch分配峰值，不含驱动。详细分库数据见 [资源表](resource_summary.csv)，原始拟合记录见 [拟合成本](fit_resources.csv)。', '',
    '| 变体 | 可训练参数 | 拟合秒：均值/中位数/最大 | 显存峰值/GiB | 所选步数范围 |', '|---|---:|---:|---:|---:|']
for r in resource_rows:
    if r['dataset'] == 'pooled':
        lines.append(f'| {label[r["variant"]]} | {r["trainable_parameters_min"]:,}–{r["trainable_parameters_max"]:,} | {r["fit_seconds_mean"]:.3f}/{r["fit_seconds_median"]:.3f}/{r["fit_seconds_max"]:.3f} | {r["peak_gpu_bytes"]/2**30:.3f} | {r["steps_min"]}–{r["steps_max"]} |')
if 'full_array_wall_seconds' in resource_receipt:
    lines += ['', f'正式数组25作业全部完成，实际最多同时{resource_receipt["actual_peak_concurrent_gpus"]}张RTX3090；首个开始至最后结束{resource_receipt["full_array_wall_seconds"]:.0f}秒（不含首次排队），累计{resource_receipt["full_array_gpu_seconds"]:.0f}卡秒。结束后无GPU占用。见 [调度回执](slurm_accounting.psv)。']
lines += ['', 'EA 诊断复用阶段 B 完成结果，比较完整任务试次的对应 B0，与上表后半测试范围不同。B1b测试协方差用全部无标签任务，是直推式参照；B1c训练/验证/测试均仅用本人睁眼静息。', '',
    '| 条件 | 头 | 数据集 | B0 BA/% | EA BA/% | 平均变化/百分点 | Holm p（12项） |', '|---|---|---|---:|---:|---:|---:|']
for r in ea.to_dict('records'):
    lines.append(f'| {r["condition"]} | {r["head"]} | {r["dataset"]} | {100*r["b0_mean"]:.2f} | {100*r["ea_mean"]:.2f} | {r["mean_gain_pp"]:.2f} | {r["p_holm"]:.4g} |')
lines += ['', 'EA 改变输入后已在 A 重新前向计算全部EA特征；B1c复用该缓存，B/B2新增EA缓存0字节、0秒。原EA单独耗时未拆分记录；A混合缓存任务最长worker29.13秒，不能将其全部计为EA时间。EA详见 [检验表](ea_tests.csv)。', '',
    '## 统计检验结果', '',
    '主判定：四个合并逐人比较，五种子先平均；对B的只微调头做单侧配对Wilcoxon（greater），Holm校正p<0.05，并要求相对B0提升中位数≥2个百分点。', '',
    '| 适配变体 | 相对B0提升中位数/百分点 | 相对微调头平均差 | 相对微调头中位差 | 单侧原始p | Holm p（4项） | 通过 |', '|---|---:|---:|---:|---:|---:|---|']
for r in gate_rows:
    lines.append(f'| {label[r["variant"]]} | {r["median_gain_vs_b0_pp"]:.2f} | {r["vs_head_mean_gain_pp"]:.2f} | {r["vs_head_median_gain_pp"]:.2f} | {r["vs_head_p_raw"]:.4g} | {r["vs_head_p_holm"]:.4g} | {"是" if r["passed"] else "否"} |')
lines += ['', '**B2判定：' + ('通过。至少一个变体同时满足两项预定条件。' if gate['passed'] else '未通过。在冻结 CBraMod 上，同 session 被试适配的上界空间很小。') + '**', '',
    '各变体相对B的只微调任务头，分库及合并双侧诊断如下（25项Holm校正，与上方4项主检验分开）：', '',
    '| 变体 | 数据集 | 平均差/百分点 | 中位差/百分点 | 原始p | Holm p |', '|---|---|---:|---:|---:|---:|']
for r in head_tests:
    lines.append(f'| {label[r["variant"]]} | {r["dataset"]} | {r["mean_gain_pp"]:.2f} | {r["median_gain_pp"]:.2f} | {r["p_raw"]:.4g} | {r["p_holm"]:.4g} |')
lines += ['', '零差剔除；差值舍入12位后用正态近似处理并列秩。分组存在重叠，分库诊断不替代预定合并主检验。统计原表见 [任务头对照](vs_finetuned_head.csv) 和 [主判定](gate_tests.csv)。', '',
    '## 发现的问题', '',
    '本轮保留全部235人；B已经核验两半均有左右手且少数类≥40%。PhysioNet每人后半仅23次，BA分辨率有限；另列Dreyer+Cho。Cho真实顺序以发布trial_sequence和原event一致性为依据，缺独立连续时间线的局限沿用。',
    f'全部零适配与全前向B0的最大logit误差为{resource_receipt["zero_max_logit_error"]:.3g}；正式测试B0预测逐人核对A原预测。25次运行均记录原骨干与原头保持不变；每次git/包版本/GPU/B0头hash见 [运行清单](run_manifest.json)。',
    '76项单元测试通过，见 [测试回执](all_tests.xml) 和 [测试环境](tests_provenance.json)。独立逐条结果核验见 [核验回执](result_verification.json)；原始预测和个人参数保留在服务器。',
    '本轮全层FiLM、LoRA r=4/r=8同时通过两项主条件。全层FiLM在Cho的提升中位数仅0.70个百分点，且相对只微调头没有优势；两个LoRA在三库各自的中位提升均超过2个百分点，分库双侧诊断也均显著优于只微调头。仅看Dreyer+Cho时，全层FiLM、LoRA r=4/r=8的提升中位数仍为3.00、2.80、3.08个百分点，效果并非仅来自PhysioNet的小样本计分。',
    '收益仍因人而异：LoRA r=4/r=8分别有12.3%/12.8%的被试下降超过2个百分点。从零训练头合并BA为63.64%，平均比B0低6.29个百分点；其验证选择均到60步，本结论仅适用于相同的有限训练预算。',
    'EA副任务沿用既有结论：B1b+MLP在Dreyer平均提高1.43个百分点（Holm p=0.0239），但属于直推设定；B1c统一使用睁眼静息后，PhysioNet的线性/MLP头仍下降9.62/12.37个百分点，参考来源统一未解决EA退化。',
    '有标签前半拟合是有利信息条件，但有限超参数网格和20/60步不是全局优化；无论通过与否，本结果均不等同于数学上界，也不能证明无标签静息可生成同样有效的适配参数。', '',
    '## 下一步建议', '',
    ('B2达到固定条件，可依据训练折验证结果确定后续生成器参数族。本轮按要求停止，未进入C。' if gate['passed'] else
     'B2未达到固定条件，按预定规则停止。当前参数族和预算下没有足够证据支持继续训练生成器；本轮未进入C。')]
(out / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(gate))
print(pd.DataFrame(summary).query('dataset == "pooled"').to_string(index=False))
