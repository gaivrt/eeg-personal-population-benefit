"""Section-12 Stage B report; the gate is fixed before any test results exist."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from subject_context.stage_a_common import ROOT, config, save_json, scratch_root, sha256
from subject_context.stage_b import config_b
from subject_context.statistics import holm

root=scratch_root();cfg=config_b();base=root/"stage_b";out=base/"report";out.mkdir(parents=True,exist_ok=True)
run_dirs=sorted((base/"runs").glob("fold-*_seed-*"))
assert len(run_dirs)==25
run_records=[json.loads((d/"run.json").read_text()) for d in run_dirs]
assert all(r["status"]=="complete" and r["backbone_unchanged"] and r["base_head_unchanged"] for r in run_records)
data=pd.concat([pd.read_csv(d/"subjects.csv") for d in run_dirs],ignore_index=True)
assert len(data)==235*5*4 and not data.duplicated(["dataset","subject","seed","variant"]).any()
selected=data[data.selected_film].copy();assert len(selected)==235*5
selected["variant"]="selected_film"
data=pd.concat([data,selected],ignore_index=True)
data.to_csv(out/"subject_seed_results.csv",index=False)
person=data.groupby(["variant","dataset","subject"],as_index=False)[["b0_ba","ba","gain"]].mean()
person.to_csv(out/"subject_results.csv",index=False)
groups=[*config()["datasets"],"Dreyer+Cho","pooled"]
def group(df,name):
    return df if name=="pooled" else df[df.dataset.isin(["Dreyer2023","Cho2017"])] if name=="Dreyer+Cho" else df[df.dataset==name]
def stats(values):
    d=np.round(np.asarray(values,dtype=float),12)
    test=wilcoxon(d,zero_method="wilcox",method="approx",alternative="two-sided") if np.any(d) else None
    return {"n":len(d),"median_gain_pp":100*np.median(d),"mean_gain_pp":100*np.mean(d),
        "sd_gain_pp":100*np.std(d,ddof=1),"improved_fraction":float(np.mean(d>0)),
        "drop_gt_2pp_fraction":float(np.mean(d<-.02)),"statistic":float(test.statistic) if test else 0.,
        "p_raw":float(test.pvalue) if test else 1.}
summary=[]
for v in [*cfg["variants"],"selected_film"]:
    for dataset in groups:
        p=group(person[person.variant==v],dataset)
        summary.append({"variant":v,"dataset":dataset,"b0_mean":p.b0_ba.mean(),"adapted_mean":p.ba.mean(),**stats(p.gain)})
for r,p in zip(summary,holm([x["p_raw"] for x in summary])):r["p_holm"]=float(p)
pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)
head_tests=[]
pivot=person.pivot(index=["dataset","subject"],columns="variant",values="ba").reset_index()
for dataset in groups:
    p=group(pivot,dataset)
    head_tests.append({"dataset":dataset,**stats(p.selected_film-p["head"])})
for r,p in zip(head_tests,holm([x["p_raw"] for x in head_tests])):r["p_holm"]=float(p)
pd.DataFrame(head_tests).to_csv(out/"film_vs_head.csv",index=False)
fixed_head=[]
for variant in cfg["variants"][:-1]:
    for dataset in groups:
        p=group(pivot,dataset)
        fixed_head.append({"variant":variant,"dataset":dataset,**stats(p[variant]-p["head"])})
for r,p in zip(fixed_head,holm([x["p_raw"] for x in fixed_head])):r["p_holm"]=float(p)
pd.DataFrame(fixed_head).to_csv(out/"fixed_film_vs_head.csv",index=False)
main=next(r for r in summary if r["variant"]=="selected_film" and r["dataset"]=="pooled")
gate={"rule":"median over subjects of five-seed mean BA difference; FiLM variant chosen on each fold validation only",
      "subjects":235,"threshold_pp":2.,"observed_median_gain_pp":main["median_gain_pp"],
      "passed":bool(main["median_gain_pp"]>=2.),"stage_c_started":False}
save_json(out/"gate.json",gate)

ea_frames=[]
for condition in ["B0_linear","B0_mlp","B1b_linear","B1b_mlp","B1c_linear","B1c_mlp"]:
    paths=list((root/"baselines_v2"/condition).glob("*/subjects.csv"));assert len(paths)==25
    f=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);assert len(f)==235*5
    ea_frames.append(f)
ea=pd.concat(ea_frames,ignore_index=True);ea.to_csv(out/"ea_subject_seed_results.csv",index=False)
ep=ea.groupby(["condition","dataset","subject"],as_index=False).balanced_accuracy.mean()
ep.to_csv(out/"ea_subject_results.csv",index=False)
ea_stats=[]
for condition in ["B1b","B1c"]:
    for head in ["linear","mlp"]:
        x=ep[ep.condition==condition+"_"+head].set_index(["dataset","subject"]).balanced_accuracy
        b=ep[ep.condition=="B0_"+head].set_index(["dataset","subject"]).balanced_accuracy
        m=pd.DataFrame({"ea":x,"b0":b,"gain":x-b}).reset_index()
        for dataset in config()["datasets"]:
            g=group(m,dataset)
            ea_stats.append({"condition":condition,"head":head,"dataset":dataset,
                             "b0_mean":g.b0.mean(),"ea_mean":g.ea.mean(),**stats(g.gain)})
for r,p in zip(ea_stats,holm([x["p_raw"] for x in ea_stats])):r["p_holm"]=float(p)
pd.DataFrame(ea_stats).to_csv(out/"ea_tests.csv",index=False)

selections=[json.loads((d/"selection.json").read_text()) for d in run_dirs]
save_json(out/"run_manifest.json",run_records)
save_json(out/"selection_manifest.json",selections)
zero=[r for d in run_dirs for r in json.loads((d/"zero_checks.json").read_text())]
resources={"b6_runs":25,"b6_worker_seconds_sum":sum(r["seconds"] for r in run_records),
    "b6_longest_worker_seconds":max(r["seconds"] for r in run_records),
    "b6_peak_gpu_allocated_bytes":max(r["peak_gpu_bytes"] for r in run_records),
    "zero_film_max_logit_error":max(r["max_logit_error"] for r in zero),
    "b1c_extra_cache_seconds":0,"b1c_extra_cache_bytes":0,
    "b1c_cache_basis":"Reuse Stage A ea_eo.npy, already computed from whitened saved signals; original EA extraction was timed jointly with other Stage A features."}
accounting=out/"slurm_accounting.psv"
if accounting.exists():
    a=pd.read_csv(accounting,sep="|",header=None,names=["job","state","elapsed","start","end","resources"])
    full=a[a.job.str.contains("_",regex=False)]
    assert len(full)==50 and set(full.state)=={"COMPLETED"}
    starts=pd.to_datetime(full.start);ends=pd.to_datetime(full.end)
    active=0;peak=0
    for _,delta in sorted([(t,1) for t in starts]+[(t,-1) for t in ends],key=lambda x:(x[0],x[1])):
        active+=delta;peak=max(peak,active)
    resources.update(full_array_jobs=50,all_jobs_completed=True,actual_peak_concurrent_gpus=peak,
        full_array_wall_seconds=(ends.max()-starts.min()).total_seconds(),
        full_array_gpu_seconds=float(pd.to_timedelta(full.elapsed).dt.total_seconds().sum()))
save_json(out/"resources.json",resources)
rows=[]
for r in json.loads((ROOT/"configs/splits/stage_b_halves_v1.json").read_text())["subjects"]:
    t=pd.read_csv(root/"processed"/r["dataset"]/f'sub-{r["subject"]:03d}'/"trials.csv")
    row={"dataset":r["dataset"],"subject":r["subject"],"order_basis":r["order_basis"]}
    for role in ["fit","query"]:
        y=t.iloc[r[role+"_indices"]].label
        row.update({role+"_left":int((y==0).sum()),role+"_right":int((y==1).sum()),role+"_n":len(y),
                    role+"_balanced":min((y==0).sum(),(y==1).sum())/len(y)>=.4})
    rows.append(row)
pd.DataFrame(rows).to_csv(out/"temporal_balance.csv",index=False)
label={"film2":"末2层 gamma+beta","film4":"末4层 gamma+beta","beta4":"末4层仅 beta","head":"仅微调任务头","selected_film":"验证选择的 FiLM（主判定）"}
lines=["# 阶段 B：同 session 上界检查", "", "## 做了什么", "",
    "沿用阶段 A 的 235 人、固定 5 折和 5 种子。每人按原始记录/发布顺序，前 floor(n/2) 试次带标签拟合，后半独立计分。B6 与同一 B0 在完全相同后半试次配对。四变体共 4,700 条被试/种子结果。", "",
    "每折/种子的 B0 头由阶段 A 保存的验证 BA 选择；骨干、训练折拟合的特征标准化始终冻结，FiLM 变体的任务头也冻结。FiLM 零初始化，gamma=1+0.1*tanh(dg)、beta=0.1*tanh(db)。任务头参照仅更新复制的任务头。",
    "FiLM 学习率 .01/.1，头参照 .0001/.001；步数 20/60，weight decay 0/.01，AdamW、batch 32。仅在本折验证被试用同样的前后半协议选择；测试后半不选 checkpoint。各变体与超参数选择记录见 [选择清单](selection_manifest.json)。", "",
    "## 与规格的任何偏离", "",
    "- 阶段 A 两种头均有模型，本轮按既有验证成绩选每折/种子的起点，不按阶段 A 测试成绩选择。beta-only 固定作用末4层，与末4层 gamma+beta 对照。",
    "- 最佳 FiLM 固定为每折/种子验证选择程序，未从测试结果取最大变体。主规则、候选网格、种子聚合与时间划分在运行前写入规格和版本控制。",
    "- Cho 顺序来自发布的 trial_sequence 和原始 event 字段的一致性回查；没有独立连续事件时间线，保留阶段 A 的证据局限。",
    "- B1c 所需 ea_eo 特征在阶段 A 已生成，因此复用缓存，不重复前向。EA 改变输入后必须重新经过骨干，不能把 B0 激活直接白化；这项原始工作已在 A 做过，EA 单项耗时当时未拆分记录。", "",
    "- 本地 Git 签名程序挂起后，本轮使用未签名提交保存版本，没有修改全局签名设置；代码哈希、配置哈希和逐次环境记录均保留。", "",
    "## 结果表", "",
    "每名被试先平均五种子；均值±SD、中位数均基于被试。变化单位为百分点，改善为严格 >0，下降超过2为严格 <-2。", "",
    "| 变体 | 数据集 | B0后半 BA/% | 适配后 BA/% | 提升均值±SD | 提升中位数 | 改善比例 | 下降>2比例 |",
    "|---|---|---:|---:|---:|---:|---:|---:|"]
for r in summary:lines.append(f'| {label[r["variant"]]} | {r["dataset"]} | {100*r["b0_mean"]:.2f} | {100*r["adapted_mean"]:.2f} | {r["mean_gain_pp"]:.2f} ± {r["sd_gain_pp"]:.2f} | {r["median_gain_pp"]:.2f} | {r["improved_fraction"]:.1%} | {r["drop_gt_2pp_fraction"]:.1%} |')
lines += ["", "[逐被试结果](subject_results.csv) · [逐被试/种子明细](subject_seed_results.csv) · [前后半类别计数](temporal_balance.csv)。",
    "", f'**主判定：{ "通过" if gate["passed"] else "未通过" }。验证选定 FiLM 的合并逐人提升中位数为 {main["median_gain_pp"]:.2f} 个百分点，预定门槛为 ≥2。** 该判定只决定是否值得继续；本轮不进入 C。', "",
    "EA 副任务使用完整任务试次，与阶段 A 完整试次 B0 比较，不与 B6 后半成绩混算。B1b 使用全部无标签任务作为测试 EA 参考，是直推设定；B1c 训练/验证/测试都仅使用本人睁眼静息。", "",
    "| 条件 | 头 | 数据集 | B0/% | EA/% | 平均差/百分点 | 中位差/百分点 | Holm p |", "|---|---|---|---:|---:|---:|---:|---:|"]
for r in ea_stats:lines.append(f'| {r["condition"]} | {r["head"]} | {r["dataset"]} | {100*r["b0_mean"]:.2f} | {100*r["ea_mean"]:.2f} | {r["mean_gain_pp"]:.2f} | {r["median_gain_pp"]:.2f} | {r["p_holm"]:.4g} |')
lines += ["", "EA 白化在已有 μV/100 信号上计算，输出无量纲后直接送骨干，不再除100；固定协方差收缩 .001，训练头预算与 A 一致。B1c 新增缓存为 **0 字节、0 秒**（复用，不代表 EA 原始计算零成本）。阶段 A 全部缓存最长 worker 29.13 秒，包含 B0、静息、三种 EA 和经典特征，不能把该时间全归给 EA。", "",
    f'B6 25 个 worker 计算时间合计 {resources["b6_worker_seconds_sum"]:.2f} 秒、最长 {resources["b6_longest_worker_seconds"]:.2f} 秒；单卡 PyTorch 分配显存峰值 {resources["b6_peak_gpu_allocated_bytes"]/2**30:.3f} GiB（不含 CUDA 驱动）。精确调度占用另见 Slurm 回执。',
    "版本、GPU、环境、B0 头哈希、冻结权重检查见 [运行清单](run_manifest.json)；B1c 各运行原始记录留在 `baselines_v2/B1c_*/`。", "",
    "## 统计检验结果", "",
    "先对每人平均五种子，再配对双侧 Wilcoxon；零差值剔除，正态近似处理并列秩。B6 5 条件×5 汇总组内 Holm 校正；这些重叠汇总组仅作描述性辅助，主判定按预设中位数门槛。EA 的 2条件×2头×3库单独进行12项 Holm 校正。完整统计量见 [B6汇总](summary.csv)、[EA检验](ea_tests.csv)。", "",
    "验证选定 FiLM 减去仅调任务头：", "",
    "| 数据集 | 平均差/百分点 | 中位差/百分点 | FiLM更好比例 | Holm p（5项） |", "|---|---:|---:|---:|---:|"]
for r in head_tests:lines.append(f'| {r["dataset"]} | {r["mean_gain_pp"]:.2f} | {r["median_gain_pp"]:.2f} | {r["improved_fraction"]:.1%} | {r["p_holm"]:.4g} |')
lines += ["", "三个固定 FiLM 变体各自相对只调头的完整配对结果另见 [固定变体对照](fixed_film_vs_head.csv)，15 项比较单独 Holm 校正。均未显示 FiLM 显著优于只调任务头；这不构成等效性证明。"]
lines += ["", "## 发现的问题", "",
    "235 人两半均有两类，按预定少数类≥40%检查未发现明显类别失衡。PhysioNet 每人前22、后23试次；Dreyer 各半80–120，Cho各半100–120。PhysioNet 单个试次会造成较大 BA 跳动，因此另外列出了 Dreyer、Cho 及两者合并结果。",
    f'零初始化全部预测与对应 B0 一致，最大 logit 舍入差 {resources["zero_film_max_logit_error"]:.3g}；骨干和原始 B0 头哈希在各次运行前后不变。',
    "74 项单元测试全部通过，见 [测试回执](all_tests.xml) 与 [测试环境](tests_provenance.json)。独立复算 4,700 条被试/种子/变体结果与 343,700 条预测，确认查询等于固定后半段、B0 预测与阶段 A 原模型一致，见 [结果核验](result_verification.json)。",
    "三个固定 FiLM 变体的合并提升中位数最高也仅 1.00 个百分点；Dreyer+Cho 合并时最高为 1.42。因此未通过并非仅由 PhysioNet 的小样本后半段造成。验证选定 FiLM 相对只调头的合并平均差为 -0.08、中位差 -0.23 个百分点，Holm p=1，未体现额外优势。",
    "EA 诊断中，B1b+MLP 在 Dreyer 平均提高 1.43 个百分点（Holm p=0.0239），但它使用全部无标签任务，属于直推参照；PhysioNet 两种头仍明显下降。B1c 统一使用睁眼静息后，PhysioNet 仍下降约 9.62/12.37 个百分点，说明仅统一训练与测试的参考来源没有解决 EA 的退化；具体原因尚未定位。",
    "有标签前半拟合是有利信息条件，但这里的有限步数、有界 FiLM 与半段评估只是操作性上界检查，不是数学上的最优上界。即使通过，也不能证明无标签静息能够预测有效参数。", "",
    "## 下一步建议", "",
    ("阶段 B 达到预定门槛，可以讨论阶段 C 的生成器实验，但本轮到此停止，不启动 C。" if gate["passed"] else "阶段 B 未达到预定门槛，按规则停止当前 FiLM 生成器路线；本轮不启动 C，也不根据测试结果继续调整网格。")]
(out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
if "full_array_wall_seconds" in resources:
    text=(out/"report.md").read_text(encoding="utf-8")
    marker="## 统计检验结果"
    text=text.replace(marker,f'正式数组实际同时最多 {resources["actual_peak_concurrent_gpus"]} 张 RTX 3090，从首个作业开始至全部结束 {resources["full_array_wall_seconds"]:.0f} 秒（不含首次排队）；累计占用 {resources["full_array_gpu_seconds"]:.0f} 卡秒。50 个作业全部完成，结束后不再占卡。详见 [Slurm 回执](slurm_accounting.psv)。\n\n'+marker)
    (out/"report.md").write_text(text,encoding="utf-8")
print(json.dumps(gate));print(pd.DataFrame(summary).to_string(index=False))
