"""Section-12 report from all fixed-fold baseline results; metadata only."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from subject_context.stage_a_common import ROOT, config, save_json, scratch_root, sha256
from subject_context.statistics import holm

root=scratch_root();out=root/"baseline_report";out.mkdir(exist_ok=True)
conditions=["B0_linear","B0_mlp","B1a_linear","B1a_mlp","B1b_linear","B1b_mlp","classic"]
frames=[]
for condition in conditions:
    paths=sorted((root/"baselines_v2"/condition).glob("*/subjects.csv"))
    if len(paths)!=25:raise RuntimeError(f"{condition}: {len(paths)}/25 results")
    df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    assert len(df)==235*5 and not df.duplicated(["dataset","subject","seed"]).any()
    frames.append(df)
data=pd.concat(frames,ignore_index=True)
data.to_csv(out/"subject_seed_results.csv",index=False)
person=data.groupby(["condition","dataset","subject"],as_index=False).balanced_accuracy.mean()
person.to_csv(out/"subject_results.csv",index=False)
summary=[]
for condition in conditions:
    for dataset in [*config()["datasets"],"pooled"]:
        p=person[person.condition==condition]
        r=data[data.condition==condition]
        if dataset!="pooled":p=p[p.dataset==dataset];r=r[r.dataset==dataset]
        seed_means=r.groupby("seed").balanced_accuracy.mean()
        summary.append({"condition":condition,"dataset":dataset,"n_subjects":len(p),
            "mean":p.balanced_accuracy.mean(),"subject_sd":p.balanced_accuracy.std(ddof=1),
            "seed_sd":seed_means.std(ddof=1)})
pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)
statistics=[]
for head in ["linear","mlp"]:
    for dataset in config()["datasets"]:
        a=person[(person.condition=="B1a_"+head)&(person.dataset==dataset)].set_index("subject").balanced_accuracy
        b=person[(person.condition=="B0_"+head)&(person.dataset==dataset)].set_index("subject").balanced_accuracy
        d=(a.sort_index()-b.sort_index()).to_numpy()
        d=np.round(d,12)
        test=wilcoxon(d,method="approx",zero_method="wilcox",alternative="two-sided") if np.any(d) else None
        statistics.append({"head":head,"dataset":dataset,"subjects":len(d),
            "statistic":float(test.statistic) if test else 0.,"p_raw":float(test.pvalue) if test else 1.,
            "median_difference_pp":100*np.median(d),"improved_fraction":float(np.mean(d>0)),
            "drop_gt_2pp_fraction":float(np.mean(d<-.02))})
for row,p in zip(statistics,holm([r["p_raw"] for r in statistics])):row["p_holm"]=float(p)
pd.DataFrame(statistics).to_csv(out/"wilcoxon.csv",index=False)
cached=[json.loads(p.read_text()) for p in (root/"features_v2").glob("*/sub-*/done.json")]
assert len(cached)==235
workers=[json.loads(p.read_text()) for p in (root/"receipts").glob("cache-worker-*.json")]
assert len(workers)==8 and all(w["status"]=="complete" for w in workers)
resources={"subjects":235,"prefix_bytes":sum(r["prefix_bytes"] for r in cached),
    "all_feature_bytes":sum(r["feature_bytes"] for r in cached),
    "cache_subject_seconds_sum":sum(r["seconds"] for r in cached),
    "cache_longest_worker_seconds":max(w["elapsed_seconds"] for w in workers),
    "cache_worker_seconds_sum":sum(w["elapsed_seconds"] for w in workers),
    "peak_gpu_allocated_bytes":max(r["peak_gpu_bytes"] for r in cached),
    "processed_bytes":sum(r["processed_bytes"] for r in json.loads((root/"receipts/prepared_v2.json").read_text())["subjects"])}
downloads=json.loads((root/"report/resource_metrics.json").read_text())
save_json(out/"download_metrics.json",[{k:v for k,v in r.items() if k.startswith("download_") or k in ["dataset","raw_disk_bytes"]} for r in downloads])
save_json(out/"channels.json",json.loads((ROOT/"configs/stage_a_channels.json").read_text()))
save_json(out/"resources.json",resources)
run_records=[json.loads(p.read_text()) for p in (root/"baselines_v2").glob("*/*/run.json")]
assert len(run_records)==175
save_json(out/"run_manifest.json",[{k:r.get(k) for k in ["git_commit","config_sha256","split_sha256","python","packages","gpu","host","slurm_job_id","condition","head","fold","seed","seconds","geometry_seconds","peak_gpu_bytes","selection","training_budget"]} for r in run_records])
prepared=json.loads((root/"receipts/prepared_v2.json").read_text())
audit=[]
for r in prepared["subjects"]:
    closed=r["rest"].get("closed",{})
    lo,hi=config()["datasets"][r["dataset"]]["rest_seconds_range"]
    audit.append({"dataset":r["dataset"],"subject":r["subject"],"left":r["left_trials"],"right":r["right_trials"],
        "imbalance_fraction":abs(r["left_trials"]-r["right_trials"])/(r["left_trials"]+r["right_trials"]),
        "eyes_open_seconds":r["rest"]["open"]["duration_seconds"],"eyes_open_stored_seconds":r["rest"]["open"]["usable_seconds"],
        "eyes_open_boundary":r["rest"]["open"].get("boundary_basis","original_record"),
        "eyes_open_duration_in_expected_range":lo<=r["rest"]["open"]["duration_seconds"]<=hi,
        "eyes_closed_seconds":closed.get("duration_seconds"),"eyes_closed_stored_seconds":closed.get("usable_seconds"),
        "eyes_closed_boundary":closed.get("boundary_basis","original_record") if closed else None,
        "eyes_closed_duration_in_expected_range":lo<=closed["duration_seconds"]<=hi if closed else None,
        "alpha_status":r["alpha"]["status"],"alpha_ratio":r["alpha"].get("closed_open_ratio"),
        "cho_order":r["checks"].get("cho_order_mapping_and_signal_roundtrip")})
pd.DataFrame(audit).to_csv(out/"data_cards.csv",index=False)
lines=["# 阶段 A 基线结果", "", "## 做了什么", "",
    "起步组 235 人、34,267 次左右手任务，固定 5 折 × 5 个种子；完成 B0 两种头、B1a/B1b 两种头及 EA+切空间+逻辑回归。所有骨干冻结，每个作业一张 RTX 3090。正式分折已提交且未按表现删人。", "",
    "输入全部来自已有 float16 预处理结果；任务和静息均已缓存前 8/12 层。骨干使用原生 EEG 通道，读出固定共同 27 通道并按时间平均。所有头的标准化、经典切空间参考仅拟合训练被试；学习率、epoch/C 只由验证被试选择。", "",
    "代码通过 Git bundle 同步到服务器；MOABB 数据、临时文件和缓存均在 scratch。下载通过联网计算节点上的 tmux 执行，逐文件预处理通过 Slurm CPU 作业执行。固定分折为仓库的 `configs/splits/stage_a_v1.json`。", "",
    "CBraMod 官方提交 `b9e961003214326972c567eff390e75b0287e32a`；权重 SHA256 `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`。官方 [PhysioNet 预处理](https://github.com/wjq-learning/CBraMod/blob/b9e961003214326972c567eff390e75b0287e32a/preprocessing/preprocessing_physio.py) 与 [输入缩放](https://github.com/wjq-learning/CBraMod/blob/b9e961003214326972c567eff390e75b0287e32a/datasets/physio_dataset.py) 对应如下。", "",
    "| 项目 | 官方 PhysioNet 流程 | 实际处理 |", "|---|---|---|",
    "| 采样率 | 200 Hz | PhysioNet 160 Hz、Dreyer/Cho 512 Hz，全部重采样至 200 Hz |",
    "| 参考与滤波 | 平均参考→0.3 Hz FIR 高通→60 Hz 陷波 | 同顺序；PhysioNet/Cho 60 Hz、Dreyer 50 Hz；不增加低通 |",
    "| 分段 | 提示后 4 秒 | PhysioNet/Dreyer 4 秒，Cho 按指令取 3 秒；静息为不重叠 4 秒，舍去不足一段的尾部 |",
    "| 幅值 | μV，再除以 100 | V × 1e4 保存 float16，读取恢复 float32；无逐试次 z-score |",
    "| 通道 | 指定 64 个 EEG 通道和顺序 | PhysioNet 同官方 64 通道顺序；Dreyer 去 EOG/EMG 后 27 通道；Cho 原生 64 EEG 通道；只对原标记坏导插值 |", "",
    "逐库通道与固定读出顺序见 [通道表](channels.json)。MNE 1.10.2；Dreyer/Cho 的陷波、窗口和静息处理为项目迁移，官方没有这两库的专用脚本。",
    "B0 的前 8 层激活量化为 float16 后再接最后 4 层，后续 FiLM 可直接复用；EA 特征由保存的 float16 信号白化后整网前向，中间激活为 float32。该数值精度差异记录为实现局限。", "",
    "## 与规格的偏离", "",
    "- 用户要求最简执行后，撤销实现者新增的幅值剔除、类别失衡门槛；alpha、时长及低准确率仅记录，不用来删人或中断。旧审计报告保留。",
    "- Dreyer S016/S032 重复标记取共同区间；S030 缺结束标记，使用发布 JSON 记录终点，属于推定边界；S040 使用实际 32 个提示，不补造试次。",
    "- 无外部 Git remote，使用 Git bundle 同步。CBraMod 官方覆盖的 PhysioNet 流程及 Dreyer/Cho 迁移参数沿用原报告；Cho 7 秒试次上 0.3 Hz FIR 的边缘风险仍在。",
    "- B1a 验证/测试的 EA 参考仅用自身睁眼静息；B1b 用全部无标签任务，属于直推参照。训练参考均用自身已保存的全部任务及静息。",
    "- EA 作用于已缩放为 μV/100 的信号，白化后为无量纲、参考平均方差约 1，直接输入骨干，不再次除以 100。固定 .001 对角收缩处理平均参考秩亏，不调参。",
    "- 经典参照从保存信号派生共同 27 通道的 8–30 Hz 特征；训练 EA 用全部自身数据，验证/测试 EA 用睁眼静息。Riemannian 切空间只拟合训练被试，逻辑回归在 GPU 上用 LBFGS。零初始化凸优化使五种子可能一致。", "",
    "- 初次下载器使用了 MOABB 的关闭 TLS 验证默认值，发现后已启用验证；早期已下载文件复用，未声称所有文件均在开启验证后重新下载。",
    "- 缓存曾用 8 张卡、B0 与经典参照同时最多 13 张，之前未先告知卡数；用户提出要求后，本轮 B1a/B1b 明确获批各 8 张、合计最多 16 张。之后新 GPU 作业须事先说明卡数。", "",
    "## 结果表", "",
    "平衡准确率为百分比。每名被试先平均五种子，再报告被试间均值 ± SD；合并结果按 235 名被试等权，不按试次数加权。", "",
    "| 条件 | PhysioNetMI | Dreyer2023 | Cho2017 | 合并 |", "|---|---:|---:|---:|---:|"]
for c in conditions:
    values=[next(r for r in summary if r["condition"]==c and r["dataset"]==d) for d in [*config()["datasets"],"pooled"]]
    lines.append("| "+c+" | "+" | ".join(f'{100*r["mean"]:.2f} ± {100*r["subject_sd"]:.2f}' for r in values)+" |")
lines += ["", "[逐被试结果](subject_results.csv) · [逐被试/种子明细](subject_seed_results.csv) · [含种子间 SD 的汇总](summary.csv)。每次运行的逐试次预测、头权重和验证搜索记录保存在服务器 `baselines_v2/`。", "",
    "B0、B1a、B1b 的每种头均搜索两种学习率 .001/.0003，每次 50 epochs、batch 128、AdamW、weight decay .01，MLP 隐层 128；取验证被试最优 epoch。未来本方法应使用相同的每候选 epoch、批量和候选数预算。经典逻辑回归 C=.1/1/10，最多 200 次 LBFGS 迭代。", "",
    f'预处理结果 {resources["processed_bytes"]/2**30:.3f} GiB；前 8 层激活缓存 {resources["prefix_bytes"]/2**30:.3f} GiB；连同基线特征共 {resources["all_feature_bytes"]/2**30:.3f} GiB。',
    f'缓存最长 worker {resources["cache_longest_worker_seconds"]:.2f} 秒，8 个 worker 用时合计 {resources["cache_worker_seconds_sum"]:.2f} 秒；单卡 PyTorch 分配显存峰值 {resources["peak_gpu_allocated_bytes"]/2**20:.2f} MiB（不含 CUDA 驱动开销）。',
    "精确资源与逐运行版本/GPU/环境见 [资源记录](resources.json) 和 [运行清单](run_manifest.json)。调度占用时长和并发见 [Slurm 记录](slurm_accounting.psv)。", "",
    "下载数据的实际文件字节数如下；包含重试的首轮开始至下载完成时间与累计作业时间分别列出，不把两者混同。网络重复传输字节未计量；Dreyer 原始占用含 ZIP 与解压文件。", "",
    "| 数据集 | 下载文件/GiB | 原始目录/GiB | 首次开始至完成/秒 | 累计下载作业/秒 |", "|---|---:|---:|---:|---:|"]
for r in downloads:lines.append(f'| {r["dataset"]} | {r["download_payload_bytes"]/2**30:.3f} | {r["raw_disk_bytes"]/2**30:.3f} | {r["download_wall_seconds_first_to_complete"]:.2f} | {r["download_slurm_elapsed_seconds_sum"]} |')
lines += ["", "精确字节数、时间戳与下载代码版本见 [下载记录](download_metrics.json)。本轮没有重新下载。", "",
    "## 统计检验结果", "",
    "B1a−B0，按数据集与同一头配对；每人先平均五种子，双侧 Wilcoxon，零差值剔除，使用正态近似处理并列秩。Holm 在 2 种头 × 3 个数据集的 6 项检验内校正。", "",
    "| 头 | 数据集 | 中位差/百分点 | 原始 p | Holm p | 改善比例 |", "|---|---|---:|---:|---:|---:|"]
for r in statistics:lines.append(f'| {r["head"]} | {r["dataset"]} | {r["median_difference_pp"]:.2f} | {r["p_raw"]:.4g} | {r["p_holm"]:.4g} | {r["improved_fraction"]:.1%} |')
lines += ["", "完整统计量及下降超过 2 个百分点的比例见 [Wilcoxon CSV](wilcoxon.csv)。", "", "## 发现的问题", ""]
low=[r for r in summary if r["condition"].startswith("B0_") and r["dataset"]!="pooled" and r["mean"]<.55]
if low:
    for r in low:lines.append(f'- {r["condition"]} 在 {r["dataset"]} 为 {r["mean"]:.2%}，低于原 55% 诊断线；按最新简化指令继续报告全部条件，未换标签、删被试或按测试成绩修改处理。')
else:lines.append("B0 两种头在三个数据集的平均 BA 均不低于 55%。")
alpha_fail=[f'{r["dataset"]} S{r["subject"]:03d}' for r in audit if r["alpha_status"]=="fail"]
lines += ["", "闭眼枕区 alpha 未高于睁眼的被试："+"、".join(alpha_fail)+"。PhysioNet 其余 101 人通过；Dreyer 缺枕区通道、Cho 缺闭眼段，alpha 不适用。",
    "所有 235 人均存在睁眼静息，原始/保存时长、名义时长范围检查和每手试次数见 [更新数据卡](data_cards.csv)。Cho 全部 52 人的已发布顺序、提示和完整 7 秒试次信号回查通过；缺独立连续事件时间线与 rest→MI 绝对间隔，一致性检查不能验证这些缺失信息。",
    "最终单元测试与泄漏检查见 [测试回执](final_tests.xml)；数据卡的时长异常和边界推定保留，未按测试准确率排除任何被试。", "",
    "## 下一步建议", "", "阶段 A 的缓存和基线已完成。本轮到此停止，不进入阶段 B。优先根据上述按数据集结果判断基线质量；保留负面结果，不再追加本轮实验。"]
(out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(pd.DataFrame(summary).to_string(index=False))
