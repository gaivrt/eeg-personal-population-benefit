# 全部预先定义检验与完整比较附录

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

| ID | 阶段 | 数据集 | 假设 | 判定规则 | 结果 | 是否通过 | 出处 |
|---|---|---|---|---|---|---|---|
| B-G | B | 起步组合并 | 验证选定末层 FiLM 相对 B0 有足够增益 | 中位增益≥2pp | 中位 1.000pp | 未通过 | [原记录](../reports/stage_b/report/gate.json) |
| B2-film_first2 | B2 | 起步组合并 | film_first2：相对 B0 有增益且优于微调头 | 中位≥2pp 且 vs头单侧 Holm(4)<.05 | 中位=1.833; pH=0.030713 | 未通过 | [原记录](../reports/stage_b2/report/gate.json) |
| B2-film_all | B2 | 起步组合并 | film_all：相对 B0 有增益且优于微调头 | 中位≥2pp 且 vs头单侧 Holm(4)<.05 | 中位=2.833; pH=1.155e-08 | 通过 | [原记录](../reports/stage_b2/report/gate.json) |
| B2-lora4 | B2 | 起步组合并 | lora4：相对 B0 有增益且优于微调头 | 中位≥2pp 且 vs头单侧 Holm(4)<.05 | 中位=2.800; pH=1.2765e-15 | 通过 | [原记录](../reports/stage_b2/report/gate.json) |
| B2-lora8 | B2 | 起步组合并 | lora8：相对 B0 有增益且优于微调头 | 中位≥2pp 且 vs头单侧 Holm(4)<.05 | 中位=3.409; pH=1.3182e-17 | 通过 | [原记录](../reports/stage_b2/report/gate.json) |
| C-M_film | C | 起步组合并 | M_film：上下文个性化满足原成功标准 | vs B4与B0 单侧 Holm(4)<.05；均值恢复≥30%；下降>2pp≤10% | 两对照显著=False;恢复=0.981;下降=22.979% | 未通过 | [原记录](../reports/stage_c/report/criteria.json) |
| C-M_lora | C | 起步组合并 | M_lora：上下文个性化满足原成功标准 | vs B4与B0 单侧 Holm(4)<.05；均值恢复≥30%；下降>2pp≤10% | 两对照显著=False;恢复=1.561;下降=12.766% | 未通过 | [原记录](../reports/stage_c/report/criteria.json) |
| C2-D1-film_offset | C2诊断1 | 起步组合并 | film_offset：本人优于交换 | U中位≥2pp 且单侧 Holm(3)<.05 | U中位=4.617;pH=4.1681e-31 | 通过 | [原记录](../reports/stage_c2/report/gate.json) |
| C2-D1-mix_offset | C2诊断1 | 起步组合并 | mix_offset：本人优于交换 | U中位≥2pp 且单侧 Holm(3)<.05 | U中位=2.580;pH=7.1335e-23 | 通过 | [原记录](../reports/stage_c2/report/gate.json) |
| C2-D1-lora8 | C2诊断1 | 起步组合并 | lora8：本人优于交换 | U中位≥2pp 且单侧 Holm(3)<.05 | U中位=3.470;pH=5.0123e-28 | 通过 | [原记录](../reports/stage_c2/report/gate.json) |
| C2-D2-film_offset | C2诊断2 | 起步组合并 | film_offset：静息最近邻优于随机 | D1通过后，单侧 Holm(3)<.05 | 差中位=0.561;pH=0.10124 | 未通过 | [原记录](../reports/stage_c2/report/gate.json) |
| C2-D2-mix_offset | C2诊断2 | 起步组合并 | mix_offset：静息最近邻优于随机 | D1通过后，单侧 Holm(3)<.05 | 差中位=0.367;pH=0.10124 | 未通过 | [原记录](../reports/stage_c2/report/gate.json) |
| C2-D2-lora8 | C2诊断2 | 起步组合并 | lora8：静息最近邻优于随机 | D1通过后，单侧 Holm(3)<.05 | 差中位=0.017;pH=0.73119 | 未通过 | [原记录](../reports/stage_c2/report/gate.json) |
| C3-D3-film_offset | C3诊断3 | 起步组合并 | film_offset：任务k=1近邻优于随机 | 单侧 Holm(2)<.05 | 差中位=0.917;pH=0.00051699 | 通过 | [原记录](../reports/stage_c3/report/gate.json) |
| C3-D3-lora8 | C3诊断3 | 起步组合并 | lora8：任务k=1近邻优于随机 | 单侧 Holm(2)<.05 | 差中位=0.189;pH=0.17997 | 未通过 | [原记录](../reports/stage_c3/report/gate.json) |
| C4-main | C4 | 起步组合并 | LoRA n=10 无标签先验优于随机先验及 B3 | P2−P1 与 P2−P0 单侧原始p均<.05 | P2-P1:p=0.89273; P2-P0:p=0.44768 | 未通过 | [原记录](../reports/stage_c4/report/endpoints.json) |
| C4-secondary-n0 | C4次要 | 起步组合并 | LoRA n=0 P2>P1 | 两标签量单侧 Holm(2)<.05 | 中位=0.180;pH=0.078435 | 未通过 | [原记录](../reports/stage_c4/report/endpoints.json) |
| C4-secondary-n5 | C4次要 | 起步组合并 | LoRA n=5 P2>P1 | 两标签量单侧 Holm(2)<.05 | 中位=0.100;pH=0.29768 | 未通过 | [原记录](../reports/stage_c4/report/endpoints.json) |
| M-main-M1 | M主终点 | 起步组合并 | M1 n=10 优于验证选出的R1 | 单侧 Holm(2)<.05；中位≥1pp；下降>2pp≤10% | 中位=0.517;pH=6.7044e-05;下降=12.340% | 未通过 | [原记录](../reports/stage_m/report/endpoints.json) |
| M-main-M2 | M主终点 | 起步组合并 | M2 n=10 优于验证选出的R1 | 单侧 Holm(2)<.05；中位≥1pp；下降>2pp≤10% | 中位=0.576;pH=6.7044e-05;下降=13.191% | 未通过 | [原记录](../reports/stage_m/report/endpoints.json) |
| M-secondary-M1-n5 | M次要 | 起步组合并 | M1 n=5 vs R1 | 同样比较；各n Holm(2)，列原效应门槛 | 中位=0.483;pH=0.0036885;下降=15.319% | 未通过 | [原记录](../reports/stage_m/report/endpoints.json) |
| M-secondary-M2-n5 | M次要 | 起步组合并 | M2 n=5 vs R1 | 同样比较；各n Holm(2)，列原效应门槛 | 中位=0.500;pH=0.00033439;下降=12.766% | 未通过 | [原记录](../reports/stage_m/report/endpoints.json) |
| M-secondary-M1-n20 | M次要 | 起步组合并 | M1 n=20 vs R1 | 同样比较；各n Holm(2)，列原效应门槛 | 中位=0.467;pH=0.0010864;下降=14.043% | 未通过 | [原记录](../reports/stage_m/report/endpoints.json) |
| M-secondary-M2-n20 | M次要 | 起步组合并 | M2 n=20 vs R1 | 同样比较；各n Holm(2)，列原效应门槛 | 中位=0.333;pH=0.0010864;下降=9.362% | 未通过 | [原记录](../reports/stage_m/report/endpoints.json) |
| M-adapt-M1 | M次要适配增益 | 起步组合并 | M1 相对 R1 的 (n10−n0) 增益 | 指定单侧 Holm(2)<.05；不改变主判定 | 均值=0.316;pH=0.041048 | 通过 | [原记录](../reports/stage_m/report/endpoints.json) |
| M-adapt-M2 | M次要适配增益 | 起步组合并 | M2 相对 R1 的 (n10−n0) 增益 | 指定单侧 Holm(2)<.05；不改变主判定 | 均值=0.235;pH=0.06045 | 未通过 | [原记录](../reports/stage_m/report/endpoints.json) |
| W-diagnostic1-film_offset- | W | 起步组合并 | film_offset 本人>交换 | U中位≥2pp且单侧Holm(3)<.05 | 中位=4.689;pH=9.1247e-32 | 通过 | [原记录](../reports/stage_w/report/diagnostic_tests.json) |
| W-diagnostic1-lora8- | W | 起步组合并 | lora8 本人>交换 | U中位≥2pp且单侧Holm(3)<.05 | 中位=3.533;pH=3.1524e-28 | 通过 | [原记录](../reports/stage_w/report/diagnostic_tests.json) |
| W-diagnostic2_3-film_offset-rest | W | 起步组合并 | film_offset rest k=1近邻>随机 | 原近邻单侧Holm：静息3项/任务2项 | 中位=0.440;pH=0.10461 | 未通过 | [原记录](../reports/stage_w/report/diagnostic_tests.json) |
| W-diagnostic2_3-lora8-rest | W | 起步组合并 | lora8 rest k=1近邻>随机 | 原近邻单侧Holm：静息3项/任务2项 | 中位=0.000;pH=1 | 未通过 | [原记录](../reports/stage_w/report/diagnostic_tests.json) |
| W-diagnostic2_3-film_offset-task | W | 起步组合并 | film_offset task k=1近邻>随机 | 原近邻单侧Holm：静息3项/任务2项 | 中位=0.767;pH=0.0042664 | 通过 | [原记录](../reports/stage_w/report/diagnostic_tests.json) |
| W-diagnostic2_3-lora8-task | W | 起步组合并 | lora8 task k=1近邻>随机 | 原近邻单侧Holm：静息3项/任务2项 | 中位=0.567;pH=0.053663 | 未通过 | [原记录](../reports/stage_w/report/diagnostic_tests.json) |
| W-Lee_diagnostic1-film_offset- | W | Lee2019_MI | film_offset 本人>交换 | U中位≥2pp且单侧Holm(3)<.05 | 中位=2.500;pH=0.00015151 | 通过 | [原记录](../reports/stage_w/report/diagnostic_tests.json) |
| W-Lee_diagnostic1-lora8- | W | Lee2019_MI | lora8 本人>交换 | U中位≥2pp且单侧Holm(3)<.05 | 中位=2.580;pH=2.2285e-06 | 通过 | [原记录](../reports/stage_w/report/diagnostic_tests.json) |
| W-Lee_diagnostic2-film_offset- | W | Lee2019_MI | film_offset rest k=1近邻>随机 | 原近邻单侧Holm：静息3项/任务2项 | 中位=0.500;pH=0.62662 | 未通过 | [原记录](../reports/stage_w/report/diagnostic_tests.json) |
| W-Lee_diagnostic2-lora8- | W | Lee2019_MI | lora8 rest k=1近邻>随机 | 原近邻单侧Holm：静息3项/任务2项 | 中位=1.190;pH=0.016588 | 通过 | [原记录](../reports/stage_w/report/diagnostic_tests.json) |
| WF-R2_sensitivity_diagnostic1-film_offset | W补充 | 起步组合并 | film_offset 本人>交换 | U中位≥2pp 且单侧Holm(3)<.05 | U中位=4.008;pH=4.4201e-29;交换N=235 | 通过 | [原记录](../reports/stage_w_followup/report/diagnostic_tests.json) |
| WF-R2_sensitivity_diagnostic1-lora8 | W补充 | 起步组合并 | lora8 本人>交换 | U中位≥2pp 且单侧Holm(3)<.05 | U中位=3.447;pH=4.3558e-26;交换N=235 | 通过 | [原记录](../reports/stage_w_followup/report/diagnostic_tests.json) |
| WF-BNCI_diagnostic1_available_subset-film_offset | W补充 | BNCI可交换8人 | film_offset 本人>交换 | U中位≥2pp 且单侧Holm(3)<.05 | U中位=6.944;pH=0.04995;交换N=8 | 通过 | [原记录](../reports/stage_w_followup/report/diagnostic_tests.json) |
| WF-BNCI_diagnostic1_available_subset-lora8 | W补充 | BNCI可交换8人 | lora8 本人>交换 | U中位≥2pp 且单侧Holm(3)<.05 | U中位=12.083;pH=0.037593;交换N=8 | 通过 | [原记录](../reports/stage_w_followup/report/diagnostic_tests.json) |

C4 只有主终点通过才支持其预定“标签节省”结论；原终点文件保存的插值量不满足该前提，不能宣称已证明节省标签。M 的按库等效标签插值为预定描述，不能替代失败的主终点；非单调或超出观察范围时不外推。R2 匹配外循环更新次数，并非 FLOPs 或墙钟。W 的 U>0 仅检验个人特异性，个人收益另以本人−G衡量。

## 3. 全部统计比较与少样本描述

以下每张表逐行列出完整源表，包括反向及未显著结果。主门槛行在上一节解释；以下描述性显著性不被升级为方法通过。没有原定 p 值的少样本曲线不事后补检验。源文件中的全精度值保留，正文只缩短显示精度。

### A · EA静息参照

B1a−B0。两头×三库共6项双侧Holm。 [完整源表](../reports/stage_a/baselines/wilcoxon.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| A-wilcoxon-001 | head=linear | PhysionetMI | 103 | -10.476 | — | 5.9494e-13 | 2.9747e-12 | 检出双侧差异；中位反向；无方法门槛 |
| A-wilcoxon-002 | head=linear | Dreyer2023 | 80 | -1.2083 | — | 0.11903 | 0.47613 | 未检出差异；无方法门槛 |
| A-wilcoxon-003 | head=linear | Cho2017 | 52 | 0.60833 | — | 0.58788 | 0.64152 | 未检出差异；无方法门槛 |
| A-wilcoxon-004 | head=mlp | PhysionetMI | 103 | -12.976 | — | 6.651e-16 | 3.9906e-15 | 检出双侧差异；中位反向；无方法门槛 |
| A-wilcoxon-005 | head=mlp | Dreyer2023 | 80 | -1.125 | — | 0.32076 | 0.64152 | 未检出差异；无方法门槛 |
| A-wilcoxon-006 | head=mlp | Cho2017 | 52 | -1.45 | — | 0.20229 | 0.60688 | 未检出差异；无方法门槛 |

### B · 全部适配−B0

各个人适配−B0。原表内统一校正；主门槛仍仅为选定FiLM中位≥2pp。 [完整源表](../reports/stage_b/report/summary.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| B-summary-001 | variant=film2 | PhysionetMI | 103 | -0.45455 | -0.027814 | 0.8849 | 1 | 未检出差异；无方法门槛 |
| B-summary-002 | variant=film2 | Dreyer2023 | 80 | 0.75 | 0.49084 | 0.019724 | 0.13807 | 未检出差异；无方法门槛 |
| B-summary-003 | variant=film2 | Cho2017 | 52 | 1 | 2.425 | 0.0020681 | 0.018613 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-004 | variant=film2 | Dreyer+Cho | 132 | 0.83333 | 1.2528 | 0.00018839 | 0.0035795 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-005 | variant=film2 | pooled | 235 | 0.6 | 0.6915 | 0.022827 | 0.13807 | 未检出差异；无方法门槛 |
| B-summary-006 | variant=film4 | PhysionetMI | 103 | 0.30303 | -0.17642 | 0.80488 | 1 | 未检出差异；无方法门槛 |
| B-summary-007 | variant=film4 | Dreyer2023 | 80 | 1.4167 | 1.6442 | 1.8344e-05 | 0.00038523 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-008 | variant=film4 | Cho2017 | 52 | 1.2 | 3.0378 | 0.0027454 | 0.021963 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-009 | variant=film4 | Dreyer+Cho | 132 | 1.3333 | 2.1932 | 1.8075e-07 | 4.5188e-06 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-010 | variant=film4 | pooled | 235 | 0.84615 | 1.1546 | 0.00055759 | 0.0081757 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-011 | variant=beta4 | PhysionetMI | 103 | 0.15152 | 0.09754 | 0.79714 | 1 | 未检出差异；无方法门槛 |
| B-summary-012 | variant=beta4 | Dreyer2023 | 80 | 1.25 | 1.3147 | 8.2348e-05 | 0.001647 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-013 | variant=beta4 | Cho2017 | 52 | 1.9 | 3.0109 | 0.00054504 | 0.0081757 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-014 | variant=beta4 | Dreyer+Cho | 132 | 1.4167 | 1.9829 | 2.2036e-07 | 5.2888e-06 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-015 | variant=beta4 | pooled | 235 | 1 | 1.1565 | 0.00020834 | 0.0037501 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-016 | variant=head | PhysionetMI | 103 | 0.22727 | 0.13313 | 0.83995 | 1 | 未检出差异；无方法门槛 |
| B-summary-017 | variant=head | Dreyer2023 | 80 | 1.1667 | 1.0988 | 0.00093122 | 0.010243 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-018 | variant=head | Cho2017 | 52 | 1.2 | 3.5942 | 0.00039495 | 0.0063192 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-019 | variant=head | Dreyer+Cho | 132 | 1.1833 | 2.0818 | 1.22e-06 | 2.8061e-05 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-020 | variant=head | pooled | 235 | 0.90909 | 1.2277 | 0.00055682 | 0.0081757 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-021 | variant=selected_film | PhysionetMI | 103 | 0.5303 | 0.22753 | 0.55684 | 1 | 未检出差异；无方法门槛 |
| B-summary-022 | variant=selected_film | Dreyer2023 | 80 | 1.1667 | 1.1014 | 0.00056831 | 0.0081757 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-023 | variant=selected_film | Cho2017 | 52 | 1.4 | 3.0333 | 0.001179 | 0.01179 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-024 | variant=selected_film | Dreyer+Cho | 132 | 1.1667 | 1.8625 | 2.5608e-06 | 5.6337e-05 | 检出双侧差异；均值正向；无方法门槛 |
| B-summary-025 | variant=selected_film | pooled | 235 | 1 | 1.1459 | 0.00025203 | 0.0042846 | 检出双侧差异；均值正向；无方法门槛 |

### B · 验证选定FiLM−微调头

selected_film−head。原表内统一校正；主门槛仍仅为选定FiLM中位≥2pp。 [完整源表](../reports/stage_b/report/film_vs_head.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| B-film_vs_head-001 |  | PhysionetMI | 103 | -0.22727 | 0.094406 | 0.8448 | 1 | 未检出差异；无方法门槛 |
| B-film_vs_head-002 |  | Dreyer2023 | 80 | -0.25 | 0.0026221 | 0.83107 | 1 | 未检出差异；无方法门槛 |
| B-film_vs_head-003 |  | Cho2017 | 52 | -0.4 | -0.5609 | 0.11935 | 0.59674 | 未检出差异；无方法门槛 |
| B-film_vs_head-004 |  | Dreyer+Cho | 132 | -0.25 | -0.21937 | 0.25334 | 1 | 未检出差异；无方法门槛 |
| B-film_vs_head-005 |  | pooled | 235 | -0.22727 | -0.081843 | 0.59086 | 1 | 未检出差异；无方法门槛 |

### B · 固定FiLM−微调头

固定FiLM−head。原表内统一校正；主门槛仍仅为选定FiLM中位≥2pp。 [完整源表](../reports/stage_b/report/fixed_film_vs_head.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| B-fixed_film_vs_head-001 | variant=film2 | PhysionetMI | 103 | -0.5303 | -0.16094 | 0.40654 | 1 | 未检出差异；无方法门槛 |
| B-fixed_film_vs_head-002 | variant=film2 | Dreyer2023 | 80 | -0.5 | -0.60794 | 0.061643 | 0.73972 | 未检出差异；无方法门槛 |
| B-fixed_film_vs_head-003 | variant=film2 | Cho2017 | 52 | -0.8 | -1.1692 | 0.031171 | 0.40523 | 未检出差异；无方法门槛 |
| B-fixed_film_vs_head-004 | variant=film2 | Dreyer+Cho | 132 | -0.5 | -0.82905 | 0.0044502 | 0.066753 | 未检出差异；无方法门槛 |
| B-fixed_film_vs_head-005 | variant=film2 | pooled | 235 | -0.5 | -0.53622 | 0.0083516 | 0.11692 | 未检出差异；无方法门槛 |
| B-fixed_film_vs_head-006 | variant=film4 | PhysionetMI | 103 | -0.83333 | -0.30955 | 0.4812 | 1 | 未检出差异；无方法门槛 |
| B-fixed_film_vs_head-007 | variant=film4 | Dreyer2023 | 80 | 0.66667 | 0.5454 | 0.096517 | 0.96924 | 未检出差异；无方法门槛 |
| B-fixed_film_vs_head-008 | variant=film4 | Cho2017 | 52 | 0 | -0.55641 | 0.11103 | 0.99931 | 未检出差异；无方法门槛 |
| B-fixed_film_vs_head-009 | variant=film4 | Dreyer+Cho | 132 | 0.18333 | 0.11135 | 0.72859 | 1 | 未检出差异；无方法门槛 |
| B-fixed_film_vs_head-010 | variant=film4 | pooled | 235 | 0 | -0.073126 | 0.74636 | 1 | 未检出差异；无方法门槛 |
| B-fixed_film_vs_head-011 | variant=beta4 | PhysionetMI | 103 | -0.15152 | -0.035587 | 0.92847 | 1 | 未检出差异；无方法门槛 |
| B-fixed_film_vs_head-012 | variant=beta4 | Dreyer2023 | 80 | 0.083333 | 0.21588 | 0.67865 | 1 | 未检出差异；无方法门槛 |
| B-fixed_film_vs_head-013 | variant=beta4 | Cho2017 | 52 | -0.4 | -0.58333 | 0.088112 | 0.96924 | 未检出差异；无方法门槛 |
| B-fixed_film_vs_head-014 | variant=beta4 | Dreyer+Cho | 132 | 0 | -0.098964 | 0.50073 | 1 | 未检出差异；无方法门槛 |
| B-fixed_film_vs_head-015 | variant=beta4 | pooled | 235 | -0.076923 | -0.071186 | 0.62202 | 1 | 未检出差异；无方法门槛 |

### B · 两种EA诊断

同一头的EA−B0。原表内统一校正；主门槛仍仅为选定FiLM中位≥2pp。 [完整源表](../reports/stage_b/report/ea_tests.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| B-ea_tests-001 | condition=B1b; head=linear | PhysionetMI | 103 | -7.8854 | -8.518 | 6.7529e-12 | 6.0776e-11 | 检出双侧差异；均值反向；无方法门槛 |
| B-ea_tests-002 | condition=B1b; head=linear | Dreyer2023 | 80 | 0.58333 | 0.49496 | 0.41347 | 1 | 未检出差异；无方法门槛 |
| B-ea_tests-003 | condition=B1b; head=linear | Cho2017 | 52 | 2 | -1.5199 | 0.85864 | 1 | 未检出差异；无方法门槛 |
| B-ea_tests-004 | condition=B1b; head=mlp | PhysionetMI | 103 | -10.179 | -9.8925 | 6.4928e-14 | 7.142e-13 | 检出双侧差异；均值反向；无方法门槛 |
| B-ea_tests-005 | condition=B1b; head=mlp | Dreyer2023 | 80 | 0.95833 | 1.433 | 0.0029872 | 0.023898 | 检出双侧差异；均值正向；无方法门槛 |
| B-ea_tests-006 | condition=B1b; head=mlp | Cho2017 | 52 | 0.65 | -2.926 | 0.6293 | 1 | 未检出差异；无方法门槛 |
| B-ea_tests-007 | condition=B1c; head=linear | PhysionetMI | 103 | -10.316 | -9.6207 | 1.1388e-13 | 1.1388e-12 | 检出双侧差异；均值反向；无方法门槛 |
| B-ea_tests-008 | condition=B1c; head=linear | Dreyer2023 | 80 | -0.083333 | -0.36665 | 0.61115 | 1 | 未检出差异；无方法门槛 |
| B-ea_tests-009 | condition=B1c; head=linear | Cho2017 | 52 | 1.8 | -0.40737 | 0.33899 | 1 | 未检出差异；无方法门槛 |
| B-ea_tests-010 | condition=B1c; head=mlp | PhysionetMI | 103 | -11.976 | -12.372 | 1.4445e-16 | 1.7334e-15 | 检出双侧差异；均值反向；无方法门槛 |
| B-ea_tests-011 | condition=B1c; head=mlp | Dreyer2023 | 80 | -0.083333 | -0.15379 | 0.79373 | 1 | 未检出差异；无方法门槛 |
| B-ea_tests-012 | condition=B1c; head=mlp | Cho2017 | 52 | -0.6 | -1.9997 | 0.20235 | 1 | 未检出差异；无方法门槛 |

### B2 · 全部适配−B0

个人适配−B0。每表25项双侧Holm；B2四项主门槛另见第2节。B2的EA表原样复用B，不计作新检验。 [完整源表](../reports/stage_b2/report/summary.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| B2-summary-001 | variant=film_first2 | PhysionetMI | 103 | 1.8182 | 1.2585 | 0.019364 | 0.058093 | 未检出差异；无方法门槛 |
| B2-summary-002 | variant=film_first2 | Dreyer2023 | 80 | 2 | 2.1375 | 1.9834e-07 | 1.9834e-06 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-003 | variant=film_first2 | Cho2017 | 52 | 1.8 | 2.9724 | 0.0034394 | 0.013758 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-004 | variant=film_first2 | Dreyer+Cho | 132 | 2 | 2.4664 | 4.5557e-09 | 5.4668e-08 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-005 | variant=film_first2 | pooled | 235 | 1.8333 | 1.937 | 8.9318e-09 | 9.825e-08 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-006 | variant=film_all | PhysionetMI | 103 | 2.5758 | 2.7137 | 0.0017605 | 0.0088025 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-007 | variant=film_all | Dreyer2023 | 80 | 3.75 | 4.2168 | 2.2896e-11 | 3.4344e-10 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-008 | variant=film_all | Cho2017 | 52 | 0.7 | 3.191 | 0.027918 | 0.058093 | 未检出差异；无方法门槛 |
| B2-summary-009 | variant=film_all | Dreyer+Cho | 132 | 3 | 3.8127 | 4.0182e-12 | 7.2327e-11 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-010 | variant=film_all | pooled | 235 | 2.8333 | 3.331 | 1.1749e-12 | 2.3498e-11 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-011 | variant=lora4 | PhysionetMI | 103 | 2.8788 | 3.4126 | 8.8729e-06 | 7.0983e-05 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-012 | variant=lora4 | Dreyer2023 | 80 | 3.1667 | 3.7295 | 3.4562e-12 | 6.5668e-11 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-013 | variant=lora4 | Cho2017 | 52 | 2.2 | 4.6853 | 2.0581e-05 | 0.00014407 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-014 | variant=lora4 | Dreyer+Cho | 132 | 2.8 | 4.106 | 6.8525e-16 | 1.5076e-14 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-015 | variant=lora4 | pooled | 235 | 2.8 | 3.8021 | 1.3741e-18 | 3.2978e-17 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-016 | variant=lora8 | PhysionetMI | 103 | 4.0152 | 4.0004 | 4.8445e-06 | 4.3601e-05 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-017 | variant=lora8 | Dreyer2023 | 80 | 3.3333 | 3.9107 | 7.6364e-12 | 1.2982e-10 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-018 | variant=lora8 | Cho2017 | 52 | 2.4 | 5.0994 | 7.8174e-05 | 0.00046904 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-019 | variant=lora8 | Dreyer+Cho | 132 | 3.0833 | 4.3789 | 3.2333e-15 | 6.7899e-14 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-020 | variant=lora8 | pooled | 235 | 3.4091 | 4.213 | 4.6519e-18 | 1.0699e-16 | 检出双侧差异；均值正向；无方法门槛 |
| B2-summary-021 | variant=scratch_head | PhysionetMI | 103 | -9.1667 | -8.8066 | 8.3319e-11 | 1.1665e-09 | 检出双侧差异；均值反向；无方法门槛 |
| B2-summary-022 | variant=scratch_head | Dreyer2023 | 80 | -7.75 | -7.1151 | 1.2128e-11 | 1.9405e-10 | 检出双侧差异；均值反向；无方法门槛 |
| B2-summary-023 | variant=scratch_head | Cho2017 | 52 | 0 | -0.048718 | 0.29262 | 0.29262 | 未检出差异；无方法门槛 |
| B2-summary-024 | variant=scratch_head | Dreyer+Cho | 132 | -4.8 | -4.3314 | 9.7015e-11 | 1.2612e-09 | 检出双侧差异；均值反向；无方法门槛 |
| B2-summary-025 | variant=scratch_head | pooled | 235 | -6.2121 | -6.2929 | 8.1258e-20 | 2.0314e-18 | 检出双侧差异；均值反向；无方法门槛 |

### B2 · 全部适配−微调头

个人适配−B阶段微调头。每表25项双侧Holm；B2四项主门槛另见第2节。B2的EA表原样复用B，不计作新检验。 [完整源表](../reports/stage_b2/report/vs_finetuned_head.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| B2-vs_finetuned_head-001 | variant=film_first2 | PhysionetMI | 103 | 0.90909 | 1.1254 | 0.095754 | 0.28726 | 未检出差异；无方法门槛 |
| B2-vs_finetuned_head-002 | variant=film_first2 | Dreyer2023 | 80 | 0.83333 | 1.0387 | 0.0050559 | 0.035392 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-003 | variant=film_first2 | Cho2017 | 52 | -0.7 | -0.62179 | 0.014453 | 0.072263 | 未检出差异；无方法门槛 |
| B2-vs_finetuned_head-004 | variant=film_first2 | Dreyer+Cho | 132 | 0.083333 | 0.38458 | 0.32947 | 0.4342 | 未检出差异；无方法门槛 |
| B2-vs_finetuned_head-005 | variant=film_first2 | pooled | 235 | 0.30303 | 0.70926 | 0.061425 | 0.2457 | 未检出差异；无方法门槛 |
| B2-vs_finetuned_head-006 | variant=film_all | PhysionetMI | 103 | 1.8182 | 2.5806 | 0.00070225 | 0.0063203 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-007 | variant=film_all | Dreyer2023 | 80 | 3.1667 | 3.1181 | 1.1825e-09 | 1.7738e-08 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-008 | variant=film_all | Cho2017 | 52 | -0.2 | -0.40321 | 0.2171 | 0.4342 | 未检出差异；无方法门槛 |
| B2-vs_finetuned_head-009 | variant=film_all | Dreyer+Cho | 132 | 1.1833 | 1.7309 | 8.063e-07 | 9.6756e-06 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-010 | variant=film_all | pooled | 235 | 1.2879 | 2.1033 | 1.155e-08 | 1.617e-07 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-011 | variant=lora4 | PhysionetMI | 103 | 2.7273 | 3.2795 | 2.4086e-06 | 2.4086e-05 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-012 | variant=lora4 | Dreyer2023 | 80 | 2.8333 | 2.6307 | 9.8432e-11 | 1.5779e-09 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-013 | variant=lora4 | Cho2017 | 52 | 0.81667 | 1.091 | 0.002935 | 0.02348 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-014 | variant=lora4 | Dreyer+Cho | 132 | 2.2 | 2.0241 | 8.6633e-13 | 1.7327e-11 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-015 | variant=lora4 | pooled | 235 | 2.4 | 2.5744 | 8.5099e-16 | 1.8722e-14 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-016 | variant=lora8 | PhysionetMI | 103 | 2.5758 | 3.8673 | 6.6156e-08 | 8.6003e-07 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-017 | variant=lora8 | Dreyer2023 | 80 | 2.8333 | 2.8119 | 9.282e-11 | 1.5779e-09 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-018 | variant=lora8 | Cho2017 | 52 | 1 | 1.5051 | 0.0059339 | 0.035603 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-019 | variant=lora8 | Dreyer+Cho | 132 | 2 | 2.2971 | 3.8658e-12 | 6.9585e-11 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-020 | variant=lora8 | pooled | 235 | 2.0455 | 2.9853 | 6.5909e-18 | 1.5159e-16 | 检出双侧差异；均值正向；无方法门槛 |
| B2-vs_finetuned_head-021 | variant=scratch_head | PhysionetMI | 103 | -9.4697 | -8.9397 | 2.3228e-12 | 4.4134e-11 | 检出双侧差异；均值反向；无方法门槛 |
| B2-vs_finetuned_head-022 | variant=scratch_head | Dreyer2023 | 80 | -8.4167 | -8.2139 | 2.4548e-14 | 5.1552e-13 | 检出双侧差异；均值反向；无方法门槛 |
| B2-vs_finetuned_head-023 | variant=scratch_head | Cho2017 | 52 | -3 | -3.6429 | 1.161e-06 | 1.2771e-05 | 检出双侧差异；均值反向；无方法门槛 |
| B2-vs_finetuned_head-024 | variant=scratch_head | Dreyer+Cho | 132 | -5.9167 | -6.4132 | 3.8617e-20 | 9.2681e-19 | 检出双侧差异；均值反向；无方法门槛 |
| B2-vs_finetuned_head-025 | variant=scratch_head | pooled | 235 | -7 | -7.5206 | 9.553e-30 | 2.3883e-28 | 检出双侧差异；均值反向；无方法门槛 |

### C · 原成功标准的组成检验

真实上下文−对照。四项主比较；两个方法的整体判定均为未通过，见第2节。 [完整源表](../reports/stage_c/report/criteria_tests.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| C-criteria_tests-001 | method=M_film; control=B4 | 起步组合并 | 235 | 0 | -0.0058811 | 0.5666 | 0.5666 | 未检出差异；无方法门槛 |
| C-criteria_tests-002 | method=M_film; control=B0 | 起步组合并 | 235 | 3.3333 | 3.2665 | 9.0756e-10 | 2.7227e-09 | 检出正向差异；无方法门槛 |
| C-criteria_tests-003 | method=M_lora; control=B4 | 起步组合并 | 235 | 0 | 0.0074705 | 0.093705 | 0.18741 | 未检出差异；无方法门槛 |
| C-criteria_tests-004 | method=M_lora; control=B0 | 起步组合并 | 235 | 6.2879 | 6.5766 | 7.3936e-25 | 2.9575e-24 | 检出正向差异；无方法门槛 |

### C · 全部诊断对照

真实上下文−所列对照。60项双侧Holm，含默认z、打乱、去间隔、T3A及EA。 [完整源表](../reports/stage_c/report/diagnostic_tests.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| C-diagnostic_tests-001 | method=M_film; versus=B4_film | PhysionetMI | 103 | 0 | -0.098208 | 0.30409 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-002 | method=M_film; versus=B4_film | Dreyer2023 | 80 | 0 | 0.022917 | 0.72012 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-003 | method=M_film; versus=B4_film | Cho2017 | 52 | 0 | 0.13269 | 0.48251 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-004 | method=M_film; versus=B4_film | Dreyer+Cho | 132 | 0 | 0.066162 | 0.48764 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-005 | method=M_film; versus=B4_film | pooled | 235 | 0 | -0.0058811 | 0.8668 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-006 | method=M_film; versus=M_film_lambda0 | PhysionetMI | 103 | 0.37879 | 0.75216 | 0.034655 | 0.92764 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-007 | method=M_film; versus=M_film_lambda0 | Dreyer2023 | 80 | 0.5 | 0.48574 | 0.051398 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-008 | method=M_film; versus=M_film_lambda0 | Cho2017 | 52 | -1.4 | -0.99872 | 0.013953 | 0.43254 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-009 | method=M_film; versus=M_film_lambda0 | Dreyer+Cho | 132 | -0.18333 | -0.099046 | 0.72346 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-010 | method=M_film; versus=M_film_lambda0 | pooled | 235 | 0.075758 | 0.27403 | 0.23234 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-011 | method=M_film; versus=B3_film | PhysionetMI | 103 | 0 | -0.22339 | 0.071427 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-012 | method=M_film; versus=B3_film | Dreyer2023 | 80 | -0.083333 | -0.30298 | 0.0027813 | 0.097345 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-013 | method=M_film; versus=B3_film | Cho2017 | 52 | 0.2 | -0.03141 | 0.5436 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-014 | method=M_film; versus=B3_film | Dreyer+Cho | 132 | 0 | -0.196 | 0.19551 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-015 | method=M_film; versus=B3_film | pooled | 235 | 0 | -0.208 | 0.033843 | 0.92764 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-016 | method=M_film; versus=B2_T3A | PhysionetMI | 103 | 11.894 | 12.635 | 8.1659e-17 | 4.0013e-15 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-017 | method=M_film; versus=B2_T3A | Dreyer2023 | 80 | 7.5 | 8.6869 | 1.9292e-14 | 8.8744e-13 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-018 | method=M_film; versus=B2_T3A | Cho2017 | 52 | -0.2 | 1.9179 | 0.69032 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-019 | method=M_film; versus=B2_T3A | Dreyer+Cho | 132 | 5.1 | 6.0203 | 3.2756e-12 | 1.4085e-10 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-020 | method=M_film; versus=B2_T3A | pooled | 235 | 7.6923 | 8.9196 | 5.3324e-28 | 3.0394e-26 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-021 | method=M_film; versus=B1a | PhysionetMI | 103 | 18.106 | 17.755 | 9.5405e-17 | 4.5795e-15 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-022 | method=M_film; versus=B1a | Dreyer2023 | 80 | 4.9167 | 4.8197 | 1.7652e-08 | 7.0609e-07 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-023 | method=M_film; versus=B1a | Cho2017 | 52 | -4.3 | -3.1045 | 0.00065853 | 0.023707 | 检出双侧差异；均值反向；无方法门槛 |
| C-diagnostic_tests-024 | method=M_film; versus=B1a | Dreyer+Cho | 132 | 1.6667 | 1.698 | 0.03313 | 0.92764 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-025 | method=M_film; versus=B1a | pooled | 235 | 6.8182 | 8.7358 | 3.9624e-17 | 1.9812e-15 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-026 | method=M_film; versus=B1c | PhysionetMI | 103 | 18.636 | 18.759 | 1.1908e-17 | 6.192e-16 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-027 | method=M_film; versus=B1c | Dreyer2023 | 80 | 4.5 | 3.9997 | 9.0293e-07 | 3.5214e-05 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-028 | method=M_film; versus=B1c | Cho2017 | 52 | -4.2 | -3.4885 | 0.00046317 | 0.017137 | 检出双侧差异；均值反向；无方法门槛 |
| C-diagnostic_tests-029 | method=M_film; versus=B1c | Dreyer+Cho | 132 | 0.5 | 1.0498 | 0.18461 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-030 | method=M_film; versus=B1c | pooled | 235 | 7.3333 | 8.8117 | 2.4175e-17 | 1.2329e-15 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-031 | method=M_lora; versus=B4_lora | PhysionetMI | 103 | 0 | 0.026162 | 0.38059 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-032 | method=M_lora; versus=B4_lora | Dreyer2023 | 80 | 0 | -7.1839e-05 | 0.73419 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-033 | method=M_lora; versus=B4_lora | Cho2017 | 52 | 0 | -0.017949 | 0.43499 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-034 | method=M_lora; versus=B4_lora | Dreyer+Cho | 132 | 0 | -0.0071142 | 0.42123 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-035 | method=M_lora; versus=B4_lora | pooled | 235 | 0 | 0.0074705 | 0.18741 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-036 | method=M_lora; versus=M_lora_lambda0 | PhysionetMI | 103 | 0.83333 | 0.97256 | 0.0055267 | 0.18791 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-037 | method=M_lora; versus=M_lora_lambda0 | Dreyer2023 | 80 | 0.5 | 0.62267 | 0.062419 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-038 | method=M_lora; versus=M_lora_lambda0 | Cho2017 | 52 | -0.4 | -0.93141 | 0.19893 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-039 | method=M_lora; versus=M_lora_lambda0 | Dreyer+Cho | 132 | 0.18333 | 0.010454 | 0.62735 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-040 | method=M_lora; versus=M_lora_lambda0 | pooled | 235 | 0.33333 | 0.43214 | 0.024633 | 0.73899 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-041 | method=M_lora; versus=B3_lora | PhysionetMI | 103 | 0 | 0.15692 | 0.16766 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-042 | method=M_lora; versus=B3_lora | Dreyer2023 | 80 | 0 | 0.015409 | 0.77472 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-043 | method=M_lora; versus=B3_lora | Cho2017 | 52 | -0.5 | -1.3712 | 0.00016958 | 0.006444 | 检出双侧差异；均值反向；无方法门槛 |
| C-diagnostic_tests-044 | method=M_lora; versus=B3_lora | Dreyer+Cho | 132 | -0.16667 | -0.53081 | 0.0083718 | 0.2679 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-045 | method=M_lora; versus=B3_lora | pooled | 235 | 0 | -0.22938 | 0.28193 | 1 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-046 | method=M_lora; versus=B2_T3A | PhysionetMI | 103 | 13.385 | 14.953 | 2.2372e-18 | 1.2081e-16 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-047 | method=M_lora; versus=B2_T3A | Dreyer2023 | 80 | 12.583 | 13.528 | 7.8398e-15 | 3.6847e-13 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-048 | method=M_lora; versus=B2_T3A | Cho2017 | 52 | 1.4 | 4.8391 | 0.0065741 | 0.21694 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-049 | method=M_lora; versus=B2_T3A | Dreyer+Cho | 132 | 8.6667 | 10.105 | 5.5924e-19 | 3.1317e-17 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-050 | method=M_lora; versus=B2_T3A | pooled | 235 | 11.5 | 12.23 | 3.2071e-36 | 1.9243e-34 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-051 | method=M_lora; versus=B1a | PhysionetMI | 103 | 20.758 | 20.072 | 9.7499e-18 | 5.1675e-16 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-052 | method=M_lora; versus=B1a | Dreyer2023 | 80 | 9.5 | 9.6607 | 5.7452e-14 | 2.5853e-12 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-053 | method=M_lora; versus=B1a | Cho2017 | 52 | -2 | -0.18333 | 0.025651 | 0.74387 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-054 | method=M_lora; versus=B1a | Dreyer+Cho | 132 | 5.8167 | 5.7827 | 5.2287e-10 | 2.196e-08 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-055 | method=M_lora; versus=B1a | pooled | 235 | 11.308 | 12.046 | 3.4308e-28 | 1.9899e-26 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-056 | method=M_lora; versus=B1c | PhysionetMI | 103 | 19.697 | 21.076 | 2.1731e-18 | 1.1952e-16 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-057 | method=M_lora; versus=B1c | Dreyer2023 | 80 | 9.1667 | 8.8407 | 1.6795e-13 | 7.39e-12 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-058 | method=M_lora; versus=B1c | Cho2017 | 52 | -2.8 | -0.56731 | 0.038679 | 0.96699 | 未检出差异；无方法门槛 |
| C-diagnostic_tests-059 | method=M_lora; versus=B1c | Dreyer+Cho | 132 | 4.9167 | 5.1345 | 7.6478e-09 | 3.1356e-07 | 检出双侧差异；均值正向；无方法门槛 |
| C-diagnostic_tests-060 | method=M_lora; versus=B1c | pooled | 235 | 11 | 12.122 | 1.0699e-28 | 6.3124e-27 | 检出双侧差异；均值正向；无方法门槛 |

### C · 各条件相对B0（后半）

各条件−B0。98行原报告诊断族，包含复用的基线/上界，不能当作98次独立实验。 [完整源表](../reports/stage_c/report/summary.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| C-summary-001 | condition=B0 | PhysionetMI | 103 | 0 | 0 | 1 | 1 | 未检出差异；无方法门槛 |
| C-summary-002 | condition=B0 | Dreyer2023 | 80 | 0 | 0 | 1 | 1 | 未检出差异；无方法门槛 |
| C-summary-003 | condition=B0 | Cho2017 | 52 | 0 | 0 | 1 | 1 | 未检出差异；无方法门槛 |
| C-summary-004 | condition=B0 | Dreyer+Cho | 132 | 0 | 0 | 1 | 1 | 未检出差异；无方法门槛 |
| C-summary-005 | condition=B0 | pooled | 235 | 0 | 0 | 1 | 1 | 未检出差异；无方法门槛 |
| C-summary-006 | condition=B1a | PhysionetMI | 103 | -11.288 | -10.839 | 6.0682e-11 | 3.7623e-09 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-007 | condition=B1a | Dreyer2023 | 80 | -0.75 | -0.85844 | 0.15637 | 1 | 未检出差异；无方法门槛 |
| C-summary-008 | condition=B1a | Cho2017 | 52 | 0.1 | -1.9256 | 0.45431 | 1 | 未检出差异；无方法门槛 |
| C-summary-009 | condition=B1a | Dreyer+Cho | 132 | -0.41667 | -1.2789 | 0.13362 | 1 | 未检出差异；无方法门槛 |
| C-summary-010 | condition=B1a | pooled | 235 | -3.8 | -5.4692 | 2.8573e-10 | 1.6001e-08 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-011 | condition=B1c | PhysionetMI | 103 | -10.682 | -11.843 | 6.3178e-13 | 4.9911e-11 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-012 | condition=B1c | Dreyer2023 | 80 | 0.33333 | -0.038506 | 0.96174 | 1 | 未检出差异；无方法门槛 |
| C-summary-013 | condition=B1c | Cho2017 | 52 | -0.2 | -1.5417 | 0.54946 | 1 | 未检出差异；无方法门槛 |
| C-summary-014 | condition=B1c | Dreyer+Cho | 132 | -0.083333 | -0.63066 | 0.6715 | 1 | 未检出差异；无方法门槛 |
| C-summary-015 | condition=B1c | pooled | 235 | -3.6 | -5.5452 | 1.2748e-10 | 7.7765e-09 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-016 | condition=B2_T3A | PhysionetMI | 103 | -5.0758 | -5.7198 | 6.6242e-12 | 4.5707e-10 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-017 | condition=B2_T3A | Dreyer2023 | 80 | -3.0833 | -4.7256 | 1.8118e-12 | 1.3589e-10 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-018 | condition=B2_T3A | Cho2017 | 52 | -2.5 | -6.9481 | 1.0962e-06 | 4.7138e-05 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-019 | condition=B2_T3A | Dreyer+Cho | 132 | -3 | -5.6011 | 1.5622e-17 | 1.4216e-15 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-020 | condition=B2_T3A | pooled | 235 | -3.6 | -5.6531 | 4.4031e-27 | 4.271e-25 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-021 | condition=B3_film | PhysionetMI | 103 | 6.5909 | 7.1389 | 8.5176e-12 | 5.6216e-10 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-022 | condition=B3_film | Dreyer2023 | 80 | 4.25 | 4.2642 | 9.3169e-13 | 7.174e-11 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-023 | condition=B3_film | Cho2017 | 52 | -4.2 | -4.9987 | 6.3232e-08 | 3.1616e-06 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-024 | condition=B3_film | Dreyer+Cho | 132 | 1.3667 | 0.61519 | 0.064405 | 1 | 未检出差异；无方法门槛 |
| C-summary-025 | condition=B3_film | pooled | 235 | 3.1667 | 3.4745 | 2.3159e-10 | 1.3432e-08 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-026 | condition=B3_lora | PhysionetMI | 103 | 7.7273 | 9.0761 | 8.9777e-16 | 7.9902e-14 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-027 | condition=B3_lora | Dreyer2023 | 80 | 8.9167 | 8.7868 | 7.8302e-15 | 6.6533e-13 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-028 | condition=B3_lora | Cho2017 | 52 | -0.8 | -0.73782 | 0.19925 | 1 | 未检出差异；无方法门槛 |
| C-summary-029 | condition=B3_lora | Dreyer+Cho | 132 | 4.6333 | 5.0347 | 1.0427e-13 | 8.3413e-12 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-030 | condition=B3_lora | pooled | 235 | 6.3333 | 6.806 | 3.4338e-28 | 3.3651e-26 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-031 | condition=B4_film | PhysionetMI | 103 | 6.1364 | 7.0137 | 7.3842e-12 | 5.0212e-10 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-032 | condition=B4_film | Dreyer2023 | 80 | 4.2083 | 3.9383 | 5.3296e-12 | 3.7307e-10 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-033 | condition=B4_film | Cho2017 | 52 | -4.9 | -5.1628 | 3.0201e-08 | 1.5403e-06 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-034 | condition=B4_film | Dreyer+Cho | 132 | 1.0833 | 0.35303 | 0.2066 | 1 | 未检出差异；无方法门槛 |
| C-summary-035 | condition=B4_film | pooled | 235 | 3.5 | 3.2724 | 2.2226e-09 | 1.2002e-07 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-036 | condition=B4_lora | PhysionetMI | 103 | 7.7273 | 9.2068 | 1.8878e-15 | 1.6424e-13 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-037 | condition=B4_lora | Dreyer2023 | 80 | 8.8333 | 8.8023 | 7.8288e-15 | 6.6533e-13 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-038 | condition=B4_lora | Cho2017 | 52 | -1.8167 | -2.091 | 0.0010321 | 0.035093 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-039 | condition=B4_lora | Dreyer+Cho | 132 | 4.75 | 4.511 | 1.8995e-10 | 1.1207e-08 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-040 | condition=B4_lora | pooled | 235 | 6.6 | 6.5692 | 1.7255e-24 | 1.6219e-22 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-041 | condition=B5_n10 | PhysionetMI | 103 | -0.22727 | -0.5636 | 0.50843 | 1 | 未检出差异；无方法门槛 |
| C-summary-042 | condition=B5_n10 | Dreyer2023 | 80 | -2 | -2.1558 | 0.0014956 | 0.049356 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-043 | condition=B5_n10 | Cho2017 | 52 | -0.3 | -1.2276 | 0.064252 | 1 | 未检出差异；无方法门槛 |
| C-summary-044 | condition=B5_n10 | Dreyer+Cho | 132 | -1.2667 | -1.7901 | 0.00027119 | 0.010305 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-045 | condition=B5_n10 | pooled | 235 | -0.66667 | -1.2525 | 0.0036374 | 0.10548 | 未检出差异；无方法门槛 |
| C-summary-046 | condition=B5_n20 | PhysionetMI | 103 | 1.8462 | 1.5219 | 0.072766 | 1 | 未检出差异；无方法门槛 |
| C-summary-047 | condition=B5_n20 | Dreyer2023 | 80 | 1 | 0.28341 | 0.57296 | 1 | 未检出差异；无方法门槛 |
| C-summary-048 | condition=B5_n20 | Cho2017 | 52 | 0.4 | -0.021154 | 0.97008 | 1 | 未检出差异；无方法门槛 |
| C-summary-049 | condition=B5_n20 | Dreyer+Cho | 132 | 0.4 | 0.16343 | 0.70293 | 1 | 未检出差异；无方法门槛 |
| C-summary-050 | condition=B5_n20 | pooled | 235 | 1 | 0.75885 | 0.10525 | 1 | 未检出差异；无方法门槛 |
| C-summary-051 | condition=B5_n40 | Dreyer2023 | 80 | 1.3333 | 1.7741 | 0.00074582 | 0.026104 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-052 | condition=B5_n40 | Cho2017 | 52 | 0.9 | 1.3468 | 0.13079 | 1 | 未检出差异；无方法门槛 |
| C-summary-053 | condition=B5_n40 | Dreyer+Cho | 132 | 1.1833 | 1.6058 | 0.00037907 | 0.014026 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-054 | condition=B5_n40 | pooled | 132 | 1.1833 | 1.6058 | 0.00037907 | 0.014026 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-055 | condition=B5_n80 | Dreyer2023 | 80 | 3 | 2.6389 | 3.0463e-07 | 1.3708e-05 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-056 | condition=B5_n80 | Cho2017 | 52 | 0.8 | 2.5577 | 0.04394 | 1 | 未检出差异；无方法门槛 |
| C-summary-057 | condition=B5_n80 | Dreyer+Cho | 132 | 2.3667 | 2.6069 | 1.7019e-07 | 8.1692e-06 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-058 | condition=B5_n80 | pooled | 132 | 2.3667 | 2.6069 | 1.7019e-07 | 8.1692e-06 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-059 | condition=B6_film_all | PhysionetMI | 103 | 2.5758 | 2.7137 | 0.0017605 | 0.056336 | 未检出差异；无方法门槛 |
| C-summary-060 | condition=B6_film_all | Dreyer2023 | 80 | 3.75 | 4.2168 | 2.2896e-11 | 1.4425e-09 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-061 | condition=B6_film_all | Cho2017 | 52 | 0.7 | 3.191 | 0.027918 | 0.75378 | 未检出差异；无方法门槛 |
| C-summary-062 | condition=B6_film_all | Dreyer+Cho | 132 | 3 | 3.8127 | 4.0182e-12 | 2.8529e-10 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-063 | condition=B6_film_all | pooled | 235 | 2.8333 | 3.331 | 1.1749e-12 | 8.9294e-11 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-064 | condition=B6_film_first2 | PhysionetMI | 103 | 1.8182 | 1.2585 | 0.019364 | 0.5422 | 未检出差异；无方法门槛 |
| C-summary-065 | condition=B6_film_first2 | Dreyer2023 | 80 | 2 | 2.1375 | 1.9834e-07 | 9.1239e-06 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-066 | condition=B6_film_first2 | Cho2017 | 52 | 1.8 | 2.9724 | 0.0034394 | 0.10318 | 未检出差异；无方法门槛 |
| C-summary-067 | condition=B6_film_first2 | Dreyer+Cho | 132 | 2 | 2.4664 | 4.5557e-09 | 2.4145e-07 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-068 | condition=B6_film_first2 | pooled | 235 | 1.8333 | 1.937 | 8.9318e-09 | 4.6446e-07 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-069 | condition=B6_lora4 | PhysionetMI | 103 | 2.8788 | 3.4126 | 8.8729e-06 | 0.00036379 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-070 | condition=B6_lora4 | Dreyer2023 | 80 | 3.1667 | 3.7295 | 3.4562e-12 | 2.4885e-10 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-071 | condition=B6_lora4 | Cho2017 | 52 | 2.2 | 4.6853 | 2.0581e-05 | 0.00082323 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-072 | condition=B6_lora4 | Dreyer+Cho | 132 | 2.8 | 4.106 | 6.8525e-16 | 6.1673e-14 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-073 | condition=B6_lora4 | pooled | 235 | 2.8 | 3.8021 | 1.3741e-18 | 1.2779e-16 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-074 | condition=B6_lora8 | PhysionetMI | 103 | 4.0152 | 4.0004 | 4.8445e-06 | 0.00020347 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-075 | condition=B6_lora8 | Dreyer2023 | 80 | 3.3333 | 3.9107 | 7.6364e-12 | 5.1164e-10 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-076 | condition=B6_lora8 | Cho2017 | 52 | 2.4 | 5.0994 | 7.8174e-05 | 0.0030488 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-077 | condition=B6_lora8 | Dreyer+Cho | 132 | 3.0833 | 4.3789 | 3.2333e-15 | 2.7806e-13 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-078 | condition=B6_lora8 | pooled | 235 | 3.4091 | 4.213 | 4.6519e-18 | 4.2798e-16 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-079 | condition=M_film | PhysionetMI | 103 | 6.0606 | 6.9155 | 1.6039e-11 | 1.0265e-09 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-080 | condition=M_film | Dreyer2023 | 80 | 3.9167 | 3.9612 | 2.5293e-12 | 1.8717e-10 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-081 | condition=M_film | Cho2017 | 52 | -4.7 | -5.0301 | 1.39e-07 | 6.8109e-06 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-082 | condition=M_film | Dreyer+Cho | 132 | 1.5 | 0.41919 | 0.12886 | 1 | 未检出差异；无方法门槛 |
| C-summary-083 | condition=M_film | pooled | 235 | 3.3333 | 3.2665 | 1.8151e-09 | 9.9832e-08 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-084 | condition=M_film_lambda0 | PhysionetMI | 103 | 5.3788 | 6.1634 | 9.9935e-12 | 6.4958e-10 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-085 | condition=M_film_lambda0 | Dreyer2023 | 80 | 3.1667 | 3.4755 | 7.9135e-13 | 6.1726e-11 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-086 | condition=M_film_lambda0 | Cho2017 | 52 | -3.6333 | -4.0314 | 7.0788e-07 | 3.1147e-05 | 检出双侧差异；均值反向；无方法门槛 |
| C-summary-087 | condition=M_film_lambda0 | Dreyer+Cho | 132 | 1.45 | 0.51823 | 0.061578 | 1 | 未检出差异；无方法门槛 |
| C-summary-088 | condition=M_film_lambda0 | pooled | 235 | 2.6 | 2.9925 | 2.6157e-10 | 1.491e-08 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-089 | condition=M_lora | PhysionetMI | 103 | 7.7273 | 9.233 | 1.4631e-15 | 1.2875e-13 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-090 | condition=M_lora | Dreyer2023 | 80 | 9 | 8.8022 | 7.8274e-15 | 6.6533e-13 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-091 | condition=M_lora | Cho2017 | 52 | -1.8 | -2.109 | 0.0028534 | 0.088456 | 未检出差异；无方法门槛 |
| C-summary-092 | condition=M_lora | Dreyer+Cho | 132 | 4.75 | 4.5039 | 1.8221e-10 | 1.0932e-08 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-093 | condition=M_lora | pooled | 235 | 6.2879 | 6.5766 | 1.4787e-24 | 1.4048e-22 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-094 | condition=M_lora_lambda0 | PhysionetMI | 103 | 7.197 | 8.2604 | 4.6788e-14 | 3.7899e-12 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-095 | condition=M_lora_lambda0 | Dreyer2023 | 80 | 8.1667 | 8.1796 | 8.729e-15 | 7.1578e-13 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-096 | condition=M_lora_lambda0 | Cho2017 | 52 | -0.6 | -1.1776 | 0.1436 | 1 | 未检出差异；无方法门槛 |
| C-summary-097 | condition=M_lora_lambda0 | Dreyer+Cho | 132 | 5.3333 | 4.4934 | 2.8017e-12 | 2.0452e-10 | 检出双侧差异；均值正向；无方法门槛 |
| C-summary-098 | condition=M_lora_lambda0 | pooled | 235 | 6.2879 | 6.1445 | 4.9658e-25 | 4.7672e-23 | 检出双侧差异；均值正向；无方法门槛 |

### C-all · 兼容的全部试次结果

各条件全部试次−B0。仅原表已有的全部试次列；与主后半设定分开，原始p，无新校正。 [完整源表](../reports/stage_c/report/summary.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| C-all-summary-001 | condition=B0 | PhysionetMI | 103 | 0 | 0 | 1 | — | 未检出差异；无方法门槛 |
| C-all-summary-002 | condition=B0 | Dreyer2023 | 80 | 0 | 0 | 1 | — | 未检出差异；无方法门槛 |
| C-all-summary-003 | condition=B0 | Cho2017 | 52 | 0 | 0 | 1 | — | 未检出差异；无方法门槛 |
| C-all-summary-004 | condition=B0 | Dreyer+Cho | 132 | 0 | 0 | 1 | — | 未检出差异；无方法门槛 |
| C-all-summary-005 | condition=B0 | pooled | 235 | 0 | 0 | 1 | — | 未检出差异；无方法门槛 |
| C-all-summary-006 | condition=B1a | PhysionetMI | 103 | -12.352 | -11.657 | 4.9419e-15 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-007 | condition=B1a | Dreyer2023 | 80 | -1.0833 | -0.72267 | 0.24671 | — | 未检出差异；无方法门槛 |
| C-all-summary-008 | condition=B1a | Cho2017 | 52 | -0.8 | -2.0087 | 0.44698 | — | 未检出差异；无方法门槛 |
| C-all-summary-009 | condition=B1a | Dreyer+Cho | 132 | -1.0417 | -1.2293 | 0.18207 | — | 未检出差异；无方法门槛 |
| C-all-summary-010 | condition=B1a | pooled | 235 | -3.8735 | -5.7995 | 1.4788e-14 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-011 | condition=B1c | PhysionetMI | 103 | -11.779 | -12.244 | 1.6129e-16 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-012 | condition=B1c | Dreyer2023 | 80 | -0.083333 | -0.17758 | 0.84501 | — | 未检出差异；无方法门槛 |
| C-all-summary-013 | condition=B1c | Cho2017 | 52 | -0.55 | -1.4478 | 0.62594 | — | 未检出差异；无方法门槛 |
| C-all-summary-014 | condition=B1c | Dreyer+Cho | 132 | -0.16667 | -0.67796 | 0.64461 | — | 未检出差异；无方法门槛 |
| C-all-summary-015 | condition=B1c | pooled | 235 | -3.9167 | -5.7473 | 1.6824e-14 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-016 | condition=B2_T3A | PhysionetMI | 103 | -4.7431 | -5.8553 | 3.0393e-14 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-017 | condition=B2_T3A | Dreyer2023 | 80 | -3 | -4.6948 | 5.8861e-13 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-018 | condition=B2_T3A | Cho2017 | 52 | -2.65 | -6.3205 | 1.9125e-07 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-019 | condition=B2_T3A | Dreyer+Cho | 132 | -2.875 | -5.3352 | 5.5253e-19 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-020 | condition=B2_T3A | pooled | 235 | -3.3333 | -5.5632 | 3.1917e-31 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-021 | condition=B3_film | PhysionetMI | 103 | 7.6482 | 6.6787 | 3.893e-13 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-022 | condition=B3_film | Dreyer2023 | 80 | 3.875 | 4.0865 | 1.6766e-13 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-023 | condition=B3_film | Cho2017 | 52 | -3.45 | -4.4968 | 3.0461e-09 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-024 | condition=B3_film | Dreyer+Cho | 132 | 1.5833 | 0.70519 | 0.027662 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-025 | condition=B3_film | pooled | 235 | 3.25 | 3.3234 | 3.7704e-12 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-026 | condition=B3_lora | PhysionetMI | 103 | 8.7747 | 8.4044 | 8.03e-16 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-027 | condition=B3_lora | Dreyer2023 | 80 | 9.1667 | 8.8717 | 7.8426e-15 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-028 | condition=B3_lora | Cho2017 | 52 | -0.25 | -0.36058 | 0.45893 | — | 未检出差异；无方法门槛 |
| C-all-summary-029 | condition=B3_lora | Dreyer+Cho | 132 | 5.45 | 5.2348 | 3.1942e-15 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-030 | condition=B3_lora | pooled | 235 | 6.9368 | 6.624 | 7.9839e-30 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-031 | condition=B4_film | PhysionetMI | 103 | 7.381 | 6.6927 | 1.5524e-13 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-032 | condition=B4_film | Dreyer2023 | 80 | 3.75 | 3.8084 | 4.6396e-13 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-033 | condition=B4_film | Cho2017 | 52 | -3.75 | -4.6907 | 1.0403e-08 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-034 | condition=B4_film | Dreyer+Cho | 132 | 1.3167 | 0.4603 | 0.085545 | — | 未检出差异；无方法门槛 |
| C-all-summary-035 | condition=B4_film | pooled | 235 | 3.1621 | 3.1919 | 1.802e-11 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-036 | condition=B4_lora | PhysionetMI | 103 | 8.913 | 8.4857 | 6.7409e-16 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-037 | condition=B4_lora | Dreyer2023 | 80 | 9.125 | 8.8735 | 7.8357e-15 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-038 | condition=B4_lora | Cho2017 | 52 | -1.5 | -1.651 | 0.0043012 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-039 | condition=B4_lora | Dreyer+Cho | 132 | 5.1583 | 4.7275 | 7.9178e-12 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-040 | condition=B4_lora | pooled | 235 | 7.25 | 6.3747 | 1.7213e-26 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-041 | condition=M_film | PhysionetMI | 103 | 7.381 | 6.6536 | 2.8354e-13 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-042 | condition=M_film | Dreyer2023 | 80 | 3.75 | 3.8558 | 3.3114e-13 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-043 | condition=M_film | Cho2017 | 52 | -3.8 | -4.6026 | 1.8149e-08 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-044 | condition=M_film | Dreyer+Cho | 132 | 1.375 | 0.52374 | 0.058188 | — | 未检出差异；无方法门槛 |
| C-all-summary-045 | condition=M_film | pooled | 235 | 3.0833 | 3.2105 | 1.3434e-11 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-046 | condition=M_film_lambda0 | PhysionetMI | 103 | 5.1786 | 5.7661 | 5.8772e-13 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-047 | condition=M_film_lambda0 | Dreyer2023 | 80 | 3.2917 | 3.3174 | 5.1557e-13 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-048 | condition=M_film_lambda0 | Cho2017 | 52 | -4 | -3.9112 | 1.7141e-07 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-049 | condition=M_film_lambda0 | Dreyer+Cho | 132 | 1.3333 | 0.46976 | 0.062119 | — | 未检出差异；无方法门槛 |
| C-all-summary-050 | condition=M_film_lambda0 | pooled | 235 | 2.9249 | 2.7911 | 2.4956e-11 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-051 | condition=M_lora | PhysionetMI | 103 | 8.6166 | 8.5075 | 5.2157e-16 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-052 | condition=M_lora | Dreyer2023 | 80 | 9.25 | 8.8923 | 7.8343e-15 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-053 | condition=M_lora | Cho2017 | 52 | -1.275 | -1.6673 | 0.0065534 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-054 | condition=M_lora | Dreyer+Cho | 132 | 5.25 | 4.7325 | 5.5528e-12 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-055 | condition=M_lora | pooled | 235 | 7.3333 | 6.3871 | 1.0112e-26 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-056 | condition=M_lora_lambda0 | PhysionetMI | 103 | 8.1818 | 7.7642 | 1.692e-14 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-057 | condition=M_lora_lambda0 | Dreyer2023 | 80 | 8.0833 | 8.1943 | 7.835e-15 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-058 | condition=M_lora_lambda0 | Cho2017 | 52 | -0.7 | -1.2638 | 0.020838 | — | 检出双侧差异；均值反向；无方法门槛 |
| C-all-summary-059 | condition=M_lora_lambda0 | Dreyer+Cho | 132 | 5.3333 | 4.4684 | 7.5739e-13 | — | 检出双侧差异；均值正向；无方法门槛 |
| C-all-summary-060 | condition=M_lora_lambda0 | pooled | 235 | 6.5833 | 5.9129 | 4.9959e-26 | — | 检出双侧差异；均值正向；无方法门槛 |

### C2 · 本人−交换分组诊断

本人参数−交换参数。15项双侧诊断；D1三项单侧联合门槛见第2节。 [完整源表](../reports/stage_c2/report/summary.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| C2-summary-001 | variant=film_offset | PhysionetMI | 103 | 4.4242 | 4.2991 | 3.157e-11 | 1.8942e-10 | 检出双侧差异；均值正向；无方法门槛 |
| C2-summary-002 | variant=film_offset | Dreyer2023 | 80 | 5.5167 | 5.9064 | 9.4796e-15 | 8.5317e-14 | 检出双侧差异；均值正向；无方法门槛 |
| C2-summary-003 | variant=film_offset | Cho2017 | 52 | 3.22 | 5.7617 | 4.4149e-08 | 1.7659e-07 | 检出双侧差异；均值正向；无方法门槛 |
| C2-summary-004 | variant=film_offset | Dreyer+Cho | 132 | 4.855 | 5.8494 | 5.4542e-22 | 5.9996e-21 | 检出双侧差异；均值正向；无方法门槛 |
| C2-summary-005 | variant=film_offset | pooled | 235 | 4.6167 | 5.1699 | 2.7787e-31 | 4.1681e-30 | 检出双侧差异；均值正向；无方法门槛 |
| C2-summary-006 | variant=mix_offset | PhysionetMI | 103 | 2.8333 | 2.5103 | 1.3608e-07 | 4.0824e-07 | 检出双侧差异；均值正向；无方法门槛 |
| C2-summary-007 | variant=mix_offset | Dreyer2023 | 80 | 2.525 | 2.4091 | 2.1888e-12 | 1.5321e-11 | 检出双侧差异；均值正向；无方法门槛 |
| C2-summary-008 | variant=mix_offset | Cho2017 | 52 | 2.24 | 3.3913 | 6.0155e-07 | 9.5455e-07 | 检出双侧差异；均值正向；无方法门槛 |
| C2-summary-009 | variant=mix_offset | Dreyer+Cho | 132 | 2.475 | 2.796 | 3.5898e-18 | 3.5898e-17 | 检出双侧差异；均值正向；无方法门槛 |
| C2-summary-010 | variant=mix_offset | pooled | 235 | 2.58 | 2.6708 | 1.4267e-22 | 1.8547e-21 | 检出双侧差异；均值正向；无方法门槛 |
| C2-summary-011 | variant=lora8 | PhysionetMI | 103 | 2.2615 | 2.2369 | 4.7727e-07 | 9.5455e-07 | 检出双侧差异；均值正向；无方法门槛 |
| C2-summary-012 | variant=lora8 | Dreyer2023 | 80 | 4.7417 | 5.307 | 9.8443e-15 | 8.5317e-14 | 检出双侧差异；均值正向；无方法门槛 |
| C2-summary-013 | variant=lora8 | Cho2017 | 52 | 4.51 | 6.3329 | 4.5377e-09 | 2.2688e-08 | 检出双侧差异；均值正向；无方法门槛 |
| C2-summary-014 | variant=lora8 | Dreyer+Cho | 132 | 4.6167 | 5.7111 | 1.9618e-22 | 2.3542e-21 | 检出双侧差异；均值正向；无方法门槛 |
| C2-summary-015 | variant=lora8 | pooled | 235 | 3.4697 | 4.1884 | 5.0123e-28 | 7.0172e-27 | 检出双侧差异；均值正向；无方法门槛 |

### C2 · 静息近邻完整比较

最近邻参数−随机参数。仅pooled三项为Holm主检验，其他分组为原始p描述。 [完整源表](../reports/stage_c2/report/diagnostic2_tests.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| C2-diagnostic2_tests-001 | variant=film_offset | PhysionetMI | 103 | 0.70455 | -0.046984 | 0.25475 | — | 未检出差异；无方法门槛 |
| C2-diagnostic2_tests-002 | variant=film_offset | Dreyer2023 | 80 | 0.058333 | 0.32351 | 0.23889 | — | 未检出差异；无方法门槛 |
| C2-diagnostic2_tests-003 | variant=film_offset | Cho2017 | 52 | 0.6 | 1.0803 | 0.033247 | — | 检出正向差异；无方法门槛 |
| C2-diagnostic2_tests-004 | variant=film_offset | Dreyer+Cho | 132 | 0.42 | 0.62162 | 0.03611 | — | 检出正向差异；无方法门槛 |
| C2-diagnostic2_tests-005 | variant=film_offset | pooled | 235 | 0.56061 | 0.32857 | 0.033745 | 0.10124 | 未检出差异；无方法门槛 |
| C2-diagnostic2_tests-006 | variant=mix_offset | PhysionetMI | 103 | -0.33333 | 0.16393 | 0.3581 | — | 未检出差异；无方法门槛 |
| C2-diagnostic2_tests-007 | variant=mix_offset | Dreyer2023 | 80 | 0.36667 | 0.34481 | 0.030866 | — | 检出正向差异；无方法门槛 |
| C2-diagnostic2_tests-008 | variant=mix_offset | Cho2017 | 52 | 0.58 | 0.7509 | 0.050102 | — | 未检出差异；无方法门槛 |
| C2-diagnostic2_tests-009 | variant=mix_offset | Dreyer+Cho | 132 | 0.49 | 0.50479 | 0.0063989 | — | 检出正向差异；无方法门槛 |
| C2-diagnostic2_tests-010 | variant=mix_offset | pooled | 235 | 0.36667 | 0.35539 | 0.045341 | 0.10124 | 未检出差异；无方法门槛 |
| C2-diagnostic2_tests-011 | variant=lora8 | PhysionetMI | 103 | -1.2121 | -1.7134 | 0.99526 | — | 未检出差异；无方法门槛 |
| C2-diagnostic2_tests-012 | variant=lora8 | Dreyer2023 | 80 | 0.43333 | 0.37648 | 0.16452 | — | 未检出差异；无方法门槛 |
| C2-diagnostic2_tests-013 | variant=lora8 | Cho2017 | 52 | 1.05 | 0.54827 | 0.06444 | — | 未检出差异；无方法门槛 |
| C2-diagnostic2_tests-014 | variant=lora8 | Dreyer+Cho | 132 | 0.85167 | 0.44415 | 0.040714 | — | 检出正向差异；无方法门槛 |
| C2-diagnostic2_tests-015 | variant=lora8 | pooled | 235 | 0.016667 | -0.5015 | 0.73119 | 0.73119 | 未检出差异；无方法门槛 |

### C3 · 静息/任务、k=1/3全部比较

近邻参数−同k随机参数。任务k=1合并主检验校正结果另见第2节；本表原始p。 [完整源表](../reports/stage_c3/report/rest_vs_task.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| C3-rest_vs_task-001 | variant=film_offset; source=rest; k=1 | PhysionetMI | 103 | 0.70455 | -0.046984 | 0.25475 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-002 | variant=film_offset; source=rest; k=1 | Dreyer2023 | 80 | 0.058333 | 0.32351 | 0.23889 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-003 | variant=film_offset; source=rest; k=1 | Cho2017 | 52 | 0.6 | 1.0803 | 0.033247 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-004 | variant=film_offset; source=rest; k=1 | Dreyer+Cho | 132 | 0.42 | 0.62162 | 0.03611 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-005 | variant=film_offset; source=rest; k=1 | pooled | 235 | 0.56061 | 0.32857 | 0.033745 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-006 | variant=film_offset; source=rest; k=3 | PhysionetMI | 103 | 0.21538 | -0.015746 | 0.34226 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-007 | variant=film_offset; source=rest; k=3 | Dreyer2023 | 80 | 0.091667 | 0.16953 | 0.11901 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-008 | variant=film_offset; source=rest; k=3 | Cho2017 | 52 | 0.58 | 0.98083 | 0.030637 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-009 | variant=film_offset; source=rest; k=3 | Dreyer+Cho | 132 | 0.275 | 0.48913 | 0.017233 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-010 | variant=film_offset; source=rest; k=3 | pooled | 235 | 0.25758 | 0.26785 | 0.037109 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-011 | variant=film_offset; source=task; k=1 | PhysionetMI | 103 | 1.0076 | 0.58063 | 0.014104 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-012 | variant=film_offset; source=task; k=1 | Dreyer2023 | 80 | 1.1583 | 0.68529 | 0.026771 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-013 | variant=film_offset; source=task; k=1 | Cho2017 | 52 | 0.82 | 0.79436 | 0.044281 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-014 | variant=film_offset; source=task; k=1 | Dreyer+Cho | 132 | 0.91667 | 0.72825 | 0.0046991 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-015 | variant=film_offset; source=task; k=1 | pooled | 235 | 0.91667 | 0.66355 | 0.00025849 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-016 | variant=film_offset; source=task; k=3 | PhysionetMI | 103 | -0.33077 | -0.12414 | 0.66676 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-017 | variant=film_offset; source=task; k=3 | Dreyer2023 | 80 | 0.775 | 0.64338 | 0.0040244 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-018 | variant=film_offset; source=task; k=3 | Cho2017 | 52 | 0.28 | 0.45583 | 0.15712 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-019 | variant=film_offset; source=task; k=3 | Dreyer+Cho | 132 | 0.57333 | 0.5695 | 0.0039504 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-020 | variant=film_offset; source=task; k=3 | pooled | 235 | 0.25 | 0.26548 | 0.055303 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-021 | variant=lora8; source=rest; k=1 | PhysionetMI | 103 | -1.2121 | -1.7134 | 0.99526 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-022 | variant=lora8; source=rest; k=1 | Dreyer2023 | 80 | 0.43333 | 0.37648 | 0.16452 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-023 | variant=lora8; source=rest; k=1 | Cho2017 | 52 | 1.05 | 0.54827 | 0.06444 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-024 | variant=lora8; source=rest; k=1 | Dreyer+Cho | 132 | 0.85167 | 0.44415 | 0.040714 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-025 | variant=lora8; source=rest; k=1 | pooled | 235 | 0.016667 | -0.5015 | 0.73119 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-026 | variant=lora8; source=rest; k=3 | PhysionetMI | 103 | -0.37879 | -0.36436 | 0.96657 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-027 | variant=lora8; source=rest; k=3 | Dreyer2023 | 80 | -0.4 | -0.10741 | 0.71348 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-028 | variant=lora8; source=rest; k=3 | Cho2017 | 52 | 0.34 | 0.54904 | 0.047607 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-029 | variant=lora8; source=rest; k=3 | Dreyer+Cho | 132 | 0.01 | 0.15119 | 0.25781 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-030 | variant=lora8; source=rest; k=3 | pooled | 235 | -0.1 | -0.074773 | 0.78018 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-031 | variant=lora8; source=task; k=1 | PhysionetMI | 103 | -0.11364 | -0.25502 | 0.65107 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-032 | variant=lora8; source=task; k=1 | Dreyer2023 | 80 | 0.525 | 0.4545 | 0.10663 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-033 | variant=lora8; source=task; k=1 | Cho2017 | 52 | 0.42 | 0.22712 | 0.1264 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-034 | variant=lora8; source=task; k=1 | Dreyer+Cho | 132 | 0.44667 | 0.36492 | 0.049062 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-035 | variant=lora8; source=task; k=1 | pooled | 235 | 0.18939 | 0.093201 | 0.17997 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-036 | variant=lora8; source=task; k=3 | PhysionetMI | 103 | 0.030303 | 0.092621 | 0.37621 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-037 | variant=lora8; source=task; k=3 | Dreyer2023 | 80 | 0.45833 | 0.39485 | 0.039143 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-038 | variant=lora8; source=task; k=3 | Cho2017 | 52 | 0.20833 | 0.016987 | 0.30145 | — | 未检出差异；无方法门槛 |
| C3-rest_vs_task-039 | variant=lora8; source=task; k=3 | Dreyer+Cho | 132 | 0.245 | 0.246 | 0.046522 | — | 检出正向差异；无方法门槛 |
| C3-rest_vs_task-040 | variant=lora8; source=task; k=3 | pooled | 235 | 0.21667 | 0.17877 | 0.071475 | — | 未检出差异；无方法门槛 |

### C3-few · 全部少样本曲线

前n个标签适配−B3。n40仅Dreyer/Cho；没有预定通过门槛，不新增检验。 [完整源表](../reports/stage_c3/report/few_shot_curve.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| C3-few-few_shot_curve-001 | variant=film_offset; shots=5 | PhysionetMI | 103 | -1.6667 | -1.8771 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-002 | variant=film_offset; shots=5 | Dreyer2023 | 80 | -5.0833 | -6.1618 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-003 | variant=film_offset; shots=5 | Cho2017 | 52 | 0.5 | 0.65385 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-004 | variant=film_offset; shots=5 | Dreyer+Cho | 132 | -2.5 | -3.4768 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-005 | variant=film_offset; shots=5 | pooled | 235 | -2.3333 | -2.7757 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-006 | variant=film_offset; shots=10 | PhysionetMI | 103 | -0.37879 | 0.026173 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-007 | variant=film_offset; shots=10 | Dreyer2023 | 80 | -2.5 | -3.5972 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-008 | variant=film_offset; shots=10 | Cho2017 | 52 | 1.7 | 2.9929 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-009 | variant=film_offset; shots=10 | Dreyer+Cho | 132 | -0.83333 | -1.0011 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-010 | variant=film_offset; shots=10 | pooled | 235 | -0.68966 | -0.55082 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-011 | variant=film_offset; shots=20 | PhysionetMI | 103 | 0.90909 | 1.5759 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-012 | variant=film_offset; shots=20 | Dreyer2023 | 80 | -0.91667 | -1.1862 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-013 | variant=film_offset; shots=20 | Cho2017 | 52 | 1.1833 | 2.8962 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-014 | variant=film_offset; shots=20 | Dreyer+Cho | 132 | -0.2 | 0.422 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-015 | variant=film_offset; shots=20 | pooled | 235 | 0.30303 | 0.92773 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-016 | variant=film_offset; shots=40 | Dreyer2023 | 80 | 0.5 | -0.062859 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-017 | variant=film_offset; shots=40 | Cho2017 | 52 | 2.4667 | 4.166 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-018 | variant=film_offset; shots=40 | Dreyer+Cho | 132 | 0.81667 | 1.6031 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-019 | variant=film_offset; shots=40 | pooled | 132 | 0.81667 | 1.6031 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-020 | variant=lora8; shots=5 | PhysionetMI | 103 | 0.76923 | 0.77728 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-021 | variant=lora8; shots=5 | Dreyer2023 | 80 | -0.5 | -0.17866 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-022 | variant=lora8; shots=5 | Cho2017 | 52 | 0.1 | 0.19295 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-023 | variant=lora8; shots=5 | Dreyer+Cho | 132 | -0.18333 | -0.032271 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-024 | variant=lora8; shots=5 | pooled | 235 | 0.2 | 0.32255 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-025 | variant=lora8; shots=10 | PhysionetMI | 103 | 0.76923 | 0.93005 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-026 | variant=lora8; shots=10 | Dreyer2023 | 80 | 0 | -0.093175 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-027 | variant=lora8; shots=10 | Cho2017 | 52 | 0.65 | 1.2692 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-028 | variant=lora8; shots=10 | Dreyer+Cho | 132 | 0.083333 | 0.44353 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-029 | variant=lora8; shots=10 | pooled | 235 | 0.38462 | 0.65677 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-030 | variant=lora8; shots=20 | PhysionetMI | 103 | 1.2121 | 1.4977 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-031 | variant=lora8; shots=20 | Dreyer2023 | 80 | 0.5 | 0.40981 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-032 | variant=lora8; shots=20 | Cho2017 | 52 | 0.2 | 1.3635 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-033 | variant=lora8; shots=20 | Dreyer+Cho | 132 | 0.5 | 0.78549 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-034 | variant=lora8; shots=20 | pooled | 235 | 0.83333 | 1.0976 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-035 | variant=lora8; shots=40 | Dreyer2023 | 80 | 1.5833 | 1.6399 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-036 | variant=lora8; shots=40 | Cho2017 | 52 | 1 | 2.3603 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-037 | variant=lora8; shots=40 | Dreyer+Cho | 132 | 1.3333 | 1.9237 | — | — | 不适用（描述） |
| C3-few-few_shot_curve-038 | variant=lora8; shots=40 | pooled | 132 | 1.3333 | 1.9237 | — | — | 不适用（描述） |

### C4 · 全部先验与标签量

所列先验左项−右项。90项完整描述；指定主/次终点及其校正见第2节。 [完整源表](../reports/stage_c4/report/comparisons.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| C4-comparisons-001 | variant=film_offset; shots=0; comparison=P2-P1 | PhysionetMI | 103 | 0 | -0.30979 | 0.85473 | — | 未检出差异；无方法门槛 |
| C4-comparisons-002 | variant=film_offset; shots=0; comparison=P2-P1 | Dreyer2023 | 80 | 0.23333 | 0.34329 | 0.020042 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-003 | variant=film_offset; shots=0; comparison=P2-P1 | Cho2017 | 52 | 0.67 | 0.69359 | 0.0023769 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-004 | variant=film_offset; shots=0; comparison=P2-P1 | Dreyer+Cho | 132 | 0.45 | 0.48128 | 0.00025211 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-005 | variant=film_offset; shots=0; comparison=P2-P1 | pooled | 235 | 0.29545 | 0.13456 | 0.049621 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-006 | variant=film_offset; shots=0; comparison=P2-P0 | PhysionetMI | 103 | 0.23077 | 0.77409 | 0.023399 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-007 | variant=film_offset; shots=0; comparison=P2-P0 | Dreyer2023 | 80 | 0.66667 | 0.67425 | 0.00023904 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-008 | variant=film_offset; shots=0; comparison=P2-P0 | Cho2017 | 52 | 1.2 | 1.5962 | 0.00081664 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-009 | variant=film_offset; shots=0; comparison=P2-P0 | Dreyer+Cho | 132 | 0.66667 | 1.0374 | 1.1013e-06 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-010 | variant=film_offset; shots=0; comparison=P2-P0 | pooled | 235 | 0.66667 | 0.922 | 2.2977e-06 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-011 | variant=film_offset; shots=0; comparison=P1-P0 | PhysionetMI | 103 | 0.74242 | 1.0839 | 4.5821e-05 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-012 | variant=film_offset; shots=0; comparison=P1-P0 | Dreyer2023 | 80 | 0.34167 | 0.33096 | 0.032168 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-013 | variant=film_offset; shots=0; comparison=P1-P0 | Cho2017 | 52 | 0.98 | 0.90256 | 0.0057467 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-014 | variant=film_offset; shots=0; comparison=P1-P0 | Dreyer+Cho | 132 | 0.52 | 0.55614 | 0.00075259 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-015 | variant=film_offset; shots=0; comparison=P1-P0 | pooled | 235 | 0.6 | 0.78744 | 2.8539e-07 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-016 | variant=film_offset; shots=10; comparison=P2-P1 | PhysionetMI | 103 | 0.15909 | 0.073566 | 0.39407 | — | 未检出差异；无方法门槛 |
| C4-comparisons-017 | variant=film_offset; shots=10; comparison=P2-P1 | Dreyer2023 | 80 | 0.23333 | 0.29325 | 0.067715 | — | 未检出差异；无方法门槛 |
| C4-comparisons-018 | variant=film_offset; shots=10; comparison=P2-P1 | Cho2017 | 52 | 0.1 | 0.2241 | 0.28307 | — | 未检出差异；无方法门槛 |
| C4-comparisons-019 | variant=film_offset; shots=10; comparison=P2-P1 | Dreyer+Cho | 132 | 0.19167 | 0.26601 | 0.061883 | — | 未检出差异；无方法门槛 |
| C4-comparisons-020 | variant=film_offset; shots=10; comparison=P2-P1 | pooled | 235 | 0.16667 | 0.18166 | 0.083034 | — | 未检出差异；无方法门槛 |
| C4-comparisons-021 | variant=film_offset; shots=10; comparison=P2-P0 | PhysionetMI | 103 | 0 | 0.50837 | 0.028789 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-022 | variant=film_offset; shots=10; comparison=P2-P0 | Dreyer2023 | 80 | 0.33333 | 0.54702 | 0.013821 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-023 | variant=film_offset; shots=10; comparison=P2-P0 | Cho2017 | 52 | 0 | 0.10833 | 0.4449 | — | 未检出差异；无方法门槛 |
| C4-comparisons-024 | variant=film_offset; shots=10; comparison=P2-P0 | Dreyer+Cho | 132 | 0.18333 | 0.3742 | 0.039555 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-025 | variant=film_offset; shots=10; comparison=P2-P0 | pooled | 235 | 0.16667 | 0.43301 | 0.0053648 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-026 | variant=film_offset; shots=10; comparison=P1-P0 | PhysionetMI | 103 | 0.7 | 0.43481 | 0.019102 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-027 | variant=film_offset; shots=10; comparison=P1-P0 | Dreyer2023 | 80 | 0.066667 | 0.25377 | 0.12843 | — | 未检出差异；无方法门槛 |
| C4-comparisons-028 | variant=film_offset; shots=10; comparison=P1-P0 | Cho2017 | 52 | -0.23 | -0.11577 | 0.6129 | — | 未检出差异；无方法门槛 |
| C4-comparisons-029 | variant=film_offset; shots=10; comparison=P1-P0 | Dreyer+Cho | 132 | 0.018333 | 0.10819 | 0.24898 | — | 未检出差异；无方法门槛 |
| C4-comparisons-030 | variant=film_offset; shots=10; comparison=P1-P0 | pooled | 235 | 0.1 | 0.25135 | 0.018859 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-031 | variant=lora8; shots=0; comparison=P2-P1 | PhysionetMI | 103 | 0.14394 | 0.081755 | 0.22959 | — | 未检出差异；无方法门槛 |
| C4-comparisons-032 | variant=lora8; shots=0; comparison=P2-P1 | Dreyer2023 | 80 | 0.20833 | 0.281 | 0.020393 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-033 | variant=lora8; shots=0; comparison=P2-P1 | Cho2017 | 52 | 0.14 | 0.022949 | 0.33657 | — | 未检出差异；无方法门槛 |
| C4-comparisons-034 | variant=lora8; shots=0; comparison=P2-P1 | Dreyer+Cho | 132 | 0.19 | 0.17934 | 0.039817 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-035 | variant=lora8; shots=0; comparison=P2-P1 | pooled | 235 | 0.18 | 0.13657 | 0.039218 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-036 | variant=lora8; shots=0; comparison=P2-P0 | PhysionetMI | 103 | 0.53846 | 0.52474 | 0.016083 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-037 | variant=lora8; shots=0; comparison=P2-P0 | Dreyer2023 | 80 | 1.4167 | 1.2618 | 4.6708e-08 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-038 | variant=lora8; shots=0; comparison=P2-P0 | Cho2017 | 52 | 1.2 | 1.0795 | 0.0044297 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-039 | variant=lora8; shots=0; comparison=P2-P0 | Dreyer+Cho | 132 | 1.2 | 1.19 | 7.024e-09 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-040 | variant=lora8; shots=0; comparison=P2-P0 | pooled | 235 | 0.90909 | 0.89841 | 7.7044e-09 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-041 | variant=lora8; shots=0; comparison=P1-P0 | PhysionetMI | 103 | 0.16667 | 0.44298 | 0.012487 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-042 | variant=lora8; shots=0; comparison=P1-P0 | Dreyer2023 | 80 | 1.0167 | 0.98082 | 2.981e-08 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-043 | variant=lora8; shots=0; comparison=P1-P0 | Cho2017 | 52 | 0.88 | 1.0565 | 0.00056218 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-044 | variant=lora8; shots=0; comparison=P1-P0 | Dreyer+Cho | 132 | 0.95 | 1.0106 | 2.9869e-10 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-045 | variant=lora8; shots=0; comparison=P1-P0 | pooled | 235 | 0.63333 | 0.76184 | 3.1124e-10 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-046 | variant=lora8; shots=5; comparison=P2-P1 | PhysionetMI | 103 | -0.10606 | -0.1606 | 0.87531 | — | 未检出差异；无方法门槛 |
| C4-comparisons-047 | variant=lora8; shots=5; comparison=P2-P1 | Dreyer2023 | 80 | 0.28333 | 0.31024 | 0.052233 | — | 未检出差异；无方法门槛 |
| C4-comparisons-048 | variant=lora8; shots=5; comparison=P2-P1 | Cho2017 | 52 | 0.02 | 0.32154 | 0.14305 | — | 未检出差异；无方法门槛 |
| C4-comparisons-049 | variant=lora8; shots=5; comparison=P2-P1 | Dreyer+Cho | 132 | 0.17333 | 0.31469 | 0.024191 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-050 | variant=lora8; shots=5; comparison=P2-P1 | pooled | 235 | 0.1 | 0.10637 | 0.29768 | — | 未检出差异；无方法门槛 |
| C4-comparisons-051 | variant=lora8; shots=5; comparison=P2-P0 | PhysionetMI | 103 | 0 | 0.36841 | 0.19321 | — | 未检出差异；无方法门槛 |
| C4-comparisons-052 | variant=lora8; shots=5; comparison=P2-P0 | Dreyer2023 | 80 | 0.5 | 0.81394 | 0.0013405 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-053 | variant=lora8; shots=5; comparison=P2-P0 | Cho2017 | 52 | 0.1 | 0.64487 | 0.059496 | — | 未检出差异；无方法门槛 |
| C4-comparisons-054 | variant=lora8; shots=5; comparison=P2-P0 | Dreyer+Cho | 132 | 0.2 | 0.74734 | 0.00055267 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-055 | variant=lora8; shots=5; comparison=P2-P0 | pooled | 235 | 0.16667 | 0.58125 | 0.0011282 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-056 | variant=lora8; shots=5; comparison=P1-P0 | PhysionetMI | 103 | 0.53846 | 0.52901 | 0.0016195 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-057 | variant=lora8; shots=5; comparison=P1-P0 | Dreyer2023 | 80 | 0.36667 | 0.5037 | 0.0030556 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-058 | variant=lora8; shots=5; comparison=P1-P0 | Cho2017 | 52 | 0.27833 | 0.32333 | 0.11239 | — | 未检出差异；无方法门槛 |
| C4-comparisons-059 | variant=lora8; shots=5; comparison=P1-P0 | Dreyer+Cho | 132 | 0.33333 | 0.43265 | 0.0018656 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-060 | variant=lora8; shots=5; comparison=P1-P0 | pooled | 235 | 0.4 | 0.47488 | 1.9192e-05 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-061 | variant=lora8; shots=10; comparison=P2-P1 | PhysionetMI | 103 | -0.061538 | -0.10881 | 0.6572 | — | 未检出差异；无方法门槛 |
| C4-comparisons-062 | variant=lora8; shots=10; comparison=P2-P1 | Dreyer2023 | 80 | 0.0083333 | -0.13445 | 0.65913 | — | 未检出差异；无方法门槛 |
| C4-comparisons-063 | variant=lora8; shots=10; comparison=P2-P1 | Cho2017 | 52 | -0.35 | -0.30147 | 0.90042 | — | 未检出差异；无方法门槛 |
| C4-comparisons-064 | variant=lora8; shots=10; comparison=P2-P1 | Dreyer+Cho | 132 | -0.15333 | -0.20025 | 0.90675 | — | 未检出差异；无方法门槛 |
| C4-comparisons-065 | variant=lora8; shots=10; comparison=P2-P1 | pooled | 235 | -0.075758 | -0.16017 | 0.89273 | — | 未检出差异；无方法门槛 |
| C4-comparisons-066 | variant=lora8; shots=10; comparison=P2-P0 | PhysionetMI | 103 | 0 | -0.24195 | 0.86696 | — | 未检出差异；无方法门槛 |
| C4-comparisons-067 | variant=lora8; shots=10; comparison=P2-P0 | Dreyer2023 | 80 | 0.5 | 0.22769 | 0.045307 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-068 | variant=lora8; shots=10; comparison=P2-P0 | Cho2017 | 52 | 0.2 | -0.053205 | 0.46115 | — | 未检出差异；无方法门槛 |
| C4-comparisons-069 | variant=lora8; shots=10; comparison=P2-P0 | Dreyer+Cho | 132 | 0.33333 | 0.11704 | 0.091811 | — | 未检出差异；无方法门槛 |
| C4-comparisons-070 | variant=lora8; shots=10; comparison=P2-P0 | pooled | 235 | 0.15152 | -0.040306 | 0.44768 | — | 未检出差异；无方法门槛 |
| C4-comparisons-071 | variant=lora8; shots=10; comparison=P1-P0 | PhysionetMI | 103 | -0.0075758 | -0.13314 | 0.71594 | — | 未检出差异；无方法门槛 |
| C4-comparisons-072 | variant=lora8; shots=10; comparison=P1-P0 | Dreyer2023 | 80 | 0.275 | 0.36214 | 0.0023979 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-073 | variant=lora8; shots=10; comparison=P1-P0 | Cho2017 | 52 | 0.22 | 0.24827 | 0.11878 | — | 未检出差异；无方法门槛 |
| C4-comparisons-074 | variant=lora8; shots=10; comparison=P1-P0 | Dreyer+Cho | 132 | 0.26333 | 0.31728 | 0.0022841 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-075 | variant=lora8; shots=10; comparison=P1-P0 | pooled | 235 | 0.13333 | 0.11986 | 0.049781 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-076 | variant=lora8; shots=20; comparison=P2-P1 | PhysionetMI | 103 | -0.068182 | -0.027489 | 0.6634 | — | 未检出差异；无方法门槛 |
| C4-comparisons-077 | variant=lora8; shots=20; comparison=P2-P1 | Dreyer2023 | 80 | -0.076437 | -0.057953 | 0.6539 | — | 未检出差异；无方法门槛 |
| C4-comparisons-078 | variant=lora8; shots=20; comparison=P2-P1 | Cho2017 | 52 | -0.4 | -0.10301 | 0.668 | — | 未检出差异；无方法门槛 |
| C4-comparisons-079 | variant=lora8; shots=20; comparison=P2-P1 | Dreyer+Cho | 132 | -0.11667 | -0.075704 | 0.74404 | — | 未检出差异；无方法门槛 |
| C4-comparisons-080 | variant=lora8; shots=20; comparison=P2-P1 | pooled | 235 | -0.075758 | -0.054571 | 0.77644 | — | 未检出差异；无方法门槛 |
| C4-comparisons-081 | variant=lora8; shots=20; comparison=P2-P0 | PhysionetMI | 103 | 0.075758 | 0.14241 | 0.20751 | — | 未检出差异；无方法门槛 |
| C4-comparisons-082 | variant=lora8; shots=20; comparison=P2-P0 | Dreyer2023 | 80 | 0.41667 | 0.37874 | 0.011125 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-083 | variant=lora8; shots=20; comparison=P2-P0 | Cho2017 | 52 | 0.3 | 0.56474 | 0.079206 | — | 未检出差异；无方法门槛 |
| C4-comparisons-084 | variant=lora8; shots=20; comparison=P2-P0 | Dreyer+Cho | 132 | 0.36667 | 0.45201 | 0.0044374 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-085 | variant=lora8; shots=20; comparison=P2-P0 | pooled | 235 | 0.33333 | 0.31631 | 0.0074017 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-086 | variant=lora8; shots=20; comparison=P1-P0 | PhysionetMI | 103 | 0.17424 | 0.16989 | 0.1261 | — | 未检出差异；无方法门槛 |
| C4-comparisons-087 | variant=lora8; shots=20; comparison=P1-P0 | Dreyer2023 | 80 | 0.26667 | 0.43669 | 0.0002954 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-088 | variant=lora8; shots=20; comparison=P1-P0 | Cho2017 | 52 | 0.22 | 0.66776 | 0.0069053 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-089 | variant=lora8; shots=20; comparison=P1-P0 | Dreyer+Cho | 132 | 0.23667 | 0.52772 | 1.1459e-05 | — | 检出正向差异；无方法门槛 |
| C4-comparisons-090 | variant=lora8; shots=20; comparison=P1-P0 | pooled | 235 | 0.23333 | 0.37088 | 5.955e-05 | — | 检出正向差异；无方法门槛 |

### M · 全部方法、标签量及适配增益

所列方法左项−右项。206项描述；ba为绝对BA差，adapt_gain为(n−0)增益之差；主/次终点见第2节。 [完整源表](../reports/stage_m/report/comparisons.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| M-comparisons-001 | shots=0; comparison=M1-R0; measure=ba | PhysionetMI | 103 | 0.075758 | 0.26975 | 0.10632 | — | 未检出差异；无方法门槛 |
| M-comparisons-002 | shots=0; comparison=M1-R0; measure=ba | Dreyer2023 | 80 | 2 | 1.8943 | 1.8526e-09 | — | 检出正向差异；无方法门槛 |
| M-comparisons-003 | shots=0; comparison=M1-R0; measure=ba | Cho2017 | 52 | 0.6 | 1.0962 | 0.011617 | — | 检出正向差异；无方法门槛 |
| M-comparisons-004 | shots=0; comparison=M1-R0; measure=ba | Dreyer+Cho | 132 | 1.8333 | 1.5799 | 7.6465e-10 | — | 检出正向差异；无方法门槛 |
| M-comparisons-005 | shots=0; comparison=M1-R0; measure=ba | pooled | 235 | 0.98485 | 1.0056 | 7.0425e-09 | — | 检出正向差异；无方法门槛 |
| M-comparisons-006 | shots=0; comparison=M1-R1; measure=ba | PhysionetMI | 103 | 0.045455 | -0.17323 | 0.67936 | — | 未检出差异；无方法门槛 |
| M-comparisons-007 | shots=0; comparison=M1-R1; measure=ba | Dreyer2023 | 80 | 0.77069 | 0.91343 | 0.00011388 | — | 检出正向差异；无方法门槛 |
| M-comparisons-008 | shots=0; comparison=M1-R1; measure=ba | Cho2017 | 52 | 0.2 | 0.039615 | 0.24102 | — | 未检出差异；无方法门槛 |
| M-comparisons-009 | shots=0; comparison=M1-R1; measure=ba | Dreyer+Cho | 132 | 0.57333 | 0.5692 | 0.00064961 | — | 检出正向差异；无方法门槛 |
| M-comparisons-010 | shots=0; comparison=M1-R1; measure=ba | pooled | 235 | 0.30303 | 0.2438 | 0.015879 | — | 检出正向差异；无方法门槛 |
| M-comparisons-011 | shots=0; comparison=M1-R2; measure=ba | PhysionetMI | 103 | 0 | -0.15571 | 0.729 | — | 未检出差异；无方法门槛 |
| M-comparisons-012 | shots=0; comparison=M1-R2; measure=ba | Dreyer2023 | 80 | 0.083333 | -0.089045 | 0.66645 | — | 未检出差异；无方法门槛 |
| M-comparisons-013 | shots=0; comparison=M1-R2; measure=ba | Cho2017 | 52 | -0.7 | -1.0135 | 0.9949 | — | 未检出差异；无方法门槛 |
| M-comparisons-014 | shots=0; comparison=M1-R2; measure=ba | Dreyer+Cho | 132 | -0.16667 | -0.45321 | 0.98902 | — | 未检出差异；无方法门槛 |
| M-comparisons-015 | shots=0; comparison=M1-R2; measure=ba | pooled | 235 | 0 | -0.32282 | 0.98328 | — | 未检出差异；无方法门槛 |
| M-comparisons-016 | shots=0; comparison=M2-R0; measure=ba | PhysionetMI | 103 | 0.075758 | 0.27281 | 0.154 | — | 未检出差异；无方法门槛 |
| M-comparisons-017 | shots=0; comparison=M2-R0; measure=ba | Dreyer2023 | 80 | 2 | 1.8829 | 3.2385e-09 | — | 检出正向差异；无方法门槛 |
| M-comparisons-018 | shots=0; comparison=M2-R0; measure=ba | Cho2017 | 52 | 1 | 1.3378 | 0.003399 | — | 检出正向差异；无方法门槛 |
| M-comparisons-019 | shots=0; comparison=M2-R0; measure=ba | Dreyer+Cho | 132 | 1.7333 | 1.6682 | 1.9068e-10 | — | 检出正向差异；无方法门槛 |
| M-comparisons-020 | shots=0; comparison=M2-R0; measure=ba | pooled | 235 | 1 | 1.0566 | 5.5023e-09 | — | 检出正向差异；无方法门槛 |
| M-comparisons-021 | shots=0; comparison=M2-R1; measure=ba | PhysionetMI | 103 | -0.022727 | -0.17017 | 0.75471 | — | 未检出差异；无方法门槛 |
| M-comparisons-022 | shots=0; comparison=M2-R1; measure=ba | Dreyer2023 | 80 | 0.89023 | 0.90205 | 5.813e-05 | — | 检出正向差异；无方法门槛 |
| M-comparisons-023 | shots=0; comparison=M2-R1; measure=ba | Cho2017 | 52 | 0.32 | 0.28128 | 0.11062 | — | 未检出差异；无方法门槛 |
| M-comparisons-024 | shots=0; comparison=M2-R1; measure=ba | Dreyer+Cho | 132 | 0.71667 | 0.6575 | 0.0001124 | — | 检出正向差异；无方法门槛 |
| M-comparisons-025 | shots=0; comparison=M2-R1; measure=ba | pooled | 235 | 0.28333 | 0.29473 | 0.0094975 | — | 检出正向差异；无方法门槛 |
| M-comparisons-026 | shots=0; comparison=M2-R2; measure=ba | PhysionetMI | 103 | 0 | -0.15266 | 0.71186 | — | 未检出差异；无方法门槛 |
| M-comparisons-027 | shots=0; comparison=M2-R2; measure=ba | Dreyer2023 | 80 | -0.16667 | -0.10043 | 0.7186 | — | 未检出差异；无方法门槛 |
| M-comparisons-028 | shots=0; comparison=M2-R2; measure=ba | Cho2017 | 52 | -0.2 | -0.77179 | 0.94211 | — | 未检出差异；无方法门槛 |
| M-comparisons-029 | shots=0; comparison=M2-R2; measure=ba | Dreyer+Cho | 132 | -0.16667 | -0.36491 | 0.93452 | — | 未检出差异；无方法门槛 |
| M-comparisons-030 | shots=0; comparison=M2-R2; measure=ba | pooled | 235 | -0.075758 | -0.27188 | 0.94437 | — | 未检出差异；无方法门槛 |
| M-comparisons-031 | shots=0; comparison=M2-M1; measure=ba | PhysionetMI | 103 | 0 | 0.0030552 | 0.46992 | — | 未检出差异；无方法门槛 |
| M-comparisons-032 | shots=0; comparison=M2-M1; measure=ba | Dreyer2023 | 80 | 0 | -0.011386 | 0.59858 | — | 未检出差异；无方法门槛 |
| M-comparisons-033 | shots=0; comparison=M2-M1; measure=ba | Cho2017 | 52 | 0.1 | 0.24167 | 0.14751 | — | 未检出差异；无方法门槛 |
| M-comparisons-034 | shots=0; comparison=M2-M1; measure=ba | Dreyer+Cho | 132 | 0 | 0.088301 | 0.28348 | — | 未检出差异；无方法门槛 |
| M-comparisons-035 | shots=0; comparison=M2-M1; measure=ba | pooled | 235 | 0 | 0.050938 | 0.32254 | — | 未检出差异；无方法门槛 |
| M-comparisons-036 | shots=5; comparison=M1-R0; measure=ba | PhysionetMI | 103 | 0 | -0.041121 | 0.63225 | — | 未检出差异；无方法门槛 |
| M-comparisons-037 | shots=5; comparison=M1-R0; measure=ba | Dreyer2023 | 80 | 1.2701 | 1.2245 | 4.3489e-06 | — | 检出正向差异；无方法门槛 |
| M-comparisons-038 | shots=5; comparison=M1-R0; measure=ba | Cho2017 | 52 | 1.6 | 1.6244 | 2.3945e-05 | — | 检出正向差异；无方法门槛 |
| M-comparisons-039 | shots=5; comparison=M1-R0; measure=ba | Dreyer+Cho | 132 | 1.4 | 1.382 | 8.1647e-10 | — | 检出正向差异；无方法门槛 |
| M-comparisons-040 | shots=5; comparison=M1-R0; measure=ba | pooled | 235 | 0.90909 | 0.75825 | 6.7759e-06 | — | 检出正向差异；无方法门槛 |
| M-comparisons-041 | shots=5; comparison=M1-R1; measure=ba | PhysionetMI | 103 | -0.27273 | -0.19926 | 0.89186 | — | 未检出差异；无方法门槛 |
| M-comparisons-042 | shots=5; comparison=M1-R1; measure=ba | Dreyer2023 | 80 | 1.175 | 1.1575 | 1.094e-05 | — | 检出正向差异；无方法门槛 |
| M-comparisons-043 | shots=5; comparison=M1-R1; measure=ba | Cho2017 | 52 | 1.13 | 0.81679 | 0.012688 | — | 检出正向差异；无方法门槛 |
| M-comparisons-044 | shots=5; comparison=M1-R1; measure=ba | Dreyer+Cho | 132 | 1.175 | 1.0233 | 2.2541e-06 | — | 检出正向差异；无方法门槛 |
| M-comparisons-045 | shots=5; comparison=M1-R1; measure=ba | pooled | 235 | 0.48333 | 0.48743 | 0.0036885 | — | 检出正向差异；无方法门槛 |
| M-comparisons-046 | shots=5; comparison=M1-R2; measure=ba | PhysionetMI | 103 | 0 | -0.32984 | 0.91158 | — | 未检出差异；无方法门槛 |
| M-comparisons-047 | shots=5; comparison=M1-R2; measure=ba | Dreyer2023 | 80 | 0 | -0.24479 | 0.8472 | — | 未检出差异；无方法门槛 |
| M-comparisons-048 | shots=5; comparison=M1-R2; measure=ba | Cho2017 | 52 | 0 | 0.078846 | 0.57143 | — | 未检出差异；无方法门槛 |
| M-comparisons-049 | shots=5; comparison=M1-R2; measure=ba | Dreyer+Cho | 132 | 0 | -0.1173 | 0.83047 | — | 未检出差异；无方法门槛 |
| M-comparisons-050 | shots=5; comparison=M1-R2; measure=ba | pooled | 235 | 0 | -0.21045 | 0.94651 | — | 未检出差异；无方法门槛 |
| M-comparisons-051 | shots=5; comparison=M2-R0; measure=ba | PhysionetMI | 103 | -0.15152 | 0.08283 | 0.46827 | — | 未检出差异；无方法门槛 |
| M-comparisons-052 | shots=5; comparison=M2-R0; measure=ba | Dreyer2023 | 80 | 1.5833 | 1.3351 | 1.5752e-07 | — | 检出正向差异；无方法门槛 |
| M-comparisons-053 | shots=5; comparison=M2-R0; measure=ba | Cho2017 | 52 | 1.4 | 1.5833 | 1.6453e-05 | — | 检出正向差异；无方法门槛 |
| M-comparisons-054 | shots=5; comparison=M2-R0; measure=ba | Dreyer+Cho | 132 | 1.5 | 1.4329 | 3.4677e-11 | — | 检出正向差异；无方法门槛 |
| M-comparisons-055 | shots=5; comparison=M2-R0; measure=ba | pooled | 235 | 0.90909 | 0.84116 | 1.3925e-07 | — | 检出正向差异；无方法门槛 |
| M-comparisons-056 | shots=5; comparison=M2-R1; measure=ba | PhysionetMI | 103 | 0 | -0.075312 | 0.64907 | — | 未检出差异；无方法门槛 |
| M-comparisons-057 | shots=5; comparison=M2-R1; measure=ba | Dreyer2023 | 80 | 0.975 | 1.2681 | 1.2238e-06 | — | 检出正向差异；无方法门槛 |
| M-comparisons-058 | shots=5; comparison=M2-R1; measure=ba | Cho2017 | 52 | 0.62 | 0.77577 | 0.011094 | — | 检出正向差异；无方法门槛 |
| M-comparisons-059 | shots=5; comparison=M2-R1; measure=ba | Dreyer+Cho | 132 | 0.855 | 1.0741 | 2.3117e-07 | — | 检出正向差异；无方法门槛 |
| M-comparisons-060 | shots=5; comparison=M2-R1; measure=ba | pooled | 235 | 0.5 | 0.57034 | 0.00016719 | — | 检出正向差异；无方法门槛 |
| M-comparisons-061 | shots=5; comparison=M2-R2; measure=ba | PhysionetMI | 103 | -0.075758 | -0.20589 | 0.85737 | — | 未检出差异；无方法门槛 |
| M-comparisons-062 | shots=5; comparison=M2-R2; measure=ba | Dreyer2023 | 80 | 0 | -0.13416 | 0.34299 | — | 未检出差异；无方法门槛 |
| M-comparisons-063 | shots=5; comparison=M2-R2; measure=ba | Cho2017 | 52 | -0.083333 | 0.037821 | 0.49422 | — | 未检出差异；无方法门槛 |
| M-comparisons-064 | shots=5; comparison=M2-R2; measure=ba | Dreyer+Cho | 132 | 0 | -0.06641 | 0.38085 | — | 未检出差异；无方法门槛 |
| M-comparisons-065 | shots=5; comparison=M2-R2; measure=ba | pooled | 235 | 0 | -0.12754 | 0.69804 | — | 未检出差异；无方法门槛 |
| M-comparisons-066 | shots=5; comparison=M2-M1; measure=ba | PhysionetMI | 103 | 0 | 0.12395 | 0.35048 | — | 未检出差异；无方法门槛 |
| M-comparisons-067 | shots=5; comparison=M2-M1; measure=ba | Dreyer2023 | 80 | 0.25 | 0.11063 | 0.012989 | — | 检出正向差异；无方法门槛 |
| M-comparisons-068 | shots=5; comparison=M2-M1; measure=ba | Cho2017 | 52 | 0 | -0.041026 | 0.67493 | — | 未检出差异；无方法门槛 |
| M-comparisons-069 | shots=5; comparison=M2-M1; measure=ba | Dreyer+Cho | 132 | 0.083333 | 0.050888 | 0.093578 | — | 未检出差异；无方法门槛 |
| M-comparisons-070 | shots=5; comparison=M2-M1; measure=ba | pooled | 235 | 0 | 0.082911 | 0.13936 | — | 未检出差异；无方法门槛 |
| M-comparisons-071 | shots=10; comparison=M1-R0; measure=ba | PhysionetMI | 103 | 0 | 0.029952 | 0.39859 | — | 未检出差异；无方法门槛 |
| M-comparisons-072 | shots=10; comparison=M1-R0; measure=ba | Dreyer2023 | 80 | 1.5 | 1.5235 | 5.6677e-10 | — | 检出正向差异；无方法门槛 |
| M-comparisons-073 | shots=10; comparison=M1-R0; measure=ba | Cho2017 | 52 | 1.3 | 1.6596 | 3.0754e-05 | — | 检出正向差异；无方法门槛 |
| M-comparisons-074 | shots=10; comparison=M1-R0; measure=ba | Dreyer+Cho | 132 | 1.45 | 1.5771 | 2.493e-13 | — | 检出正向差异；无方法门槛 |
| M-comparisons-075 | shots=10; comparison=M1-R0; measure=ba | pooled | 235 | 0.90909 | 0.899 | 3.2202e-09 | — | 检出正向差异；无方法门槛 |
| M-comparisons-076 | shots=10; comparison=M1-R1; measure=ba | PhysionetMI | 103 | 0.068182 | -0.097649 | 0.48311 | — | 未检出差异；无方法门槛 |
| M-comparisons-077 | shots=10; comparison=M1-R1; measure=ba | Dreyer2023 | 80 | 1.1333 | 1.2315 | 6.9551e-08 | — | 检出正向差异；无方法门槛 |
| M-comparisons-078 | shots=10; comparison=M1-R1; measure=ba | Cho2017 | 52 | 0.2 | 0.8266 | 0.029564 | — | 检出正向差异；无方法门槛 |
| M-comparisons-079 | shots=10; comparison=M1-R1; measure=ba | Dreyer+Cho | 132 | 1.0583 | 1.072 | 9.0493e-08 | — | 检出正向差异；无方法门槛 |
| M-comparisons-080 | shots=10; comparison=M1-R1; measure=ba | pooled | 235 | 0.51667 | 0.55935 | 3.3522e-05 | — | 检出正向差异；无方法门槛 |
| M-comparisons-081 | shots=10; comparison=M1-R2; measure=ba | PhysionetMI | 103 | 0 | -0.18912 | 0.73666 | — | 未检出差异；无方法门槛 |
| M-comparisons-082 | shots=10; comparison=M1-R2; measure=ba | Dreyer2023 | 80 | 0 | 0.092565 | 0.3273 | — | 未检出差异；无方法门槛 |
| M-comparisons-083 | shots=10; comparison=M1-R2; measure=ba | Cho2017 | 52 | 0 | -0.010897 | 0.61311 | — | 未检出差异；无方法门槛 |
| M-comparisons-084 | shots=10; comparison=M1-R2; measure=ba | Dreyer+Cho | 132 | 0 | 0.051807 | 0.47445 | — | 未检出差异；无方法门槛 |
| M-comparisons-085 | shots=10; comparison=M1-R2; measure=ba | pooled | 235 | 0 | -0.053789 | 0.64489 | — | 未检出差异；无方法门槛 |
| M-comparisons-086 | shots=10; comparison=M2-R0; measure=ba | PhysionetMI | 103 | 0 | -0.11373 | 0.57872 | — | 未检出差异；无方法门槛 |
| M-comparisons-087 | shots=10; comparison=M2-R0; measure=ba | Dreyer2023 | 80 | 1.6667 | 1.5477 | 2.6221e-10 | — | 检出正向差异；无方法门槛 |
| M-comparisons-088 | shots=10; comparison=M2-R0; measure=ba | Cho2017 | 52 | 1.4 | 1.7724 | 3.4327e-06 | — | 检出正向差异；无方法门槛 |
| M-comparisons-089 | shots=10; comparison=M2-R0; measure=ba | Dreyer+Cho | 132 | 1.6667 | 1.6363 | 1.0185e-14 | — | 检出正向差异；无方法门槛 |
| M-comparisons-090 | shots=10; comparison=M2-R0; measure=ba | pooled | 235 | 0.90909 | 0.86924 | 2.6769e-09 | — | 检出正向差异；无方法门槛 |
| M-comparisons-091 | shots=10; comparison=M2-R1; measure=ba | PhysionetMI | 103 | -0.015152 | -0.24133 | 0.75948 | — | 未检出差异；无方法门槛 |
| M-comparisons-092 | shots=10; comparison=M2-R1; measure=ba | Dreyer2023 | 80 | 1.175 | 1.2558 | 7.7556e-08 | — | 检出正向差异；无方法门槛 |
| M-comparisons-093 | shots=10; comparison=M2-R1; measure=ba | Cho2017 | 52 | 0.62 | 0.93942 | 0.0050465 | — | 检出正向差异；无方法门槛 |
| M-comparisons-094 | shots=10; comparison=M2-R1; measure=ba | Dreyer+Cho | 132 | 1.1167 | 1.1311 | 8.0849e-09 | — | 检出正向差异；无方法门槛 |
| M-comparisons-095 | shots=10; comparison=M2-R1; measure=ba | pooled | 235 | 0.57576 | 0.52959 | 6.5275e-05 | — | 检出正向差异；无方法门槛 |
| M-comparisons-096 | shots=10; comparison=M2-R2; measure=ba | PhysionetMI | 103 | -0.075758 | -0.3328 | 0.89075 | — | 未检出差异；无方法门槛 |
| M-comparisons-097 | shots=10; comparison=M2-R2; measure=ba | Dreyer2023 | 80 | 0.16667 | 0.11681 | 0.16232 | — | 未检出差异；无方法门槛 |
| M-comparisons-098 | shots=10; comparison=M2-R2; measure=ba | Cho2017 | 52 | 0 | 0.10192 | 0.37561 | — | 未检出差异；无方法门槛 |
| M-comparisons-099 | shots=10; comparison=M2-R2; measure=ba | Dreyer+Cho | 132 | 0 | 0.11095 | 0.20415 | — | 未检出差异；无方法门槛 |
| M-comparisons-100 | shots=10; comparison=M2-R2; measure=ba | pooled | 235 | 0 | -0.083548 | 0.64028 | — | 未检出差异；无方法门槛 |
| M-comparisons-101 | shots=10; comparison=M2-M1; measure=ba | PhysionetMI | 103 | 0 | -0.14368 | 0.75255 | — | 未检出差异；无方法门槛 |
| M-comparisons-102 | shots=10; comparison=M2-M1; measure=ba | Dreyer2023 | 80 | 0.083333 | 0.024246 | 0.18977 | — | 未检出差异；无方法门槛 |
| M-comparisons-103 | shots=10; comparison=M2-M1; measure=ba | Cho2017 | 52 | 0 | 0.11282 | 0.30184 | — | 未检出差异；无方法门槛 |
| M-comparisons-104 | shots=10; comparison=M2-M1; measure=ba | Dreyer+Cho | 132 | 0 | 0.059139 | 0.17464 | — | 未检出差异；无方法门槛 |
| M-comparisons-105 | shots=10; comparison=M2-M1; measure=ba | pooled | 235 | 0 | -0.029758 | 0.359 | — | 未检出差异；无方法门槛 |
| M-comparisons-106 | shots=20; comparison=M1-R0; measure=ba | PhysionetMI | 103 | 0.30769 | 0.39901 | 0.087807 | — | 未检出差异；无方法门槛 |
| M-comparisons-107 | shots=20; comparison=M1-R0; measure=ba | Dreyer2023 | 80 | 0.91667 | 1.1748 | 1.2077e-07 | — | 检出正向差异；无方法门槛 |
| M-comparisons-108 | shots=20; comparison=M1-R0; measure=ba | Cho2017 | 52 | 1 | 1.4346 | 0.00012355 | — | 检出正向差异；无方法门槛 |
| M-comparisons-109 | shots=20; comparison=M1-R0; measure=ba | Dreyer+Cho | 132 | 1 | 1.2772 | 1.3036e-10 | — | 检出正向差异；无方法门槛 |
| M-comparisons-110 | shots=20; comparison=M1-R0; measure=ba | pooled | 235 | 0.83333 | 0.89227 | 4.5314e-08 | — | 检出正向差异；无方法门槛 |
| M-comparisons-111 | shots=20; comparison=M1-R1; measure=ba | PhysionetMI | 103 | 0 | -0.064052 | 0.63581 | — | 未检出差异；无方法门槛 |
| M-comparisons-112 | shots=20; comparison=M1-R1; measure=ba | Dreyer2023 | 80 | 1.0583 | 1.0474 | 4.467e-06 | — | 检出正向差异；无方法门槛 |
| M-comparisons-113 | shots=20; comparison=M1-R1; measure=ba | Cho2017 | 52 | 0.57 | 0.79474 | 0.018503 | — | 检出正向差异；无方法门槛 |
| M-comparisons-114 | shots=20; comparison=M1-R1; measure=ba | Dreyer+Cho | 132 | 0.95833 | 0.9479 | 1.2284e-06 | — | 检出正向差异；无方法门槛 |
| M-comparisons-115 | shots=20; comparison=M1-R1; measure=ba | pooled | 235 | 0.46667 | 0.50436 | 0.00054322 | — | 检出正向差异；无方法门槛 |
| M-comparisons-116 | shots=20; comparison=M1-R2; measure=ba | PhysionetMI | 103 | 0 | 0.15514 | 0.41138 | — | 未检出差异；无方法门槛 |
| M-comparisons-117 | shots=20; comparison=M1-R2; measure=ba | Dreyer2023 | 80 | 0 | 0.33215 | 0.27615 | — | 未检出差异；无方法门槛 |
| M-comparisons-118 | shots=20; comparison=M1-R2; measure=ba | Cho2017 | 52 | -0.1 | -0.16859 | 0.82027 | — | 未检出差异；无方法门槛 |
| M-comparisons-119 | shots=20; comparison=M1-R2; measure=ba | Dreyer+Cho | 132 | 0 | 0.13489 | 0.53184 | — | 未检出差异；无方法门槛 |
| M-comparisons-120 | shots=20; comparison=M1-R2; measure=ba | pooled | 235 | 0 | 0.14376 | 0.45145 | — | 未检出差异；无方法门槛 |
| M-comparisons-121 | shots=20; comparison=M2-R0; measure=ba | PhysionetMI | 103 | 0.83333 | 0.36639 | 0.10162 | — | 未检出差异；无方法门槛 |
| M-comparisons-122 | shots=20; comparison=M2-R0; measure=ba | Dreyer2023 | 80 | 1 | 1.2143 | 5.9643e-08 | — | 检出正向差异；无方法门槛 |
| M-comparisons-123 | shots=20; comparison=M2-R0; measure=ba | Cho2017 | 52 | 1.1 | 1.3064 | 0.00038794 | — | 检出正向差异；无方法门槛 |
| M-comparisons-124 | shots=20; comparison=M2-R0; measure=ba | Dreyer+Cho | 132 | 1 | 1.2506 | 2.7271e-10 | — | 检出正向差异；无方法门槛 |
| M-comparisons-125 | shots=20; comparison=M2-R0; measure=ba | pooled | 235 | 0.83333 | 0.86305 | 3.5114e-08 | — | 检出正向差异；无方法门槛 |
| M-comparisons-126 | shots=20; comparison=M2-R1; measure=ba | PhysionetMI | 103 | 0 | -0.096674 | 0.6415 | — | 未检出差异；无方法门槛 |
| M-comparisons-127 | shots=20; comparison=M2-R1; measure=ba | Dreyer2023 | 80 | 0.68333 | 1.087 | 1.2529e-06 | — | 检出正向差异；无方法门槛 |
| M-comparisons-128 | shots=20; comparison=M2-R1; measure=ba | Cho2017 | 52 | 0.32 | 0.66654 | 0.052943 | — | 未检出差异；无方法门槛 |
| M-comparisons-129 | shots=20; comparison=M2-R1; measure=ba | Dreyer+Cho | 132 | 0.56167 | 0.92134 | 2.1929e-06 | — | 检出正向差异；无方法门槛 |
| M-comparisons-130 | shots=20; comparison=M2-R1; measure=ba | pooled | 235 | 0.33333 | 0.47515 | 0.000681 | — | 检出正向差异；无方法门槛 |
| M-comparisons-131 | shots=20; comparison=M2-R2; measure=ba | PhysionetMI | 103 | 0 | 0.12251 | 0.50723 | — | 未检出差异；无方法门槛 |
| M-comparisons-132 | shots=20; comparison=M2-R2; measure=ba | Dreyer2023 | 80 | -0.083333 | 0.37166 | 0.14473 | — | 未检出差异；无方法门槛 |
| M-comparisons-133 | shots=20; comparison=M2-R2; measure=ba | Cho2017 | 52 | -0.4 | -0.29679 | 0.84854 | — | 未检出差异；无方法门槛 |
| M-comparisons-134 | shots=20; comparison=M2-R2; measure=ba | Dreyer+Cho | 132 | -0.16667 | 0.10833 | 0.48055 | — | 未检出差异；无方法门槛 |
| M-comparisons-135 | shots=20; comparison=M2-R2; measure=ba | pooled | 235 | 0 | 0.11455 | 0.46773 | — | 未检出差异；无方法门槛 |
| M-comparisons-136 | shots=20; comparison=M2-M1; measure=ba | PhysionetMI | 103 | 0 | -0.032623 | 0.49439 | — | 未检出差异；无方法门槛 |
| M-comparisons-137 | shots=20; comparison=M2-M1; measure=ba | Dreyer2023 | 80 | 0 | 0.039511 | 0.31739 | — | 未检出差异；无方法门槛 |
| M-comparisons-138 | shots=20; comparison=M2-M1; measure=ba | Cho2017 | 52 | 0 | -0.12821 | 0.78389 | — | 未检出差异；无方法门槛 |
| M-comparisons-139 | shots=20; comparison=M2-M1; measure=ba | Dreyer+Cho | 132 | 0 | -0.026559 | 0.60992 | — | 未检出差异；无方法门槛 |
| M-comparisons-140 | shots=20; comparison=M2-M1; measure=ba | pooled | 235 | 0 | -0.029217 | 0.62251 | — | 未检出差异；无方法门槛 |
| M-comparisons-141 | shots=40; comparison=M1-R0; measure=ba | Dreyer2023 | 80 | 1.1667 | 1.1989 | 5.0653e-09 | — | 检出正向差异；无方法门槛 |
| M-comparisons-142 | shots=40; comparison=M1-R0; measure=ba | Cho2017 | 52 | 0.4 | 1.0731 | 0.0032462 | — | 检出正向差异；无方法门槛 |
| M-comparisons-143 | shots=40; comparison=M1-R0; measure=ba | Dreyer+Cho | 132 | 1 | 1.1493 | 3.8136e-10 | — | 检出正向差异；无方法门槛 |
| M-comparisons-144 | shots=40; comparison=M1-R0; measure=ba | pooled | 132 | 1 | 1.1493 | 3.8136e-10 | — | 检出正向差异；无方法门槛 |
| M-comparisons-145 | shots=40; comparison=M1-R1; measure=ba | Dreyer2023 | 80 | 0.84167 | 0.71746 | 1.4485e-05 | — | 检出正向差异；无方法门槛 |
| M-comparisons-146 | shots=40; comparison=M1-R1; measure=ba | Cho2017 | 52 | 0.29 | 0.74026 | 0.032396 | — | 检出正向差异；无方法门槛 |
| M-comparisons-147 | shots=40; comparison=M1-R1; measure=ba | Dreyer+Cho | 132 | 0.63 | 0.72644 | 8.4275e-06 | — | 检出正向差异；无方法门槛 |
| M-comparisons-148 | shots=40; comparison=M1-R1; measure=ba | pooled | 132 | 0.63 | 0.72644 | 8.4275e-06 | — | 检出正向差异；无方法门槛 |
| M-comparisons-149 | shots=40; comparison=M1-R2; measure=ba | Dreyer2023 | 80 | 0.16667 | 0.28477 | 0.10179 | — | 未检出差异；无方法门槛 |
| M-comparisons-150 | shots=40; comparison=M1-R2; measure=ba | Cho2017 | 52 | 0 | -0.10705 | 0.69066 | — | 未检出差异；无方法门槛 |
| M-comparisons-151 | shots=40; comparison=M1-R2; measure=ba | Dreyer+Cho | 132 | 0 | 0.13042 | 0.25414 | — | 未检出差异；无方法门槛 |
| M-comparisons-152 | shots=40; comparison=M1-R2; measure=ba | pooled | 132 | 0 | 0.13042 | 0.25414 | — | 未检出差异；无方法门槛 |
| M-comparisons-153 | shots=40; comparison=M2-R0; measure=ba | Dreyer2023 | 80 | 1.1667 | 1.1936 | 1.39e-08 | — | 检出正向差异；无方法门槛 |
| M-comparisons-154 | shots=40; comparison=M2-R0; measure=ba | Cho2017 | 52 | 0.6 | 0.95769 | 0.00065334 | — | 检出正向差异；无方法门槛 |
| M-comparisons-155 | shots=40; comparison=M2-R0; measure=ba | Dreyer+Cho | 132 | 1 | 1.1006 | 8.3857e-11 | — | 检出正向差异；无方法门槛 |
| M-comparisons-156 | shots=40; comparison=M2-R0; measure=ba | pooled | 132 | 1 | 1.1006 | 8.3857e-11 | — | 检出正向差异；无方法门槛 |
| M-comparisons-157 | shots=40; comparison=M2-R1; measure=ba | Dreyer2023 | 80 | 0.85833 | 0.71211 | 1.2077e-05 | — | 检出正向差异；无方法门槛 |
| M-comparisons-158 | shots=40; comparison=M2-R1; measure=ba | Cho2017 | 52 | 0.72 | 0.62487 | 0.023191 | — | 检出正向差异；无方法门槛 |
| M-comparisons-159 | shots=40; comparison=M2-R1; measure=ba | Dreyer+Cho | 132 | 0.815 | 0.67774 | 5.1312e-06 | — | 检出正向差异；无方法门槛 |
| M-comparisons-160 | shots=40; comparison=M2-R1; measure=ba | pooled | 132 | 0.815 | 0.67774 | 5.1312e-06 | — | 检出正向差异；无方法门槛 |
| M-comparisons-161 | shots=40; comparison=M2-R2; measure=ba | Dreyer2023 | 80 | 0 | 0.27942 | 0.082242 | — | 未检出差异；无方法门槛 |
| M-comparisons-162 | shots=40; comparison=M2-R2; measure=ba | Cho2017 | 52 | -0.16667 | -0.22244 | 0.78279 | — | 未检出差异；无方法门槛 |
| M-comparisons-163 | shots=40; comparison=M2-R2; measure=ba | Dreyer+Cho | 132 | 0 | 0.081718 | 0.32299 | — | 未检出差异；无方法门槛 |
| M-comparisons-164 | shots=40; comparison=M2-R2; measure=ba | pooled | 132 | 0 | 0.081718 | 0.32299 | — | 未检出差异；无方法门槛 |
| M-comparisons-165 | shots=40; comparison=M2-M1; measure=ba | Dreyer2023 | 80 | 0 | -0.005352 | 0.72716 | — | 未检出差异；无方法门槛 |
| M-comparisons-166 | shots=40; comparison=M2-M1; measure=ba | Cho2017 | 52 | -0.083333 | -0.11538 | 0.6621 | — | 未检出差异；无方法门槛 |
| M-comparisons-167 | shots=40; comparison=M2-M1; measure=ba | Dreyer+Cho | 132 | 0 | -0.048698 | 0.75521 | — | 未检出差异；无方法门槛 |
| M-comparisons-168 | shots=40; comparison=M2-M1; measure=ba | pooled | 132 | 0 | -0.048698 | 0.75521 | — | 未检出差异；无方法门槛 |
| M-comparisons-169 | shots=5; comparison=M1-R1; measure=adapt_gain | PhysionetMI | 103 | -0.098485 | -0.026033 | 0.65255 | — | 未检出差异；无方法门槛 |
| M-comparisons-170 | shots=5; comparison=M1-R1; measure=adapt_gain | Dreyer2023 | 80 | 0.075 | 0.24402 | 0.22218 | — | 未检出差异；无方法门槛 |
| M-comparisons-171 | shots=5; comparison=M1-R1; measure=adapt_gain | Cho2017 | 52 | 0.6 | 0.77718 | 0.014744 | — | 检出正向差异；无方法门槛 |
| M-comparisons-172 | shots=5; comparison=M1-R1; measure=adapt_gain | Dreyer+Cho | 132 | 0.20833 | 0.45405 | 0.019458 | — | 检出正向差异；无方法门槛 |
| M-comparisons-173 | shots=5; comparison=M1-R1; measure=adapt_gain | pooled | 235 | 0 | 0.24363 | 0.088092 | — | 未检出差异；无方法门槛 |
| M-comparisons-174 | shots=5; comparison=M2-R1; measure=adapt_gain | PhysionetMI | 103 | 0 | 0.094863 | 0.40754 | — | 未检出差异；无方法门槛 |
| M-comparisons-175 | shots=5; comparison=M2-R1; measure=adapt_gain | Dreyer2023 | 80 | 0.21667 | 0.36604 | 0.066952 | — | 未检出差异；无方法门槛 |
| M-comparisons-176 | shots=5; comparison=M2-R1; measure=adapt_gain | Cho2017 | 52 | 0.53 | 0.49449 | 0.06666 | — | 未检出差异；无方法门槛 |
| M-comparisons-177 | shots=5; comparison=M2-R1; measure=adapt_gain | Dreyer+Cho | 132 | 0.35 | 0.41664 | 0.017085 | — | 检出正向差异；无方法门槛 |
| M-comparisons-178 | shots=5; comparison=M2-R1; measure=adapt_gain | pooled | 235 | 0.016667 | 0.27561 | 0.039279 | — | 检出正向差异；无方法门槛 |
| M-comparisons-179 | shots=10; comparison=M1-R1; measure=adapt_gain | PhysionetMI | 103 | 0.10606 | 0.075581 | 0.40342 | — | 未检出差异；无方法门槛 |
| M-comparisons-180 | shots=10; comparison=M1-R1; measure=adapt_gain | Dreyer2023 | 80 | 0.091667 | 0.31809 | 0.065408 | — | 未检出差异；无方法门槛 |
| M-comparisons-181 | shots=10; comparison=M1-R1; measure=adapt_gain | Cho2017 | 52 | 0.55 | 0.78699 | 0.031065 | — | 检出正向差异；无方法门槛 |
| M-comparisons-182 | shots=10; comparison=M1-R1; measure=adapt_gain | Dreyer+Cho | 132 | 0.2 | 0.50281 | 0.0065923 | — | 检出正向差异；无方法门槛 |
| M-comparisons-183 | shots=10; comparison=M1-R1; measure=adapt_gain | pooled | 235 | 0.16667 | 0.31556 | 0.020524 | — | 检出正向差异；无方法门槛 |
| M-comparisons-184 | shots=10; comparison=M2-R1; measure=adapt_gain | PhysionetMI | 103 | 0 | -0.071159 | 0.7359 | — | 未检出差异；无方法门槛 |
| M-comparisons-185 | shots=10; comparison=M2-R1; measure=adapt_gain | Dreyer2023 | 80 | 0.23333 | 0.35372 | 0.023346 | — | 检出正向差异；无方法门槛 |
| M-comparisons-186 | shots=10; comparison=M2-R1; measure=adapt_gain | Cho2017 | 52 | 0.6 | 0.65814 | 0.040213 | — | 检出正向差异；无方法门槛 |
| M-comparisons-187 | shots=10; comparison=M2-R1; measure=adapt_gain | Dreyer+Cho | 132 | 0.36667 | 0.47365 | 0.0035759 | — | 检出正向差异；无方法门槛 |
| M-comparisons-188 | shots=10; comparison=M2-R1; measure=adapt_gain | pooled | 235 | 0.16667 | 0.23486 | 0.06045 | — | 未检出差异；无方法门槛 |
| M-comparisons-189 | shots=20; comparison=M1-R1; measure=adapt_gain | PhysionetMI | 103 | -0.075758 | 0.10918 | 0.54316 | — | 未检出差异；无方法门槛 |
| M-comparisons-190 | shots=20; comparison=M1-R1; measure=adapt_gain | Dreyer2023 | 80 | -0.083333 | 0.13402 | 0.46465 | — | 未检出差异；无方法门槛 |
| M-comparisons-191 | shots=20; comparison=M1-R1; measure=adapt_gain | Cho2017 | 52 | 0.38 | 0.75513 | 0.0344 | — | 检出正向差异；无方法门槛 |
| M-comparisons-192 | shots=20; comparison=M1-R1; measure=adapt_gain | Dreyer+Cho | 132 | 0.05 | 0.3787 | 0.092858 | — | 未检出差异；无方法门槛 |
| M-comparisons-193 | shots=20; comparison=M1-R1; measure=adapt_gain | pooled | 235 | 0 | 0.26057 | 0.18817 | — | 未检出差异；无方法门槛 |
| M-comparisons-194 | shots=20; comparison=M2-R1; measure=adapt_gain | PhysionetMI | 103 | -0.075758 | 0.0735 | 0.60969 | — | 未检出差异；无方法门槛 |
| M-comparisons-195 | shots=20; comparison=M2-R1; measure=adapt_gain | Dreyer2023 | 80 | -0.016667 | 0.18491 | 0.20097 | — | 未检出差异；无方法门槛 |
| M-comparisons-196 | shots=20; comparison=M2-R1; measure=adapt_gain | Cho2017 | 52 | 0.01 | 0.38526 | 0.24102 | — | 未检出差异；无方法门槛 |
| M-comparisons-197 | shots=20; comparison=M2-R1; measure=adapt_gain | Dreyer+Cho | 132 | 0 | 0.26384 | 0.13764 | — | 未检出差异；无方法门槛 |
| M-comparisons-198 | shots=20; comparison=M2-R1; measure=adapt_gain | pooled | 235 | 0 | 0.18041 | 0.29154 | — | 未检出差异；无方法门槛 |
| M-comparisons-199 | shots=40; comparison=M1-R1; measure=adapt_gain | Dreyer2023 | 80 | -0.28333 | -0.19597 | 0.84878 | — | 未检出差异；无方法门槛 |
| M-comparisons-200 | shots=40; comparison=M1-R1; measure=adapt_gain | Cho2017 | 52 | 0.09 | 0.70064 | 0.11972 | — | 未检出差异；无方法门槛 |
| M-comparisons-201 | shots=40; comparison=M1-R1; measure=adapt_gain | Dreyer+Cho | 132 | -0.018333 | 0.15724 | 0.53933 | — | 未检出差异；无方法门槛 |
| M-comparisons-202 | shots=40; comparison=M1-R1; measure=adapt_gain | pooled | 132 | -0.018333 | 0.15724 | 0.53933 | — | 未检出差异；无方法门槛 |
| M-comparisons-203 | shots=40; comparison=M2-R1; measure=adapt_gain | Dreyer2023 | 80 | -0.15833 | -0.18994 | 0.86083 | — | 未检出差异；无方法门槛 |
| M-comparisons-204 | shots=40; comparison=M2-R1; measure=adapt_gain | Cho2017 | 52 | 0.35 | 0.34359 | 0.21967 | — | 未检出差异；无方法门槛 |
| M-comparisons-205 | shots=40; comparison=M2-R1; measure=adapt_gain | Dreyer+Cho | 132 | 0.03 | 0.020241 | 0.56659 | — | 未检出差异；无方法门槛 |
| M-comparisons-206 | shots=40; comparison=M2-R1; measure=adapt_gain | pooled | 132 | 0.03 | 0.020241 | 0.56659 | — | 未检出差异；无方法门槛 |

### W · 群体模型比较

G−指定群体对照。按库原始p描述；B3行是历史并列，不是独立新实验。所有主门槛以第2节为准。 [完整源表](../reports/stage_w/report/population_comparisons.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| W-population_comparisons-001 | method=m_film; control=B0 | Cho2017 | 52 | -4.1 | -4.4917 | 1 | — | 未检出差异；无方法门槛 |
| W-population_comparisons-002 | method=m_film; control=B0 | Dreyer2023 | 80 | 4.25 | 4.3534 | 1.2461e-12 | — | 检出正向差异；无方法门槛 |
| W-population_comparisons-003 | method=m_film; control=B0 | PhysionetMI | 103 | 7.4242 | 7.4813 | 4.2467e-12 | — | 检出正向差异；无方法门槛 |
| W-population_comparisons-004 | method=m_film; control=B3 | Cho2017 | 52 | 0.41667 | 0.50705 | 0.031959 | — | 检出正向差异；无方法门槛 |
| W-population_comparisons-005 | method=m_film; control=B3 | Dreyer2023 | 80 | 0.25 | 0.089188 | 0.27513 | — | 未检出差异；无方法门槛 |
| W-population_comparisons-006 | method=m_film; control=B3 | PhysionetMI | 103 | 0.075758 | 0.34237 | 0.13914 | — | 未检出差异；无方法门槛 |
| W-population_comparisons-007 | method=m_lora; control=B0 | Cho2017 | 52 | -0.083333 | -0.063462 | 0.57928 | — | 未检出差异；无方法门槛 |
| W-population_comparisons-008 | method=m_lora; control=B0 | Dreyer2023 | 80 | 9.0833 | 9.1791 | 3.9106e-15 | — | 检出正向差异；无方法门槛 |
| W-population_comparisons-009 | method=m_lora; control=B0 | PhysionetMI | 103 | 7.7692 | 9.1445 | 4.2353e-16 | — | 检出正向差异；无方法门槛 |
| W-population_comparisons-010 | method=m_lora; control=B3 | Cho2017 | 52 | 0.2 | 0.67436 | 0.02961 | — | 检出正向差异；无方法门槛 |
| W-population_comparisons-011 | method=m_lora; control=B3 | Dreyer2023 | 80 | 0.5 | 0.39224 | 0.0015548 | — | 检出正向差异；无方法门槛 |
| W-population_comparisons-012 | method=m_lora; control=B3 | PhysionetMI | 103 | 0 | 0.068459 | 0.31596 | — | 未检出差异；无方法门槛 |
| W-population_comparisons-013 | method=m_lora; control=R2 | Cho2017 | 52 | -1 | -1.4353 | 0.99761 | — | 未检出差异；无方法门槛 |
| W-population_comparisons-014 | method=m_lora; control=R2 | Dreyer2023 | 80 | -1.6667 | -1.5973 | 1 | — | 未检出差异；无方法门槛 |
| W-population_comparisons-015 | method=m_lora; control=R2 | PhysionetMI | 103 | 0 | -0.3651 | 0.88316 | — | 未检出差异；无方法门槛 |

### W · 本人/交换逐库并列

本人−交换。按库原始p描述；B3行是历史并列，不是独立新实验。所有主门槛以第2节为准。 [完整源表](../reports/stage_w/report/diagnostic1_side_by_side.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| W-diagnostic1_side_by_side-001 | start=B3; variant=film_offset | Cho2017 | 52 | 3.22 | 5.7617 | 2.2074e-08 | — | 检出正向差异；无方法门槛 |
| W-diagnostic1_side_by_side-002 | start=B3; variant=film_offset | Dreyer2023 | 80 | 5.5167 | 5.9064 | 4.7398e-15 | — | 检出正向差异；无方法门槛 |
| W-diagnostic1_side_by_side-003 | start=B3; variant=film_offset | PhysionetMI | 103 | 4.4242 | 4.2991 | 1.5785e-11 | — | 检出正向差异；无方法门槛 |
| W-diagnostic1_side_by_side-004 | start=B3; variant=lora8 | Cho2017 | 52 | 4.51 | 6.3329 | 2.2688e-09 | — | 检出正向差异；无方法门槛 |
| W-diagnostic1_side_by_side-005 | start=B3; variant=lora8 | Dreyer2023 | 80 | 4.7417 | 5.307 | 4.9221e-15 | — | 检出正向差异；无方法门槛 |
| W-diagnostic1_side_by_side-006 | start=B3; variant=lora8 | PhysionetMI | 103 | 2.2615 | 2.2369 | 2.3864e-07 | — | 检出正向差异；无方法门槛 |
| W-diagnostic1_side_by_side-007 | start=G; variant=film_offset | Cho2017 | 52 | 3.2 | 6.0153 | 6.1169e-09 | — | 检出正向差异；无方法门槛 |
| W-diagnostic1_side_by_side-008 | start=G; variant=film_offset | Dreyer2023 | 80 | 5.6615 | 5.731 | 5.7204e-15 | — | 检出正向差异；无方法门槛 |
| W-diagnostic1_side_by_side-009 | start=G; variant=film_offset | Lee2019_MI | 54 | 2.5 | 3.9056 | 7.5754e-05 | — | 检出正向差异；无方法门槛 |
| W-diagnostic1_side_by_side-010 | start=G; variant=film_offset | PhysionetMI | 103 | 4.5455 | 4.6221 | 4.2005e-12 | — | 检出正向差异；无方法门槛 |
| W-diagnostic1_side_by_side-011 | start=G; variant=lora8 | Cho2017 | 52 | 4.3 | 6.4279 | 5.2315e-08 | — | 检出正向差异；无方法门槛 |
| W-diagnostic1_side_by_side-012 | start=G; variant=lora8 | Dreyer2023 | 80 | 4.8917 | 5.2762 | 3.9237e-15 | — | 检出正向差异；无方法门槛 |
| W-diagnostic1_side_by_side-013 | start=G; variant=lora8 | Lee2019_MI | 54 | 2.58 | 3.7804 | 7.4282e-07 | — | 检出正向差异；无方法门槛 |
| W-diagnostic1_side_by_side-014 | start=G; variant=lora8 | PhysionetMI | 103 | 1.8712 | 2.554 | 5.7361e-08 | — | 检出正向差异；无方法门槛 |

### W · 近邻逐库并列

近邻−随机。按库原始p描述；B3行是历史并列，不是独立新实验。所有主门槛以第2节为准。 [完整源表](../reports/stage_w/report/neighbours_side_by_side.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| W-neighbours_side_by_side-001 | start=B3; variant=film_offset; source=rest; k=1 | Cho2017 | 52 | 0.6 | 1.0803 | 0.033247 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-002 | start=B3; variant=film_offset; source=rest; k=1 | Dreyer2023 | 80 | 0.058333 | 0.32351 | 0.23889 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-003 | start=B3; variant=film_offset; source=rest; k=1 | PhysionetMI | 103 | 0.70455 | -0.046984 | 0.25475 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-004 | start=B3; variant=film_offset; source=rest; k=3 | Cho2017 | 52 | 0.58 | 0.98083 | 0.030637 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-005 | start=B3; variant=film_offset; source=rest; k=3 | Dreyer2023 | 80 | 0.091667 | 0.16953 | 0.11901 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-006 | start=B3; variant=film_offset; source=rest; k=3 | PhysionetMI | 103 | 0.21538 | -0.015746 | 0.34226 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-007 | start=B3; variant=film_offset; source=task; k=1 | Cho2017 | 52 | 0.82 | 0.79436 | 0.044281 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-008 | start=B3; variant=film_offset; source=task; k=1 | Dreyer2023 | 80 | 1.1583 | 0.68529 | 0.026771 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-009 | start=B3; variant=film_offset; source=task; k=1 | PhysionetMI | 103 | 1.0076 | 0.58063 | 0.014104 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-010 | start=B3; variant=film_offset; source=task; k=3 | Cho2017 | 52 | 0.28 | 0.45583 | 0.15712 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-011 | start=B3; variant=film_offset; source=task; k=3 | Dreyer2023 | 80 | 0.775 | 0.64338 | 0.0040244 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-012 | start=B3; variant=film_offset; source=task; k=3 | PhysionetMI | 103 | -0.33077 | -0.12414 | 0.66676 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-013 | start=B3; variant=lora8; source=rest; k=1 | Cho2017 | 52 | 1.05 | 0.54827 | 0.06444 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-014 | start=B3; variant=lora8; source=rest; k=1 | Dreyer2023 | 80 | 0.43333 | 0.37648 | 0.16452 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-015 | start=B3; variant=lora8; source=rest; k=1 | PhysionetMI | 103 | -1.2121 | -1.7134 | 0.99526 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-016 | start=B3; variant=lora8; source=rest; k=3 | Cho2017 | 52 | 0.34 | 0.54904 | 0.047607 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-017 | start=B3; variant=lora8; source=rest; k=3 | Dreyer2023 | 80 | -0.4 | -0.10741 | 0.71348 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-018 | start=B3; variant=lora8; source=rest; k=3 | PhysionetMI | 103 | -0.37879 | -0.36436 | 0.96657 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-019 | start=B3; variant=lora8; source=task; k=1 | Cho2017 | 52 | 0.42 | 0.22712 | 0.1264 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-020 | start=B3; variant=lora8; source=task; k=1 | Dreyer2023 | 80 | 0.525 | 0.4545 | 0.10663 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-021 | start=B3; variant=lora8; source=task; k=1 | PhysionetMI | 103 | -0.11364 | -0.25502 | 0.65107 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-022 | start=B3; variant=lora8; source=task; k=3 | Cho2017 | 52 | 0.20833 | 0.016987 | 0.30145 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-023 | start=B3; variant=lora8; source=task; k=3 | Dreyer2023 | 80 | 0.45833 | 0.39485 | 0.039143 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-024 | start=B3; variant=lora8; source=task; k=3 | PhysionetMI | 103 | 0.030303 | 0.092621 | 0.37621 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-025 | start=G; variant=film_offset; source=rest; k=1 | Cho2017 | 52 | 0.465 | 0.8409 | 0.027327 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-026 | start=G; variant=film_offset; source=rest; k=1 | Dreyer2023 | 80 | 0.33333 | 0.29107 | 0.21787 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-027 | start=G; variant=film_offset; source=rest; k=1 | Lee2019_MI | 54 | 0.5 | 0.087037 | 0.31331 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-028 | start=G; variant=film_offset; source=rest; k=1 | PhysionetMI | 103 | 0.40909 | -0.0051712 | 0.2261 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-029 | start=G; variant=film_offset; source=rest; k=3 | Cho2017 | 52 | 0.61 | 0.64859 | 0.04601 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-030 | start=G; variant=film_offset; source=rest; k=3 | Dreyer2023 | 80 | 0.44167 | 0.34692 | 0.074348 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-031 | start=G; variant=film_offset; source=rest; k=3 | PhysionetMI | 103 | 0.40909 | 0.15307 | 0.24019 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-032 | start=G; variant=film_offset; source=task; k=1 | Cho2017 | 52 | 0.72 | 0.51013 | 0.034497 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-033 | start=G; variant=film_offset; source=task; k=1 | Dreyer2023 | 80 | 0.66667 | 0.6229 | 0.024729 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-034 | start=G; variant=film_offset; source=task; k=1 | PhysionetMI | 103 | 1.0769 | 0.30719 | 0.075027 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-035 | start=G; variant=film_offset; source=task; k=3 | Cho2017 | 52 | 0.31 | 0.14859 | 0.30966 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-036 | start=G; variant=film_offset; source=task; k=3 | Dreyer2023 | 80 | 0.79167 | 0.84606 | 0.0012175 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-037 | start=G; variant=film_offset; source=task; k=3 | PhysionetMI | 103 | 0.40909 | 0.22653 | 0.16223 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-038 | start=G; variant=lora8; source=rest; k=1 | Cho2017 | 52 | 0.45 | 0.35096 | 0.11823 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-039 | start=G; variant=lora8; source=rest; k=1 | Dreyer2023 | 80 | 0.33333 | 0.32454 | 0.15527 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-040 | start=G; variant=lora8; source=rest; k=1 | Lee2019_MI | 54 | 1.19 | 1.073 | 0.0055295 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-041 | start=G; variant=lora8; source=rest; k=1 | PhysionetMI | 103 | -0.98485 | -1.3909 | 0.99461 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-042 | start=G; variant=lora8; source=rest; k=3 | Cho2017 | 52 | 0.15 | 0.48635 | 0.12835 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-043 | start=G; variant=lora8; source=rest; k=3 | Dreyer2023 | 80 | -0.21667 | 0.040029 | 0.51754 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-044 | start=G; variant=lora8; source=rest; k=3 | PhysionetMI | 103 | 0.12121 | -0.050295 | 0.49867 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-045 | start=G; variant=lora8; source=task; k=1 | Cho2017 | 52 | 0.45 | 0.31314 | 0.16833 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-046 | start=G; variant=lora8; source=task; k=1 | Dreyer2023 | 80 | 0.78333 | 0.53144 | 0.037978 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-047 | start=G; variant=lora8; source=task; k=1 | PhysionetMI | 103 | 0.2197 | 0.0082988 | 0.40001 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-048 | start=G; variant=lora8; source=task; k=3 | Cho2017 | 52 | 0.24 | 0.067756 | 0.16359 | — | 未检出差异；无方法门槛 |
| W-neighbours_side_by_side-049 | start=G; variant=lora8; source=task; k=3 | Dreyer2023 | 80 | 0.29167 | 0.39068 | 0.035857 | — | 检出正向差异；无方法门槛 |
| W-neighbours_side_by_side-050 | start=G; variant=lora8; source=task; k=3 | PhysionetMI | 103 | -0.068182 | 0.061276 | 0.43985 | — | 未检出差异；无方法门槛 |

### W · 少样本逐库并列

少样本适配−对应起点。按库原始p描述；B3行是历史并列，不是独立新实验。所有主门槛以第2节为准。 [完整源表](../reports/stage_w/report/few_shot_side_by_side.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| W-few_shot_side_by_side-001 | start=B3; shots=5 | Cho2017 | 52 | 0.1 | 0.19295 | 0.39217 | — | 未检出差异；无方法门槛 |
| W-few_shot_side_by_side-002 | start=B3; shots=5 | Dreyer2023 | 80 | -0.5 | -0.17866 | 0.52826 | — | 未检出差异；无方法门槛 |
| W-few_shot_side_by_side-003 | start=B3; shots=5 | PhysionetMI | 103 | 0.76923 | 0.77728 | 0.0037918 | — | 检出正向差异；无方法门槛 |
| W-few_shot_side_by_side-004 | start=B3; shots=10 | Cho2017 | 52 | 0.65 | 1.2692 | 0.02527 | — | 检出正向差异；无方法门槛 |
| W-few_shot_side_by_side-005 | start=B3; shots=10 | Dreyer2023 | 80 | 0 | -0.093175 | 0.47367 | — | 未检出差异；无方法门槛 |
| W-few_shot_side_by_side-006 | start=B3; shots=10 | PhysionetMI | 103 | 0.76923 | 0.93005 | 0.0019442 | — | 检出正向差异；无方法门槛 |
| W-few_shot_side_by_side-007 | start=B3; shots=20 | Cho2017 | 52 | 0.2 | 1.3635 | 0.040659 | — | 检出正向差异；无方法门槛 |
| W-few_shot_side_by_side-008 | start=B3; shots=20 | Dreyer2023 | 80 | 0.5 | 0.40981 | 0.040045 | — | 检出正向差异；无方法门槛 |
| W-few_shot_side_by_side-009 | start=B3; shots=20 | PhysionetMI | 103 | 1.2121 | 1.4977 | 4.2253e-05 | — | 检出正向差异；无方法门槛 |
| W-few_shot_side_by_side-010 | start=B3; shots=40 | Cho2017 | 52 | 1 | 2.3603 | 0.0023702 | — | 检出正向差异；无方法门槛 |
| W-few_shot_side_by_side-011 | start=B3; shots=40 | Dreyer2023 | 80 | 1.5833 | 1.6399 | 7.5412e-07 | — | 检出正向差异；无方法门槛 |
| W-few_shot_side_by_side-012 | start=G; shots=5 | Cho2017 | 52 | -0.6 | -0.39872 | 0.87772 | — | 未检出差异；无方法门槛 |
| W-few_shot_side_by_side-013 | start=G; shots=5 | Dreyer2023 | 80 | -0.33333 | -0.46458 | 0.89582 | — | 未检出差异；无方法门槛 |
| W-few_shot_side_by_side-014 | start=G; shots=5 | Lee2019_MI | 54 | -1.1 | -1.2222 | 0.99996 | — | 未检出差异；无方法门槛 |
| W-few_shot_side_by_side-015 | start=G; shots=5 | PhysionetMI | 103 | 0.75758 | 0.81185 | 0.0030023 | — | 检出正向差异；无方法门槛 |
| W-few_shot_side_by_side-016 | start=G; shots=10 | Cho2017 | 52 | 0.4 | 1.0474 | 0.025724 | — | 检出正向差异；无方法门槛 |
| W-few_shot_side_by_side-017 | start=G; shots=10 | Dreyer2023 | 80 | 0.25287 | -0.047845 | 0.37811 | — | 未检出差异；无方法门槛 |
| W-few_shot_side_by_side-018 | start=G; shots=10 | Lee2019_MI | 54 | -0.6 | -0.7037 | 0.99478 | — | 未检出差异；无方法门槛 |
| W-few_shot_side_by_side-019 | start=G; shots=10 | PhysionetMI | 103 | 0.83333 | 0.82899 | 0.0045323 | — | 检出正向差异；无方法门槛 |
| W-few_shot_side_by_side-020 | start=G; shots=20 | Cho2017 | 52 | 0.2 | 1.2987 | 0.092284 | — | 未检出差异；无方法门槛 |
| W-few_shot_side_by_side-021 | start=G; shots=20 | Dreyer2023 | 80 | 0.33333 | 0.25 | 0.1384 | — | 未检出差异；无方法门槛 |
| W-few_shot_side_by_side-022 | start=G; shots=20 | Lee2019_MI | 54 | 0 | -0.33704 | 0.75759 | — | 未检出差异；无方法门槛 |
| W-few_shot_side_by_side-023 | start=G; shots=20 | PhysionetMI | 103 | 1.5152 | 1.5731 | 1.4012e-06 | — | 检出正向差异；无方法门槛 |
| W-few_shot_side_by_side-024 | start=G; shots=40 | Cho2017 | 52 | 1.2 | 2.1519 | 0.0032738 | — | 检出正向差异；无方法门槛 |
| W-few_shot_side_by_side-025 | start=G; shots=40 | Dreyer2023 | 80 | 1.1667 | 1.4075 | 1.3792e-06 | — | 检出正向差异；无方法门槛 |

### W-pooled · 合并近邻主/次描述

近邻−随机。核验后生成的合并描述；k=3为次要，k=1校正门槛见第2节。 [完整源表](../reports/stage_w/neighbour_audit/pooled_descriptive.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| W-pooled-pooled_descriptive-001 | start=B3; variant=film_offset; source=rest; k=1 | 起步组合并 | 235 | 0.56061 | 0.32857 | 0.033745 | — | 检出正向差异；无方法门槛 |
| W-pooled-pooled_descriptive-002 | start=B3; variant=film_offset; source=rest; k=3 | 起步组合并 | 235 | 0.25758 | 0.26785 | 0.037109 | — | 检出正向差异；无方法门槛 |
| W-pooled-pooled_descriptive-003 | start=B3; variant=film_offset; source=task; k=1 | 起步组合并 | 235 | 0.91667 | 0.66355 | 0.00025849 | — | 检出正向差异；无方法门槛 |
| W-pooled-pooled_descriptive-004 | start=B3; variant=film_offset; source=task; k=3 | 起步组合并 | 235 | 0.25 | 0.26548 | 0.055303 | — | 未检出差异；无方法门槛 |
| W-pooled-pooled_descriptive-005 | start=B3; variant=lora8; source=rest; k=1 | 起步组合并 | 235 | 0.016667 | -0.5015 | 0.73119 | — | 未检出差异；无方法门槛 |
| W-pooled-pooled_descriptive-006 | start=B3; variant=lora8; source=rest; k=3 | 起步组合并 | 235 | -0.1 | -0.074773 | 0.78018 | — | 未检出差异；无方法门槛 |
| W-pooled-pooled_descriptive-007 | start=B3; variant=lora8; source=task; k=1 | 起步组合并 | 235 | 0.18939 | 0.093201 | 0.17997 | — | 未检出差异；无方法门槛 |
| W-pooled-pooled_descriptive-008 | start=B3; variant=lora8; source=task; k=3 | 起步组合并 | 235 | 0.21667 | 0.17877 | 0.071475 | — | 未检出差异；无方法门槛 |
| W-pooled-pooled_descriptive-009 | start=G; variant=film_offset; source=rest; k=1 | 起步组合并 | 235 | 0.44 | 0.28289 | 0.034872 | — | 检出正向差异；无方法门槛 |
| W-pooled-pooled_descriptive-010 | start=G; variant=film_offset; source=rest; k=3 | 起步组合并 | 235 | 0.42308 | 0.32871 | 0.019445 | — | 检出正向差异；无方法门槛 |
| W-pooled-pooled_descriptive-011 | start=G; variant=film_offset; source=task; k=1 | 起步组合并 | 235 | 0.76667 | 0.45957 | 0.0021332 | — | 检出正向差异；无方法门槛 |
| W-pooled-pooled_descriptive-012 | start=G; variant=film_offset; source=task; k=3 | 起步组合并 | 235 | 0.5 | 0.42019 | 0.0050867 | — | 检出正向差异；无方法门槛 |
| W-pooled-pooled_descriptive-013 | start=G; variant=lora8; source=rest; k=1 | 起步组合并 | 235 | 0 | -0.42148 | 0.79471 | — | 未检出差异；无方法门槛 |
| W-pooled-pooled_descriptive-014 | start=G; variant=lora8; source=rest; k=3 | 起步组合并 | 235 | 0 | 0.099199 | 0.28994 | — | 未检出差异；无方法门槛 |
| W-pooled-pooled_descriptive-015 | start=G; variant=lora8; source=task; k=1 | 起步组合并 | 235 | 0.56667 | 0.25384 | 0.053663 | — | 未检出差异；无方法门槛 |
| W-pooled-pooled_descriptive-016 | start=G; variant=lora8; source=task; k=3 | 起步组合并 | 235 | 0.16154 | 0.17485 | 0.061664 | — | 未检出差异；无方法门槛 |

### WF-BNCI · 群体适配与任务头/显式线性探针

G−冻结超参数的头对照。BNCI9人，session2二分类；原始p描述，不与Compass四分类LOSO数值直接相减。 [完整源表](../reports\stage_w_followup\report\bnci_population_comparisons.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| WF-BNCI-bnci_population_comparisons-001 | method=G_film_offset; control=B0 | BNCI2014_001 | 9 | 2.5 | 2.5309 | 0.054561 | — | 未检出差异；无方法门槛 |
| WF-BNCI-bnci_population_comparisons-002 | method=G_film_offset; control=B0_linear | BNCI2014_001 | 9 | 1.3889 | 1.821 | 0.18671 | — | 未检出差异；无方法门槛 |
| WF-BNCI-bnci_population_comparisons-003 | method=G_lora8; control=B0 | BNCI2014_001 | 9 | 1.9444 | 2.9321 | 0.061282 | — | 未检出差异；无方法门槛 |
| WF-BNCI-bnci_population_comparisons-004 | method=G_lora8; control=B0_linear | BNCI2014_001 | 9 | 2.2222 | 2.2222 | 0.17141 | — | 未检出差异；无方法门槛 |

### WF-R2-own_minus_population · 三个起点逐库敏感性描述

本人−对应群体起点。G主参照，R2敏感性；B3/G复用历史样本，描述p不构成独立验证或新通过标准。 [完整源表](../reports/stage_w_followup/report/population_strength_by_dataset.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| WF-R2-own_minus_population-population_strength_by_dataset-001 | start=B3; variant=film_offset; population_family=m_film | Cho2017 | 52 | 3.4667 | 5.8122 | 1.6233e-06 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-002 | start=B3; variant=film_offset; population_family=m_film | Dreyer2023 | 80 | 2.5 | 3.1737 | 4.7996e-12 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-003 | start=B3; variant=film_offset; population_family=m_film | PhysionetMI | 103 | 1.8182 | 1.951 | 0.00017831 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-004 | start=B3; variant=lora8; population_family=m_lora | Cho2017 | 52 | 3.4 | 5.5974 | 9.2701e-08 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-005 | start=B3; variant=lora8; population_family=m_lora | Dreyer2023 | 80 | 3 | 3.5652 | 3.367e-13 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-006 | start=B3; variant=lora8; population_family=m_lora | PhysionetMI | 103 | 0.90909 | 1.4232 | 0.0012843 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-007 | start=G; variant=film_offset; population_family=m_film | Cho2017 | 52 | 2.3 | 5.3237 | 1.0832e-05 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-008 | start=G; variant=film_offset; population_family=m_film | Dreyer2023 | 80 | 2.5 | 3.0236 | 6.9305e-11 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-009 | start=G; variant=film_offset; population_family=m_film | PhysionetMI | 103 | 1.1364 | 1.8229 | 0.00064083 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-010 | start=G; variant=lora8; population_family=m_lora | Cho2017 | 52 | 3.5 | 5.1295 | 1.1756e-06 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-011 | start=G; variant=lora8; population_family=m_lora | Dreyer2023 | 80 | 3.2917 | 3.5798 | 1.4898e-13 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-012 | start=G; variant=lora8; population_family=m_lora | PhysionetMI | 103 | 0.98485 | 1.64 | 0.00072949 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-013 | start=R2; variant=film_offset; population_family=m_lora | Cho2017 | 52 | 1.0667 | 3.0769 | 0.0072902 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-014 | start=R2; variant=film_offset; population_family=m_lora | Dreyer2023 | 80 | 1.8333 | 1.7403 | 2.0444e-06 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-015 | start=R2; variant=film_offset; population_family=m_lora | PhysionetMI | 103 | -0.37879 | 0.097755 | 0.54921 | — | 未检出差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-016 | start=R2; variant=lora8; population_family=m_lora | Cho2017 | 52 | 2 | 4.1897 | 1.7927e-05 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-017 | start=R2; variant=lora8; population_family=m_lora | Dreyer2023 | 80 | 2 | 2.2601 | 1.2842e-09 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_population-population_strength_by_dataset-018 | start=R2; variant=lora8; population_family=m_lora | PhysionetMI | 103 | 0.75758 | 1.0055 | 0.023709 | — | 检出正向差异；无方法门槛 |

### WF-R2-own_minus_swap · 三个起点逐库敏感性描述

本人−交换。G主参照，R2敏感性；B3/G复用历史样本，描述p不构成独立验证或新通过标准。 [完整源表](../reports/stage_w_followup/report/population_strength_by_dataset.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| WF-R2-own_minus_swap-population_strength_by_dataset-001 | start=B3; variant=film_offset; population_family=m_film | Cho2017 | 52 | 3.22 | 5.7617 | 2.2074e-08 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-002 | start=B3; variant=film_offset; population_family=m_film | Dreyer2023 | 80 | 5.5167 | 5.9064 | 4.7398e-15 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-003 | start=B3; variant=film_offset; population_family=m_film | PhysionetMI | 103 | 4.4242 | 4.2991 | 1.5785e-11 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-004 | start=B3; variant=lora8; population_family=m_lora | Cho2017 | 52 | 4.51 | 6.3329 | 2.2688e-09 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-005 | start=B3; variant=lora8; population_family=m_lora | Dreyer2023 | 80 | 4.7417 | 5.307 | 4.9221e-15 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-006 | start=B3; variant=lora8; population_family=m_lora | PhysionetMI | 103 | 2.2615 | 2.2369 | 2.3864e-07 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-007 | start=G; variant=film_offset; population_family=m_film | Cho2017 | 52 | 3.2 | 6.0153 | 6.1169e-09 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-008 | start=G; variant=film_offset; population_family=m_film | Dreyer2023 | 80 | 5.6615 | 5.731 | 5.7204e-15 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-009 | start=G; variant=film_offset; population_family=m_film | PhysionetMI | 103 | 4.5455 | 4.6221 | 4.2005e-12 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-010 | start=G; variant=lora8; population_family=m_lora | Cho2017 | 52 | 4.3 | 6.4279 | 5.2315e-08 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-011 | start=G; variant=lora8; population_family=m_lora | Dreyer2023 | 80 | 4.8917 | 5.2762 | 3.9237e-15 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-012 | start=G; variant=lora8; population_family=m_lora | PhysionetMI | 103 | 1.8712 | 2.554 | 5.7361e-08 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-013 | start=R2; variant=film_offset; population_family=m_lora | Cho2017 | 52 | 3.96 | 6.206 | 5.0703e-08 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-014 | start=R2; variant=film_offset; population_family=m_lora | Dreyer2023 | 80 | 4.5583 | 4.9083 | 1.483e-14 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-015 | start=R2; variant=film_offset; population_family=m_lora | PhysionetMI | 103 | 3.5152 | 3.4145 | 1.4242e-10 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-016 | start=R2; variant=lora8; population_family=m_lora | Cho2017 | 52 | 4.04 | 6.276 | 2.648e-09 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-017 | start=R2; variant=lora8; population_family=m_lora | Dreyer2023 | 80 | 4.1 | 4.6095 | 6.9072e-15 | — | 检出正向差异；无方法门槛 |
| WF-R2-own_minus_swap-population_strength_by_dataset-018 | start=R2; variant=lora8; population_family=m_lora | PhysionetMI | 103 | 2.0455 | 2.1575 | 2.2573e-06 | — | 检出正向差异；无方法门槛 |

### WF-few · 冻结设置的逐人均值描述

少样本适配−G。9人，不新增显著性门槛；从完整个人明细按所列条件聚合，结构化清单记录聚合规则，不伪装成单条源行。 [完整源表](../reports/stage_w_followup/report/bnci_few_shot_subjects.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| WF-few-bnci_few_shot_subjects-001 | shots=5 | BNCI2014_001 | 9 | -0.83333 | -0.64815 | — | — | 不适用（描述） |
| WF-few-bnci_few_shot_subjects-002 | shots=10 | BNCI2014_001 | 9 | 2.7778 | 2.6852 | — | — | 不适用（描述） |
| WF-few-bnci_few_shot_subjects-003 | shots=20 | BNCI2014_001 | 9 | 1.1111 | 0.67901 | — | — | 不适用（描述） |

### WF-own · 冻结设置的逐人均值描述

本人−G。9人，不新增显著性门槛；从完整个人明细按所列条件聚合，结构化清单记录聚合规则，不伪装成单条源行。 [完整源表](../reports/stage_w_followup/report/bnci_diagnostic1_subjects.csv)。

| ID | 条件 | 数据集 | N | 中位差pp | 均值差pp | 原始p | 校正p | 判定范围 |
|---|---|---|---|---|---|---|---|---|
| WF-own-bnci_diagnostic1_subjects-001 | variant=film_offset | BNCI2014_001 | 9 | 5.5556 | 5.1235 | — | — | 不适用（描述） |
| WF-own-bnci_diagnostic1_subjects-002 | variant=lora8 | BNCI2014_001 | 9 | 5 | 4.8148 | — | — | 不适用（描述） |

## 4. 平台判据与未执行/不可估项目

W 平台是描述性检查，不是模型有效性门槛：最后8个检查点损失极差≤max(1e−4, |均值|×1%)，训练和验证均满足才记平台。实际停止步数与所选步数分开报告。历史W四组各25条均未满足平台；单条结果如下。准确率轨迹没有记录，不能据CE判定其趋势。

| ID | 数据群 | 模型 | 折 | 种子 | 训练平台 | 验证平台 | 共同平台 |
|---|---|---|---|---|---|---|---|
| W-plateau-001 | starter | m_film | 0 | 11 | 否 | 否 | 未通过 |
| W-plateau-002 | starter | m_lora | 0 | 11 | 否 | 否 | 未通过 |
| W-plateau-003 | starter | m_film | 0 | 23 | 否 | 否 | 未通过 |
| W-plateau-004 | starter | m_lora | 0 | 23 | 否 | 否 | 未通过 |
| W-plateau-005 | starter | m_film | 0 | 37 | 否 | 否 | 未通过 |
| W-plateau-006 | starter | m_lora | 0 | 37 | 否 | 否 | 未通过 |
| W-plateau-007 | starter | m_film | 0 | 53 | 否 | 否 | 未通过 |
| W-plateau-008 | starter | m_lora | 0 | 53 | 否 | 否 | 未通过 |
| W-plateau-009 | starter | m_film | 0 | 71 | 否 | 否 | 未通过 |
| W-plateau-010 | starter | m_lora | 0 | 71 | 否 | 否 | 未通过 |
| W-plateau-011 | starter | m_film | 1 | 11 | 否 | 否 | 未通过 |
| W-plateau-012 | starter | m_lora | 1 | 11 | 否 | 否 | 未通过 |
| W-plateau-013 | starter | m_film | 1 | 23 | 否 | 否 | 未通过 |
| W-plateau-014 | starter | m_lora | 1 | 23 | 否 | 否 | 未通过 |
| W-plateau-015 | starter | m_film | 1 | 37 | 否 | 否 | 未通过 |
| W-plateau-016 | starter | m_lora | 1 | 37 | 否 | 否 | 未通过 |
| W-plateau-017 | starter | m_film | 1 | 53 | 否 | 否 | 未通过 |
| W-plateau-018 | starter | m_lora | 1 | 53 | 否 | 否 | 未通过 |
| W-plateau-019 | starter | m_film | 1 | 71 | 否 | 否 | 未通过 |
| W-plateau-020 | starter | m_lora | 1 | 71 | 否 | 否 | 未通过 |
| W-plateau-021 | starter | m_film | 2 | 11 | 否 | 否 | 未通过 |
| W-plateau-022 | starter | m_lora | 2 | 11 | 否 | 否 | 未通过 |
| W-plateau-023 | starter | m_film | 2 | 23 | 否 | 否 | 未通过 |
| W-plateau-024 | starter | m_lora | 2 | 23 | 否 | 否 | 未通过 |
| W-plateau-025 | starter | m_film | 2 | 37 | 否 | 否 | 未通过 |
| W-plateau-026 | starter | m_lora | 2 | 37 | 否 | 否 | 未通过 |
| W-plateau-027 | starter | m_film | 2 | 53 | 否 | 否 | 未通过 |
| W-plateau-028 | starter | m_lora | 2 | 53 | 否 | 否 | 未通过 |
| W-plateau-029 | starter | m_film | 2 | 71 | 否 | 否 | 未通过 |
| W-plateau-030 | starter | m_lora | 2 | 71 | 否 | 否 | 未通过 |
| W-plateau-031 | starter | m_film | 3 | 11 | 否 | 否 | 未通过 |
| W-plateau-032 | starter | m_lora | 3 | 11 | 否 | 否 | 未通过 |
| W-plateau-033 | starter | m_film | 3 | 23 | 否 | 否 | 未通过 |
| W-plateau-034 | starter | m_lora | 3 | 23 | 否 | 否 | 未通过 |
| W-plateau-035 | starter | m_film | 3 | 37 | 否 | 否 | 未通过 |
| W-plateau-036 | starter | m_lora | 3 | 37 | 否 | 否 | 未通过 |
| W-plateau-037 | starter | m_film | 3 | 53 | 否 | 否 | 未通过 |
| W-plateau-038 | starter | m_lora | 3 | 53 | 否 | 否 | 未通过 |
| W-plateau-039 | starter | m_film | 3 | 71 | 否 | 否 | 未通过 |
| W-plateau-040 | starter | m_lora | 3 | 71 | 否 | 否 | 未通过 |
| W-plateau-041 | starter | m_film | 4 | 11 | 否 | 否 | 未通过 |
| W-plateau-042 | starter | m_lora | 4 | 11 | 否 | 否 | 未通过 |
| W-plateau-043 | starter | m_film | 4 | 23 | 否 | 否 | 未通过 |
| W-plateau-044 | starter | m_lora | 4 | 23 | 否 | 否 | 未通过 |
| W-plateau-045 | starter | m_film | 4 | 37 | 否 | 否 | 未通过 |
| W-plateau-046 | starter | m_lora | 4 | 37 | 否 | 否 | 未通过 |
| W-plateau-047 | starter | m_film | 4 | 53 | 否 | 否 | 未通过 |
| W-plateau-048 | starter | m_lora | 4 | 53 | 否 | 否 | 未通过 |
| W-plateau-049 | starter | m_film | 4 | 71 | 否 | 否 | 未通过 |
| W-plateau-050 | starter | m_lora | 4 | 71 | 否 | 否 | 未通过 |
| W-plateau-051 | Lee2019_MI | m_film | 0 | 11 | 否 | 否 | 未通过 |
| W-plateau-052 | Lee2019_MI | m_lora | 0 | 11 | 否 | 否 | 未通过 |
| W-plateau-053 | Lee2019_MI | m_film | 0 | 23 | 否 | 否 | 未通过 |
| W-plateau-054 | Lee2019_MI | m_lora | 0 | 23 | 否 | 否 | 未通过 |
| W-plateau-055 | Lee2019_MI | m_film | 0 | 37 | 否 | 否 | 未通过 |
| W-plateau-056 | Lee2019_MI | m_lora | 0 | 37 | 否 | 否 | 未通过 |
| W-plateau-057 | Lee2019_MI | m_film | 0 | 53 | 否 | 否 | 未通过 |
| W-plateau-058 | Lee2019_MI | m_lora | 0 | 53 | 否 | 否 | 未通过 |
| W-plateau-059 | Lee2019_MI | m_film | 0 | 71 | 否 | 否 | 未通过 |
| W-plateau-060 | Lee2019_MI | m_lora | 0 | 71 | 否 | 否 | 未通过 |
| W-plateau-061 | Lee2019_MI | m_film | 1 | 11 | 否 | 否 | 未通过 |
| W-plateau-062 | Lee2019_MI | m_lora | 1 | 11 | 否 | 否 | 未通过 |
| W-plateau-063 | Lee2019_MI | m_film | 1 | 23 | 否 | 否 | 未通过 |
| W-plateau-064 | Lee2019_MI | m_lora | 1 | 23 | 否 | 否 | 未通过 |
| W-plateau-065 | Lee2019_MI | m_film | 1 | 37 | 否 | 否 | 未通过 |
| W-plateau-066 | Lee2019_MI | m_lora | 1 | 37 | 否 | 否 | 未通过 |
| W-plateau-067 | Lee2019_MI | m_film | 1 | 53 | 否 | 否 | 未通过 |
| W-plateau-068 | Lee2019_MI | m_lora | 1 | 53 | 否 | 否 | 未通过 |
| W-plateau-069 | Lee2019_MI | m_film | 1 | 71 | 否 | 否 | 未通过 |
| W-plateau-070 | Lee2019_MI | m_lora | 1 | 71 | 否 | 否 | 未通过 |
| W-plateau-071 | Lee2019_MI | m_film | 2 | 11 | 否 | 否 | 未通过 |
| W-plateau-072 | Lee2019_MI | m_lora | 2 | 11 | 否 | 否 | 未通过 |
| W-plateau-073 | Lee2019_MI | m_film | 2 | 23 | 否 | 否 | 未通过 |
| W-plateau-074 | Lee2019_MI | m_lora | 2 | 23 | 否 | 否 | 未通过 |
| W-plateau-075 | Lee2019_MI | m_film | 2 | 37 | 否 | 否 | 未通过 |
| W-plateau-076 | Lee2019_MI | m_lora | 2 | 37 | 否 | 否 | 未通过 |
| W-plateau-077 | Lee2019_MI | m_film | 2 | 53 | 否 | 否 | 未通过 |
| W-plateau-078 | Lee2019_MI | m_lora | 2 | 53 | 否 | 否 | 未通过 |
| W-plateau-079 | Lee2019_MI | m_film | 2 | 71 | 否 | 否 | 未通过 |
| W-plateau-080 | Lee2019_MI | m_lora | 2 | 71 | 否 | 否 | 未通过 |
| W-plateau-081 | Lee2019_MI | m_film | 3 | 11 | 否 | 否 | 未通过 |
| W-plateau-082 | Lee2019_MI | m_lora | 3 | 11 | 否 | 否 | 未通过 |
| W-plateau-083 | Lee2019_MI | m_film | 3 | 23 | 否 | 否 | 未通过 |
| W-plateau-084 | Lee2019_MI | m_lora | 3 | 23 | 否 | 否 | 未通过 |
| W-plateau-085 | Lee2019_MI | m_film | 3 | 37 | 否 | 否 | 未通过 |
| W-plateau-086 | Lee2019_MI | m_lora | 3 | 37 | 否 | 否 | 未通过 |
| W-plateau-087 | Lee2019_MI | m_film | 3 | 53 | 否 | 否 | 未通过 |
| W-plateau-088 | Lee2019_MI | m_lora | 3 | 53 | 否 | 否 | 未通过 |
| W-plateau-089 | Lee2019_MI | m_film | 3 | 71 | 否 | 否 | 未通过 |
| W-plateau-090 | Lee2019_MI | m_lora | 3 | 71 | 否 | 否 | 未通过 |
| W-plateau-091 | Lee2019_MI | m_film | 4 | 11 | 否 | 否 | 未通过 |
| W-plateau-092 | Lee2019_MI | m_lora | 4 | 11 | 否 | 否 | 未通过 |
| W-plateau-093 | Lee2019_MI | m_film | 4 | 23 | 否 | 否 | 未通过 |
| W-plateau-094 | Lee2019_MI | m_lora | 4 | 23 | 否 | 否 | 未通过 |
| W-plateau-095 | Lee2019_MI | m_film | 4 | 37 | 否 | 否 | 未通过 |
| W-plateau-096 | Lee2019_MI | m_lora | 4 | 37 | 否 | 否 | 未通过 |
| W-plateau-097 | Lee2019_MI | m_film | 4 | 53 | 否 | 否 | 未通过 |
| W-plateau-098 | Lee2019_MI | m_lora | 4 | 53 | 否 | 否 | 未通过 |
| W-plateau-099 | Lee2019_MI | m_film | 4 | 71 | 否 | 否 | 未通过 |
| W-plateau-100 | Lee2019_MI | m_lora | 4 | 71 | 否 | 否 | 未通过 |

BNCI补充运行同样记录平台；新增准确率日志不参与选择检查点。

| ID | 数据集 | 模型 | 折 | 种子 | 训练平台 | 验证平台 | 共同平台 |
|---|---|---|---|---|---|---|---|
| WF-plateau-001 | BNCI2014_001 | m_film | 0 | 11 | 否 | 否 | 未通过 |
| WF-plateau-002 | BNCI2014_001 | m_lora | 0 | 11 | 否 | 否 | 未通过 |
| WF-plateau-003 | BNCI2014_001 | m_film | 0 | 23 | 否 | 否 | 未通过 |
| WF-plateau-004 | BNCI2014_001 | m_lora | 0 | 23 | 否 | 否 | 未通过 |
| WF-plateau-005 | BNCI2014_001 | m_film | 0 | 37 | 否 | 否 | 未通过 |
| WF-plateau-006 | BNCI2014_001 | m_lora | 0 | 37 | 否 | 否 | 未通过 |
| WF-plateau-007 | BNCI2014_001 | m_film | 0 | 53 | 否 | 否 | 未通过 |
| WF-plateau-008 | BNCI2014_001 | m_lora | 0 | 53 | 否 | 否 | 未通过 |
| WF-plateau-009 | BNCI2014_001 | m_film | 0 | 71 | 否 | 否 | 未通过 |
| WF-plateau-010 | BNCI2014_001 | m_lora | 0 | 71 | 否 | 否 | 未通过 |
| WF-plateau-011 | BNCI2014_001 | m_film | 1 | 11 | 否 | 否 | 未通过 |
| WF-plateau-012 | BNCI2014_001 | m_lora | 1 | 11 | 否 | 否 | 未通过 |
| WF-plateau-013 | BNCI2014_001 | m_film | 1 | 23 | 否 | 否 | 未通过 |
| WF-plateau-014 | BNCI2014_001 | m_lora | 1 | 23 | 是 | 否 | 未通过 |
| WF-plateau-015 | BNCI2014_001 | m_film | 1 | 37 | 否 | 否 | 未通过 |
| WF-plateau-016 | BNCI2014_001 | m_lora | 1 | 37 | 否 | 否 | 未通过 |
| WF-plateau-017 | BNCI2014_001 | m_film | 1 | 53 | 否 | 否 | 未通过 |
| WF-plateau-018 | BNCI2014_001 | m_lora | 1 | 53 | 否 | 否 | 未通过 |
| WF-plateau-019 | BNCI2014_001 | m_film | 1 | 71 | 否 | 否 | 未通过 |
| WF-plateau-020 | BNCI2014_001 | m_lora | 1 | 71 | 否 | 否 | 未通过 |
| WF-plateau-021 | BNCI2014_001 | m_film | 2 | 11 | 否 | 否 | 未通过 |
| WF-plateau-022 | BNCI2014_001 | m_lora | 2 | 11 | 否 | 否 | 未通过 |
| WF-plateau-023 | BNCI2014_001 | m_film | 2 | 23 | 否 | 否 | 未通过 |
| WF-plateau-024 | BNCI2014_001 | m_lora | 2 | 23 | 否 | 否 | 未通过 |
| WF-plateau-025 | BNCI2014_001 | m_film | 2 | 37 | 否 | 否 | 未通过 |
| WF-plateau-026 | BNCI2014_001 | m_lora | 2 | 37 | 否 | 否 | 未通过 |
| WF-plateau-027 | BNCI2014_001 | m_film | 2 | 53 | 否 | 否 | 未通过 |
| WF-plateau-028 | BNCI2014_001 | m_lora | 2 | 53 | 否 | 否 | 未通过 |
| WF-plateau-029 | BNCI2014_001 | m_film | 2 | 71 | 否 | 否 | 未通过 |
| WF-plateau-030 | BNCI2014_001 | m_lora | 2 | 71 | 否 | 否 | 未通过 |
| WF-plateau-031 | BNCI2014_001 | m_film | 3 | 11 | 否 | 否 | 未通过 |
| WF-plateau-032 | BNCI2014_001 | m_lora | 3 | 11 | 否 | 否 | 未通过 |
| WF-plateau-033 | BNCI2014_001 | m_film | 3 | 23 | 否 | 否 | 未通过 |
| WF-plateau-034 | BNCI2014_001 | m_lora | 3 | 23 | 否 | 否 | 未通过 |
| WF-plateau-035 | BNCI2014_001 | m_film | 3 | 37 | 否 | 否 | 未通过 |
| WF-plateau-036 | BNCI2014_001 | m_lora | 3 | 37 | 否 | 否 | 未通过 |
| WF-plateau-037 | BNCI2014_001 | m_film | 3 | 53 | 否 | 否 | 未通过 |
| WF-plateau-038 | BNCI2014_001 | m_lora | 3 | 53 | 否 | 否 | 未通过 |
| WF-plateau-039 | BNCI2014_001 | m_film | 3 | 71 | 否 | 否 | 未通过 |
| WF-plateau-040 | BNCI2014_001 | m_lora | 3 | 71 | 否 | 否 | 未通过 |
| WF-plateau-041 | BNCI2014_001 | m_film | 4 | 11 | 否 | 否 | 未通过 |
| WF-plateau-042 | BNCI2014_001 | m_lora | 4 | 11 | 是 | 否 | 未通过 |
| WF-plateau-043 | BNCI2014_001 | m_film | 4 | 23 | 否 | 否 | 未通过 |
| WF-plateau-044 | BNCI2014_001 | m_lora | 4 | 23 | 否 | 否 | 未通过 |
| WF-plateau-045 | BNCI2014_001 | m_film | 4 | 37 | 否 | 否 | 未通过 |
| WF-plateau-046 | BNCI2014_001 | m_lora | 4 | 37 | 否 | 否 | 未通过 |
| WF-plateau-047 | BNCI2014_001 | m_film | 4 | 53 | 否 | 否 | 未通过 |
| WF-plateau-048 | BNCI2014_001 | m_lora | 4 | 53 | 否 | 否 | 未通过 |
| WF-plateau-049 | BNCI2014_001 | m_film | 4 | 71 | 否 | 否 | 未通过 |
| WF-plateau-050 | BNCI2014_001 | m_lora | 4 | 71 | 否 | 否 | 未通过 |

| ID | 原计划 / 假设 | 数据集 | 原判定规则或执行前提 | 状态与原因 |
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
## 6. 阶段 X 的结果前检验清单（2026-09-27，X1核心完成）

本节由 `docs/cross_model_tests.md` 维护，并由汇总脚本附在历史表后。规则见 [X协议](cross_model_protocol.md)，纠错依据见 [新旧任务对齐](cross_model_alignment.md)。REVE与LaBraM均已完成25次运行、全部结果与日志核验；扩展未执行不记未通过。

| ID / 家族 | 计划范围 | 固定判定或用途 | 当前状态 |
|---|---|---|---|
| X0-source | REVE/LaBraM×五库 | 别名核查；重合保留单列，unlisted不是逐记录排重 | 审计完成；REVE重合Dreyer/Cho/Lee，LaBraM重合PhysioNet |
| X0-structure | 两模型全层适配 | 固定官方源码、29份文件哈希、通道/参数算术 | 静态核查完成；两模型完整权重严格加载，Lee全通道核验仍待执行 |
| X-engineering-CPU | 两模型接口及现有代码回归 | 合成数据零适配、Q/K/V/输出梯度、冻结、重载、分批累积、断点恢复；原生三库形状 | 此前127项CPU回归通过；本次18项跨模型回归通过（含新增6项来源校验）；完整PT骨干独立合成检查通过；科学结果为0 |
| X-formal-entry | 两模型各25折/种子，共50完整运行，另两份共享缓存 | 正式执行并保存计时；完整群体/个人选择先于测试；逐模型预测复核 | 入口和合成流程检查通过；235人准备及两模型缓存完成；两模型各25/25及合计3,744,910条预测复算通过；50次配置/划分/来源/冻结与验证选择核验通过 |
| X0-resource-review | 历史本地Slurm回执 | 按作业ID去重、按GPU分配时间核算，不重复加同卡worker | 历史167.44卡时；本轮总计66.91卡时（REVE48.08、LaBraM18.83），含缓存及22秒失败；GPU墙钟4.52小时、峰值16卡 |
| X-G-trajectory | 全部B0/G候选 | 逐epoch验证CE/accuracy/BA；初训1,000/3,000候选按原增益排序；选定点按W续训/CE早停，平台另报 | 两模型各25个G均结束，0/50满足平台；两模型全部逐epoch、选择与耐心复核通过；49次CE耐心、1次步数上限停止；收敛后结论不可判定 |
| X-core-U | 每模型起步235人合并 | 原C2/W三槽位Holm，本人−交换中位≥2pp且p_Holm<.05；未执行族p=1占位、结果NA | 均通过：REVE N235、中位3.50pp、Holm p=1.38e-35；LaBraM N235、中位3.17pp、Holm p=1.19e-29；分库不新增主家族 |
| X-core-population | 两模型×三库 | G−B0效果量、原始描述性p | 两模型六格完成，均值均为正 |
| X-core-personal | 两模型×三库 | 本人−G效果量，与本人−交换分开 | 两模型六格完成，均值均为正 |
| X-core-few | 两模型×(Physio三个n+Dreyer四个n+Cho四个n) | 原少样本描述及全前半恢复分母，无新成功门槛 | 22格齐全；REVE Physio/Dreyer稳定、Cho不稳定；LaBraM Physio/Cho稳定、Dreyer不稳定 |
| X-film-U | 扩展1，每模型合并 | 补入同一原三槽位家族；本人−G另报 | 未执行，不重训已完成LoRA |
| X-neighbour-rest | 扩展2，每模型合并 | 原W：诊断1通过的适配族主分析，Holm含遗漏混合偏移p=1槽；按库描述 | 未执行 |
| X-neighbour-task | 扩展2，每模型合并 | 原C3：k1、两适配族Holm；无诊断1前置门槛，k3次要 | 未执行；撤销误加前置条件 |
| X-external-U | 扩展3，每模型×Lee/BNCI | 各库原三槽位诊断1，起步超参数冻结；不新建八项家族 | 未执行；BNCI交换最多八人 |
| X-external-rest | 扩展3，原外部静息项目 | 原资格和校正；BNCI选择价值不可辨识记NA | 未执行，不追加外部任务近邻 |
| X-M1-R2-n10 | 扩展4，每模型合并 | 用户指定M1−R2，更新数匹配、墙钟另报；原≥1pp/p_Holm<.05/下降>2pp≤10%，M2槽p=1 | 未执行；分库描述 |
| X-M1-R2-secondary | n5/n20、n0和适配增益分解 | 原次要报告，不替换n10主终点 | 未执行 |
| X-consistency | 每库群体、本人相对G、少样本 | 协议§6.2描述性一致标签，不是等效性检验 | 核心已判定：群体Physio/Dreyer一致正、Cho不一致；本人−G三库一致正；少样本仅Physio一致稳定，Dreyer/Cho不一致；外部待扩展 |

每模型×库完成即更新，无需等另一模型；合并主判定与分库描述分别标明。FiLM未做时用p=1保守占原槽，若预定扩展完成再列补齐版本，注明原未执行槽位。缺供体、预算未执行不造阴性结果；seen/unlisted限制随合并及分库结果保留。

初版新增跨模型6/12/8项主家族已撤销，历史主判定保持原样。资源和日期缺口见 [X0报告](stage_x0_report_2026-09-27.md)，本轮实现与核验见 [工程报告](cross_model_engineering_2026-09-27/README.md)。初次核心50项因REVE缓存失败在启动前取消，已替换依赖；有效核心仍50项。失败缓存22秒计入用量，LaBraM缓存保留；两模型完整诊断及报告均已完成，本阶段停止。提交回执见[记录](cross_model_engineering_2026-09-27/submission_receipt.json)。

直接全量的执行调整见[正式记录](cross_model_engineering_2026-09-27/formal_execution.md)；取消计时前置不改变训练预算、选择规则或统计判定。

REVE三库核验结果见[模型报告](cross_model_REVE_report_2026-09-27.md)。原主诊断通过不替代本人−G比较，不意味着G已充分收敛；seen标签与未执行槽位保留。

REVE附件核验：25次运行的逐epoch与验证选择独立复核通过；1,106份工件逐文件SHA256校验后保留至HPC home，本地核对731份非检查点原始工件。逐种子/单类前缀/分段显存与Slurm用量见模型报告。报告附件复核脚本为`scripts/cross_model_report_annex.py`，仅复算与核对，不训练或重选。

X1核心最终结果见[阶段报告](stage_x1_core_report_2026-09-27.md)、[LaBraM模型报告](cross_model_LaBraM_report_2026-09-27.md)及[三模型汇总](cross_model_summary.md)。两模型完整工件2,212文件、13.36GB已逐文件校验保存home；本地1,462份非检查点原始工件核验通过。全部核心于9月27日08:47结束，逐试次核验08:48完成，未舍弃核心组合、种子或n。扩展因阶段边界未执行，不是阴性或工期落后舍弃。

X1呈现补充（用户要求，未增加实验）：已补充CBraMod/REVE/LaBraM×三库本人−交换均值、中位、SD和原始p，九格方向均正；原合并诊断1三模型均通过，保留各自原Holm家族状态。与本人−G分列，不将正差值解释为全部收益均个人特异。见[跨模型汇总](cross_model_summary.md)。
## 群体训练量曲线及唯一数值重跑：最终冻结检验表（2026-09-27）

原科学规则v1.31，唯一重跑例外v1.34，结果登记v1.35；[原协议](population_budget_protocol.md)、[重跑前协议](population_budget_retry_protocol.md)。1×为现有G实际步数S，2×/4×为总2S/4S；个人r8与原网格只在验证被试选择，测试在选择冻结后读取。历史CBraMod优化器重启与群体参数化差异继续披露。

| ID | 预定规则 | 最终状态 |
|---|---|---|
| GB-G | 三模型三库三预算点G成绩、逐epoch与端点验证损失/准确率 | 全75轨迹完整；逐模型/库/点均25次运行，均到4×且未达联合平台 |
| GB-own-G | 同人五种子均值，完整235人中位；2×/4×六槽Holm | 三模型三点中位正，2×/4×均显著为正 |
| GB-own-swap | 原供体及10次抽样，全矩阵；中位≥2pp且原三槽Holm<.05 | 三模型三个预算点合并均通过 |
| GB-undertraining | M1≥M2≥M4且至少一处下降，M4<0.5pp；优先 | 三模型均未触发，不等于排除训练不足 |
| GB-stable | abs(M4−M2)<0.5pp，两点中位正且六槽Holm<.05 | CBraMod通过（0.400pp）；REVE/LaBraM不通过变化门槛 |
| GB-other | 其余如实描述；缺完整队列不缩小分母 | REVE/LaBraM描述性；CBraMod唯一重跑成功，已无缺失槽 |

| 模型 | 本人−G中位1×→2×→4×（pp） | 2× / 4×六槽Holm p | 本人−交换中位1×→2×→4×（pp） | 本人−交换各点三槽Holm p | 判读 |
| --- | --- | --- | --- | --- | --- |
| CBraMod | 2.600 → 1.600 → 2.000 | 5.69e-12 / 6.57e-13 | 3.533 → 3.400 → 3.233 | 4.73e-28 / 9.59e-26 / 3.52e-26 | 持续训练后稳定为正 |
| REVE | 2.424 → 1.833 → 1.000 | 3.02e-21 / 6.57e-13 | 3.500 → 3.100 → 2.240 | 1.38e-35 / 1.31e-32 / 1.23e-27 | 如实描述，不作强结论 |
| LaBraM | 2.424 → 2.576 → 1.970 | 5.62e-20 / 1.62e-20 | 3.174 → 2.867 → 2.939 | 1.19e-29 / 2.7e-29 / 2.51e-29 | 如实描述，不作强结论 |

CBraMod的2×到4×本人−G中位变化为0.400pp，小于0.5pp；两点中位均正，六槽Holm p分别为5.69e-12、6.57e-13，按原规则判为“在群体模型持续训练后，个人收益稳定为正”。REVE、LaBraM的变化分别为0.833pp、0.606pp，仍为“如实描述，不作强结论”。三个模型均未触发“个人收益主要反映群体模型训练不足”，这不等于排除训练不足。

三模型在三个预算点的本人−交换均通过原三槽位诊断门槛，支持既定供体程序下的个人参数匹配价值；不表示全部收益都是个人独有。最终75条成功群体轨迹均未达到联合平台，不能将CBraMod的预算稳定判读写成“已充分收敛后”的结论。

REVE/LaBraM原始成绩和原始p逐项保持不变；原失败CBraMod的两个p=1缺失槽替换为真实p后，按相同六槽家族重算，REVE 4×校正p由首批9.60e-13更新为6.57e-13。原三槽本人−交换检验按各预算点分别报告，未运行变体p=1；不改历史W/X1家族。

唯一重跑取消适配模型第8层激活的fp16往返、保留fp32，并关闭TF32；原群体梯度裁剪1.0不变，其余配置、数据、预算、选择和供体保持原协议。完整CBraMod队列包含24条原精度运行和1条上述例外。该运行1×历史对比56,706条预测中3条硬预测变化，G/本人BA最大差均为0，交换BA最大单人单种子差0.300pp。首批另外24条中已有1条G硬预测变化（单人单种子G BA差4.1667pp）继续保留；历史W/X1原报告不改。这些差异与数值措施均已披露，重跑成功不证明原故障根因。

本次GPU1912202_57于19:12:04完成，用时41分36秒、0.693卡时；CPU1912210于19:17:06完成，用时4分36秒、0.307核时、零GPU。群体新增9750步，总13000步，在4×上限停止。首批含原失败88.096卡时，本检查累计88.789卡时。原3,720份保留文件重新核验，新126份工件已保留home，本地120份非检查点工件全部SHA256通过；合并复算13,092,075条预测，只替换原索引57。

19项准备CPU检查全部通过。最终合并75条来源、原失败保留、端点、逐epoch日志、个人验证选择及供体均已核验；另从最终逐人×种子表独立复算中位数、Wilcoxon及六槽Holm，验证完整分母与36组验证端点覆盖。工程核验不增加科学检验家族。

首批74成功/1失败及缺失NA是当时状态，保存在[首批报告](population_budget_initial_report_2026-09-27.md)和[重跑前检验快照](population_budget_tests_before_retry.md)。[最终完整报告](population_budget_report_2026-09-27.md)提供模型×库均值±SD、全部个体及验证日志。[独立复核回执](../reports/population_budget/retry1/final_verification.json)。**本阶段停止，之后全部实验冻结。**
