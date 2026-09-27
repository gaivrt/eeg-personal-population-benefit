# 基础模型预训练语料与数据重合核查

初次核查：2026-09-25；X0追加核查：2026-09-27。原部分覆盖CBraMod、LaBraM与7库；文末X0部分增加REVE，覆盖本轮指定5库。结论针对论文对应的官方**预训练权重**，不适用于任意同架构模型或下游微调checkpoint。没有重新预训练，也未获得原作者逐记录训练清单。

## 结论及 7 库映射

**CBraMod 声明在 TUEG 上预训练，未发现与本项目 7 库的数据集级重合。LaBraM 的预训练清单明确包含 PhysioNet EEGMMIDB，与 PhysioNetMI 重合。**“未列入公开清单”只表示来源清单核查结果，不等于完成逐被试、逐记录哈希排重。[CBraMod 论文 §3.1](https://arxiv.org/abs/2412.07236)、[LaBraM 论文附录 D，第 17–18 页](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf)。

| 本项目数据集 | CBraMod 原版预训练 | LaBraM 原版预训练 | 身份核对与解释 |
|---|---|---|---|
| PhysioNetMI | 未列入 TUEG 来源 | **明确重合** | EEGMMIDB / EEG Motor Movement/Imagery Dataset / Schalk 2004 是同一来源；不能因 MOABB 类名不同而漏判。 |
| Dreyer2023（A+B+C） | 未列入 | 未列入公开清单 | 未见 Dreyer 或其 A/B/C 子组。 |
| Cho2017 | 未列入 | 未列入公开清单 | 未见 Cho / GigaDB 100295。 |
| Lee2019_MI | 未列入 | 未列入公开清单 | 未见 OpenBMI / Lee2019 / GigaDB 100542。 |
| Stieger2021 | 未列入 | 未列入公开清单 | 未见该纵向在线 BCI 数据 / Figshare 13123148。 |
| Yang2025（2C） | 未列入 | 未列入公开清单 | 按本项目 DOI 10.1038/s41597-025-04826-y 及 Figshare 22671172 核对；CBraMod 下游 SHU-MI 是 Ma 2022 的 25 人、32 通道、250 Hz 数据，不是本库。 |
| BNCI2014-001 | 未列入；作者做过下游评测 | 未列入公开清单 | 本库是 **BCI Competition IV-2a**，LaBraM 预训练列的是 **IV-1**，两者不同。 |

上表是将下列论文语料清单与[本项目数据身份](data_download_access.md)对照得到的判断。尤其注意：本项目没有用测试被试训练适配器，不足以证明基础模型预训练也未见过其信号。

## CBraMod 的预训练语料

公开主设定只有 **Temple University Hospital EEG Corpus（TUEG）**。论文列原始规模 69,652 记录、14,987 人、26,846 session、27,062 小时；清洗后为 1,109,545 个 30 秒样本，超过 9,000 小时。它们是原始语料规模与保留样本规模两个口径，不能混写。[论文 §3.1，第 6 页](https://arxiv.org/abs/2412.07236)。

仓库交叉检查：[作者仓库](https://github.com/wjq-learning/CBraMod)，固定 commit `b9e961003214326972c567eff390e75b0287e32a`。README 指向预训练权重；`pretrain_main.py` 接受本地数据目录，`datasets/pretraining_dataset.py` 从 LMDB 键读取样本。已检查代码没有提供可复核每一条预训练记录的 manifest，因此具体语料身份依据论文，不能从通用 loader 推出更强排重结论。[固定版本入口](https://github.com/wjq-learning/CBraMod/blob/b9e961003214326972c567eff390e75b0287e32a/pretrain_main.py)。

需要区分的情况：

- PhysioNet-MI 和 BCIC-IV-2a 被作者用于**下游评测**；后者见附录 E.9、表 15（第 27 页）。它们不因此成为该论文声明的预训练语料，但 BNCI 也不能称为“基础模型开发过程中从未使用的全新基准”。
- 附录 G 的 `LaBraM (ours)` 是 CBraMod 作者在其预训练语料上重新训练的对比模型，不能用于描述 LaBraM 官方原版权重的来源。
- `CBraMod (excluding TUAB)` 是另一个实验变体，不应默认认为下载的常规权重就是此版本。[论文附录 E.8、E.9、G](https://arxiv.org/abs/2412.07236)。

本项目已有权重配置指向作者 README 链接的 Hugging Face 仓库 `weighting666/CBraMod`，固定 revision `500543c7e30bda1b22bfd51a49301b238dee21fd`；已有本地下载回执记录 `pretrained_weights.pth` 为 19,775,842 字节，SHA256 为 `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`。该回执用于标识文件，不等于权重内部附带了逐记录来源证明。

## LaBraM 的完整公开预训练清单

论文附录 D 说明以下清单同时用于 neural tokenizer（VQNSP）训练与 LaBraM 预训练。按原论文的分组列出，SEED 系列在同一行展开；不能把“约 20 个数据集”的摘要数字当成另有未列出的精确 20 条。表中小时数为论文报告值。[原论文附录 D，第 17–18 页](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf)。

| 语料 | 小时 |
|---|---:|
| BCI Competition IV-1 | 8.21 |
| Emobrain | 4.94 |
| Grasp and Lift EEG Challenge | 11.72 |
| Inria BCI Challenge | 29.98 |
| **EEG Motor Movement/Imagery Dataset（EEGMMIDB）** | **47.30** |
| Raw EEG Data（Trujillo 2020） | 34.35 |
| Resting State EEG Data（Trujillo 2017） | 3.04 |
| SEED、SEED-IV、SEED-GER、SEED-FRA | 166.75 |
| Siena Scalp EEG Database | 30.47 |
| SPIS Resting State Dataset | 0.83 |
| Target Versus Non-Target（Brain Invaders） | 16.00 |
| TUAR | 92.22 |
| TUEP | 591.22 |
| TUSZ | 1,138.53 |
| TUSL | 20.59 |
| 作者自采 EEG（Jiang 2023/2021、Luo 2022、Li 2021、Tao & Lu 2020） | 342.23 |

论文总时长写作 **2,534.78 小时**；上述逐项报告值相加为 **2,538.38 小时**，相差 3.60 小时。本核查保留两种口径，未擅自改动任何条目，也不能在没有原始 manifest 的情况下解释差额。

仓库交叉检查：[作者仓库](https://github.com/935963004/LaBraM)，固定 commit `c431221e6cfd23dbfa9950e0180682fb322b0548`。README 的约 2,500 小时、约 20 库描述与论文量级相符；`run_vqnsp_training.py`、`run_labram_pretraining.py` 中的训练路径是示例占位符，未给出用于发布权重的完整文件清单。[固定版预训练脚本](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/run_labram_pretraining.py)、[tokenizer 脚本](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/run_vqnsp_training.py)。

EEGMMIDB 条目明确描述 109 人，并提及睁眼/闭眼基线及运动/想象记录；没有公开到 run 的纳入清单。因此 R04/R08/R12 或 R01/R02 **不能被声称已从 LaBraM 预训练中排除**，也不能断言本项目每条记录均确实参与过预训练。[PhysioNet 原始数据身份](https://physionet.org/content/eegmmidb/1.0.0/)、[LaBraM 附录 D](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf)。

## 对实验报告的要求

- 默认 CBraMod 的选择保持不变，使用官方预训练权重；本项目数据划分与预训练来源审计分别报告。
- 若采用 LaBraM 备选，保留 PhysioNet 但标记“预训练已见来源”，单列结果；不能以该库证明严格的预训练未见被试泛化。是否改动主分析范围需要在执行该阶段前明确，不在看到测试结果后改协议。
- BNCI 维持本项目外部测试隔离。披露其在 CBraMod 原论文中的下游基准使用史；本项目禁止使用基于 BNCI 微调的权重。
- 更换 checkpoint 后重新核对模型卡、训练语料与文件校验值；该架构的名称本身不足以继承本审计结论。

## 证据与未完成项

官方仓库的已读小型源码快照、URL、commit、字节数及 SHA256 见 [source_manifest.json](pretraining_audit_2026-09-25/source_manifest.json)。论文全文仅保存在本地 `.cache/pretraining_audit/`，不作为代码或数据提交；manifest 同时记录其原始 URL 和哈希，便于重取。OpenReview 的 CBraMod PDF 下载返回 HTTP 403，使用作者仓库链接的 arXiv 论文核验，第 6、27、28 页与 LaBraM 第 17–18 页均已查看。

这次工作记录公开来源与检查结论，尚未做原作者训练集级哈希排重、重训练或独立泛化评测。按用户指定的 WikiSkill 最小范围原则保留证据，不因文档核查新增技能训练、付费评测或声称技能已经改进。

## 阶段 X0 追加审计（2026-09-27）

本轮核查REVE-Base官方自监督预训练PT与LaBraM-Base原版权重。**REVE明确使用Dreyer2023、Cho2017、Lee2019_MI；LaBraM明确使用PhysioNet EEGMMIDB。** 因此不能把新基础模型的本项目测试被试隔离自动解释成预训练未见被试。五库核查结果如下；“未列入”仅为公开来源级结论。

| 本项目身份/别名 | REVE PT | LaBraM原版 | 证据定位与核查 |
|---|---|---|---|
| PhysionetMI / EEGMMIDB / Schalk2004 | 未列入；下游基准 | **明确重合** | REVE附录B的PhysioNet来源是Siena、ICARE，未列EEGMMIDB；正文§3.1.1称排除了下游记录，§3.3/C.5列PhysioNet MI为下游。LaBraM附录D明确109人运动/想象与两基线；不能因MOABB名字不同漏判 |
| Dreyer2023 A+B+C / **Inria Large** / Scientific Data 10:580 / DOI 10.1038/s41597-023-02445-z | **明确重合** | 未列入 | REVE附录B“Other sources”列Inria Large（Dreyer et al., 2023），参考文献题名、作者、卷页与本项目相符。不能只搜索Dreyer类名，也不能与LaBraM的Inria BCI Challenge（P300、Margaux2012）混同 |
| Cho2017 / GigaDB100295 / gix034 | **明确重合** | 未列入 | REVE附录B MOABB清单直接列Cho2017；LaBraM公开清单无此项 |
| Lee2019_MI / Lee2019MI / OpenBMI / GigaDB100542 | **明确重合** | 未列入 | REVE清单同时列MI、ERP、SSVEP；其中MI本身即已列入，不能靠仅用session1推断无重合 |
| BNCI2014-001 / BNCI2014001 / BCI Competition IV-2a | 未列入；下游基准 | 未列入 | REVE附录B出现BNCI2014004/2015001等，非IV-2a；IV-2a在下游§3.3/C.5。LaBraM列BCI Competition IV-1，非IV-2a |

主要来源：[REVE NeurIPS原论文，§3.1.1、§3.3、附录B pp.21–22、参考文献p.12、附录C.5](https://papers.nips.cc/paper_files/paper/2025/file/20a917f77773ac0fa8bea2bdd6606b66-Paper-Conference.pdf)、[Dreyer数据作者论文](https://www.nature.com/articles/s41597-023-02445-z)、[LaBraM ICLR原论文附录D pp.17–18](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf)。REVE的PhysioNet线索获得作者原论文支持，结论仍限官方PT来源清单，未获得逐记录manifest。

特别排除REVE附录E.2的**XFT**权重：其监督跨库MI微调使用PhysioNet、Cho、Lee等；原PT的来源结论不能套用于XFT。本项目固定 [brain-bzh/reve-base](https://huggingface.co/brain-bzh/reve-base/tree/dc2a075c287bb2f6c04ee5875bd79535a0f7dba6) 的普通预训练权重，X1加载前核实文件身份；不按下游成绩挑选checkpoint。LaBraM沿用本文件初审的固定官方commit，未替换为CBraMod作者重预训练的LaBraM变体。

REVE作者公开清单声称覆盖92个来源，但没有每一发布权重的逐被试、逐run文件哈希清单。Hugging Face可重分发语料只占部分，不能以某库未出现在可下载子集就断言未参与预训练。此次没有推定重合比例、删除部分被试伪造“干净子集”或执行除重预训练。

**X的结果前处理决定：保留全部用户指定组合，已见来源单独报告，不进入预训练未见来源的证据汇总。** REVE起步unlisted为PhysioNet，LaBraM为Dreyer+Cho；二者样本不同，不能直接比较这两个合并量。加上原CBraMod后，三模型共同unlisted的起步库为空；五库中仅BNCI为共同unlisted候选，仍须披露其开发阶段下游使用史。没有严格三模型共同未见来源的起步组复现结论。

新源码/元数据回执见 [X0 source_manifest](cross_model_audit_2026-09-27/source_manifest.json)，权重版本和下载后核验要求见 [X协议](cross_model_protocol.md)。旧7库表及历史结论保留；本轮不扩展到Stieger/Yang、不重新预训练、不根据未来结果改变上述分层。
