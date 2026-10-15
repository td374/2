# REVIEWER_PACK.md —— 交给外部审核者（GPT 等）的文件清单

> 目的：让审核者**能自己核对**，而不是复述结论。
> 本次审计（`audit/` 9 份文件）的每条判定都挂在 `文件 → 函数 → 行号` 与真实数据上，
> 因此**必须给代码与数据**，否则审核退化为「转述」。

---

## 0. 先回答：只给对话文字够吗？

**不够。** 三个具体理由：

1. 对话里只有**摘要**，没有代码原文 → 审核者无法验证 `metrics.py L234-244` 到底写了什么；
2. 对话里引用的数字（0.4936 / 642.19 / 1.064 / p=0.153 / 0.07×–18.35×）都来自
   `artifacts/reports/*.json` 与 `artifacts/data_matched/*.npz` → 不给文件就没法复算；
3. 最容易被漏掉的一点：**审计的价值在于「我说错了什么」**。
   只给对话文字，审核者只能看到我的论证链，**看不到被论证的对象**，等于让他给我打分而不是查我。

> 结论：**对话文字可以作为「导读」，但不能作为唯一材料。** 附件是必需的。

---

## 1. Tier A —— 必需（**约 2.4 MB**）

| # | 路径 | 体积 | 支撑审计中的什么 |
|---|---|---|---|
| A1 | `audit/`（9 个 md） | 0.11 MB | 被审核的对象本身 |
| A2 | `新方案与框架文件/SCI研究设计方案_V2.0.md` | 0.03 MB | **唯一主基线**；C1–C14、H1–H5、P0–P3 的出处 |
| A3 | `xjtu_sy_rul/rulbench/`（66 文件） | 0.4 MB | 全部 `rulbench.*` 行号引用的实体：`data/labels.py`、`data/windows.py`、`data/build.py`、`data/scaling.py`、`evaluation/metrics.py`、`evaluation/report.py`、`evaluation/inference.py`、`train/protocols.py`、`losses.py`、`models/registry.py`、`models/trivial.py` |
| A4 | `xjtu_sy_rul/scripts/`（32 文件） | 0.2 MB | `09`/`10`/`11`/`15`/`18`/`22` 六个被点名的脚本 |
| A5 | `xjtu_sy_rul/configs/`（16 文件） | 0.05 MB | `base.yaml`、`linear.yaml`、`cross_matched_linear.yaml` |
| A6 | `xjtu_sy_rul/tests/`（2 文件） | 0.01 MB | 判断是否有针对锚点/指标的单元测试（审计里说「指标单元测试」缺失） |
| A7 | `xjtu_sy_rul/paper/manuscript.md` | 0.1 MB | 审计点名的 §3.2/§3.3/Table 1（L229-235）、§6.6 Table 7（L500-510）、§7.4 |
| A8 | `xjtu_sy_rul/docs/`（16 个 md） | 0.2 MB | D9 点名的 6 处 `LI(h)=3(1-h)/4` 误用：`contributions.md` L11、`experiment_plan.md` L923、`comparison_strategy_qa.md` L94/169/234、`S2L_S5L_results.md` L63、`STATUS_review.md` L83 |

**A 包足以核查**：所有「定义/公式/代码行号/措辞」类判定（A1、A2 的推导、A3 的机制、A6 的对称性逻辑）。

---

## 2. Tier B —— 建议（**+ 约 31 MB**）：能独立复算数字

| # | 路径 | 体积 | 支撑审计中的什么 |
|---|---|---|---|
| B1 | `xjtu_sy_rul/artifacts/reports/` **仅 `*.json` / `*.csv` / `*.yaml`**（225 文件） | 0.9 MB | 数字核对的关键：`cross_transfer_summary.json`（`oracle_const_task__rmse=642.1947`、`n_windows=6879`）、`matched_rcross.json`、`matched_fold_rcross.json`、`matched_comparison.json`、`volume_audit.json`、`label_sensitivity.json` |
| B2 | `xjtu_sy_rul/artifacts/data_matched/`（26 个 npz） | 30.6 MB | 独立验证三项最强主张：① 划分无泄漏；② **9/9 任务跨工况 test 集与单工况 test 集逐位相同**；③ 训练量比 0.07×–18.35× |
| B3 | `xjtu_sy_rul/artifacts/data_linear/`、`data/`（npz） | 各约 30 MB | 验证「旧配置 single=轴承5、cross=全5轴承」→ 「946.63 vs 35.32」伪影 |
| B4 | `xjtu_sy_rul/artifacts/reports/label_anchors_linear.json` | — | `09` 锚点的落盘结果 |

> B2/B3 是**最高性价比**的一项：给了它，审核者可以**自己跑**
> `scripts/22_train_volume_audit.py --data-dir artifacts/data_matched`
> 和 `scripts/09_label_baseline_anchors.py --config configs/linear.yaml --data-dir artifacts/data_linear`
> 得到与我完全相同的表。**这才叫审核。**

---

## 3. Tier C —— **不必给**（会被体积淹没）

| 路径 | 体积 | 为什么不必给 |
|---|---|---|
| `xjtu_sy_rul/artifacts/reports/**/*.pt` | **778 MB**（951 个检查点） | 审计**未重训**；要复核 `D2/D3`（`report.py` 的区间/方差）时，B1 的 json 已含 `per_bearing` 与 `predictions` |
| `xjtu_sy_rul/datasets/` | **3.18 GB**（43,595 文件） | 原始 XJTU-SY 振动信号；审计的结论全部在**特征/标签层**，不追到原始信号 |
| `xjtu_sy_rul/artifacts/feature_cache/` | — | 中间缓存，与审计结论无关 |
| `xjtu_sy_rul/legacy/` | 0.04 MB | 旧实现；仅在追问「原论文怎么做」时有用（可留可去） |
| `output/`、`logs/` | 0.2 MB | 前次 docx 流水线的运行日志 |

**一句话**：给了 C 包，审核者会被 1 GB 噪音淹没，反而降低审核质量。

---

## 4. 给审核者的建议指令（可直接粘贴）

> 以下是**推荐的任务描述**，不是必须。重点是让它「找反例」而不是「复述」。

```
你是一个独立审计者，任务是**证伪**下面这份审计报告，不是复述它。

材料：
- 被审对象：audit/ 下 9 个 md（先读 STAGE1_AUDIT_REPORT.md 和 audit_02_constant_baseline.md）
- 唯一主基线设计：新方案与框架文件/SCI研究设计方案_V2.0.md
- 源码：xjtu_sy_rul/rulbench/、scripts/、configs/
- 论文现状：xjtu_sy_rul/paper/manuscript.md
- 可复算的产物：xjtu_sy_rul/artifacts/reports/*.json 与 artifacts/data_matched/*.npz

请在以下 6 个点逐一找反例或确认，每条都要给（文件 + 行号）或（复算命令 + 结果）：

1. audit_02 主张「报告指标是分钟空间无权重 RMSE，所以最优常数是 L² 加权均值，
   scripts/15_cross_transfer.py::_oracle_constant 正确」。
   请独立复核：metrics.py / report.py 里 RMSE 到底在哪个空间、有没有隐含权重；
   并用 data_matched 里的 npz 暴力搜索验证 c*。

2. audit_02 主张 scripts/11_make_figures.py::analytic_anchors 把 sqrt 放在 mean 之前、
   算成 MAE、使 hurdle 低估 13.4%。请核对行号并复算。

3. audit_03 主张无泄漏：轴承不重叠、ChannelScaler 与 mRMR 只在 train 上拟合。
   请指出任何**未被覆盖**的泄漏路径（例如缓存指纹 build.py::_fingerprint 只含
   denoise/fft/multi_domain → 换标签不换 data_dir 会静默复用）。

4. audit_04 主张训练量不可比（0.07×–18.35×，符号随目标工况翻转）。
   请用 scripts/22_train_volume_audit.py 复算，并检查「有效优化步数」是否也不匹配。

5. audit_05 主张 §6.6 Table 7 用的是未配对结果。
   请核对 manuscript.md L500-510 的数字与 cross_transfer_summary.json / matched_* 的对应关系。

6. 请指出这份审计**自身的**方法论问题：有没有「未核实」被当成结论、
   有没有用对话文字里的数字而未落到文件、有没有把「缺实验」与「定义错误」混淆。

输出格式：每条给「确认 / 推翻 / 无法判定 + 证据」，最后给一份你认为**最该先修**的 3 项排序。
```

---

## 5. 三个「别让审核者被带跑」的提醒（可选，附在材料里）

1. **区分「定义错误」与「缺实验」**：`A4 = FAIL` 是**缺训练量匹配臂**，不是定义错。
   如果审核者把它读成「论文数学错了」，那是误读。
2. **`audit/` 是过程文件，不是论文**：里面直接写「FAIL/MODIFY」是审计语言，
   不代表论文里会出现这些词。
3. **「未核实」清单要一起给**：`STAGE1_AUDIT_REPORT.md` §9 列了 U1–U10，
   这些是**我明确说没查**的，审核者可以优先补查这些，而不是重复已查过的。

---

## 6. 打包建议（若要一次性给出）

最小可用：**Tier A + Tier B1**（约 3.3 MB），一个 zip 即可。
完整可复算：**Tier A + B1 + B2**（约 34 MB）。

排除规则（务必加进打包命令）：
```
排除：**/*.pt   **/datasets/**   **/feature_cache/**   **/.git/**
```

*End of REVIEWER_PACK.md*
