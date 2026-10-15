# STAGE4_FINAL_INTERPRETATION_MEMO.md

**改稿前的唯一依据。** 本文件冻结 Stage 4 的最终解释与措辞边界。**在改 `paper/manuscript.md`
之前，任何措辞若与本文件冲突，以本文件为准；若要越界，必须先改本文件并说明理由。**

- 上游依据：`audit/STAGE2_EVIDENCE_CHAIN.md`、`STAGE3_EVIDENCE_CHAIN.md`、`STAGE5_RESULTS.md`、
  `STAGE4_PROTOCOL_FREEZE.md`（含 §9.5 B' 结果与 S4-T3 裁决、红线 28–37）。
- 证据根：**`artifacts/reports/s4/`**（825 ckpt）+ `artifacts/reports/stage4_analysis/`；
  表 = `paper/tables/stage4_tables.md`。
- 定位：**Stage 4 已完成，不再补实验**；本 memo → 再改 Abstract → Contributions → Results/Discussion。

---

## 0. 一句话结论（论文的转向）

**旧叙事（证据不支持，禁止）**
> 我们提出 SE-ResNet-BiGRU，通过复杂结构（SE 注意力 / 残差 / 双向）提升 RUL 预测性能。

**新叙事（证据支持，唯一允许）**
> 在严格协议与多工况验证下，RUL 预测性能的提升并非主要来源于网络复杂度增加，
> 而更多受**标签结构、任务可学习性与工况迁移难度**共同限制。

转向性质：**"模型创新型论文" → "严格评估与机制分析型论文"。**
这不是"负面结果"，而是**"重新解释已有改进策略的性能来源"**——比普通模型论文更接近方法论研究。

因果拆解（三层）：
```
architecture complexity  ──►  limited / unstable contribution      (S4-T1, S4-T3)
label & task structure   ──►  dominant factor                      (Stage 5, S4-T2)
```

---

## 1. 三层证据的最终结论

### 1.1 Stage 5（标签结构层）—— 回答「为什么模型在不同 RUL 分布下表现变化」
| 项 | 结论 |
|---|---|
| headline | proposed 六点 `gap` 随 `h` **单调**（−2.14 → +44.27 min），**变号仍 h≈0.60** |
| crossover | **只报 adjacent 区间**（两口径 h≈0.55–0.60）；**禁精确 `h*`** |
| h=0.8 全阵容 | **无深度模型击败平凡基线**（best deep = LSTM 56.2 vs hurdle 43.0）|
| ⚠ 红线 25 | **禁恢复"RMSE 随 h 严格单调"**（h0.8 77.15 > h0.9 76.18，严格单调**已被证伪**）|

⇒ 措辞：**「标签结构改变了任务的信息可学习程度」**，不是"某个模型突然失效"。

### 1.2 S4-T1（结构因素拆解）—— 回答「SE / ResNet / BiGRU 是否贡献主要性能」
| 因素 | 边 | median `ΔG` | 95% CI | CI 跨 0 |
|---|---|---|---|---|
| ResNet extractor | BiGRU → ResNet-BiGRU | +0.074 | [−0.086, +0.485] | 是 |
| **SE** | ResNet-BiGRU → SE-ResNet-BiGRU | **−0.023** | [−0.123, +0.058] | 是 |
| Bidirectional | SE-ResNet-GRU → SE-ResNet-BiGRU | +0.108 | [−0.200, +0.165] | 是 |
| *CNN → BiGRU*（**补充，不入链**） | *family comparison* | *+0.255* | *[+0.118, +0.382]* | 否 |

⇒ **无稳定组件增益**。SE 的**连方向都为负**，是很有价值的讨论点：
**attention 的收益依赖任务条件，而非普遍有效**（大量论文默认 `SE attention = improvement`）。

### 1.3 S4-T2（标签结构 × 架构）—— 回答「限制来自标签还是架构」
`G_m(h)` 六点对四个模型**均单调降且变号**；observed crossover：

| Model | crossover |
|---|---|
| SE-ResNet-BiGRU | h=0.5 → h=0.6 |
| ResNet-BiGRU | h=0.5 → h=0.6 |
| SE-ResNet-GRU | h=0.5 → h=0.6 |
| BiGRU | linear → h=0.5 |

⇒ **不同架构 crossover 接近** ⇒ **独立新训复现了 Stage 5 的 h≈0.55–0.60 零点**（headline 二次验证）。
措辞：**「限制来自标签信息结构，而不是某个架构」**；**禁"架构无效"**（红线 31）。

### 1.4 S4-T3（迁移层）—— 回答「复杂结构能否增强跨工况能力」
unit = `(fold, target condition)`，**n = 15**（= 全部 15 轴承；单元内 3 个源任务为**重复测量**，按中位折叠）。

| Model | n | `R_cross` 中位 | 95% CI | 跨 1? | `ΔRMSE` 中位 (min) | 95% CI | 跨 0? | cross 更优 |
|---|---|---|---|---|---|---|---|---|
| SE-ResNet-BiGRU | 15 | 1.287 | [1.151, 1.817] | 否 | +14.59 | [+8.62, +71.53] | 否 | 13 % |
| ResNet-BiGRU | 15 | 1.202 | [1.012, 2.118] | 否 | +11.06 | [+1.09, +23.19] | 否 | 27 % |
| **BiGRU** | 15 | **0.991** | [0.792, 1.690] | **是** | −8.23 | [−19.79, +22.76] | **是** | 60 % |

**裁决（严格按预注册）**：
1. **最简架构 BiGRU 的 `R_cross` CI 跨 1** ⇒ 对它跨工况**未表现出稳定代价**。
2. 两个更深架构 CI **不跨 1**、`ΔRMSE` CI **不跨 0** ⇒ 代价**稳定**，方向 = **"更深 ⇒ 跨工况更差"**。
3. **但不足以说 SE/ResNet 有害**：配对核验 `extractor` 差 **+0.410 [0.021, 0.602] 不跨 0，
   却仅 3/15 单元更优、逐单元 +3.41↔−2.83 翻转**；`SE` 差 **+0.067 [−0.405, +0.348] 跨 0**（7/15）。
   ⇒ 结论上限：**"架构复杂度未表现出对工况迁移的稳健优势；在有代价的架构上该代价随深度略增"**。

**唯一允许的英文措辞（用户 2026-10-11 明确认可）**：
> **Increased architectural complexity did not translate into robust cross-condition benefits.**

---

## 2. 可以说什么 / 禁止说什么（措辞对照表）

| # | ✅ 可以说 | ❌ 禁止说 | 依据 |
|---|---|---|---|
| 1 | 性能提升**并非主要**来源于网络复杂度 | 网络复杂度**无用** / **无效** | 红线 30/36；`ΔG` CI 跨 0 只说明"无稳定增益" |
| 2 | SE / ResNet / 双向 **未表现出稳定一致**的组件增益 | SE **有害** / SE **降低了**性能 | 红线 36；SE 中位 −0.023 但 CI 跨 0 |
| 3 | attention 的收益**依赖任务条件**，非普遍有效 | attention **没有价值** | 语义等同 #2 |
| 4 | 跨工况训练**未表现出稳定一致的优势** | 跨工况**等效** / **无差异** | 红线 21（CI 是"宽"不是"紧贴 0"）|
| 5 | 架构复杂度**未转化为稳健的跨工况收益** | 复杂结构**降低**泛化能力 | 用户认可措辞；配对方向翻转 + 尺度陷阱 |
| 6 | 最简架构跨工况**未观察到稳定代价** | BiGRU **迁移更好** | 只有它的 CI 跨 1 = 未拒绝 null |
| 7 | headline 零点在 **h≈0.55–0.60 区间** | 零点精确为 **h\* = 0.6** | 红线 31；只报 observed interval |
| 8 | `gap` 随 `h` **单调** | **RMSE 随 `h` 严格单调** | 红线 25（h0.8 77.15 > h0.9 76.18）|
| 9 | 跨协议数字**各自内部**比较 | S4-T3 的 1.287/1.202/0.991 与 Stage 3 的 **1.086 并列** | **A9**：B' 是**双源** cross，Stage 3 是**单源**，不可比 |
| 10 | `R_cross` 与 `ΔRMSE` **并报**（协议 S5）| 只报比值 / 只报绝对差 | **尺度陷阱**（见下）|
| 11 | 这是**方法论 / 机制分析**贡献 | 负面结果 = **失败** / 我们**没能**做出模型 | §0 转向 |
| 12 | 平凡基线是**可部署**主对手（`const_mean`） | `best_const` 当**可部署**对手 | 三角色冻结定义 |

**三个必须记牢的数字陷阱**：
- **A9 口径**：B' cross = **双源**（`1+2→3` 等）；Stage 3 = **单源**（`1→2` 等）。**两组数字禁止并列。**
- **尺度陷阱**：`ΔRMSE` 中位被**低 RMSE 折**主导（fold4/cond2 的 single RMSE 仅 1.69–6.30 min，
  比值 6.5× 其实只 5–10 min 绝对差）⇒ **必须并报比值与绝对差**（协议 S5）。
- **SE 参数不是"参数更多"**：`pure_bigru` 153,729 →`resnet_bigru` 280,705 →`se_resnet_bigru` 283,265
  （SE 仅 +2,560 ≈ **0.9 %**）⇒ **SE 的无效不能用参数量解释**；这是支持"结构选择而非容量"论点的硬事实。

---

## 3. Abstract 需要怎么调整

**当前 Abstract（已转向，但 Stage 4 完全缺席）**：标题已改为
"The label, not the architecture, decides the verdict: a label-sensitivity analysis and a
disclosure-aware evaluation protocol for rolling-bearing RUL prediction"，
主体是 Stage 5 + 十规则 + Stage 3 的 `R_cross` 1.09。**没有 T1/T2/T3。**

**改法（增补，不推翻）**：
1. **保留** Stage 5 的 headline（`gap` 单调 + h≈0.55–0.60 变号）与 h=0.8 负结果 —— 这是地基。
2. **新增一句架构层证据**（T1）：*a paired component ablation finds no stable contribution from
   the residual extractor, the squeeze-and-excitation block, or bidirectionality
   (all Δ G confidence intervals straddle zero; the SE block's median is slightly negative).*
3. **新增一句标签层复核**（T2）：*the crossing point is reproduced independently under a fresh
   training run for four architectures, whose crossovers coincide between h = 0.5 and h = 0.6* ——
   把"标签决定"从单次观察升级为**跨架构复现**。
4. **新增一句迁移层**（T3）：*under a paired cross-condition design, increased architectural
   complexity did not translate into robust cross-condition benefits
   (R_cross 1.29 / 1.20 for the two deeper models versus 0.99 for the simplest, whose interval
   is the only one straddling 1).*
5. **收束句**改为：性能来源被系统性分解为**标签结构（主导）+ 任务可学习性（主导）+
   工况迁移难度（限制）**，而**架构复杂度贡献有限**。
6. ⚠ Abstract 中**不得**出现 "propose SE-ResNet-BiGRU"、**不得**出现 `h* = 0.6`、
   **不得**把 Stage 3 的 1.09 与 S4-T3 的 1.29/1.20/0.99 并列成同一序列。

---

## 4. Contributions 需要怎么调整

**现状（Introduction 末尾，4 条）**：① closed-form LI 理论；② controlled label-sensitivity sweep；
③ external validation (PHM2012)；④ disclosure-aware protocol (A1–A6)。
→ 已经是非模型中心，**但缺 Stage 4 的三段机制证据**，且规则数已扩到 A1–A10（文中仍写 A1–A6，需同步）。

**建议改为 3 层（用户 2026-10-11 指定方向）**：

| 新 | 内容 | 证据 |
|---|---|---|
| **C1 严格评估协议** | label definition + 完整平凡基线阶梯 + 解析下界 + 评分口径 + 披露规则 | §3、§5、Stage 5、A1–A10 |
| **C2 标签结构敏感性分析** | LI 闭式解 + `h` 扫描 + crossover 区间 + **跨架构独立复现** | Stage 5、**S4-T2** |
| **C3 架构贡献分解** | 组件消融（T1）+ 跨工况迁移（T3）**双向**拆解 | **S4-T1、S4-T3** |

对应 RQ（用户指定，建议替换现有单一 RQ）：

- **RQ1：深度 RUL 模型是否真正超过简单基线？** — **部分**。并非所有标签结构；
  存在 `h` 阈值（crossover，区间 h≈0.55–0.60）。
- **RQ2：网络复杂度是否带来稳定收益？** — **否**。证据：S4-T1 三因素消融（CI 全跨 0）+
  S4-T2 crossover 一致性。
- **RQ3：复杂结构是否提升跨工况迁移？** — **未观察到稳健优势**。证据：S4-T3。

⚠ 若保留"carrier architecture"那段（SE-ResNet + BiGRU + MK-MMD 作为实验载体而非主张），
**必须保留**——它是 C3 成立的前提（模型是载体，不是卖点）。

---

## 5. 结果/讨论章节的插入方案（Stage 4 落在哪里）

Stage 4 **不是**新章节主体，而是**给已有论断补上机制证据**。建议：

1. **§6.6 之后新增 §6.7 "Decomposing the architectural contribution"**（或并入 §6.6 末尾）：
   - §6.7.1 组件消融（S4-T1 表：extractor / SE / bidirectional，三边 CI 跨 0；
     附 supplementary `CNN→BiGRU` +0.255 [+0.118, +0.382] 作**家族比较**，**明确声明不入链**，红线 33）。
   - §6.7.2 标签谱跨架构复核（S4-T2 表：四模型 crossover，附 `ΔG_arch(h)` 边）；
     **明确写"独立新训复现"**——这是对 §6.3 headline 的加固（A10 + 复现）。
   - §6.7.3 跨工况架构比较（S4-T3 表，n=15）；
     **必须并报 `R_cross` 与 `ΔRMSE`**；**必须加 A9 脚注**说明与 §6.6 的双源/单源口径不同、不可比。
2. **§7.1「The verdict is a property of the benchmark」**：把标题级论断从"标签"扩展为
   **"标签结构 + 任务定义"**，并引用 S4-T2 的跨架构复现作为加强证据。
3. **§7.2 Practical guidance**：补一条——**报告架构消融时必须给配对 CI**（因为 CI 跨 0 是常态）；
   **报告迁移时必须并报比值与绝对差**。
4. **§7.3 What this does *not* say**：补两条
   - 不说"深层架构无法迁移"（配对方向翻转、尺度陷阱）；
   - 不说"SE 无用"（仅"未表现出稳定增益"；且参数增量仅 0.9 %，不是容量问题）。
5. **§7.4 Limitations**：补一条——S4-T3 的 cross 臂为**双源**设计，与 §6.6 的单源设计
   **不构成同一比较**；两者各自内部有效，**不跨设计比较**（A9）。
6. **§5 ten disclosure rules**：文中仍写 **A1–A6**，需同步为 **A1–A10**，
   并把 **A10（评测播种）** 与 **B8/B10/B11 属基础设施修复、不入 Contributions** 一并写清。

⚠ **B8 / B9 / B10 / B11 全部禁入 Contributions**（均为实验基础设施缺陷/修复，不是科学发现）。
B11（共享臂目录写表必须 merge，红线 37）尤其要留在工程文档，不写进论文。

---

## 6. 不得再做的事（用户 2026-10-11 明确）

- ❌ 不再补实验；❌ 不跑完整 **L3vm**；❌ 不扩大模型阵容（不加 CNN/LSTM/Informer/SE-ResNet-GRU 到 L3）；
- ❌ 不恢复"模型优势证明"路线；
- ❌ 不在同一次操作里既改协议又改正文；
- ❌ 不把负面结果包装成失败，也不夸大（"首次""最优"类词禁用）。

---

## 7. 复现与审计锚点

| 项 | 值 |
|---|---|
| `s4/` ckpt 总数 | **825**（L1 75 + L2 300 + L3 450）|
| B' unit / 耗时 | **30/30，零失败，67.7 min** |
| `36 verify` | **PASS**（9 复用行 × 9 列与冻结 Stage 5 逐位相同）|
| 确定性探针 | Probe A（训练）+ Probe B（评测）**OVERALL PASS** |
| `G(h)` 交叉校验 | `s4/se_resnet_bigru` 与冻结 `s5` 在 **6 个 h 上 diff = +0.0000** |
| S4-T1/T2 vs B11 前 | **逐位相同**（+0.074 / −0.023 / +0.108；crossover h=0.5→0.6）|
| 生成脚本 | `scripts/36_run_stage4.py`、`37_stage4_analysis.py`、`38_stage4_determinism_probe.py` |

*End of STAGE4_FINAL_INTERPRETATION_MEMO.md*
