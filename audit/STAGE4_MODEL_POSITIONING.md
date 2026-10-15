# STAGE4_MODEL_POSITIONING.md —— 核心模型（SE-ResNet-BiGRU）在现框架中的定位与 Stage 4 预留设计

> **状态**：**设计 / 规划**（未训练、未改代码）。
> **触发**：外部审阅意见——「按当前框架，如果什么都不调整，你的主要模型的重要性会被明显削弱」。
> **上游**：`audit/STAGE1_PROTOCOL_FREEZE.md`（红线 1–23）、`audit/STAGE2_EVIDENCE_CHAIN.md`（封板）、
> `audit/STAGE3_EVIDENCE_CHAIN.md`（封板）、`audit/SCI_scope.md`（RQ1–RQ4 / H1–H5 / Non-Goals）。
> **结论（一句话）**：诊断**成立**，但**不需要改主线**；补丁是**预留 Stage 4**，而且
> **模型梯子已全部在 `rulbench/models/registry.py` 中注册、走同一训练循环与同一损失**——
> 即 **Stage 4 几乎零实现成本**。

### 执行顺序（已拍板，2026-10-08）

```
① 命名纠错（零成本）                          ✅ 已完成
② Stage 3 → manuscript §6.6 + Table 7 改写     ✅ 本次完成（§6.6 已重写为严格配对；§6.2–6.4 的数字仍待重跑）
③ 冻结当前论文主线
④ Stage 4（L1 架构增益 → L2 标签结构依赖 → L3 有限跨工况）
```

**先 ② 再 ④**：Stage 3 已封板，正文里不能再留**已被项目审计判定无效**的未配对 / 修复前数字。
Stage 4 **不打断** Stage 2/2.1/3（均已封板）。

---

## 1. 直接回答：当前框架还体现模型的重要性吗？

**部分体现——但只是「叙事级」，不是「实测级」。**

**✅ 现框架已经给模型的**：

1. 它是 Stage 2 / 2.1 / 3 的**唯一 backbone**，全文 headline
   （标签结构限制有效 learning gain；跨工况无稳定一致优势）**都是用这个模型建立的**；
2. 机制分析（bias / slope / **prediction-response compression** / 阶段依赖偏差）**以它为对象**；
3. 因此有一句**有份量**的主张已经成立：
   > **即使使用一个具有较强表征能力的深度 RUL 模型，标签结构本身仍会限制其能够获得的
   > 有效学习增益。**

**❌ 现框架没给模型的**：

1. **没有任何证据**表明这个架构优于替代方案（"架构溢价"未被测量过）；
2. **没有分解** SE 模块 / 残差 CNN 提取器 / 双向 RNN 各自的贡献；
3. ⇒ 读者（与审稿人）完全可以说："换成 CNN、LSTM 甚至普通 GRU，可能也一样。"

**⇒ 现状定性**：模型目前是 **carrier（实验载体）**，不是 **subject（研究对象）**。
要把 subject 身份补上，**只能靠 Stage 4**，且**不必打断现有顺序**（见 §4）。

---

## 2. 现状审计：模型出现在哪里、什么版本状态

| 结果位置 | 覆盖的模型 | 版本状态 | 对「模型重要性」的作用 |
|---|---|---|---|
| **S2L 单工况主表**（`artifacts/reports/single_linear/`） | `cnn, gru, lstm, informer_senet, ridge, se_resnet_bigru` + 2 trivial = **8** | ⚠ **修复前**（须重跑） | 唯一的多模型表，**但无架构消融梯子** |
| **S5L 标签谱**（`artifacts/reports/sweep_h{0.5..0.9}/`） | `const_mean, gru, se_resnet_bigru` = **3** | ⚠ **修复前**（须重跑） | 只对比 proposed vs **一个** baseline |
| **Stage 2 / 2.1**（`s2_chain/`, `stage2_bias.json`） | 仅 `se_resnet_bigru` | ✅ 修复后（封板） | 单模型，机制分析 |
| **Stage 3 全部臂**（`matched_fold*_b8`, `vm_fold*`） | 仅 `se_resnet_bigru` | ✅ 修复后（封板） | 单模型，跨工况裁决 |

> ⚠ **关键事实**：全仓库**没有任何结果目录包含 `resnet_bigru` 或 `pure_bigru`**
> （已用 `listdir` 核实）⇒ **架构消融从未运行过**。这正是外部意见所指的缺口。

---

## 3. 好消息：梯子已经在代码里（Stage 4 无须新实现）

`rulbench/models/registry.py` + `rulbench/models/ablations.py` 已注册全部消融变体，
且 `ablations.py` 的模块 docstring 明确写道：**消融与完整模型走同一 loop、同一 loss，
唯一差异 = 被消融的那个因子**——这恰好是论文版被批评的「消融与主模型训练协议不一致」的**代码级修复**。

**可用的因子分解（比线性梯子更强，因为它做的是因子隔离而非排序）**：

| 消融因子 | 关闭（对照） | 打开（处理） | 隔离出什么 | registry key |
|---|---|---|---|---|
| 残差 CNN 提取器 | `pure_bigru` | `resnet_bigru` | **提取器** 的贡献 | `pure_bigru` / `resnet_bigru` |
| SE 通道注意力 | `resnet_bigru` | `se_resnet_bigru` | **SE** 的贡献 | `resnet_bigru` → **proposed** |
| 双向 RNN | `se_resnet_gru` | `se_resnet_bigru` | **双向** 的贡献 | `se_resnet_gru` → **proposed** |
| 时序建模 | `cnn` | `pure_bigru` | **RNN** 的贡献 | `cnn` / `pure_bigru` |

> 外部意见提的线性梯子「CNN → ResNet → SE-ResNet → SE-ResNet-GRU」在本代码里没有
> 「纯 ResNet / 纯 SE-ResNet（**不带** RNN）」两级——提取器在本仓库里**总是**接一个 RNN。
> 上表的**因子分解**是等价且更严格的替代：它把「提取器 / SE / 双向」三个因子**各自单独**隔离。

---

## 4. 定位建议：采纳外部意见，但做三点收紧

**总原则（与外部意见一致）**：**不回到"架构创新"主线**，保留
`Label Structure → Zero-learning Baseline → Learning Gain → Cross-condition → Matched Volume`；
把 `SE-ResNet-BiGRU` 定位成 **core predictor**。

**三点收紧（与外部意见的差异）**：

1. **用因子分解，不用线性梯子**——理由见 §3 末注。这更能回答「SE 到底贡献了什么」这类审稿人必问的问题。
2. **三层验证全部用配对 G，而不是裸 RMSE 排行**：
   ```
   G_model = 1 − RMSE_model / RMSE_const_mean        （const_mean = 可部署零学习基准）
   ```
   这与你 Stage 2 已经建立的 `G` 口径**完全一致**，不引入新指标。
3. **预注册的表述是研究问题，不是方向性假设**（避免被读成"为迎合 Stage 2 而设计"，也避免被读成 leaderboard）：
   > **RQ（Stage 4）：不同网络架构所获得的 learning gain，对标签结构是否具有相似或不同的敏感性？**
   这与 Stage 2.1 的 **prediction-response compression** 自然衔接——
   Stage 2 已证**模型侧**在高 `h` 变差（`h ≥ 0.7` 时甚至劣于 `const_mean`）；
   Stage 4 只负责**测量**这种塌陷在不同架构上是否同步发生，**不预设**结论
   （若结果确实同步，再升格为发现；见 §6 R3）。

---

## 5. Stage 4 三层设计（与外部意见的三层对应）

**L1 · 架构增益是否存在**（单工况 linear）
在**同一 15 单元 / 同一测试集**上配对比较
`{cnn, pure_bigru, resnet_bigru, se_resnet_gru, se_resnet_bigru}` 的 `G`。
→ 回答「SE / 提取器 / 双向分别贡献多少」。

**L2 · 架构增益是否随标签结构变化**（与 Stage 2 完美衔接）
对 `h ∈ {linear, 0.5, 0.6, 0.7, 0.8, 0.9}` 重复 L1 → 得 `G_model(h)` 曲面。
→ 回答「架构溢价是否随 h 衰减、在哪塌到 0」。

**L3 · 架构增益能否跨工况保持**（核心落脚点）
**只对有限模型做跨工况**，避免几百个 checkpoint 的灾难：
`se_resnet_bigru`（核心）+ **1–2 个最有代表性的对手**（首选 `resnet_bigru`，其次 `pure_bigru`）。
直接复用 `scripts/28/29/30/31` 的 **5 折 + 训练量匹配臂**框架，比较 single / cross 两臂的 `G_model`。
→ 回答「强表征模型学到的退化表征，究竟能否跨工况迁移」。
**其余模型只在 L1（单工况 linear）跑，不进 L3。**

> 外部意见说「Stage 4 不需要模型海选，4 个左右足够」——**本仓库的因子分解恰好只要 4–5 个 key**，
> 且其中 2 个（`cnn` / `se_resnet_bigru`）**已有 checkpoint**，新增训练量很小。

---

## 6. 预先注册的 Stage 4 裁决规则（防事后择结论）

- **R1** 主判据 = **配对 `G` 差**（同一 `(condition, bearing)`），报分布 + 轴承层级自助 CI；**禁裸比 RMSE**。
- **R2** 架构增益必须**先在 linear 下成立**，才允许谈 L2 / L3。
- **R3** L2 读法（**不预设**结论）：若各架构的 `G` 在 `h ≥ 0.6` **同步**塌到 0，可写
  **「架构优势受标签结构门控」**，**不得**写「架构无效 / 没用」；若**不同步**（某些架构更耐受），
  则写「不同架构的 learning gain 对标签结构的**敏感性不同**」。两种结果都可发表。
- **R4** L3 沿用 Stage 3 **红线 21**（禁「等效 / 无差异」）。
- **R5** 每个新增模型必须**重新过确定性探针**（A7/A10：同配置 ≥3 遍逐位一致），才可引用。
- **（候选红线 24）** 架构主张必须先有 linear 下的配对 `G` 增益，再谈标签结构依赖与迁移。

---

## 7. 命名：**已拍板并执行（a + b）**

**硬事实（已核实代码）**：主表里标为 **"BiGRU" 的列实际是单向 `gru`**
（`GRURegressor.__init__` 默认 `bidirectional=False`）；而 proposed `se_resnet_bigru`
（`SEResNetBiGRU`）里 `bidirectional=True`，**确实是双向**。

### 7.1 规范命名表（**已执行**，零成本纠错）

| registry key | 论文名称 |
|---|---|
| `gru` | **GRU** |
| `pure_bigru` | **BiGRU** |
| `resnet_bigru` | **ResNet-BiGRU** |
| `se_resnet_gru` | **SE-ResNet-GRU** |
| `se_resnet_bigru` | **SE-ResNet-BiGRU**（核心模型） |

**已改位置**：`paper/manuscript.md`（Table 3/4/5 标签 + §6.2/§7.3 正文）、
`paper/tables/tables.md`（4 处），以及**根因**——生成器的显示名映射：
`scripts/12_paper_tables.py::DISPLAY`（原 `"gru": "BiGRU"`）与
`scripts/11_make_figures.py::MODEL_LABEL`（原 `"gru": "BiGRU"` + 图例 `label="BiGRU"`）。
两处均已改为 `"gru": "GRU"`，并补齐 `pure_bigru/resnet_bigru/se_resnet_gru` 的显示名。

### 7.2 `pure_bigru` **必须跑**（补齐因子分解）

它不是"多一个模型"，而是闭合**双向**这一环：

```
CNN  →  BiGRU  →  ResNet-BiGRU  →  SE-ResNet-BiGRU
                      （= pure_bigru → resnet_bigru → se_resnet_bigru）

SE-ResNet-GRU  →  SE-ResNet-BiGRU     ← 单独回答「双向时序建模是否有贡献」
```

> 注：`configs/base.yaml` 中**已经声明**了 `experiments.ablation_single`
> （`se_resnet_bigru, resnet_bigru, se_resnet_gru, se_resnet_lstm, pure_bigru`）与
> `experiments.ablation_cross`（`se_resnet_bigru, resnet_bigru, pure_bigru`）两个实验组，
> 但**从未运行**（§2 已核）。Stage 4 只需按这两个组驱动即可。

---

## 8. 成本、风险与不做的事

- **运行成本**：L1 ≈ (5 模型 − 已有 2) × 3 工况 × 5 seed；L2 = L1 × 6 个标签点；L3 复用 `scripts/28–31`。
- **风险**：
  - 每新增模型都须**重过确定性探针**（R5）；`informer_senet` 内部有自采样（已在代码里 deterministic 化）；
  - **不得**用修复前的 S2L / S5L 数字论证架构增益（它们是 ⚠）。
- **明确不做**：**不改 Stage 2 / 2.1 / 3**（Stage 2 红线 18 封板；Stage 3 已封板）。
  外部意见主张「现在不要因为模型重要性不够而打断 Stage 2」——**同意**。

---

## 9. 与现有红线的接口

- 复用：红线 7（"无稳定优势"的普适性需 > 1 模型）、A1–A10 披露规则、红线 19–23。
- 新增：**候选红线 24**（架构主张的前置条件，见 §6）。
- **Stage 4 与 Stage 3 的关系**：Stage 3 已在**单模型**上回答"跨工况是否提供额外信息"；
  Stage 4 的 L3 是**同一问题的架构稳健性检验**（把 Stage 3 的结论从 1 个模型推广到 4–5 个）。

---

## 10. 论文结构影响（双主线，交汇点即 Stage 4）

```
主线 A（模型）        CNN/GRU → SE-ResNet → [SE-ResNet-BiGRU]      答：怎样提取有效退化特征（Stage 4 L1/L2）
主线 B（框架）  标签结构 → 零学习基准 → Learning Gain → 跨工况 → 匹配训练量   答：何时真的学到、能否迁移（Stage 2/2.1/3）
                                         ↓ 交汇
                [SE-ResNet-BiGRU] + [Learning-Gain / Cross-Condition 框架]
```

**交汇点就是 Stage 4**：模型的价值不再靠"RMSE 最低"证明，而是靠
**"在框架内可被量化、可被定界（bounded）的有效学习增益"** 证明。

---

*End of STAGE4_MODEL_POSITIONING.md*
