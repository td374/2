# REVIEW_ROUND1_RESPONSE.md —— 对第一轮独立审核的响应与处置

| 项 | 值 |
|---|---|
| 触发 | 用户转来的独立审核意见（Round 1），核心：**Stage 0→1 合格（8.5/10）**，但 **`p=0.153` 不能当 H4 的正式统计证据** |
| 审核者裁定 | 采纳「读法 B（按阶段）」：先处理 **B2/B3/B4/B5 + 统计单位协议**，再进 Stage 2；**训练量匹配臂留 Stage 3** |
| 本次动作 | 冻结统计单位协议 + 新增再分析脚本；修 3 处代码缺陷并重算受影响产物；修订 `audit_05` 与 `STAGE1_AUDIT_REPORT` 的相应结论 |
| 硬边界 | **未改论文正文**；未启动任何训练（所有重算均复用既有 checkpoint） |
| 日期 | 2026-10-07 |

---

## 1. 逐条采纳

| 审核意见 | 原审计的立场 | 本代理的处置 |
|---|---|---|
| **A5 的 `p=0.153` 不能被当成 H4 的正式统计证据** | ❌ 原 `audit_05` §4.2、`STAGE1_AUDIT_REPORT` §2.4 把 pooled `p=0.153` 与「支持 H4」并列 | ✅ **接受**。统计单位冻结为**轴承**；p 降级为**探索性**；主证据改为**分布 + 效应量 + 轴承层级自助 CI**。见 §2 |
| 统计单位必须是轴承，不能用 pooled cell Wilcoxon | ⚠ 原审计指出「折间依赖未处理」，但仍把 pooled p 当正式推断 | ✅ 见 §2（`audit/STATISTICAL_UNIT_PROTOCOL.md` + `scripts/23_stat_unit_rcross.py`） |
| 不要现在改论文、不要跑 Stage 2 全部实验 | — | ✅ 遵守：**未改论文**，未跑 Stage 2 |
| 先处理 B2/B3/B4/B5 | — | ✅ B3 见 §2；B2 见 §3；B4 见 §4；B5 见 §5 |
| training-volume matched arm 留 Stage 3 | A4=FAIL 读法乙 | ✅ 维持：`R_cross`/`ΔRMSE` 在匹配臂完成前**只作描述性**，禁因果措辞 |

---

## 2. B3 —— 统计单位与推断协议（本次核心）

### 2.1 冻结内容

见 `audit/STATISTICAL_UNIT_PROTOCOL.md`。要点：

- **统计单位 = 测试轴承实例 = `(fold, target condition)` = `(condition, bearing)`**。
  五折 × 三目标工况 = **15 个单元 = XJTU-SY 全部 15 个轴承**。
- 每个单元内的 **3 个源任务变体是重复测量**（同一轴承上换源工况），**不是**新样本。
  故 `45 cells = 15 units × 3 repeats`，**不能**当 45 个独立样本做检验。
- 主证据 = **分布（中位数/分位/逐折符号）+ 效应量（`R_cross` 与 `ΔRMSE` 分钟）+ 轴承层级自助 CI**。
- p 值一律降级：cell 级标 `EXPLORATORY_ONLY`；单元级标 `indicative`（n=15，功效低）。
- 必须**按目标工况分层**（训练量混淆符号随目标翻转）。
- 自助 CI 是**下界**：跨折的训练轴承重叠，15 个单元并非严格独立（协议 S10）。

### 2.2 实现与数值

`scripts/23_stat_unit_rcross.py`（纯再分析，不训练）→ `docs/stat_unit_rcross.md`、`artifacts/reports/stat_unit_rcross.json`。

**主口径 = 未校正（uncorrected），n = 15**：

| 量 | 值 |
|---|---|
| 单元级中位 `R_cross` | **1.086**（95% CI **[0.889, 1.276]**，**跨 1**） |
| IQR / 极差 | 0.468 / 0.654–3.007 |
| 跨工况更优的单元占比 | **33%** |
| 单元级中位 `ΔRMSE` | **+3.44 min**（95% CI **[−1.81, +12.62]**，**跨 0**） |
| 逐折中位 | 0.889 / 0.956 / 1.276 / 1.357 / 1.061 → **>1 者 3/5** |
| 效应量 vs 异质度 | \|median−1\| = **8.6%** ≪ 折间跨度 **42.3%** |
| cell 级 Wilcoxon（45 cells） | p = 0.153 → **探索性** |
| 单元级 Wilcoxon（n=15） | p = 0.252 → **indicative** |

**按目标工况分层**（暴露训练量混淆的方向）：

| target | n | 中位 `R_cross` | 95% CI | 更优占比 | 中位 `ΔRMSE` |
|---|---|---|---|---|---|
| 1 | 5 | 1.061 | [0.760, 1.276] | 40% | +0.52 min |
| 2 | 5 | 0.956 | [0.654, 3.007] | 60% | −1.81 min |
| 3 | 5 | 1.181 | [1.019, 1.425] | **0%** | +166.44 min |

> 目标 3 的「一致更差」**不能**归因于域漂移：该工况跨工况臂训练量仅 **0.07–0.27×**（A4/D6）。
> 且其 `ΔRMSE` 数值大主要是 cond3 轴承**寿命最长**（寿命 111–2538 min）→ 同样的比例恶化换算成分钟被放大，
> 属量级效应而非比例效应（比较时应看 `R_cross`）。

**敏感口径 = 校正（corrected）**：中位 1.264、CI [1.082, 1.610]、更优占比 13%。**这不是主证据**：
校正因子只在跨工况臂施加（`report.py` 仅 `protocol=="cross"` 时启用），单工况臂 factor≡1.0 →
**跨协议对比在校正口径下是口径错配**（规则 A9）。它只用于敏感性。

### 2.3 对 H4 的影响

**修订后的 H4 措辞（采纳审核者版本）**：

> 在严格配对的跨工况实验中，跨工况训练相对于目标工况内训练**未表现出稳定一致的优势**，
> 其**效应方向随留出轴承和任务而变化**。

**禁止**：把 `p=0.153`（cell 级）或 `p=0.252`（单元级）写成「显著/不显著」的正式结论；
**禁止**把结论升级为「等效/无差异」（`stage0_scope_freeze.md` §2.1）。

---

## 3. B2 —— `ΔRMSE` 字段（已补）

`ΔRMSE = RMSE_cross − RMSE_single`（分钟；规则 D4/C6）现已在三处产出：

| 位置 | 字段 |
|---|---|
| `scripts/23_stat_unit_rcross.py` | 单元级 `median_dRMSE` + 自助 CI；JSON `effect_size.median_dRMSE_min` |
| `scripts/21_fold_rcross.py` | 池化 `median_delta_rmse`；每 cell 的 `d`；JSON `median_delta_rmse_min` |
| `scripts/20_matched_rcross.py` | 每 (target,model) 的 `median_delta_rmse`；池化 `median_delta_rmse` |

重算结果（`matched_fold_rcross.json`，`se_resnet_bigru`，45 cells；B8 修复后）：

```
uncorrected: pooled median R_cross = 1.0639   median dRMSE = +3.44 min
corrected  : pooled median R_cross = 1.2515   median dRMSE = +13.46 min
```

`R_cross` 的池化中位数与 p 与修复前**一致**（1.0639 / 0.1530）——B8 修复后重算回到了原始值；
个位格有 0.00x 级微调（见 §4.1）。

---

## 4. B4 —— `report.py` 两处口径缺陷（已修 + 已重算）

`rulbench/evaluation/report.py::evaluate_dataset`：

- **D2**：`std_min` 原先**未乘 `factor`**，而 `lower/upper` 乘了 → 跨工况 `crps`（源自 `std`）与区间**不同源**。
  → 现改为 `std_min = std_raw * factor`。
- **D3**：「未校正」行原先**复用已校正的 `lower/upper`** → 其 `picp` 是拿未校正点估计配校正区间算的。
  → 现为未校正行单独构造 `lower_unc/upper_unc/std_raw`。

**验证（fold1 跨工况，`se_resnet_bigru`）** —— 修复后校正/未校正两行的 `picp`/`crps` **不再相同**：

| task | picp (corr / uncorr) | crps (corr / uncorr) |
|---|---|---|
| `1->2` | 0.4739 / 0.4630 | 179.64 / 84.94 |
| `2+3->1` | 1.000 / 0.9783 | 21.86 / 13.87 |
| `1+2->3` | 0.4747 / 0.3355 | 795.11 / 882.05 |

（五折全覆盖：`crps` 在 5/5 折的 9/9 任务上都不同；`picp` 除少数两口径都饱和到 1.000 的任务外均不同。）

**影响面**：仅跨工况臂的 `picp`/`crps`；**`rmse` 只依赖预测均值，本应不受影响** —— 但实测**它还是变了**，
由此发现了下面的 B8。

**重算范围**：`matched_fold{1..5}/cross` 的 5 个报告目录已复用既有 checkpoint **重算**（无训练）。
`repro_*`（试点）目录未重算——见 §6.3。

### 4.1 ⚠ 重算过程中**新发现**一个缺陷（B8）：纯再评测不可复现

**现象**：重算 fold1 后 `R_cross` 的 pooled 中位数从 **1.0639 → 1.0644**、p 从 0.1530 → 0.1498；
而 §4 的改动**只涉及** `std`/`lower`/`upper`，**不触及** `rmse`（只依赖预测均值）。即 **`rmse` 本身漂了**。

**第二次独立重算**（同配置、同 checkpoint、全新进程）给出**不同**的 rmse：

| task | 第 1 次重算 | 第 2 次重算 |
|---|---|---|
| `1->2` | 268.66 | **267.01** |
| `2->1` | 18.93 | **19.32** |
| `1+2->3` | 1190.30 | **1190.12** |

**根因**（`rulbench/evaluation/inference.py`）：`predict_with_uncertainty`（L157–168）先 `_stochastic()`
再抽 **`evaluation.mc_samples = 50`** 次**随机 dropout 前向**取均值 → 预测依赖**全局 torch RNG 状态**；
而 `seed_everything` **只在 `train_one()` 里调用，评测路径从不播种**。
- 因此：训练过的进程「顺带」把 RNG 置为确定态 → `scripts/17` 的「**train→eval 全链**」测得 **0.0%**（该结论对全链仍成立）；
- 但 **纯再评测（复用 checkpoint、不训练）没有播种**，RNG 来自进程启动的 OS 熵 → **不可复现**（约 **0.05–2%/格**）。

**修复（3 行）**：`predict_seed(..., seed=...)` 在 MC-Dropout 循环**之前**调用 `seed_everything(seed)`；
`predict_ensemble` 把该 seed 传下去 → 预测成为 `(checkpoint, data, seed)` 的**纯函数**。

**验证**：同一 fold1 跨工况在**两个独立进程**中重算 → **逐位一致（36/36 行）** ✔。

**影响与披露（新增）**：
- A7 的「0.0% run-to-run 极差」**只对 train→eval 全链成立**；**由 checkpoint 纯再评测另有一层 0.05–2%/格 的抖动**——
  修复前从未披露。修复后两层都归零。
- 所有 `matched_fold*` 跨工况报告已在**修复后**重算，故与论文的 train→eval 产物**同属一个可复现口径**。
- **建议新增披露规则 A10**：*评测路径的 MC-Dropout 采样已播种；由 checkpoint 再评测与完整重训给出相同数字（逐位）。*
- 这是本次审核的**额外产出**：不在 B2–B5 清单内，但属同一「可复现性/口径」家族，且不修则 §4 的重算本身不可复现。

---

## 5. B5 —— `11_make_figures.py` 的 MAE/RMSE 混用（已修 + 已重绘）

`scripts/11_make_figures.py::analytic_anchors`（L99–L101）原先

```python
one    = life * np.sqrt((ratio - 1.0) ** 2)          # elementwise |误差|
best_c = life * np.sqrt((ratio - ratio.mean()) ** 2)
out[...] = {"const_one": one.mean(), "best_const": best_c.mean()}   # -> 这是 MAE
```

即**先取绝对值再求均值** → 报的是 **MAE**，却被当作 `best_const` 的 **RMSE** 地板（低估约 13%）。
现改为 `sqrt(mean((life*(ratio-c))**2))`。

**验证**（`artifacts/data_linear`，重跑后）：

| 工况 | `const_one`（RMSE） | `best_const`（RMSE，修复后） | 修复前（误为 MAE） |
|---|---|---|---|
| cond1 | 41.445 | **6.055** | 5.24 |
| cond2 | 204.806 | **88.911** | 77.00 |
| cond3 | 75.881 | **23.958** | 20.75 |

与 `audit_02` §4 预告的 6.06 / 88.91 / 23.96 **完全吻合**。`fig1–fig4` 已重绘。

---

## 6. 顺序决策与门禁

### 6.1 采纳「读法 B（按阶段映射）」

- A4 的 FAIL（训练量不可比）**阻塞的是 Stage 3 的因果结论**，**不阻塞 Stage 2（单工况 / 标签结构）**；
- 因此 Stage 2 可先行，但**冻结一条禁令**：

> **在训练量匹配臂（源工况训练窗口下采样到与目标同量 / 统一 K=min）完成之前，
> 任何 `R_cross` / `ΔRMSE` 的表述不得含因果措辞（「由于工况迁移」等），H5 记为「无法判定」。**

### 6.2 本轮已清账的阻塞项

| 编号 | 状态 |
|---|---|
| **B2** `ΔRMSE` 缺失 | ✅ 已补（三处脚本）+ 重算 |
| **B3** 统计单位 + 折间依赖 | ✅ 已冻结协议 + 新增 `scripts/23` + 数值产出 |
| **B4** `report.py` 区间/方差不同源 | ✅ 已修 + `matched_fold{1..5}/cross` 已重算 |
| **B5** `11_make_figures` MAE 当 RMSE | ✅ 已修 + 图已重绘 |
| **B8**（新发现）纯再评测不可复现（MC-Dropout 未播种） | ✅ 已修（`predict_seed` 播种）+ 已验证逐位一致 |

### 6.3 仍留待处理（随后续阶段）

| 编号 | 事项 | 归属 |
|---|---|---|
| **B1** | 训练量匹配臂（三种训练方式） | **Stage 3 核心实验** |
| **B6** | `manuscript.md` §6.6 Table 7 换成配对数据源 | Stage 5 文件级重构（**未改论文**） |
| **B7** | `LI(h)` 适用条件 + 命名降级（6 处文档） | 纯文字，Stage 5 |
| 新增 | 早于 B4/B8 修复的 `results_*.csv`（`single`、`single_linear`、`cross_uda_linear`、`cross_source_only_linear`、`sweep_h*`、`matched_fold*` 的**单工况**臂）其 `picp`/`crps` 口径仍属修复前；引用前须重算 | 引用前 |
| 新增 | **全部 Stage 1 前的训练产物须重跑**（确定性修复时点 2026-10-07 09:44 之前） | Stage 2/3 开工时 |
| 新增 | `artifacts/reports/matched_rcross.json`（单折试点）的**输入目录已不存在**（`repro_single_fixed` / `repro_matched_fixed` 已不在 `artifacts/reports/`）→ 该 JSON 目前**无法从仓库状态复现**；若论文引用其值须重建试点或改用 `matched_fold*` | 引用前 |

---

## 7. 对 `audit_05` 与 `STAGE1_AUDIT_REPORT` 的修订

已就地更新（见两文件内的「Round-1 修订」标注）：

- `audit/audit_05_cross_protocol.md`：§4.2 增补单元级结果并**把 pooled p 降级为探索性**；§6 缺陷 2 标记为「协议已冻结 + 已实现」；§10.2/§11 更新。
- `audit/STAGE1_AUDIT_REPORT.md`：§2.4**改写**（不再用 pooled p 论证 H4）；§3 的 D5 已解决；§5 的 P9 升级为「已冻结并实现」；§6 的 B3 标记 resolved。

---

*End of REVIEW_ROUND1_RESPONSE.md*
