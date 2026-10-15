# audit_05_cross_protocol.md —— Audit 5：测试对象与比较协议

| 项 | 值 |
|---|---|
| 审计对象 | 单/跨工况的**测试对象**是否相同；`R_cross` / `ΔRMSE` 是否按同一规则计算；五折的统计单位与折间依赖 |
| 涉及代码/产物 | `scripts/18_matched_incondition_vs_cross.py`、`scripts/15_cross_transfer.py`、`artifacts/reports/matched_rcross.json`、`artifacts/reports/matched_fold_rcross.json`、`artifacts/reports/matched_comparison.json`、`configs/cross_matched_linear.yaml`、`paper/manuscript.md` §6.6 |
| 审计日期 | 2026-10-07 |
| 结论 | **MODIFY**（协议已具备可比性，但报告口径与 ΔRMSE 缺失须补；并存在「manuscript 用未配对结果」的一致性缺陷） |

---

## 1. 两个量的定义（V2.0 §1.3(c) 要求）

```
R_cross  = RMSE_cross / RMSE_single              配对：同测试轴承、同窗口、同评价规则
ΔRMSE    = RMSE_cross − RMSE_single              「必须并列」
```

**前置条件**（缺一即不可比）：两臂必须评在**同一测试总体**上（同轴承、同窗口、同误差定义）。

---

## 2. 测试对象配对：**已做到**（配对配置）

`configs/cross_matched_linear.yaml` L54–L58：

```yaml
splits:
  cross:
    source_train: [1, 2, 3]   # matched to splits.single.train
    source_val:  [4]          # matched to splits.single.val
    target_test: [5]          # matched to splits.single.test
```

该文件的 L6–L24 注释已把设计意图写死：

| 臂 | train | val | test |
|---|---|---|---|
| in-condition | 目标条件轴承 1–3 | 轴承 4 | **目标条件轴承 5** |
| cross-matched | 源条件轴承 1–3 | 轴承 4 | **目标条件轴承 5** |

**即「唯一变化的因子 = 训练数据来自哪个工况」。** 数值验证（`audit_04` §2.1）：
9/9 任务的跨工况 test 集与同目标单工况 test 集**逐位相同**（同 `condition`/`bearing_id`/`window_end`/`ratio`）。✔

---

## 3. 三类结果的**口径差异**（必须在论文中区分）

现有产物有两类，**口径不同**，混用即不可比：

| 产物 | 模型训练时的配置 | 评测总体 | 备注 |
|---|---|---|---|
| `artifacts/reports/matched_comparison.json`（`scripts/18…`） | **旧未配对跨工况**（`cross_uda_linear` / `cross_source_only_linear`，`target_test: [1..5]`） | **事后只用轴承 5**（`18` L146–L151、L156–L163） | 纯再分析（`18` L27）；**模型的 MK-MMD 目标集仍是 5 个轴承** |
| `artifacts/reports/matched_rcross.json`、`matched_fold_rcross.json`（源自 `repro_matched_fixed`、`matched_fold*`） | **真配对配置**（`cross_matched_linear.yaml`） | 轴承 5（= `target_test: [5]`） | 训练与评测**都**是配对口径 |

**关键差异**：`scripts/18…` 是**事后重聚合**（把旧模型的 per-bearing 指标只留轴承 5），
但**旧模型的 UDA 臂在训练时见过 5 个目标轴承的输入**（`protocols.py` L65 返回 `test["X"]` 全量）。
真配对配置下 UDA 臂只见过**轴承 5** 的输入。

→ **结论**：`18` 的结果**比未配对聚合可比得多**，但**不等于**真配对实验；
**论文的 `R_cross` 必须取自真配对产物**（`matched_rcross.json` / `matched_fold_rcross.json`）。

---

## 4. 配对结果数值（真配对口径）

### 4.1 三重复（`matched_rcross.json`，`repro_matched_fixed` × 3 repeat）

27 个 cell（3 target × 3 model × 3 repeat）：

| 口径 | min | **中位数** | max |
|---|---|---|---|
| **未校正（uncorrected）** | 0.867 | **1.245** | 2.365 |
| 校正（corrected） | 0.909 | 1.249 | 2.684 |

逐目标工况的**符号构成**（`matched_rcross.json` → `metrics.*.cells.*.cells_below_one / n_cells`）：

| target | `cells_below_one` / `n_cells`（未校正） | 含义 |
|---|---|---|
| cond1 | **0 / 3** | 跨工况**全部更差**（`R>1`），且训练量也更多（18.35×） |
| cond2 | **1 / 3** | 混合 |
| cond3 | **2 / 3** | 跨工况多数**更好**，且训练量更少（0.07×–0.27×） |

→ **无稳定方向**：`R_cross` 跨越 1，符号随目标工况翻转。**与 H4「未发现显著系统性差异」一致。
不得写成「等效」。**

### 4.2 五折（`matched_fold_rcross.json`，5 折 LOBO × 9 任务 = 45 cells）

| 口径 | pooled 中位数 | win rate（跨工况更优） | IQR | n | Wilcoxon p |
|---|---|---|---|---|---|
| **未校正** | **1.0639** | 0.40 | 0.5095 | 45 | **0.1530** |
| 校正 | 1.2503 | 0.20 | 0.5532 | 45 | （见文件） |

逐折中位数（未校正）：`[0.889, 0.958, 1.274, 1.256, 0.984]` —— **跨折跨越 1**。

**核对**：V2.0 §1.7 引用「`median(R_cross)=1.064, p=0.153`」→ 与本文件**完全一致** ✔。
即 **V2.0 用的是真配对五折结果**。

> **Round-1 修订（2026-10-07）**：上表的 pooled `p=0.153` **不得**用作 H4 的正式统计证据——
> 45 个 cell **不是 45 个独立样本**（5 折共享训练轴承；每单元内的 3 个源任务变体是重复测量）。
> 统计单位已冻结为**测试轴承实例（n = 15 = 5 折 × 3 目标工况 = 全部 15 个轴承）**，
> 主证据改为**分布 + 效应量 + 轴承层级自助 CI**。单元级结果
> （`scripts/23_stat_unit_rcross.py` → `docs/stat_unit_rcross.md`）：
> 未校正中位 `R_cross` = **1.086**、95% CI **[0.889, 1.276]（跨 1）**、跨工况更优 **33%**、
> 中位 `ΔRMSE` **+3.44 min**（CI [−1.81, +12.62]，跨 0）；折中位 >1 者 **3/5**；
> \|median−1\| **8.6%** ≪ 折间跨度 **42.3%**；单元级 `p = 0.252`（仅 indicative）。
> （数字为 B8 修复后、可逐位复现的口径；见 `REVIEW_ROUND1_RESPONSE.md` §4.1。）
> 协议：`audit/STATISTICAL_UNIT_PROTOCOL.md`；处置：`audit/REVIEW_ROUND1_RESPONSE.md`。
> ⚠ 口径提示：本表「逐折中位数」是**每折 9 个 cell** 的中位数；`scripts/23` 的逐折中位数
> 是**每折 3 个单元**（单元 = 该目标工况内 3 任务的 median）的中位数，两者定义不同、不必相等。

---

## 5. 缺陷 1：`ΔRMSE` **完全缺失**

全仓库检索 `delta_rmse` / `ΔRMSE` / `rmse_delta`：
**没有任何计算 `RMSE_cross − RMSE_single` 的代码或字段。**

检索到的 `correction_delta_rmse_pct`（`rulbench/evaluation/report.py` L81–L83）
是「**校正 vs 未校正**」的相对变化，**不是**「跨工况 vs 单工况」的绝对差：

```python
"correction_delta_rmse_pct": float(
    100.0 * (metrics_uncorrected["rmse"] - metrics["rmse"]) / (metrics_uncorrected["rmse"] + 1e-12)
)                                                                    # 同臂内，校正前后
```

**判定**：V2.0 §1.3(c)「**必须并列** `ΔRMSE`」**未实现** → **FAIL**。
**理由**：`R_cross` 是无量纲比值，在长寿命与短寿命任务上不可直接聚合；
`ΔRMSE`（分钟）才能表出实际量级差。缺它则 §6.6 的表述无法完整。

> **Round-1 修订：已补。** `ΔRMSE = RMSE_cross − RMSE_single`（分钟）现已在
> `scripts/23_stat_unit_rcross.py`（单元级 `median_dRMSE` + CI）、
> `scripts/21_fold_rcross.py`（池化 `median_delta_rmse_min` + 每 cell 的 `d`）、
> `scripts/20_matched_rcross.py`（每 (target,model) `median_delta_rmse`）产出。
> 重算值（B8 修复后）：未校正 **+3.44 min**、校正 +13.46 min；`R_cross` 池化中位数与 p 回到 1.0639 / 0.1530。
> 见 `audit/REVIEW_ROUND1_RESPONSE.md` §3。

> **注**：`scripts/18…` 的 `delta_pct`（L201：`100*(ratio−1)`）是**比值的百分比**，
> 也不是绝对 `ΔRMSE`，不能替代。

## 6. 缺陷 2：五折的**统计单位与折间依赖**未处理

`matched_fold_rcross.json` 报：

```
"n": 45,  "pooled_median": 1.0639,  "wilcoxon_p": 0.1530
```

**问题**：

1. `n = 45` = 5 折 × 9 任务，但 **5 折共享同一批轴承数据**（LOBO 的不同折只是换了留出轴承）；
   把 45 个 cell 当独立样本做 Wilcoxon → **p 值的独立性假设不成立**；
2. **统计单位**（轴承还是窗口）未在代码或文档中声明；
3. V2.0 C4 明确要求：「不能只报一个汇总 p 值，也不能把五折当作五个独立数据集——统计推断须考虑
   **折间共享数据造成的依赖性**，并**明确统计单位是轴承还是窗口**」。

**判定**：**FAIL**（与 `audit_03` A3-j 同一问题，此处从协议角度复述）。
未处理的后果：`p = 0.153` 应标注为「**未考虑折间依赖**的参考值」，或改用轴承层级重采样/分层模型。

> **Round-1 修订：已解决（判定改为 RESOLVED）。** 统计单位已冻结为**测试轴承实例（n=15，5 折 × 3 目标工况 = 全部 15 个轴承）**，
> pooled `p` 降级为 `EXPLORATORY_ONLY`，主证据改为**分布 + 效应量（`R_cross` & `ΔRMSE`）+ 轴承层级自助 CI**，
> 并按目标工况分层。协议见 `audit/STATISTICAL_UNIT_PROTOCOL.md`（S1–S10），实现见 `scripts/23_stat_unit_rcross.py`。
> **口径错配警示**：校正口径把跨工况臂 ×1.3–2.5 而单工况臂 ×1.0，故其「跨工况更差」是口径产物（规则 A9），不得作主证据。

## 7. 缺陷 3：`scripts/18…` 的 **best-of-source 选择**引入乐观偏置

`scripts/18_matched_incondition_vs_cross.py::cross_gain_table` L185–L189：

```python
best_task, best = None, float("inf")
for task in tasks_by_target.get(tgt, []):
    bmap = cross[arm].get(task, {}).get(m, {})
    if 5 in bmap and bmap[5]["rmse"] < best:
        best_task, best = task, bmap[5]["rmse"]      # 取 3 个源工况里最差的…还是最好的？
```

代码取的是**最小 RMSE 的源任务**（`< best`），即 **oracle 选出最好的源变体**。
docstring L172–L174 自述这是「most conservative choice against the 'cross is worse' claim」——
即**刻意偏向跨工况**。

**判定**：这是**有意的保守设计**，不是错误；但必须与
`matched_fold_rcross.json`（报**全部 cell**、不挑最好源）**区分开**：
两者回答不同问题（前者「最好的跨工况能否赢」、后者「跨工况整体分布如何」）。
**论文若引用 18 的 R_cross，必须写明「best source per (target, model)」。**

## 8. 缺陷 4：`paper/manuscript.md` §6.6 的表**不是配对结果**

`paper/manuscript.md` L500–L510 的 Table 7，例：

```
| 1+2→3 | 642.2 | 653.0 | 946.6 | … |
```

- `642.2` = `cross_transfer_summary.json` 的 `oracle_const_task__rmse = 642.1947`；
- 该文件的 `1+2->3` 记录 **`n_windows: 6879`、`n_bearings: 5`** →
  **未配对口径**（`artifacts/data_linear`，`target_test: [1..5]`）；
- 配对口径下 `1+2→3` 的 `n_test` 只有 **83**（= 轴承 5），oracle 应为 **23.96**（见 `audit_02` §4）。

**判定**：§6.6 的 headline 表与 V2.0 的配对要求 **不一致** → **MODIFY**
（V2.0 §4.2 已把 §6.6 列为「重写 + 重跑」，本审计为该处置提供**证据**）。

> 同时注意：`docs/matched_cross_vs_incondition.md` L106 仍把 `946.63` 与 `35.32` 并列
> → 该文档也是**未配对**产物，须随稿重写。

## 9. 两个必须写入方法章的「比较规则」

| # | 规则 | 依据 |
|---|---|---|
| R1 | `R_cross` 与 `ΔRMSE` **必须并列**，并**同时给出 `n_train`**（否则不能把差异全归因于工况迁移） | V2.0 §1.3(c)、C6；`audit_04` §3 |
| R2 | 单/跨工况比较**只能用配对产物**；`best-of-source` 的 `R_cross` 须明确标注选择规则 | §3、§7 |

---

## 10. 审计结论

| 子项 | 判定 | 依据 |
|---|---|---|
| A5-a 配对配置存在且设计正确 | **PASS** | `cross_matched_linear.yaml` L6–L24、L54–L58 |
| A5-b 配对配置下测试对象逐位相同 | **PASS** | `audit_04` §2.1（9/9） |
| A5-c `R_cross` 可按配对口径计算 | **PASS** | `matched_rcross.json`、`matched_fold_rcross.json` |
| A5-d `R_cross` 无稳定方向（符号随目标翻转） | **PASS**（支持 H4 的「未发现显著系统性差异」） | §4.1、§4.2 |
| A5-e `ΔRMSE` 与 `R_cross` 并列 | ~~FAIL~~ → **RESOLVED**（Round-1） | §5 |
| A5-f 统计单位 + 折间依赖已处理 | ~~FAIL~~ → **RESOLVED**（Round-1） | §6；V2.0 C4 |
| A5-g `best-of-source` 选择规则已标注 | **PASS（代码已自述）/ 论文须落实** | §7 |
| A5-h 论文用的表与配对口径一致 | **FAIL** | §8；§6.6 Table 7 用的是未配对结果 |

### 10.1 总判定：**MODIFY**

- **协议本身是对的**：配对配置把单/跨工况的差异压缩到「训练数据来源」这一个因子，
  这是本项目相对旧稿的**关键改进**；
- 但**报告层**有四处必须补/改：
  1. **补 `ΔRMSE`**（分钟）并与其 `R_cross` 并列（P0，影响 §6.6 全表）；
  2. **声明统计单位 = 轴承** + 处理折间依赖（P1，影响所有 p 值与 CI）；
  3. **区分**「事后重聚合（`18`）」与「真配对（`matched_*`）」两类产物；
  4. **换掉** §6.6 Table 7 的未配对数字（P0，V2.0 §4.2 已列）。
- **H4 的当前证据支持「未发现显著系统性差异」**（未校正 pooled 中位数 1.064、p = 0.153、
  win rate 0.40、符号随目标翻转）。**不得**升级为「等效」（措辞纪律，`stage0_scope_freeze.md` §2.1）。

### 10.2 对 Protocol Freeze 的输入

见 `STAGE1_AUDIT_REPORT.md` 的 **P4（测试对象）、P7（R_cross + ΔRMSE + n_train）、P3（统计单位）**。

> **Round-1 修订**：其中 **P3（统计单位）** 已由 `audit/STATISTICAL_UNIT_PROTOCOL.md` 冻结并实现
> （单位 = 轴承实例 n=15；p 降级；主证据 = 分布 + 效应量 + 轴承自助 CI）；
> **P7 的 `ΔRMSE`** 已补齐。**P4/P6（训练量匹配）** 仍留 Stage 3。
> 见 `audit/REVIEW_ROUND1_RESPONSE.md`。

---

## 11. 未核实项

| 项 | 状态 |
|---|---|
| `matched_fold_rcross.json` 的 **corrected** 臂完整统计（pooled 中位数/IQR/p） | ✅ **已核实（Round-1）**：corrected pooled median 1.2503、p=2.29e-6、median dRMSE +13.46 min（`scripts/23` 复核） |
| 五折的**折定义**（`configs/matched_folds/fold1…5.yaml`）是否真的覆盖 5 个不同留出轴承 | ✅ **已核实（Round-1）**：fold`k` 的 `test:[k]`，k=1…5 → 5 个不同留出轴承；`single` 与 `cross` 两臂的 train/val/test 轴承索引**逐折相同** |
| `scripts/18…` 输出是否已随稿更新（`docs/matched_cross_vs_incondition.md` 的 `946.63`） | **仍未核实是否已被覆盖**（属 Stage 5 文件级重构，见 B6） |

*End of audit_05_cross_protocol.md*
