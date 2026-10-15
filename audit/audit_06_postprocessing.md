# audit_06_postprocessing.md —— Audit 6：后处理、缩放与评价指标的对称性

| 项 | 值 |
|---|---|
| 审计对象 | ① median-ratio 校正是否只作用于一个臂；② 区间/方差量是否与均值同源；③ 指标族是否有两套口径；④ 图中锚点与柱是否同一量纲 |
| 涉及代码 | `rulbench/evaluation/inference.py`、`rulbench/evaluation/report.py`、`rulbench/pipelines/run.py`、`rulbench/evaluation/metrics.py`、`rulbench/models/trivial.py`、`scripts/15_cross_transfer.py`、`scripts/11_make_figures.py` |
| 审计日期 | 2026-10-07 |
| 结论 | **MODIFY**（不对称已识别且已被 V2.0 排除为主要证据；但存在 3 处实现级不对称须修） |

---

## 1. 后处理链（代码事实）

### 1.1 神经网络臂 —— `report.py::evaluate_dataset`

| 行 | 内容 |
|---|---|
| L43 | `is_cross = artifact["meta"].get("protocol") == "cross"` |
| L48–L49 | `val_true, test_true = ratio × lifetime`（分钟） |
| L51 | `val_pred_min = ratio_to_minutes(val_pred["mean"], val_life)` |
| **L52** | `correction_enabled = bool(cfg…correction.enabled) and is_cross` ← **只有跨工况才可能校正** |
| L53–L61 | `factor = median_scaling_factor(val_true, val_pred_min) if correction_enabled else 1.0` |
| L63 | `test_mean_min = ratio_to_minutes(test_pred["mean"], test_life)` |
| L64 | `corrected_min = clip_to_lifetime(test_mean_min * factor, test_life)` |
| L65 | `uncorrected_min = clip_to_lifetime(test_mean_min, test_life)` |
| L70–L71 | `metrics` 用 `corrected_min`；`metrics_uncorrected` 用 `uncorrected_min` |

### 1.2 平凡臂 —— `run.py::evaluate_trivial`

| 行 | 内容 |
|---|---|
| L34 | `model.fit(train["X"], train["ratio"])` ← `constant_mean` 用**训练集**比值均值（`trivial.py` L33–L34） |
| L45–L48 | `is_cross` 为真且 `correction.enabled` 才估 `factor`；否则 `factor = 1.0`（L46） |
| L50–L52 | 同 `report.py` 的 `corrected` / `uncorrected` 双口径 |

### 1.3 校正因子本身

`rulbench/evaluation/inference.py::median_scaling_factor`，L101–L115：

```
factor = median( y_true[valid] / y_pred[valid] )        # L112
bounds = (0.1, 10.0)                                    # L104，越界即返回 1.0（L113–L114）
```

`ratio_to_minutes`（L118–L120）与 `clip_to_lifetime`（L123–L124）分别是「× lifetime」与「clip 到 [0, lifetime]」。

**数值核对（单工况）**：`artifacts/repro_single_fixed/repeat{1,2,3}/results_full.json` 中
所有记录的 `correction_delta_rmse_pct` **全为 0.0** → 单工况确实 `factor = 1.0`。✔
（跨工况同字段取值散布于 `−188.9% … +64.4%`，如 `cross_uda_linear/results_full.json`。）

---

## 2. 不对称 1（**最重要**）：median-scaling 只作用于跨工况臂

由 §1 可见：

| 臂 | 后处理 |
|---|---|
| 单工况 | **恒不校正**（factor ≡ 1.0） |
| 跨工况 | 校正（factor = 源域 val 上的 `median(true/pred)`） |

**后果（V2.0 §1.7 排除条款 ①、C7）**：

1. **两臂不同质** → 校正后的 `R_cross` 不能作主要证据；
2. **校正抹平基线阶梯**：对常数预测 `c`，`factor = median(val_true)/(c·L)`（因 `pred = c·L` 恒定，
   `median(true/pred) = median(true)/(c·L)`），故校正后预测
   = `c·L · median(val_true)/(c·L)` = **`median(val_true)`** —— **与 `c` 无关**。

   `scripts/15_cross_transfer.py` 的文档字符串 L9–L14 已把这条写成正式性质：

   > 「the correction divides the constant back out, so **every** constant collapses to the same
   > prediction, `median(val_ratio) * lifetime`. That is exactly why `constant_one` and
   > `constant_mean` come out numerically identical in the cross tables -- not a bug, a property.」

   `paper/manuscript.md` L480–L487 亦复述（「agree to four decimals on all nine tasks in the
   corrected table while differing substantially in the uncorrected one」）。

3. **源域因子被搬到目标域**：`factor` 在**源域 val** 上估（`report.py` L48、L54），
   却作用在**目标域 test**。`paper/manuscript.md` L531–L533 记录其**不稳定**：
   「observed range 0.21–5.60」，最大放大 5.6× 后再 clip。

**判定**：**不对称确实存在**；V2.0 已明令其**不得作为主要证据**（一致）。
**必须**：主表并列**未校正**口径（A4 双口径），且任何跨工况结论以未校正为主。

---

## 3. 不对称 2：区间与方差量**不同源**

`rulbench/evaluation/report.py` L66–L68：

```python
lower = clip_to_lifetime(ratio_to_minutes(test_pred["lower"], test_life) * factor, test_life)   # L66
upper = clip_to_lifetime(ratio_to_minutes(test_pred["upper"], test_life) * factor, test_life)   # L67
std_min = ratio_to_minutes(test_pred["std"], test_life)                                          # L68
```

- **L66–L67**：区间**乘了 `factor`**；
- **L68**：`std_min` **没有乘 `factor`**。

于是 `evaluate_predictions(test_true, corrected_min, lower, upper, std_min)`（L70）里：

| 量 | 是否含 `factor` |
|---|---|
| `lower` / `upper`（→ `picp`、`mpiw`） | ✔ 含 |
| `std_min`（→ `crps`，`metrics.py` L253–L254） | ✘ 不含 |

→ **当 `factor ≠ 1` 时，区间与 CRPS 用的方差不同源**，`crps` 与 `picp/mpiw` **不自洽**。
（单工况 `factor = 1.0`，故问题只在跨工况出现。）

**更严重的一处**：L71

```python
metrics_uncorrected = evaluate_predictions(test_true, uncorrected_min, lower, upper, std_min)
```

用 `uncorrected_min`（**未**校正的均值）配 `lower/upper`（**已**校正的区间）→
**「未校正」口径的 `picp`/`mpiw` 实际用的是校正后的区间**，
`correction_delta_rmse_pct`（L81–L83）也因此只反映 RMSE，而 PICP 的「校正前后差」是错的。

**判定**：**FAIL（实现级）**。修法：`std_min` 同步乘 `factor`；
未校正口径应使用**未乘 factor** 的 `lower/upper`。

---

## 4. 指标族的对称性

### 4.1 两套 PHM 口径 —— 必须并列

`rulbench/evaluation/metrics.py`

| 行 | 函数 | 相对误差口径 |
|---|---|---|
| L71–L78 | `phm_score_legacy` | 相对误差取**分数**（`err = (t−p)/t`） |
| L81–L88 | `phm_score_official` | 相对误差取**百分数**（`err = 100·(t−p)/t`，L87） |
| L62–L68 | `_phm_from_relative` | `err>0`（低估）用 `/20`，`err<0`（高估）用 `/5` → **本身即不对称** |

`evaluate_predictions` 同时输出 `phm_legacy` 与 `phm_official`（L247–L248）。
**判定**：**已实现双报**（符合 A4）✔；但 **PHM 分数固有不对称**（高估罚 4×），
必须在方法章声明，且**不得**与对称指标（RMSE/MAE）混为同质证据。

### 4.2 `nrmse` 的分母依赖测试总体

`metrics.py::nrmse`，L45–L58：`mode="range"` 时分母 = `y_true.max() − y_true.min()`。
该量**随测试总体变化**（不同轴承集合 → 不同极差）→ `nrmse` **不可跨协议横比**。
**判定**：须在方法章声明 `nrmse` 只在本协议内可比。

### 4.3 RMSE / MAE 是**对称**的

`rmse`（L35–L37）、`mae`（L40–L42）均为对称量、分钟空间、无权重 → **可作为主指标** ✔。
与 `audit_02` 的结论一致（`best_const` 定义在分钟空间无权重 RMSE 上）。

---

## 5. 不对称 3：图 3 的 hurdle 与柱**不同量纲**（跨引 Audit 2）

`scripts/11_make_figures.py`

| 行 | 内容 |
|---|---|
| L83–L84 | 柱高来自 `results_corrected.csv` 的 `rmse`（**RMSE**） |
| L99–L101 | hurdle 用 `analytic_anchors` 的 `best_const` = `mean(life·|r−mean|)`（**MAE**） |
| L242–L243 | `ax.plot(x, hv, "D", …)` 把该值画成「best constant (hurdle)」 |

**后果**：同一张图里柱是 RMSE、点是 MAE → **hurdle 被系统性低估约 13.4%**
（`audit_02` §5 已量化：6.06→5.24、88.91→77.00、23.96→20.75），视觉上放大模型对地板的领先。

**判定**：**FAIL（实现级，影响交付图 `paper/figures/fig3_main_results.{png,pdf}`）**。

---

## 6. 正确的地方（记录以界定问题范围）

| 项 | 位置 | 说明 |
|---|---|---|
| 双口径（校正/未校正）都计算并存 | `report.py` L70–L71；`run.py` L51–L52 | A4 要求满足 ✔ |
| 校正因子只在**源域 val** 上估 | `report.py` L48、L54 | 不看目标标签 ✔ |
| 单工况强制 `factor = 1.0` | `report.py` L52；`run.py` L46 | 消除了一种更坏的不对称 ✔ |
| 预测被 clip 到 `[0, lifetime]` | `inference.py` L123–L124 | 防止超额预测 ✔ |
| `scaling_factor` 被写进每条结果 | `report.py` L78；`run.py` L59 | 可审计 ✔ |
| 校正越界回退 1.0 | `inference.py` L113–L114 | 鲁棒 ✔ |
| 三个平凡基准在 `run.py` 与神经臂走**同一套** eval 路径 | `run.py` L30–L52 vs `report.py` L30–L99 | 口径统一 ✔ |

---

## 7. 审计结论

| 子项 | 判定 | 依据 |
|---|---|---|
| A6-a median-scaling 只在跨工况臂生效 | **已识别（V2.0 已排除其为主要证据）** | `report.py` L52；`run.py` L46；C7 |
| A6-b 校正抹平基线阶梯 | **已识别（代码与论文均已声明为「性质」）** | `15` L9–L14；`paper` L480–L487 |
| A6-c 校正因子跨域搬运且不稳定（0.21–5.60） | **须披露** | `paper` L531–L533 |
| A6-d 双口径（校正/未校正）均报 | **PASS** | `report.py` L70–L71 |
| A6-e `std_min` 与 `lower/upper` 同源 | **FAIL** | §3 |
| A6-f 「未校正」口径使用未校正区间 | **FAIL** | `report.py` L71 |
| A6-g PHM 双口径并列 | **PASS** | `metrics.py` L247–L248 |
| A6-h PHM 分数固有不对称已声明 | **PARTIAL**（代码有注释，论文须写） | `metrics.py` L62–L68、L5–L15 |
| A6-i `nrmse` 分母随总体变化，须限制可比范围 | **须声明** | `metrics.py` L52–L53 |
| A6-j 图 3 的 hurdle 与柱同量纲 | **FAIL** | §5；`11` L99–L101、L242–L243 |

### 7.1 总判定：**MODIFY**

- **核心的不对称（A6-a/b）是「已知且已处置」的**：V2.0 已把不对称后处理排除出主要证据，
  并要求双口径并列 → **该风险关闭**；
- **但三处实现级不对称须修**（不修则跨工况的 PICP/CRPS/图 3 不可用）：

| 编号 | 缺陷 | 严重度 | 影响交付物 |
|---|---|---|---|
| **F1** | `std_min` 未随 `factor` 缩放 → CRPS 与区间不自洽 | **高** | 跨工况 `crps` |
| **F2** | 「未校正」口径误用已校正区间 → PICP 的校正前后差失真 | **高** | 跨工况 `picp` / `correction_delta_rmse_pct` 的解读 |
| **F3** | `11_make_figures.py` 用 MAE 当 RMSE 画 hurdle | **中**（仅图） | Fig 3 |

### 7.2 对 Protocol Freeze 的输入

见 `STAGE1_AUDIT_REPORT.md` 的 **P8（后处理规则）、P9（指标清单与可比范围）**。

---

## 8. 未核实项

| 项 | 状态 |
|---|---|
| `cross_uda_linear` / `cross_source_only_linear` 的 `constant_one` 与 `constant_mean` 在**校正口径下逐位相同** | **未逐位数值复核**（docstring `15` L9–L14 与 `paper` L480–L487 均已断言；本次未实跑比对） |
| `metrics.py` 中 `crps`（L104–L116）是否在 `std=0` 的平凡基准上退化 | **未核实**（L109 有 `maximum(std, _EPS)` 保护，但未数值检验） |
| `results_full.json` 中 `picp` 的具体数值是否已受 F2 影响 | **未核实**（须重算或读现成字段比对） |

*End of audit_06_postprocessing.md*
