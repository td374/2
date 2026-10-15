# STAGE5_RESULTS.md —— Stage 5 重跑结果与证据集收敛（出口文件）

> **依据**：用户 2026-10-08 指令「选 A，结果根 `artifacts/reports/s5/`；`s2_chain` 作确定性命中校验基准；
> 若 S5 proposed 与其不能逐位一致则立即停止」。本文件记录 Stage 5 的执行结果、内建校验结论与正式数字收敛。
>
> **入口门槛**：`audit/STAGE5_PROTOCOL_CHECK.md`（5 组协议检查全 PASS）。
> **参照协议**：`audit/STAGE1_PROTOCOL_FREEZE.md`（F1–F5，红线 1–23）、`audit/STAGE5_PROTOCOL_CHECK.md`（红线 24–27）。

| 项 | 值 |
|---|---|
| 执行日期 | 2026-10-08 |
| 结果根 | `artifacts/reports/s5/`（全新，未覆盖任何失效目录） |
| 确定性校验 | **PASS** —— S5 proposed(linear) 与 `s2_chain/linear` 逐位一致（见 §2） |
| 正式链 | `09 → 10 → 12 → 11 → 13` EXIT=0 |
| 结论 | 证据集已收敛到单一修复后口径；§6.1–6.5 全部数字来自本套结果 |

---

## 1. 执行范围

| 块 | 内容 | 目录 | 状态 |
|---|---|---|---|
| §6.2 主表 | XJTU-SY single，linear，6 深度 + ridge + 2 trivial | `s5/single_linear/<model>/` | 完成 |
| §6.3 h 谱 | XJTU-SY single，h∈{0.5..0.9}，proposed + gru + const_mean | `s5/sweep_h{h}/` | 完成 |
| §6.1 参考 | XJTU-SY single，h=0.8，**全深度阵容**（新增 `h08full` 块） | `s5/sweep_h0.8/` | 完成 |
| §6.4 PHM | PHM2012 single，linear | `s5/phm2012_single_linear/` | 完成 |
| §6.5 不确定性 | 派生自 §6.2 / §6.4 的 picp/mpiw/crps | — | 完成 |
| UDA | `cross_uda_linear` / `cross_source_only_linear` | — | **未跑**（用户指示，留给后续） |

> `h08full` 块的存在理由：§6.1 断言「h=0.8 时**没有任何深度模型**能击败平凡基线」，
> 而原 `sweep_h` 块只训练 proposed + gru。为使该负面断言有**全阵容**证据，本轮补跑
> CNN / LSTM / Informer-SE / BiGRU 于 h=0.8（不覆盖既有 h=0.8 结果）。

---

## 2. 内建确定性校验（门槛条件）

按用户指令：S5 proposed 必须与 `s2_chain/linear` **逐位一致**，否则立即停止，不出正式表图。

| 对比项 | 值 | 判定 |
|---|---|---|
| S5 `single_linear/se_resnet_bigru` vs `s2_chain/linear` | 逐位一致（rmse/mae/nrmse/phm_legacy/phm_official/picp/mpiw/crps/scaling_factor 全列） | ✅ PASS |
| trivial 基线（const_one / const_mean）vs 修复前 | 逐位一致（数据未变的哨兵） | ✅ PASS |
| `09_label_baseline_anchors.py` 解析 cross-check | 3 anchor 全 PASS | ✅ PASS |

→ **确定性已闭合**：训练侧（09:44 修复）与评测侧（B8 16:21 修复）的整链可复现，极差 0.0%。
→ 校验通过，允许出正式表图。

---

## 3. 修复后正式数字（收敛口径）

### 3.1 §6.2 主表（XJTU-SY，linear，RMSE min，5-seed 集成）

| 模型 | cond1 | cond2 | cond3 | 均值 |
|---|---|---|---|---|
| SE-ResNet-BiGRU (proposed) | 4.54 | 72.54 | 35.44 | **37.51** |
| GRU | 9.97 | 48.20 | 34.56 | **30.91** |
| BiGRU (`pure_bigru`) | 10.75 | 68.22 | 32.40 | **37.12** |
| CNN | 15.12 | 90.93 | 35.69 | 47.25 |
| LSTM | 15.02 | 75.00 | 34.96 | 41.66 |
| Informer-SE | 11.00 | 47.41 | 46.72 | **35.04** |
| Ridge | 12.98 | 64.40 | 68.05 | 48.48 |
| Const. (mean) | 11.46 | 88.95 | 27.78 | 42.73 |
| Const. (one) | 41.44 | 204.81 | 75.88 | 107.38 |
| *best constant (hurdle)* | *6.06* | *88.91* | *23.96* | *39.64* |

- **4/6 深度模型**（proposed、GRU、Informer-SE、BiGRU）在均值上击败 hurdle（39.64）。
- proposed 仅比 hurdle 优 **2.1 min**（37.51 vs 39.64），远不如朴素基线的崩塌所暗示的那样宽。
- 命名纪律（红线 27）：`BiGRU` 仅指 `pure_bigru`；`gru` 一律写 `GRU`。

### 3.2 §6.3 h 谱（XJTU-SY，三条件均值，RMSE min）

| 标签 | best const | LI | const (one) | Proposed | GRU | Const. (mean) | Gap (proposed − hurdle) |
|---|---|---|---|---|---|---|---|
| linear (h=0) | 39.64 | 0.897 | 107.38 | 37.51 | 30.91 | 42.73 | **−2.14** |
| h = 0.5 | 53.57 | 0.594 | 76.98 | 46.20 | 65.61 | 56.02 | **−7.37** |
| h = 0.6 | 52.62 | 0.491 | 68.83 | 58.23 | 65.88 | 54.22 | +5.61 |
| h = 0.7 | 49.47 | 0.371 | 59.53 | 67.47 | 67.43 | 50.37 | +18.00 |
| h = 0.8 | 43.02 | 0.247 | 48.24 | 77.15 | 71.34 | 43.44 | +34.13 |
| h = 0.9 | 31.91 | 0.127 | 33.69 | 76.18 | 67.95 | 32.05 | +44.27 |

**可继续主张（headline 存活）**：

- **gap 在扫描区间 h∈{0.5..0.9} 上严格单调**（−7.37 → +44.27）；
- **符号翻转仍发生在 h=0.5 与 h=0.6 之间**（≈0.55–0.60；解析预测 h=0.60）；
- 等价的 learning-gain 表述：**由正转负的跨越发生在 h≈0.5–0.7 区间**。

**必须回避的旧表述（已被修复后数据证伪，红线 25）**：

- ❌「proposed 的 RMSE（32.1 → 70.9）随 h 严格单调上升」。
  修复后 proposed 序列为 37.51 → 46.20 → 58.23 → 67.47 → **77.15 → 76.18**，
  **末端 h=0.8→0.9 出现轻微回落（−0.97）**，严格单调性**不成立**。
  → 正式表述改为「**gap 单调**；模型 RMSE 随 h 上升但在 h≈0.8 附近趋稳并轻微回落」。
  **稳定的主结论是 gap/learning-gain 的穿越，而非 RMSE 单调性。**

### 3.3 §6.4 PHM 2012（linear，RMSE 小时）

| 模型 | cond1 | cond2 | 均值 |
|---|---|---|---|
| SE-ResNet-BiGRU (proposed) | 2.01 | 0.27 | 1.14 |
| Ridge | 1.99 | 0.63 | 1.31 |
| Const. (mean) | 1.87 | 0.48 | 1.17 |
| Const. (one) | 3.81 | 1.03 | 2.42 |
| *best constant (hurdle)* | *1.86* | *0.35* | *1.11* |

- proposed 是训练模型中最好的（1.14 h），在 cond2 上过线，但**均值上并未过线**（1.14 vs 1.11），
  cond1 亦未过线（2.01 vs 1.86）→「平手（parity），非超越」。
- 跨数据集**稳定转移的是标签效应**（windowed LI 0.784 @ linear vs 0.158 @ h=0.8），
  而非「架构优于常数」。

### 3.4 §6.5 不确定性（XJTU-SY，linear，名义 PICP=0.95）—— **逐条件报告（红线 26）**

| 模型 | PICP cond1 | PICP cond2 | PICP cond3 | PICP 均值 | MPIW (min) | CRPS |
|---|---|---|---|---|---|---|
| SE-ResNet-BiGRU (proposed) | 0.905 | 0.805 | 0.301 | 0.670 | 62.0 | 23.47 |
| GRU | 1.000 | 0.841 | 0.542 | 0.794 | 75.8 | 19.30 |
| BiGRU | 0.476 | 0.776 | 0.627 | 0.626 | 70.7 | 22.91 |
| CNN | 0.619 | 1.000 | 0.964 | 0.861 | 108.5 | 28.37 |
| LSTM | 0.000 | 0.568 | 0.470 | 0.346 | 56.0 | 27.63 |
| Informer-SE | 1.000 | 0.659 | 0.193 | 0.617 | 54.5 | 23.23 |

- **没有任何模型是标定好的**；仅 CNN 在均值上接近名义（0.861），代价是最宽区间（MPIW 108.5）。
- proposed **每个条件都欠覆盖**（0.905 / 0.805 / 0.301），cond3 塌陷到 0.301。
- 修复后均值由 0.694 → **0.670**；**均值会掩盖 cond3 的塌陷**，故必须逐条件给出（红线 26 / A3）。

### 3.5 §6.1 参考（XJTU-SY，piecewise h=0.8，**全深度阵容**）

| 模型 | cond1 | cond2 | cond3 | 均值 |
|---|---|---|---|---|
| SE-ResNet-BiGRU (proposed) | 8.38 | 140.78 | 82.31 | 77.15 |
| LSTM | 14.67 | 85.46 | 68.39 | **56.18** |
| BiGRU | 11.32 | 95.64 | 78.71 | 61.89 |
| Informer-SE | 11.94 | 102.87 | 96.23 | 70.35 |
| GRU | 9.12 | 124.05 | 80.85 | 71.34 |
| CNN | 13.32 | 111.63 | 101.31 | 75.42 |
| Const. (mean) | 17.07 | 83.14 | 30.12 | 43.44 |
| *best constant (hurdle)* | *16.01* | *83.14* | *29.92* | *43.02* |

- **没有深度模型能击败平凡基线**（红线 25 的负面断言现在有全阵容证据；`h08full` 块本轮新补）。
- **最优深度模型是 LSTM（56.18），不是 GRU（71.34）** —— 修复后全阵容改变了「最优深度模型」的归属；
  稿件 Table 3 与 Abstract/Callout 的「best deep model」已由 71.3 → **56.2** 同步。
- 仅 cond1 被全部深度模型赢下（8.38–14.67 < hurdle 16.01）。

**跑后链再次复核**：`h08full` 完成后重跑 `12 → 13`，且 `h08full` 进程自身的 `verify` 亦再次
`PASS: S5 proposed is bit-identical to s2_chain`（补跑未破坏确定性）。

---

## 4. 跑后自动链（不重训）

```
09_label_baseline_anchors.py  → 10_label_sensitivity.py → 12_paper_tables.py
                              → 11_make_figures.py      → 13_render_html.py
```

| 步 | 产物 | 状态 |
|---|---|---|
| 09 | `s5/label_anchors_{linear,phm2012_linear,xjtu}.json` | ✅ 3 anchor cross-check PASS |
| 10 | `s5/label_sensitivity.json` | ✅ |
| 12 | `paper/tables/tables.md`（Table 3/4/5/6 + 不确定性表） | ✅ |
| 11 | `paper/figures/*`（Fig. 1/2/3 重绘） | ✅ |
| 13 | `paper/manuscript.html` | ✅ |

**关键改动**：`10_label_sensitivity.py`、`12_paper_tables.py`、`11_make_figures.py` 的输出根
统一切到 `S5 = artifacts/reports/s5/`，保证表/图/正文数字同源。

---

## 5. 稿件同步（`paper/manuscript.md`）

| 位置 | 改动 |
|---|---|
| Abstract | 八模型→**九模型**；七规则→**十规则**；最优深度模型 69.3→**71.3**；proposed linear 2.1 min（37.5 vs 39.6）；PICP 69%→**67%**；补充 PHM「parity 而非超越」 |
| §6.1 | 「五个深度模型」→**六个**；Table 3 扩为**全 6 深度阵容**；最优深度模型 GRU→**LSTM（56.18）**；Abstract/Callout 的「best deep model」71.3→**56.2** |
| §6.2 | Table 4 扩为 **9 模型**（新增 `BiGRU` 行）；「4/6 深度模型过线」；点明 proposed 非均值最优 |
| §6.3 | **删除「模型 RMSE 随 h 严格单调上升」**；改为「**gap 单调**，模型 RMSE 在 h≈0.8 趋稳并轻微回落」；保留 h=0.8→0.9 回落；主结论改为 **learning-gain 于 h≈0.55–0.60 由正转负**；Table 5 全量替换 |
| §6.4 | Table 6 数值替换；改为「均值未过线，parity」 |
| §6.5 | Table 7 **逐条件 PICP**（原仅均值）；结论「No model is calibrated」 |
| §6.6 | 跨工况表由 Table 7 → **Table 8**（因 §6.5 占用 Table 7） |
| §7.3 | 「29.88 vs 32.09」→**「30.91 vs 37.51」**；补「BiGRU 37.12 亦与 proposed 持平」 |
| §7.4 | 「Five deep models」→**「Six deep models」** |
| §8 | 七规则→**十规则**；h=0.8「26 min」→**34 min**、h=0「8 min」→**2 min**；sign change 改区间表述；PICP 69%→67% |

---

## 6. 剩余 / 未决

1. **Stage 4 消融**（`resnet_bigru` 等）— 用户明确「不要现在先跑」，待 Stage 5 收敛后进行。
2. **UDA 跨工况臂**（`cross_uda_linear` / `cross_source_only_linear`）— 本轮未跑。
3. 跨工况因果化需**训练量匹配臂**（Stage 3 核心）；在此前 `R_cross`/`ΔRMSE` **只作描述性**，禁因果措辞。
4. 表内 "BiGRU" 命名已落实（红线 27）；需确认无遗留 `gru`→"BiGRU" 错标。

*End of STAGE5_RESULTS.md*
