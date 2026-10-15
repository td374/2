# audit_01_LI.md —— Audit 1：`LI(h)` 的数学推导与适用条件

| 项 | 值 |
|---|---|
| 审计对象 | `LI(h) = 3(1−h)/4` 的推导、适用条件、命名 |
| 涉及代码 | `rulbench/data/labels.py`、`rulbench/data/windows.py`、`rulbench/evaluation/metrics.py`、`scripts/09_label_baseline_anchors.py`、`scripts/10_label_sensitivity.py` |
| 审计日期 | 2026-10-07 |
| 结论 | **MODIFY** —— 数学**成立**；但必须补「适用条件」并执行命名降级 |

---

## 1. 代码事实（先看代码，再看论文）

### 1.1 标签的定义

`rulbench/data/labels.py`

| 位置 | 内容 |
|---|---|
| `piecewise_linear_rul(n_samples, healthy_ratio=0.8)`，L27–L42 | `threshold = int(n*healthy_ratio)`（L37–38）；`ratio[:threshold]=1`；`ratio[threshold:] = (n − arange(threshold,n))/span`（L40–41），其中 `span = n − threshold`（L39） |
| `linear_rul(n_samples)`，L45–L51 | `tail = arange(1,n+1)/n`（L48）；`ratio = tail[::-1]`（L50）→ 即 `r_i = (n − i)/n`，`r_0 = 1`，`r_{n−1} = 1/n` |

**关键事实 1**：`h` 在代码中的实际含义 = **平台段占比**（`[0, h·n)` 段恒为 1.0）。
这就是 `healthy_ratio`，与 V2.0 §2.4 的「平台段占比 h」**一致**，「h 的实际含义」**已核实**。

**关键事实 2**：linear 标签的末端是 `1/n`，**不是 0**（labels.py 文档字符串 L12–L17 亦明确说明）。
即连续极限中「r 触达 0」这一前提，在离散实现里**不成立**。

### 1.2 标签的分窗与「被打分」的集合

`rulbench/data/windows.py`

| 位置 | 内容 |
|---|---|
| `make_windows`，L17–L63 | `n_win = n_time − window_len + 1`（L50）；`idx = arange(n_win)[:,None] + arange(window_len)[None,:]`（L56）；`ends = idx[:, −1]`（L58）；**`ratio: ratio[ends]`**（L61） |

**关键事实 3**：窗口的标签 = **窗口最后一步**的标签值；因此**被打分的窗口端点**是
`ends = window_len−1 … n_time−1`，即**前 `window_len−1` 个快照永远不参与打分**。
`window_len = 32`（`configs/base.yaml` L20），故**前 31 个快照的标签被丢弃**。

`scripts/09_label_baseline_anchors.py::_scored_ends`（L52–54）复现了这条规则：

```python
return np.arange(window_len - 1, n_snap)
```

### 1.3 `LI` 的计算式

| 位置 | 内容 |
|---|---|
| `scripts/09_label_baseline_anchors.py::_anchors`，L57–L69 | `rmse_one = life * sqrt(mean((ratio−1)²))`（L58）；`c_best = ratio.mean()`（L59）；`rmse_best = life * sqrt(mean((ratio−c_best)²))`（L60）；`li = 1 − (rmse_best/rmse_one)²`（L62） |
| `scripts/10_label_sensitivity.py::analytic_anchors`，L71–L99 | 同构，仅在**每轴承**上计算后对轴承取**算术平均**（L99） |
| `rulbench/evaluation/metrics.py::constant_baseline_rmse`，L120–L172 | 提供**连续极限**（L153–154：`lifetime*sqrt((1−h)/3)`）与**窗口精确**（L156–L172：`ends=arange(window_len−1, n_samples)`，L170）两种闭式 |

---

## 2. 推导复核

### 2.1 定义式

```
LI = 1 − (RMSE_best / RMSE_one)²  =  1 − Var_e[r] / E_e[(r−1)²]                        (A1.1)
```

由 `RMSE_best = L·sqrt(Var[r])`（09 L60 + 展开）与 `RMSE_one = L·sqrt(E[(r−1)²])`（09 L58），
两者的 `L` 相消，得 (A1.1)。**（A1.1）成立**，与 `paper/manuscript.md` 式 (8) 一致。

> **限定**：`L` 相消要求**同一受评集合内 `L` 为常数**。跨轴承池化时 `L` 随窗口变化，
> 该式**不成立**（须把 `L` 放进均值内部）。见 `audit_02_constant_baseline.md` §4。

### 2.2 连续极限

设 `r ~ U`：平台段 `[0,h)` 上 `r=1`，衰减段 `[h,1]` 上 `r` 自 1 线性到 0。

```
E[r]   = h·1 + ∫_h^1 r dr / (1−h) · (1−h) = h + (1−h)/2 = (1+h)/2
E[r²]  = h + ∫_h^1 r² dr = h + (1−h³)/3
Var[r] = E[r²] − E[r]²  = (1−h)(1+3h)/12
E[(r−1)²] = ∫_h^1 (r−1)² dr = (1−h)/3
LI(h) = 1 − [(1−h)(1+3h)/12] / [(1−h)/3] = 1 − (1+3h)/4 = 3(1−h)/4                (A1.2)
```

**(A1.2) 推导成立。** 与 `paper/manuscript.md` 式 (9) 及 L218–L220 的证明一致。
数值上 h = 0.8 → 0.150 = `3(1−0.8)/4` ✔。

### 2.3 三个必要时条件（**这是本审计要补的**）

(A1.2) 的成立隐含**三个必要条件**，缺一不可：

| 条件 | 内容 | 实现是否满足 |
|---|---|---|
| **C-a** | `N → ∞`（连续、均匀采样） | ✘ XJTU-SY 最短轴承仅 **52** 个快照 |
| **C-b** | `r` 覆盖**整个** `[0,1]`（触达 0） | ✘ `linear_rul` 末端 = `1/n` ≠ 0；且**前 `L−1` 个快照不被打分** |
| **C-c** | 受评集合内 `L` 为常数 | ✔ 单工况成立；✘ 旧跨工况池化不成立 |

**C-b 的量级**：`window_len = 32`，`ends = 31…n−1`。
对 `linear`，被打分的 `r` 只有 `{1/n, 2/n, …, (n−31)/n}` 这一段（**低 RUL 段**），
`r > (n−31)/n` 的高 RUL 段**从不被打分**。这直接改变 `Var[r]` 与 `E[(r−1)²]` 的比值。

### 2.4 窗口精确闭式（本审计建议正式采用）

`metrics.py::constant_baseline_rmse`（L156–L172）已实现正确闭式：

```
threshold = max(1, min(int(n·h), n−1));  span = n − threshold
ratio[threshold:] = (n − arange(threshold, n)) / span
ends  = arange(window_len − 1, n)                       # 与 windows.py L58 一致
RMSE_one_windowed = lifetime · sqrt(mean((1 − ratio[ends])²))                     (A1.3)
```

该式与代码的窗口构造**逐位对应**（`windows.py` L50/L58/L61 ↔ `metrics.py` L164–L170）。
**判据**：任何与「实测 RMSE」比较的锚点，必须用 (A1.3)，**不得**用连续极限。

---

## 3. 数值验证（真实数据）

用 `scripts/09_label_baseline_anchors.py --config configs/linear.yaml --data-dir artifacts/data_linear`
实跑（window_len = 32，三个单工况测试轴承 cond1::5 / cond2::5 / cond3::5）：

| label | const_one | best_const | LI（窗口，实测） | LI（连续，式 A1.2） | 偏差 |
|---|---|---|---|---|---|
| linear (h=0) | 107.38 | 39.64 | **0.897** | 0.750 | +19.6% |
| piecewise h=0.5 | 76.98 | 53.57 | **0.594** | 0.375 | +58.4% |
| piecewise h=0.6 | 68.83 | 52.62 | **0.491** | 0.300 | +63.7% |
| piecewise h=0.7 | 59.53 | 49.47 | **0.371** | 0.225 | +64.9% |
| piecewise h=0.8 | 48.24 | 43.02 | **0.247** | 0.150 | +64.7% |
| piecewise h=0.9 | 33.69 | 31.91 | **0.127** | 0.075 | +69.3% |

（LU 为「三轴承 LI 的算术平均」，与 `09` L177 的聚合方式一致。）

逐轴承明细（`09` 输出）：

| bearing | n_snap | life | linear const_one / best_const / LI | h=0.8 const_one / best_const / LI |
|---|---|---|---|---|
| cond1::5 | 52 | 52.0 | 41.44 / 6.06 / **0.979** | 20.24 / 16.01 / **0.374** |
| cond2::5 | 339 | 339.0 | 204.81 / 88.91 / **0.812** | 90.95 / 83.14 / **0.164** |
| cond3::5 | 114 | 114.0 | 75.88 / 23.96 / **0.900** | 33.52 / 29.92 / **0.203** |

**结论（数值）**：

1. **单调性成立**：窗口 `LI` 随 `h` 严格单调下降 ✔（H1 的标签侧前提成立）。
2. **(A1.2) 与窗口实测的偏差随 `h` 增大**（+19.6% → +69.3%），
   即在**最需要精度的低 LI 区（h ≥ 0.8，LI ≈ 0.15）偏差最大**。
3. **偏差随寿命缩短而放大**：cond1（52 快照）h=0.8 的窗口 LI = 0.374，是连续值 0.150 的 **2.5 倍**。
   `metrics.py` L140–L143 已记载同一现象（1_5 连续 13.4 min vs 窗口 20.2 min，差 51%）。
4. **交叉点会被连续式系统性偏移**：连续式给 h=0.60 → 0.300；窗口实测 h=0.6 → 0.491。
   因此 `h*` 的**数值位置依赖所用口径**，进一步支持「`h≈0.60` 只能是待复核现象」。

---

## 4. 「只是标签分布的数学结果」核查（V2.0 §1.3(a) 第 2 项）

**核查结论：是。** `LI` 完全由 `{标签定义 h, 窗口长度, 寿命 n}` 三个量决定，
**不含任何特征、模型、数据信息**；`09`/`10` 两个脚本均**只读 `test/ratio`，不建模型**（`09` L124–L147、`10` L78–L98）。

因此 V2.0 的硬约束**必须执行**：

> `LI(h)` 只能称为「**标签结构指标**」（label-structure index），
> **不得**宣称它测量「真实退化信息 / 可预测性 / 可学习性」。

**当前措辞状态**：

| 载体 | 措辞 | 判定 |
|---|---|---|
| `paper/manuscript.md` 式 (8)(9)、Table 1（L229–L235） | 明确写「in the continuous limit」，并**并列**窗口实测值 | ✔ 合规 |
| `paper/manuscript.md` L211 | 「it is computable before training a single model」 | ✔ 合规（陈述的是可计算性，非物理含义） |
| `docs/contributions.md` L11 | 「**LI(h)=3(1−h)/4**（Prop. 1）；PHM 2012 同序复现」 | ✘ **未加「连续极限」限定** |
| `docs/experiment_plan.md` L923 | 「解析理论：**LI(h) = 3(1−h)/4**（Prop. 1，已数值验证）」 | ✘ 同上；且「已数值验证」易被读成「离散实测**等于**该式」 |
| `docs/comparison_strategy_qa.md` L94/L169/L234 | 多处按**等式**使用 | ✘ 同上 |
| `docs/S2L_S5L_results.md` L63 | 「`LI(h) = 1 − Var/E[(r−1)²] = 3(1−h)/4`」 | ✘ **写成两个等号**，等于断言离散等连续 |
| `docs/STATUS_review.md` L83 | 「解析式 `LI(h) = 3(1−h)/4`」 | ✘ 未加限定 |

> 注：`docs/*.md` 属**过程记录**（V2.0 §4.2 已把它们多列为「重写」对象），
> 因此这是**文档级 MODIFY**，不是数学错误。

---

## 5. 审计结论

### 5.1 结论明细

| 子项 | 判定 | 依据 |
|---|---|---|
| A1-a 定义式 (A1.1) 成立 | **PASS** | §2.1；`paper` 式 (8) |
| A1-b 连续极限 `3(1−h)/4` 推导成立 | **PASS** | §2.2；数值 h=0.8 → 0.150 ✔ |
| A1-c 单调性成立 | **PASS** | §3 表格：0.897 → 0.594 → … → 0.127 |
| A1-d 适用条件已明示（N→∞ / r→0 / L 常数） | **FAIL**（未明示） | §2.3；`paper` §3.2 只提「continuous limit」，未列三条 |
| A1-e 命名降级为「标签结构指标」 | **FAIL**（未执行） | §4；6 处文档仍按等式使用 |
| A1-f 离散口径必须用窗口闭式 (A1.3) | **已实现但未强制** | `metrics.py` L156–L172 有闭式；`paper` L226–L227 已提，但 `09` 的 headline 仍以连续式对照 |

### 5.2 总判定：**MODIFY**

- **数学没错**，(A1.1)(A1.2) 均可直接使用；
- 但必须补三件事（否则会构成 V2.0 §1.3(a) 明令禁止的过度宣称）：

1. **补适用条件**：正式写「式 (9) 成立于 N→∞、均匀采样、r 触达 0、受评集合内 L 为常数」；
2. **命名降级**：全稿（含 `docs/`）统一为「**标签结构指标**」，删除任何「度量退化管理/可预测性」的措辞；
3. **口径双报**：凡引用 `LI(h)` 数值，必须**同时**给出连续值与**窗口实测值**（现 `paper` Table 1 已做到，需推广到所有图与表）。

### 5.3 对 Definition Freeze 的输入

见 `STAGE1_AUDIT_REPORT.md` 的 **Definition Freeze 条目 D2**。

---

## 6. 未核实项

| 项 | 状态 |
|---|---|
| PHM 2012 的窗口 `LI` 谱是否与 XJTU-SY 同序 | **未核实**（`artifacts/phm2012/data_linear` 存在，但本次未跑；V2.0 §4.2 未列此项为必做） |
| `h* ` 在窗口口径下的具体数值 | **未核实**（须在 Stage 2 h-sweep 重跑后确定；现在给任何数值都是超范围） |

*End of audit_01_LI.md*
