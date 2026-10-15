# audit_02_constant_baseline.md —— Audit 2：`c*` / `best_const` 与实际损失的一致性

> **本项是 Stage 1 的最高优先级**（V2.0 §6.2 P0-1、冲突 C1）。
> 若 `c*` 的定义与评价损失不一致，则**整个「事后最优常数地板」失效**，
> §6.6 的「无模型越过地板」结论、Fig 3 的 hurdle 线、Table 5 的 oracle 列全部需重算。

| 项 | 值 |
|---|---|
| 审计对象 | `c* = Σwᵢrᵢ/Σwᵢ` 的权重定义；三种零学习基准的角色边界；损失空间 |
| 涉及代码 | `rulbench/losses.py`、`rulbench/evaluation/metrics.py`、`rulbench/evaluation/report.py`、`rulbench/models/trivial.py`、`scripts/09…`、`scripts/10…`、`scripts/11…`、`scripts/15_cross_transfer.py` |
| 审计日期 | 2026-10-07 |
| 结论 | **MODIFY**（定义需补「空间」限定；`15` 实现正确）+ **1 项实现缺陷（`11`）** |

---

## 1. 代码里**同时存在**三种「最优常数」——先把这个事实摆出来

| # | 出处 | 公式 | 空间/权重 | 作用域 |
|---|---|---|---|---|
| **c₁** | `scripts/09_label_baseline_anchors.py::_anchors` L59 | `c_best = ratio.mean()` | **比值空间**，无权重（普通均值） | **每轴承**（`09` L135–L147 循环 `b in sorted(set(bid))`） |
| **c₂** | `scripts/10_label_sensitivity.py::analytic_anchors` L93 | `best = life·sqrt(mean((r−r.mean())²))` | 同上（比值空间普通均值） | 每轴承，再对轴承取算术平均（L99） |
| **c₃** | `scripts/11_make_figures.py::analytic_anchors` L99–L100 | `one = life·sqrt((r−1)²)`；`best_c = life·sqrt((r−r.mean())²)` | ⚠ **既非 RMSE 也非普通均值**（见 §5） | 每轴承，再取平均（L101） |
| **c₄** | `scripts/15_cross_transfer.py::_oracle_constant` L60–L69 | `w = lifetime²; c = Σ(w·r)/Σw` | **分钟空间**，权重 `w = L²` | **任务级池化**（`anchors_for` L72–L96 对全部窗口） |

**c₁ = c₂ ≠ c₃（缺陷） ≠ c₄（当 L 变化时不等价）。**
若方法章只写「最优常数」而不指明是哪一个，读者无法判断地板到底画在哪。

---

## 2. 评价损失在哪个空间？—— 代码事实

### 2.1 报告指标 = **分钟空间、无权重 RMSE**

`rulbench/evaluation/metrics.py`

| 位置 | 内容 |
|---|---|
| `rmse`，L35–L37 | `sqrt(mean((y_true − y_pred)²))` —— **无权重** |
| `evaluate_predictions`，L234–L255 | 形参名 `y_true_minutes, y_pred_minutes`（L235–L236）；`out["rmse"] = rmse(y_true_minutes, y_pred_minutes)`（L244） |
| `metrics.py` L242 | 文档字符串：`"""Compute the full metric block for one prediction vector (minutes)."""` |

`rulbench/evaluation/report.py`

| 位置 | 内容 |
|---|---|
| `_split_truth`，L23–L27 | `truth = ratio · lifetime`（L26） |
| `evaluate_dataset`，L70–L71 | `metrics = evaluate_predictions(test_true, corrected_min, …)`；两者**均为分钟** |

**事实 A**：headline 指标 = **分钟空间无权重 RMSE**。`configs/base.yaml` L124 `headline_metric: rmse`。

### 2.2 训练损失 = **比值空间、阈值加权 MSE**

`rulbench/losses.py`

| 位置 | 内容 |
|---|---|
| `weighted_mse`，L25–L42 | `w = 2` if `target < 0.3` else `1`（L37–L41）；`mean(w·(pred−target)²)`（L42） |
| L33 | 文档字符串：`"""… ``target`` is the normalised RUL ratio …"""` |
| `configs/base.yaml` L90–L94 | `loss: weighted_mse`；`threshold: 0.3`；`weight: 2.0`；`val_loss: mse` |

**事实 B**：训练目标 = **比值空间、阈值加权 MSE（threshold=0.3, weight=2）**；模型选择用 `val_loss: mse`（`base.yaml` L94）。

> **事实 A ≠ 事实 B**：**训练在比值空间，评价在分钟空间**。这是一个**必须披露的口径差**（见 §6）。

---

## 3. 推理：在「评价指标」下，最优常数到底是谁？

评价量为分钟空间无权重 RMSE。设常数预测 `pred = c·L_i`（预测比值 c），真值 `L_i·r_i`：

```
RMSE(c)² = (1/N)·Σᵢ (Lᵢrᵢ − Lᵢc)² = (1/N)·Σᵢ Lᵢ²(rᵢ − c)²                        (A2.1)
```

对 c 求导置零：

```
c* = Σᵢ Lᵢ²rᵢ / Σᵢ Lᵢ²                     ← 寿命平方加权均值                      (A2.2)
```

**两种情况**：

| 情形 | `Lᵢ` 是否随窗口变化 | `c*` |
|---|---|---|
| 受评集合内**同一轴承**（或若干**同寿命**轴承） | 常数 | (A2.2) 退化为 **普通均值 `Σrᵢ/N`** |
| 受评集合**跨多轴承、寿命不同** | 变化 | **必须是** (A2.2) 的 `L²` 加权均值 |

**因此**：

- `c₄`（`15_cross_transfer.py` L67–L68 的 `w = L²`）**正是 (A2.2)**，即**在评价损失下正确的**事后最优常数 ✔
- `c₁`/`c₂`（普通均值）**只在受评集合内 L 为常数时正确**；在「每轴承独立计算」这一实现下（`09` L135、`10` L88）**恰好正确** ✔
- 若把普通均值直接套到跨轴承池化任务上，则**错误** ✘

### 3.1 与 V2.0 §1.3(b) 的关系

V2.0 §1.3(b) 写：

> 「现有 `c* = ΣLᵢ²rᵢ/ΣLᵢ²` 对应 `wᵢ = Lᵢ²`，**只在评价损失确为 L² 加权 MSE 时成立**；
> 若评价用**普通 RMSE**，最优常数就是**普通均值**（无 L² 权重）。⚠ 待核实。」

**本审计的裁决**：这句需要**一处修正**——

> 「普通 RMSE」有两种空间，结论相反：
> - **比值空间**普通 RMSE → 最优常数 = 普通均值（V2.0 原句成立）；
> - **分钟空间**普通 RMSE（**本文实际使用的**）→ 最优常数 = **`L²` 加权均值**，
>   因为「分钟 = 比值 × L」这一步**自动引入了 `L²` 权重**（式 A2.1）。
>
> 所以 `L²` 权重**不是**「额外加的一个权重」，而是**比值→分钟换算的必然结果**。
> V2.0 的「待核实」在此**关闭**：**现实现（c₄）与评价损失一致。**

---

## 4. 数值验证（真实数据）

对常数比值 `c` 在 `[0,1]` 上暴力网格搜索，求「分钟空间 RMSE」的最小点，
与 `c_plain`（普通均值）、`c_life2`（L² 加权）、`c_lossopt`（`weighted_mse` 最优）对比：

| 任务 | n | 平均寿命 | `c_plain` | `c_life2` | `c_lossopt` | 暴力最小点 | 暴力最小值 |
|---|---|---|---|---|---|---|---|
| **旧** `data_linear/1+2→3`（未配对，5 轴承池化） | 6879 | 2165.9 | 0.4900 (642.25) | **0.4936 (642.19)** | 0.4103 (669.11) | **0.4936** | 642.19 |
| **配对** `data_matched/1+2→3` | 83 | **114.0（单轴承）** | 0.3684 (**23.96**) | 0.3684 (23.96) | 0.3060 (24.99) | 0.3684 | 23.96 |
| **配对** `data_matched/2+3→1` | 21 | **52.0（单轴承）** | 0.2115 (**6.06**) | 0.2115 (6.06) | 0.1875 (6.18) | 0.2116 | 6.06 |

（括号内为该常数对应的**分钟空间 RMSE**。）

**读法**：

1. **旧未配对任务**：暴力最小点 = `0.4936` = `c_life2` ✔（=`15` 的 `oracle_const_ratio_task`，
   与 `artifacts/reports/cross_transfer_summary.json` 的 `oracle_const_task__rmse = 642.1947` **完全吻合**）。
   普通均值给 642.25，差 0.01%——**数值上极小**，因为该任务里绝大多数窗口集中在少数几个长寿命轴承上。
   → **结论：`c₄` 正确，且与普通均值在该任务上的数值差可忽略；但定义上必须用 `L²` 加权。**
2. **配对任务**：目标测试集只有**一个轴承**（`target_test: [5]`，`configs/cross_matched_linear.yaml` L58），
   因此 `L` 在受评集合内**恒定** → `c_plain ≡ c_life2 ≡` 暴力最小点。
   → **配对协议下，三种定义自动合一，C1 的风险在配对协议里天然消失。**
3. **`c_lossopt`（`weighted_mse` 的最优）与评价最优不同**（0.4103 vs 0.4936；0.3060 vs 0.3684）。
   若把 `best_const` 定义成「训练损失下的最优常数」，地板会**抬高**（669.11 vs 642.19）。
   → 说明「地板定义在哪个损失上」是**实质性选择**，必须在方法章写死：**本文的地板定义在
   「报告指标 = 分钟空间无权重 RMSE」上**。

---

## 5. 实现缺陷：`scripts/11_make_figures.py` 把 MAE 当成 RMSE

`scripts/11_make_figures.py::analytic_anchors`，L87–L102：

```python
one     = life * np.sqrt((ratio - 1.0) ** 2)                 # L99
best_c  = life * np.sqrt((ratio - ratio.mean()) ** 2)        # L100
out[rec["name"]] = {"const_one": float(one.mean()),
                    "best_const": float(best_c.mean())}      # L101
```

`np.sqrt((x)**2) = |x|`（逐元素），再 `.mean()` → 得到的是 **分钟空间 MAE**，
即 `mean(L·|r−1|)`，**不是** `sqrt(mean((L·r − L)²))`。
**`sqrt` 与 `mean` 的次序颠倒**（须先 `mean` 再 `sqrt`）。

对照正确的是 `scripts/09…::_anchors` L58 / L60：`life * sqrt(mean((ratio − c)²))`。

**量化偏差**（`artifacts/data_linear` 三个单工况测试轴承）：

| bearing | life | `09`/`10` 正确 `best_const`（RMSE） | `11` 给出的值（MAE 形式） | 相对低估 |
|---|---|---|---|---|
| cond1::5 | 52 | 6.06 | 5.24 | **−13.5%** |
| cond2::5 | 339 | 88.91 | 77.00 | **−13.4%** |
| cond3::5 | 114 | 23.96 | 20.75 | **−13.4%** |

**影响面**：该函数被 `fig3()` 调用（L214 XJTU-SY、L221 PHM 2012），
经 `_grouped`（L232–L248）在 **L242–L243** 画出「best constant (hurdle)」菱形：

```python
hv = [anch.get(c, {}).get("best_const", np.nan) * scale for c in conds]   # L242
ax.plot(x, hv, "D", color="black", ms=6, label="best constant (hurdle)")  # L243
```

→ **`paper/figures/fig3_main_results.{png,pdf}` 的 hurdle 线系统性偏低约 13.4%**，
视觉上**放大了模型相对地板的领先幅度**。图为交付物，故这是**实质缺陷**。
（`paper/tables/*` 与 `manuscript.md` 的数字来自 `09`/`10`/`12`，**未受此缺陷影响**。）

**修复**：把 L99–L100 改为与 `09` L58/L60 同构：

```python
one    = life[0] * np.sqrt(np.mean((ratio - 1.0) ** 2))
best_c = life[0] * np.sqrt(np.mean((ratio - ratio.mean()) ** 2))
```

（或统一改为「分钟空间池化」：`sqrt(mean((life*ratio − life)²))` / `sqrt(mean((life*ratio − c*)²))`。）
**本审计不修改代码**（审计阶段只读）；修复动作记入 `STAGE1_AUDIT_REPORT.md` 的 Blocking Issues。

---

## 6. 口径差必须披露：训练在比值空间，评价在分钟空间

| 环节 | 空间 | 依据 |
|---|---|---|
| 监督损失 `weighted_mse` | **比值** | `losses.py` L33、L42 |
| 模型选择 `val_loss: mse` | **比值** | `base.yaml` L94；模型输出为比值 |
| 报告 `rmse` | **分钟** | `metrics.py` L235、L244；`report.py` L70 |
| 事后地板 `best_const` | **分钟** | 式 (A2.2)；`15` L67 |

**含义**：模型是被"教"去最小化**比值空间**的加权误差，却被**分钟空间**的无权重 RMSE 打分。
两者对「哪些样本更重要」的偏好不同（分钟空间更看重长寿命样本，比值空间+阈值权更看重低 RUL 样本）。

**披露要求（写入方法章，非选项）**：
> 「训练目标为比值空间的阈值加权 MSE（threshold = 0.3, weight = 2.0），
> 报告指标为分钟空间无权重 RMSE；事后最优常数按报告的分钟空间 RMSE 定义（式 A2.2）。」

---

## 7. 三种零学习基准的**角色边界**（V2.0 §1.3(b) 要求分列）

| 基准 | 定义（代码位置） | 拟合所用数据 | 角色 | 可否当「可部署模型」 |
|---|---|---|---|---|
| `const_one` | `models/trivial.py`：`value="one" → 1.0` | 不用数据 | 朴素参照（预定义固定常数） | ✘ |
| `const_mean` | `models/trivial.py`：`value="mean" → np.mean(y)`；`y` = **训练**标签 | **训练集**标签 | **可部署**的零特征基准（公平比较的对手） | ✔ |
| `best_const` | `09` L59 / `15` L60–L69，在**受评集合**上求最优 | **测试集**标签（事后） | 测试集**事后诊断**，度量常数预测极限 | ✘ **（禁止）** |

**注意**：`const_mean` 用的是**训练集**标签均值，而 `best_const` 用**测试集**（=`val` 或 `test`）的最优值。
两者**不在同一信息条件下**，因此：

- **H2 / 主分析**：`G` 必须相对 **`const_mean`** 计算（V2.0 §1.7：主分析用 `const_mean`）；
- **`best_const`** 只用于回答「是否有模型越过常数极限」这一**诊断**问题。

**核对现状**：

| 载体 | 是否分列角色 | 判定 |
|---|---|---|
| `scripts/10_label_sensitivity.py` L136–L139 注释 | 明确写了「hurdle 是解析 best_const」，并说明 `const_mean ≥ best_const` | ✔ |
| `paper/manuscript.md` §3.3 / Table 1 | 有 `best_const`，但**未集中分列三角色** | ⚠ 须在方法章集中说明（C13） |
| `configs/cross_matched_linear.yaml` L66–L72 | 「constant_mean is the strongest *deployable* constant; the post-hoc optimal constant is computed analytically by 15」 | ✔ |

---

## 8. 审计结论

| 子项 | 判定 | 依据 |
|---|---|---|
| A2-a `15` 的 `c₄`（`L²` 加权）与评价损失一致 | **PASS** | §3 推导 + §4 暴力验证吻合 642.19 |
| A2-b `09`/`10` 的普通均值在**每轴承**作用域下正确 | **PASS** | §3 表格（L 常数 → 退化） |
| A2-c 配对协议下三种定义合一，C1 风险消失 | **PASS** | §4 配对任务 `c_plain ≡ c_life2` |
| A2-d 定义式必须明示「比值空间 vs 分钟空间」 | **FAIL**（未明示） | §3.1；V2.0 §1.3(b) 原表述会引起误读 |
| A2-e `11_make_figures.py` 锚点正确性 | **FAIL** | §5；低估 13.4%，影响 Fig 3 交付物 |
| A2-f 三角色分列 + 地板定义在哪个损失上写死 | **PARTIAL** | §7；须在方法章集中说明 |
| A2-g 训练空间 ≠ 评价空间须披露 | **FAIL**（未披露） | §6 |

### 8.1 总判定：**MODIFY**

**好消息（C1 的核心风险解除）**：
- 评价是**分钟空间**无权重 RMSE；其最优常数**就是** `L²` 加权均值；
- `15_cross_transfer.py` 用的正是它 → **「事后最优常数地板」的定义与实现一致**，
  §6.6「无特征模型越过地板」的结论**不因 C1 而失效**；
- 配对协议（`cross_matched_linear.yaml`）下目标测试集只有单轴承，三种定义**自动合一**，
  因此**配对比较对 C1 免疫**。

**必须 MODIFY 的三点**：
1. **定义式补空间限定**（V2.0 §1.3(b) 的「待核实」可关闭，但原句须改写为「比值空间/分钟空间」二分）；
2. **修 `11_make_figures.py`** 的 MAE/RMSE 次序缺陷（图交付物，属 P0 缺陷）；
3. **披露训练/评价空间差异** + **集中分列三角色**（C13）。

### 8.2 对 Definition Freeze 的输入

见 `STAGE1_AUDIT_REPORT.md` 的 **D1（c*/best_const）、D3（三角色边界）、D5（G/ΔRMSE）**。

---

## 9. 未核实项

| 项 | 状态 |
|---|---|
| `ridge` 基线对 `L` 的依赖（是否也受空间差影响） | **未核实**（本次未审 `rulbench/models/trivial.py` 的 ridge 路径实现细节） |
| PHM 2012 单位（秒/小时）下 `L²` 权重是否引入新的量纲偏差 | 结构上不影响（`L` 的常数倍在 (A2.2) 中相消）；**但未在 PHM 数据上数值验证** |
| `val_split` 上的 `best_const` 与 `test_split` 上的差异 | **未核实** |

*End of audit_02_constant_baseline.md*
