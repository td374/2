# STAGE5_PROTOCOL_CHECK.md —— Stage 5 重跑前的协议一致性检查（入口门槛）

> **依据**：用户 2026-10-08 指令「重跑前先做一次『协议一致性检查』。这一步比直接启动 GPU 更重要」。
> 本文件是 **Stage 5 的入口门槛**：§2 全部 PASS 才允许启动重跑；§8 的决策点确认后即可执行。
>
> **参照协议**：`audit/STAGE1_PROTOCOL_FREEZE.md`（F1–F5、红线 1–23）、`audit/STATISTICAL_UNIT_PROTOCOL.md`。

| 项 | 值 |
|---|---|
| 检查日期 | 2026-10-08 |
| 检查范围 | §6.2 `single_linear` / §6.3 `sweep_h` / §6.4 PHM2012 / §6.5 不确定性 |
| 结论 | **门槛通过（2 项已在代码层修复，0 项未处置）**，但**发现修复后正式数字发生实质变化**（§5），必须整体收敛 |
| GPU | 未启动（严格遵守「先检查、后重跑」） |

---

## 1. 有效性基准：两个修复时点

后续一切「有效 / 失效」判定都以下述时点为界（`STAGE1_PROTOCOL_FREEZE.md` §F5）：

| 修复 | 落点（文件） | 时点 |
|---|---|---|
| **确定性修复**（训练侧） | `rulbench/train/protocols.py`：`seed_everything(seed)` 前移到 `build_for_artifact` **之前**；`rulbench/__init__.py`：`CUBLAS_WORKSPACE_CONFIG` 提前到包导入 | **2026-10-07 09:44** |
| **B8**（评测侧） | `rulbench/evaluation/inference.py::predict_seed(..., seed=)`：MC-Dropout 采样前播种 | **2026-10-07 16:21** |

→ **有效产物必须晚于 16:21 生成**（且训练晚于 09:44）。

---

## 2. 逐项协议一致性检查（对照用户指定的 5 组）

### ① 数据：划分 / 泄漏 / 统计单位 —— **PASS**

| 检查点 | 证据 | 结论 |
|---|---|---|
| 单工况划分 | `configs/base.yaml::splits.single = {train:[1,2,3], val:[4], test:[5]}`；`data/build.py::build_single_condition` | 轴承级互斥，无跨轴承泄漏 |
| 跨工况划分 | `splits.cross = {source_train:[1,2,3,4], source_val:[5], target_test:[1,2,3,4,5]}` | 源/目标工况分离 |
| 特征选择泄漏 | `build.py::_select_features` 仅用 `groups["train"]` 拟合 mRMR；relevance 目标为**训练轴承**的寿命位置代理，不使用测试标签 | 无泄漏 |
| 标准化泄漏 | `build.py::_build`：`ChannelScaler().fit(windows["train"]["X"])`，仅训练集 | 无泄漏 |
| 统计单位 | 单工况单元 = 测试轴承 `(condition, bearing)`，共 3 个（cond1/2/3 的 b5）；**禁**按 `bearing_id` 单独分组（三工况 b5 同名，红线 9） | 与 F1 一致 |
| 每格样本量 | 5 seeds 集成，窗口级 RMSE；主结果口径 = **测试实例等权平均**（红线 13） | 一致 |

> 数据侧**不受训练侧修复影响** → `artifacts/data_linear`、`data_h{0.5..0.9}`、`artifacts/phm2012/data` 可原样复用（`STAGE1_PROTOCOL_FREEZE.md` §4.2）。

### ② 模型命名 —— **PASS（已在源码层修复）**

根因：`scripts/12_paper_tables.py::DISPLAY` 与 `11_make_figures.py::MODEL_LABEL` 曾把 `"gru" → "BiGRU"`，
而 `rulbench/models/baselines.py::GRURegressor` 默认 `bidirectional=False`（实为**单向 GRU**）。

| 现状 | 位置 |
|---|---|
| `"gru" → "GRU"`，并补 `pure_bigru→BiGRU`、`resnet_bigru→ResNet-BiGRU`、`se_resnet_gru→SE-ResNet-GRU` | `12_paper_tables.py`、`11_make_figures.py`（已改） |
| 真 BiGRU 由 `pure_bigru` 承担（`ablations.py::build_pure_bigru`，`bidirectional=True`） | `models/registry.py::MODELS` 已注册、可 `resolve` |
| 稿件 Table 3/4/5 与 §7.3 的 "BiGRU" 标签已全部改为 "GRU" | `paper/manuscript.md`、`paper/tables/tables.md` |

→ Stage 5 主表**必须新增 `pure_bigru` 行**，否则「BiGRU 优于基线」的表述无对应数字。本轮纳入（见 §6 Block 1）。

### ③ 推理：B8 播种 / A10 一致性 —— **PASS（已抽样实证，见 §4.1）**

- `predict_ensemble` → `predict_seed(..., seed=seed)` → 采样前 `seed_everything(seed)` → 预测 = f(checkpoint, data, seed)。
- 实测：对**已修复**的 `s2_chain/linear` 复用 checkpoint 再评测，与存储 CSV **逐位一致（0.001% 舍入）**。
- 对**失效**的 `single_linear` 复用 checkpoint 再评测，与存储 CSV **差 1.04%** → 证明「老 checkpoint + 新评测」是无效组合。

### ④ 指标口径 —— **PASS**

| 检查点 | 结论 |
|---|---|
| headline 指标 | `rmse`（**分钟空间、无权重**），`report.headline_metric: rmse` |
| 双口径 | `results_corrected.csv` / `results_uncorrected.csv` 同时产出（红线 13 / A9） |
| 单工况后处理 | `report.py`：`correction_enabled = ... and is_cross` → **单工况 factor ≡ 1.0**，校正/未校正一致（F §3.4） |
| `best_const` 角色 | 仅事后诊断（解析 hurdle），**禁**当可部署基线（红线 8） |
| B4 口径 | `std_min` 随 `factor` 缩放；未校正行不复用校正区间（已修） |

### ⑤ 输出留痕 —— **PASS（但需新增分支，见 §8-Q1）**

现状：每次运行为 `resolved_config.yaml` + `checkpoints/*.pt` + `results_corrected.csv` + `results_uncorrected.csv` + `results_full.json` + `summary_by_model.csv`。
`run()` 默认 `train_missing=True`，**checkpoint 已存在则静默跳过训练** → 这是 Stage 5 唯一的高危陷阱（§4.2）。

---

## 3. 存量产物失效台账（Stage 5 四个目标块）

| 块 | 目录 | 生成窗口 | 判定 |
|---|---|---|---|
| §6.2 主表 | `artifacts/reports/single_linear/` | 10-06 23:59 → 10-07 00:12 | **失效**（早于确定性修复） |
| §6.3 h 谱 | `artifacts/reports/sweep_h{0.5..0.9}/` | 10-07 00:14 → 00:38 | **失效** |
| §6.1 参考（h=0.8） | 同上 `sweep_h0.8` | 10-07 00:28 → 00:32 | **失效**（§6.1/Table 3 亦引用此目录） |
| §6.4 PHM | `artifacts/phm2012/reports/single_linear/` | 10-07 00:41 → 00:46 | **失效** |
| §6.5 不确定性 | 派生自 `single_linear/` 的 `picp/mpiw/crps` 列 | — | **失效** |
| （暂停）UDA | `cross_uda_linear/`、`cross_source_only_linear/` | 10-07 01:38 → 01:57 | **失效**；按用户指示**本轮不跑** |
| 遗留 | `artifacts/reports/single/`（piecewise h=0.8 原始跑） | 10-06 16:35 → 16:58 | 失效；未被 `12_paper_tables.py` 引用，仅存档 |

**可复用的有效产物**（`STAGE1_PROTOCOL_FREEZE.md` §4.2 + §6 登记）：

| 目录 | 内容 | 生成时间 |
|---|---|---|
| `artifacts/reports/s2_chain/{linear,h0.5..h0.9}` | **proposed + const_one + const_mean**，6 个标签点，修复后 | 10-07 21:49 → 22:12 |

---

## 4. 抽样实证（本轮新做，脚本 `xjtu_sy_rul/_s5_probe.py`）

### 4.1 A10 / B8 复评测抽样（3 条件中的 cond1，5 seeds 集成）

| 案例 | 再评测 RMSE | 存储 RMSE | 相对差 | 判定 |
|---|---|---|---|---|
| `s2_chain/linear`（修复后） | **4.5381** | 4.5381 | **0.001%** | ✅ 逐位一致 → 可复用 |
| `single_linear`（修复前） | 5.3872 | 5.4438 | **1.041%** | ❌ 不一致 → 不可复用 |

→ 同时验证了 B8 修复有效、且「老 checkpoint + 新评测」确实产生口径错配。

### 4.2 checkpoint 复用陷阱（确认存在）

`s2_chain` 之外的所有 Stage-5 目标目录**仍存有修复前的 `.pt`**（如 `single_linear` 含 75 个）。
若在原目录直接重跑，`pipelines/run.py` 会因 `ckpt.exists()` 为真而**跳过训练**、只做新评测 →
产出「旧权重 + 新评测」的**混血结果**。**因此重跑必须落在全新结果根目录**（不覆盖、不就地重跑）。

---

## 5. 关键发现：修复后正式数字发生**实质变化**

由于确定性修复改变了训练初始权重，**proposed 模型（SE-ResNet-BiGRU）的 RMSE 整体移动**；
trivial 基线（`const_one` / `const_mean`）**逐位不变**（它们不训练），故对比是干净的。

### 5.1 §6.2 主表（linear，RMSE min）—— proposed 行

| | cond1 | cond2 | cond3 | 均值 |
|---|---|---|---|---|
| 稿件现用（修复前） | 5.44 | 57.97 | 32.87 | **32.09** |
| 修复后（`s2_chain/linear`） | **4.54** | **72.54** | **35.44** | **37.51** |
| trivial 基线（两者一致） | — | — | — | const_mean 42.73 / const_one 107.38 |

→ 「proposed 比 hurdle 优 **7.6 min**」将变为 **约 2.1 min**（hurdle 39.64 不变）。

### 5.2 §6.1 / §6.3 h 谱 —— proposed 均值与 gap

| 标签 | hurdle | LI | 稿件均值（修复前） | 修复后均值 | 稿件 gap | 修复后 gap |
|---|---|---|---|---|---|---|
| linear (h=0) | 39.64 | 0.897 | 32.09 | **37.51** | −7.55 | **−2.13** |
| h = 0.5 | 53.57 | 0.594 | 41.89 | **46.20** | −11.68 | **−7.37** |
| h = 0.6 | 52.62 | 0.491 | 52.77 | **58.23** | +0.15 | **+5.61** |
| h = 0.7 | 49.47 | 0.371 | 66.56 | **67.47** | +17.09 | **+18.00** |
| h = 0.8 | 43.02 | 0.247 | 69.34 | **77.15** | +26.31 | **+34.13** |
| h = 0.9 | 31.91 | 0.127 | 70.91 | **76.18** | +39.00 | **+44.27** |

**维持在结论上的（可继续主张）**：

- **gap 随 h 严格单调递增**（−2.13 → +44.27）；
- **符号翻转仍发生在 h = 0.5 与 h = 0.6 之间**（≈0.60）——headline 存活。

**必须改写的（旧表述将被证伪）**：

- 稿件写「proposed 的 RMSE（32.1 → 70.9）**单调**上升」。修复后序列为
  37.51 → 46.20 → 58.23 → 67.47 → **77.15 → 76.18**，**末端 h=0.8→0.9 出现轻微回落**，
  严格单调性**不成立** → §6.3 读法 1 须改为「**gap 单调**；模型 RMSE 随 h 上升但在 h≈0.8 附近趋稳」。

### 5.3 §6.5 不确定性 —— proposed 的 PICP 逐条件重排

| | cond1 | cond2 | cond3 | 均值 |
|---|---|---|---|---|
| 修复前 | 0.667 | 0.909 | 0.506 | **0.694** |
| 修复后 | **0.905** | **0.805** | **0.301** | **0.670** |

→ 「PICP ≈ 0.694 平均」需改为「≈0.67，但**逐条件差异极大**（cond3 仅 0.30）」；
「只有 CNN/GRU 接近名义覆盖率」这一句必须用**新跑的全模型 PICP** 重判（旧值失效）。

### 5.4 §6.4 PHM 2012
数字待重跑（旧目录整体失效）；预期同为修复后重排，须整体替换。

---

## 6. Stage 5 重跑计划（P0 顺序，与用户指定一致）

> **统一落在全新根目录** `artifacts/reports/s5/`（避免 §4.2 混血陷阱）。
> 若与 `s2_chain` 复用比对（§8-Q1 选项 A），则 `s5` 的 proposed 数字须与 `s2_chain` **逐位相同**——
> 作为**内建确定性交叉验证**。

| 块 | 内容 | 模型 × 条件 × seed × 标签点 | 新增 checkpoint | 预计 |
|---|---|---|---|---|
| **P0-1 §6.2** | XJTU-SY single，linear | (se_resnet_bigru, cnn, lstm, gru, informer_senet, **pure_bigru**) × 3 × 5 × 1 + trivial(=one/mean/ridge，不训练) | 6×15 = 90 | ~12 min |
| **P0-2 §6.3** | XJTU-SY single，h∈{0.5..0.9} | (se_resnet_bigru, gru) × 3 × 5 × 5 + const_mean | 2×75 = 150 | ~20 min |
| **P0-3 §6.4** | PHM2012 single，linear | se_resnet_bigru × 2 × 5 + trivial | 10 | ~3 min |
| **P0-4 §6.5** | 不确定性 | 派生自 P0-1 / P0-3 的 `picp/mpiw/crps` | 0 | 0 |
| ~~P1 UDA~~ | `cross_uda_linear` / `cross_source_only_linear` | — | — | **暂停（用户指示）** |

**跑后自动链**（不重训）：
`09_label_baseline_anchors.py`（解析 anchors，含 linear 行）→ `10_label_sensitivity.py`（h 谱聚合，需改指向 `s5`）
→ `12_paper_tables.py`（Table 3/4/5/6 + 不确定性表）→ `11_make_figures.py`（图重绘）→ `13_render_html.py`。

**内建检查（跑后立即断言）**：
1. `s5[proposed, linear]` 与 `s2_chain[linear]` **逐位一致**（否则确定性仍未闭合）；
2. trivial 基线与修复前**逐位一致**（数据未变的哨兵）；
3. `09_label_baseline_anchors.py` 的 cross-check 全 PASS（确认 label 构造未被静默改动）。

---

## 7. 红线（Stage 5 全程有效，继承 1–23）

新增/强调：

24. **禁止复用失效目录内的 checkpoint**；重跑必须换结果根（§4.2）。
25. **禁止**恢复「模型 RMSE 随 h 严格单调」的旧表述（§5.2，已被修复后数据证伪）。
26. §6.5 的不确定性结论**必须逐条件**给出，不得只报三条件均值（§5.3）。
27. `pure_bigru` 进入主表后，**"BiGRU" 仅指 `pure_bigru`**；`gru` 一律写 "GRU"。

---

## 8. 待用户确认的决策点

- **Q1 结果根策略**：
  - (A) 全新 `s5/`，**并**以 `s2_chain` 作确定性命中校验（推荐：单一口径 + 内建自检）；
  - (B) 只跑缺口模型（`cnn/lstm/gru/informer_senet/pure_bigru` + h 谱 `gru` + PHM），proposed 直接复用 `s2_chain`
    （省 ~40% GPU，但证据集分布在两个目录）；
  - (C) 就地覆盖 §3 的失效目录（**不推荐**：违反 §4.2，且丧失审计留痕）。
- **Q2 §6.2 模型阵容**：在现 8 模型外**新增 `pure_bigru`**（真 BiGRU，落实命名修复）——建议采纳；
  是否同时把 `resnet_bigru` 也纳入主表？（默认**否**，留给 Stage 4 消融）

*End of STAGE5_PROTOCOL_CHECK.md*
