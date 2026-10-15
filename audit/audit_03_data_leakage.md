# audit_03_data_leakage.md —— Audit 3：数据划分、窗口构造与泄漏

| 项 | 值 |
|---|---|
| 审计对象 | 划分合理性、归一化/特征选择的拟合范围、窗口构造、跨工况的 transductive 边界、验证集来源 |
| 涉及代码 | `rulbench/data/build.py`、`rulbench/data/scaling.py`、`rulbench/data/windows.py`、`rulbench/data/labels.py`、`rulbench/train/protocols.py`、`rulbench/evaluation/report.py`、`configs/base.yaml`、`configs/cross_matched_linear.yaml` |
| 审计日期 | 2026-10-07 |
| 结论 | **PASS**（含 1 项须披露的 transductive 设置与 1 项统计单位问题） |

---

## 1. 划分协议（代码事实）

`configs/base.yaml` L53–L61：

```yaml
splits:
  single:  { train: [1,2,3], val: [4], test: [5] }
  cross:   { source_train: [1,2,3,4], source_val: [5], target_test: [1,2,3,4,5] }
```

`rulbench/data/build.py`

| 位置 | 内容 |
|---|---|
| `build_single_condition`，L276–L305 | `train/val/test` 由 `splits.single` 解析（L290–L293），**同一 condition 内**：train 1–3 / val 4 / test 5 |
| `build_cross_experiment`，L308–L347 | `src_train`/`src_val` 由 `splits.cross` 解析并**对每个 source condition** 取（L319–L328）；`tgt_test` 取**目标 condition** 的轴承（L329–L331） |
| `_build`，L350–L409 | `groups` → 每 split 独立构造窗口（L374–L386）；`meta.groups` 记录实际使用的 `(cond,bearing)`（L403） |
| `build.py` L344 | `"transductive_target": True` —— **显式标记**跨工况为直推式 |

`configs/cross_matched_linear.yaml` L54–L58：配对协议把 `source_train/val` 与 `splits.single`
**逐轴承对齐**（`[1,2,3]` / `[4]` / `[5]`），使两臂**只在「训练数据来自哪个工况」这一个因子上不同**。

---

## 2. 泄漏核查（逐项）

### 2.1 轴承级不重叠 —— **数值验证 PASS**

在 `artifacts/data_matched` 上实测（以 `(condition, bearing_id)` 为键）：

```
single_cond1: train=[(1,1),(1,2),(1,3)]  val=[(1,4)]  test=[(1,5)]
             train∩test=∅   train∩val=∅   val∩test=∅
single_cond2: train=[(2,1),(2,2),(2,3)]  val=[(2,4)]  test=[(2,5)]   三交均为 ∅
single_cond3: train=[(3,1),(3,2),(3,3)]  val=[(3,4)]  test=[(3,5)]   三交均为 ∅
```

**结论**：训练/验证/测试**在轴承层面完全不相交**，不存在「同一轴承的窗口跨 split」的泄漏。✔

### 2.2 归一化统计量只在训练集上拟合 —— **PASS**

`rulbench/data/build.py` L388–L391：

```python
# scaler: fit on the training split only, never on val/test
scaler = ChannelScaler().fit(windows["train"]["X"])
for split in windows:
    windows[split]["X"] = scaler.transform(windows[split]["X"])
```

`rulbench/data/scaling.py::ChannelScaler.fit`（L39–L47）只接受传入的 `X`；
`transform`（L49–L54）只做 `(x−mean)/std`。文档字符串 L4–L7 明确说明这是相对原论文的关键修正
（原实现池化了全部 15 个轴承，包含测试轴承）。

**结论**：`mean/std` **未见过 val/test 数据**。✔

### 2.3 mRMR 特征选择只在训练集上拟合 —— **PASS**

`rulbench/data/build.py`

| 位置 | 内容 |
|---|---|
| L371 | `indices, feat_dim = _select_features(feat_map, groups["train"], cfg)` —— **传入的只有 train 键** |
| `_select_features`，L183–L209 | `train_mat = concatenate([feat_map[k][:, n_fft:] for k in train_keys])`（L196）—— **只拼 train** |
| L199–L204 | relevance 目标 = `linspace(0,1, T)`（归一化寿命位置）—— **由 train 轴承自身的长度生成**，不引用任何测试信息 |
| L208 | `indices = range(n_fft) + [n_fft + i for i in picked]` |

**结论**：`mrmr_indices` 与 `feat_dim` **只依赖 train 轴承**；缓存于 artefact（L400、L417–L420），
可被审阅者直接核查。✔

> 说明：relevance 目标用「寿命位置」作代理，这在标签构造里也用到（`labels.py`）。但两者都**只在 train 轴承上**计算，
> 因此**不构成测试泄漏**；只是需要在方法章说明「relevance 代理 = 寿命位置」。

### 2.4 验证集来源 —— **PASS**

- **单工况**：`val = 轴承4`，与 train（1–3）同 condition、与 test（5）不同轴承。✔
- **跨工况**：`src_val` 由 **`sources` 条件**解析（`build.py` L325–L328），即**源域轴承**；
  目标域轴承只出现在 `test`。→ **模型选择与后处理拟合都不看目标标签**。✔

后处理因子只在**源域 val** 上拟合（`rulbench/evaluation/report.py` L45–L61）：
`val_pred = predict_ensemble(..., split="val")`（L45）→ `median_scaling_factor(val_true, val_pred_min)`（L54–L58）。
**未使用目标域任何标签。** ✔

### 2.5 窗口构造是否泄漏标签 —— **PASS（但须披露「离线基准标签」性质）**

`rulbench/data/windows.py` L50/L58/L61：窗口标签 = **窗口最后一步**的比值。
`rulbench/data/labels.py` L12–L17 明确：比值 × 寿命 = **nominal RUL**，且因平台段恒为 1.0，
**早期 nominal RUL 高估真实剩余寿命**——这是「离线基准标签」的固有性质，**须在论文声明**。

**结论**：窗口构造本身不泄漏（标签只取该窗口末端时刻的值，未使用未来窗口的标签）；
但「标签 = 全寿命归一化位置」意味着**评测是离线基准**，不是在线预测。V2.0 §1.3 与 `labels.py` L10–L17 已列此点。

### 2.6 跨工况的 transductive 设置 —— **须披露（不是泄漏，但等同「测试输入参与训练」）**

`rulbench/train/protocols.py`

| 位置 | 内容 |
|---|---|
| `target_split_for_mmd`，L55–L65 | 仅当 `meta.protocol == "cross"`（L61）**且** `uda.enabled`（L63）时，
返回 `{"X": artifact["splits"]["test"]["X"]}`（L65） |
| `train_one`，L94 | `target = target_split_for_mmd(artifact, cfg)` |
| `train_one`，L105 | `used_target = target is not None and hasattr(model, "mmd_representation")` |
| L99–L104 注释 | 明确：`target is not None` 只表示「**被提供**」；只有暴露 `mmd_representation` 的模型才真正使用 |

**含义**：

- 跨工况「UDA 臂」把**测试集的输入特征**（无标签）用于 MK-MMD 对齐 → **直推式（transductive）**；
- 因此跨工况的「测试集」**不是严格留出集**，其 RMSE 与**非直推**的单工况 RMSE **不同质**；
- 这**不是偷偷作弊**（已由 `build.py` L344 `transductive_target: True` 与 `protocols.py` L1–L9 显式声明），
  但**必须**在方法章与结果处披露，并**必须并列 source-only 对照臂**（`uda.enabled: false`，`protocols.py` L63）→ A5。

**A5 只对带对齐头的模型有意义**：`rulbench/models/registry.py` L138：

```python
MMD_CAPABLE = ("se_resnet_bigru", "resnet_bigru", "se_resnet_gru", "se_resnet_lstm", "pure_bigru")
```

CNN / LSTM / GRU / Informer **不在其中**（registry L46–L49），故这两臂对它们**恒等**，
报「UDA 增益」= 报噪声。✔（与本文档 §2.6 一致）

---

## 3. 一个必须在方法章处理的问题：窗口不是独立样本

`windows.py` L50：`n_win = n_time − window_len + 1`，**stride = 1**（`base.yaml` L21）。
同一轴承相邻窗口**共享 `window_len−1 = 31` 个时间步**，高度自相关。

**后果**：

- 把窗口当独立样本会**严重低估**方差 → p 值过小、CI 过窄；
- V2.0 §1.7 / C4 已规定：**统计单位 = 轴承**，折间依赖须显式处理，不得把相关窗口当独立样本。

**核对现状**：

| 载体 | 是否说明统计单位 | 判定 |
|---|---|---|
| `paper/manuscript.md` | §7.4 提「three test bearings… per-condition claims are based on one bearing」 | ⚠ 提了样本量，但**未明确宣布「统计单位 = 轴承」** |
| `matched_fold_rcross.json` | 报 `n = 45`（5 折 × 9 任务，**任务级**） | ✔ 任务级聚合，未用窗口数当 n |
| `rulbench/evaluation/metrics.py::bootstrap_ci`（L201–L216） | 提供 bootstrap，但**默认对传入数组重采样** | ⚠ 调用方若传窗口数组则违反 C4 |

---

## 4. 缓存指纹问题（非泄漏，但属「静默复用」风险，记录备查）

`rulbench/data/build.py::_fingerprint` L74–L82：指纹只含 `denoise / fft / multi_domain`：

```python
payload = {"denoise":…, "fft":…, "multi_domain":…}
```

**不含标签、不含划分**。因此：

- 换标签（linear / 不同 h）或换划分时，**特征缓存仍会命中** → 必须**隔离 `paths.data_dir`**；
- 本项目已按此执行：`linear → artifacts/data_linear`（`configs/linear.yaml` L23）、
  `h 谱 → artifacts/data_h*`（`scripts/10…` L38–L42）、`配对 → artifacts/data_matched`（`cross_matched_linear.yaml` L51）、
  PHM → `artifacts/phm2012/data_linear`；
- 该风险**已被正确规避**，但**不是自动的**：任何新实验若忘记换 `data_dir`，会**静默复用错标签**。
  → 建议在 Stage 2 的 runbook 里加一条硬检查（读 `index.json` 的 `name` 与标签策略比对）。

**判定**：现状 **PASS**（已隔离），风险**已登记**。

---

## 5. 审计结论

| 子项 | 判定 | 依据 |
|---|---|---|
| A3-a 轴承级 train/val/test 不重叠 | **PASS** | §2.1 数值验证（三交均 ∅） |
| A3-b scaler 只在 train 上拟合 | **PASS** | `build.py` L388–L391 |
| A3-c mRMR 只在 train 上拟合 | **PASS** | `build.py` L371、L196 |
| A3-d 验证集来自本域（单=条件内、跨=源域），模型选择不看目标标签 | **PASS** | §2.4 |
| A3-e 后处理因子只用源域 val | **PASS** | `report.py` L48–L61 |
| A3-f 窗口构造不泄漏（标签只取窗口末端） | **PASS** | `windows.py` L58/L61 |
| A3-g 「离线基准标签」性质须声明 | **PARTIAL**（`labels.py` 有，论文须正式写） | `labels.py` L12–L17 |
| A3-h transductive 设置须披露 + 并列 source-only | **PASS（已实现）** / 论文披露仍需落到 §6.6 与 A5 | §2.6；`protocols.py` L61–L63 |
| A3-i A5 只对 MMD_CAPABLE 有效 | **PASS** | `registry.py` L138 |
| A3-j 统计单位 = 轴承（窗口不独立） | **FAIL**（未在论文中宣布） | §3；V2.0 C4 |

### 5.1 总判定：**PASS**

**泄漏层面无可指摘**：轴承不重叠、归一化与特征选择**均只在训练集拟合**、
验证集来自本域、后处理因子只用源域 val。这是本项目相对原实现的**主要修正**，
且在代码里以注释与 artefact 字段（`build.py` L10–L12、`meta.groups` L403）**可核查**。

两项须随稿处理（不构成泄漏）：

1. **transductive 披露**（A5）：跨工况 UDA 臂使用测试**输入**；须并列 source-only 对照，
   且只对 `MMD_CAPABLE` 模型谈「对齐增益」。
2. **统计单位**（C4）：必须写明「统计单位 = 轴承」，窗口不作为独立样本。

### 5.2 对 Protocol Freeze 的输入

见 `STAGE1_AUDIT_REPORT.md` 的 **P3（统计单位）、P5（transductive 声明）、P1/P2（划分与预处理）**。

---

## 6. 未核实项

| 项 | 状态 |
|---|---|
| `artifacts/data_matched` 之外的数据集（`data_h*`、`phm2012/data_linear`）是否同样无轴承重叠 | **未逐个数值验证**（构造代码同一路径 `build.py::_build`，结构上应一致；未对每个 npz 复核） |
| `feature_cache` 内是否存在**跨标签误用**的历史文件 | **未核实**（须比对 `index.json` 与 `_fingerprint` 记录） |
| `pca_health_indicator`（`build.py` L43–L55）是否可能在别处被当作特征/标签 | **未核实**（L46 注释声明「never used as a training target」，未全局搜索确认） |

*End of audit_03_data_leakage.md*
