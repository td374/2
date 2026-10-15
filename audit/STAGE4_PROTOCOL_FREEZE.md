# STAGE4_PROTOCOL_FREEZE.md —— Stage 4 协议冻结检查（入口门槛，先于任何训练）

> **依据**：用户 2026-10-08 指令「**不要现在直接让模型开始训练 Stage 4**。先做一次 Stage 4 的协议冻结检查」。
> 本文件是 **Stage 4 的入口门槛**：§2 全部 PASS、§7 六个问题全部有答案，才允许启动训练。
>
> **上游**：`audit/STAGE1_PROTOCOL_FREEZE.md`（F1–F5、红线 1–23）、`audit/STAGE5_PROTOCOL_CHECK.md`（红线 24–27）、
> `audit/STAGE4_MODEL_POSITIONING.md`（三层设计 + R1–R5）、`audit/STAGE5_RESULTS.md`（Stage 5 已封板）、
> `audit/STATISTICAL_UNIT_PROTOCOL.md`。

| 项 | 值 |
|---|---|
| 检查日期 | 2026-10-08 |
| 检查范围 | L1 结构贡献（单工况 linear）/ L2 架构×标签结构（h 谱）/ L3 跨工况小规模验证 |
| 结论 | **门槛通过：0 项未处置**；所有 Stage 4 消融变体**从未训练过**（§1 已核），必须全部新训 |
| GPU | **未启动**（严格遵守「先冻结、后训练」） |

**状态定义（用户拍板 2026-10-08）**：

> **Stage 2 / 2.1 / 3 / 5：封存。**  **Stage 4：下一阶段唯一新增实验。**
> **Stage 4 完成后，再进行最终论文重构，而不是边跑边改论文。**

---

## 1. 地基事实核查（决定「必须新训」清单）

### 1.1 消融变体在代码里的真实结构（读 `rulbench/models/ablations.py`）

`SequenceRegressor` 是唯一实现，四个因子通过构造参数切换：

| registry key | `use_extractor` | `use_se` | `cell` | `bidirectional` | 论文名称 |
|---|---|---|---|---|---|
| `pure_bigru` | ✗ | ✗ | gru | **True** | **BiGRU** |
| `resnet_bigru` | ✓ | ✗ | gru | **True** | **ResNet-BiGRU** |
| `se_resnet_gru` | ✓ | ✓ | gru | **False** | **SE-ResNet-GRU** |
| `se_resnet_lstm` | ✓ | ✓ | lstm | False | SE-ResNet-LSTM |
| `se_resnet_bigru` | ✓ | ✓ | gru | **True** | **SE-ResNet-BiGRU**（proposed） |
| `resnet_lstm` | ✓ | ✗ | lstm | True | ResNet-BiLSTM（额外对照） |
| `cnn`（baseline） | ✗（CNN1D） | ✗ | — | — | CNN |

**因子隔离（Stage 4 的核心，比线性梯子更严格）**：

| 因子 | 关闭 | 打开 | 隔离量 |
|---|---|---|---|
| 残差 CNN 提取器 | `pure_bigru` | `resnet_bigru` | 提取器贡献 |
| SE 通道注意力 | `resnet_bigru` | `se_resnet_bigru` | SE 贡献 |
| 双向 RNN | `se_resnet_gru` | `se_resnet_bigru` | 双向贡献 |
| 时序建模（RNN） | `cnn` | `pure_bigru` | RNN 贡献 |

> ⚠ `cnn` 属 `baselines.py::CNN1D`（不同类），与 `SequenceRegressor` 不是同构切换；
> 因此「RNN 贡献」一行**只能作方向性参考**，主因子链用前三行（同为 `SequenceRegressor`）。

### 1.1b 冒烟验证 + 参数增量核算（本机实跑，`input_dim=140`）

七个变体**全部可构造**（`ALL BUILT OK`）；参数增量与因子一一对应：

| 模型 | 参数量 | 因子增量 |
|---|---|---|
| `cnn` | 52,161 | — |
| `pure_bigru` | 153,729 | — |
| `se_resnet_gru` | 196,417 | — |
| `resnet_bigru` | 280,705 | **提取器** = +126,976 |
| `se_resnet_bigru` | 283,265 | **SE** = +2,560；**双向** = +86,848（vs `se_resnet_gru`） |

> ⇒ 因子是**可干净隔离的**：提取器与双向是大额参数变化，SE 仅 +2,560（**≈0.9% 参数**）——
> 因此若 SE 带来可观 `G` 增益，其解释**不会**是「参数更多」；反之若 SE 无增益，也**不能**用「参数少」开脱。
> `resnet_lstm` 的 `supports_mmd=False`（无 `mmd_representation` 走提取器分支）——**它不在 L3 阵容内**，无影响。

### 1.2 checkpoint 存量核查（已用脚本枚举全仓库 `.pt`，2026-10-08 实测）

> **枚举口径**：`artifacts/**/checkpoints/*.pt`，共 **88 个 checkpoint 目录 / 2174 个 `.pt`**。

| 目录族 | 实际覆盖的模型 | 状态 |
|---|---|---|
| `s5/single_linear/*` | `se_resnet_bigru, cnn, lstm, gru, informer_senet, pure_bigru`（各 15） | ✅ 修复后（Stage 5） |
| `s5/sweep_h0.8/*` | **全阵容**（`se_resnet_bigru, cnn, lstm, gru, informer_senet, pure_bigru`） | ✅ 修复后 |
| `s5/sweep_h{0.5,0.6,0.7,0.9}/*` | **只有** `se_resnet_bigru` + `gru` | ✅ 修复后（**不是全阵容**） |
| `s2_chain/{linear,h0.5..h0.9}/*` | `se_resnet_bigru`（各 15） | ✅ 修复后 |
| `matched_fold{1..5}/{single,cross}/checkpoints` | `se_resnet_bigru`（single 15 / cross 45） | ✅ 修复后（Stage 3 实体） |
| `matched_fold{1..5}_b8/` | **无 checkpoint**（纯重评测产物，脚本 `33`） | ✅ 只作口径核对 |
| `vm_fold{1..5}/{single,cross}/checkpoints` | `se_resnet_bigru` | ✅ 修复后（Stage 3） |
| `single_linear/*`、`sweep_h*/*`、`single/*`（**无 `s5/` 前缀**） | 多个 | ❌ **修复前，禁用** |
| **`resnet_bigru` / `se_resnet_gru` / `se_resnet_lstm` / `resnet_lstm`** | — | **🚫 全仓库 0 个 checkpoint** |

**⚠ 对原表述的三处更正**（本次实测）：

1. **`s5/sweep_h{0.5,0.6,0.7,0.9}` 不是全阵容**——只有 `se_resnet_bigru` 与 `gru`。原 §4 把
   「L2 可复用 `pure_bigru`」写进估算，**只对 h=0.8 成立**；h≠0.8 处 `pure_bigru` 必须新训。
2. **`matched_fold{k}_b8` 没有任何 checkpoint**——它是 `scripts/33` 用 `matched_fold{k}` 的
   checkpoint 重算出的 CSV，只用于口径核对，**不可作为训练来源**。
3. **⚠⚠ 关键：`run()` 的复用不会自动发生。** `run()` 的判定是
   `run_dir = out_dir/"checkpoints"`，`if train_missing and not ckpt.exists(): train`。
   由于 Stage 4 写入**全新根 `s4/`**，其 `run_dir` 为空 ⇒ **所有模型（含 `se_resnet_bigru`/`pure_bigru`/`cnn`）
   都会被重新训练**，§2⑤ 承诺的「免费 A10 命中校验」**不会自动获得**。
   ⇒ 若确实想要该命中校验，**必须显式把 `s5/` 对应 `.pt` 预置到 `s4/.../checkpoints/`**
   （见 §2⑤-补：`36` 的 `--seed-reuse` 步骤）。**否则 Stage 4 是「全量重训」而非「部分复用」**。
   两者都合法，但**必须在冻结文档里二选一写清**，不得含糊。

**⇒ 结论（回答用户问题 1/2/6）**：

- **必须新训**：`resnet_bigru`、`se_resnet_gru`（L1/L2/L3 核心）；L2 在 **h≠0.8** 处还需新训
  `pure_bigru`（h=0.8 可复用 `s5/sweep_h0.8/pure_bigru`）；`se_resnet_lstm`、`resnet_lstm`（可选对照）。
- **可复用（作基准）**：`pure_bigru`、`se_resnet_bigru`、`cnn`（`s5/single_linear`）；`se_resnet_bigru` 在
  h=0.8 全阵容、h=0.5/0.6/0.7/0.9 的 `s5/sweep_h*`。**但仅在「同一 `data_dir` + 同一配置 + 修复后版本」
  下成立**，且需按 §2⑤-补**显式预置**才生效。
- **绝不复用**：任何无 `s5/` 前缀的 `single_linear/`、`sweep_h*/`、`single/` 内 checkpoint（修复前，红线 24）。

---

## 2. 逐项协议一致性检查（对照用户指定的 6 点）

### ① 数据划分 —— **PASS**

| 检查点 | 证据 | 结论 |
|---|---|---|
| 单工况（L1/L2） | `configs/linear.yaml`（继承 base）`splits.single = {train:[1,2,3], val:[4], test:[5]}` | 轴承级互斥，与 Stage 5 完全一致 |
| 标签根 | L1 用 `artifacts/data_linear`；L2 用 `artifacts/data_h{h}`（h∈{0.5..0.9}） | **与 Stage 5 同根**，保证 G 可比 |
| 特征选择 | `build.py::_select_features` 仅用 `groups["train"]` 拟合 mRMR | 无泄漏 |
| 标准化 | `ChannelScaler().fit(windows["train"]["X"])` | 无泄漏 |
| 跨工况（L3） | `configs/matched_folds/fold{k}.yaml`（leave-one-bearing-out）+ `matched_folds_vm`（训练量匹配） | 复用 Stage 3 既定框架 |

### ② seed —— **PASS**

`runtime.seeds = [42, 123, 456, 789, 1011]`（`configs/base.yaml`）——**与 Stage 2/3/5 逐位相同**的五种子。
**不新增、不删减**，保证跨 Stage 可比。

### ③ 训练协议（loss / optimizer / epoch / early stopping）—— **PASS**

全部读自 `configs/base.yaml`，Stage 4 **一律不改**：

| 项 | 值 |
|---|---|
| loss | `weighted_mse`（threshold 0.3、weight 2.0）——**同一损失** |
| batch_size | 128 |
| epochs | 150 |
| val_loss（模型选择） | `mse` |
| patience | 15 |
| optimizer | Adam，lr 1e-3，weight_decay 0.0 |
| scheduler | reduce_on_plateau（factor 0.5, patience 5, min_lr 1e-6） |
| grad_clip | null |
| 结构超参（所有消融共用） | hidden_dim 64 / num_layers 2 / dropout 0.2 / se_reduction 16 / channel_plan (64,128) |

> ⭐ **这是 Stage 4 唯一合法性的基础**：`ablations.py` 的 `SequenceRegressor` 保证消融与 proposed
> **除被消融因子外完全同构**，且走**同一 `train_one` 循环**。这正是论文版被批评「消融与主模型协议不一致」的代码级修复。

### ④ 评测 —— **PASS**

| 检查点 | 证据 | 结论 |
|---|---|---|
| headline 指标 | `report.headline_metric: rmse`（**分钟空间、无权重**） | 与 G 定义一致 |
| MC-Dropout | `evaluation.mc_samples: 50`，**已在 B8 后播种**（`predict_seed(..., seed=)`） | 可复现 |
| 单工况后处理 | `report.py`：`correction_enabled = ... and is_cross` → **单工况 factor ≡ 1.0** | L1/L2 校正/未校正一致 |
| 双口径 | `results_corrected.csv` + `results_uncorrected.csv` | A9 |
| `best_const` 角色 | 仅事后诊断（解析 hurdle），**禁**当可部署基线（红线 8） | — |

### ⑤ 输出留痕 & 结果根 —— **PASS（本文件新冻结）**

- **结果根（新，红线 24 的落实）**：`artifacts/reports/s4/`
  - L1：`s4/l1_linear/{model}/`
  - L2：`s4/l2_h{h}/{model}/`
  - L3：`s4/l3_fold{k}/{single,cross}/`、`s4/l3_vm_fold{k}/{single,cross}/`
- **`resolved_config.yaml` + `checkpoints/*.pt` + `results_*.csv` + `results_full.json`** 随每次运行落盘。
- ⛔ **禁止**把 Stage 4 结果写进 `s5/`、`s2_chain/`、`matched_fold*/`（污染已封存证据集）。
- ⚠ **复用已有 checkpoint 的唯一合法条件**：`data_dir` 与 Stage 5 **逐字符相同**、模型 key 相同、seed 相同、
  且 checkpoint 由**修复后**版本产出。

### ⑤-补 复用机制的真实行为与两选一（2026-10-08 代码级核实）

`rulbench/pipelines/run.py::run()`：

```python
out_dir  = pathlib.Path(out_dir)
run_dir  = out_dir / "checkpoints"
...
ckpt = checkpoint_path(run_dir, artifact["name"], model_name, seed)
if train_missing and not ckpt.exists():
    train_one(artifact, model_name, cfg, seed=seed, out_path=ckpt, device=device)
results.append(evaluate_dataset(artifact, model_name, cfg, run_dir, seeds, device=device))
```

因此 **checkpoint 的查找根是 `out_dir/checkpoints`，与 `cfg.paths.report_dir` 无关**。
Stage 4 的 `out_dir` 是全新的 `s4/...`，其 `checkpoints/` 为空 ⇒ **默认行为 = 全量重训**。

**两个方案（必须二选一，写进冻结决议）**：

| 方案 | 做法 | 得到什么 | 代价 | 建议 |
|---|---|---|---|---|
| **A. 全量重训（默认/最干净）** | 不做任何预置；Stage 4 所有模型全部新训 | **完全独立**于 Stage 5 的证据；无版本耦合 | 多算 `se_resnet_bigru/pure_bigru/cnn` 的重复训练（≈1/3 预算） | ✅ **推荐**：论文主张是「组件贡献」，独立性比省 GPU 更重要；且与 Stage 5 的**任何**共享都可能被审稿人质疑 |
| **B. 预置复用（省算力 + 免费命中校验）** | 显式把 `s5/` 对应 `.pt` 复制进 `s4/.../checkpoints/`，令 `run()` 跳过训练只重评测 | Stage 4 与 Stage 5 在重叠模型上**逐位可比**（一次免费 A10 校验） | 若校验不过，说明复用前提破裂 → 必须整体回退方案 A | ⚪ 仅在**预算紧张**时采用；且必须**先跑通校验**再继续 |

> **冻结决定（本文件）：采用方案 A（全量重训）。**
> 理由：① Stage 5 的教训正是「旧产物污染新结果」，方案 A 从根上消除该风险；
> ② 论文的 Stage 4 主张是「同一 `SequenceRegressor` 下逐因子 `ΔG`」，**独立性优先于算力**；
> ③ 方案 B 的「命中校验」虽然诱人，但其价值 < 被审稿人质疑「Stage 4 的 proposed 数字与 Stage 5 同源」的风险。
> ⇒ `36_run_stage4.py` 保留 `verify()` 作为**可选**命中校验（方案 B 时使用）；**方案 A 下 `verify()` 返回
> 「no reusable output」并 PASS（0），属预期行为**，不得误判为失败。

### ⑥ 统计单位 —— **PASS（沿用冻结协议）**

- **单位 = 测试轴承实例** = `(fold, 目标工况)` = `(condition, bearing)`；单工况 L1/L2 只有 3 个单元（cond1/2/3 的 b5）；
  L3 五折 × 3 工况 = **n = 15 = 全部 15 轴承**（`audit/STATISTICAL_UNIT_PROTOCOL.md`）。
- 每单元内 5 seed 是**重复测量**（集成后为 1 个预测），3 source-task 亦为重复测量。
- **主证据 = 分布 + 效应量 + 轴承层级自助 CI**；p 值降级（cell→探索性；单元→indicative）。

---

## 3. Stage 4 统计结构（用户 2026-10-08 最终固定，**不得更改**）

```text
Stage 4
│
├── L1 组件贡献
│   ├── pure_bigru → resnet_bigru          提取器
│   ├── resnet_bigru → se_resnet_bigru     SE
│   └── se_resnet_gru → se_resnet_bigru    双向
│
├── L1 补充架构比较
│   └── cnn vs pure_bigru                  CNN 时序表征 vs BiGRU
│
├── L2 标签结构 × 架构
│   ├── G_model(h)
│   ├── ΔG_architecture(h)
│   └── observed crossover interval
│
└── L3 架构 × 跨工况
    ├── SE-ResNet-BiGRU
    ├── ResNet-BiGRU
    └── BiGRU
```

### 3.1 L1 =「组件贡献分析 + 补充架构比较」，**不是排行榜**

前三条是**组件增量消融**（同名 `SequenceRegressor`、单一因子切换，可直接相减求 `ΔG`）；
`cnn → pure_bigru` 是**架构家族比较**（`CNN1D` 与 `SequenceRegressor` 不同类），
**只能作方向性参考，不得与前三行并列成"谁最好"**。

| 层 | 模型 | 说明 |
|---|---|---|
| **L1 组件贡献** | `pure_bigru`、`resnet_bigru`、`se_resnet_bigru`、`se_resnet_gru` | 三个单因子切换 → 提取器 / SE / 双向 各自的 `ΔG` |
| **L1 补充架构比较** | `cnn`（仅作方向性对照） | CNN 时序表征 vs BiGRU；**不参与组件 ΔG 链** |

### 3.2 L2 的核心是 `ΔG`，**不是比较 RMSE**

```
G_m(h)            = 1 − RMSE_m(h) / RMSE_const_mean(h)
ΔG_{A→B}(h)       = G_B(h) − G_A(h)
ΔG_SE(h)          = G_{se_resnet_bigru}(h) − G_{resnet_bigru}(h)      （示例）
```

回答的是「**同一标签结构下，加入某组件后，模型相对零学习基线的真实学习增益是否增加？**」——
**不是**「该组件的 RMSE 有没有下降」。这是 Stage 4 与 Stage 2 真正接上的地方。

### 3.3 零点：只报 **observed crossover interval**，不报精确临界值

Stage 2 只支持「learning gain 符号转换发生在约 **h≈0.55–0.60**」。
Stage 4 **沿用同一措辞**，输出：

- 相邻 `h` 的**符号变化**；
- `G(h) > 0 / < 0` 逐点标记；
- **observed crossover interval**（跨越发生在哪两个相邻网格点之间）；
- 无法确定时**明确标 `not identified`**。

⛔ **禁止**写 `h* = 0.573` 之类的精确临界值——除非新增连续 `h` 训练设计并给出明确的插值/估计模型（本轮不做）。

### 3.4 L3 = 架构 × 工况的**迁移行为**，不是第二个 Stage 3

L3 阵容 **最多三模型**：`se_resnet_bigru` / `resnet_bigru` / `pure_bigru`。
它回答的是：

> **Stage 4 得到的架构增益，能否在跨工况条件下保留？**

**不是**「cross 是否优于 single」（那是 Stage 3 的职责）。呈现
`ΔG_architecture`（或至少 `R_cross`），解释为**架构 × 工况的迁移行为**。
**禁止**扩成「6 模型 × 9 任务」。

---

## 4. 新增 checkpoint 工程量（**2026-10-10 启动前实测修正**）

> 依据 §2⑤-补 的冻结决定（方案 A），下列全部按「新训」计。
> ⚠ **原估 ≈885 / 2–3h 是错误的**：它把 L3 的 cross 臂误当成 3 个任务，实际 `configs/matched_folds/fold{k}.yaml`
> 的 `prepare.cross_tasks` 有 **9 个源任务变体**（`1->2,1->3,2->1,2->3,3->1,3->2,1+2->3,1+3->2,2+3->1`），
> `run_cross` 逐个训练 ⇒ L3/L3vm 各 **900** 个 checkpoint。已用 `data_fold1` 的 12 个 npz 与 Stage 3 的
> `matched_fold1`（15 single + 45 cross = 60 ckpt/模型）实地点算复核。

| 层 | 计算（task × model × seed） | checkpoint 数 | 实测单次耗时 |
|---|---|---|---|
| L1 | 3 single × 5 模型 × 5 seed | **75** | ~1.3 s |
| L2 | 5 h ×（3 single × 4 模型 × 5 seed） | **300** | ~1.3 s |
| L3 | 5 折 ×（3 single + **9 cross**）× 3 模型 × 5 seed | **900** | single ~1.3 s / **cross ~18 s** |
| L3-vm | 同 L3（训练量匹配臂） | **900** | 同 L3 |
| **合计** | | **2175** | — |

**实测总时长（RTX 4050 Laptop，6.4 GB）**：

| 层 | 训练耗时 |
|---|---|
| L1 | 1.6 min |
| L2 | 6.5 min |
| L3 | **3.46 h** |
| L3-vm | **3.46 h** |
| **训练合计** | **≈ 7.0 h** |
| + 集成评测（MC-Dropout 50）开销 ≈ 30% | **≈ 9–9.2 h** |

> ⇒ **正式全量训练需预留约 9 小时**（远高于原估 2–3h），瓶颈是 L3/L3vm 的 cross 臂（各 ≈3.5h）。
### 4.1 分阶段执行策略（**用户 2026-10-11 拍板**）

**L3/L3vm 的计算量主要由 cross-task enumeration 导致，而非模型数量增加**——L3 的模型数与单工况一致
（3 个），但每个折要跑 **9 个 source-task variant** 的 cross 臂，于是 900 ckpt/层。
一个**辅助性**的迁移实验因此占据了全 Stage 4 的 **~90% 计算成本**，这在实验设计与复现说明上都站不住。

> **因此 Stage 4 采用分阶段执行策略，L3 在 L1/L2 结果确认后启动。**

| 阶段 | 内容 | ckpt | 预计 | 产出 |
|---|---|---|---|---|
| **Phase 1（先行）** | **L1 + L2** | **375** | **≈10 min** | **S4-T1**（组件贡献 `ΔG`）+ **S4-T2**（`G(h)` 曲面 + observed crossover interval） |
| Phase 2（条件触发） | L3 + L3vm | 1800 | ≈7 h | S4-T3（架构 × 跨工况迁移） |

**Phase 2 的触发条件（用户判定）**：

- **若 L1/L2 强**（架构差异明显 / proposed 在多数 h 有优势 / 组件贡献稳定）→ **跑 L3**，回答
  「架构优势能否跨工况迁移」。
- **若 L1/L2 弱**（各架构 `G` 同步塌到 ≈0、高 h 全部失效）→ **L3 价值下降**：仍需跑，但可**重新考虑
  cross-task 数量**（保持三模型不变，不扩模型）。

> ⛔ **Phase 1 期间禁止启动 L3/L3vm**。且 **Phase 1 结果不到位就不改论文**（用户：先证明"模型为什么重要"，
> 再证明"模型能力是否能迁移"）。

---

## 5. 输出表（冻结）

| 表 | 内容 | 来源 |
|---|---|---|
| **S4-T1** | L1 **组件贡献**：提取器 / SE / 双向 三个 `ΔG`（配对，含 CI）+ 补充架构比较 `cnn vs pure_bigru`（单独标注） | `s4/l1_linear/` |
| **S4-T2** | L2 **标签结构 × 架构**：`G_m(h)` 曲面 + `ΔG_architecture(h)` + **observed crossover interval**（逐模型） | `s4/l2_h*/` |
| **S4-T3** | L3 **架构 × 跨工况**：三模型的 `ΔG_architecture`（或 `R_cross`）+ ΔRMSE(min)，解释为迁移行为 | `s4/l3_*` |
| **S4-T4** | 确定性探针：**Probe A（训练）** ckpt hash / history / val trajectory 逐位；**Probe B（评测）** RMSE / 预测向量 / 不确定性 逐位 | 探针 |

**G 定义（不引入新指标，Stage 2 已冻结）**：`G = 1 − RMSE_model / RMSE_const_mean`。
**hurdle 定义**：`best_const`（`scripts/09` 解析，`scripts/15::_oracle_constant` 复核）——**仅事后诊断**。

---

## 6. 预先注册的裁决规则（承接 STAGE4_MODEL_POSITIONING §6）

- **R1** 主判据 = **配对 `G` 差**（同一 `(condition, bearing)`）；报分布 + 轴承层级自助 CI；**禁裸比 RMSE**。
- **R2** 架构增益必须**先在 linear 下成立**，才允许谈 L2 / L3。
- **R3** L2 读法（**不预设**）：各架构 `G` 在 `h ≥ 0.6` **同步**塌到 0 → 可写
  **「架构优势受标签结构门控」**，**不得**写「架构无效」；**不同步** → 写「不同架构 learning gain 的标签敏感性不同」。
  两种结果都可发表。
- **R4** L3 沿用 Stage 3 **红线 21**（禁「等效 / 无差异」）。
- **R5 · 两层确定性探针（必过，才可引用）**：
  - **Probe A · 训练确定性**：同一 (model, config, fold, seed) **重复训练 ≥3 遍**，逐位比对
    **checkpoint hash** + **training history** + **validation trajectory**。
  - **Probe B · 评测确定性**：同一 checkpoint **重复 inference ≥3 遍**，逐位比对
    **RMSE** + **prediction vector** + **uncertainty outputs**（MC-Dropout 已播种）。
  - 两者**分开证明**：Stage 5 已证 B8 的评测问题，但 Stage 4 引入**三个全新模型** → **重做一次完整闭环**。
- **候选红线 24**（升格）：架构主张必须先有 linear 下的配对 `G` 增益，再谈标签结构依赖与迁移。

---

## 7. 用户六个问题的逐条答案（门槛结论）

| # | 问题 | 答案 |
|---|---|---|
| 1 | 哪些 checkpoint 必须新训？ | **`resnet_bigru` + `se_resnet_gru`（L1/L2 核心）**；L2 再加 `pure_bigru`（h 谱需全因子）；L3 全部三模型的新折。 |
| 2 | 哪些现有 checkpoint 可作验证/基准？ | `s5/` 的 `se_resnet_bigru`、`pure_bigru`、`cnn`（**仅当 data_dir 逐字符相同**）；`matched_fold*_b8` 供 L3 口径核对。 |
| 3 | 协议是否完全一致？ | **是**：data 划分 / seed[42,123,456,789,1011] / loss=weighted_mse / Adam 1e-3 / 150 epoch / patience 15 / eval=rmse 分钟空间 / MC-Dropout 50 已播种。全部继承 `base.yaml`，Stage 4 不改。 |
| 4 | 统计单位？ | 测试轴承实例 `(condition, bearing)`；L1/L2 = 3 单元，L3 = 15 单元。主证据 = 分布 + 效应量 + 轴承层级自助 CI。 |
| 5 | 最终输出哪些表？ | **S4-T1（L1 组件贡献 ΔG + 补充架构比较）/ S4-T2（L2 `G(h)` 曲面 + `ΔG_architecture(h)` + observed crossover interval）/ S4-T3（L3 架构 × 跨工况）/ S4-T4（两层确定性探针）**（§5）。 |
| 6 | 是否会混入 Stage 5 旧 checkpoint？ | **不会**：Stage 4 写入全新根 `s4/`；复用仅在「同 data_dir + 同配置 + 修复后」三条件成立时发生，且**强制逐位命中校验**（红线 24）。 |

---

## 8. 红线（Stage 4 全程有效）

继承 1–27，新增/强调：

28. **Stage 4 结果写入全新根 `s4/`**；禁止写进 `s5/`、`s2_chain/`、`matched_fold*/`。
29. **每个新增模型必须先过两层确定性探针**（R5：Probe A 训练 ≥3 遍 + Probe B 评测 ≥3 遍，逐位一致）才可引用。
30. **架构增益只报配对 `G` / `ΔG` 与其 CI**；禁裸 RMSE 排行（R1）。
31. **L2 读法冻结为 R3 的两种措辞**，不得写「架构无效 / 没用」；零点只报 **observed crossover interval**，
    **禁**精确临界值 `h*`（§3.3）。
32. **L3 最多三模型**（`se_resnet_bigru, resnet_bigru, pure_bigru`）；L3 回答「架构增益能否跨工况保留」，
    **不重复** Stage 3 的 cross-vs-single 问题；禁扩成排行榜。
33. **L1 的 `cnn` 只作补充架构比较**，**不并入**组件 `ΔG` 链，不得与前三行并列为排行榜（§3.1）。
34. **`save_table` 禁止重新引入 pandas 的 `DataFrame` 构造**（本机 CUDA 上下文存在后构造 `DataFrame`
    必段错误，见 §9.1c B10）；如确需 pandas，只能在**模块导入期**（torch/CUDA 之前）导入。
35. **多模型 block 必须可恢复**：任一 unit 硬崩溃不得导致整块重跑；`36` 默认每 unit 一个子进程。

---

## 9. 进入训练的前置动作（冻结后一次性执行）

1. ✅ 写 `scripts/36_run_stage4.py`（镜像 `35_run_stage5.py` 的结构：blocks `l1 / l2 / l3 / l3vm` + `verify`；
   全部写入 `s4/`；`verify` = 复用命中校验，**方案 A 下为可选**）。
2. ✅ 写 `scripts/37_stage4_analysis.py`（`G`/`ΔG` 配对、`G(h)` 曲面、**observed crossover interval**、
   轴承层级自助 CI；按 §3 的四段结构输出 S4-T1…T4）。
   - **§3.4 统计单位已落地**：L3 的 `n_units` 固定为 `(fold, 工况)` = **15**；单元内 3 个 source-task
     在统计前按**中位数**折叠（协议 S2）。已用真实 `matched_fold{k}_b8` 数据构造合成布局验证 → `n=15` ✔。
3. ✅ 写 `scripts/38_stage4_determinism_probe.py`（**两层探针**：Probe A 训练确定性 + Probe B 评测确定性）。
4. **代码静态检查 → dry-run → 跑两层确定性探针 → PASS → 才允许正式训练。**
   - ✅ **静态检查**：`py_compile` 36/37/38 全通过。
   - ✅ **dry-run**：CLI `--help` 通过（GPU 环境）；`36 ... verify` 返回 0（方案 A 的预期行为）；
     `37` 对合成布局端到端跑通（L1 `ΔG` 三边有限、L2 crossover 以**相邻点区间**给出、L3 `n=15`）。
   - ✅ **Probe B 干跑（不训练，复用 Stage 3 真实 ckpt）**：以 `matched_fold1/single` 的
     `se_resnet_bigru` 五种子集成对 `artifacts/data_fold1` 重评测 3 遍 →
     `rmse = 31.395056` **三次逐位相同**，`prediction vector` / `uncertainty` 均逐位相同；
     且与冻结的 `matched_fold1_b8/single/results_corrected.csv`（`31.3951`）**全精度一致**
     ⇒ ① 评测口径正确（集成 + 分钟空间 + clip）；② **评测确定性成立**；③ **A10 命中校验通过**。
   - ✅✅ **Probe A（训练确定性）已跑完并通过（2026-10-08 20:40）**——用户授权的最小范围：
     `resnet_bigru` + `se_resnet_gru`，folder=1、condition=cond1、`seed=42`、**各独立重复 3 次**。

     | 模型 | sha256（3 次） | hash | history | val_traj | **final weights** | n_tensors | 结论 |
     |---|---|---|---|---|---|---|---|
     | `resnet_bigru` | `2457e818…` ×3 同 | ✅ | ✅ | ✅ | ✅ | 58 | **PASS** |
     | `se_resnet_gru` | `c7642f1f…` ×3 同 | ✅ | ✅ | ✅ | ✅ | 54 | **PASS** |

     - 逐位比对四项：checkpoint `sha256` / `history` / `best_val_loss`·`best_epoch`·`epochs_run` /
       **逐张量最终权重**（58、54 个张量，`np.array_equal` 全等，`first_diff=None`）。
     - 早停轨迹也逐位一致：`resnet_bigru` 三次均 `early stop @51, best 36, val 0.043708`；
       `se_resnet_gru` 三次均 `early stop @23, best 8, val 0.046066`。
     - 产物 `artifacts/reports/stage4_probe/stage4_probe.json`（`probe: "a"`，OVERALL **PASS**）；
       仅 6 个 ckpt（2 模型 × 3 重复 × seed42），**未触碰 885 规模的任何训练**。
     - ⇒ **R5 双层的「Probe A 训练确定性」在新增架构上成立**。
   - ⛔ **闸门状态**：Probe A + Probe B 均 PASS ⇒ **正式 Stage 4 训练已解锁**，但按用户指令
     **到此停止并汇报，不自动进入** 885 ckpt 全量实验。
5. **训练完成后才改论文**（用户明确：不边跑边改）。

### 9.1 GPU 启动记录

- **2026-10-08 20:39–20:40**：首次启动 GPU，**仅执行 Probe A**（用户授权范围）。全程 27 秒，
  仅训练 6 个 ckpt（2 模型 × 3 重复 × seed42、fold1/cond1）——**远未触及全量实验**。
- Probe A + Probe B **双 PASS** ⇒ 闸门解锁。
- **2026-10-11 15:20**：首次启动 Phase 1（单进程 `l1,l2`）→ **EXIT=139 段错误**，定位为 §9.1c 的 B10
  （pandas 在 CUDA 上下文后构造 `DataFrame`）。修复（标准库 csv + 导入期 pandas 守卫 + 每 unit 子进程）后重启。
- **2026-10-11 16:01–16:11**：**L1 完成**（5 unit，9m12s，EXIT=0）→ 75 ckpt。
- **2026-10-11 16:19–16:58**：**L2 完成**（20 unit，39m04s，EXIT=0）→ 300 ckpt。
- **2026-10-11 16:59–17:00**：`baselines` 块（12 unit，秒级）→ `l1_linear` 与 `l2_h*` 各自的
  `constant_mean` / `constant_one` 落地 `s4/`（`G` 的分母，自包含，不再跨根取 `s5/`）。
- **Phase 1 总计 25 unit（+12 基线）/ 375 + 12 ckpt ⇒ 与 §4 预测量完全一致。**

**Phase 1 运行后核验（2026-10-11，全绿）**

| 核验项 | 结果 |
|---|---|
| 单元完成度 | **25/25**（L1 5 + L2 20），无缺失、无失败 unit |
| ckpt 数 | **375**（L1 75 + L2 300）+ 基线 12 ⇒ 精确等于冻结预测 |
| `36 ... verify` | **PASS**：9 个复用行 × 9 列与冻结 Stage 5 **逐位相同** |
| 两层确定性探针 | **Probe A + B OVERALL PASS**（3 模型：`resnet_bigru`/`se_resnet_gru`/`pure_bigru`） |
| **`G(h)` 复现** | `s4` 的 `se_resnet_bigru` 与冻结 `s5` 的 `G(h)` 在 **全部 6 个 h 上 diff = +0.0000** ⇒ 加固 A10 |
| **`const_mean` 地标** | `s4` 与 `s5` 在 **全部 18 个值**（6 根 × 3 工况）上完全一致 |

> `G(h)` 的逐位复现是本次最强的交叉校验：`s4/se_resnet_bigru` 是**独立新训**（另一析出、另一进程），
> 却与冻结 `s5` 完全重合 ⇒ 同时验证了数据管线、配置继承、评测口径与 **A10**。

### 9.1b 正式训练启动前核验（2026-10-10，**仅核验，未训练**）

| 核验项 | 结果 |
|---|---|
| 脚本版本 | `36` sha256 `943ae4a0…` / `37` `fd89da99…` / `38` `d6770ab2…`（均已编译通过） |
| 冻结文档 | `STAGE4_PROTOCOL_FREEZE.md` sha256 `d8a677b1…` |
| 模型名单 | L1 `pure_bigru,resnet_bigru,se_resnet_bigru,se_resnet_gru,cnn`；L2 去 `cnn` × h0.5–0.9；L3 三模型 — **与冻结一致** |
| 配置（真实 load） | L1/L3/L3vm `label.strategy=linear`；**L2 `piecewise`+`healthy_ratio`**（L2 即标签结构谱，正确）；`weighted_mse` / 150 epoch / patience 15 / lr 1e-3 / headline `rmse` / mc_samples 50 — **全部继承 base.yaml** |
| 随机种子 | `[42,123,456,789,1011]`（四类配置实测一致） |
| 输出路径 | `artifacts/reports/s4/` — **当前不存在**（全新根，方案 A 无污染） |
| 依赖数据 | `data_linear` / `data_h{0.5..0.9}` / `data_fold{1..5}` / `data_vm_fold{1..5}` — 各 12 npz，**齐全** |
| **工程量（修正）** | **2175 ckpt / ≈7.0 h 训练 / ≈9.2 h 含评测**（原估 885 / 2–3h **低估**，见 §4） |

> `linear_rul(n_samples)` 不含 `healthy_ratio` 形参 ⇒ L1/L3 的 `healthy_ratio: 0.8` **确为惰性**，非隐患。

**尚未开始的**：Stage 4 正式全量训练（L1/L2/L3/l3vm，2175 ckpt，方案 A）。

### 9.1c **B10 —— `save_table` 的 pandas 段错误（2026-10-11 发现并修复，属实验基础设施缺陷）**

**现象**：首次启动 Phase 1（`--blocks l1,l2`，单进程）在 1m43s 后 **EXIT=139（SIGSEGV）**，
`l1_linear/pure_bigru` 已完整产出（15 ckpt + 4 表），崩溃发生在切换到 `resnet_bigru` 之前。

**根因（已最小化复现）**：本机 `pandas 3.0.3` 在 **CUDA 上下文已创建** 之后再
**构造 `DataFrame`** 会段错误：`import pandas` 本身安全，`pd.DataFrame(...)` 崩。
```
import torch
torch.zeros(4, device="cuda"); torch.cuda.synchronize()   # 建立 CUDA 上下文
import pandas as pd
pd.DataFrame([{"a": 1.5}])                                 # ← SIGSEGV
```
与项目代码无关，是环境级原生库冲突（pandas 3.0.3 / torch 2.6.0+cu124 / Windows）。
`rulbench/evaluation/report.py::save_table` 原先在 `run()` 尾部惰性 `import pandas` 并
`pd.DataFrame(records).to_csv(...)`——**CUDA 已在上下文里**，于是**每个 block 的第一个模型
写完 CSV 就杀死整个进程**，多模型 block 永远跑不完。旧架构在单模型脚本下从不触发，所以 Stage 5 未暴露。

**修复（行为保持）**：
1. `save_table` 改为**纯标准库 `csv`** 写出（BOM `utf-8-sig`、列序按插入序）。已逐字节比对
   与旧 pandas 输出**完全一致**（表头 + 数据行）。
2. `scripts/36_run_stage4.py` 在 **import torch 之前** 防御性 `import pandas`，杜绝复发。

**证据**：修复前同一顺序复现 EXIT=139（`pure_bigru→resnet_bigru` 与 `resnet_bigru→pure_bigru` 均崩，
`resnet_bigru→resnet_bigru` 亦崩 ⇒ 与模型身份无关，**第二次 `save_table` 必崩**）；
修复后同一脚本 **EXIT=0** 两模型跑完；单模型 3 条件下 RMSE **10.75 / 68.22 / 32.40**
与崩溃前那次**逐位相同** ⇒ 修复不扰动结果。

**顺带的健壮性改进**：`36` 增加 `--one <unit>` 与**监督式（每 unit 一个子进程）** 默认执行：
任何 unit 硬崩溃（段错误 / OOM）最多损失一个 unit，重跑自动跳过已存在的 ckpt
（`train_missing=True`）。unit 数：L1 = 5、L2 = 20、L3 = 30、L3vm = 30。

**红线 34**：`save_table` **禁止**重新引入 pandas 的 `DataFrame` 构造；如确需 pandas，
必须在**模块导入期**（任何 CUDA 上下文建立之前）导入。B10 属**实验基础设施修复**，
与 B8/B9 同级，**不得写入 Contributions**。

### 9.2 Stage 4 数据依赖核查（本次实测，全部就位）

| 依赖 | 状态 |
|---|---|
| `artifacts/data_linear`、`data_h{0.5..0.9}` | ✅ 各 12 npz |
| `artifacts/data_fold{1..5}` | ✅ 各 12 npz |
| `artifacts/data_vm_fold{1..5}` | ✅ 各 12 npz |
| `configs/linear.yaml`、`configs/matched_folds/fold{1..5}.yaml`、`configs/matched_folds_vm/fold{1..5}.yaml` | ✅ 全部存在 |

> `run_single` glob `single_cond*.npz`、`run_cross` glob `cross_*.npz`；`linear.yaml` 的
> `paths.report_dir` 与 `experiments.single.models` **不会泄漏**进 Stage 4——
> `36` 对每个模型显式传 `out_dir` 与 `models=[model]`（已代码级核实）。

### 9.3 **Phase 2 = B'（减配 L3）—— 用户 2026-10-11 定案**

**用户否决**「全量 900+900 ckpt / ≈7h」的 L3+L3vm，改采**减配 L3（B'）**。理由（用户原话要点）：
S4-T1 已答「架构增益是否存在」（答：不存在）；原 L3 问题「更复杂架构的增益能否迁移」已削弱为
「若没有稳定单工况增益，跨工况能否恢复」。**但整段跳过 L3 会留下审稿可见的缺口**——
「架构因子不存在」与「架构因子无法迁移」是两句不同的话。故保留**最小验证单元**。

**B' 冻结范围（不得扩）**：

| 项 | 范围 |
|---|---|
| 模型 | `pure_bigru` / `resnet_bigru` / `se_resnet_bigru`（**3 个**，红线 32） |
| 折 | 5 折（原 `matched_folds/fold{1..5}.yaml`，原 `cross_tasks`，原指标） |
| **single 臂** | **全量**：3 模型 × 5 折 × 3 工况 = **45 ckpt** |
| **cross 臂** | **仅三个双源任务** `1+2→3` / `1+3→2` / `2+3→1`（3 目标 × 5 折 = 15 ckpt/model）= **45 ckpt** |
| **L3 合计** | **90 ckpt**（**非** 900） |
| **不跑** | **L3vm**、单源 `1→2/1→3/2→1/2→3/3→1/3→2`、其余六个 cross task |

**为何 cross 用双源而非单源**：「两个已知工况迁移到未知目标」比单源更贴合论文的问题表述
（`run_cross` 原生的九个任务即 = 六个单源 + 三个双源）。

**为何不跑 L3vm**：Batch 1（T1）已显示 SE ≈ 0、ResNet ≈ 0、BiGRU ≈ 0，**训练量已不是主要混杂**
（Stage 3.2 已在主线上证明训练量不是 `R_cross` 的主导解释）⇒ 此时再跑 L3vm 属过度防御。

**实现（无新增依赖）**：
- `scripts/36_run_stage4.py`：常量 `L3_CROSS_TASKS = ['cross_1+2_to_3','cross_1+3_to_2','cross_2+3_to_1']`；
  `block_l3min()`；`run_one(..., cross=False, datasets=None)` 透传 `datasets`；`BLOCKS["l3min"]`。
  `_units('l3min')` = **30 unit**（3 模型 × 5 折 × {single, cross}）。
- `rulbench/pipelines/run.py`：`_datasets_for(cfg, data_dir, pattern, include=None)` 支持按名过滤
  （完整 stem 或去掉 `cross_` 前缀的任务名皆可）；`run_single`/`run_cross` 由 kwargs 取 `datasets` 转发；
  **`include=None` 完全复现原 glob ⇒ 既有全部调用点行为不变**（不改变 Stage 5 / Stage 3 产出）。

**工程量实测修正（重要）**：用户原估 50–80 min 基于「1 unit = 1 模型」；实际
**1 unit = 1 模型 × 1 折 × 1 臂 = 15 ckpt**（3 工况或 3 任务 × 5 种子）。实测：
`single` ≈ 87–102 s、`cross` ≈ 292–310 s ⇒ 每折 ≈ 3×(95+300) ≈ 20 min ⇒ **5 折 ≈ 100 min**
（非 900 ckpt，但也非 50–80 min）。**结论仍远小于全量 7h。**

**预先注册裁决（承 §6，跑前写死）**：
- 若 `R_cross ≈ 1`（CI 跨 1）⇒ 只能写「**架构复杂度无法抵抗工况迁移难度**」；
- 若 `resnet_bigru > pure_bigru` 且 CI 不跨 0 ⇒ 才进入训练量讨论（并须并报 `n_train` 与 `ΔRMSE`）；
- **禁**「等效 / 无差异」（红线 21）；**禁**裸 RMSE 排行（红线 30）；`R_cross>1` 单独不构成证据（协议 S5）。

**B11 —— 监督式子进程共享臂目录导致结果表被截断（2026-10-11 发现并修复，属实验基础设施缺陷）**

**现象**：B' 首轮跑到 15/30 unit 时核验产物，发现 `l3_fold{k}/single/results_corrected.csv` 与
`.../cross/...` **都只有 3 行、只含最后一个模型**（fold1 剩 `pure_bigru`、fold3/single 剩 `resnet_bigru`
而 fold3/cross 剩 `se_resnet_bigru`），另外两个模型的 6 行被**静默丢弃**。
checkpoint 完全正常（每臂 45 个 = 3 模型 × 15），**只有结果表被截断**。

**根因**：`block_l3` / `block_l3min` 让**三个模型共用同一个臂目录**（`l3_fold{k}/single`），
而监督式驱动把**每个模型拆成独立子进程**。`run()` 尾部 `save_table` 是**覆盖写**，于是每个模型
写完自己的 3 行就把前两个模型的行**截掉**。旧架构（L1/L2）给每个模型一个**独立目录**
（`l1_linear/<model>`、`l2_h{h}/<model>`），所以从不触发——**Phase 1 的 L1/L2 产物不受影响**，
这也与 Phase 1 的 `G(h)` diff = +0.0000 交叉验证一致。

**修复（行为保持，opt-in）**：
1. `rulbench/evaluation/report.py::save_table` 新增 `merge_keys=None` 形参。非 `None` 时先读回已有 CSV，
   按 key（此处 `("dataset","model")`）剔除将被覆盖的行，再与本次记录**合并**并按 key 排序写回。
   **`merge_keys=None`（默认）完全保留原覆盖语义 ⇒ 所有既有调用点（Stage 5 / Stage 3 / L1 / L2）行为不变。**
2. `rulbench/pipelines/run.py::run` 新增 `merge_tables=False` 形参，转成
   `merge_keys=("dataset","model")`（`summary_by_model.csv` 用 `("model",)`）传给 `save_table`。
   `run_single`/`run_cross` 经 `**kwargs` 透传，默认 `False` 不影响他人。
3. `scripts/36_run_stage4.py::run_one` 新增 `merge=False`，仅 **L3/L3vm 臂目录**传 `merge=True`；
   **关键：`_units()` 里另有一份单元清单（`--one` 实际走的是这份）也必须同步**——首版只改了
   `block_l3*` 而漏了 `_units()`，导致修复"无效"（表现为 CSV 仍被截断）。两处均已改。

**证据（逐项实测）**：
- 单元测试：`merge_keys=None` 覆盖语义不变；`merge_keys=("dataset","model")` 三次写入累积 9 行、
  顺序确定；同 key 重写**原地更新不重复**（⇒ 幂等、可重跑）。
- 端到端：对 fold1/single 依次单跑三模型 → CSV 由 3 行累加到 **9 行 / 3 模型**。
- 数值不变：重评测得到的 `se_resnet_bigru` cond1 = **31.3951**，与冻结 Stage 5 的 `31.395056`
  及 §9.1b 的 Probe B 一致 ⇒ 修复不扰动结果，**A10 仍成立**。
- 重跑**复用全部已有 ckpt**（`train_missing=True`），评测耗时 18–39 s/unit（含训练时约 95–300 s）。

**红线 37**：**多个模型共享同一臂目录时，写表必须 `merge_keys` 合并，不得覆盖**；
新增 block 若沿用共享臂目录（L3/L3vm 风格）必须同时在 `block_*` **与** `_units()` 两处传 `merge=True`。
B11 属**实验基础设施修复**，与 B8/B9/B10 同级，**不得写入 Contributions**。

### 9.4 B' 首轮（B11 前）的处置

首轮 15/30 unit 的 **checkpoint 全部有效并保留**（`s4/l3_fold{1,2}` 完整、`fold3` 部分）；
**结果表全部作废并删除重生成**（截断不可修复，须以合并写入重建）。重跑不重复训练。

### 9.5 **B' 运行结果与 S4-T3 裁决（2026-10-11 完成）**

**工程量（实测）**：30/30 unit，**零失败**，评测+训练合计 **67.7 min**（ckpt 复用省下 folds1–3 的训练）。
`s4/` 总计 **825 ckpt**（L1 75 + L2 300 + **L3 450**）；`36 verify` **PASS**（9 复用行 × 9 列与冻结 S5 逐位相同）。

**S4-T3（unit = `(fold, target condition)`，n = 15）**：

| Model | n | `R_cross` 中位 | 95% CI | 跨 1? | `ΔRMSE` 中位 (min) | 95% CI | 跨 0? | cross 更优 |
|---|---|---|---|---|---|---|---|---|
| SE-ResNet-BiGRU | 15 | **1.287** | [1.151, 1.817] | 否 | +14.59 | [+8.62, +71.53] | 否 | 13% |
| ResNet-BiGRU | 15 | **1.202** | [1.012, 2.118] | 否 | +11.06 | [+1.09, +23.19] | 否 | 27% |
| **BiGRU** | 15 | **0.991** | [0.792, 1.690] | **是** | −8.23 | [−19.79, +22.76] | **是** | 60% |

**裁决（严格按 §9.3 预注册；不得临时改写措辞）**：

1. **最简架构 BiGRU 的 `R_cross` CI 跨 1**（0.991 [0.792, 1.690]）⇒ 对它而言**跨工况未表现出稳定代价**。
2. 两个更深架构（ResNet-BiGRU、SE-ResNet-BiGRU）的 `R_cross` CI **不跨 1**，`ΔRMSE` CI **不跨 0**
   ⇒ 对它们而言**跨工况的代价是稳定的（`R_cross > 1`，即跨工况显著更差）**。方向是**"更深 ⇒ 跨工况更差"**。
3. **但不能写成"SE/ResNet 有害"或排行榜**（红线 30/36）。配对核验：
   - `pure_bigru → resnet_bigru`（extractor）：`R_cross` 配对差中位 **+0.410 [0.021, 0.602] 不跨 0**，
     **但方向不一致**——15 个单元里只有 **3/15** 个是 resnet 更优，逐单元从 **+3.41 翻到 −2.83**。
   - `resnet_bigru → se_resnet_bigru`（SE）：配对差中位 **+0.067 [−0.405, +0.348] 跨 0**，
     7/15 更优 ⇒ **无稳定 SE 效应**。
   ⇒ 结论上限：**"架构复杂度未表现出对工况迁移的稳健优势；在有代价的架构上，该代价随深度略增"**，
   **禁**"深层架构无法迁移"、**禁**"等效/无差异"（红线 21）。

4. ⚠ **协议口径红线（A9）**：B' 的 cross 臂是**双源**（`1+2→3` 等），而 Stage 3 的 `R_cross` 是
   **单源**（`1→2` 等）——**两者数字不可直接相比**，只能各自内部比较。S4-T3 的 1.287/1.202/0.991
   **不得**与 Stage 3 的 1.086 并列成"跨工况更差/更好"的结论。
5. ⚠ **尺度陷阱**：`ΔRMSE` 中位被低 RMSE 折主导（如 fold4/cond2 的 single RMSE 仅 1.69–6.30 min，
   比值 6.5× 却是 5–10 min 的绝对差）。**必须并报 `R_cross` 与 `ΔRMSE`**(协议 S5)，不得只报比值。

**S4-T1 / S4-T2 未受 B11 影响**（逐位复现 B11 前数值）：`extractor` +0.074、`SE` −0.023、
`bidirectional` +0.108；crossover 仍为 h=0.5→0.6。

**Stage 4 四段全部收口**：T1 = 无稳定组件增益；T2 = 标签结构是主因、独立复现 h≈0.55–0.60 零点；
T3 = 架构复杂度未表现出对迁移的稳健优势；T4 = 可复现性（两层探针 + verify PASS）。

*End of STAGE4_PROTOCOL_FREEZE.md*
