# STAGE3_EVIDENCE_CHAIN.md —— Stage 3：跨工况泛化（严格配对 / 训练量匹配 / 15 单位统计）

> **状态**：**已完成**（3.1 严格配对 ✅、3.3 单位统计 ✅、3.2 训练量匹配臂 ✅ —— 训练 300/300
> checkpoint、构建 60/60 验证通过、配对聚合与零学习对照均已产出）。裁决见 §5.5。
> **上游**：`audit/STAGE1_PROTOCOL_FREEZE.md`（F1–F5、红线 22 条；**红线 19–22 为本阶段预先注册**）、
> `audit/STATISTICAL_UNIT_PROTOCOL.md`（S1–S10）、`audit/STAGE2_EVIDENCE_CHAIN.md`（Stage 2 已封板）。
> **模型**：`se_resnet_bigru`（与 Stage 2 同一 backbone，**不做模型间排名**）。
> **产物**：`docs/stat_unit_rcross.md`、`artifacts/reports/stat_unit_rcross.json`（3.1/3.3）；
> `artifacts/data_vm_fold{1..5}/`、`configs/matched_folds_vm/fold{k}.yaml`、
> `artifacts/reports/vm_fold{1..5}/{single,cross}`（3.2）；
> `docs/volume_matched_design.md`、`docs/vm_vs_unmatched.md`、`docs/vm_vs_unmatched_constmean.md`、
> `docs/stat_unit_rcross.md`、`docs/stat_unit_rcross_vm.md`、`paper/figures/fig_stage3_vm_rcross.png`。
> **复现命令**：§7。

---

## 1. 本阶段回答的科学问题

Stage 3 是全文主战场。问题**不是**「跨工况模型好不好」，而是：

> **在控制了测试对象与训练样本量之后，跨工况数据是否提供了超出目标工况内部数据的
> 额外退化信息？**

三条必做（用户 2026-10-07 指令）：

| 编号 | 内容 | 状态 |
|---|---|---|
| **3.1** | **严格配对**：同一 `(condition, bearing)` 上比 `RMSE_cross` vs `RMSE_single`，报 `R_cross` 并列 `ΔRMSE`(分钟) 与 `n_train` | ✅ 已完成（= M1 五折） |
| **3.2** | **训练量匹配臂**：使 single / cross 的训练样本量严格相等 —— **本阶段的核心新工作** | ✅ 已完成（§5） |
| **3.3** | **15 个测试实例层面的统计**：单位 = `(fold, 目标工况)` = `(condition, bearing)`，n = 15；轴承层级自助 CI | ✅ 已完成 |

---

## 2. 口径纪律（Stage 1 冻结 + Stage 3 新增）

1. **统计单位 = 测试轴承实例** = `(fold, target)` = `(condition, bearing)`，**n = 15**
   = XJTU-SY 全部 15 个轴承。单元内 3 个 source-task 变体是**重复测量** →
   `45 cell = 15 × 3`，**不是 45 个独立样本**；cell 级 p 仅 `EXPLORATORY_ONLY`。
2. **`R_cross` 必须并列 `ΔRMSE`（分钟）与 `n_train`**。
3. **跨协议对比只用未校正口径**（A9）：`median_scaling_factor` **只施加于跨工况臂**
   （单工况恒 1.0，跨工况 1.3–2.5）→ 校正口径下两臂不同口径，**不可**用于跨协议对比。
4. **`best_const` 只作事后诊断**；可部署零学习基准是 `const_mean`。
5. **（新，红线 19）训练量匹配臂的构接口径**：单元内（1 单工况 + 3 跨工况）**四条臂一律下采样到同一
   `K = min(n_train)`**；**val / test 与匹配前逐位一致**；**scaler 在裁剪后训练集上重拟合**；
   **mRMR 索引不动**（快照级、只依赖「哪几个训练轴承」，而「哪个工况」正是处理变量）。
6. **（新）已知残留不对称（必须披露）**：本设计匹配的是 **train** 样本量；
   两臂的 **val 集大小仍不相等**（如 fold3/target2 的单工况 val 仅 11 窗口，跨工况 91），
   早停因此不是等信息的。这不影响「训练样本量」这一处理维度的配对，但限制了
   「训练动力学完全等价」的强主张。
7. **（新，红线 20）裁决规则预先注册**（见 §5），禁止事后择结论。
8. **（新，红线 21）结论上限**：只能写「**未表现出稳定一致的优势**，效应方向随留出轴承与任务变化」；
   **禁止**「等效 / 无差异 / 证明无增益」。

---

## 3. 设计总览

### 3.1 五折留一轴承设计（`scripts/19_make_matched_folds.py`）

fold `k`：**test = 每个工况的轴承 `k`**，val = 轴承 `(k mod 5)+1`，train = 其余三个轴承。
单工况臂与跨工况臂**共用同一划分**，因此

* 两臂的 **测试轴承逐位相同** → 常数地基相同，`R_cross` 无需重标定；
* **唯一变化 = 训练数据来自哪个工况**。

跨工况任务 = 对每个目标工况 `t` 的 3 个源变体（两个单源 + 一个双源），
5 折 × 3 目标 × (1 单 + 3 跨) → **15 个单元、45 个 cell**。

### 3.2 训练量混杂（`scripts/22` + 本轮复算）

匹配前（**五折设计口径**，非 pilot）：

| 目标工况 | cross/single `n_train` 比值范围 | 方向 |
|---|---|---|
| cond1 | **1.85× – 24.50×** | 跨工况**恒多** |
| cond2 | 0.26× – 7.21× | 混合 |
| cond3 | **0.048× – 0.556×** | 跨工况**恒少** |

整体 **0.048× – 24.50×（510 倍跨度）**，且**符号随目标工况翻转** → 不抵消。
⇒ 匹配前的 `R_cross` 混淆「换了工况」与「训练量变了」，**只能描述、不能因果**。

---

## 4. Stage 3.1 / 3.3 —— 严格配对的未匹配结果（基准）

来源：`docs/stat_unit_rcross.md`、`artifacts/reports/stat_unit_rcross.json`
（模型 `se_resnet_bigru`，**未校正口径**；**已按 §4.1 的 B9 修正统一到当前评测版本**，
输入为 `artifacts/reports/matched_fold{k}_b8/`）。

**单元级（n = 15）**：median `R_cross` = **1.086**（95% CI [0.889, 1.281]，**跨 1**）、
IQR 0.468、range 0.654–3.002、跨工况更优 **33%**；
median `ΔRMSE` = **+3.43 min**（CI [−1.83, +12.61]，**跨 0**）。

| 目标工况 | n | median `R_cross` | 95% CI | 跨工况更优 | median `ΔRMSE` (min) | 训练量比 |
|---|---|---|---|---|---|---|
| cond1 | 5 | 1.058 | [0.761, 1.281] | 40% | +0.50 | 恒 **多** 1.85–24.5× |
| cond2 | 5 | 0.955 | [0.654, 3.002] | 60% | −1.83 | 混合 |
| cond3 | 5 | **1.180** | [1.019, 1.425] | **0%** | +166.19 | 恒 **少** 0.048–0.556× |

* 折中位数（未校正）：0.889 / 0.955 / 1.281 / 1.357 / 1.058 → **> 1 者 3/5**；折间跨度 **42.3%**。
* `|median − 1| = 8.6%`  **≪** 折间跨度 `42.3%`。
* 统计（已降级）：cell 级 Wilcoxon `p = 0.153`（**探索性**，45 cell 非独立）；
  单元级 `p = 0.252`（n = 15，**indicative**）。
* 校正口径（**仅敏感性，不可跨协议用**）：median 1.269（CI [1.082, 1.612]），跨工况更优 13%。

**读法**：三个目标工况中，只有 **cond3 的 median `R_cross` > 1**，而 cond3 的跨工况臂
训练量恰好**只有 0.05–0.56×** ⇒ 该超额**首先由训练量解释**，不能当作域漂移证据（F4/红线 3）。
这正是必须做 3.2 的原因。

### 4.1 ⚠ 基线一致性问题 **B9**：上表的分子/分母评测版本不一致（本轮发现并修正）

上表的 `R_cross` = `RMSE_cross / RMSE_single`，而这两个数**不是同一代码版本产出的**：

| 产物 | mtime | 相对 B8 修复（`evaluation/inference.py` = 10-07 **16:21**） |
|---|---|---|
| `matched_fold{k}/single/results_*.csv` | 10:51–12:09 | **之前**（MC-Dropout 评测**未播种**） |
| `matched_fold{k}/cross/results_*.csv` | 16:25–16:30 | **之后**（已播种） |

**证据链**（三步定位，全部可复算）：

1. fold1 的 `cond1` 单工况 RMSE 由 **31.4329 → 31.3951**，但它的训练集
   **K = 239 = 自然窗口数**（不应被裁剪）。
2. `data_fold1/single_cond1.npz` 与 `data_vm_fold1/single_cond1.npz` 逐数组比对：
   `train`/`val`/`test` 全部数组 + `mrmr_indices` **`array_equal = True`**；
   cond1 的 5 个 seed **checkpoint SHA-256 逐字节相同** ⇒ 数据与训练都相同，
   差异**只能在评测侧**。
3. mtime 比对（上表）⇒ 原因 = **B8**。

**量级**：用当前代码重评测 fold1 单工况臂（复用 checkpoint，纯推理）→
**12 行中 9 行完全不变**（trivial 基线无 MC-Dropout），变的 3 行全是 `se_resnet_bigru`，
**max |ΔRMSE| = 0.1203%**、mean −0.0047%（可复现，非 run-to-run 噪声——噪声实测 0.0%）。

**处置**（`scripts/33_reeval_checkpoints.py`）：用**当前代码**从既有 checkpoint 重评测
（依据 A10，与完整 train→eval 逐位一致），输出到 **新目录** `artifacts/reports/matched_fold{k}_b8/`，
**不就地覆盖**原始目录；审计留痕写入 `artifacts/reports/reeval_audit.json`。
本阶段所有对照（§5.3）**一律使用 `_b8` 目录**作为未匹配基准。

⇒ **B9 不改变任何裁决**（0.12% ≪ 目标效应），但**必须统一口径**，否则配对差里混入一处
系统性版本偏差。已登记为冻结文件**红线 23**。

---

---

## 5. Stage 3.2 —— 训练量匹配臂（核心）
### 5.1 构接（红线 19）

`scripts/28_make_volume_matched.py`：对每个 `(fold, target)` 单元，

```
K = min( n_train(single_cond_t),
         n_train(cross_{a}_to_t), n_train(cross_{b}_to_t), n_train(cross_{a+b}_to_t) )
```

四条臂**一律**下采样到恰好 `K` 个训练窗口：

* **逐轴承按比例分配**（最大余数法），**轴承内按时间等距**取窗
  —— 等价于「抬高有效 stride」，保留全长寿命阶段覆盖与轴承间相对权重；
* **scaler 在裁剪后的训练集上重拟合**（该臂端到端只见过自己的 `K` 个样本）；
* **mRMR 索引不动**（快照级、与窗口数无关；「哪个工况」是处理变量）；
* **val / test 与匹配前逐位一致** → 常数地基不变，`R_cross` 无需重标定。

实测 `K ∈ [204, 349]`（15 个单元各一个 `K`；由最短命的 cond1 训练量决定）。

### 5.2 构建验证（`artifacts/reports/vm_build_verification.txt`）

| 检查 | 方法 | 结果 |
|---|---|---|
| 训练量精确相等 | 逐 npz 读 `train/ratio` 长度 vs `K` | **60/60 等于 `K`** |
| val / test 未变 | 与 `data_fold{k}` 对应数组 `np.array_equal` | **60/60 逐位相同** |
| 保留窗口是原窗口 | 每个保留窗口的 `X` 字节串均存在于原 train 集 | **60/60 全部命中** |

### 5.3 结果（匹配后）· `docs/vm_vs_unmatched.md`

模型 `se_resnet_bigru`，**未校正口径**（A9），n = 15 单元。数字来自
`artifacts/reports/vm_vs_unmatched.json`；逐单元表见 `docs/stat_unit_rcross_vm.md`。

**单元级对照（n = 15）**

| 口径 | median `R_cross` | 95% CI（轴承层级自助） | IQR | range | 跨工况更优 |
|---|---|---|---|---|---|
| 未匹配（`_b8`） | **1.086** | [0.889, 1.281] | 0.468 | 0.654–3.002 | 5/15 |
| **训练量匹配** | **1.158** | [0.885, 1.402] | 0.517 | 0.755–3.160 | 6/15 |

* **配对差 `dR = R_matched − R_unmatched`**：median **−0.004**
  （95% CI [−0.093, 0.158]，**跨 0**）、IQR 0.255、range −0.497…+0.971；
  Wilcoxon `p = 0.847`（n = 15，**indicative**）。
* 匹配后 `n_train(cross)/n_train(single)` = **1.00×（15/15 单元）** ⇒ 处理维度已严格相等。
* 折间跨度（`(max−min)/mean` of fold-medians）：未匹配 **42.2%** → 匹配后 **48.7%**；
  `|median−1|` 未匹配 8.6% → 匹配后 **15.8%**。两者**均小于**折间跨度。

**按目标工况分层（决定性读法）**

| 目标 | n | median `R_cross` 未匹配 | 匹配后 | `dR` | `|median_matched−1|` | 层内折间跨度 | 训练量方向 |
|---|---|---|---|---|---|---|---|
| cond1 | 5 | 1.058 | **0.902** | −0.156 | 9.8% | 55.9% | 跨工况**恒多**（15.2×）→ 匹配=**减量** |
| cond2 | 5 | 0.955 | **1.158** | +0.202 | 15.8% | 157.9% | 跨工况**恒多**（5.2×）→ 匹配=**减量** |
| cond3 | 5 | 1.180 | **1.402** | +0.221 | 40.2% | 74.1% | 跨工况**恒少**（0.19×）→ 匹配=**增量** |

（跨度 = `(max−min)/mean`，与本文件 §4 及 `docs/stat_unit_rcross_vm.md` 口径一致。）

**零学习对照（`constant_mean`，`docs/vm_vs_unmatched_constmean.md`）**

同一测试轴承、同口径下，常数预测器（预测训练标签均值）的 `R_cross`
median **1.018 → 1.018**、配对 `dR` median **−0.000**（Wilcoxon `p = 0.978`）。三重含义：

1. **两臂测试集逐位相同** —— 否则常数预测器不会在两臂给出相同 RMSE（这也独立复核了 §5.1 的 val/test 未动）；
2. **训练量裁剪对常数预测器无影响** —— 管道自洽，裁剪只改「用了多少窗口」，不改「窗口来自哪个工况」；
3. **纯标签分布漂移只有 ≈1.8%** —— `se_resnet_bigru` 的 `R_cross`≈1.16 中，超出常数预测器的部分
   （≈1.16/1.018 ≈ 1.14×）才是**学习侧**迁移效应，**量级很小**。

⇒ 分层 + 零学习对照共同说明：`R_cross` 的偏离**既不是标签漂移**（≈1.8%），
**也没有被训练量匹配系统性消除**（配对 `dR`≈0）。

### 5.4 裁决规则（红线 20，**预先注册**）

1. 主判据 = 单元级（n = 15）`median R_cross`（**未校正**）在匹配前后的**配对差** `dR`，
   及**按目标工况分层**的中位数移动。
2. **若某目标工况的 `median R_cross` 匹配后落入「与 1.0 不可区分」区间**
   （`|median − 1|` 小于该层的折间跨度）→ 把匹配前的超额**归因于训练量**，
   **不得**归因于域漂移。
3. 只有匹配后**仍稳定 > 1 且超出折间跨度**时，才可提出**域效应**，
   且必须与训练量讨论**并列**陈述。
4. 无论结果如何，结论上限见红线 21（禁止「等效 / 无差异」）。

### 5.5 裁决（红线 20 逐条应用）

**规则 1（主判据 = 单元级 median `dR` + 分层移动）**
单元级配对 `dR` median = **−0.004**、CI **跨 0**、`p = 0.847`
⇒ **匹配训练量未产生稳定的系统性变化**。
分层移动为 −0.156 / +0.202 / +0.221，**方向不一致**。

**规则 2（匹配后落入「与 1.0 不可区分」→ 归因训练量，**不得**归因域漂移）**

| 目标 | `|median_matched−1|` | 折间跨度 | 判定 |
|---|---|---|---|
| cond1 | 9.8% | 55.9% | **不可区分** ✓ |
| cond2 | 15.8% | 157.9% | **不可区分** ✓ |
| cond3 | 40.2% | 74.1% | **仍不可区分** ✓ |

⇒ 三个目标工况匹配后**均不能**把偏离归因于**域漂移**。

**规则 3（只有匹配后仍稳定 > 1 且超出折间跨度，才可提域效应）**
**无**目标满足 ⇒ **不提域效应**。

**方向性反证（对「归因训练量」的进一步反驳，关键）**
若 `R_cross` 的超额真由训练量驱动，则按匹配方向应看到
「跨工况臂**减量** → `R_cross` **升**；跨工况臂**增量** → `R_cross` **降**」。实测：

* cond1、cond2 匹配 = 跨工况臂**减量**，`R_cross` 却 **下降**（1.058→0.902）+ / 略升（0.955→1.158）；
* cond3 匹配 = 跨工况臂**增量**（0.19×→1.00×，跨工况臂拿到**更多**数据），
  `R_cross` 却 **上升**（1.180→1.402）。

即 cond1 与 cond3 的移动**方向都与训练量预测相反** ⇒
**训练量不是残留超额的驱动因素**；这是对规则 2「归因训练量」的**直接证伪**，
残留因此成为**未归属残差**（既非训练量，也未达域效应的证据门槛）。

> **Stage 3 结论（红线 21 结论上限）**
> 在**同时控制测试对象（同一轴承、同一测试集）与训练样本量（`K` 严格相等）**之后，
> 跨工况数据**未表现出稳定一致的优势**：单元级 median `R_cross` 匹配前后为
> **1.086 / 1.158**，配对 `dR` **≈ 0**（CI 跨 0、`p = 0.847`），
> 效应方向随留出轴承与任务变化。
> 唯一方向性信号（cond3 略 > 1）在匹配后**不降反升**，故**既不能归因于训练量**（方向相反），
> **也不能归因于域漂移**（`|median−1|` 仍小于折间跨度、n = 5 功效不足）——属**未归属残差**。
> **禁止**表述为「等效 / 无差异 / 证明无增益」（红线 21）。

**对未匹配基准结论的影响（要正面写进正文）**
匹配前后 `R_cross` 几乎不变（median 1.086→1.158、配对 `dR`≈0）⇒
**训练量混杂虽然真实存在于 `n_train`，但它并不是 `R_cross` 的主导解释**；
即 §4 未匹配设计下「无稳定优势」的结论**对训练量混杂是稳健的**——
这反过来**修正了 Stage 2 遗留的担忧**（此前担心 cond3 的 `R_cross>1` 只是训练量假象）。


---

## 6. 验证日志

| 检查 | 方法 | 结果 |
|---|---|---|
| 训练确定性 | （继承 M0）`seed_everything` 前移 + `CUBLAS_WORKSPACE_CONFIG` 提前 | 修复后单工况 3/3、跨工况 9/9 逐位一致，极差 **0.0%** |
| A10（评测可复现） | 仅复制 checkpoint 重评测 vs 完整 train→eval | `results_corrected.csv` 逐字节相同 |
| 匹配臂构建 | 见 §5.2（脚本 `28` 内置验证） | **60/60 三项全过** |
| 匹配臂训练覆盖 | 5 折 × 3 目标 × (1 单 + 3 跨) × 5 seed | **300/300 checkpoint 全部产出**（可续跑：已存在即跳过） |
| **零学习对照（`constant_mean`）** | 匹配前后两臂 `R_cross` 及配对 `dR`（脚本 `30`） | median **1.018 → 1.018**、`dR`≈0（`p = 0.978`）⇒ 测试集两臂**逐位相同**、裁剪对常数预测器**无影响** |
| 匹配臂 `n_train` 严格相等 | 逐单元读聚合产物 `ratio vm` 列 | **15/15 单元 = 1.00×** |
| **A10 大规模复验（B9 附带产出）** | 对 M1 五折**跨工况臂**用当前代码重评测（复用 checkpoint，300 个 cell） | **36/36 每折逐位复现，`max|Δ| = 0.0000%`**（5 折全同） |
| B9 版本偏差量化 | 对 M1 五折**单工况臂**同法重评测 | 12 行/折中 9 行不变；变化仅 `se_resnet_bigru`，**max|Δ| 0.05–0.39%** |
| 解析↔实测 | （继承 Stage 2）`constant_one` 模型 RMSE ≡ 解析 `const_one` | 6/6 点、双口径 diff = 0.0000 |
| 测试集未动 | 匹配前后 `constant_one` 的 RMSE 是否变化（它在两臂都恒预测 1.0） | 5 折 × 3 工况**全部相同** ⇒ 测试集逐位未变 |

---

## 7. 复现命令

```bash
cd xjtu_sy_rul
PY=D:/miniconda3/envs/data_competition/python.exe

# 3.1 / 3.3 单元级统计（未匹配基准）
$PY scripts/23_stat_unit_rcross.py

# B9 修正：把 M1 基线统一到当前（post-B8）评测版本（纯推理，不训练）
for k in 1 2 3 4 5; do
  $PY scripts/33_reeval_checkpoints.py \
      --src-report artifacts/reports/matched_fold$k \
      --data-dir   artifacts/data_fold$k \
      --config     configs/matched_folds/fold$k.yaml
done
$PY scripts/23_stat_unit_rcross.py \
    --fold artifacts/reports/matched_fold1_b8 ... --fold artifacts/reports/matched_fold5_b8 \
    --out docs/stat_unit_rcross_b8.md --json artifacts/reports/stat_unit_rcross_b8.json

# 3.2 训练量匹配臂：构建 -> 训练 -> 聚合
$PY scripts/28_make_volume_matched.py
$PY scripts/29_run_volume_matched.py
$PY scripts/30_vm_vs_unmatched.py
# 零学习对照：把"纯标签分布漂移"从"学习侧迁移损失"里分离出来
$PY scripts/30_vm_vs_unmatched.py --model constant_mean \
    --md docs/vm_vs_unmatched_constmean.md --json artifacts/reports/vm_vs_unmatched_constmean.json
$PY scripts/31_stage3_figure.py
```

---

## 8. 未做 / 不越界

* **本阶段已完成**：3.1 严格配对 ✅、3.2 训练量匹配臂（构接 + 训练 + 配对聚合 + 零学习对照）✅、
  3.3 15 单元统计 ✅；B9 基线一致性缺陷已发现并修正（§4.1、红线 23）。
* **未扩模型**：Stage 3 与 Stage 2 同为单一 backbone `se_resnet_bigru`；
  「无稳定优势」的普适性需要 > 1 模型，列为后续 robustness（红线 7 的延伸）。
* **未匹配 val 集大小**（§2.6）；如需「训练动力学完全等价」的更强主张，须补 val-capped 变体。
* **未做多重检验校正**的正式化（n = 15，功效本就低；主证据是分布 + CI）。
* **未归属残差（cond3）**：匹配后仍 `R_cross > 1`（1.402）且**不降反升**，
  既非训练量、也未达域效应门槛 → 作为**开放项**登记，**禁止**在正文升格为结论。
* manuscript §6.6 / Table 7 仍为**未配对**数字（`642.2 / 946.6`），须按 A8 + 本文件 §4/§5 改写
  （改为同一 `(condition,bearing)` 上的配对 `R_cross` + `ΔRMSE` + `n_train`）。

---

*End of STAGE3_EVIDENCE_CHAIN.md*
