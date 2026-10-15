# STAGE1_AUDIT_REPORT.md —— 定义与协议审计报告

> **Stage 1 收口文件。** 六项审计的汇总判定、Definition Freeze、Protocol Freeze、Blocking Issues。
> 本文件写出后**立即停止，不进入 Stage 2**（`stage0_scope_freeze.md` §5）。

| 项 | 值 |
|---|---|
| 依据 | `新方案与框架文件/SCI研究设计方案_V2.0.md`（唯一主基线）、`audit/SCI_scope.md`、`audit/stage0_scope_freeze.md` |
| 方法 | **先读代码，再对论文**；每条结论附 `文件 → 函数 → 行号`；无证据处写「未核实」 |
| 审计日期 | 2026-10-07 |
| 改动 | **未修改论文正文、未修改任何代码、未启动任何训练**（Stage 1 当时）；**Round-1** 仅修 3 处代码缺陷 + 新增 1 个再分析脚本，仍未改论文、未训练 |
| 交付 | 本文件 + `audit_01…audit_06`（7 个文件）；**Round-1 追加** `STATISTICAL_UNIT_PROTOCOL.md`、`REVIEW_ROUND1_RESPONSE.md`、`scripts/23_stat_unit_rcross.py` |

---

## 1. 总判定表

| # | 审计项 | V2.0 出处 | 总判定 | 一句话结论 |
|---|---|---|---|---|
| **A1** | `LI(h)` 数学推导与命名 | §2.8-1、C2 | **MODIFY** | 推导成立；须补三条适用条件 + 命名降级为「标签结构指标」+ 双口径并列 |
| **A2** | `c*` / `best_const` 与实际损失 | §2.8-2、**C1（最高优先）** | **MODIFY** | 地板定义与实现**一致**（C1 风险解除）；须补「比值/分钟空间」限定、披露训练≠评价空间、修 1 处图脚本缺陷 |
| **A3** | 数据划分与窗口泄漏 | §2.8-3、§2.8-4 | **PASS** | 轴承不重叠、scaler/mRMR 只在 train 拟合、val 来自本域；两项须随稿披露 |
| **A4** | 训练量与测试对象可比性 | §2.8-5、C3 | **FAIL** | 测试对象**已配对（PASS）**；**训练量不可比（FAIL，0.07×–18.35×）** |
| **A5** | 测试对象与比较协议 | §2.8-5、C4、C6 | **MODIFY** | 配对协议正确且 `R_cross` 无稳定方向；`ΔRMSE` **缺失**、统计单位未声明、论文表用错口径 |
| **A6** | 后处理 / 指标对称性 | §2.8-6、C7 | **MODIFY** | 核心不对称已识别并已排除出主要证据；3 处实现级不对称须修 |

**统计**：PASS 1 项、MODIFY 4 项、FAIL 1 项。

---

## 2. 关键正面结论（先记「确认无误」的部分）

1. **C1（最高优先，`c*` 与损失一致性）——风险解除。**
   报告指标是**分钟空间无权重 RMSE**（`metrics.py` L35–L37、L234–L244；`report.py` L70）；
   该指标下的最优常数**就是** `L²` 加权均值（推导 A2.2；暴力网格验证：旧 `1+2→3` 的最小点 =
   `0.4936` = `c_life2`，RMSE 642.19，与 `15_cross_transfer.py` L60–L69 的输出
   `oracle_const_task__rmse = 642.1947` **完全吻合**）。
   → **`15_cross_transfer.py` 的「事后最优常数地板」定义正确**，
   §6.6「无特征模型越过地板」的结论**不因 C1 失效**。
   且在**配对协议**下目标测试集只有单轴承 → `L` 恒定 → 三种定义自动合一，**配对比较对 C1 免疫**。

2. **A3（泄漏）——干净。**
   `build.py` L388–L391（scaler 只 fit train）、L371/L196（mRMR 只 fit train）、
   `windows.py` L58/L61（标签只取窗口末端）；数值验证：三个单工况的 train/val/test **三交均为空集**。✔

3. **A4 的测试对象维度——已修好。**
   `cross_matched_linear.yaml` L6–L24/L54–L58 把两臂压到「只差训练数据来源」；
   数值验证：9/9 任务的跨工况 test 集与同目标单工况 test 集**逐位相同**。✔

4. **A5 的 `R_cross`——无稳定方向，但证据形态须改（Round-1 修订）。**
   真配对五折（`matched_fold_rcross.json`，uncorrected）：45 个 cell 的 pooled 中位数 **1.064**、
   win rate **0.40**、IQR 0.509；逐折 cell 中位数 **跨越 1**；三重复（`matched_rcross.json`）
   min/median/max = **0.867 / 1.245 / 2.365**。
   ⚠ **不得**再用 pooled **`p = 0.153`** 作为 H4 的正式统计证据：45 个 cell **不是 45 个独立样本**。
   统计单位已冻结为**测试轴承实例（n = 15 = 5 折 × 3 目标工况 = 全部 15 个轴承）**，
   主证据 = **分布 + 效应量 + 轴承层级自助 CI**：单元级中位 `R_cross` = **1.086**
   （95% CI **[0.889, 1.276]（跨 1）**）、跨工况更优 **33%**、中位 `ΔRMSE` **+3.44 min**
   （CI [−1.81, +12.62]（跨 0））、折中位 >1 者 **3/5**、\|median−1\| **8.6%** ≪ 折间跨度 **42.3%**；
   cell `p=0.153`（探索性）、单元 `p=0.252`（indicative）。
   协议/实现/处置：`audit/STATISTICAL_UNIT_PROTOCOL.md`、`scripts/23_stat_unit_rcross.py`、
   `audit/REVIEW_ROUND1_RESPONSE.md`。
   → **只能主张「未表现出稳定一致的优势，效应方向随留出轴承与任务变化」**，**不得**写「等效」。

5. **`LI(h)` 的单调性——成立。**
   窗口实测：linear 0.897 → h=0.5 **0.594** → 0.6 **0.491** → 0.7 **0.371** → 0.8 **0.247** → 0.9 **0.127**。
   （`09_label_baseline_anchors.py` 实跑，`[PASS] 3 'linear' anchor(s) agree`。）

---

## 3. 缺陷清单（按严重度）

| 编号 | 缺陷 | 严重度 | 证据 | 影响交付物 |
|---|---|---|---|---|
| **D1** | ~~`11_make_figures.py::analytic_anchors` 把 **MAE 当 RMSE**~~ → **RESOLVED（Round-1）** | ~~高~~ 已修 | 改为 `sqrt(mean((life*(ratio-c))²))`；重跑得 6.055 / 88.911 / 23.958（原误值 5.24 / 77.00 / 20.75），图已重绘 | `fig3_main_results.{png,pdf}` |
| **D2** | ~~`report.py` L68 `std_min` 未随 `factor` 缩放~~ → **RESOLVED（Round-1）** | ~~高~~ 已修 | `std_min = std_raw * factor`，与 `lower/upper` 同源 | 跨工况 `crps` |
| **D3** | ~~`report.py` L71 「未校正」口径**误用已校正区间**~~ → **RESOLVED（Round-1）** | ~~高~~ 已修 | 未校正行单独构造 `lower_unc/upper_unc/std_raw`；fold1 验证 `1->2` picp 0.4913/0.4609、crps 179.32/85.14（不再相同） | 跨工况 `picp` |
| **D4** | ~~**`ΔRMSE` 全仓库不存在**~~ → **RESOLVED（Round-1）** | ~~高~~ 已补 | 已在 `scripts/23`(单元级)、`21`(池化 `median_delta_rmse_min`)、`20`(每 target/model `median_delta_rmse`) 产出 | §6.6 全表、H5 |
| **D5** | ~~统计单位未声明 + 折间依赖未处理~~ → **RESOLVED（Round-1）** | ~~高~~ 已修 | 冻结为轴承实例 n=15 + 轴承自助 CI：`audit/STATISTICAL_UNIT_PROTOCOL.md`、`scripts/23_stat_unit_rcross.py` | 所有 p 值 / CI |
| **D6** | **训练量不可比**（0.07×–18.35×，符号随目标工况翻转），且**无匹配臂** | **高** | `22_train_volume_audit.py` 实跑；`artifacts/` 无匹配臂产物 | RQ3/RQ4 的因果结论、H5 |
| **D7** | `paper/manuscript.md` §6.6 Table 7 用的是**未配对**结果 | **高** | L502 `1+2→3 = 946.6` ↔ `cross_transfer_summary.json` 的 `n_windows=6879 / n_bearings=5`；配对口径应为 `n_test=83`、oracle 23.96 | §6.6 全表 |
| **D8** | `LI(h)` **适用条件**未在论文明示（N→∞ / r 触达 0 / L 恒定） | 中 | `paper` §3.2 只写 "continuous limit" | §3.2/§3.3 措辞 |
| **D9** | `LI(h)` 命名未降级为「标签结构指标」（6 处文档按等式使用） | 中 | `docs/contributions.md` L11、`experiment_plan.md` L923、`comparison_strategy_qa.md` L94/169/234、`S2L_S5L_results.md` L63、`STATUS_review.md` L83 | 全稿措辞（V2.0 C2） |
| **D10** | **训练空间（比值）≠ 评价空间（分钟）** 未披露 | 中 | `losses.py` L33/L42 vs `metrics.py` L235/L244 | §4 Method |
| **D11** | 三角色（`const_one`/`const_mean`/`best_const`）未在方法章集中分列 | 中 | `10` L136–L139 有注释；论文未集中 | §3 章（V2.0 C13） |
| **D12** | 校正因子跨域搬运且不稳定（0.21–5.60）未正式披露 | 中 | `report.py` L48/L54；`paper` L531–L533 | §6.6 |
| **D13** | 特征缓存指纹**不含标签/划分** → 换标签若不隔离 `data_dir` 会**静默复用** | 低（现状已隔离） | `build.py` L74–L82 | 实验 runbook |
| **D14** | **纯再评测（复用 checkpoint、不训练）不可复现**：`inference.py::predict_with_uncertainty` 抽 50 次 MC-Dropout，评测路径**从不播种**，RNG 取自进程启动 OS 熵 → 约 **0.05–2%/格** 抖动（`17` 的 0.0% 只覆盖 train→eval 全链） | **中**（Round-1 新发现） | `inference.py` L157–168、`run.py` L111 无 `seed_everything` | 由 checkpoint 复算的一切数字 |

---

## 4. Definition Freeze（定义冻结）

> 以下为**冻结后的正式定义**。方法章必须以本节为准；与本节不一致的既有表述一律作废。

### D1. `h` —— 平台段占比

```
h ∈ (0,1)：标签在 [0, h·n) 上恒为 1.0，之后线性衰减至 1/(n−threshold)
threshold = max(1, min(int(n·h), n−1));  span = n − threshold
```
代码：`rulbench/data/labels.py::piecewise_linear_rul` L27–L42；配置键 `data.label.healthy_ratio`。
**已核实**：代码含义与 V2.0 §2.4 的「平台段占比」**一致**。
**linear 标签** = `r_i = (n−i)/n`（`labels.py::linear_rul` L45–L51），是 `h→0` 的**参照臂**，
但**末端为 `1/n` ≠ 0**。

### D2. `LI(h)` —— 标签结构指标（label-structure index）

```
LI = 1 − Var_e[r] / E_e[(r−1)²]              （定义式，恒成立）
LI(h) = 3(1−h)/4                            （连续极限：N→∞、均匀采样、r 触达 0、受评集合内 L 恒定）
LI_window(h)                                （正式采用：窗口精确闭式，metrics.py L156–L172）
```
**强制**：
1. **名称**：只能叫「标签结构指标」；**禁止**「退化信息 / 可预测性 / 可学习性」。
2. **双口径**：凡给 `LI` 数值，必须**同时**给连续值与**窗口实测值**。
3. **适用条件**：引用 `3(1−h)/4` 时，必须写出 D2 括号里的三个条件。
   （实测偏差：`h=0.8` 连续 0.150 vs 窗口 0.247；`h=0` 连续 0.750 vs 窗口 0.897。）

### D3. 三种零学习基准 —— 角色与边界

| 量 | 定义（代码） | 拟合数据 | 角色 | 可部署 |
|---|---|---|---|---|
| `const_one` | `ratio ≡ 1.0`（`trivial.py` L31–L32） | 无 | 朴素参照 | ✘ |
| `const_mean` | `ratio ≡ mean(train_ratio)`（`trivial.py` L33–L34；`run.py` L34） | **训练集**标签 | **可部署**零特征基准；**主分析对手** | ✔ |
| `best_const` | 受评集合上的事后最优常数（`09` L59；`15` L60–L69） | **测试集**标签（事后） | **事后诊断**：常数预测极限 | ✘ **禁止** |

**强制**：`best_const` 与 `const_mean` **不得**混用同一口径；`G` 的主计算分母 = `const_mean`。

### D4. `c*` / `best_const` 的闭式与作用域

```
受评集合内 L 恒定（单工况、配对跨工况）:  c* = Σ rᵢ / N                （普通均值）
受评集合内 L 变化（池化多轴承）:          c* = Σ Lᵢ²rᵢ / Σ Lᵢ²          （L² 加权均值）
```
**强制**：
1. 定义在**报告指标**（分钟空间无权重 RMSE）上 —— **这是唯一的 `best_const` 口径**；
2. **必须**说明「`L²` 权重来自比值→分钟换算，而非额外加权」；
3. **披露**训练目标（比值空间、`threshold=0.3`、`weight=2.0`）≠ 报告指标（分钟空间无权重），
   故 **`c*_lossopt ≠ c*`**（旧 `1+2→3`：0.4103 vs 0.4936，对应 RMSE 669.11 vs 642.19）。

### D5. 学习增益 `G`

```
G = 1 − RMSE_model / RMSE_baseline        baseline 主用 const_mean，best_const 仅作诊断
```
**强制**：① 同测试对象、同窗口、同误差定义；② **必须并列 RMSE 绝对差**；
③ 声明「`G>0` 只说明在该评价集合上优于所选基准，不证明学到因果退化机理」。
代码现状：`G` 尚未作为一等字段（须在 Stage 2 新增）。

### D6. 跨工况比 `R_cross` 与 `ΔRMSE`

```
R_cross = RMSE_cross / RMSE_single          （配对：同测试轴承、同窗口、同评价规则）
ΔRMSE   = RMSE_cross − RMSE_single          （分钟；必须与 R_cross 并列）
必须并列 n_train                             （否则差异不能全归因于工况迁移）
```
**强制**：① **只能用配对产物**（`matched_rcross.json` / `matched_fold_rcross.json`），
**不得**用事后重聚合（`scripts/18…`）或未配对聚合（`cross_transfer_summary.json`）；
② 引用 best-of-source 结果时**必须标明选择规则**。

---

## 5. Protocol Freeze（协议冻结）

| # | 项目 | 冻结内容 | 依据 |
|---|---|---|---|
| **P1** | **划分** | 单工况 `train[1,2,3] / val[4] / test[5]`（条件内）；跨工况**配对** `source[1,2,3] / val[4] / target_test[5]` | `base.yaml` L53–L61；`cross_matched_linear.yaml` L54–L58 |
| **P2** | **归一化** | `ChannelScaler` **只在训练 split 上 fit**，val/test 仅 transform | `build.py` L388–L391 |
| **P3** | **特征选择** | mRMR **只在训练轴承上 fit**（relevance 代理 = 寿命位置） | `build.py` L371、L196、L199–L204 |
| **P4** | **窗口** | `window_len=32`、`stride=1`；标签 = 窗口**末端**比值；前 31 个快照不打分 | `base.yaml` L20–L21；`windows.py` L50/L58/L61 |
| **P5** | **transductive 声明** | 跨工况 UDA 臂使用**目标 test 的输入**（无标签）；必须并列 source-only 臂；「对齐增益」只对 `MMD_CAPABLE` 模型报 | `protocols.py` L61–L65、L105；`registry.py` L138 |
| **P6** | **训练量** | **当前不可比（0.07×–18.35×）**；**阶段 3 必须补①原始/②跨原始/③跨匹配 三种训练方式**；在补齐前 `R_cross`/`ΔRMSE` 只作描述性 | `22_train_volume_audit.py`；V2.0 §1.5 C3 |
| **P7** | **后处理** | median-scaling **仅跨工况**；**主证据用未校正口径**；校正口径仅作敏感性；`factor` 只在**源域 val** 估 | `report.py` L52–L61；V2.0 §1.7 排除 ① |
| **P8** | **指标** | headline = **分钟空间无权重 RMSE**；`MAE` 并列；**PHM 两口径并列**并声明其固有不对称；`nrmse(range)` 仅**本协议内**可比 | `metrics.py` L35–L58、L71–L88、L247–L248 |
| **P9** | **统计单位** | **统计单位 = 轴承（实例）**；窗口**不得**当独立样本；折间共享数据依赖须显式处理（分层/重采样），不得把五折当 5 个独立数据集。**已冻结并实现**：单位 = `(condition, bearing)` n=15，p 降级为探索性/indicative，主证据 = 分布 + 效应量 + 轴承自助 CI（`STATISTICAL_UNIT_PROTOCOL.md` S1–S10；`scripts/23_stat_unit_rcross.py`） | V2.0 §1.7、C4 |
| **P10** | **确定性** | `seed_everything(seed)` 在**建模前**调用；`CUBLAS_WORKSPACE_CONFIG` 在 torch 导入前设置；**必须报 run-to-run 极差**，任何小于该极差的差异不得作结论 | `protocols.py` L87；`rulbench/__init__.py`；V2.0 §1.5 阶段A |
| **P11** | **数据目录隔离** | 齿轮换标签/划分时**必须**换 `paths.data_dir`（缓存指纹不含标签） | `build.py` L74–L82；`linear.yaml` L23；`cross_matched_linear.yaml` L51 |

---

## 6. Blocking Issues（阻塞项）

> 判定规则：**存在 FAIL 即不得进入 Stage 2**。本节同时给出**该项阻塞的是哪个阶段**，供裁定。

| 编号 | 阻塞项 | 来源 | 阻塞的阶段 | 是否可「先修后放行」 |
|---|---|---|---|---|
| **B1** | **A4：训练量不可比** —— 无 ③ 训练量匹配臂 | D6 | **阶段 3 的因果结论**（RQ3/RQ4、H5） | 其修复**本身就是 V2.0 阶段 3 的核心实验**（§1.5 C3、§6.2 P1-4） |
| **B2** | ~~`ΔRMSE` 缺失~~ → **RESOLVED（Round-1）** | D4 | 阶段 3 的表（§6.6 / Table 5） | ✅ 已补：`scripts/20/21/23` 三处产出 `median_delta_rmse` |
| **B3** | ~~统计单位 + 折间依赖未处理~~ → **RESOLVED（Round-1）** | D5 | 全部统计结论 | ✅ 已冻结协议（n=15 轴承实例）+ `scripts/23` 实现 |
| **B4** | ~~`report.py` 区间/方差不同源、「未校正」误用校正区间~~ → **RESOLVED（Round-1）** | D2、D3 | 跨工况的 `crps`/`picp` | ✅ 已修 + `matched_fold{1..5}/cross` 复用 checkpoint 重算（无重训） |
| **B5** | ~~`11_make_figures.py` 用 MAE 当 RMSE~~ → **RESOLVED（Round-1）** | D1 | 交付图 Fig 3 | ✅ 已修 + `fig1–4` 已重绘 |
| **B6** | §6.6 Table 7 用未配对结果 | D7 | 阶段 5 的文件级重构 | 可，换数据源（**未改论文**） |
| **B7** | `LI(h)` 适用条件 + 命名降级未落实 | D8、D9 | 全稿措辞（§3.2/§3.3/§7） | 可，纯文字 |
| **B8** | **纯再评测不可复现**（MC-Dropout 未播种） | D14 | 一切「由 checkpoint 复算」的数字 | ✅ **RESOLVED（Round-1）**：`predict_seed` 采样前 `seed_everything(seed)`；两独立进程重算逐位一致 |

---

## 7. 门禁裁定（Gate Decision）

### 7.1 按规则的字面结论

> **A4 = FAIL → 依 `stage0_scope_freeze.md` §5 的规则，本代理不进入 Stage 2，就此停止。**

### 7.2 对 A4 这一 FAIL 的性质说明（供你裁定）

必须把「**定义级失败**」与「**缺一个实验臂**」区分开：

| 维度 | 结论 |
|---|---|
| **定义 / 数学** | **0 项 FAIL**。`c*`、`LI`、`G`、`R_cross` 的定义均已核实可用（`c*` 甚至比 V2.0 担心的更安全）。 |
| **协议 / 可比性** | **1 项 FAIL**：训练量不可比。**这不是定义错误，而是缺 V2.0 已排入阶段 3 的实验臂。** |
| **实现** | 3 项高严重度缺陷（D1/D2/D3）+ 1 项缺失字段（D4），**均为纯计算或数行代码可修**。 |

因此有两种同样站得住的读法：

| 读法 | 含义 | 后果 |
|---|---|---|
| **读法甲（严格）** | A4 未过 → 不得进入 Stage 2；先把 B1 的训练量匹配臂补上 | 阶段 2 被推迟到阶段 3 的核心实验之后（顺序倒置，与 V2.0 §2.8 的阶段划分冲突） |
| **读法乙（按阶段映射）** | A4 阻塞的是**阶段 3 的因果结论**，而**不阻塞阶段 2（标签结构 / 单工况内）**；阶段 2 可先行，但**任何跨工况措辞在补齐匹配臂前必须禁用** | 与 V2.0 §2.8 的阶段划分一致 |

**本代理不代替你裁定**（这属于研究设计决策，不属于审计）。倾向性建议：**读法乙**，
理由是训练量匹配在 V2.0 里明确是**阶段 3 的核心实验**，把它当作阶段 2 的前置会使阶段划分自相矛盾；
但必须同时**冻结一条禁令**：

> **在 ③ 训练量匹配臂完成之前，任何 `R_cross` / `ΔRMSE` 的表述不得含因果措辞
> （「由于工况迁移」「工况迁移导致」等），H5 状态记为「无法判定」。**

### 7.3 无论采哪种读法都成立的「必须先做」清单

按「定义冻结后、正式实验前」的合理顺序（**Round-1 已执行的标 ✅**）：

1. ~~**修 B4 / B5**~~ ✅ **已修 + 已重算/重绘**（`report.py` 的区间-方差同源与未校正口径、`11_make_figures.py` 的 MAE/RMSE）；
2. ~~**补 B2 / B3**~~ ✅ **已补**（`ΔRMSE` 字段、轴承层级协议 + 自助 CI）；
3. **落实 B7**（`LI` 适用条件 + 命名降级）——纯文字，**留 Stage 5**；
4. **替换 B6**（§6.6 换配对数据源）——**留 Stage 5**；
5. **再动手阶段 2**；阶段 3 开工前先建 B1 的匹配臂。

> **Round-1 处置详见** `audit/REVIEW_ROUND1_RESPONSE.md`；统计单位协议见 `audit/STATISTICAL_UNIT_PROTOCOL.md`。

---

## 8. V2.0 冲突清单（C1–C14）的审计回执

| # | V2.0 关切 | 本报告结论 |
|---|---|---|
| **C1** | `best_const` 权重 vs 评价损失 | ✅ **关闭**：评价是分钟空间 RMSE，最优常数 = `L²` 加权均值，与代码一致；V2.0 §1.3(b) 的「待核实」可改为「已核实：比值/分钟空间二分」 |
| **C2** | `LI(h)` 地位 | ⚠ **维持 MODIFY**：推导成立，但须降级命名 + 补适用条件（D8/D9） |
| **C3** | 训练量匹配为核心 | ⚠ **确认未做**（D6/B1）；保持 P1 |
| **C4** | 统计单位与折间依赖 | ⚠ **确认未做**（D5/B3） |
| **C5** | 「不可区分」措辞 | ✅ 已按「未发现显著系统性差异」冻结（`stage0_scope_freeze.md` §2.1） |
| **C6** | `ΔRMSE` | ⚠ **确认缺失**（D4/B2） |
| **C7** | 不对称后处理 | ✅ **确认存在、且已禁用为主要证据**（A6） |
| **C8** | 修复前结果 | ✅ 一致：旧产物已定位（§6.6、`docs/matched_cross_vs_incondition.md`） |
| **C9** | 章节结构 | ⏸ 超出 Stage 1 范围（属阶段 5） |
| **C10** | 贡献压缩三层 | ✅ Stage 0 已冻结（`SCI_scope.md` §4） |
| **C11** | 题目 | ⏸ 待定稿阶段 |
| **C12** | `h≈0.60` 降级 | ✅ 已冻结为「待复核现象」；且本审计发现**交叉点位置依赖口径**（连续 vs 窗口），进一步支持降级 |
| **C13** | 三角色分列 | ⚠ **未落实**（D11） |
| **C14** | 消融重设计 | ⏸ 超出 Stage 1 范围 |

---

## 9. 未核实项（汇总，供后续补齐）

| # | 项 | 来源 |
|---|---|---|
| U1 | PHM 2012 的窗口 `LI` 谱与 XJTU-SY 是否同序 | A1 §6 |
| U2 | `h*` 在窗口口径下的数值 | A1 §6 |
| U3 | 各数据集的轴承重叠是否逐个 npz 复核（除 `data_matched`） | A3 §6 |
| U4 | `feature_cache` 是否存在跨标签误用的历史文件 | A3 §6 |
| U5 | `pca_health_indicator` 是否在别处被用作特征/标签 | A3 §6 |
| U6 | `data_fold2…5`、PHM 2012 的训练量比 | A4 §6 |
| U7 | 两臂的**有效优化步数**（样本量/批大小）是否匹配 | A4 §6 |
| U8 | `matched_fold_rcross.json` corrected 臂的完整统计 | A5 §11 |
| U9 | 五折的折定义是否覆盖 5 个不同留出轴承 | A5 §11 |
| U10 | 校正口径下 `constant_one` 与 `constant_mean` 是否逐位相同 | A6 §8 |

---

## 10. 停止声明

1. 本报告与六份审计（`audit_01_LI.md` … `audit_06_postprocessing.md`）已产出；
2. **未修改论文正文、未修改代码、未启动训练**（Round-1 前）；
3. 依 `stage0_scope_freeze.md` §5 与你的指令：**本报告写出后停止，不进入 Stage 2**；
4. ~~**待你裁定**：§7.2 的读法甲 / 读法乙。~~ → **已裁定：读法乙（Round-1）**，
   并追加更严格的统计要求（统计单位 = 轴承）。
   → 已按 §7.3 清单执行 **B2/B3/B4/B5**（见 `audit/REVIEW_ROUND1_RESPONSE.md`），
   **未改论文**；下一步 = Stage 2 的**非跨工况**部分，Stage 3 的**训练量匹配臂**为其前置红线的解除条件。

---

## 11. Round-1 修订索引（2026-10-07）

| 位置 | 修订 |
|---|---|
| §2.4 | 不再用 pooled `p=0.153`；改为单元级分布 + 效应量 + 轴承自助 CI |
| §3 D1–D4 | 全部标记 RESOLVED（已修 + 已重算/重绘） |
| §3 D5 | 标记 RESOLVED（协议冻结 + `scripts/23`） |
| §3 D14 | **新增**（Round-1 新发现：纯再评测不可复现） |
| §5 P9 | 升级为「已冻结并实现」 |
| §6 B2–B5 | 标记 RESOLVED |
| §6 B8 | **新增并 RESOLVED**（评测路径播种） |
| §7.3 | 已完成项打勾 |
| §10 | 读法已裁定为「按阶段（乙）」 |
| 披露清单 | **建议新增 A10**（评测 MC-Dropout 已播种，checkpoint 再评测 = 完整重训） |

*End of STAGE1_AUDIT_REPORT.md*
