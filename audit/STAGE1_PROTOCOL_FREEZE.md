# STAGE1_PROTOCOL_FREEZE.md —— Stage 1 统计协议冻结（出口文件）

> **本文件是 Stage 1 的正式出口。** 自此，下列口径**不得**再被后续阶段改写；
> 若确需修改，必须回到本文件按「修订」流程留痕。Stage 2 及其后一律以本文件为准。

| 项 | 值 |
|---|---|
| 冻结日期 | 2026-10-07 |
| 依据 | 独立审核 Round-1 意见（用户转达）+ `audit/STATISTICAL_UNIT_PROTOCOL.md` + `audit/REVIEW_ROUND1_RESPONSE.md` |
| 状态 | **FROZEN**（Stage 1 结束，Stage 2 开启） |
| 相关文件 | `audit/STATISTICAL_UNIT_PROTOCOL.md`（S1–S10）、`audit/STAGE1_AUDIT_REPORT.md`、`audit/audit_05_cross_protocol.md` |

---

## 1. 核心口径（5 条，全部冻结）

### F1｜统计单位 = 15 个测试轴承实例
- 单位 = `(fold, target condition)` = `(condition, bearing)`；
- 5 折 × 3 目标工况 = **n = 15 = XJTU-SY 全部 15 个轴承**；
- 每个单元内的 **3 个 source-task 是重复测量**，不是新样本；
- 因此 **45 个 cell 不是 45 个独立样本**，不得据此做任何确认性检验。

> 实现：`xjtu_sy_rul/scripts/23_stat_unit_rcross.py`（单元聚合 = 单元内 3 任务取中位数），
> 输出 `xjtu_sy_rul/docs/stat_unit_rcross.md` 与 `artifacts/reports/stat_unit_rcross.json`。

### F2｜H4 不再依赖 `p = 0.153`
- `p = 0.153` **可以作为探索性 cell-level 描述**保留（明确标注 `EXPLORATORY_ONLY`）；
- **正式结论**依据：**效应方向** + **15 个测试实例上的分布** + **轴承层级自助 CI** + **`ΔRMSE`（分钟）**；
- 单元级 `p = 0.252` 亦仅为 `indicative`（n=15，功效低），不作正式判据；
- **严禁**写「等效 / 无差异」。

### F3｜H4 当前正式措辞（逐字使用）

> 在严格配对的跨工况实验中，跨工况训练相对于目标工况内训练未表现出稳定一致的优势，
> 其效应方向随留出轴承和任务而变化。

### F4｜A4 / H5 继续保留
- **训练量匹配实验属于 Stage 3**；
- 在它完成以前，**`R_cross` / `ΔRMSE` 一律不作因果解释**（禁「由于工况迁移」「工况迁移导致」等）；
- **H5 状态 = 「无法判定」**，**不是**「得到支持」。

### F5｜只有「修复后重生成」的结果才是有效版本
下述三类修复**之后**重新生成的结果才是后续实验的有效版本：

| 修复 | 内容 | 时点 |
|---|---|---|
| 确定性修复 | `train_one` 中 `seed_everything` 前移到建模之前；`CUBLAS_WORKSPACE_CONFIG` 提前到包导入 | 2026-10-07 09:44 |
| B4 | `report.py`：`std_min` 随 `factor` 缩放；未校正行不再复用校正区间 | 2026-10-07 下午 |
| B8 | `inference.py`：MC-Dropout 评测路径播种（`predict_seed(..., seed=)`） | 2026-10-07 下午 |

**旧 CSV / 报告中的 PICP、CRPS 以及一切 `rmse` 数值不得混入最终结果。**
失效清单见 §4。

---

## 2. B8 的归类：**实验基础设施修复**，不是研究创新

**结论（冻结）**：B8 按「**实验基础设施修复 / 可复现性修复**」记录，**禁止**包装为新的研究贡献、
**禁止**写入 Contributions，**不得**作为「方法创新」出现在摘要/引言/结论。

**允许出现的位置**：
- 论文的 *Reproducibility* 小节 或 *Implementation details* 附录（一句话级）：
  > MC-Dropout 推断路径已播种；由 checkpoint 再评测与完整重训给出逐位相同的数字。
- 披露清单中的 **A10** 条目（披露 ≠ 贡献）。

**理由**：它不是原研究设计里的问题，而是评测管线的可复现性缺口；把它当创新会**虚增贡献层数**，
与本文「三层贡献」的收束（`SCI_scope.md` §4）冲突。

**可作为内部工程价值保留的东西**（不进论文）：
- 教训：「整体流程可复现」≠「任一子步骤可复现」——凡下游含未播种随机算子（此处 MC-Dropout），
  复用中间产物（checkpoint）就会脱离原来的确定性轨迹。审计可复现性时必须**单独测「从中间产物再计算」**这条路径。

---

## 3. Stage 2 的入口条件（本次开启）

**Stage 2 = 单工况 + 标签结构 + 零学习基准 + Learning Gain。**

### 3.1 证据链（必须完整跑通一条，再扩展）

```
h  →  label structure (LI)  →  const_mean  →  best_const  →  RMSE  →  Learning Gain
```

### 3.2 验证模型：**先只用 `se_resnet_bigru`**

**规则：一次只跑一个验证模型，先产出完整证据链，确认无新的协议问题后，再扩到 CNN / GRU / 其它。**

### 3.3 Stage 2 只回答两个问题

- **RQ1**：标签结构改变后，RUL 任务本身的预测难度如何变化？
- **RQ2**：模型到底获得了多少**超越零学习基准**的真实学习增益？

**不回答**（留作后续 robustness）：「SE-ResNet 比其他模型好多少」。

### 3.4 公式与角色（沿用冻结定义）

| 量 | 定义 | 角色 |
|---|---|---|
| `LI(h)` =「标签结构指标」 | `1 − Var[r]/E[(r−1)²]`；连续极限 `3(1−h)/4` **只**在 N→∞+均匀采样+r 触达 0+受评集合内 L 恒定时成立 → **必须双口径报** | RQ1 的自变量刻画 |
| `const_one` | `ratio ≡ 1.0` | 朴素参照（稻草人） |
| `const_mean` | `ratio ≡ mean(train_ratio)` | **可部署**零学习基准；`G` 的分母 |
| `best_const` | 受评集合上的事后最优常数 | **仅事后诊断**（常数预测极限），**禁止**当可部署 |
| `G` | `1 − RMSE_model / RMSE_const_mean` | RQ2 的答案；**必须并列 RMSE 绝对差** |

**口径纪律**：单工况下 `factor ≡ 1`，故校正/未校正一致；Stage 2 全部数字**只标一个口径**即可，
但必须在文中声明「单工况无后处理」。

---

## 4. 有效版本基线（Stage 2 开工时的产物台账）

### 4.1 失效（**禁止引用**）
| 产物 | 原因 |
|---|---|
| `artifacts/reports/single`、`single_linear` | 确定性修复前（10-06 16:47 / 10-07 00:07） |
| `artifacts/reports/sweep_h{0.5..0.9}` | 确定性修复前（00:17–00:32）——**正是 Stage 2 的标签谱** |
| `artifacts/reports/phm2012/single_linear` | 修复前（00:46） |
| `artifacts/reports/cross_uda_linear`、`cross_source_only_linear` | 修复前（01:44–01:57） |
| 上述目录的 `picp` / `crps` 列 | 另叠加 B4 修复前的口径缺陷 |

> ⚠ **陷阱**：上述目录内**仍存有 checkpoint**（如 `single_linear` 有 75 个 `.pt`）。
> 在原目录重跑会因「checkpoint 已存在」而**静默跳过训练**、直接复用修复前的模型。
> → 重跑**必须换结果根目录**（Stage 2 使用 `artifacts/reports/s2_chain/`）。

### 4.2 有效（可引用）
| 产物 | 说明 |
|---|---|
| `artifacts/reports/matched_fold{1..5}` | 确定性修复后；跨工况臂已按 B4/B8 重算 |
| `artifacts/reports/stat_unit_rcross.json` | F1 的单元级结果 |
| `artifacts/reports/matched_fold_rcross.json` | 含 `median_delta_rmse_min`（B2） |
| `paper/figures/fig1–fig4` | B5 修复后重绘 |
| 各标签的**数据集** `artifacts/data_linear`、`data_h{0.5..0.9}` | 由标签/特征决定，不受训练侧修复影响 |

---

## 5. 红线清单（Stage 2 全程有效）

1. **禁止**把 45 个 cell 当独立样本；统计单位是轴承实例（F1）。
2. **禁止**用 `p=0.153` 支撑 H4；**禁止**写「等效/无差异」（F2/F3）。
3. **禁止**在训练量匹配臂完成前对 `R_cross`/`ΔRMSE` 作因果解释；H5 = 无法判定（F4）。
4. **禁止**混用修复前的 PICP/CRPS/rmse（F5）。
5. **禁止**把 B8 写成研究创新或贡献（§2）。
6. **禁止**在 Stage 2 一开始就铺开全部模型/数据集；先 `se_resnet_bigru` 跑通一条链（§3.2）。
7. **禁止**用「模型间排名」回答 Stage 2 的 RQ1/RQ2（§3.3）。
8. `best_const` **只能**作事后诊断，**不得**当作可部署基线（§3.4）。

---

## 6. 修订记录（append-only）

| 日期 | 动作 | 说明 |
|---|---|---|
| 2026-10-07 | 增补 §6（本次） | **不修改任何已冻结口径**，仅登记 Stage 2 产物并指向其证据链文件。 |

**Stage 2 产物登记**（新增可引用项，追加到 §4.2）：

| 产物 | 说明 |
|---|---|
| `artifacts/reports/s2_chain/{linear,h0.5..h0.9}` | `se_resnet_bigru` + `constant_one`/`constant_mean`，确定性修复 + B4 + B8 之后生成 |
| `artifacts/reports/stage2_chain.json`、`docs/stage2_chain_table.md` | Stage 2 证据链（双口径） |
| `paper/figures/fig_stage2_label_gap.png` | Stage 2 主图 |
| `audit/STAGE2_EVIDENCE_CHAIN.md` | Stage 2 出口文件（RQ1/RQ2、验证日志、复现命令） |

**新增口径钉子（Stage 2 发现的陷阱，追加到红线）**：

9. **按轴承分组时必需带上工况/运行键**。三个工况的测试轴承 `bearing_id` **都是 5**；
   只用 `bearing_id` 分组会把三工况误并为「一个轴承」，从而把「三轴承平均」口径
   静默算成「池化」口径（实测：h=0.8 的 LI 会从 0.247 错成 0.183）。
   分组键必须是 `(condition, bearing_id)`。

### Stage 2.1 追加（误差机制审查，2026-10-07 晚）

10. **机制陈述必须有测量支撑**。在算出 `bias` 之前，**禁止**出现「系统性高估 / 低估 / 偏移」等措辞。
    （本次实测：Stage 2 初稿的「系统性高估」被**推翻**；真实方向是**低估**。见 `STAGE2_EVIDENCE_CHAIN.md` §8.1。）
11. **`bias` 必须并列归一化 RUL 单位**（`pred_ratio − ratio`）。三测试轴承寿命 52 / 339 / 114 分钟，
    `−66 min` 在各轴承上对应完全不同的 RUL 比例，**分钟口径的 bias 不可跨轴承平均或比较**。
12. **寿命阶段（early / middle / late）分箱必须用归一化 `ratio`**，不得用绝对分钟。
13. **口径分工**：**主结果 = 测试实例等权平均**；**pooled = 敏感性分析**。两套结果**不得**混入同一主表。
14. **`G` 的交叉只能写作 learning-gain crossover**，位置表述为「h ≈ 0.6 附近，随口径在 0.5–0.7 移动」；
    **禁止**给出精确「临界点 / 最优阈值 h\*」（h 仅取了 0.5/0.6/0.7 三个离散值）。

**Stage 2.1 产物登记**（追加到 §4.2）：

| 产物 | 说明 |
|---|---|
| `artifacts/reports/stage2_bias.json`、`stage2_bias_scatter.json` | 标签结构统计 + 偏差机制（无训练） |
| `paper/figures/fig_stage21_pred_vs_true.png`、`fig_stage21_bias_profile.png` | 归一化散点 + 偏差剖面 |
| `scripts/26_stage2_bias.py`、`scripts/27_stage21_figure.py` | 实现 |

### Stage 2 封板追加（2026-10-07 深夜）

15. **术语**：高 h 下的现象只能叫 **prediction-response compression（预测响应压缩）**。
    **禁止**写「塌缩为常数 / collapse to a constant」——预测仍随真值变化，只是**幅度持续衰减**
    （`slope` 0.763→0.234、三轴承离散度比 0.841→0.513），证据**不足以**支持「完全塌缩为常数」。
16. **「偏移主导」类结论必须用正式指标**：`Bias²/MSE = mean(d)²/mean(d²)`（有界 [0,1]；0 = 纯方差，1 = 纯偏移）。
    **不得**用 `|bias|/RMSE` 代替正式表述；行文只能是
    「平均预测偏差的绝对幅度相对于 RMSE 明显增加，表明误差呈现更强的系统性偏差特征」。
17. **机制表必须列全部 6 个标签点**（linear、h=0.5、0.6、0.7、0.8、0.9）。
    主表用 6 点、机制表只用 4 点会造成「选择性展示」的质疑。
18. **Stage 2 正式封板**：Stage 2 / 2.1 到此结束，**不再重训、不再扩模型、不再改 Stage 2 结论**。
    后续引用 h 谱一律以 §4.2 登记的产物为准。

### Stage 3 入口追加（2026-10-07 晚）

19. **训练量匹配臂的构接口径**（`scripts/28_make_volume_matched.py`）：`(fold, target)` 单元内
    **全部四条臂**（1 个单工况 + 3 个跨工况变体）一律下采样到 `K = min(n_train over the four arms)`；
    取窗方式 = 归一到各训练轴承后**按时间等距**（等价于抬高有效 stride，保留全长寿命阶段覆盖与轴承间权重）。
    **val / test 与匹配前逐位一致**（常数地基不变）；**scaler 在裁剪后的训练集上重拟合**；
    **mRMR 索引不动**——它由**快照级**训练轴承特征决定、与窗口数无关，而「哪个工况」正是处理变量，
    不动是**正确**行为而非泄漏。
20. **Stage 3.2 裁决规则（预先注册，禁止事后择结论）**：
    - 主判据 = 单元级（n=15）`median R_cross`（**未校正口径**）在匹配前后的**配对差** `dR`，
      以及**按目标工况分层**的中位数移动（因为混混杂符号随目标翻转）；
    - 若某目标工况的 `median R_cross` 在匹配后落入「与 1.0 不可区分」区间
      （|median−1| 小于该层的折间跨度），则把匹配前的超额**归因于训练量**，
      **不得**归因于域漂移；
    - 只有匹配后仍稳定 > 1 且超出折间跨度时，才可提出**域效应**，且必须与训练量讨论并列陈述。
21. **Stage 3 结论上限（继承 F2/F3）**：只能写「**未表现出稳定一致的优势**，
    其效应方向随留出轴承与任务而变化」；**禁止**「等效 / 无差异 / 证明无增益 / 证明无信息」。
22. **跨协议对比只用未校正口径**（A9）：`median_scaling_factor` 只在跨工况臂施加（单工况恒 1.0），
    校正口径**不得**用于跨协议对比；校正口径仅可作协议内部 headline。

23. **评测版本一致性（B9，2026-10-07 深夜新发现）**：`R_cross` 的**分子与分母必须来自同一代码版本的评测**。
    M1 基线 `matched_fold*/single` 生成于 B8 修复（`evaluation/inference.py`，10-07 16:21）**之前**，
    而 `matched_fold*/cross` 是 B8 之后（16:25–16:30）重算的 → 该基线的 `R_cross` 分母/分子**版本不一致**。
    修法：`scripts/33_reeval_checkpoints.py`（复用既有 checkpoint、**纯推理**，依据 A10 与 train→eval 逐位一致）
    统一到当前版本，输出到 `artifacts/reports/matched_fold{k}_b8/`，**不得**就地覆盖原始目录。
    实测版本偏差 ≤ **0.12%**（仅 `se_resnet_bigru` 行变化，trivial 基线行不变），
    量级不改变任何裁决，但**配对比较前必须统一**。

*End of STAGE1_PROTOCOL_FREEZE.md*
