# STATISTICAL_UNIT_PROTOCOL.md —— 统计单位与推断协议（冻结）

> **触发**：独立审核（2026-10-07）指出——审计把 `Wilcoxon p = 0.153` 写进了「支持 H4」的证据行，
> 这是本轮审计**最弱的一环**：指出了「折间依赖未处理」，却仍让 pooled p 充当正式推断。
>
> **本文就是把那句提醒落实成规则。** 冻结后：
> - ✅ 允许：分布 / 效应量 / 轴承层级重采样区间 / 符号一致性；
> - ❌ 禁止：把 pooled cell-level `p = 0.153` 当作 H4 的正式统计证明。
>
> 依据：V2.0 §1.7、C4；`audit/stage0_scope_freeze.md` §2.1（H4 措辞纪律）。

---

## 1. 数据结构的真实形状（先看清，再谈统计）

`configs/matched_folds/fold{k}.yaml` 实测：

| fold | single train | val | **test** | cross target_test |
|---|---|---|---|---|
| fold1 | [3,4,5] | [2] | **[1]** | **[1]** |
| fold2 | [1,4,5] | [3] | **[2]** | **[2]** |
| fold3 | [1,2,5] | [4] | **[3]** | **[3]** |
| fold4 | [1,2,3] | [5] | **[4]** | **[4]** |
| fold5 | [2,3,4] | [1] | **[5]** | **[5]** |

**关键事实**：`fold{k}` 的测试轴承恒为**轴承 k**；且同一折内 `single.test ≡ cross.target_test`（逐位相同，
`audit_04` §2.1 已验证）。

因此把 5 折 × 3 目标工况铺开后：

```
5 折 × 3 目标工况 = 15 个「测试轴承实例」
                ≡ XJTU-SY 的全部 15 个轴承（3 工况 × 5 轴承）
```

**这 15 个实例才是独立统计单位。** 而每个实例内含 **3 个源任务变体**
（两个单源 + 一个多源，如目标 3 有 `1->3 / 2->3 / 1+2->3`），是**同一测试轴承上的重复测量**。

```
45 个 cell  = 15 个独立单位  ×  3 次重复测量        ← 不能当 45 个独立样本
```

> **最常见的误读**：「5 折 = 5 次重复实验」。**不是。** 五折是把**同一批 15 个轴承**换了角色重新标注
> （LOBO）——留出轴承从 1 轮到 5，测试轴承也随之从 1 变到 5。折不是复制，是**遍历轴承**。

---

## 2. 冻结规则

| # | 规则 |
|---|---|
| **S1** | **统计单位 = 测试轴承实例**（= `(fold, target condition)`，等价于 `(condition, bearing)`）。XJTU-SY 单模型下 **n = 15**。 |
| **S2** | **单位内聚合**：同一单位内的 3 个源任务变体为重复测量，先按**中位数**聚合成 1 个数，再进入推断。**不得**把 3 个变体当 3 个样本。 |
| **S3** | **折不是重复**：五折不提供额外样本量。折间结果只用于**稳健性**陈述（「符号在哪些留出轴承下翻转」），**不得**当 n=5 的复制。 |
| **S4** | **主报告量** = 分布而非单点：单位级 `R_cross` 的 **中位数、IQR、min–max、跨 1 的比例（win rate）、符号一致性**。 |
| **S5** | **效应量必报**：`R_cross = RMSE_cross/RMSE_single` **并列** `ΔRMSE = RMSE_cross − RMSE_single`（分钟）。 |
| **S6** | **区间用轴承层级重采样**：对 **n=15 个单位**做 percentile bootstrap（≥10 000 次，固定种子）得到中位数的 CI。**不得**对窗口或 45 个 cell 重采样。 |
| **S7** | **p 值降级**：cell-level pooled Wilcoxon 只能标为 `exploratory / descriptive`。若报单位级检验（n=15），也须标注「n=15，检验力低，仅作参考」。**H4 的主证据是 S4+S5+S6，不是任何 p 值。** |
| **S8** | **分层报告**：训练量混杂的方向**随目标工况翻转**（`audit_04` §3.2），故单位级统计**必须按目标工况分层**给出，不得只给总体。 |
| **S9** | **等效性**：**禁止**声称「等效」。若确需，必须**事先**设定有实际意义的等效界值并使用等效性检验（TOST）。 |
| **S10** | **残余依赖须披露**：即便 15 个单位，跨折的训练轴承仍有重叠 → 单位之间**并非严格独立**。因此 S6 的 CI 应表述为「真不确定性的**下界**（乐观）」。 |

---

## 3. 主报告格式（冻结）

每个（模型 × 口径）一张表，字段固定：

| 字段 | 定义 |
|---|---|
| `n_units` | 独立单位数（XJTU-SY 单模型 = 15） |
| `median_R_unit` | 单位级 `R_cross` 的中位数 |
| `IQR_R` | 单位级四分位距 |
| `win_rate_unit` | 单位级 `R_cross < 1` 的比例 |
| `median_dRMSE_min` | 单位级 `ΔRMSE`（分钟）的中位数 |
| `CI_R` / `CI_dRMSE` | 轴承层级 bootstrap 95% CI（S6） |
| `sign_consistency_folds` | 折中位数 > 1 的折数 / 总折数 |
| `effect_vs_fold_spread` | `|median_R−1|` 与折间跨度的对比 |
| `cells_descriptive` | 45 cell 的中位数等（**明确标 descriptive**） |
| `p_cell_exploratory` | pooled cell-level Wilcoxon（**明确标 exploratory，非核心证据**） |

---

## 4. 与 H4 的衔接（改写后的唯一允许措辞）

冻结后的 H4 结论句，**只能**是这一形态：

> **在严格配对的跨工况实验中，跨工况训练相对于目标工况内训练未表现出稳定一致的优势；
> 其效应方向随留出轴承与源任务而变化。**
> （单位级 `R_cross` 中位数 = X.XXX，95% CI [a, b]，`ΔRMSE` 中位数 = ±c 分钟，
> 单位级获胜比例 = p%，折中位数在 n/5 折中位于 1 的两侧。）

**禁止**：
- ❌「跨工况与单工况无显著差异（p = 0.153）」——把 cell-level p 当证据；
- ❌「两者等效 / 等价 / 不可区分」；
- ❌「跨工况训练没有价值」；
- ❌ 只报一个汇总 p 值而不报分布。

---

## 5. 对现有产物的处置

| 产物 | 处置 |
|---|---|
| `artifacts/reports/matched_fold_rcross.json` 的 `wilcoxon_p` | **降级**为 `wilcoxon_p_cell_level_exploratory`；引用时必须带 `exploratory` 字样 |
| `docs/matched_fold_rcross.md` 的 pooled p 段 | 加「exploratory only」标注；补单位级统计 |
| 新的单位级分析 | `scripts/23_stat_unit_rcross.py` → `artifacts/reports/stat_unit_rcross.json` + `docs/stat_unit_rcross.md` |
| `audit/audit_05_cross_protocol.md` §4.2 / `STAGE1_AUDIT_REPORT.md` §2.4 | 按本协议**改述**（见 `audit/REVIEW_ROUND1_RESPONSE.md` §3） |

---

## 6. 与 Stage 3 的接口

Stage 3 补齐**训练量匹配臂**后，本协议不变，只需把单位表的行数从「1 个训练方式」扩成
「自然跨工况 / 训练量匹配」两列，即可回答：

```
A（单工况）→ B（自然跨工况）：cross-condition 本身是否有优势？
B（自然跨工况）→ C（训练量匹配）：观察到的差异有多少来自训练量？
```

**在 C 臂完成前**（`audit/STAGE1_AUDIT_REPORT.md` §7.2 红线）：
任何 `R_cross` / `ΔRMSE` **只能作描述性陈述，不得写成工况迁移的因果结论**。

---

## 7. 冻结状态

- 冻结日期：2026-10-07（回应独立审核第 1 轮）。
- 本协议**替代**此前任何以 pooled p 值为核心的统计表述。
- 变更须走「新增标注为事后的修订」，不得静默改写。

*End of STATISTICAL_UNIT_PROTOCOL.md*
