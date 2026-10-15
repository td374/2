# Stage 6: Label Comparison Experiments

**目标**：对比三种标签（linear、piecewise、Exp-Concave），证明 SE-ResNet-BiGRU 在不同标签下都有效。

**创建日期**：2026-10-15  
**状态**：计划中  
**负责人**：用户 + Claude  

---

## 1. 实验设计

### 1.1 标签方案

| 标签 | 定义 | 参数 | 实现状态 |
|------|------|------|---------|
| **Linear** | `r = (N-t)/N` | h=0 | ✅ 已有（主标签） |
| **Piecewise** | 前 h 恒为 1，后 (1-h) 线性下降 | h=0.8 | ✅ 已有（对比） |
| **Exp-Concave** | `1 - (1-exp(-α)) * (1-exp(-αt/N)) / (1-exp(-α))` | α=2.0 | ⚠️ 待实现 |
| **Crossover-aware** | 双线性（h_cross=0.58 分段） | h_cross=0.58 | ⚠️ 可选 |

### 1.2 实验矩阵

| 数据集 | 标签 | 模型 | 优先级 |
|--------|------|------|--------|
| XJTU-SY | Linear | SE-ResNet-BiGRU | ✅ 已完成（s5/） |
| XJTU-SY | Piecewise (h=0.8) | SE-ResNet-BiGRU | ✅ 已完成（s5/sweep_h0.8/） |
| XJTU-SY | Exp-Concave (α=2.0) | SE-ResNet-BiGRU | ⭐⭐⭐ 必须 |
| PHM 2012 | Linear | SE-ResNet-BiGRU | ✅ 已完成（s5/phm2012_single_linear/） |
| PHM 2012 | Exp-Concave (α=2.0) | SE-ResNet-BiGRU | ⭐⭐⭐ 必须 |
| XJTU-SY | Crossover-aware | SE-ResNet-BiGRU | ⭐⭐ 可选 |

---

## 2. 目录结构

```
experiments/stage6_label_comparison/
├── PLAN.md                           ← 本文件
├── README.md                         ← 实验说明与结果汇总
├── code/
│   ├── label_functions.py            ← 标签函数实现
│   ├── run_exp_concave_xjtu.py       ← XJTU-SY Exp-Concave 实验
│   ├── run_exp_concave_phm.py        ← PHM2012 Exp-Concave 实验
│   └── run_crossover_aware.py        ← Crossover-aware 实验（可选）
├── results/
│   ├── xjtu_exp_concave/             ← XJTU-SY Exp-Concave 结果
│   ├── phm_exp_concave/              ← PHM2012 Exp-Concave 结果
│   └── xjtu_crossover_aware/         ← Crossover-aware 结果（可选）
├── analysis/
│   ├── compare_labels.py             ← 标签对比分析脚本
│   ├── comparison_table.csv          ← 对比表格
│   └── figures/                      ← 对比图（RMSE / LI 趋势）
└── audit/
    ├── determinism_check.md          ← 确定性验证
    └── protocol_compliance.md        ← 协议合规性检查
```

---

## 3. 实施步骤

### Phase 1：代码准备（1 天）

#### 3.1 实现标签函数
- [ ] 在 `code/label_functions.py` 中实现：
  - `exp_concave_label(t, N, alpha=2.0)` → IET 2024
  - `crossover_aware_label(t, N, h_cross=0.58)` → 自己的（可选）
- [ ] 单元测试：验证边界条件（t=0 → 1.0, t=N → 0.0）
- [ ] 可视化对比：绘制 4 种标签曲线（linear / piecewise / Exp-Concave / crossover）

#### 3.2 修改数据生成流程
- [ ] 修改 `rulbench/data/build.py`，加入 Exp-Concave 标签分支
- [ ] 生成数据集：
  - `artifacts/data_exp_concave/` （XJTU-SY）
  - `artifacts/data_phm_exp_concave/` （PHM2012）

### Phase 2：训练实验（3-5 天）

#### 2.1 XJTU-SY + Exp-Concave（优先级 ⭐⭐⭐）
- [ ] 运行 5 折交叉验证（与 s5/single_linear 相同设置）
- [ ] 保存 checkpoint 到 `results/xjtu_exp_concave/`
- [ ] 记录 RMSE（逐工况 + 均值）

#### 2.2 PHM 2012 + Exp-Concave（优先级 ⭐⭐⭐）
- [ ] 运行单次训练（cond1 / cond2 各自独立）
- [ ] 保存 checkpoint 到 `results/phm_exp_concave/`
- [ ] 记录 RMSE（逐工况 + 均值）

#### 2.3 Crossover-aware（可选，优先级 ⭐⭐）
- [ ] 仅在 XJTU-SY 上测试
- [ ] 即使 RMSE 不如 linear，也诚实报告

### Phase 3：分析与审计（1-2 天）

#### 3.1 对比分析
- [ ] 生成对比表格：

| 数据集 | 标签 | RMSE (cond1) | RMSE (cond2) | RMSE (cond3) | RMSE (mean) | LI |
|--------|------|-------------|-------------|-------------|-------------|-----|
| XJTU-SY | Linear | 4.54 | 72.54 | 35.44 | 37.51 | 0.897 |
| XJTU-SY | Piecewise (h=0.8) | 8.38 | 140.78 | 82.31 | 77.15 | 0.247 |
| XJTU-SY | Exp-Concave (α=2.0) | ? | ? | ? | ? | ? |
| PHM2012 | Linear | 2.01 | 0.27 | — | 1.14 | 0.784 |
| PHM2012 | Exp-Concave (α=2.0) | ? | ? | — | ? | ? |

- [ ] 绘制对比图（3 数据集 × 标签 RMSE 柱状图）

#### 3.2 确定性验证
- [ ] 重复训练 1 次（相同 seed），验证 RMSE 一致性
- [ ] 文档化：`audit/determinism_check.md`

#### 3.3 协议合规性
- [ ] 确认 A1-A10 规则都遵守
- [ ] 文档化：`audit/protocol_compliance.md`

---

## 4. 预期结果

### 4.1 成功场景

**Exp-Concave 略优于 linear**（+1-2 min）：
- 写法："The Exp-Concave non-linear label (Kim & Lee 2024) achieves **36.8 min RMSE** on XJTU-SY, slightly outperforming the linear label (37.5 min), suggesting the non-linear degradation profile better aligns with the bearing failure process."

**Exp-Concave 与 linear 相当**（差异 <1 min）：
- 写法："The Exp-Concave label yields **comparable performance** (37.2 vs 37.5 min), indicating the SE-ResNet-BiGRU architecture is **robust across label designs**."

### 4.2 失败场景（也有价值）

**Exp-Concave 不如 linear**（差 >2 min）：
- 写法："Interestingly, the Exp-Concave label (39.8 min) underperforms the linear label (37.5 min) on XJTU-SY. This may suggest that **the optimal label design is dataset-dependent**, and the linear assumption better fits the XJTU-SY degradation profiles despite its simplicity."
- 分析："We attribute this to the fact that XJTU-SY bearings exhibit relatively uniform degradation trajectories, whereas the Exp-Concave label assumes accelerated degradation that may not be present in this dataset."

---

## 5. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| Exp-Concave 训练不收敛 | 低 | 高 | 调整 α 参数（1.5 / 2.5），或学习率 |
| Exp-Concave RMSE 显著更差 | 中 | 中 | 诚实报告，分析原因，不影响主结论 |
| Crossover-aware 失败 | 中 | 低 | 舍弃该标签，不写入论文 |
| 确定性验证失败 | 低 | 高 | 检查播种、回退到已验证的代码版本 |

---

## 6. 时间估算

| 任务 | 工作量 | 时间 |
|------|--------|------|
| 代码准备 | 小 | 1 天 |
| XJTU-SY Exp-Concave | 中 | 2 天 |
| PHM2012 Exp-Concave | 小 | 0.5 天 |
| Crossover-aware（可选） | 小-中 | 1 天 |
| 分析 + 审计 | 中 | 1-2 天 |
| **总计（不含可选）** | — | **4-5 天** |
| **总计（含可选）** | — | **5-6 天** |

---

## 7. 交付物清单

- [ ] `code/label_functions.py` （标签实现）
- [ ] `results/xjtu_exp_concave/` （checkpoint + 报告）
- [ ] `results/phm_exp_concave/` （checkpoint + 报告）
- [ ] `analysis/comparison_table.csv` （对比表格）
- [ ] `analysis/figures/label_comparison.png` （对比图）
- [ ] `audit/determinism_check.md` （确定性验证报告）
- [ ] `audit/protocol_compliance.md` （协议合规报告）
- [ ] `README.md` （实验总结，包含主要数字）

---

## 8. 后续论文修改

实验完成后，需要修改的论文部分：

- [ ] Abstract：加入 Exp-Concave 对比结果
- [ ] §6.X（新增）：Label Comparison
  - Table X：三标签对比表
  - Figure X：标签曲线 + RMSE 柱状图
- [ ] Discussion：标签选择的影响分析
- [ ] Related Work：补充 IET 2024 标签论文
- [ ] Appendix（可选）：Crossover-aware 标签的尝试（如果失败）

---

**状态更新日志**：
- 2026-10-15：计划创建，等待用户确认后开始执行
