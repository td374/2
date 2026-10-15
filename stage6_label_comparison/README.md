# Stage 6: Label Comparison Experiments

**目标**：验证 SE-ResNet-BiGRU 在不同 RUL 标签设计下的鲁棒性，补充标签消融实验。

**状态**：🟡 进行中（代码准备完成，等待训练）  
**创建日期**：2026-10-15  
**预计完成**：2026-10-20  

---

## 快速导航

- **实验计划**：[PLAN.md](PLAN.md)
- **文件管理规范**：[_DIRECTORY_MANAGEMENT.md](_DIRECTORY_MANAGEMENT.md)
- **标签函数代码**：[code/label_functions.py](code/label_functions.py)
- **结果目录**：[results/](results/)
- **分析脚本**：[analysis/](analysis/)

---

## 实验设计摘要

### 标签方案（4 种）

| 标签 | 定义 | 参数 | LI | 状态 |
|------|------|------|-----|------|
| **Linear** | `r = (N-t)/N` | h=0 | 0.750 | ✅ 已完成（s5/） |
| **Piecewise** | 前 h 平台 + 后线性 | h=0.8 | 0.150 | ✅ 已完成（s5/） |
| **Exp-Concave** | 加速非线性（IET 2024） | α=2.0 | 0.846 | 🟡 进行中 |
| **Crossover-aware** | 双线性（基于 Stage 5 发现） | h_cross=0.58 | 0.611 | ⚪ 可选 |

**关键洞察**：
- Exp-Concave 的 LI=0.846 **最高**（理论上信息量最丰富）
- 但 LI 高不等于 RMSE 低（取决于标签与真实退化的匹配度）

### 实验矩阵

| 数据集 | 标签 | 优先级 | 时间估算 |
|--------|------|--------|---------|
| XJTU-SY | Exp-Concave (α=2.0) | ⭐⭐⭐ 必须 | 2 天 |
| PHM 2012 | Exp-Concave (α=2.0) | ⭐⭐⭐ 必须 | 0.5 天 |
| XJTU-SY | Crossover-aware | ⭐⭐ 可选 | 1 天 |

---

## 当前进度

### Phase 1: 代码准备 ✅ **完成**

- [x] 实现 4 种标签函数（`code/label_functions.py`）
- [x] 单元测试：边界条件验证（t=0 → 1.0, t=N → 0.0）
- [x] 自检通过：所有标签 r(0)=1.0, r(N)=0.0
- [x] LI 计算验证（与理论值一致）
- [ ] ⚠️ 标签曲线可视化（plot_label_comparison 有 bug，待修复）

### Phase 2: 数据生成 🟡 **进行中**

- [ ] 修改 `rulbench/data/build.py`，加入 Exp-Concave 标签
- [ ] 生成 `artifacts/data_exp_concave/`（XJTU-SY）
- [ ] 生成 `artifacts/data_phm_exp_concave/`（PHM2012）

### Phase 3: 训练实验 ⚪ **未开始**

- [ ] XJTU-SY + Exp-Concave（5 折交叉验证）
- [ ] PHM 2012 + Exp-Concave（单次训练）
- [ ] Crossover-aware（可选）

### Phase 4: 分析 & 审计 ⚪ **未开始**

- [ ] 生成对比表格（comparison_table.csv）
- [ ] 绘制对比图（RMSE 柱状图、标签曲线）
- [ ] 确定性验证（重复训练 1 次）
- [ ] 协议合规性检查（A1-A10）

---

## 预期结果

### 场景 A：Exp-Concave 优于 Linear（最理想）

**RMSE 对比**（假设）：
```
XJTU-SY:  Linear 37.5 min  →  Exp-Concave 36.2 min  (提升 3.5%)
PHM2012:  Linear 1.14 h    →  Exp-Concave 1.08 h    (提升 5.3%)
```

**写法**：
> "The Exp-Concave non-linear label (Kim & Lee 2024) achieves **36.2 min RMSE** on XJTU-SY, outperforming the linear label (37.5 min) by 3.5%, suggesting the accelerated degradation profile better captures the bearing failure progression."

---

### 场景 B：Exp-Concave 与 Linear 相当（也可接受）

**RMSE 对比**（假设）：
```
XJTU-SY:  Linear 37.5 min  →  Exp-Concave 37.2 min  (差异 <1%)
PHM2012:  Linear 1.14 h    →  Exp-Concave 1.16 h    (差异 <2%)
```

**写法**：
> "The Exp-Concave label yields **comparable performance** (37.2 vs 37.5 min on XJTU-SY, 1.16 vs 1.14 h on PHM2012), indicating the SE-ResNet-BiGRU architecture is **robust across label designs** and does not rely on a specific degradation assumption."

---

### 场景 C：Exp-Concave 不如 Linear（仍有价值）

**RMSE 对比**（假设）：
```
XJTU-SY:  Linear 37.5 min  →  Exp-Concave 40.1 min  (差 7%)
PHM2012:  Linear 1.14 h    →  Exp-Concave 1.22 h    (差 7%)
```

**写法**：
> "Interestingly, the Exp-Concave label (40.1 min) underperforms the linear label (37.5 min) on XJTU-SY despite its higher label informativeness (LI=0.846 vs 0.750). This suggests that **optimal label design is dataset-dependent**: the linear assumption may better fit XJTU-SY's relatively uniform degradation trajectories, whereas Exp-Concave's accelerated profile may suit datasets with pronounced end-of-life acceleration (e.g., turbofan engines in C-MAPSS)."

**分析**：
- 承认负面结果
- 给出合理解释（数据集特性不同）
- 不影响主结论（SE-ResNet-BiGRU 的方法创新）

---

## 交付物清单

### 代码（`code/`）
- [x] `label_functions.py` — 4 种标签实现 + LI 计算
- [ ] `run_exp_concave_xjtu.py` — XJTU-SY 训练脚本
- [ ] `run_exp_concave_phm.py` — PHM2012 训练脚本
- [ ] `run_crossover_aware.py` — Crossover-aware 训练脚本（可选）

### 结果（`results/`）
- [ ] `xjtu_exp_concave/` — checkpoint + report.json
- [ ] `phm_exp_concave/` — checkpoint + report.json
- [ ] `xjtu_crossover_aware/` — checkpoint + report.json（可选）

### 分析（`analysis/`）
- [ ] `comparison_table.csv` — 三标签对比表
- [ ] `comparison_summary.json` — 机器可读结果
- [ ] `figures/label_curves.png` — 标签曲线对比
- [ ] `figures/rmse_comparison.png` — RMSE 柱状图
- [ ] `figures/li_vs_rmse.png` — LI-RMSE 散点图

### 审计（`audit/`）
- [ ] `determinism_check.md` — 确定性验证报告
- [ ] `protocol_compliance.md` — A1-A10 合规性检查
- [ ] `seed_log.json` — 播种记录

---

## 后续论文修改

实验完成后，需要更新的论文部分：

### Abstract
```diff
- On XJTU-SY with the full-life linear label, SE-ResNet-BiGRU achieves 37.5 min RMSE...
+ On XJTU-SY with the full-life linear label, SE-ResNet-BiGRU achieves 37.5 min RMSE...
+ The Exp-Concave non-linear label (Kim & Lee 2024) yields [XX.X] min, 
+ demonstrating architectural robustness across label designs.
```

### New Section: §6.X Label Comparison
- Table X: 三标签对比（XJTU-SY + PHM2012）
- Figure X: 标签曲线 + RMSE 柱状图
- 分析：标签选择对性能的影响

### Discussion
- 补充：标签-数据集匹配性的讨论
- 补充：LI 理论与实际 RMSE 的关系

### Related Work
- 补充：Kim & Lee (2024) IET 非线性标签论文
- 对比：你的系统性扫描 vs 他们的单点提出

---

## 时间估算

| 阶段 | 任务 | 时间 | 负责人 |
|------|------|------|--------|
| Phase 1 | 代码准备 | ✅ 1 天（已完成） | Claude |
| Phase 2 | 数据生成 | 0.5 天 | 用户 |
| Phase 3 | 训练实验 | 2.5 天（XJTU 2d + PHM 0.5d） | 用户 |
| Phase 4 | 分析审计 | 1 天 | Claude + 用户 |
| **总计** | — | **4-5 天** | — |

---

## 联系与支持

- **实验执行**：用户负责运行训练脚本、监控进度
- **代码开发**：Claude 负责脚本框架、分析脚本、绘图
- **质量保证**：双方共同审计（确定性、协议合规）

**问题反馈**：遇到问题请在 `_deprecated/` 中记录失败的尝试，便于追溯。

---

**最后更新**：2026-10-15 13:17  
**下一步行动**：修改 `rulbench/data/build.py`，加入 Exp-Concave 标签生成逻辑
