# Exp-Concave标签实验执行清单

**日期**: 2026-10-15  
**目标**: 在XJTU-SY和PHM2012上训练Exp-Concave标签,对比linear标签

---

## ✅ 已完成(Step 1)

### 1.1 代码实现
- [x] `rulbench/data/labels.py`: 新增`exp_concave_rul`函数
- [x] 注册到`LABEL_FACTORIES`
- [x] 更新`build_labels`函数支持`exp_concave`策略
- [x] 添加`alpha`参数(默认2.0)

### 1.2 配置文件
- [x] `configs/exp_concave.yaml`: XJTU-SY配置
- [x] `configs/exp_concave_phm2012.yaml`: PHM2012配置

### 1.3 测试脚本
- [x] `scripts/test_exp_concave_label.py`: 验证实现+可视化

---

## 📋 待执行(按顺序)

### Step 2: 验证实现(预计10分钟)

**命令**:
```bash
cd C:\Users\28129\Desktop\论文改写整理发表？\修改版——1
python scripts/test_exp_concave_label.py
```

**检查点**:
- [ ] 边界检查通过(r(0)=1.0, r(N)=0.0)
- [ ] 单调性检查通过
- [ ] LI计算正确(Exp-Concave LI ≈ 0.84-0.85)
- [ ] 生成2张图: `label_comparison.png`, `alpha_sensitivity.png`

**预期LI值**:
- Linear (h=0): LI ≈ 0.75
- Piecewise (h=0.8): LI ≈ 0.15
- Exp-Concave (α=2.0): LI ≈ 0.846

---

### Step 3: 数据准备 - XJTU-SY(预计1小时)

**⚠️ 前置条件**: 修改`configs/exp_concave.yaml`中的`data.root`为实际路径

**命令**:
```bash
python -m rulbench.cli prepare --config configs/exp_concave.yaml
```

**产出**:
- `artifacts/data_exp_concave/`: NPZ文件(15个轴承)
- 每个NPZ包含: X(特征), y_ratio(标签), y_minutes, metadata

**检查点**:
- [ ] 15个NPZ文件生成
- [ ] 标签范围检查: `y_ratio ∈ [0, 1]`
- [ ] LI验证: 读取一个NPZ,计算LI是否≈0.85

---

### Step 4: 训练评测 - XJTU-SY(预计6-8小时,可后台)

**命令**:
```bash
# 方式1: 前台运行
python -m rulbench.cli single --config configs/exp_concave.yaml

# 方式2: 后台运行(推荐)
nohup python -m rulbench.cli single --config configs/exp_concave.yaml > exp_concave.log 2>&1 &
```

**产出**:
- `artifacts/checkpoints/exp_concave/`: 75个checkpoint(5种子×15轴承)
- `artifacts/reports/exp_concave/results_full.json`: 完整结果
- `artifacts/reports/exp_concave/summary_by_model.csv`: RMSE汇总

**检查点**:
- [ ] 训练完成(150 epochs × 15 bearings × 5 seeds)
- [ ] RMSE范围合理(预期35-45 min)
- [ ] 与linear对比(预期差异±2 min)

**预期结果**(基于IET 2024):
- XJTU-SY Exp-Concave: RMSE ≈ 37-39 min(vs Linear 37.5 min)
- 可能比linear好1-2 min,也可能相当

---

### Step 5: 数据准备 - PHM2012(预计20分钟)

**⚠️ 前置条件**: 确认`configs/phm2012.yaml`中`data.root`正确

**命令**:
```bash
python -m rulbench.cli prepare --config configs/exp_concave_phm2012.yaml
```

**产出**:
- `artifacts/data_exp_concave_phm2012/`: 7个NPZ

---

### Step 6: 训练评测 - PHM2012(预计3-4小时)

**命令**:
```bash
python -m rulbench.cli single --config configs/exp_concave_phm2012.yaml
```

**产出**:
- `artifacts/checkpoints/exp_concave_phm2012/`: 35个checkpoint
- `artifacts/reports/exp_concave_phm2012/results_full.json`

**预期结果**:
- PHM2012 Exp-Concave: RMSE ≈ 1.1-1.2 h(vs Linear 1.14 h)

---

### Step 7: 结果对比分析(预计1小时)

**创建分析脚本**(`scripts/compare_labels.py`):
```python
import json
import pandas as pd

# 读取三种标签的结果
linear = json.load(open("artifacts/reports/single_linear/results_full.json"))
piecewise = json.load(open("artifacts/reports/single_piecewise/results_full.json"))
exp_concave = json.load(open("artifacts/reports/exp_concave/results_full.json"))

# 提取RMSE
# ... (生成对比表格)
```

**产出**:
- 标签对比表(3标签×2数据集×5种子)
- 统计显著性检验(paired t-test)
- 可视化图(箱线图/条形图)

---

## 📊 预期实验结果

### 情况A: Exp-Concave优于Linear(理想)

| 数据集 | Linear RMSE | Exp-Concave RMSE | 改进 |
|--------|-------------|------------------|------|
| XJTU-SY | 37.5 min | 36.2 min | ✓ -3.5% |
| PHM2012 | 1.14 h | 1.09 h | ✓ -4.4% |

**论文措辞**:
> "The Exp-Concave label (α=2.0) outperforms the linear label on both datasets, supporting the hypothesis that **matching the label curvature to accelerated degradation improves prediction**."

---

### 情况B: Exp-Concave与Linear相当(中性)

| 数据集 | Linear RMSE | Exp-Concave RMSE | 差异 |
|--------|-------------|------------------|------|
| XJTU-SY | 37.5 min | 37.8 min | +0.8% |
| PHM2012 | 1.14 h | 1.13 h | -0.9% |

**论文措辞**:
> "The Exp-Concave label yields **comparable performance** to the linear label (37.8 vs 37.5 min on XJTU-SY), indicating the architecture is **robust across label designs**. This suggests task learnability is not strongly dependent on matching label curvature to degradation profile."

---

### 情况C: Exp-Concave劣于Linear(需解释)

| 数据集 | Linear RMSE | Exp-Concave RMSE | 差异 |
|--------|-------------|------------------|------|
| XJTU-SY | 37.5 min | 40.1 min | +6.9% |
| PHM2012 | 1.14 h | 1.21 h | +6.1% |

**论文措辞**:
> "The Exp-Concave label underperforms the linear label (40.1 vs 37.5 min on XJTU-SY), suggesting that for the bearing datasets tested, **the uniform decay assumption (linear) is more robust** than the accelerated degradation hypothesis. This may indicate that bearing degradation in these datasets is more uniform than Paris crack growth would predict."

---

## ⏱️ 时间估算

| 步骤 | 时长 | 可并行 |
|------|------|--------|
| Step 2: 验证实现 | 10分钟 | — |
| Step 3: XJTU-SY数据准备 | 1小时 | — |
| Step 4: XJTU-SY训练 | 6-8小时 | ✓ 后台 |
| Step 5: PHM2012数据准备 | 20分钟 | ✓ 可与Step 4并行 |
| Step 6: PHM2012训练 | 3-4小时 | ✓ 后台 |
| Step 7: 结果分析 | 1小时 | — |
| **总计** | **12-15小时** | 实际1-2天(后台运行) |

---

## 🚨 常见问题

### Q1: alpha应该用多少?
**A**: 默认2.0(IET 2024),可选扫描{1.5, 2.0, 2.5}

### Q2: 如果Exp-Concave不如linear怎么办?
**A**: 诚实报告(见情况C措辞),仍然有学术价值

### Q3: 需要调超参数吗?
**A**: 不需要,保持与linear相同的超参数(公平对比)

### Q4: 如何确认实现正确?
**A**: 
1. LI ≈ 0.846(理论值)
2. r(0)=1.0, r(N)=0.0
3. 单调递减
4. 曲线形状符合"先慢后快"

---

## 📝 下一步

完成Step 2-7后:
- [ ] 生成标签对比表
- [ ] 更新`完整实施方案与框架.md`的§11.2关键数字
- [ ] 撰写论文Results章节的标签对比小节
- [ ] 更新Abstract提及Exp-Concave

---

**准备好后,从Step 2开始执行!**
