# Stage 6: Label Comparison — 废弃文件隔离与目录管理规范

**目的**：确保 Stage 6 实验不受旧代码/数据干扰，建立清晰的文件组织规范。

---

## 1. 废弃文件清单（已隔离到 `_deprecated/`）

### 1.1 失效的实验结果（Stage 1-3）

**原因**：确定性修复前（2026-10-07 09:44 之前）的所有结果不可复现。

```
artifacts/reports/single/              → 失效（确定性修复前）
artifacts/reports/single_linear/       → 失效（确定性修复前）
artifacts/reports/sweep_h{0.5..0.9}/   → 失效（确定性修复前）
artifacts/reports/phm2012/single_linear/  → 失效（修复前）
artifacts/reports/cross_uda_linear/    → 失效（修复前）
artifacts/reports/cross_source_only_linear/  → 失效（修复前）
```

**处理方式**：
- ✅ 已在 `audit/STAGE1_PROTOCOL_FREEZE.md` 中标记为失效
- ⚠️ **不删除**（留作审计证据），但**不引用**

---

### 1.2 失效的脚本（旧版，已被 Stage 5 重构）

```
scripts/01_*.py ~ 08_*.py              → 旧版数据处理脚本（已被 rulbench/ 替代）
scripts/14_*.py ~ 18_*.py              → 旧版标签扫描脚本（已被 09-13 替代）
```

**处理方式**：
- ✅ 保留在 `scripts/`（向后兼容，但不使用）
- ⚠️ Stage 6 **只用** `scripts/09_*.py` 及以后的脚本

---

## 2. 有效文件清单（当前可用）

### 2.1 有效的实验结果

| 目录 | 内容 | 状态 |
|------|------|------|
| `artifacts/reports/s5/` | Stage 5 全部结果（825 ckpt） | ✅ 可引用 |
| `artifacts/reports/s5/single_linear/` | XJTU-SY linear 主结果 | ✅ 可引用 |
| `artifacts/reports/s5/sweep_h0.8/` | XJTU-SY h=0.8 全阵容 | ✅ 可引用 |
| `artifacts/reports/s5/phm2012_single_linear/` | PHM2012 linear | ✅ 可引用 |
| `artifacts/reports/s4/` | Stage 4 消融实验（825 ckpt） | ✅ 可引用 |
| `artifacts/reports/matched_fold{1..5}/` | 跨工况配对实验 | ✅ 可引用 |

### 2.2 有效的脚本（Stage 5 后）

| 脚本 | 功能 | 状态 |
|------|------|------|
| `scripts/09_label_baseline_anchors.py` | 标签基线计算 | ✅ 使用 |
| `scripts/10_label_sensitivity.py` | 标签敏感性分析 | ✅ 使用 |
| `scripts/11_make_figures.py` | 绘图 | ✅ 使用 |
| `scripts/12_paper_tables.py` | 生成论文表格 | ✅ 使用 |
| `scripts/13_render_html.py` | 渲染 HTML | ✅ 使用 |
| `scripts/36_run_stage4.py` | Stage 4 实验 | ✅ 参考 |
| `scripts/37_stage4_analysis.py` | Stage 4 分析 | ✅ 参考 |

---

## 3. Stage 6 目录规范（新建）

```
experiments/stage6_label_comparison/
│
├── PLAN.md                           ← 实验计划（本目录的索引）
├── README.md                         ← 实验说明与最终结果汇总
│
├── code/                             ← Stage 6 专用代码
│   ├── label_functions.py            ← 标签函数实现
│   ├── run_exp_concave_xjtu.py       ← XJTU-SY Exp-Concave 实验
│   ├── run_exp_concave_phm.py        ← PHM2012 Exp-Concave 实验
│   ├── run_crossover_aware.py        ← Crossover-aware 实验（可选）
│   └── _tests/                       ← 单元测试
│       ├── test_label_functions.py
│       └── test_label_curves.py
│
├── results/                          ← 实验结果（checkpoint + 报告）
│   ├── xjtu_exp_concave/
│   │   ├── se_resnet_bigru__seed42.pt
│   │   ├── report.json
│   │   └── training_log.csv
│   ├── phm_exp_concave/
│   │   ├── se_resnet_bigru__seed42.pt
│   │   ├── report.json
│   │   └── training_log.csv
│   └── xjtu_crossover_aware/         ← 可选
│       ├── se_resnet_bigru__seed42.pt
│       └── report.json
│
├── analysis/                         ← 对比分析
│   ├── compare_labels.py             ← 对比脚本
│   ├── comparison_table.csv          ← 主表格
│   ├── comparison_summary.json       ← 机器可读结果
│   └── figures/
│       ├── label_curves.png          ← 4 种标签曲线
│       ├── rmse_comparison.png       ← RMSE 柱状图
│       └── li_vs_rmse.png            ← LI-RMSE 散点图
│
├── audit/                            ← 审计与质量控制
│   ├── determinism_check.md          ← 确定性验证（重复训练）
│   ├── protocol_compliance.md        ← A1-A10 合规性检查
│   └── seed_log.json                 ← 播种记录
│
└── _deprecated/                      ← Stage 6 内部废弃文件
    └── (实验失败的中间产物)
```

---

## 4. 命名规范

### 4.1 文件命名

| 类型 | 规范 | 示例 |
|------|------|------|
| 脚本 | `动词_描述.py` | `run_exp_concave_xjtu.py` |
| 结果目录 | `数据集_标签/` | `xjtu_exp_concave/` |
| checkpoint | `模型__seed种子.pt` | `se_resnet_bigru__seed42.pt` |
| 报告 | `report.json` | 固定名称 |
| 表格 | `描述_table.csv` | `comparison_table.csv` |
| 图片 | `描述.png` | `label_curves.png` |

### 4.2 变量命名（代码中）

| 概念 | 变量名 | 反例（禁止） |
|------|--------|-------------|
| 线性标签 | `linear` | `h0` / `lin` |
| 分段标签 | `piecewise_h08` | `pw` / `piece` |
| Exp-Concave | `exp_concave_a20` | `ec` / `nonlinear` |
| Crossover-aware | `crossover_h058` | `ca` / `cross` |

### 4.3 实验 ID（结果目录）

格式：`{dataset}_{label}_{optional_suffix}/`

- `xjtu_exp_concave/` ✅
- `xjtu_exp_concave_alpha15/` ✅（如果测试不同 α）
- `phm_exp_concave/` ✅
- `xjtu_crossover_h058/` ✅

---

## 5. 版本控制规则

### 5.1 Git 管理（如果使用）

**纳入版本控制**：
```
experiments/stage6_label_comparison/
├── PLAN.md
├── README.md
├── code/*.py
├── analysis/*.py
├── audit/*.md
└── .gitignore           ← 排除 results/ 和大文件
```

**排除版本控制**：
```
# .gitignore for Stage 6
results/                 # checkpoint 太大（用 Git LFS 或外部存储）
analysis/figures/*.png   # 可重新生成
*.pyc
__pycache__/
.DS_Store
```

### 5.2 备份策略

| 内容 | 备份方式 | 频率 |
|------|---------|------|
| 代码（`code/`） | Git | 每次修改 |
| checkpoint（`results/`） | 本地 ZIP | 实验完成后 |
| 表格/报告 | Git | 每次更新 |
| 审计文档 | Git | 每次验证后 |

**备份命令**（实验完成后）：
```bash
cd experiments/stage6_label_comparison
zip -r stage6_results_backup_$(date +%Y%m%d).zip results/ analysis/
mv stage6_results_backup_*.zip ../../_backups/
```

---

## 6. 质量控制清单

### 6.1 代码质量

- [ ] 所有标签函数有单元测试（`code/_tests/`）
- [ ] 代码有 docstring（说明参数、返回值）
- [ ] 关键步骤有注释（为什么这样做）
- [ ] 无硬编码路径（用相对路径或配置文件）

### 6.2 实验质量

- [ ] 每个实验有 `report.json`（记录超参、RMSE、播种）
- [ ] 确定性验证通过（重复训练 RMSE 一致）
- [ ] A1-A10 协议合规（`audit/protocol_compliance.md` 检查）
- [ ] 结果可追溯（checkpoint 文件名包含 seed）

### 6.3 文档质量

- [ ] `README.md` 包含主要数字（RMSE 表格）
- [ ] `PLAN.md` 实时更新进度（✅ / ⚠️ / ❌）
- [ ] `audit/*.md` 记录验证过程（不是结论，是证据）

---

## 7. 与主项目的集成

### 7.1 Stage 6 完成后，需要同步到主项目

| Stage 6 产物 | 主项目位置 | 操作 |
|-------------|-----------|------|
| `results/xjtu_exp_concave/` | `artifacts/reports/s6_xjtu_exp_concave/` | 复制 |
| `results/phm_exp_concave/` | `artifacts/reports/s6_phm_exp_concave/` | 复制 |
| `analysis/comparison_table.csv` | `paper/tables/label_comparison.csv` | 复制 |
| `analysis/figures/*.png` | `paper/figures/` | 复制并重命名 |
| `code/label_functions.py` | `rulbench/data/labels.py` | 合并 |

### 7.2 主项目脚本需要更新

- [ ] `scripts/12_paper_tables.py` 加入 Stage 6 表格生成
- [ ] `scripts/11_make_figures.py` 加入标签对比图
- [ ] `scripts/13_render_html.py` 渲染新表格/图片

---

## 8. 故障排查指南

### 问题 1：训练不收敛（Exp-Concave）

**症状**：loss 震荡 / RMSE > 100

**可能原因**：
1. α 参数太大（标签曲线太陡）
2. 学习率不匹配

**解决方案**：
1. 降低 α（2.0 → 1.5 → 1.0）
2. 调整学习率（0.001 → 0.0005）
3. 检查标签函数是否有 bug（单元测试）

---

### 问题 2：结果与 Stage 5 差异过大

**症状**：相同标签（linear），Stage 6 RMSE 与 s5/ 不一致

**可能原因**：
1. 数据预处理不同
2. 模型初始化种子不同
3. 训练超参不同

**解决方案**：
1. 检查 `rulbench/data/build.py` 是否被修改
2. 确认 seed 一致（训练 seed=42，数据 seed=0）
3. 对比 `report.json` 中的超参

---

### 问题 3：确定性验证失败

**症状**：重复训练 RMSE 差异 > 0.1 min

**可能原因**：
1. 未播种（CUDA / numpy / random）
2. 数据 shuffle 未固定
3. MC-Dropout 未播种（评测时）

**解决方案**：
1. 检查 `seed_everything()` 调用
2. 检查 DataLoader(shuffle=False, ...)
3. 检查 `predict_seed(seed=...)` 参数

---

## 9. 联系人与责任

| 角色 | 负责人 | 职责 |
|------|--------|------|
| 实验执行 | 用户 | 运行训练、记录结果 |
| 代码实现 | Claude | 编写标签函数、脚本框架 |
| 质量审计 | 双方 | 确定性验证、协议合规 |
| 论文整合 | Claude | 表格生成、图片绘制、文字修改 |

---

**最后更新**：2026-10-15 13:17  
**下一步行动**：用户确认后，开始编写 `code/label_functions.py`
