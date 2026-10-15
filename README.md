# 修改版——1：SCI 投稿准备项目

**项目名称**：基于 SE-ResNet-BiGRU 的滚动轴承剩余寿命预测：多数据集与多标签验证  
**创建日期**：2026-10-15  
**状态**：🟡 Stage 6 准备中  
**目标期刊**：Measurement（2 区）/ Sensors（3 区）

---

## 📁 项目结构

```
修改版——1/
│
├── README.md                         ← 本文件（项目总览）
│
├── stage6_label_comparison/          ← 🟡 Stage 6：标签对比实验
│   ├── PLAN.md                       ← 实验计划（8 章节）
│   ├── README.md                     ← 进度追踪
│   ├── _DIRECTORY_MANAGEMENT.md      ← 文件管理规范
│   ├── code/
│   │   └── label_functions.py        ← ✅ 4 种标签函数（已自检）
│   ├── results/                      ← 实验结果（待训练）
│   ├── analysis/                     ← 分析脚本 + 图表
│   └── audit/                        ← 审计文档
│
├── scripts/                          ← 有效脚本（35 个）
│   ├── 09_label_baseline_anchors.py
│   ├── 10_label_sensitivity.py
│   ├── 11_make_figures.py
│   ├── 12_paper_tables.py
│   ├── 13_render_html.py
│   ├── 36_run_stage4.py
│   ├── 37_stage4_analysis.py
│   ├── 38_stage4_determinism_probe.py
│   └── ... (其他 27 个脚本)
│
├── audit/                            ← 审计文档（22 个）
│   ├── STAGE1_PROTOCOL_FREEZE.md     ← 协议冻结（F1-F5, 红线 1-23）
│   ├── STAGE4_FINAL_INTERPRETATION_MEMO.md  ← 改稿唯一依据
│   ├── STAGE4_PROTOCOL_FREEZE.md     ← Stage 4 协议
│   ├── STAGE5_RESULTS.md             ← Stage 5 结果收敛
│   ├── CLAUDE_REVIEW_2026-10-14.md   ← Claude 代码+逻辑审查
│   └── ... (其他 17 个审计文档)
│
├── paper/                            ← 论文文件（4 个）
│   ├── manuscript.md                 ← 英文定稿（62,995 字符）
│   ├── manuscript.html               ← 英文 HTML 渲染
│   ├── manuscript_zh.md              ← 中文核验版（29,261 字符）
│   └── manuscript_zh.html            ← 中文 HTML 渲染
│
├── rulbench/                         ← 代码库（模块化）
│   ├── data/
│   │   ├── build.py                  ← 数据预处理入口
│   │   ├── labels.py                 ← 标签函数（待加 Exp-Concave）
│   │   └── ...
│   ├── models/
│   │   ├── se_resnet_bigru.py
│   │   └── ...
│   ├── training/
│   └── evaluation/
│
├── datasets_reference.txt            ← 数据集位置记录
└── _DEPRECATED_FILES_README.md       ← 废弃文件说明
```

---

## 🎯 当前任务：Stage 6 标签对比实验

### 目标
验证 SE-ResNet-BiGRU 在不同 RUL 标签设计下的鲁棒性。

### 标签方案（4 种）

| 标签 | LI | 状态 | 优先级 |
|------|-----|------|--------|
| Linear (h=0) | 0.750 | ✅ 已完成（s5/） | — |
| Piecewise (h=0.8) | 0.150 | ✅ 已完成（s5/） | — |
| **Exp-Concave (α=2.0)** | **0.846** | 🟡 进行中 | ⭐⭐⭐ 必须 |
| Crossover-aware (h=0.58) | 0.611 | ⚪ 可选 | ⭐⭐ 可选 |

### 时间估算
- Phase 1（数据生成）：0.5 天
- Phase 2（训练实验）：2.5 天
- Phase 3（分析审计）：1 天
- Phase 4（论文修改）：1 天
- **总计**：**4-5 天**

### 下一步行动
1. [ ] 修改 `rulbench/data/build.py`，加入 Exp-Concave 标签
2. [ ] 生成 `artifacts/data_exp_concave/`
3. [ ] 训练 XJTU-SY + Exp-Concave（5 折）
4. [ ] 训练 PHM2012 + Exp-Concave（单次）
5. [ ] 分析 + 生成表格/图表
6. [ ] 更新论文（Abstract + §6.X + Discussion）

---

## 📊 已完成的工作

### Stage 1-5（已封板）
- ✅ Stage 1：协议冻结（F1-F5，红线 1-23）
- ✅ Stage 2：标签敏感性分析（crossover h≈0.55-0.60）
- ✅ Stage 3：跨工况迁移（R_cross 未稳定优势）
- ✅ Stage 4：组件消融（ΔG 全部跨 0，无稳定增益）
- ✅ Stage 5：全阵容重跑（825 ckpt，零失败）

### 数据集（2 个）
- ✅ XJTU-SY（3 工况，15 轴承）
- ✅ IEEE PHM 2012 / PRONOSTIA（3 工况，17 轴承，**同一数据集**）

### 论文（英文定稿 + 中文核验）
- ✅ `manuscript.md`（62,995 字符，954 行）
- ✅ `manuscript_zh.md`（29,261 字符，757 行）
- ✅ Claude 代码+逻辑审查（3 必修 + 2 建议 + 3 预判问题）

---

## 🔗 数据集位置（不迁移，仅记录）

### 原始数据
```
D:\1毕业论文文献，方法\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets.part06\XJTU-SY_Bearing_Datasets
```
**大小**：3.2 GB  
**内容**：XJTU-SY 原始振动信号（15 个轴承，3 工况）

### 预处理数据
```
C:\Users\28129\Desktop\论文改写整理发表？\xjtu_sy_rul\datasets\PHM2012\
```
**大小**：约 500 MB  
**内容**：IEEE PHM 2012（= PRONOSTIA）预处理后数据（17 个轴承，3 工况）

### 有效实验结果（checkpoint）
```
C:\Users\28129\Desktop\论文改写整理发表？\xjtu_sy_rul\artifacts\reports\
├── s5/                  ← Stage 5 全部结果（825 ckpt，可引用）
├── s4/                  ← Stage 4 消融实验（825 ckpt，可引用）
├── s2_chain/            ← Stage 2 确定性校验基准
└── matched_fold{1..5}/  ← 跨工况配对实验
```
**大小**：约 3.6 GB（checkpoint）  
**访问方式**：直接从原位置读取，不复制到 `修改版——1/`

---

## 🎓 核心贡献（定版）

### 论文定位
**方法创新（SE-ResNet-BiGRU）+ 标签探索 + 评估严格**

### 四大贡献
1. **SE-ResNet-BiGRU 架构**（方法创新）
2. **多数据集验证**（XJTU-SY + PHM2012）
3. **A1-A10 评估协议**（可复现性保证）
4. **标签结构分析**（crossover 现象 h≈0.55-0.60）

### 投稿期刊（SCI 2-3 区）
- **首选**：Measurement（2 区，IF≈5.1）
- **次选**：Sensors（3 区，IF≈3.4）
- **保底**：Applied Sciences（3 区，IF≈2.5）

---

## ⚠️ 重要约定

### 有效结果（可引用）
- ✅ `s5/`（Stage 5 全部结果）
- ✅ `s4/`（Stage 4 消融实验）
- ✅ `matched_fold{1..5}/`（跨工况配对）

### 失效结果（禁止引用）
- ❌ `single/`、`single_linear/`（确定性修复前）
- ❌ `sweep_h{0.5..0.9}/`（确定性修复前）
- ❌ `cross_uda_linear/`（确定性修复前）

### 脚本使用规则
- ✅ **使用**：`09-13.py`、`36-38.py`
- ⚠️ **参考但不运行**：`01-08.py`、`14-18.py`（已被重构）

### 数据访问规则
- ✅ **直接读取原位置**（不复制大文件）
- ✅ **记录位置**（`datasets_reference.txt`）
- ❌ **禁止混用新旧路径**

---

## 📞 联系与协作

| 角色 | 负责人 | 职责 |
|------|--------|------|
| 实验执行 | 用户 | 运行训练、监控进度、记录结果 |
| 代码开发 | Claude | 脚本框架、标签实现、分析工具 |
| 质量审计 | 双方 | 确定性验证、协议合规检查 |
| 论文修改 | Claude | 表格生成、图片绘制、文字改写 |

---

## 📝 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-10-15 | 项目重组，迁移有效文件到 `修改版——1/` |
| v1.1 | 2026-10-15 | 创建 Stage 6 实验框架，实现标签函数 |

---

**最后更新**：2026-10-15 13:38  
**下一次审查**：Stage 6 完成后（预计 2026-10-20）
