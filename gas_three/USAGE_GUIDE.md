# 气体光谱分析系统 - 完整运行指南

## 📋 系统概述

本系统是一个基于**偏最小二乘法（PLS）回归**的气体光谱分析工具，专门用于预测**NO、NO2、SO2**三种气体的浓度。系统包含两个训练好的模型：Enhanced模型（基于增强数据集）和Standard模型（基于HITRAN标准数据库）。

### 🎯 主要功能
- **光谱数据预处理**：自动插值、标准化、噪声处理
- **多模型预测**：Enhanced + Standard 双模型预测
- **智能数据分割**：避免数据泄露的训练策略
- **综合结果输出**：提供多种格式的预测结果

---

## 🚀 快速运行指南

### 1. 系统状态检查（推荐第一步）
```bash
cd gas_three
python check_status.py
```

**输出示例**：
```
ENHANCED GAS SPECTROSCOPY SYSTEM STATUS
============================================================
1. CORE FILES STATUS:
   Base spectra: OK
   Enhanced dataset: OK
   Training data: OK
   Enhanced model: OK
   Data scalers: OK

5. SYSTEM STATUS:
   Status: READY
```

### 2. 最简单的使用方法

#### 方法A：一键检测（推荐）
```bash
python how_to_detect.py your_spectrum_file.csv
```

**输出示例**：
```
QUICK SPECTRUM DETECTION
========================================
File: your_spectrum_file.csv
Loaded: 3104 data points

DETECTION RESULTS:
------------------------------
Standard Model:
  NO: 0.274 (27.4%)
  NO2: 0.301 (30.1%)
  SO2: 0.426 (42.6%)

Enhanced Model:
  NO: 0.667 (66.7%)
  NO2: 0.333 (33.3%)
  SO2: 0.000 (0.0%)

Consensus (Average):
  NO: 0.470 (47.0%)
  NO2: 0.317 (31.7%)
  SO2: 0.213 (21.3%)
```

#### 方法B：完整系统使用
```bash
python how_to_use.py
```

#### 方法C：快速演示
```bash
python quick_start_example.py
```

### 3. 预测真实数据的多种方式

#### 方式1：标准数据路径
```bash
# 将您的光谱数据保存为 data/raw/X_real.csv，然后运行：
python predict_real.py
```

#### 方式2：指定文件路径
```bash
python detect_spectrum.py your_spectrum_file.csv
```

#### 方式3：使用特定模型
```bash
# 仅使用Standard模型
python predict_with_standard.py your_spectrum_file.csv
```

#### 方式4：Python代码集成
```python
from how_to_detect import quick_detect

# 快速检测
result = quick_detect('your_spectrum.csv')

# 结果格式
# result = {
#     'Standard': [NO_conc, NO2_conc, SO2_conc],
#     'Enhanced': [NO_conc, NO2_conc, SO2_conc]
# }
```

---

## 🔄 完整的训练流水线

如果需要重新训练模型或系统初次安装，按以下顺序执行：

### 步骤1：数据预处理
```bash
# 预处理HITRAN数据（需要../hitran_csv/目录下的原始数据）
python "01_preprocess.py（三气体版）.py"
```
**功能**：从HITRAN数据库读取NO、NO2、SO2的光谱数据，进行插值处理，生成统一波数轴的参考光谱。

### 步骤2：生成混合样本
```bash
# 生成单一混合样本
python "02_generate_mixture.py（三气体版）.py"
```
**功能**：根据不同浓度比例生成混合气体光谱样本。

### 步骤3：生成增强训练数据集
```bash
# 生成大规模训练数据集
python "03_generate_dataset.py（三气体版）.py"
```
**功能**：生成包含多层次噪声和非线性效应的增强训练数据集。

### 步骤4：数据分割和模型训练
```bash
# 智能数据分割
python data_splitter.py

# 训练增强模型
python enhanced_model_trainer.py
```

### 步骤5：训练标准PLS模型（可选）
```bash
# 默认参数PLS模型
python "04a_pls_model_default.py（三气体版）.py"

# 自定义参数PLS模型
python "04b_pls_model_custom.py（三气体版）.py"
```

### 一键重建系统
```bash
# 完整重建Enhanced模型
python how_to_use.py --retrain

# 重建Standard模型
python build_standard_model.py
```

---

## 📊 数据格式要求

### 输入数据格式
```csv
wavenumber,intensity
587.0,9.643139671328989e-26
588.0,8.733509209604609e-23
589.0,2.2142589032825055e-22
590.0,4.8356789012345678e-22
...
```

**要求**：
- **文件格式**：CSV
- **列1**：wavenumber（波数，cm⁻¹）或 frequency
- **列2**：intensity（强度）或 absorbance
- **数据范围**：建议覆盖 586.7-3690.6 cm⁻¹
- **数据点数**：系统会自动插值到3104个标准波数点

### 输出结果格式
```csv
Model,NO_concentration,NO_percentage,NO2_concentration,NO2_percentage,SO2_concentration,SO2_percentage
standard,0.2736,27.4,0.3006,30.1,0.4258,42.6
enhanced,0.6669,66.7,0.3331,33.3,0.0,0.0
CONSENSUS,0.4702,47.0,0.3169,31.7,0.2129,21.3
```

---

## 🗂️ 目录结构详解

```
gas_three/
├── 📁 核心训练文件
│   ├── 01_preprocess.py（三气体版）.py     # HITRAN数据预处理和插值
│   ├── 02_generate_mixture.py（三气体版）.py # 单一混合样本生成
│   ├── 03_generate_dataset.py（三气体版）.py # 增强训练数据集生成
│   ├── 04a_pls_model_default.py（三气体版）.py # 默认PLS模型训练
│   ├── 04b_pls_model_custom.py（三气体版）.py  # 自适应PLS模型训练
│   └── build_standard_model.py            # 标准模型构建器
│
├── 🔧 实用工具
│   ├── how_to_detect.py                    # 快速检测工具（推荐）
│   ├── detect_spectrum.py                  # 完整检测脚本
│   ├── predict_with_standard.py            # Standard模型预测
│   ├── predict_real.py                     # Enhanced模型预测
│   ├── how_to_use.py                       # 系统使用指南
│   ├── check_status.py                     # 系统状态检查
│   ├── demo.py                             # 演示脚本
│   └── quick_start_example.py              # 快速开始示例
│
└── 📊 数据管理
    └── data/
        ├── processed/                      # 预处理数据
        │   ├── interpolated_spectra.csv    # Enhanced模型参考光谱
        │   ├── standard_interpolated_spectra.csv # Standard模型参考光谱
        │   ├── X_dataset.csv               # 训练特征数据
        │   ├── Y_labels.csv                # 训练标签数据
        │   ├── X_train.csv, X_val.csv, X_test.csv # 数据分割结果
        │   └── generation_params.json      # 数据生成参数
        ├── raw/                            # 原始数据
        │   ├── X_real.csv                  # 待预测光谱数据
        │   └── Y_real.csv                  # 真实浓度（可选）
        ├── models/                         # 模型文件
        │   ├── enhanced_pls_model.pkl      # Enhanced PLS模型
        │   ├── scaler_X.pkl                # Enhanced模型X标准化器
        │   ├── scaler_Y.pkl                # Enhanced模型Y标准化器
        │   └── standard/
        │       ├── standard_pls_model.pkl  # Standard PLS模型
        │       ├── standard_scaler_X.pkl   # Standard模型X标准化器
        │       └── standard_scaler_Y.pkl   # Standard模型Y标准化器
        ├── results/                        # 预测结果
        │   ├── detection/                  # 检测结果
        │   └── standard/                   # Standard模型结果
        └── figures/                        # 可视化图表
```

---

## 🧠 模型技术详解

### 🔬 算法原理
- **算法**：偏最小二乘法（PLS）回归
- **实现**：`sklearn.cross_decomposition.PLSRegression`
- **原理**：将高维光谱数据降维到低维主成分空间，同时考虑输入和输出的协方差关系

### 📈 模型参数对比

| 参数 | Enhanced模型 | Standard模型 |
|------|-------------|-------------|
| **主成分数** | 10 | 10 |
| **输入特征** | 3,104 | 3,104 |
| **输出目标** | 3（NO, NO2, SO2） | 3（NO, NO2, SO2） |
| **训练样本** | 10,545 | 3,690 |
| **唯一组合** | 703 | 369 |
| **Test R²** | 0.86160 | 0.99122 |
| **Test RMSE** | 0.06311 | 0.01686 |

### 🎯 两个模型的区别

**Enhanced模型**：
- 基于增强数据集训练
- 包含复杂噪声模型和非线性效应
- 更真实的性能表现（R² ≈ 0.86）
- 适合一般光谱数据

**Standard模型**：
- 基于HITRAN标准数据库
- 专门针对标准气体优化
- 更高的预测精度（R² ≈ 0.99）
- 适合标准气体光谱

---

## 📋 故障排除

### 常见问题及解决方案

**1. 模型文件未找到**
```bash
# 检查系统状态
python check_status.py

# 重新构建模型
python how_to_use.py  # Enhanced模型
python build_standard_model.py  # Standard模型
```

**2. 特征维度不匹配**
- 系统会自动调整特征维度
- 如果数据点过少，会自动填充零值
- 如果数据点过多，会截取前3104个点

**3. 数据格式错误**
```python
# 检查数据格式
import pandas as pd
df = pd.read_csv('your_file.csv')
print(df.head())
print(df.columns)
```

**4. 预测结果异常**
- 检查输入光谱的波数范围是否合理
- 确认强度数值在合理范围内
- 比较两个模型的预测结果

**5. 系统未就绪**
```bash
# 如果check_status.py显示NOT READY，运行：
python how_to_use.py
```

---

## 🔧 高级配置

### 自定义数据生成参数
编辑 `03_generate_dataset.py（三气体版）.py`：
```python
# 噪声水平调整
base_noise_level = 0.08        # 基础噪声（默认8%）
spectral_noise_level = 0.05    # 光谱噪声（默认5%）
nonlinear_strength = 0.15      # 非线性强度（默认15%）

# 浓度范围调整
no_ratios = np.arange(0.05, 0.65, 0.02)   # NO浓度范围
no2_ratios = np.arange(0.05, 0.65, 0.02)  # NO2浓度范围
```

### 自定义模型参数
编辑训练脚本中的PLS参数：
```python
# 修改主成分数
model = PLSRegression(n_components=10)  # 可调整为5-20

# 修改交叉验证参数
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 📊 性能基准测试

### Enhanced模型性能
- **训练集 R²**：0.91134
- **测试集 R²**：0.86160
- **测试集 RMSE**：0.06311
- **泛化误差**：0.01415（合理范围）

### Standard模型性能
- **训练集 R²**：≈0.99
- **测试集 R²**：0.99122
- **测试集 RMSE**：0.01686
- **训练样本**：3,690个

### 单气体性能（Enhanced模型）
- **NO**：R² = 0.87017, RMSE = 0.05633
- **NO2**：R² = 0.84804, RMSE = 0.06008
- **SO2**：R² = 0.86658, RMSE = 0.07189

---

## 🔍 系统特色

### ✅ 解决的关键问题
1. **数据泄露**：训练测试集浓度组合零重叠
2. **过拟合**：从不现实的R²≈1.0降至合理的R²≈0.86
3. **数据复杂性**：多层次噪声模型模拟真实条件
4. **模型鲁棒性**：严格的交叉验证和数据分割

### 🎯 技术亮点
1. **智能数据分割**：基于浓度组合的分组分割
2. **多层次噪声**：高斯噪声 + 光谱相关噪声 + 系统偏差
3. **非线性效应**：模拟真实气体分子间相互作用
4. **双模型融合**：Enhanced + Standard 模型互补

---

## 📞 技术支持

### 如需帮助
1. **系统状态检查**：`python check_status.py`
2. **运行演示程序**：`python demo.py`
3. **查看详细日志**：检查命令行输出信息
4. **重建系统**：`python how_to_use.py --retrain`

### 联系信息
- **项目路径**：`E:\generate_mixture\gas_three\`
- **模型存储**：`data/models/` 目录
- **结果输出**：`data/results/` 目录

---

## 📄 引用信息

如果您在研究中使用了本系统，请引用：
- **算法**：Partial Least Squares (PLS) Regression
- **数据源**：HITRAN Database
- **实现**：scikit-learn PLSRegression
- **模型**：Enhanced + Standard 双模型架构

---

## 🎬 使用流程总结

### 新用户首次使用
```bash
1. cd gas_three
2. python check_status.py        # 检查状态
3. python how_to_use.py          # 如果系统未就绪，自动设置
4. python how_to_detect.py your_spectrum.csv  # 开始预测
```

### 日常使用
```bash
# 最简单的方式
python how_to_detect.py your_spectrum.csv

# 或者使用标准流程
python predict_real.py  # 需要将数据放在 data/raw/X_real.csv
```

### 系统维护
```bash
python check_status.py           # 定期检查系统状态
python how_to_use.py --retrain   # 重新训练模型
python build_standard_model.py   # 重建标准模型
```

---

*最后更新：2025年1月*