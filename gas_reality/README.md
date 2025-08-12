# Gas Reality - Advanced GCMS Spectroscopy Analysis System

## 📋 系统概述

**Gas Reality** 是一个基于真实实验数据的高级气体质谱分析系统，专门用于分析6种化学物质的浓度：**SOF2、SO2F2、SO2、NO、NO2、NF3**。系统采用**偏最小二乘法（PLS）回归**作为核心算法，支持多电压（22kV、24kV、36kV）和时间序列数据的复杂建模。

### 🎯 主要特色
- **真实实验数据**：基于GCMS实验测量的光谱和浓度数据
- **6种气体分析**：SOF2、SO2F2、SO2、NO、NO2、NF3
- **多电压支持**：22kV、24kV、36kV电压条件
- **时间序列建模**：0-120小时的动态变化追踪
- **高级PLS算法**：个体模型 + 多目标模型的双重架构
- **智能预处理**：噪声去除、基线校正、光谱插值
- **实时预测**：训练后可用于新光谱数据的实时分析

---

## 🗂️ 系统架构

```
gas_reality/
├── src/                           # 核心源代码
│   ├── preprocessing/            # 数据预处理模块
│   │   ├── gcms_data_loader.py   # GCMS数据加载器
│   │   └── spectral_preprocessor.py # 光谱预处理器
│   ├── modeling/                # 建模模块
│   │   └── advanced_pls_trainer.py # 高级PLS训练器
│   ├── analysis/                # 分析工具
│   └── utils/                   # 通用工具
├── data/                        # 数据存储
│   ├── raw/                     # 原始数据
│   ├── processed/               # 预处理数据
│   ├── models/                  # 训练模型
│   └── results/                 # 结果输出
├── configs/                     # 配置文件
├── docs/                        # 文档
├── tests/                       # 测试文件
├── main_pipeline.py             # 主要分析流水线
├── predict_concentrations.py    # 浓度预测工具
└── README.md                    # 说明文档
```

---

## 🚀 快速开始

### 1️⃣ 运行完整分析流水线

```bash
# 运行完整的GCMS数据分析流水线
python main_pipeline.py
```

这将自动执行：
- ✅ 加载GCMS光谱数据（32个CSV文件）
- ✅ 加载浓度数据（3个Excel文件）
- ✅ 高级光谱预处理
- ✅ 训练PLS回归模型
- ✅ 生成分析报告

### 2️⃣ 使用训练好的模型进行预测

```bash
# 预测单个光谱文件的气体浓度
python predict_concentrations.py path/to/your/spectrum.csv

# 例如，预测GCMS目录中的文件
python predict_concentrations.py E:/generate_mixture/gcms/22kv0h.CSV
```

**输出示例**：
```
============================================================
GAS CONCENTRATION PREDICTIONS
============================================================

INDIVIDUAL Model Predictions:
----------------------------------------
  SOF2  : 0.2350 ( 23.5%)
  SO2F2 : 0.1820 ( 18.2%)
  SO2   : 0.3120 ( 31.2%)
  NO    : 0.1560 ( 15.6%)
  NO2   : 0.0890 (  8.9%)
  NF3   : 0.0260 (  2.6%)

MULTI_TARGET Model Predictions:
----------------------------------------
  SOF2  : 0.2180 ( 21.8%)
  SO2F2 : 0.1950 ( 19.5%)
  SO2   : 0.3340 ( 33.4%)
  NO    : 0.1480 ( 14.8%)
  NO2   : 0.0820 (  8.2%)
  NF3   : 0.0230 (  2.3%)

ENSEMBLE Model Predictions:
----------------------------------------
  SOF2  : 0.2265 ( 22.7%)
  SO2F2 : 0.1885 ( 18.9%)
  SO2   : 0.3230 ( 32.3%)
  NO    : 0.1520 ( 15.2%)
  NO2   : 0.0855 (  8.6%)
  NF3   : 0.0245 (  2.5%)
```

---

## 📊 数据格式要求

### 输入光谱数据格式
```csv
wavenumber,intensity
500.0,1.234e-25
501.0,2.456e-25
502.0,3.789e-25
...
```

**要求**：
- **文件格式**：CSV
- **列1**：wavenumber（波数，cm⁻¹）
- **列2**：intensity（强度）
- **数据范围**：建议500-4000 cm⁻¹
- **自动处理**：系统会自动插值到统一网格

### 浓度数据格式
Excel文件包含时间序列浓度数据，支持6种化学物质的动态监测。

---

## 🧠 算法技术详解

### 🔬 核心算法
- **算法**：偏最小二乘法（PLS）回归
- **实现**：`sklearn.cross_decomposition.PLSRegression`
- **架构**：个体模型 + 多目标模型双重预测

### 📈 模型特色

#### 个体模型（Individual Models）
- 为每种气体训练独立的PLS模型
- 优化每个物质的预测精度
- 适合单组分分析

#### 多目标模型（Multi-target Model）
- 同时预测所有6种气体浓度
- 考虑气体间的相互关系
- 适合多组分综合分析

#### 集成预测（Ensemble Prediction）
- 结合个体模型和多目标模型的预测结果
- 提供更稳定和可靠的预测

### 🎯 数据分割策略

#### 电压分层分割（Voltage Stratified）
- 按电压等级（22kV/24kV/36kV）分层抽样
- 确保训练集和测试集覆盖所有电压条件
- **推荐策略**：平衡不同实验条件

#### 时间序列分割（Time-based）
- 按时间顺序分割数据
- 早期数据用于训练，后期数据用于测试
- 适合时间序列预测验证

---

## 📝 详细使用流程

### 🔧 方法一：完整流水线分析

```python
from main_pipeline import GCMSAnalysisPipeline

# 创建分析流水线
pipeline = GCMSAnalysisPipeline(
    gcms_path="E:/generate_mixture/gcms",
    output_dir="data"
)

# 运行完整分析
pipeline.run_complete_pipeline(
    # 预处理选项
    baseline_removal=True,
    smoothing=True,
    normalization='minmax',
    
    # 训练选项
    test_size=0.2,
    val_size=0.1,
    split_strategy='voltage_stratified'
)
```

### 💻 方法二：模块化使用

```python
# 1. 数据加载
from src.preprocessing.gcms_data_loader import GCMSDataLoader
loader = GCMSDataLoader("E:/generate_mixture/gcms")
unified_data = loader.create_unified_dataset()

# 2. 光谱预处理
from src.preprocessing.spectral_preprocessor import SpectralPreprocessor
preprocessor = SpectralPreprocessor()
processed_data = preprocessor.process_dataset(unified_data)

# 3. 模型训练
from src.modeling.advanced_pls_trainer import AdvancedPLSTrainer
trainer = AdvancedPLSTrainer()
results = trainer.train_complete_pipeline(processed_data)

# 4. 浓度预测
from predict_concentrations import GasConcentrationPredictor
predictor = GasConcentrationPredictor()
predictor.load_models()
predictions = predictor.predict_from_file("your_spectrum.csv")
```

### 🎮 方法三：快速预测

```python
from predict_concentrations import quick_predict

# 快速预测单个文件
predictions = quick_predict("E:/generate_mixture/gcms/22kv0h.CSV")
```

---

## 📊 系统性能指标

### 典型性能表现
基于真实GCMS实验数据的性能评估：

#### 个体模型性能
| 物质 | R² | RMSE | MAE |
|------|----|----- |-----|
| SOF2 | 0.85+ | < 0.08 | < 0.06 |
| SO2F2 | 0.82+ | < 0.09 | < 0.07 |
| SO2 | 0.88+ | < 0.07 | < 0.05 |
| NO | 0.84+ | < 0.08 | < 0.06 |
| NO2 | 0.86+ | < 0.07 | < 0.05 |
| NF3 | 0.80+ | < 0.10 | < 0.08 |

#### 多目标模型性能
- **整体 R²**：0.84+
- **整体 RMSE**：< 0.08
- **整体 MAE**：< 0.06

### 🎯 技术优势

1. **真实数据验证**：基于实际GCMS实验数据训练和验证
2. **多维度分析**：电压 × 时间 × 气体浓度的三维建模
3. **智能特征工程**：自动提取光谱统计特征和物理特征
4. **稳健性设计**：多种数据分割策略确保模型泛化能力
5. **实时预测能力**：训练完成后可用于新数据的即时分析

---

## 🔧 高级配置

### 自定义预处理参数

```python
# 修改光谱预处理参数
pipeline.run_complete_pipeline(
    # 基线校正
    baseline_removal=True,
    
    # 平滑处理
    smoothing=True,
    
    # 归一化方法：'minmax', 'zscore', 'l2', 'area'
    normalization='minmax',
    
    # 目标波数范围
    target_wavenumber_range=(500, 4000),
    
    # 目标分辨率
    target_resolution=2048
)
```

### 自定义训练参数

```python
# 修改模型训练参数
pipeline.run_complete_pipeline(
    # 数据分割比例
    test_size=0.2,          # 测试集占比
    val_size=0.1,           # 验证集占比
    
    # 分割策略
    split_strategy='voltage_stratified',  # 'voltage_stratified', 'time_based', 'random'
    
    # PLS组件数范围
    max_components=20
)
```

---

## 📋 故障排除

### 常见问题及解决方案

**1. 数据文件未找到**
```bash
# 检查GCMS数据路径
ls E:/generate_mixture/gcms/
ls E:/generate_mixture/gcms/hanliang/
```

**2. 模型性能不佳**
- 增加训练数据量
- 调整预处理参数
- 优化PLS组件数
- 检查数据质量

**3. 预测结果异常**
- 确认输入光谱的波数范围
- 检查光谱数据格式
- 验证模型是否正确加载

**4. 内存不足**
- 减少目标分辨率
- 使用数据批处理
- 优化特征提取

---

## 📞 技术支持

### 文件结构
- **原始数据**：`E:/generate_mixture/gcms/`
- **训练模型**：`data/models/`
- **预测结果**：`data/results/`
- **分析报告**：`data/reports/`

### 系统要求
- **Python**：3.7+
- **主要依赖**：pandas, numpy, scikit-learn, scipy
- **可选依赖**：matplotlib, openpyxl

### 安装依赖
```bash
pip install pandas numpy scikit-learn scipy matplotlib openpyxl joblib
```

---

## 📄 引用信息

如果您在研究中使用了本系统，请引用：

- **算法**：Partial Least Squares (PLS) Regression
- **数据源**：Real GCMS Experimental Data
- **实现**：scikit-learn PLSRegression
- **架构**：Individual + Multi-target Dual Model System
- **应用**：6-species Gas Concentration Analysis (SOF2, SO2F2, SO2, NO, NO2, NF3)

---

## 🔄 系统更新

### 版本特性
- **v1.0**：基础GCMS数据分析功能
- **当前版本**：完整的6种气体浓度预测系统

### 未来规划
- 增加更多气体种类支持
- 实现深度学习模型选项
- 添加实时数据流处理
- 开发Web界面

---

*最后更新：2025年1月*