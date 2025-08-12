# 气体光谱分析项目

基于光谱数据和机器学习的综合气体混合物分析系统，用于浓度预测。

## 项目概述

本项目实现了先进的气体光谱分析工具，用于检测和定量气体浓度，包括：
- FTIR光谱数据处理
- 机器学习模型（偏最小二乘回归）
- GCMS数据分析
- 基于HITRAN数据库的Voigt谱线拟合

## 项目结构

```
├── gas_two/           # 初始气体分析实现
├── gas_three/         # 高级三组分气体分析
├── gas_reality/       # 真实实验数据处理
├── gcms/              # 气相色谱-质谱数据
├── gas_hdf5/          # HDF5光谱数据库
├── output_hdf5/       # 处理后的光谱输出
└── batch_voigt_from_hitran.py  # HITRAN数据处理
```

## 主要功能

### 🔬 **光谱分析**
- FTIR光谱预处理和归一化
- 基线校正和噪声降低
- 峰检测和分析

### 🤖 **机器学习模型**
- 偏最小二乘（PLS）回归
- 交叉验证和模型评估
- 单个和多目标浓度预测

### 📊 **数据处理**
- GCMS实验数据集成
- HDF5数据库管理
- 基于HITRAN的Voigt谱线分析

### 📈 **可视化**
- 光谱绘制和比较
- 模型性能可视化
- 浓度预测结果展示

## 快速开始

### 1. 标准模型使用
```python
# 构建和使用标准模型
python gas_three/build_standard_model.py
python gas_three/predict_with_standard.py
```

### 2. 增强管道
```python
# 运行完整分析管道
python gas_three/run_enhanced_pipeline.py
```

### 3. 真实数据处理
```python
# 处理实验GCMS数据
python gas_reality/main_pipeline.py
```

## 支持的气体类型

- **NO₂** (二氧化氮)
- **NO** (一氧化氮)  
- **SO₂** (二氧化硫)
- **CS₂** (二硫化碳)
- **NF₃** (三氟化氮)
- **SO₂F₂** (硫酰氟)
- **SOF₂** (亚硫酰氟)

## 数据文件

大型数据文件使用Git LFS管理：
- `*.csv` - 光谱和浓度数据
- `*.pkl` - 训练好的机器学习模型
- `*.npy` - 用于高效数据存储的NumPy数组
- `*.hdf5` - 高性能光谱数据库

## 模型性能

系统在气体浓度预测方面达到高精度：
- 大多数组分的交叉验证R² > 0.95
- 检测限达到ppm级别
- 具备实时预测能力

## 文档说明

- **gas_three/README.md** - 详细的三组分分析指南
- **gas_three/USAGE_GUIDE.md** - 分步使用说明
- **gas_three/PROJECT_DOCUMENTATION.md** - 技术文档
- **gas_reality/README.md** - 真实数据处理指南

## 系统要求

- Python 3.7+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn（用于可视化）
- H5py（用于HDF5文件处理）

## 安装方法

```bash
git clone https://github.com/Daniel-Starr/generate_mixture.git
cd generate_mixture
pip install -r requirements.txt  # 如果存在requirements.txt文件
```

## 使用示例

### 检测气体浓度
```python
from gas_three.detect_spectrum import detect_gas_mixture
results = detect_gas_mixture('path/to/spectrum.csv')
```

### 训练自定义模型
```python
from gas_three.enhanced_model_trainer import train_model
model = train_model(X_train, y_train)
```

## 贡献

本项目是气体光谱分析持续研究的一部分。欢迎贡献和改进。

## 许可证

[在此添加许可证信息]

## 引用

如果您在研究中使用此代码，请引用：
```
[如适用，请添加引用信息]
```

---

**注意**：本项目使用Git LFS存储大型数据文件。克隆仓库时请确保已安装Git LFS。