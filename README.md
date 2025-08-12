# Gas Spectroscopy Analysis Project

A comprehensive gas mixture analysis system using spectroscopy data and machine learning for concentration prediction.

## Project Overview

This project implements advanced gas spectroscopy analysis tools for detecting and quantifying gas concentrations using:
- FTIR spectroscopy data processing
- Machine learning models (PLS regression)
- GCMS data analysis
- Voigt profile fitting from HITRAN database

## Project Structure

```
├── gas_two/           # Initial gas analysis implementation
├── gas_three/         # Advanced 3-component gas analysis
├── gas_reality/       # Real experimental data processing
├── gcms/              # Gas Chromatography-Mass Spectrometry data
├── gas_hdf5/          # HDF5 spectral databases
├── output_hdf5/       # Processed spectral outputs
└── batch_voigt_from_hitran.py  # HITRAN data processing
```

## Key Features

### 🔬 **Spectral Analysis**
- FTIR spectrum preprocessing and normalization
- Baseline correction and noise reduction
- Peak detection and analysis

### 🤖 **Machine Learning Models**
- Partial Least Squares (PLS) regression
- Cross-validation and model evaluation
- Individual and multi-target concentration prediction

### 📊 **Data Processing**
- GCMS experimental data integration
- HDF5 database management
- Voigt profile analysis from HITRAN

### 📈 **Visualization**
- Spectral plotting and comparison
- Model performance visualization
- Concentration prediction results

## Quick Start

### 1. Standard Model Usage
```python
# Build and use standard model
python gas_three/build_standard_model.py
python gas_three/predict_with_standard.py
```

### 2. Enhanced Pipeline
```python
# Run complete analysis pipeline
python gas_three/run_enhanced_pipeline.py
```

### 3. Real Data Processing
```python
# Process experimental GCMS data
python gas_reality/main_pipeline.py
```

## Supported Gas Types

- **NO₂** (Nitrogen Dioxide)
- **NO** (Nitric Oxide)  
- **SO₂** (Sulfur Dioxide)
- **CS₂** (Carbon Disulfide)
- **NF₃** (Nitrogen Trifluoride)
- **SO₂F₂** (Sulfuryl Fluoride)
- **SOF₂** (Thionyl Fluoride)

## Data Files

Large data files are managed using Git LFS:
- `*.csv` - Spectral and concentration data
- `*.pkl` - Trained machine learning models
- `*.npy` - NumPy arrays for efficient data storage
- `*.hdf5` - High-performance spectral databases

## Model Performance

The system achieves high accuracy in gas concentration prediction:
- Cross-validation R² > 0.95 for most components
- Detection limits in ppm range
- Real-time prediction capability

## Documentation

- **gas_three/README.md** - Detailed 3-component analysis guide
- **gas_three/USAGE_GUIDE.md** - Step-by-step usage instructions
- **gas_three/PROJECT_DOCUMENTATION.md** - Technical documentation
- **gas_reality/README.md** - Real data processing guide

## Requirements

- Python 3.7+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualization)
- H5py (for HDF5 file handling)

## Installation

```bash
git clone https://github.com/Daniel-Starr/generate_mixture.git
cd generate_mixture
pip install -r requirements.txt  # If requirements.txt exists
```

## Usage Examples

### Detect Gas Concentrations
```python
from gas_three.detect_spectrum import detect_gas_mixture
results = detect_gas_mixture('path/to/spectrum.csv')
```

### Train Custom Model
```python
from gas_three.enhanced_model_trainer import train_model
model = train_model(X_train, y_train)
```

## Contributing

This project is part of ongoing research in gas spectroscopy analysis. Contributions and improvements are welcome.

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:
```
[Add citation information if applicable]
```

---

**Note**: This project uses Git LFS for large data files. Ensure Git LFS is installed when cloning the repository.