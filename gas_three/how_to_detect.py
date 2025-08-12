# how_to_detect.py
# 简单的光谱检测使用指南

import pandas as pd
import numpy as np
import joblib
import os
from scipy.interpolate import interp1d

def quick_detect(file_path):
    """快速检测光谱文件中的气体浓度"""
    
    print("QUICK SPECTRUM DETECTION")
    print("=" * 40)
    print(f"File: {file_path}")
    
    if not os.path.exists(file_path):
        print("ERROR: File not found!")
        return None
    
    # 1. 读取光谱数据
    try:
        df = pd.read_csv(file_path)
        
        # 自动识别列
        if 'wavenumber' in df.columns and 'intensity' in df.columns:
            wavenumbers = df['wavenumber'].values
            intensities = df['intensity'].values
        elif len(df.columns) >= 2:
            wavenumbers = df.iloc[:, 0].values
            intensities = df.iloc[:, 1].values
        else:
            print("ERROR: Cannot identify wavenumber and intensity columns")
            return None
            
        print(f"Loaded: {len(wavenumbers)} data points")
        
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return None
    
    # 2. 预处理数据
    try:
        # 加载参考波数轴
        ref_file = "data/processed/standard_interpolated_spectra.csv"
        if os.path.exists(ref_file):
            ref_df = pd.read_csv(ref_file)
            ref_wavenumbers = ref_df['wavenumber'].values
            
            # 插值到参考轴
            interpolator = interp1d(wavenumbers, intensities, 
                                  kind='linear', bounds_error=False, fill_value=0)
            X_test = interpolator(ref_wavenumbers).reshape(1, -1)
        else:
            X_test = intensities.reshape(1, -1)
            
    except Exception as e:
        print(f"ERROR in preprocessing: {e}")
        return None
    
    # 3. 使用模型预测
    results = {}
    gas_names = ['NO', 'NO2', 'SO2']
    
    # 尝试Standard模型
    try:
        model = joblib.load("data/models/standard/standard_pls_model.pkl")
        scaler_X = joblib.load("data/models/standard/standard_scaler_X.pkl")
        scaler_Y = joblib.load("data/models/standard/standard_scaler_Y.pkl")
        
        # 调整维度
        expected = model.n_features_in_
        actual = X_test.shape[1]
        
        if actual == expected:
            X_adjusted = X_test
        elif actual > expected:
            X_adjusted = X_test[:, :expected]
        else:
            X_adjusted = np.hstack([X_test, np.zeros((1, expected - actual))])
        
        # 预测
        X_scaled = scaler_X.transform(X_adjusted)
        Y_pred_scaled = model.predict(X_scaled)
        Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
        Y_pred = np.maximum(Y_pred, 0)
        Y_pred = Y_pred / Y_pred.sum()
        
        results['Standard'] = Y_pred[0]
        
    except:
        pass
    
    # 尝试Enhanced模型
    try:
        model = joblib.load("data/models/enhanced_pls_model.pkl")
        scaler_X = joblib.load("data/models/scaler_X.pkl")
        scaler_Y = joblib.load("data/models/scaler_Y.pkl")
        
        # 调整维度
        expected = model.n_features_in_
        actual = X_test.shape[1]
        
        if actual == expected:
            X_adjusted = X_test
        elif actual > expected:
            X_adjusted = X_test[:, :expected]
        else:
            X_adjusted = np.hstack([X_test, np.zeros((1, expected - actual))])
        
        # 预测
        X_scaled = scaler_X.transform(X_adjusted)
        Y_pred_scaled = model.predict(X_scaled)
        Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
        Y_pred = np.maximum(Y_pred, 0)
        Y_pred = Y_pred / Y_pred.sum()
        
        results['Enhanced'] = Y_pred[0]
        
    except:
        pass
    
    # 4. 显示结果
    if not results:
        print("ERROR: No models available for prediction")
        return None
    
    print()
    print("DETECTION RESULTS:")
    print("-" * 30)
    
    for model_name, prediction in results.items():
        print(f"\n{model_name} Model:")
        for i, gas in enumerate(gas_names):
            conc = prediction[i]
            print(f"  {gas}: {conc:.3f} ({conc*100:.1f}%)")
    
    # 如果有多个模型，显示平均值
    if len(results) > 1:
        avg_pred = np.mean(list(results.values()), axis=0)
        print(f"\nConsensus (Average):")
        for i, gas in enumerate(gas_names):
            conc = avg_pred[i]
            print(f"  {gas}: {conc:.3f} ({conc*100:.1f}%)")
    
    return results

def main():
    """主函数 - 使用示例"""
    
    print("GAS CONCENTRATION DETECTOR")
    print("=" * 50)
    print("Detects NO, NO2, SO2 concentrations from spectral data")
    print()
    
    # 示例用法
    print("USAGE EXAMPLES:")
    print("1. Detect specific file:")
    print("   python how_to_detect.py your_spectrum.csv")
    print()
    print("2. Use in Python:")
    print("   from how_to_detect import quick_detect")
    print("   result = quick_detect('your_file.csv')")
    print()
    
    # 如果提供了命令行参数
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        quick_detect(file_path)
    else:
        # 默认检测示例文件
        example_file = "test/mixed_spectrum_244_clean_20250725_151622.csv"
        if os.path.exists(example_file):
            print(f"Running example detection on: {example_file}")
            print()
            quick_detect(example_file)
        else:
            print("No file specified and no example file found.")
            print("Usage: python how_to_detect.py your_spectrum_file.csv")

if __name__ == "__main__":
    main()