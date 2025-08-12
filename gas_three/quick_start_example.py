# quick_start_example.py
# 快速开始示例 - 演示如何使用增强版系统

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def create_sample_data():
    """创建示例数据来演示系统功能"""
    print("Creating sample spectral data for demonstration...")
    
    # 模拟一个真实光谱数据样本
    # 使用与训练数据相同的波数轴
    if os.path.exists("data/processed/interpolated_spectra.csv"):
        ref_df = pd.read_csv("data/processed/interpolated_spectra.csv")
        wavenumbers = ref_df['wavenumber'].values
        
        # 模拟一个混合光谱 (NO:30%, NO2:40%, SO2:30%)
        no_spectrum = ref_df['NO'].values
        no2_spectrum = ref_df['NO2'].values  
        so2_spectrum = ref_df['SO2'].values
        
        # 创建混合光谱并添加一些噪声
        np.random.seed(123)
        mixed_spectrum = (0.3 * no_spectrum + 
                         0.4 * no2_spectrum + 
                         0.3 * so2_spectrum)
        
        # 添加现实的噪声
        noise = np.random.normal(0, 0.05 * mixed_spectrum.std(), len(mixed_spectrum))
        noisy_spectrum = mixed_spectrum + noise
        noisy_spectrum = np.maximum(noisy_spectrum, 0)  # 确保非负
        
        # 保存为模拟的"真实"数据
        os.makedirs("data/raw", exist_ok=True)
        sample_df = pd.DataFrame([noisy_spectrum])
        sample_df.to_csv("data/raw/X_real.csv", index=False)
        
        # 保存真实的浓度（用于验证）
        true_concentrations = pd.DataFrame([[0.3, 0.4, 0.3]], 
                                         columns=['NO_conc', 'NO2_conc', 'SO2_conc'])
        true_concentrations.to_csv("data/raw/Y_real.csv", index=False)
        
        print("Sample data created:")
        print(f"  - True concentrations: NO=30%, NO2=40%, SO2=30%")
        print(f"  - Spectral data saved to: data/raw/X_real.csv")
        print(f"  - True labels saved to: data/raw/Y_real.csv")
        
        return True
    else:
        print("ERROR: Reference spectra not found. Please run system setup first.")
        return False

def demonstrate_prediction():
    """演示预测功能"""
    print("\n" + "="*60)
    print("DEMONSTRATION: Predicting Gas Concentrations")
    print("="*60)
    
    try:
        # 方法1: 使用更新后的predict_real.py
        print("\nMethod 1: Using updated predict_real.py")
        exec(open("predict_real.py").read())
        
    except Exception as e:
        print(f"Method 1 failed: {str(e)}")
        
        # 方法2: 直接调用模型
        print("\nMethod 2: Direct model call")
        try:
            import joblib
            from sklearn.preprocessing import StandardScaler
            
            # 加载模型
            model = joblib.load("data/models/enhanced_pls_model.pkl")
            scaler_X = joblib.load("data/models/scaler_X.pkl")
            scaler_Y = joblib.load("data/models/scaler_Y.pkl")
            
            # 加载数据
            X_real = pd.read_csv("data/raw/X_real.csv").values
            
            # 预测
            X_scaled = scaler_X.transform(X_real)
            Y_pred_scaled = model.predict(X_scaled)
            Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
            
            # 归一化
            Y_pred = np.maximum(Y_pred, 0)
            Y_pred = Y_pred / Y_pred.sum(axis=1, keepdims=True)
            
            # 显示结果
            print("\nPREDICTION RESULTS:")
            gas_names = ['NO', 'NO2', 'SO2']
            for i, gas in enumerate(gas_names):
                predicted = Y_pred[0, i]
                print(f"  {gas}: {predicted:.4f} ({predicted*100:.1f}%)")
            
            # 如果有真实值，显示比较
            if os.path.exists("data/raw/Y_real.csv"):
                Y_true = pd.read_csv("data/raw/Y_real.csv").values
                print("\nCOMPARISON WITH TRUE VALUES:")
                for i, gas in enumerate(gas_names):
                    true_val = Y_true[0, i]
                    pred_val = Y_pred[0, i]
                    error = abs(pred_val - true_val)
                    print(f"  {gas}: True={true_val:.1%}, Predicted={pred_val:.1%}, Error={error:.1%}")
                    
        except Exception as e2:
            print(f"Method 2 also failed: {str(e2)}")

def demonstrate_real_data_processor():
    """演示真实数据处理器"""
    print("\n" + "="*60)
    print("DEMONSTRATION: Real Data Processor")  
    print("="*60)
    
    try:
        from real_data_processor import process_real_data_pipeline
        
        # 创建一个CSV格式的光谱文件来演示
        if os.path.exists("data/processed/interpolated_spectra.csv"):
            ref_df = pd.read_csv("data/processed/interpolated_spectra.csv")
            
            # 创建示例光谱文件
            sample_spectrum = pd.DataFrame({
                'wavenumber': ref_df['wavenumber'].values,
                'intensity': 0.3 * ref_df['NO'].values + 
                           0.4 * ref_df['NO2'].values + 
                           0.3 * ref_df['SO2'].values +
                           np.random.normal(0, 1e-23, len(ref_df))
            })
            
            sample_file = "demo_spectrum.csv"
            sample_spectrum.to_csv(sample_file, index=False)
            print(f"Created demo spectrum file: {sample_file}")
            
            # 处理数据
            result = process_real_data_pipeline(
                sample_file, 
                wavenumber_col='wavenumber',
                intensity_col='intensity',
                predict=True
            )
            
            if result and result['predictions']:
                print("✅ Real data processor demonstration successful!")
            
            # 清理
            if os.path.exists(sample_file):
                os.remove(sample_file)
                
    except Exception as e:
        print(f"Real data processor demonstration failed: {str(e)}")

def main():
    print("Enhanced Gas Spectroscopy System - Quick Start Example")
    print("="*60)
    
    # 检查系统状态
    required_files = [
        "data/models/enhanced_pls_model.pkl",
        "data/models/scaler_X.pkl", 
        "data/models/scaler_Y.pkl"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ System not ready. Missing files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease run: python how_to_use.py")
        return
    
    print("✅ System is ready!")
    
    # 创建示例数据
    if create_sample_data():
        # 演示预测
        demonstrate_prediction()
        
        # 演示真实数据处理器  
        demonstrate_real_data_processor()
        
        print("\n" + "="*60)
        print("QUICK START COMPLETE!")
        print("="*60)
        print("Now you can:")
        print("1. Replace data/raw/X_real.csv with your actual spectral data")
        print("2. Run: python predict_real.py")
        print("3. Or use: python how_to_use.py your_data.csv")
        print("4. Or use the real_data_processor in your Python scripts")
        
    else:
        print("❌ Quick start failed. Please check system setup.")

if __name__ == "__main__":
    main()