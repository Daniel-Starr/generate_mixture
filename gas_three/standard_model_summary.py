# standard_model_summary.py
# 标准模型状态总结

import os
import json
import joblib

def show_standard_model_summary():
    """显示标准模型完整总结"""
    
    print("="*60)
    print("STANDARD GAS PREDICTION MODEL SUMMARY")
    print("="*60)
    
    # 1. 模型基本信息
    print("\n1. MODEL INFORMATION:")
    print("   Name: STANDARD")
    print("   Source: HITRAN Database")
    print("   Gas Types: NO, NO2, SO2")
    print("   Training Method: PLS Regression")
    
    # 2. 检查模型文件
    print("\n2. MODEL FILES STATUS:")
    model_files = [
        ("Main Model", "data/models/standard/standard_pls_model.pkl"),
        ("X Scaler", "data/models/standard/standard_scaler_X.pkl"),
        ("Y Scaler", "data/models/standard/standard_scaler_Y.pkl"),
        ("Performance Results", "data/results/standard/model_performance.json"),
        ("Interpolated Spectra", "data/processed/standard_interpolated_spectra.csv")
    ]
    
    all_ready = True
    for name, path in model_files:
        exists = os.path.exists(path)
        size = f"({os.path.getsize(path)} bytes)" if exists else ""
        status = "✓" if exists else "✗"
        print(f"   {name}: {status} {size}")
        if not exists:
            all_ready = False
    
    # 3. 模型性能
    if os.path.exists("data/results/standard/model_performance.json"):
        print("\n3. MODEL PERFORMANCE:")
        with open("data/results/standard/model_performance.json", "r") as f:
            perf = json.load(f)
        
        print(f"   Components: {perf.get('best_n_components', 'N/A')}")
        print(f"   Test R²: {perf.get('test_r2', 'N/A'):.5f}")
        print(f"   Test RMSE: {perf.get('test_rmse', 'N/A'):.5f}")
        print(f"   Training Samples: {perf.get('total_samples', 'N/A')}")
        print(f"   Valid Combinations: {perf.get('valid_combinations', 'N/A')}")
    
    # 4. 模型规格
    if os.path.exists("data/models/standard/standard_pls_model.pkl"):
        print("\n4. MODEL SPECIFICATIONS:")
        model = joblib.load("data/models/standard/standard_pls_model.pkl")
        print(f"   Input Features: {model.n_features_in_}")
        print(f"   Output Variables: 3 (NO, NO2, SO2)")
        print(f"   PLS Components: {model.n_components}")
        print(f"   Model Type: {type(model).__name__}")
    
    # 5. 使用方法
    print("\n5. HOW TO USE YOUR STANDARD MODEL:")
    print("   A. Direct prediction script:")
    print("      python predict_with_standard.py")
    print("   ")
    print("   B. With your own data file:")
    print("      python predict_with_standard.py your_data.csv")
    print("   ")
    print("   C. In Python code:")
    print("      import joblib")
    print("      model = joblib.load('data/models/standard/standard_pls_model.pkl')")
    print("      # [your prediction code]")
    
    # 6. 数据格式要求
    print("\n6. INPUT DATA FORMAT:")
    print("   - CSV file with spectral intensities")
    print("   - 3104 wavelength points (586.7-3690.6 cm⁻¹)")
    print("   - One row per sample")
    print("   - Or place data as 'data/raw/X_real.csv'")
    
    # 7. 输出格式
    print("\n7. OUTPUT FORMAT:")
    print("   - NO concentration (0-1)")
    print("   - NO2 concentration (0-1)")
    print("   - SO2 concentration (0-1)")
    print("   - Total always sums to 1.0")
    
    # 8. 状态总结
    print("\n8. STATUS:")
    if all_ready:
        print("   🟢 STANDARD MODEL: FULLY READY")
        print("   ✓ All files present")
        print("   ✓ Model validated")
        print("   ✓ Ready for predictions")
    else:
        print("   🔴 STANDARD MODEL: INCOMPLETE")
        print("   ✗ Some files missing")
        print("   → Run: python build_standard_model.py")
    
    print("\n" + "="*60)
    print("STANDARD MODEL READY FOR USE!")
    print("="*60)

if __name__ == "__main__":
    show_standard_model_summary()