# check_status.py
# 检查系统状态

import os
import pandas as pd
import json

def check_status():
    print("="*60)
    print("ENHANCED GAS SPECTROSCOPY SYSTEM STATUS")
    print("="*60)
    
    # 1. 检查核心文件
    print("\n1. CORE FILES STATUS:")
    core_files = [
        ("Base spectra", "data/processed/interpolated_spectra.csv"),
        ("Enhanced dataset", "data/processed/X_dataset.csv"),
        ("Training data", "data/processed/X_train.csv"),
        ("Enhanced model", "data/models/enhanced_pls_model.pkl"),
        ("Data scalers", "data/models/scaler_X.pkl"),
    ]
    
    all_ready = True
    for name, path in core_files:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"   {name}: {status}")
        if not exists:
            all_ready = False
    
    # 2. 检查数据集统计
    if os.path.exists("data/processed/generation_params.json"):
        print("\n2. DATASET STATISTICS:")
        with open("data/processed/generation_params.json", "r") as f:
            params = json.load(f)
        
        print(f"   Total samples: {params.get('total_samples', 'N/A')}")
        print(f"   Unique combinations: {params.get('unique_combinations', 'N/A')}")
        print(f"   Base noise level: {params.get('base_noise_level', 'N/A')}")
        print(f"   Nonlinear strength: {params.get('nonlinear_strength', 'N/A')}")
    
    # 3. 检查模型性能
    if os.path.exists("data/results/enhanced_results.json"):
        print("\n3. MODEL PERFORMANCE:")
        with open("data/results/enhanced_results.json", "r") as f:
            results = json.load(f)
        
        print(f"   Components: {results.get('best_n_components', 'N/A')}")
        print(f"   Training R2: {results.get('train_r2', 'N/A'):.5f}")
        print(f"   Test R2: {results.get('test_r2', 'N/A'):.5f}")
        print(f"   Test RMSE: {results.get('test_rmse', 'N/A'):.5f}")
    
    # 4. 检查数据分割
    split_files = ["X_train.csv", "X_val.csv", "X_test.csv"]
    if all(os.path.exists(f"data/processed/{f}") for f in split_files):
        print("\n4. DATA SPLIT:")
        for split_name in ["train", "val", "test"]:
            df = pd.read_csv(f"data/processed/X_{split_name}.csv")
            print(f"   {split_name.capitalize()}: {len(df)} samples")
    
    # 5. 系统就绪状态
    print("\n5. SYSTEM STATUS:")
    if all_ready:
        print("   Status: READY")
        print("   You can now:")
        print("     - Process real data: python how_to_use.py your_data.csv")
        print("     - Direct prediction: python predict_real.py")
        print("     - Use Python API: from real_data_processor import process_real_data_pipeline")
    else:
        print("   Status: NOT READY")
        print("   Please run: python how_to_use.py")
    
    # 6. 检查示例数据
    print("\n6. SAMPLE DATA:")
    if os.path.exists("data/raw/X_real.csv"):
        real_data = pd.read_csv("data/raw/X_real.csv")
        print(f"   Sample real data: {real_data.shape[0]} samples, {real_data.shape[1]} features")
        
        if os.path.exists("data/raw/Y_real.csv"):
            true_labels = pd.read_csv("data/raw/Y_real.csv")
            print(f"   True concentrations available: {len(true_labels)} samples")
    else:
        print("   No sample data found")
    
    print("\n" + "="*60)
    print("STATUS CHECK COMPLETE")
    print("="*60)

if __name__ == "__main__":
    check_status()