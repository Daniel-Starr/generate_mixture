# predict_with_standard.py
# 使用STANDARD模型进行气体浓度预测

import pandas as pd
import numpy as np
import joblib
import os
import json

def load_standard_model():
    """加载标准模型"""
    print("Loading STANDARD model...")
    
    model_dir = "data/models/standard"
    
    # 检查模型文件
    model_path = os.path.join(model_dir, "standard_pls_model.pkl")
    scaler_X_path = os.path.join(model_dir, "standard_scaler_X.pkl")
    scaler_Y_path = os.path.join(model_dir, "standard_scaler_Y.pkl")
    
    if not all(os.path.exists(p) for p in [model_path, scaler_X_path, scaler_Y_path]):
        raise FileNotFoundError("STANDARD model files not found. Please run build_standard_model.py first.")
    
    # 加载模型和标准化器
    model = joblib.load(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_Y = joblib.load(scaler_Y_path)
    
    print(f"  Model components: {model.n_components}")
    print(f"  Input features: {model.n_features_in_}")
    print("  Standard model loaded successfully!")
    
    return model, scaler_X, scaler_Y

def load_model_info():
    """加载模型信息"""
    info_path = "data/results/standard/model_performance.json"
    
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        return info
    else:
        return {"model_name": "standard", "info": "Model info not available"}

def predict_concentrations(spectral_data, model, scaler_X, scaler_Y):
    """预测气体浓度"""
    print("Predicting gas concentrations...")
    
    # 确保输入是2D数组
    if spectral_data.ndim == 1:
        spectral_data = spectral_data.reshape(1, -1)
    
    # 检查特征维度
    expected_features = model.n_features_in_
    actual_features = spectral_data.shape[1]
    
    if actual_features != expected_features:
        print(f"  WARNING: Feature dimension mismatch!")
        print(f"    Expected: {expected_features}, Got: {actual_features}")
        
        if actual_features > expected_features:
            print(f"    Truncating to first {expected_features} features")
            spectral_data = spectral_data[:, :expected_features]
        else:
            print(f"    Padding with zeros to {expected_features} features")
            padding = np.zeros((spectral_data.shape[0], expected_features - actual_features))
            spectral_data = np.hstack([spectral_data, padding])
    
    # 标准化输入
    X_scaled = scaler_X.transform(spectral_data)
    
    # 预测
    Y_pred_scaled = model.predict(X_scaled)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    
    # 确保浓度为正且和为1
    Y_pred = np.maximum(Y_pred, 0)
    Y_pred = Y_pred / Y_pred.sum(axis=1, keepdims=True)
    
    return Y_pred

def display_results(predictions, sample_names=None):
    """显示预测结果"""
    gas_names = ['NO', 'NO2', 'SO2']
    
    print("\n" + "="*60)
    print("STANDARD MODEL PREDICTION RESULTS")
    print("="*60)
    
    for i, pred in enumerate(predictions):
        if sample_names:
            print(f"\n{sample_names[i]}:")
        else:
            print(f"\nSample {i+1}:")
        
        for j, gas in enumerate(gas_names):
            conc = pred[j]
            print(f"  {gas}: {conc:.4f} ({conc*100:.1f}%)")
        
        total = pred.sum()
        print(f"  Total: {total:.4f}")

def predict_from_file(file_path):
    """从文件预测"""
    print(f"Loading spectral data from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # 读取数据
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
        
        # 检查是否有多个样本
        if 'wavenumber' in data.columns or 'frequency' in data.columns:
            # 假设这是单个光谱，波数在第一列，强度在第二列
            if len(data.columns) >= 2:
                spectral_data = data.iloc[:, 1].values.reshape(1, -1)
            else:
                raise ValueError("File format not recognized")
        else:
            # 假设每行是一个样本
            spectral_data = data.values
    else:
        raise ValueError("Only CSV files are supported")
    
    print(f"  Data shape: {spectral_data.shape}")
    
    return spectral_data

def main():
    """主函数"""
    print("STANDARD GAS CONCENTRATION PREDICTOR")
    print("="*50)
    print("Based on HITRAN standard gas database")
    print("Predicts: NO, NO2, SO2 concentrations")
    print("="*50)
    
    try:
        # 加载标准模型
        model, scaler_X, scaler_Y = load_standard_model()
        model_info = load_model_info()
        
        # 显示模型信息
        print(f"\nModel Performance:")
        if 'test_r2' in model_info:
            print(f"  R²: {model_info['test_r2']:.5f}")
            print(f"  RMSE: {model_info['test_rmse']:.5f}")
        
        # 检查是否有输入数据
        default_data_path = "data/raw/X_real.csv"
        
        if os.path.exists(default_data_path):
            print(f"\nFound default data file: {default_data_path}")
            spectral_data = predict_from_file(default_data_path)
            
            # 预测
            predictions = predict_concentrations(spectral_data, model, scaler_X, scaler_Y)
            
            # 显示结果
            display_results(predictions)
            
            # 保存结果
            results_df = pd.DataFrame(predictions, columns=['NO_conc', 'NO2_conc', 'SO2_conc'])
            
            os.makedirs("data/results/standard", exist_ok=True)
            output_path = "data/results/standard/predictions.csv"
            results_df.to_csv(output_path, index=False)
            
            print(f"\nResults saved to: {output_path}")
            
            # 如果有真实值，进行比较
            true_data_path = "data/raw/Y_real.csv"
            if os.path.exists(true_data_path):
                print("\nComparing with true values...")
                Y_true = pd.read_csv(true_data_path).values
                
                if Y_true.shape == predictions.shape:
                    gas_names = ['NO', 'NO2', 'SO2']
                    print("\nCOMPARISON:")
                    for i in range(len(predictions)):
                        print(f"\nSample {i+1}:")
                        for j, gas in enumerate(gas_names):
                            true_val = Y_true[i, j]
                            pred_val = predictions[i, j]
                            error = abs(pred_val - true_val)
                            print(f"  {gas}: True={true_val:.1%}, Pred={pred_val:.1%}, Error={error:.1%}")
        
        else:
            print(f"\nNo default data found at: {default_data_path}")
            print("To use the STANDARD model:")
            print("1. Place your spectral data as 'data/raw/X_real.csv'")
            print("2. Run this script again")
            print("3. Or use: python predict_with_standard.py your_data.csv")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 如果提供了文件路径
        file_path = sys.argv[1]
        
        try:
            model, scaler_X, scaler_Y = load_standard_model()
            spectral_data = predict_from_file(file_path)
            predictions = predict_concentrations(spectral_data, model, scaler_X, scaler_Y)
            display_results(predictions)
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
    else:
        main()