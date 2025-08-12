# predict_real.py
# 加载已训练模型，预测真实样本光谱的气体浓度

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# 1. 读取增强模型和新数据
model_path = "data/models/enhanced_pls_model.pkl"
scaler_X_path = "data/models/scaler_X.pkl"
scaler_Y_path = "data/models/scaler_Y.pkl"
real_data_path = "data/raw/X_real.csv"

print("Enhanced Gas Concentration Prediction")
print("="*40)

if not os.path.exists(model_path):
    print("ERROR: Enhanced model not found!")
    print("Please run: python how_to_use.py")
    raise FileNotFoundError(f"Model file not found: {model_path}")

if not os.path.exists(real_data_path):
    print("ERROR: Real data file not found!")
    print(f"Please provide your spectral data as: {real_data_path}")
    print("Format: CSV file with spectral intensities")
    raise FileNotFoundError(f"Real data file not found: {real_data_path}")

# 加载增强模型和标准化器
pls = joblib.load(model_path)
print(f"Loaded enhanced PLS model with {pls.n_components} components")

use_scaling = os.path.exists(scaler_X_path) and os.path.exists(scaler_Y_path)
if use_scaling:
    scaler_X = joblib.load(scaler_X_path)
    scaler_Y = joblib.load(scaler_Y_path)
    print("Loaded data standardization scalers")

X_real = pd.read_csv(real_data_path).values
print(f"Loaded real data: {X_real.shape[0]} samples, {X_real.shape[1]} features")

# 2. 预测（使用增强模型）
print("\nPredicting gas concentrations...")

# 检查特征维度匹配
expected_features = pls.n_features_in_
actual_features = X_real.shape[1]

if actual_features != expected_features:
    print(f"WARNING: Feature dimension mismatch!")
    print(f"  Expected: {expected_features}, Got: {actual_features}")
    
    if actual_features > expected_features:
        print(f"  Truncating to first {expected_features} features")
        X_real = X_real[:, :expected_features]
    else:
        print(f"  Padding with zeros to {expected_features} features")
        padding = np.zeros((X_real.shape[0], expected_features - actual_features))
        X_real = np.hstack([X_real, padding])

# 应用标准化（如果有）
if use_scaling:
    X_real_scaled = scaler_X.transform(X_real)
    Y_pred_scaled = pls.predict(X_real_scaled)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    print("Applied data standardization")
else:
    Y_pred = pls.predict(X_real)
    print("No standardization applied")

# 3. 验证和修正预测结果
print("\nProcessing prediction results...")

for i, sample_pred in enumerate(Y_pred):
    original_sum = sample_pred.sum()
    
    # 将负值置为0
    sample_pred_positive = np.maximum(sample_pred, 0)
    
    # 归一化确保总和为1
    if sample_pred_positive.sum() > 0:
        sample_pred_normalized = sample_pred_positive / sample_pred_positive.sum()
    else:
        # 如果所有预测都为负，使用等比例分配
        sample_pred_normalized = np.ones(len(sample_pred)) / len(sample_pred)
    
    Y_pred[i] = sample_pred_normalized
    
    normalized_sum = sample_pred_normalized.sum()
    print(f"Sample {i+1}: Original sum={original_sum:.3f} -> Normalized sum={normalized_sum:.3f}")

# 4. 显示最终结果
print("\n" + "="*50)
print("FINAL PREDICTION RESULTS")
print("="*50)

gas_names = ['NO', 'NO2', 'SO2']
for i, sample_pred in enumerate(Y_pred):
    print(f"\nSample {i+1}:")
    for j, gas in enumerate(gas_names):
        conc = sample_pred[j]
        print(f"  {gas}: {conc:.4f} ({conc*100:.1f}%)")

# 5. 保存结果
os.makedirs("data/results", exist_ok=True)
Y_pred_df = pd.DataFrame(Y_pred, columns=['NO_conc', 'NO2_conc', 'SO2_conc'])
Y_pred_df.to_csv("data/results/Y_pred_real.csv", index=False)
print(f"\nResults saved to: data/results/Y_pred_real.csv")

# 4. 可视化结果（仅当真实标签存在时）
true_label_path = "data/raw/Y_real.csv"
if os.path.exists(true_label_path):
    Y_real = pd.read_csv(true_label_path).values
    if Y_real.shape == Y_pred.shape:
        if Y_real.shape[1] == 1:
            plt.scatter(Y_real, Y_pred)
            plt.xlabel("True Concentration")
            plt.ylabel("Predicted Concentration")
            plt.title("PLS Prediction on Real Data")
            plt.plot([Y_real.min(), Y_real.max()], [Y_real.min(), Y_real.max()], 'r--')
            plt.savefig("data/figures/pls_prediction_real.png")
            plt.close()
        else:
            for i in range(Y_real.shape[1]):
                plt.figure()
                plt.scatter(Y_real[:, i], Y_pred[:, i])
                plt.xlabel(f"True Component {i+1}")
                plt.ylabel(f"Predicted Component {i+1}")
                plt.title(f"Real Prediction - Component {i+1}")
                plt.plot([Y_real[:, i].min(), Y_real[:, i].max()],
                         [Y_real[:, i].min(), Y_real[:, i].max()], 'r--')
                plt.savefig(f"data/figures/pls_prediction_real_component{i+1}.png")
                plt.close()
        print("真实 vs. 预测图已保存。")
    else:
        print("注意：data/raw/Y_real.csv 存在但维度与预测结果不一致，跳过绘图。")
else:
    print("未找到 data/raw/Y_real.csv（真实浓度），跳过可视化。")
