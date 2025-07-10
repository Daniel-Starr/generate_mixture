# predict_real.py
# 加载已训练模型，预测真实样本光谱的气体浓度

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# 1. 读取模型和新数据
model_path = "trained_pls_model.pkl"
real_data_path = "X_real.csv"

if not os.path.exists(model_path):
    raise FileNotFoundError("模型文件未找到，请先运行 train_model.py 训练模型。")
if not os.path.exists(real_data_path):
    raise FileNotFoundError("未找到 X_real.csv，请提供真实样本光谱数据。")

pls = joblib.load(model_path)
X_real = pd.read_csv(real_data_path).values

# 2. 预测
Y_pred = pls.predict(X_real)

# 3. 保存结果
Y_pred_df = pd.DataFrame(Y_pred)
Y_pred_df.to_csv("Y_pred_real.csv", index=False)
print("预测完成，结果已保存为 Y_pred_real.csv")

# 4. 可视化结果（仅当真实标签存在时）
true_label_path = "Y_real.csv"
if os.path.exists(true_label_path):
    Y_real = pd.read_csv(true_label_path).values
    if Y_real.shape == Y_pred.shape:
        if Y_real.shape[1] == 1:
            plt.scatter(Y_real, Y_pred)
            plt.xlabel("True Concentration")
            plt.ylabel("Predicted Concentration")
            plt.title("PLS Prediction on Real Data")
            plt.plot([Y_real.min(), Y_real.max()], [Y_real.min(), Y_real.max()], 'r--')
            plt.savefig("pls_prediction_real.png")
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
                plt.savefig(f"pls_prediction_real_component{i+1}.png")
                plt.close()
        print("真实 vs. 预测图已保存。")
    else:
        print("注意：Y_real.csv 存在但维度与预测结果不一致，跳过绘图。")
else:
    print("未找到 Y_real.csv（真实浓度），跳过可视化。")
