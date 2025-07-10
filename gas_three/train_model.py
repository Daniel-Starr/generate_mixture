# train_model.py
# 用仿真数据训练 PLS 模型并保存

import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

# 1. 读取仿真数据
X = pd.read_csv("X_dataset.csv").values
Y = pd.read_csv("Y_labels.csv").values

# 2. 拆分训练/测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 3. 训练模型
pls = PLSRegression(n_components=10)
pls.fit(X_train, Y_train)

# 4. 模型评估
Y_pred = pls.predict(X_test)
print("MSE:", mean_squared_error(Y_test, Y_pred))
print("R^2:", r2_score(Y_test, Y_pred))

# 5. 可视化（支持单目标和多目标）
if Y.shape[1] == 1:
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("True Concentration")
    plt.ylabel("Predicted Concentration")
    plt.title("PLS Prediction (Simulated Data)")
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
    plt.savefig("pls_prediction_simulated.png")
    plt.close()
else:
    for i in range(Y.shape[1]):
        plt.figure()
        plt.scatter(Y_test[:, i], Y_pred[:, i])
        plt.xlabel(f"True Component {i+1}")
        plt.ylabel(f"Predicted Component {i+1}")
        plt.title(f"PLS Prediction - Component {i+1}")
        plt.plot([Y_test[:, i].min(), Y_test[:, i].max()],
                 [Y_test[:, i].min(), Y_test[:, i].max()], 'r--')
        plt.savefig(f"pls_prediction_simulated_component{i+1}.png")
        plt.close()

# 6. 保存模型
joblib.dump(pls, "trained_pls_model.pkl")
print("模型已保存为 trained_pls_model.pkl")
