# 04b_pls_model_custom.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ✅ 可调参数
train_ratio = 0.7         # 训练集占比
max_components = 10       # 尝试的最大成分数
enable_plot = True        # 是否可视化

# 加载数据
X = pd.read_csv("X_dataset.csv").values
Y = pd.read_csv("Y_labels.csv").values

# 拆分数据
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1-train_ratio, random_state=0)

# 自动选择最优成分数（根据 NO2）
best_score = -np.inf
best_n = 1
for n in range(1, max_components + 1):
    pls = PLSRegression(n_components=n)
    scores = cross_val_score(pls, X_train, Y_train[:, 0], cv=5, scoring='r2')
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_n = n

# 拟合模型
pls_final = PLSRegression(n_components=best_n)
pls_final.fit(X_train, Y_train)
Y_pred = pls_final.predict(X_test)

# 评估
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)
print(f"✅ 自定义模型评估：Best n={best_n}, RMSE = {rmse:.5f}, R² = {r2:.5f}")

# 保存结果
with open("evaluation_custom.txt", "w") as f:
    f.write(f"Best n: {best_n}\nRMSE: {rmse:.5f}\nR2: {r2:.5f}")

pd.DataFrame(Y_test, columns=["NO2_true", "SO2_true"]).to_csv("Y_test_custom.csv", index=False)
pd.DataFrame(Y_pred, columns=["NO2_pred", "SO2_pred"]).to_csv("Y_pred_custom.csv", index=False)

# 可视化
if enable_plot:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(Y_test[:, 0], Y_pred[:, 0])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("True NO2")
    plt.ylabel("Predicted NO2")
    plt.title("NO2 Prediction")

    plt.subplot(1, 2, 2)
    plt.scatter(Y_test[:, 1], Y_pred[:, 1])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("True SO2")
    plt.ylabel("Predicted SO2")
    plt.title("SO2 Prediction")
    plt.tight_layout()
    plt.savefig("pls_prediction_custom.png", dpi=300)
    plt.show()
