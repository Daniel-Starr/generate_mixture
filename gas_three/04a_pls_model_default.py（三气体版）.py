import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 加载数据
X = pd.read_csv("data/processed/X_dataset.csv").values
Y = pd.read_csv("data/processed/Y_labels.csv").values

# 拆分（80% 训练，20% 测试）
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# PLS 模型
pls = PLSRegression(n_components=5)
pls.fit(X_train, Y_train)

# 预测
Y_pred = pls.predict(X_test)

# 评估
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)

print(f"✅ 三气体默认模型: RMSE = {rmse:.5f}, R² = {r2:.5f}")

# 保存
pd.DataFrame(Y_test, columns=["NO_true", "NO2_true", "SO2_true"]).to_csv("data/results/Y_test_default.csv", index=False)
pd.DataFrame(Y_pred, columns=["NO_pred", "NO2_pred", "SO2_pred"]).to_csv("data/results/Y_pred_default.csv", index=False)
with open("data/results/evaluation_default.txt", "w") as f:
    f.write(f"RMSE: {rmse:.5f}\nR2: {r2:.5f}\n")

# 可视化
plt.figure(figsize=(15, 4))
for i, name in enumerate(["NO", "NO2", "SO2"]):
    plt.subplot(1, 3, i+1)
    plt.scatter(Y_test[:, i], Y_pred[:, i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"{name} Prediction")
plt.tight_layout()
plt.savefig("data/figures/pls_prediction_default.png", dpi=300)
plt.show()
