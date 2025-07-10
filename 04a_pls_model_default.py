# 04a_pls_model_default.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 加载数据
X = pd.read_csv("X_dataset.csv").values
Y = pd.read_csv("Y_labels.csv").values

# 拆分数据（80% 训练，20% 测试）
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 拟合 PLS 模型（默认5个成分）
pls = PLSRegression(n_components=5)
pls.fit(X_train, Y_train)

# 预测
Y_pred = pls.predict(X_test)

# 评估（手动计算RMSE）
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)

# 输出评估指标
print(f"✅ 默认模型评估：RMSE = {rmse:.5f}, R² = {r2:.5f}")
with open("evaluation_default.txt", "w") as f:
    f.write(f"RMSE: {rmse:.5f}\nR2: {r2:.5f}")

# 保存预测与真实标签
pd.DataFrame(Y_test, columns=["NO2_true", "SO2_true"]).to_csv("Y_test_default.csv", index=False)
pd.DataFrame(Y_pred, columns=["NO2_pred", "SO2_pred"]).to_csv("Y_pred_default.csv", index=False)

# 可视化
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
plt.savefig("pls_prediction_default.png", dpi=300)
plt.show()
