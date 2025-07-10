import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 配置
train_ratio = 0.7
max_components = 10
enable_plot = True

# 加载数据
X_df = pd.read_csv("X_dataset.csv")
Y_df = pd.read_csv("Y_labels.csv")

X = X_df.values
Y = Y_df.values

# 列名即为波数
wavenumbers = X_df.columns

# 数据拆分
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1 - train_ratio, random_state=0
)

# 交叉验证选择最优成分
best_score = -np.inf
best_n = 1
for n in range(1, max_components + 1):
    pls = PLSRegression(n_components=n)
    scores = cross_val_score(pls, X_train, Y_train[:, 0], cv=5, scoring="r2")
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_n = n

print(f"🔍 最优成分数: {best_n}, CV-R² (NO) = {best_score:.5f}")

# 最终模型
pls_final = PLSRegression(n_components=best_n)
pls_final.fit(X_train, Y_train)
Y_pred = pls_final.predict(X_test)

# 评估
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)
print(f"✅ 三气体自适应模型: RMSE = {rmse:.5f}, R² = {r2:.5f}")

# 保存预测结果
pd.DataFrame(Y_test, columns=["NO_true", "NO2_true", "SO2_true"]).to_csv(
    "Y_test_custom.csv", index=False
)
pd.DataFrame(Y_pred, columns=["NO_pred", "NO2_pred", "SO2_pred"]).to_csv(
    "Y_pred_custom.csv", index=False
)
with open("evaluation_custom.txt", "w") as f:
    f.write(f"Best n: {best_n}\nRMSE: {rmse:.5f}\nR2: {r2:.5f}\n")

# 保存系数
coefficients = pls_final.coef_.T
pd.DataFrame(
    coefficients,
    columns=["NO", "NO2", "SO2"],
    index=wavenumbers
).to_csv("pls_coefficients.csv")

print("✅ 已保存三气体的 PLS 加权系数到 pls_coefficients.csv")

# 可视化
if enable_plot:
    plt.figure(figsize=(15, 4))
    for i, name in enumerate(["NO", "NO2", "SO2"]):
        plt.subplot(1, 3, i + 1)
        plt.scatter(Y_test[:, i], Y_pred[:, i])
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"{name} Prediction")
    plt.tight_layout()
    plt.savefig("pls_prediction_custom.png", dpi=300)
    plt.show()
