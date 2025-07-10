# 03_generate_dataset.py
import pandas as pd
import numpy as np

# 读取插值后的 NO2 和 SO2 光谱数据
df = pd.read_csv("interpolated_spectra.csv")
wavenumbers = df['wavenumber'].values
no2 = df['NO2'].values
so2 = df['SO2'].values

# 设置 NO2 浓度范围：30% 到 50%，步长 0.05
no2_ratios = np.arange(0.30, 0.55, 0.05)
num_samples_per_ratio = 10  # 每个比例生成 10 个样本
noise_level = 0.01          # 高斯噪声强度（1% 波动）

X_data = []
Y_labels = []

# 逐个比例生成样本
for r_no2 in no2_ratios:
    r_so2 = 1 - r_no2
    mixed = r_no2 * no2 + r_so2 * so2
    for _ in range(num_samples_per_ratio):
        # 添加噪声（乘以 1±小量）
        noise = np.random.normal(loc=0, scale=noise_level, size=mixed.shape)
        noisy_sample = mixed * (1 + noise)
        X_data.append(noisy_sample)
        Y_labels.append([r_no2, r_so2])

# 转为 DataFrame
X_df = pd.DataFrame(X_data, columns=[f"{w}cm-1" for w in wavenumbers])
Y_df = pd.DataFrame(Y_labels, columns=['NO2_conc', 'SO2_conc'])

# 保存结果
X_df.to_csv("X_dataset.csv", index=False)
Y_df.to_csv("Y_labels.csv", index=False)

print("✅ 已生成混合光谱数据集（带噪声）")
print("→ X_dataset.csv: 混合光谱矩阵")
print("→ Y_labels.csv: 对应浓度标签")
