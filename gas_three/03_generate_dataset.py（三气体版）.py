import pandas as pd
import numpy as np

# 读取插值后的三气体光谱
df = pd.read_csv("interpolated_spectra.csv")
wavenumbers = df['wavenumber'].values
no  = df['NO'].values
no2 = df['NO2'].values
so2 = df['SO2'].values

# 生成比例组合
no_ratios  = np.arange(0.2, 0.45, 0.05)
no2_ratios = np.arange(0.3, 0.55, 0.05)

X_data = []
Y_labels = []

noise_level = 0.01  # 1%噪声
samples_per_ratio = 10

for r_no in no_ratios:
    for r_no2 in no2_ratios:
        r_so2 = 1.0 - r_no - r_no2
        if r_so2 < 0:
            continue  # 无效比例，跳过
        for _ in range(samples_per_ratio):
            mixed = r_no * no + r_no2 * no2 + r_so2 * so2
            noise = np.random.normal(0, noise_level, size=mixed.shape)
            noisy_sample = mixed * (1 + noise)
            X_data.append(noisy_sample)
            Y_labels.append([r_no, r_no2, r_so2])

# 转为 DataFrame
X_df = pd.DataFrame(X_data, columns=[f'{w}cm-1' for w in wavenumbers])
Y_df = pd.DataFrame(Y_labels, columns=['NO_conc', 'NO2_conc', 'SO2_conc'])

# 保存
X_df.to_csv("X_dataset.csv", index=False)
Y_df.to_csv("Y_labels.csv", index=False)

print(f"✅ 生成多比例三气体数据集，共 {len(X_df)} 条样本")

