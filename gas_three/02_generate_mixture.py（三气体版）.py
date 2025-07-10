import pandas as pd
import numpy as np

# 读取插值后的三气体光谱
df = pd.read_csv("interpolated_spectra.csv")

# 提取
X_no  = df['NO'].values
X_no2 = df['NO2'].values
X_so2 = df['SO2'].values

# 比例
ratio_no  = 0.3
ratio_no2 = 0.4
ratio_so2 = 0.3

# 混合
X_mix = ratio_no * X_no + ratio_no2 * X_no2 + ratio_so2 * X_so2

# 浓度标签
y_mix = np.array([ratio_no, ratio_no2, ratio_so2])

# 保存
mix_df = pd.DataFrame({
    'wavenumber': df['wavenumber'],
    'mixed_spectrum': X_mix
})
mix_df.to_csv("mixed_spectrum.csv", index=False)

# 浓度标签
label_df = pd.DataFrame([y_mix], columns=['NO_conc', 'NO2_conc', 'SO2_conc'])
label_df.to_csv("mixed_concentration.csv", index=False)

print("✅ 已生成三气体混合样本")
