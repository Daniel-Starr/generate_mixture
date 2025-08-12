# 02_generate_mixture.py
import pandas as pd
import numpy as np

# 读取插值后的光谱文件
df = pd.read_csv("interpolated_spectra.csv")

# 获取 NO2 和 SO2 光谱向量
X_no2 = df['NO2'].values
X_so2 = df['SO2'].values

# 混合比例（40% NO2, 60% SO2）
ratio_no2 = 0.4
ratio_so2 = 0.6

# 合成混合光谱
X_mix = ratio_no2 * X_no2 + ratio_so2 * X_so2

# 浓度标签
y_mix = np.array([ratio_no2, ratio_so2])

# 保存混合光谱
mix_df = pd.DataFrame({
    'wavenumber': df['wavenumber'],
    'mixed_spectrum': X_mix
})
mix_df.to_csv("mixed_spectrum.csv", index=False)

# 保存浓度标签
label_df = pd.DataFrame([y_mix], columns=['NO2_conc', 'SO2_conc'])
label_df.to_csv("mixed_concentration.csv", index=False)

print("✅ 混合光谱已保存为 mixed_spectrum.csv")
print("✅ 浓度标签已保存为 mixed_concentration.csv")
