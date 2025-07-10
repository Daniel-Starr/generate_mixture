import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# 读取三种气体的 CSV
no_df  = pd.read_csv(r"E:\generate_mixture\hitran_csv\NO.csv")
no2_df = pd.read_csv(r"E:\generate_mixture\hitran_csv\NO2.csv")
so2_df = pd.read_csv(r"E:\generate_mixture\hitran_csv\SO2.csv")

# 查看它们的波数范围
no_min, no_max   = no_df['nu'].min(), no_df['nu'].max()
no2_min, no2_max = no2_df['nu'].min(), no2_df['nu'].max()
so2_min, so2_max = so2_df['nu'].min(), so2_df['nu'].max()

# 取三者共同的波数范围
common_min = max(no_min, no2_min, so2_min)
common_max = min(no_max, no2_max, so2_max)

print(f"统一波数区间: {common_min:.2f} ~ {common_max:.2f}")

# 构建统一波数轴，步长 1
common_nu = np.arange(np.ceil(common_min), np.floor(common_max) + 1, 1)

# 插值
interp_no  = interp1d(no_df['nu'],  no_df['sw'],  bounds_error=False, fill_value=0)
interp_no2 = interp1d(no2_df['nu'], no2_df['sw'], bounds_error=False, fill_value=0)
interp_so2 = interp1d(so2_df['nu'], so2_df['sw'], bounds_error=False, fill_value=0)

# 生成在统一波数轴上的吸收系数
sw_no_interp  = interp_no(common_nu)
sw_no2_interp = interp_no2(common_nu)
sw_so2_interp = interp_so2(common_nu)

# 合成 DataFrame
interpolated_df = pd.DataFrame({
    'wavenumber': common_nu,
    'NO':  sw_no_interp,
    'NO2': sw_no2_interp,
    'SO2': sw_so2_interp
})

# 保存
interpolated_df.to_csv("interpolated_spectra.csv", index=False)
print("✅ 已生成插值后的三气体光谱: interpolated_spectra.csv")
