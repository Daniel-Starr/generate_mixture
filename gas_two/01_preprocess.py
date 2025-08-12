# 01_preprocess.py
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# 路径设置（使用你本地的实际路径）
no2_path = r"E:\generate_mixture\hitran_csv\NO2.csv"
so2_path = r"E:\generate_mixture\hitran_csv\SO2.csv"

# 读取 NO2 和 SO2 数据
no2_df = pd.read_csv(no2_path)
so2_df = pd.read_csv(so2_path)

# 构建统一波数轴（587–3689 cm⁻¹）
common_nu = np.arange(587, 3690, 1)

# 插值（超出范围填充为 0）
interp_no2 = interp1d(no2_df['nu'], no2_df['sw'], bounds_error=False, fill_value=0)
interp_so2 = interp1d(so2_df['nu'], so2_df['sw'], bounds_error=False, fill_value=0)

# 插值后的吸收强度
sw_no2_interp = interp_no2(common_nu)
sw_so2_interp = interp_so2(common_nu)

# 合并为 DataFrame
interpolated_df = pd.DataFrame({
    'wavenumber': common_nu,
    'NO2': sw_no2_interp,
    'SO2': sw_so2_interp
})

# 保存到当前目录
interpolated_df.to_csv("interpolated_spectra.csv", index=False)
print("✅ 插值完成，已保存为 interpolated_spectra.csv")
