import pandas as pd
import numpy as np
from scipy.stats import truncnorm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重现性
np.random.seed(42)

# 读取插值后的三气体光谱
df = pd.read_csv("data/processed/interpolated_spectra.csv")
wavenumbers = df['wavenumber'].values
no  = df['NO'].values
no2 = df['NO2'].values
so2 = df['SO2'].values

# 大幅增加浓度组合的复杂性和多样性
print("Generating high-complexity gas mixture dataset...")

# 1. 使用更精细的浓度网格
no_ratios  = np.arange(0.05, 0.65, 0.02)   # 从5%到65%，步长2%
no2_ratios = np.arange(0.05, 0.65, 0.02)   # 从5%到65%，步长2%

# 2. 增强噪声模型：多层次噪声（提升到10%水平）
base_noise_level = 0.10  # 基础噪声10%
spectral_noise_level = 0.08  # 光谱相关噪声8%
systematic_noise_level = 0.05  # 系统性偏差5%

# 3. 样本数量动态调整
samples_per_ratio = 15  # 每个组合增加到15个样本

X_data = []
Y_labels = []
valid_combinations = 0
invalid_combinations = 0

# 4. 添加非线性混合效应的权重
nonlinear_strength = 0.15  # 非线性效应强度

for r_no in no_ratios:
    for r_no2 in no2_ratios:
        r_so2 = 1.0 - r_no - r_no2
        
        # 更严格的浓度约束：确保每种气体至少3%
        if r_so2 < 0.03 or r_no < 0.03 or r_no2 < 0.03:
            invalid_combinations += 1
            continue
        
        # 限制高浓度组合，避免不现实的情况
        if r_no > 0.6 or r_no2 > 0.6 or r_so2 > 0.8:
            invalid_combinations += 1
            continue
            
        valid_combinations += 1
        
        for sample_idx in range(samples_per_ratio):
            # 5. 基础线性混合
            mixed_base = r_no * no + r_no2 * no2 + r_so2 * so2
            
            # 6. 添加非线性交互效应（模拟真实气体分子间相互作用）
            nonlinear_term = nonlinear_strength * (
                r_no * r_no2 * (no * no2) / (no.max() * no2.max()) +
                r_no * r_so2 * (no * so2) / (no.max() * so2.max()) +
                r_no2 * r_so2 * (no2 * so2) / (no2.max() * so2.max())
            )
            
            mixed = mixed_base + nonlinear_term
            
            # 7. 多层次噪声模型
            # 基础高斯噪声
            gaussian_noise = np.random.normal(0, base_noise_level, size=mixed.shape)
            
            # 光谱相关噪声（相邻波数相关）
            spectral_noise = np.zeros_like(mixed)
            for i in range(1, len(spectral_noise)):
                spectral_noise[i] = 0.7 * spectral_noise[i-1] + np.random.normal(0, spectral_noise_level)
            
            # 系统性偏差（模拟仪器漂移）
            systematic_bias = systematic_noise_level * np.sin(2 * np.pi * np.arange(len(mixed)) / len(mixed))
            
            # 8. 应用综合噪声
            noisy_sample = mixed * (1 + gaussian_noise + spectral_noise) + systematic_bias
            
            # 9. 确保物理意义（非负值）
            noisy_sample = np.maximum(noisy_sample, 0)
            
            X_data.append(noisy_sample)
            Y_labels.append([r_no, r_no2, r_so2])

# 10. 数据质量验证和统计
print(f"Enhanced dataset generation completed:")
print(f"   Valid combinations: {valid_combinations}")
print(f"   Invalid combinations: {invalid_combinations}")
print(f"   Total samples: {len(X_data)}")
print(f"   Concentration ranges: NO={no_ratios.min():.2f}-{no_ratios.max():.2f}, NO2={no2_ratios.min():.2f}-{no2_ratios.max():.2f}")

# 转为 DataFrame
X_df = pd.DataFrame(X_data, columns=[f'{w:.1f}cm-1' for w in wavenumbers])
Y_df = pd.DataFrame(Y_labels, columns=['NO_conc', 'NO2_conc', 'SO2_conc'])

# 11. 数据质量检查
concentration_sums = Y_df.sum(axis=1)
min_sum, max_sum = concentration_sums.min(), concentration_sums.max()
print(f"   Concentration sum range: {min_sum:.6f} - {max_sum:.6f} (should be 1.0)")

# 统计每种气体的浓度分布
print(f"   NO concentration: {Y_df['NO_conc'].min():.3f} - {Y_df['NO_conc'].max():.3f} (mean: {Y_df['NO_conc'].mean():.3f})")
print(f"   NO2 concentration: {Y_df['NO2_conc'].min():.3f} - {Y_df['NO2_conc'].max():.3f} (mean: {Y_df['NO2_conc'].mean():.3f})")
print(f"   SO2 concentration: {Y_df['SO2_conc'].min():.3f} - {Y_df['SO2_conc'].max():.3f} (mean: {Y_df['SO2_conc'].mean():.3f})")

# 检查光谱数据的范围和质量
print(f"   Spectral intensity range: {X_df.values.min():.2e} - {X_df.values.max():.2e}")
print(f"   Spectral dimensions: {X_df.shape[1]} wavenumbers")

# 12. 唯一组合验证
unique_combinations = Y_df.drop_duplicates()
print(f"   Unique concentration combinations: {len(unique_combinations)}")

# 保存数据
X_df.to_csv("data/processed/X_dataset.csv", index=False)
Y_df.to_csv("data/processed/Y_labels.csv", index=False)

# 13. 保存数据生成参数记录
generation_params = {
    'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    'base_noise_level': base_noise_level,
    'spectral_noise_level': spectral_noise_level,
    'systematic_noise_level': systematic_noise_level,
    'nonlinear_strength': nonlinear_strength,
    'samples_per_ratio': samples_per_ratio,
    'valid_combinations': valid_combinations,
    'total_samples': len(X_data),
    'unique_combinations': len(unique_combinations)
}

import json
with open("data/processed/generation_params.json", "w") as f:
    json.dump(generation_params, f, indent=2)

print(f"   Generation parameters saved to: data/processed/generation_params.json")
print(f"   Dataset saved to: data/processed/X_dataset.csv and Y_labels.csv")

