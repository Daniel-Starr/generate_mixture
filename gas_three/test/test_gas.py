# path_specific_mixer.py
# 指定路径的气体混合器 - 适配你的文件路径

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from datetime import datetime


def mix_gases_with_your_path():
    """使用你指定路径的气体混合器"""

    print("🧪 指定路径气体混合器")
    print("生成 NO:NO2:SO2 = 2:4:4 混合光谱")
    print("=" * 50)

    # 你的文件路径
    base_path = r"E:\generate_mixture\hitran_csv"
    no_file = os.path.join(base_path, "NO.csv")
    no2_file = os.path.join(base_path, "NO2.csv")
    so2_file = os.path.join(base_path, "SO2.csv")

    print(f"📂 数据路径: {base_path}")
    print(f"📄 文件列表:")
    print(f"   NO:  {no_file}")
    print(f"   NO2: {no2_file}")
    print(f"   SO2: {so2_file}")

    # 检查文件是否存在
    files_info = {
        'NO': no_file,
        'NO2': no2_file,
        'SO2': so2_file
    }

    missing_files = []
    for gas_name, file_path in files_info.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"   ✅ {gas_name}: 存在 ({file_size:.1f} MB)")
        else:
            print(f"   ❌ {gas_name}: 不存在")
            missing_files.append(gas_name)

    if missing_files:
        print(f"\n❌ 缺少文件: {', '.join(missing_files)}")
        return False

    try:
        # 1. 读取数据
        print(f"\n📖 读取气体数据...")

        no_df = pd.read_csv(no_file)
        no2_df = pd.read_csv(no2_file)
        so2_df = pd.read_csv(so2_file)

        print(f"   NO:  {len(no_df)} 行数据")
        print(f"   NO2: {len(no2_df)} 行数据")
        print(f"   SO2: {len(so2_df)} 行数据")

        # 检查数据格式
        for name, df in [('NO', no_df), ('NO2', no2_df), ('SO2', so2_df)]:
            if 'nu' not in df.columns or 'sw' not in df.columns:
                print(f"❌ {name} 文件格式错误，需要 'nu' 和 'sw' 列")
                print(f"   实际列名: {list(df.columns)}")
                return False

        print("✅ 数据格式检查通过")

        # 2. 数据清理和统计
        print(f"\n🔧 数据预处理...")

        # 清理数据
        no_clean = no_df[(no_df['nu'].notna()) & (no_df['sw'].notna()) & (no_df['nu'] > 0)].copy()
        no2_clean = no2_df[(no2_df['nu'].notna()) & (no2_df['sw'].notna()) & (no2_df['nu'] > 0)].copy()
        so2_clean = so2_df[(so2_df['nu'].notna()) & (so2_df['sw'].notna()) & (so2_df['nu'] > 0)].copy()

        # 排序
        no_clean = no_clean.sort_values('nu').reset_index(drop=True)
        no2_clean = no2_clean.sort_values('nu').reset_index(drop=True)
        so2_clean = so2_clean.sort_values('nu').reset_index(drop=True)

        # 统计信息
        print(f"   NO  清理后: {len(no_clean)} 点, 范围: {no_clean['nu'].min():.1f}-{no_clean['nu'].max():.1f} cm⁻¹")
        print(f"   NO2 清理后: {len(no2_clean)} 点, 范围: {no2_clean['nu'].min():.1f}-{no2_clean['nu'].max():.1f} cm⁻¹")
        print(f"   SO2 清理后: {len(so2_clean)} 点, 范围: {so2_clean['nu'].min():.1f}-{so2_clean['nu'].max():.1f} cm⁻¹")

        # 3. 确定共同波数范围
        print(f"\n🎯 分析波数范围...")

        no_min, no_max = no_clean['nu'].min(), no_clean['nu'].max()
        no2_min, no2_max = no2_clean['nu'].min(), no2_clean['nu'].max()
        so2_min, so2_max = so2_clean['nu'].min(), so2_clean['nu'].max()

        # 计算共同范围
        common_min = max(no_min, no2_min, so2_min)
        common_max = min(no_max, no2_max, so2_max)

        print(f"   NO  范围: {no_min:.1f} - {no_max:.1f} cm⁻¹")
        print(f"   NO2 范围: {no2_min:.1f} - {no2_max:.1f} cm⁻¹")
        print(f"   SO2 范围: {so2_min:.1f} - {so2_max:.1f} cm⁻¹")
        print(f"   共同范围: {common_min:.1f} - {common_max:.1f} cm⁻¹")

        if common_min >= common_max:
            print("   ⚠️ 没有完全重叠，使用扩展范围")
            # 使用更大范围
            extended_min = min(no_min, no2_min, so2_min)
            extended_max = max(no_max, no2_max, so2_max)
            common_min, common_max = extended_min, extended_max
            print(f"   扩展范围: {common_min:.1f} - {common_max:.1f} cm⁻¹")

        # 4. 创建统一网格
        step_size = 1.0  # 1 cm⁻¹ 间隔
        wavenumber_grid = np.arange(
            np.ceil(common_min),
            np.floor(common_max) + step_size,
            step_size
        )

        print(f"   统一网格: {len(wavenumber_grid)} 点 (步长: {step_size} cm⁻¹)")

        # 5. 插值处理
        print(f"\n🔄 插值到统一网格...")

        # 创建插值函数
        no_interp = interp1d(no_clean['nu'], no_clean['sw'],
                             kind='linear', bounds_error=False, fill_value=0)
        no2_interp = interp1d(no2_clean['nu'], no2_clean['sw'],
                              kind='linear', bounds_error=False, fill_value=0)
        so2_interp = interp1d(so2_clean['nu'], so2_clean['sw'],
                              kind='linear', bounds_error=False, fill_value=0)

        # 执行插值
        no_intensity = no_interp(wavenumber_grid)
        no2_intensity = no2_interp(wavenumber_grid)
        so2_intensity = so2_interp(wavenumber_grid)

        # 计算覆盖度
        no_coverage = np.sum(no_intensity != 0) / len(wavenumber_grid)
        no2_coverage = np.sum(no2_intensity != 0) / len(wavenumber_grid)
        so2_coverage = np.sum(so2_intensity != 0) / len(wavenumber_grid)

        print(f"   NO  覆盖度: {no_coverage:.1%} ({np.sum(no_intensity != 0)} 有效点)")
        print(f"   NO2 覆盖度: {no2_coverage:.1%} ({np.sum(no2_intensity != 0)} 有效点)")
        print(f"   SO2 覆盖度: {so2_coverage:.1%} ({np.sum(so2_intensity != 0)} 有效点)")

        # 6. 按2:4:4混合
        print(f"\n🧪 按2:4:4比例混合...")

        # 混合比例
        ratios = [2, 4, 4]  # NO:NO2:SO2
        total = sum(ratios)
        ratio_no = ratios[0] / total  # 2/10 = 0.2
        ratio_no2 = ratios[1] / total  # 4/10 = 0.4
        ratio_so2 = ratios[2] / total  # 4/10 = 0.4

        print(f"   NO:  {ratios[0]}/{total} = {ratio_no:.1%}")
        print(f"   NO2: {ratios[1]}/{total} = {ratio_no2:.1%}")
        print(f"   SO2: {ratios[2]}/{total} = {ratio_so2:.1%}")

        # 生成混合光谱
        mixed_spectrum = (ratio_no * no_intensity +
                          ratio_no2 * no2_intensity +
                          ratio_so2 * so2_intensity)

        # 添加轻微噪声
        noise_level = 0.01  # 1%
        noise = np.random.normal(0, noise_level * np.max(mixed_spectrum), size=mixed_spectrum.shape)
        mixed_spectrum_noisy = mixed_spectrum + noise
        mixed_spectrum_noisy = np.maximum(mixed_spectrum_noisy, 0)  # 确保非负

        print(f"   添加 {noise_level * 100:.1f}% 噪声")

        # 7. 保存结果到你的目录
        print(f"\n💾 保存结果...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存路径（保存到当前工作目录）
        output_dir = "."  # 当前目录，你也可以改为你想要的路径

        # 干净的混合光谱
        clean_df = pd.DataFrame({
            'wavenumber': wavenumber_grid,
            'intensity': mixed_spectrum
        })
        clean_filename = f'mixed_spectrum_244_clean_{timestamp}.csv'
        clean_filepath = os.path.join(output_dir, clean_filename)
        clean_df.to_csv(clean_filepath, index=False)

        # 含噪声的混合光谱
        noisy_df = pd.DataFrame({
            'wavenumber': wavenumber_grid,
            'intensity': mixed_spectrum_noisy
        })
        noisy_filename = f'mixed_spectrum_244_noisy_{timestamp}.csv'
        noisy_filepath = os.path.join(output_dir, noisy_filename)
        noisy_df.to_csv(noisy_filepath, index=False)

        print(f"   ✅ 干净光谱: {clean_filepath}")
        print(f"   ✅ 含噪声光谱: {noisy_filepath}")

        # 8. 生成可视化
        print(f"\n📊 生成可视化图表...")

        create_visualization(wavenumber_grid, no_intensity, no2_intensity, so2_intensity,
                             mixed_spectrum, mixed_spectrum_noisy, [ratio_no, ratio_no2, ratio_so2], timestamp)

        # 9. 输出详细统计
        print(f"\n📈 混合光谱统计:")
        print(f"   波数范围: {wavenumber_grid.min():.1f} - {wavenumber_grid.max():.1f} cm⁻¹")
        print(f"   数据点数: {len(wavenumber_grid)}")
        print(f"   非零点数: {np.sum(mixed_spectrum > 0)}")
        print(f"   最大强度: {mixed_spectrum.max():.3e}")
        print(f"   平均强度: {mixed_spectrum.mean():.3e}")
        print(f"   强度范围: {mixed_spectrum.min():.3e} - {mixed_spectrum.max():.3e}")

        # 各组分贡献分析
        no_contribution = ratio_no * np.max(no_intensity)
        no2_contribution = ratio_no2 * np.max(no2_intensity)
        so2_contribution = ratio_so2 * np.max(so2_intensity)

        print(f"\n🔍 各组分最大贡献:")
        print(f"   NO:  {no_contribution:.3e}")
        print(f"   NO2: {no2_contribution:.3e}")
        print(f"   SO2: {so2_contribution:.3e}")

        print(f"\n🎉 混合光谱生成完成！")
        print(f"📁 输出文件:")
        print(f"   📄 {clean_filename}")
        print(f"   📄 {noisy_filename}")
        print(f"   📊 mixture_244_visualization_{timestamp}.png")

        return True

    except Exception as e:
        print(f"❌ 处理出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_visualization(wavenumber, no_intensity, no2_intensity, so2_intensity,
                         mixed_clean, mixed_noisy, ratios, timestamp):
    """创建详细的可视化图表"""

    plt.rcParams['font.size'] = 10
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    colors = ['red', 'blue', 'green']
    gas_names = ['NO', 'NO2', 'SO2']
    spectra = [no_intensity, no2_intensity, so2_intensity]

    # 第一行：单独气体光谱
    for i in range(3):
        axes[0, i].plot(wavenumber, spectra[i], color=colors[i], linewidth=1, alpha=0.8)
        axes[0, i].set_title(f'{gas_names[i]} Spectrum\n(Ratio: {ratios[i]:.1%})', fontweight='bold')
        axes[0, i].set_xlabel('Wavenumber (cm⁻¹)')
        axes[0, i].set_ylabel('Intensity')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        # 添加统计信息
        max_val = spectra[i].max()
        non_zero = np.sum(spectra[i] > 0)
        axes[0, i].text(0.02, 0.98, f'Max: {max_val:.2e}\nPoints: {non_zero}',
                        transform=axes[0, i].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                        verticalalignment='top', fontsize=9)

    # 第二行左：加权光谱叠加
    for i in range(3):
        weighted = ratios[i] * spectra[i]
        axes[1, 0].plot(wavenumber, weighted, color=colors[i],
                        label=f'{gas_names[i]} ({ratios[i]:.1%})', alpha=0.7, linewidth=1)

    axes[1, 0].plot(wavenumber, mixed_clean, 'black',
                    label='Mixed (clean)', linewidth=2, alpha=0.9)
    axes[1, 0].set_title('Weighted Components & Mixed Result', fontweight='bold')
    axes[1, 0].set_xlabel('Wavenumber (cm⁻¹)')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # 第二行中：干净vs噪声对比
    axes[1, 1].plot(wavenumber, mixed_clean, 'blue',
                    label='Clean', linewidth=2, alpha=0.8)
    axes[1, 1].plot(wavenumber, mixed_noisy, 'red',
                    label='With 1% Noise', linewidth=1, alpha=0.6)
    axes[1, 1].set_title('Clean vs Noisy Spectrum', fontweight='bold')
    axes[1, 1].set_xlabel('Wavenumber (cm⁻¹)')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # 第二行右：混合比例和统计
    # 饼图
    ratio_percentages = [r * 100 for r in ratios]
    wedges, texts, autotexts = axes[1, 2].pie(ratios,
                                              labels=[f'{name}\n{ratio_percentages[i]:.1f}%'
                                                      for i, name in enumerate(gas_names)],
                                              colors=colors, autopct='', startangle=90)

    axes[1, 2].set_title('Mixture Ratios (2:4:4)\nNO:NO2:SO2', fontweight='bold')

    # 添加详细统计
    stats_text = f"""Mixed Spectrum Statistics:
Max Intensity: {mixed_clean.max():.2e}
Mean Intensity: {mixed_clean.mean():.2e}
Total Data Points: {len(wavenumber):,}
Non-zero Points: {np.sum(mixed_clean > 0):,}
Coverage: {np.sum(mixed_clean > 0) / len(wavenumber):.1%}
Wavenumber Range: {wavenumber.min():.0f}-{wavenumber.max():.0f} cm⁻¹"""

    axes[1, 2].text(0.0, -1.4, stats_text,
                    transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
                    fontsize=9, verticalalignment='top')

    # 添加总标题
    plt.suptitle(f'Gas Mixture Analysis: NO:NO2:SO2 = 2:4:4\n'
                 f'Generated: {timestamp} | Source: E:\\generate_mixture\\hitran_csv\\',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存图表
    plot_filename = f'mixture_244_visualization_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"   📊 可视化图表: {plot_filename}")

    plt.show()


# 主程序
if __name__ == "__main__":
    print("🔬 指定路径气体混合器")
    print("适用于你的HITRAN数据路径")
    print("=" * 60)

    success = mix_gases_with_your_path()

    if success:
        print(f"\n✅ 混合光谱生成成功！")
        print(f"💡 提示:")
        print(f"   • 生成的文件可直接用于光谱分析")
        print(f"   • 干净光谱适合理论分析")
        print(f"   • 含噪声光谱更接近实际测量")
        print(f"   • 可以修改代码中的混合比例")
    else:
        print(f"\n❌ 生成失败")
        print(f"💡 请检查:")
        print(f"   • 文件路径是否正确")
        print(f"   • 文件是否包含 'nu' 和 'sw' 列")
        print(f"   • 数据格式是否正确")