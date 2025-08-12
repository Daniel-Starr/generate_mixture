"""
调试脚本 - 检查光谱数据问题
"""
import pandas as pd
import numpy as np
from pathlib import Path

def check_spectral_data(file_path):
    """检查光谱数据文件的问题"""
    print(f"\n检查文件: {file_path}")

    # 读取数据
    df = pd.read_csv(file_path)
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    # 检查波数列
    if 'nu' in df.columns:
        nu_col = 'nu'
    elif df.columns[0].lower() in ['wavenumber', 'wave', 'wn']:
        nu_col = df.columns[0]
    else:
        nu_col = df.columns[0]

    print(f"\n波数列: {nu_col}")
    print(f"波数范围: {df[nu_col].min():.2f} - {df[nu_col].max():.2f}")

    # 检查重复值
    duplicates = df[nu_col].duplicated().sum()
    print(f"重复的波数值: {duplicates} 个")

    if duplicates > 0:
        print("\n前10个重复的波数值:")
        dup_values = df[df[nu_col].duplicated(keep=False)][nu_col].unique()[:10]
        for val in dup_values:
            count = (df[nu_col] == val).sum()
            print(f"  {val}: 出现 {count} 次")

    # 检查NaN值
    nan_count = df.isna().sum().sum()
    print(f"\nNaN值总数: {nan_count}")

    # 检查数据类型
    print(f"\n数据类型:")
    print(df.dtypes)

    return df

def fix_spectral_data(input_file, output_file):
    """修复光谱数据中的问题"""
    print(f"\n修复文件: {input_file}")

    # 读取数据
    df = pd.read_csv(input_file)

    # 确定列名
    if 'nu' in df.columns and 'sw' in df.columns:
        nu_col, sw_col = 'nu', 'sw'
    else:
        # 假设第一列是波数，第二列是强度
        nu_col, sw_col = df.columns[0], df.columns[1]

    print(f"使用列: {nu_col} (波数), {sw_col} (强度)")

    # 重命名列为标准名称
    df_fixed = df[[nu_col, sw_col]].copy()
    df_fixed.columns = ['nu', 'sw']

    # 移除NaN
    original_len = len(df_fixed)
    df_fixed = df_fixed.dropna()
    print(f"移除 {original_len - len(df_fixed)} 个NaN行")

    # 按波数排序
    df_fixed = df_fixed.sort_values('nu')

    # 处理重复值 - 取平均
    original_len = len(df_fixed)
    df_fixed = df_fixed.groupby('nu').agg({'sw': 'mean'}).reset_index()
    print(f"合并 {original_len - len(df_fixed)} 个重复波数")

    # 保存修复后的数据
    df_fixed.to_csv(output_file, index=False)
    print(f"修复后的数据已保存到: {output_file}")
    print(f"最终数据点数: {len(df_fixed)}")

    return df_fixed

def main():
    """主函数"""
    print("=== 光谱数据诊断工具 ===")

    # 检查当前目录下的CSV文件
    csv_files = list(Path('.').glob('*.csv'))
    print(f"\n找到 {len(csv_files)} 个CSV文件:")
    for f in csv_files:
        print(f"  - {f}")

    # 检查hitran_csv目录
    hitran_dir = Path('hitran_csv')
    if hitran_dir.exists():
        print(f"\n在 hitran_csv 目录中找到文件:")
        hitran_files = list(hitran_dir.glob('*.csv'))
        for f in hitran_files:
            print(f"  - {f}")

    # 检查气体文件
    gas_files = ['NO.csv', 'NO2.csv', 'SO2.csv']

    for gas_file in gas_files:
        # 先检查 hitran_csv 目录
        file_path = hitran_dir / gas_file if hitran_dir.exists() else Path(gas_file)

        if not file_path.exists():
            # 如果不存在，尝试当前目录
            file_path = Path(gas_file)

        if file_path.exists():
            # 检查原始数据
            df = check_spectral_data(file_path)

            # 如果有问题，修复它
            if df['nu'].duplicated().sum() > 0 or df.isna().sum().sum() > 0:
                fixed_file = f"{gas_file.split('.')[0]}_fixed.csv"
                fix_spectral_data(file_path, fixed_file)
                print(f"\n建议：使用修复后的文件 {fixed_file} 替换原文件")
        else:
            print(f"\n⚠️ 文件不存在: {gas_file}")

    print("\n" + "="*50)
    print("诊断完成！")
    print("\n如果有数据问题，可以：")
    print("1. 使用生成的 _fixed.csv 文件")
    print("2. 或运行以下命令修复所有文件：")
    print("   python debug_script.py --fix-all")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--fix-all":
        # 自动修复所有气体文件
        print("=== 自动修复模式 ===")

        # 检查 hitran_csv_fixed 目录中的修复文件
        fixed_dir = Path('hitran_csv_fixed')
        if fixed_dir.exists():
            print(f"从 {fixed_dir} 目录复制修复后的文件...")
            for gas in ['NO', 'NO2', 'SO2']:
                fixed_file = fixed_dir / f"{gas}_fixed.csv"
                target_file = Path(f"{gas}.csv")

                if fixed_file.exists():
                    import shutil
                    shutil.copy2(fixed_file, target_file)
                    print(f"✅ {fixed_file} -> {target_file}")
                else:
                    # 如果 _fixed 文件不存在，尝试直接的文件名
                    alt_file = fixed_dir / f"{gas}.csv"
                    if alt_file.exists():
                        import shutil
                        shutil.copy2(alt_file, target_file)
                        print(f"✅ {alt_file} -> {target_file}")
                    else:
                        print(f"❌ 未找到 {gas} 的修复文件")
        else:
            # 如果没有 fixed 目录，尝试从当前目录
            print("从当前目录查找修复后的文件...")
            for gas in ['NO', 'NO2', 'SO2']:
                fixed_file = Path(f"{gas}_fixed.csv")
                if fixed_file.exists():
                    import shutil
                    shutil.copy2(fixed_file, f"{gas}.csv")
                    print(f"✅ {fixed_file} -> {gas}.csv")

        print("\n文件准备完成！")
        print("现在可以运行: python gas_spectroscopy_system.py")
    else:
        main()