# -*- coding: utf-8 -*-
# generate_concentration_table.py
# 生成气体浓度组合表格

import numpy as np
import pandas as pd

def generate_concentration_table():
    """生成并打印气体浓度组合表格"""
    
    # 生成比例组合（与03_generate_dataset.py中相同的设置）
    no_ratios = np.arange(0.2, 0.45, 0.05)   # [0.2, 0.25, 0.3, 0.35, 0.4]
    no2_ratios = np.arange(0.3, 0.55, 0.05)  # [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    
    print("气体浓度组合详细表格")
    print("=" * 70)
    print("| NO (%) | NO2 (%) | SO2 (%) | 总和 | 是否有效 |")
    print("|--------|---------|---------|------|----------|")
    
    # 存储所有组合的数据
    table_data = []
    valid_count = 0
    invalid_count = 0
    
    for r_no in no_ratios:
        for r_no2 in no2_ratios:
            r_so2 = 1.0 - r_no - r_no2
            total = r_no + r_no2 + r_so2
            
            # 检查有效性（与代码中的约束条件一致）
            if r_so2 < 0.05:  # SO2浓度至少5%
                status = "无效"
                is_valid = False
                invalid_count += 1
            else:
                status = "有效"
                is_valid = True
                valid_count += 1
            
            # 格式化输出
            no_pct = r_no * 100
            no2_pct = r_no2 * 100
            so2_pct = r_so2 * 100
            total_pct = total * 100
            
            print(f"| {no_pct:4.0f}   | {no2_pct:5.0f}   | {so2_pct:5.0f}   | {total_pct:3.0f}  | {status} |")
            
            # 保存数据用于后续分析
            table_data.append({
                'NO_pct': no_pct,
                'NO2_pct': no2_pct, 
                'SO2_pct': so2_pct,
                'total': total_pct,
                'is_valid': is_valid,
                'NO_ratio': r_no,
                'NO2_ratio': r_no2,
                'SO2_ratio': r_so2
            })
    
    print("=" * 70)
    print(f"统计结果:")
    print(f"   有效组合: {valid_count} 个")
    print(f"   无效组合: {invalid_count} 个") 
    print(f"   总组合数: {valid_count + invalid_count} 个")
    print(f"   每种组合生成样本: 10个（带1%噪声）")
    print(f"   最终训练样本: {valid_count * 10} 个")
    
    # 保存为CSV文件
    df = pd.DataFrame(table_data)
    df.to_csv("data/results/concentration_combinations.csv", index=False)
    print(f"\n表格数据已保存到: data/results/concentration_combinations.csv")
    
    # 分析浓度范围
    valid_data = df[df['is_valid']]
    if len(valid_data) > 0:
        print(f"\n有效组合的浓度范围:")
        print(f"   NO:  {valid_data['NO_pct'].min():.0f}% - {valid_data['NO_pct'].max():.0f}%")
        print(f"   NO2: {valid_data['NO2_pct'].min():.0f}% - {valid_data['NO2_pct'].max():.0f}%")
        print(f"   SO2: {valid_data['SO2_pct'].min():.0f}% - {valid_data['SO2_pct'].max():.0f}%")
    
    return table_data

if __name__ == "__main__":
    # 确保结果目录存在
    import os
    os.makedirs("data/results", exist_ok=True)
    
    # 生成表格
    concentration_data = generate_concentration_table()
    
    print(f"\n脚本执行完成！")