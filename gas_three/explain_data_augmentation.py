# -*- coding: utf-8 -*-
# explain_data_augmentation.py  
# 详细解释数据增强、约束条件和最终数据集的概念

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def explain_concepts():
    """详细解释数据增强相关概念"""
    
    print("=" * 80)
    print("数据增强、约束条件和最终数据集概念详解")
    print("=" * 80)
    
    # 1. 基础浓度组合
    print("\n1. 基础浓度组合生成")
    print("-" * 40)
    
    no_ratios = np.arange(0.2, 0.45, 0.05)   # [0.2, 0.25, 0.3, 0.35, 0.4]
    no2_ratios = np.arange(0.3, 0.55, 0.05)  # [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    
    print(f"NO浓度选项: {[f'{x*100:.0f}%' for x in no_ratios]}")
    print(f"NO2浓度选项: {[f'{x*100:.0f}%' for x in no2_ratios]}")
    print(f"理论组合数: {len(no_ratios)} × {len(no2_ratios)} = {len(no_ratios) * len(no2_ratios)} 种")
    
    # 2. 约束条件检查
    print("\n2. 约束条件检查")
    print("-" * 40)
    
    valid_combinations = []
    invalid_combinations = []
    
    for r_no in no_ratios:
        for r_no2 in no2_ratios:
            r_so2 = 1.0 - r_no - r_no2
            
            # 约束条件检查
            if r_so2 < 0.05:  # SO2浓度至少5%
                invalid_combinations.append((r_no, r_no2, r_so2))
                print(f"❌ 无效: NO={r_no*100:.0f}%, NO2={r_no2*100:.0f}%, SO2={r_so2*100:.1f}% (SO2<5%)")
            else:
                valid_combinations.append((r_no, r_no2, r_so2))
    
    print(f"\n约束条件: SO2浓度 ≥ 5%")
    print(f"有效组合: {len(valid_combinations)} 种")
    print(f"无效组合: {len(invalid_combinations)} 种")
    
    # 3. 数据增强演示
    print("\n3. 数据增强演示")
    print("-" * 40)
    
    # 选择一个具体的浓度组合进行演示
    demo_combo = valid_combinations[0]  # 取第一个有效组合
    r_no, r_no2, r_so2 = demo_combo
    
    print(f"演示组合: NO={r_no*100:.0f}%, NO2={r_no2*100:.0f}%, SO2={r_so2*100:.0f}%")
    
    # 模拟光谱数据（简化版）
    np.random.seed(42)  # 固定随机种子便于演示
    wavenumbers = np.arange(1000, 1100, 10)  # 简化的波数范围
    
    # 创建简化的单气体光谱
    no_spectrum = np.exp(-(wavenumbers - 1020)**2 / 100)   # NO在1020附近有峰
    no2_spectrum = np.exp(-(wavenumbers - 1040)**2 / 100)  # NO2在1040附近有峰  
    so2_spectrum = np.exp(-(wavenumbers - 1060)**2 / 100)  # SO2在1060附近有峰
    
    # 理想混合光谱（无噪声）
    ideal_mixed = r_no * no_spectrum + r_no2 * no2_spectrum + r_so2 * so2_spectrum
    
    print(f"\n生成10个噪声样本 (噪声水平=1%):")
    
    samples_per_ratio = 10
    noise_level = 0.01  # 1%噪声
    
    noisy_samples = []
    for i in range(samples_per_ratio):
        # 生成高斯噪声
        noise = np.random.normal(0, noise_level, size=ideal_mixed.shape)
        # 添加乘性噪声 (相对噪声)
        noisy_sample = ideal_mixed * (1 + noise)
        noisy_samples.append(noisy_sample)
        
        # 显示前3个样本的统计信息
        if i < 3:
            noise_magnitude = np.std(noise) * 100
            signal_change = np.std(noisy_sample - ideal_mixed) / np.mean(ideal_mixed) * 100
            print(f"  样本{i+1}: 噪声强度={noise_magnitude:.2f}%, 信号变化={signal_change:.2f}%")
    
    # 4. 最终数据集结构
    print("\n4. 最终数据集结构")
    print("-" * 40)
    
    total_samples = len(valid_combinations) * samples_per_ratio
    print(f"有效浓度组合数: {len(valid_combinations)}")
    print(f"每种组合生成样本数: {samples_per_ratio}")
    print(f"最终训练样本总数: {len(valid_combinations)} × {samples_per_ratio} = {total_samples}")
    
    # 5. 数据结构说明
    print("\n5. 数据结构说明")
    print("-" * 40)
    
    print("X_dataset.csv (特征矩阵):")
    print(f"  - 行数: {total_samples} (每行一个光谱样本)")
    print(f"  - 列数: 波数点个数 (每列一个波数的强度值)")
    print(f"  - 列名格式: '1000cm-1', '1001cm-1', ..., '2000cm-1'")
    
    print("\nY_labels.csv (标签矩阵):")
    print(f"  - 行数: {total_samples} (与X_dataset对应)")
    print(f"  - 列数: 3 ('NO_conc', 'NO2_conc', 'SO2_conc')")
    print(f"  - 每行浓度之和: 1.0 (100%)")
    
    # 6. 实际数据验证
    print("\n6. 实际数据验证")
    print("-" * 40)
    
    try:
        # 读取实际生成的数据
        Y_actual = pd.read_csv("data/processed/Y_labels.csv")
        print(f"实际Y_labels.csv:")
        print(f"  - 样本数: {len(Y_actual)}")
        print(f"  - 唯一浓度组合数: {len(Y_actual.drop_duplicates())}")
        print(f"  - 每种组合重复次数: {len(Y_actual) // len(Y_actual.drop_duplicates())}")
        
        # 验证浓度约束
        concentration_sums = Y_actual.sum(axis=1)
        min_concentrations = Y_actual.min()
        
        print(f"\n约束验证:")
        print(f"  - 浓度和范围: {concentration_sums.min():.6f} - {concentration_sums.max():.6f}")
        print(f"  - 最小浓度: NO={min_concentrations['NO_conc']:.3f}, NO2={min_concentrations['NO2_conc']:.3f}, SO2={min_concentrations['SO2_conc']:.3f}")
        print(f"  - SO2最小值 ≥ 0.05? {min_concentrations['SO2_conc'] >= 0.05}")
        
    except FileNotFoundError:
        print("  实际数据文件未找到，请先运行03_generate_dataset.py")
    
    print("\n" + "=" * 80)
    print("概念解释完成！")
    print("=" * 80)

def create_visualization():
    """创建可视化图表帮助理解"""
    
    print("\n正在生成可视化图表...")
    
    # 创建噪声效果演示图
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    ideal_signal = np.sin(x) + 0.5 * np.sin(3*x)
    
    plt.figure(figsize=(12, 8))
    
    # 子图1: 理想信号vs噪声信号
    plt.subplot(2, 2, 1)
    plt.plot(x, ideal_signal, 'b-', linewidth=2, label='理想信号')
    
    # 生成3个噪声样本
    colors = ['r--', 'g--', 'm--']
    for i in range(3):
        noise = np.random.normal(0, 0.01, size=ideal_signal.shape)
        noisy_signal = ideal_signal * (1 + noise)
        plt.plot(x, noisy_signal, colors[i], alpha=0.7, label=f'噪声样本{i+1}')
    
    plt.title('1%噪声对光谱信号的影响')
    plt.xlabel('波数')
    plt.ylabel('强度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 浓度组合分布
    plt.subplot(2, 2, 2)
    no_ratios = np.arange(0.2, 0.45, 0.05)
    no2_ratios = np.arange(0.3, 0.55, 0.05)
    
    valid_no = []
    valid_no2 = []
    valid_so2 = []
    
    for r_no in no_ratios:
        for r_no2 in no2_ratios:
            r_so2 = 1.0 - r_no - r_no2
            if r_so2 >= 0.05:
                valid_no.append(r_no * 100)
                valid_no2.append(r_no2 * 100)
                valid_so2.append(r_so2 * 100)
    
    scatter = plt.scatter(valid_no, valid_no2, c=valid_so2, cmap='viridis', s=100)
    plt.colorbar(scatter, label='SO2浓度 (%)')
    plt.xlabel('NO浓度 (%)')
    plt.ylabel('NO2浓度 (%)')
    plt.title('有效浓度组合分布')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 数据增强示意图
    plt.subplot(2, 2, 3)
    base_combinations = len(valid_no)
    samples_per_combo = 10
    total_samples = base_combinations * samples_per_combo
    
    categories = ['基础组合', '增强后样本']
    values = [base_combinations, total_samples]
    bars = plt.bar(categories, values, color=['skyblue', 'lightcoral'])
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(value), ha='center', va='bottom', fontweight='bold')
    
    plt.title('数据增强效果')
    plt.ylabel('样本数量')
    
    # 子图4: 约束条件示意图
    plt.subplot(2, 2, 4)
    
    # 创建所有可能的组合（包括无效的）
    all_no = []
    all_no2 = []
    all_so2 = []
    all_valid = []
    
    for r_no in no_ratios:
        for r_no2 in no2_ratios:
            r_so2 = 1.0 - r_no - r_no2
            all_no.append(r_no * 100)
            all_no2.append(r_no2 * 100) 
            all_so2.append(r_so2 * 100)
            all_valid.append(r_so2 >= 0.05)
    
    # 分别绘制有效和无效组合
    valid_mask = np.array(all_valid)
    plt.scatter(np.array(all_no)[valid_mask], np.array(all_so2)[valid_mask], 
               c='green', label='有效组合', s=100, alpha=0.7)
    plt.scatter(np.array(all_no)[~valid_mask], np.array(all_so2)[~valid_mask], 
               c='red', label='无效组合', s=100, alpha=0.7, marker='x')
    
    plt.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='SO2最低限制(5%)')
    plt.xlabel('NO浓度 (%)')
    plt.ylabel('SO2浓度 (%)')
    plt.title('约束条件效果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/figures/data_augmentation_explanation.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存到: data/figures/data_augmentation_explanation.png")
    
    plt.show()

if __name__ == "__main__":
    # 确保目录存在
    import os
    os.makedirs("data/figures", exist_ok=True)
    
    # 执行解释
    explain_concepts()
    
    # 创建可视化图表
    create_visualization()