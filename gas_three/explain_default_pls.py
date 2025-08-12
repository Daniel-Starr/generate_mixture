# -*- coding: utf-8 -*-
# explain_default_pls.py
# 详细解释默认PLS模型的各个参数和设计

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def explain_default_pls_model():
    """详细解释默认PLS模型的设计和参数"""
    
    print("=" * 80)
    print("默认PLS模型详细参数解释")
    print("=" * 80)
    
    # 1. 算法选择解释
    print("\n1. 算法: 偏最小二乘回归 (PLS Regression)")
    print("-" * 60)
    print("选择理由:")
    print("  - PLS适合高维输入、多输出的回归问题")
    print("  - 光谱数据通常有数百到数千个波数点(高维)")
    print("  - 需要同时预测3种气体浓度(多输出)")
    print("  - PLS能处理输入变量间的多重共线性问题")
    print("  - 相比普通回归，PLS在光谱分析中表现更稳定")
    
    # 2. 主成分数解释
    print("\n2. 主成分数: 5 (固定)")
    print("-" * 60)
    print("参数说明:")
    print("  - n_components=5: 提取5个潜变量(主成分)")
    print("  - 固定值: 不进行优化，直接使用经验值")
    print("  - 选择5的原因:")
    print("    * 通常光谱数据的有效信息集中在前几个主成分")
    print("    * 5个成分足以捕获主要的光谱特征")
    print("    * 避免过拟合(成分数过多)")
    print("    * 保持模型简单快速")
    
    # 3. 数据分割解释
    print("\n3. 数据分割: 80%训练，20%测试")
    print("-" * 60)
    print("分割策略:")
    print("  - test_size=0.2: 20%数据用于测试")
    print("  - random_state=42: 固定随机种子确保结果可重复")
    print("  - 分层采样: 否(因为是回归问题，不是分类)")
    
    # 加载实际数据进行演示
    try:
        X = pd.read_csv("data/processed/X_dataset.csv").values
        Y = pd.read_csv("data/processed/Y_labels.csv").values
        
        print(f"\n实际数据规模:")
        print(f"  - 总样本数: {len(X)}")
        print(f"  - 特征维度: {X.shape[1]} (光谱波数点)")
        print(f"  - 输出维度: {Y.shape[1]} (气体种类)")
        
        # 执行数据分割
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print(f"\n分割结果:")
        print(f"  - 训练集: {len(X_train)} 样本 ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  - 测试集: {len(X_test)} 样本 ({len(X_test)/len(X)*100:.1f}%)")
        
        # 4. 预处理解释
        print("\n4. 预处理: 无额外标准化")
        print("-" * 60)
        print("预处理策略:")
        print("  • sklearn的PLSRegression内部会自动中心化数据")
        print("  • 中心化: X = X - mean(X), Y = Y - mean(Y)")
        print("  • 不进行额外的标准化(如z-score归一化)")
        print("  • 保留原始光谱强度的相对关系")
        
        print(f"\n原始数据统计:")
        print(f"  • X均值范围: {X.mean(axis=0).min():.6f} ~ {X.mean(axis=0).max():.6f}")
        print(f"  • X标准差范围: {X.std(axis=0).min():.6f} ~ {X.std(axis=0).max():.6f}")
        print(f"  • Y均值: {Y.mean(axis=0)}")
        print(f"  • Y标准差: {Y.std(axis=0)}")
        
        # 5. 模型训练和评估
        print("\n5. 模型训练和评估")
        print("-" * 60)
        
        # 创建和训练模型
        pls = PLSRegression(n_components=5)
        pls.fit(X_train, Y_train)
        
        # 预测
        Y_pred = pls.predict(X_test)
        
        # 评估
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        r2 = r2_score(Y_test, Y_pred)
        
        print(f"性能指标:")
        print(f"  • RMSE: {rmse:.5f}")
        print(f"  • R²: {r2:.5f}")
        
        # 各气体分别评估
        print(f"\n各气体单独评估:")
        gas_names = ['NO', 'NO2', 'SO2']
        for i, gas in enumerate(gas_names):
            rmse_gas = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred[:, i]))
            r2_gas = r2_score(Y_test[:, i], Y_pred[:, i])
            print(f"  • {gas}: RMSE={rmse_gas:.5f}, R²={r2_gas:.5f}")
        
        # 6. 模型内部参数分析
        print("\n6. 模型内部参数分析")
        print("-" * 60)
        
        print(f"PLS模型属性:")
        print(f"  • 主成分数: {pls.n_components}")
        print(f"  • X权重矩阵形状: {pls.x_weights_.shape}")
        print(f"  • Y权重矩阵形状: {pls.y_weights_.shape}")
        print(f"  • 回归系数形状: {pls.coef_.shape}")
        print(f"  • 解释的X方差比例: {pls.x_scores_.var(axis=0).sum() / X_train.var().sum():.4f}")
        
        # 7. 与自适应模型的对比
        print("\n7. 与自适应PLS模型的区别")
        print("-" * 60)
        print("默认模型特点:")
        print("  • 固定参数，无优化过程")
        print("  • 训练速度快")
        print("  • 适合快速原型和基线模型")
        print("  • 可能不是最优配置")
        
        print("\n自适应模型特点:")
        print("  • 通过交叉验证选择最优主成分数")
        print("  • 训练时间较长")
        print("  • 性能通常更好")
        print("  • 适合最终部署")
        
        # 8. 使用场景
        print("\n8. 适用场景")
        print("-" * 60)
        print("默认PLS模型适合:")
        print("  • 快速验证数据质量")
        print("  • 建立基线性能")
        print("  • 算法原型开发")
        print("  • 计算资源有限的环境")
        print("  • 对精度要求不是极高的应用")
        
        # 创建可视化
        create_pls_visualization(X_train, X_test, Y_train, Y_test, Y_pred, pls)
        
    except FileNotFoundError:
        print("\n实际数据文件未找到，请先运行数据生成脚本")
    
    print("\n" + "=" * 80)
    print("默认PLS模型解释完成！")
    print("=" * 80)

def create_pls_visualization(X_train, X_test, Y_train, Y_test, Y_pred, pls):
    """创建PLS模型的可视化图表"""
    
    print("\n正在生成默认PLS模型可视化图表...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：预测vs真实值散点图
    gas_names = ['NO', 'NO2', 'SO2']
    colors = ['red', 'blue', 'green']
    
    for i, (gas, color) in enumerate(zip(gas_names, colors)):
        ax = axes[0, i]
        ax.scatter(Y_test[:, i], Y_pred[:, i], c=color, alpha=0.6, s=50)
        
        # 添加理想线 y=x
        min_val = min(Y_test[:, i].min(), Y_pred[:, i].min())
        max_val = max(Y_test[:, i].max(), Y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
        
        # 计算R²
        r2_individual = r2_score(Y_test[:, i], Y_pred[:, i])
        ax.set_xlabel(f'True {gas} Concentration')
        ax.set_ylabel(f'Predicted {gas} Concentration')
        ax.set_title(f'{gas} Prediction (R² = {r2_individual:.4f})')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 第二行：模型分析图
    
    # 子图4：主成分解释方差
    ax = axes[1, 0]
    # 计算每个主成分解释的方差
    X_scores = pls.transform(X_train)
    explained_var = []
    for i in range(pls.n_components):
        var_explained = np.var(X_scores[:, i]) / np.sum(np.var(X_scores, axis=0))
        explained_var.append(var_explained)
    
    bars = ax.bar(range(1, len(explained_var)+1), explained_var, color='skyblue', alpha=0.7)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PLS Components Contribution')
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, var in zip(bars, explained_var):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{var:.3f}', ha='center', va='bottom')
    
    # 子图5：残差分析
    ax = axes[1, 1]
    residuals = Y_test - Y_pred
    residuals_flat = residuals.flatten()
    predicted_flat = Y_pred.flatten()
    
    ax.scatter(predicted_flat, residuals_flat, alpha=0.6, c='purple', s=30)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Predicted')
    ax.grid(True, alpha=0.3)
    
    # 子图6：训练vs测试数据分布
    ax = axes[1, 2]
    
    # 计算训练集和测试集在第一主成分上的投影
    X_train_scores = pls.transform(X_train)[:, 0]
    X_test_scores = pls.transform(X_test)[:, 0]
    
    ax.hist(X_train_scores, bins=20, alpha=0.7, label=f'Training ({len(X_train)} samples)', 
            color='lightblue', density=True)
    ax.hist(X_test_scores, bins=20, alpha=0.7, label=f'Testing ({len(X_test)} samples)', 
            color='lightcoral', density=True)
    
    ax.set_xlabel('First PLS Component Score')
    ax.set_ylabel('Density')
    ax.set_title('Data Distribution on PC1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/figures/default_pls_analysis.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存到: data/figures/default_pls_analysis.png")
    
    # 不显示图表以避免终端问题
    plt.close()

if __name__ == "__main__":
    # 确保目录存在
    import os
    os.makedirs("data/figures", exist_ok=True)
    
    # 执行解释
    explain_default_pls_model()