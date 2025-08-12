# -*- coding: utf-8 -*-
# get_model_performance.py
# 获取模型性能指标

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

def get_default_pls_performance():
    """获取默认PLS模型的完整性能指标"""
    
    # 加载数据
    X = pd.read_csv('data/processed/X_dataset.csv').values
    Y = pd.read_csv('data/processed/Y_labels.csv').values
    
    # 数据分割
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # 测量训练时间
    start_time = time.time()
    
    # 创建和训练PLS模型
    pls = PLSRegression(n_components=5)
    pls.fit(X_train, Y_train)
    
    training_time = time.time() - start_time
    
    # 预测
    Y_pred = pls.predict(X_test)
    
    # 评估
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    r2 = r2_score(Y_test, Y_pred)
    
    return {
        'rmse': rmse,
        'r2': r2,
        'training_time': training_time,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X.shape[1]
    }

def get_custom_pls_performance():
    """获取自适应PLS模型性能指标"""
    try:
        # 读取已保存的结果
        with open('data/results/evaluation_custom.txt', 'r') as f:
            lines = f.readlines()
            
        results = {}
        for line in lines:
            if 'Best n:' in line:
                results['best_components'] = int(line.split(':')[1].strip())
            elif 'RMSE:' in line:
                results['rmse'] = float(line.split(':')[1].strip())
            elif 'R2:' in line:
                results['r2'] = float(line.split(':')[1].strip())
        
        return results
    except:
        return None

if __name__ == "__main__":
    print("模型性能测试结果")
    print("=" * 50)
    
    # 默认PLS模型
    print("\n3.1.1 默认PLS模型表现:")
    default_results = get_default_pls_performance()
    print(f"RMSE: {default_results['rmse']:.5f}")
    print(f"R2: {default_results['r2']:.5f}")
    print(f"训练时间: {default_results['training_time']:.4f} 秒")
    print(f"训练样本: {default_results['train_samples']}")
    print(f"测试样本: {default_results['test_samples']}")
    print(f"特征维度: {default_results['features']}")
    
    # 自适应PLS模型  
    print("\n3.1.2 自适应PLS模型表现:")
    custom_results = get_custom_pls_performance()
    if custom_results:
        print(f"最优主成分数: {custom_results['best_components']}")
        print(f"RMSE: {custom_results['rmse']:.5f}")
        print(f"R2: {custom_results['r2']:.5f}")
    else:
        print("自适应模型结果文件未找到")
    
    # 保存完整结果到文件
    with open('data/results/complete_performance.txt', 'w') as f:
        f.write("模型性能测试完整结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("3.1.1 默认PLS模型表现:\n")
        f.write(f"RMSE: {default_results['rmse']:.5f}\n")
        f.write(f"R2: {default_results['r2']:.5f}\n")
        f.write(f"训练时间: {default_results['training_time']:.4f} 秒\n")
        f.write(f"训练样本: {default_results['train_samples']}\n")
        f.write(f"测试样本: {default_results['test_samples']}\n")
        f.write(f"特征维度: {default_results['features']}\n\n")
        
        if custom_results:
            f.write("3.1.2 自适应PLS模型表现:\n")
            f.write(f"最优主成分数: {custom_results['best_components']}\n")
            f.write(f"RMSE: {custom_results['rmse']:.5f}\n")
            f.write(f"R2: {custom_results['r2']:.5f}\n")
    
    print(f"\n完整结果已保存到: data/results/complete_performance.txt")