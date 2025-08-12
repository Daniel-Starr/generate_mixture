# data_splitter.py
# 智能数据分割模块 - 避免数据泄露，确保模型评估的可靠性

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import json
import os

class IntelligentDataSplitter:
    """
    智能数据分割器，确保：
    1. 浓度组合不重叠（避免数据泄露）
    2. 训练测试集分布相似
    3. 支持多种分割策略
    """
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.split_info = {}
        
    def concentration_based_split(self, X: pd.DataFrame, Y: pd.DataFrame, 
                                strategy: str = 'grid') -> Tuple[pd.DataFrame, ...]:
        """
        基于浓度组合的数据分割
        
        Parameters:
        - strategy: 'grid' (网格分割) 或 'random' (随机分割)
        """
        print("🔄 执行智能数据分割...")
        
        # 创建浓度组合标识符
        Y_copy = Y.copy()
        Y_copy['combination_id'] = Y_copy.groupby(['NO_conc', 'NO2_conc', 'SO2_conc']).ngroup()
        
        unique_combinations = Y_copy['combination_id'].unique()
        n_unique = len(unique_combinations)
        
        print(f"   📊 总计 {n_unique} 个唯一浓度组合")
        
        if strategy == 'grid':
            # 网格策略：确保训练测试集在浓度空间中均匀分布
            test_combinations = self._grid_based_selection(Y_copy, unique_combinations)
        else:
            # 随机策略：随机选择测试组合
            np.random.seed(self.random_state)
            n_test = max(1, int(n_unique * self.test_size))
            test_combinations = np.random.choice(unique_combinations, size=n_test, replace=False)
        
        # 验证集组合（从训练集中选择）
        remaining_combinations = [c for c in unique_combinations if c not in test_combinations]
        n_val = max(1, int(len(remaining_combinations) * self.val_size / (1 - self.test_size)))
        
        np.random.seed(self.random_state + 1)
        val_combinations = np.random.choice(remaining_combinations, size=min(n_val, len(remaining_combinations)), 
                                          replace=False)
        
        train_combinations = [c for c in remaining_combinations if c not in val_combinations]
        
        # 创建数据分割
        train_mask = Y_copy['combination_id'].isin(train_combinations)
        val_mask = Y_copy['combination_id'].isin(val_combinations)
        test_mask = Y_copy['combination_id'].isin(test_combinations)
        
        X_train = X[train_mask].reset_index(drop=True)
        X_val = X[val_mask].reset_index(drop=True)
        X_test = X[test_mask].reset_index(drop=True)
        
        Y_train = Y[train_mask].reset_index(drop=True)
        Y_val = Y[val_mask].reset_index(drop=True)
        Y_test = Y[test_mask].reset_index(drop=True)
        
        # 记录分割信息
        self.split_info = {
            'strategy': strategy,
            'train_combinations': len(train_combinations),
            'val_combinations': len(val_combinations),
            'test_combinations': len(test_combinations),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'test_combination_ids': test_combinations.tolist(),
            'val_combination_ids': val_combinations.tolist(),
            'train_combination_ids': train_combinations
        }
        
        self._print_split_summary()
        self._verify_no_overlap(Y_train, Y_val, Y_test)
        
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
    def _grid_based_selection(self, Y_copy: pd.DataFrame, unique_combinations: np.ndarray) -> np.ndarray:
        """网格策略：在浓度空间中均匀选择测试样本"""
        
        # 获取每个组合的代表性浓度
        combination_centers = Y_copy.groupby('combination_id')[['NO_conc', 'NO2_conc', 'SO2_conc']].first()
        
        # 使用K-means聚类选择代表性测试组合
        from sklearn.cluster import KMeans
        
        n_test_clusters = max(1, int(len(unique_combinations) * self.test_size))
        kmeans = KMeans(n_clusters=n_test_clusters, random_state=self.random_state, n_init=10)
        
        # 聚类浓度空间
        concentration_coords = combination_centers[['NO_conc', 'NO2_conc', 'SO2_conc']].values
        cluster_labels = kmeans.fit_predict(concentration_coords)
        
        # 从每个聚类中选择最接近中心的组合作为测试集
        test_combinations = []
        for cluster_id in range(n_test_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_combinations = combination_centers[cluster_mask]
            
            if len(cluster_combinations) > 0:
                cluster_center = kmeans.cluster_centers_[cluster_id]
                # 计算到聚类中心的距离
                distances = np.sum((cluster_combinations[['NO_conc', 'NO2_conc', 'SO2_conc']].values - cluster_center) ** 2, axis=1)
                closest_idx = cluster_combinations.index[np.argmin(distances)]
                test_combinations.append(closest_idx)
        
        return np.array(test_combinations)
    
    def _print_split_summary(self):
        """打印分割摘要"""
        info = self.split_info
        print(f"   ✅ 数据分割完成:")
        print(f"      🚂 训练集: {info['train_combinations']} 组合, {info['train_samples']} 样本")
        print(f"      🔍 验证集: {info['val_combinations']} 组合, {info['val_samples']} 样本")
        print(f"      🧪 测试集: {info['test_combinations']} 组合, {info['test_samples']} 样本")
        print(f"      📊 总体比例: {info['train_samples']/(info['train_samples']+info['val_samples']+info['test_samples']):.1%} / "
              f"{info['val_samples']/(info['train_samples']+info['val_samples']+info['test_samples']):.1%} / "
              f"{info['test_samples']/(info['train_samples']+info['val_samples']+info['test_samples']):.1%}")
    
    def _verify_no_overlap(self, Y_train: pd.DataFrame, Y_val: pd.DataFrame, Y_test: pd.DataFrame):
        """验证数据集间没有浓度组合重叠"""
        
        train_combinations = set(Y_train.groupby(['NO_conc', 'NO2_conc', 'SO2_conc']).ngroup().unique())
        val_combinations = set(Y_val.groupby(['NO_conc', 'NO2_conc', 'SO2_conc']).ngroup().unique())
        test_combinations = set(Y_test.groupby(['NO_conc', 'NO2_conc', 'SO2_conc']).ngroup().unique())
        
        train_val_overlap = train_combinations.intersection(val_combinations)
        train_test_overlap = train_combinations.intersection(test_combinations)
        val_test_overlap = val_combinations.intersection(test_combinations)
        
        total_overlap = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
        
        if total_overlap == 0:
            print(f"   ✅ 验证通过: 无浓度组合重叠")
        else:
            print(f"   ⚠️  警告: 发现 {total_overlap} 个重叠组合")
            
        return total_overlap == 0
    
    def visualize_split(self, Y_train: pd.DataFrame, Y_val: pd.DataFrame, Y_test: pd.DataFrame, 
                       save_path: str = "data/figures/data_split_visualization.png"):
        """可视化数据分割在浓度空间中的分布"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 3D散点图的2D投影
        projections = [
            ('NO_conc', 'NO2_conc', 'NO vs NO2'),
            ('NO_conc', 'SO2_conc', 'NO vs SO2'),
            ('NO2_conc', 'SO2_conc', 'NO2 vs SO2')
        ]
        
        colors = ['blue', 'green', 'red']
        labels = ['Train', 'Validation', 'Test']
        datasets = [Y_train, Y_val, Y_test]
        
        for i, (x_col, y_col, title) in enumerate(projections):
            for dataset, color, label in zip(datasets, colors, labels):
                if len(dataset) > 0:
                    axes[i].scatter(dataset[x_col], dataset[y_col], 
                                  c=color, alpha=0.6, s=50, label=label)
            
            axes[i].set_xlabel(x_col.replace('_conc', ' Concentration'))
            axes[i].set_ylabel(y_col.replace('_conc', ' Concentration'))
            axes[i].set_title(title)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 数据分割可视化已保存到: {save_path}")
    
    def save_split_info(self, save_path: str = "data/processed/split_info.json"):
        """保存分割信息"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 转换numpy数组为列表以便JSON序列化
        save_info = self.split_info.copy()
        for key, value in save_info.items():
            if isinstance(value, np.ndarray):
                save_info[key] = value.tolist()
            elif isinstance(value, np.integer):
                save_info[key] = int(value)
                
        with open(save_path, 'w') as f:
            json.dump(save_info, f, indent=2)
        
        print(f"   💾 分割信息已保存到: {save_path}")

def apply_intelligent_split():
    """应用智能数据分割"""
    
    # 检查数据文件是否存在
    if not os.path.exists("data/processed/X_dataset.csv"):
        print("❌ 数据文件不存在，请先运行 03_generate_dataset.py")
        return None
    
    # 加载数据
    X = pd.read_csv("data/processed/X_dataset.csv")
    Y = pd.read_csv("data/processed/Y_labels.csv")
    
    print(f"📥 加载数据: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    # 创建分割器
    splitter = IntelligentDataSplitter(test_size=0.2, val_size=0.1, random_state=42)
    
    # 执行分割
    X_train, X_val, X_test, Y_train, Y_val, Y_test = splitter.concentration_based_split(
        X, Y, strategy='grid'
    )
    
    # 保存分割后的数据
    data_splits = {
        'X_train': X_train, 'Y_train': Y_train,
        'X_val': X_val, 'Y_val': Y_val,
        'X_test': X_test, 'Y_test': Y_test
    }
    
    for name, data in data_splits.items():
        data.to_csv(f"data/processed/{name}.csv", index=False)
    
    print(f"   💾 分割后数据已保存到 data/processed/")
    
    # 可视化和保存信息
    splitter.visualize_split(Y_train, Y_val, Y_test)
    splitter.save_split_info()
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

if __name__ == "__main__":
    apply_intelligent_split()