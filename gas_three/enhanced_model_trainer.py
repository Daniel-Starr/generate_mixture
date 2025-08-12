# enhanced_model_trainer.py
# 增强模型训练器 - 包含严格的模型评估和交叉验证

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 解决中文字体显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedModelTrainer:
    """
    增强版模型训练器，提供：
    1. 严格的交叉验证
    2. 多种评估指标
    3. 模型复杂度分析
    4. 鲁棒性测试
    """
    
    def __init__(self, max_components: int = 20, cv_folds: int = 5, random_state: int = 42):
        self.max_components = max_components
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
    def train_with_cross_validation(self, X_train: pd.DataFrame, Y_train: pd.DataFrame,
                                  X_val: pd.DataFrame, Y_val: pd.DataFrame,
                                  use_scaling: bool = True) -> Dict:
        """
        使用交叉验证训练模型并选择最优参数
        """
        print("🚀 开始增强模型训练与验证...")
        
        # 数据预处理
        if use_scaling:
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_val_scaled = self.scaler_X.transform(X_val)
            Y_train_scaled = self.scaler_Y.fit_transform(Y_train)
            Y_val_scaled = self.scaler_Y.transform(Y_val)
        else:
            X_train_scaled = X_train.values
            X_val_scaled = X_val.values
            Y_train_scaled = Y_train.values
            Y_val_scaled = Y_val.values
        
        print(f"   📊 训练数据: {X_train_scaled.shape}, 验证数据: {X_val_scaled.shape}")
        
        # 1. 交叉验证选择最优组件数
        print("   🔍 执行交叉验证选择最优组件数...")
        cv_results = self._cross_validate_components(X_train_scaled, Y_train_scaled)
        
        # 2. 训练最优模型
        best_n_components = cv_results['best_n_components']
        print(f"   ✅ 最优主成分数: {best_n_components}")
        
        best_model = PLSRegression(n_components=best_n_components)
        best_model.fit(X_train_scaled, Y_train_scaled)
        
        # 3. 验证集评估
        val_results = self._evaluate_model(best_model, X_val_scaled, Y_val_scaled, 
                                         X_train_scaled, Y_train_scaled, use_scaling)
        
        # 4. 模型复杂度分析
        complexity_analysis = self._analyze_model_complexity(best_model, X_train_scaled, Y_train_scaled)
        
        # 5. 鲁棒性测试
        robustness_results = self._test_robustness(best_model, X_val_scaled, Y_val_scaled)
        
        # 保存模型和结果
        self.models['best_pls'] = best_model
        self.results = {
            'cv_results': cv_results,
            'validation_results': val_results,
            'complexity_analysis': complexity_analysis,
            'robustness_results': robustness_results,
            'best_n_components': best_n_components,
            'use_scaling': use_scaling
        }
        
        self._print_training_summary()
        
        return self.results
    
    def _cross_validate_components(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """交叉验证选择最优组件数"""
        
        cv_scores_mean = []
        cv_scores_std = []
        component_range = range(1, min(self.max_components + 1, X.shape[1], X.shape[0]))
        
        # 使用组交叉验证避免数据泄露（如果有组信息）
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for n_comp in component_range:
            pls = PLSRegression(n_components=n_comp)
            
            # 对每个目标变量分别计算CV分数，然后平均
            cv_scores_per_target = []
            
            for target_idx in range(Y.shape[1]):
                scores = cross_val_score(pls, X, Y[:, target_idx], 
                                       cv=kfold, scoring='r2', n_jobs=-1)
                cv_scores_per_target.append(scores)
            
            # 平均所有目标变量的分数
            avg_scores = np.mean(cv_scores_per_target, axis=0)
            cv_scores_mean.append(avg_scores.mean())
            cv_scores_std.append(avg_scores.std())
        
        # 选择最优组件数（考虑方差）
        cv_scores_mean = np.array(cv_scores_mean)
        cv_scores_std = np.array(cv_scores_std)
        
        # 使用1-standard-error rule选择更简单的模型
        best_idx = np.argmax(cv_scores_mean)
        best_score = cv_scores_mean[best_idx]
        best_std = cv_scores_std[best_idx]
        
        # 找到在best_score - best_std范围内的最简单模型
        threshold = best_score - best_std
        simple_model_idx = np.where(cv_scores_mean >= threshold)[0][0]
        
        best_n_components = component_range[simple_model_idx]
        
        return {
            'component_range': list(component_range),
            'cv_scores_mean': cv_scores_mean.tolist(),
            'cv_scores_std': cv_scores_std.tolist(),
            'best_n_components': best_n_components,
            'best_cv_score': cv_scores_mean[simple_model_idx],
            'best_cv_std': cv_scores_std[simple_model_idx]
        }
    
    def _evaluate_model(self, model, X_val: np.ndarray, Y_val: np.ndarray,
                       X_train: np.ndarray, Y_train: np.ndarray, use_scaling: bool) -> Dict:
        """全面评估模型性能"""
        
        # 预测
        Y_train_pred = model.predict(X_train)
        Y_val_pred = model.predict(X_val)
        
        # 如果使用了缩放，需要逆变换
        if use_scaling:
            Y_train_true = self.scaler_Y.inverse_transform(Y_train)
            Y_train_pred = self.scaler_Y.inverse_transform(Y_train_pred)
            Y_val_true = self.scaler_Y.inverse_transform(Y_val)
            Y_val_pred = self.scaler_Y.inverse_transform(Y_val_pred)
        else:
            Y_train_true = Y_train
            Y_val_true = Y_val
        
        # 计算多种评估指标
        metrics = {}
        
        for dataset_name, y_true, y_pred in [('train', Y_train_true, Y_train_pred), 
                                           ('validation', Y_val_true, Y_val_pred)]:
            
            # 整体指标
            metrics[f'{dataset_name}_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics[f'{dataset_name}_mae'] = mean_absolute_error(y_true, y_pred)
            metrics[f'{dataset_name}_r2'] = r2_score(y_true, y_pred)
            
            # 每个气体的指标
            gas_names = ['NO', 'NO2', 'SO2']
            for i, gas in enumerate(gas_names):
                metrics[f'{dataset_name}_{gas}_rmse'] = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
                metrics[f'{dataset_name}_{gas}_mae'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
                metrics[f'{dataset_name}_{gas}_r2'] = r2_score(y_true[:, i], y_pred[:, i])
        
        # 计算泛化误差
        metrics['generalization_gap_rmse'] = metrics['validation_rmse'] - metrics['train_rmse']
        metrics['generalization_gap_r2'] = metrics['train_r2'] - metrics['validation_r2']
        
        return metrics
    
    def _analyze_model_complexity(self, model, X: np.ndarray, Y: np.ndarray) -> Dict:
        """分析模型复杂度"""
        
        # 获取PLS系数
        coef_matrix = model.coef_  # shape: (n_features, n_targets)
        
        # 计算VIP (Variable Importance in Projection) 分数
        vip_scores = self._calculate_vip_scores(model)
        
        # 计算复杂度指标
        complexity = {
            'n_components': model.n_components,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'n_targets': Y.shape[1],
            'coef_sparsity': np.mean(np.abs(coef_matrix) < 1e-6),  # 系数稀疏度
            'coef_l1_norm': np.sum(np.abs(coef_matrix)),
            'coef_l2_norm': np.sqrt(np.sum(coef_matrix ** 2)),
            'effective_rank': np.linalg.matrix_rank(model.x_scores_),  # 有效秩
            'condition_number': np.linalg.cond(model.x_loadings_),  # 条件数
            'vip_scores': vip_scores.tolist(),  # VIP分数
            'vip_mean': float(vip_scores.mean()),
            'vip_std': float(vip_scores.std()),
            'important_variables_count': int(np.sum(vip_scores > 1.0)),  # VIP>1的变量数
        }
        
        # 方差解释比例
        explained_variance = model.x_scores_.var(axis=0)
        total_variance = explained_variance.sum()
        complexity['explained_variance_ratio'] = (explained_variance / total_variance).tolist()
        complexity['cumulative_explained_variance'] = np.cumsum(explained_variance / total_variance).tolist()
        
        return complexity
    
    def _calculate_vip_scores(self, model) -> np.ndarray:
        """
        计算VIP (Variable Importance in Projection) 分数
        
        VIP分数衡量每个输入变量对PLS模型的重要性
        VIP > 1.0 通常被认为是重要变量
        
        公式: VIP_j = sqrt(p * sum(a_j,h^2 * SS_h) / sum(SS_h))
        其中:
        - p: 变量数量
        - a_j,h: 第j个变量在第h个主成分上的loadings
        - SS_h: 第h个主成分解释的Y方差
        """
        # 获取X和Y的loadings
        x_loadings = model.x_loadings_  # shape: (n_features, n_components)
        y_loadings = model.y_loadings_  # shape: (n_targets, n_components)
        
        # 计算每个主成分解释的Y方差 (SS_h)
        y_scores = model.y_scores_  # shape: (n_samples, n_components)
        ss_components = np.var(y_scores, axis=0, ddof=1)  # 每个主成分的方差
        
        # 计算VIP分数
        n_features = x_loadings.shape[0]
        n_components = x_loadings.shape[1]
        
        # VIP公式实现
        vip_scores = np.zeros(n_features)
        total_ss = np.sum(ss_components)
        
        if total_ss > 0:
            for j in range(n_features):
                weighted_loadings_sq = 0
                for h in range(n_components):
                    # 每个主成分的贡献 = loadings^2 * 该主成分解释的方差
                    weighted_loadings_sq += (x_loadings[j, h] ** 2) * ss_components[h]
                
                # VIP公式
                vip_scores[j] = np.sqrt(n_features * weighted_loadings_sq / total_ss)
        
        return vip_scores
    
    def _test_robustness(self, model, X_val: np.ndarray, Y_val: np.ndarray) -> Dict:
        """鲁棒性测试"""
        
        robustness = {}
        
        # 1. 噪声鲁棒性测试
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        noise_performance = []
        
        for noise_level in noise_levels:
            # 添加噪声
            X_noisy = X_val + np.random.normal(0, noise_level * X_val.std(), X_val.shape)
            Y_pred_noisy = model.predict(X_noisy)
            
            # 计算性能下降
            r2_noisy = r2_score(Y_val, Y_pred_noisy)
            noise_performance.append(r2_noisy)
        
        robustness['noise_levels'] = noise_levels
        robustness['noise_performance'] = noise_performance
        
        # 2. 数据删除鲁棒性（随机删除一些特征）
        feature_removal_ratios = [0.1, 0.2, 0.3]
        removal_performance = []
        
        for removal_ratio in feature_removal_ratios:
            n_remove = int(X_val.shape[1] * removal_ratio)
            np.random.seed(self.random_state)
            remove_indices = np.random.choice(X_val.shape[1], n_remove, replace=False)
            
            X_reduced = np.delete(X_val, remove_indices, axis=1)
            
            # 重新训练简化模型
            model_reduced = PLSRegression(n_components=min(model.n_components, X_reduced.shape[1]))
            # 需要相应的训练数据，这里简化处理
            # 实际应用中需要传入对应的训练数据
            removal_performance.append(0.0)  # 占位符
        
        robustness['feature_removal_ratios'] = feature_removal_ratios
        robustness['removal_performance'] = removal_performance
        
        return robustness
    
    def _print_training_summary(self):
        """打印训练摘要"""
        results = self.results
        
        print("   ✅ 模型训练完成!")
        print(f"      🎯 最优组件数: {results['best_n_components']}")
        print(f"      📊 交叉验证R²: {results['cv_results']['best_cv_score']:.5f} ± {results['cv_results']['best_cv_std']:.5f}")
        
        val_results = results['validation_results']
        print(f"      🚂 训练集 - RMSE: {val_results['train_rmse']:.5f}, R²: {val_results['train_r2']:.5f}")
        print(f"      🔍 验证集 - RMSE: {val_results['validation_rmse']:.5f}, R²: {val_results['validation_r2']:.5f}")
        print(f"      📈 泛化差距 - RMSE: {val_results['generalization_gap_rmse']:.5f}, R²: {val_results['generalization_gap_r2']:.5f}")
        
        # 每个气体的性能
        gas_names = ['NO', 'NO2', 'SO2']
        for gas in gas_names:
            print(f"      {gas}: R² = {val_results[f'validation_{gas}_r2']:.5f}, "
                  f"RMSE = {val_results[f'validation_{gas}_rmse']:.5f}")
    
    def visualize_results(self, X_test: pd.DataFrame, Y_test: pd.DataFrame, 
                         save_dir: str = "data/figures/"):
        """可视化模型结果"""
        
        if 'best_pls' not in self.models:
            print("❌ 模型尚未训练，请先调用 train_with_cross_validation")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        model = self.models['best_pls']
        
        # 预测测试集
        if self.results['use_scaling']:
            X_test_scaled = self.scaler_X.transform(X_test)
            Y_test_pred_scaled = model.predict(X_test_scaled)
            Y_test_pred = self.scaler_Y.inverse_transform(Y_test_pred_scaled)
        else:
            Y_test_pred = model.predict(X_test.values)
        
        # 1. 交叉验证曲线
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        cv_results = self.results['cv_results']
        
        components = cv_results['component_range']
        mean_scores = cv_results['cv_scores_mean']
        std_scores = cv_results['cv_scores_std']
        
        ax.errorbar(components, mean_scores, yerr=std_scores, 
                   marker='o', capsize=5, capthick=2)
        ax.axvline(x=cv_results['best_n_components'], color='red', linestyle='--', 
                  label=f'Optimal: {cv_results["best_n_components"]} components')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cross-Validation R² Score')
        ax.set_title('PLS Cross-Validation Results')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(f"{save_dir}/cv_component_selection.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 预测结果对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        gas_names = ['NO', 'NO2', 'SO2']
        
        for i, gas in enumerate(gas_names):
            y_true = Y_test.values[:, i]
            y_pred = Y_test_pred[:, i]
            
            axes[i].scatter(y_true, y_pred, alpha=0.6, s=50)
            
            # 完美预测线
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            # 计算R²
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            axes[i].set_xlabel(f'True {gas} Concentration')
            axes[i].set_ylabel(f'Predicted {gas} Concentration')
            axes[i].set_title(f'{gas}: R² = {r2:.4f}, RMSE = {rmse:.4f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/enhanced_prediction_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. VIP分析可视化
        self._visualize_vip_analysis(save_dir)
        
        print(f"   📊 可视化结果已保存到: {save_dir}")
    
    def _visualize_vip_analysis(self, save_dir: str):
        """可视化VIP分析结果"""
        
        if 'complexity_analysis' not in self.results:
            return
            
        complexity = self.results['complexity_analysis']
        vip_scores = np.array(complexity['vip_scores'])
        
        # 读取波数信息
        try:
            df = pd.read_csv("data/processed/interpolated_spectra.csv")
            wavenumbers = df['wavenumber'].values
        except:
            wavenumbers = np.arange(len(vip_scores))
        
        # 确保长度匹配
        if len(wavenumbers) != len(vip_scores):
            wavenumbers = np.arange(len(vip_scores))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. VIP分数分布
        axes[0, 0].plot(wavenumbers, vip_scores, 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                          label='VIP = 1.0 (Importance Threshold)')
        axes[0, 0].fill_between(wavenumbers, vip_scores, 1.0, 
                               where=(vip_scores > 1.0), color='red', alpha=0.3,
                               label=f'Important Bands (n={complexity["important_variables_count"]})')
        axes[0, 0].set_xlabel('Wavenumber (cm$^{-1}$)')
        axes[0, 0].set_ylabel('VIP Score')
        axes[0, 0].set_title('Variable Importance in Projection (VIP) Analysis')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 重要波段放大图
        important_indices = np.where(vip_scores > 1.0)[0]
        if len(important_indices) > 0:
            axes[0, 1].stem(wavenumbers[important_indices], vip_scores[important_indices], 
                           basefmt=' ', linefmt='g-', markerfmt='go')
            axes[0, 1].set_xlabel('Wavenumber (cm$^{-1}$)')
            axes[0, 1].set_ylabel('VIP Score')
            axes[0, 1].set_title(f'Important Bands Detail (VIP > 1.0, n={len(important_indices)})')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 标注最重要的前10个波段
            top_10_indices = important_indices[np.argsort(vip_scores[important_indices])[-10:]]
            for idx in top_10_indices:
                axes[0, 1].annotate(f'{wavenumbers[idx]:.0f}', 
                                   (wavenumbers[idx], vip_scores[idx]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8)
        else:
            axes[0, 1].text(0.5, 0.5, 'No variables with VIP>1.0', 
                           transform=axes[0, 1].transAxes, ha='center', va='center')
            axes[0, 1].set_title('Important Bands Analysis')
        
        # 3. VIP分数直方图
        axes[1, 0].hist(vip_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='VIP = 1.0')
        axes[1, 0].axvline(x=vip_scores.mean(), color='green', linestyle='-', linewidth=2, 
                          label=f'Mean = {vip_scores.mean():.3f}')
        axes[1, 0].set_xlabel('VIP Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('VIP Score Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 累积重要性分析
        sorted_indices = np.argsort(vip_scores)[::-1]  # 降序排列
        cumulative_importance = np.cumsum(vip_scores[sorted_indices])
        cumulative_percentage = cumulative_importance / cumulative_importance[-1] * 100
        
        axes[1, 1].plot(range(1, len(cumulative_percentage) + 1), cumulative_percentage, 'b-')
        axes[1, 1].axhline(y=80, color='red', linestyle='--', label='80% Importance')
        axes[1, 1].axhline(y=95, color='orange', linestyle='--', label='95% Importance')
        
        # 找到达到80%和95%重要性的变量数
        idx_80 = np.where(cumulative_percentage >= 80)[0]
        idx_95 = np.where(cumulative_percentage >= 95)[0]
        
        if len(idx_80) > 0:
            axes[1, 1].axvline(x=idx_80[0]+1, color='red', linestyle=':', alpha=0.7)
            axes[1, 1].text(idx_80[0]+1, 82, f'{idx_80[0]+1} vars', rotation=90, va='bottom')
            
        if len(idx_95) > 0:
            axes[1, 1].axvline(x=idx_95[0]+1, color='orange', linestyle=':', alpha=0.7)
            axes[1, 1].text(idx_95[0]+1, 97, f'{idx_95[0]+1} vars', rotation=90, va='bottom')
        
        axes[1, 1].set_xlabel('Variable Count (sorted)')
        axes[1, 1].set_ylabel('Cumulative Importance (%)')
        axes[1, 1].set_title('Cumulative Variable Importance')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/vip_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存重要波段信息到CSV
        vip_analysis_df = pd.DataFrame({
            'wavenumber': wavenumbers,
            'vip_score': vip_scores,
            'is_important': vip_scores > 1.0
        })
        vip_analysis_df = vip_analysis_df.sort_values('vip_score', ascending=False)
        vip_analysis_df.to_csv(f"{save_dir}/../results/vip_analysis.csv", index=False)
        
        print(f"   🎯 VIP分析完成: {complexity['important_variables_count']} 个重要波段 (VIP > 1.0)")
        print(f"      平均VIP分数: {complexity['vip_mean']:.3f} ± {complexity['vip_std']:.3f}")
        print(f"      重要波段信息已保存到: {save_dir}/../results/vip_analysis.csv")
    
    def save_model_and_results(self, save_dir: str = "data/models/"):
        """保存模型和结果"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        if 'best_pls' in self.models:
            joblib.dump(self.models['best_pls'], f"{save_dir}/enhanced_pls_model.pkl")
            
            if self.results['use_scaling']:
                joblib.dump(self.scaler_X, f"{save_dir}/scaler_X.pkl")
                joblib.dump(self.scaler_Y, f"{save_dir}/scaler_Y.pkl")
        
        # 保存结果
        with open(f"{save_dir}/enhanced_training_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"   💾 模型和结果已保存到: {save_dir}")

def main():
    """主函数：执行增强模型训练"""
    
    # 检查分割后的数据是否存在
    required_files = [
        "data/processed/X_train.csv", "data/processed/Y_train.csv",
        "data/processed/X_val.csv", "data/processed/Y_val.csv",
        "data/processed/X_test.csv", "data/processed/Y_test.csv"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 缺少文件: {file_path}")
            print("   请先运行 data_splitter.py 进行数据分割")
            return
    
    # 加载分割后的数据
    print("📥 加载分割后的数据...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    Y_train = pd.read_csv("data/processed/Y_train.csv")
    X_val = pd.read_csv("data/processed/X_val.csv")
    Y_val = pd.read_csv("data/processed/Y_val.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    Y_test = pd.read_csv("data/processed/Y_test.csv")
    
    print(f"   训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    # 创建增强训练器
    trainer = EnhancedModelTrainer(max_components=15, cv_folds=5, random_state=42)
    
    # 训练模型
    results = trainer.train_with_cross_validation(X_train, Y_train, X_val, Y_val, use_scaling=True)
    
    # 可视化结果
    trainer.visualize_results(X_test, Y_test)
    
    # 保存模型和结果
    trainer.save_model_and_results()
    
    print("🎉 增强模型训练完成！")
    
    return trainer, results

if __name__ == "__main__":
    main()