# cross_validation_analysis.py
# 详细分析和可视化5折交叉验证过程

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CrossValidationAnalyzer:
    def __init__(self, X, Y, cv_folds=5):
        self.X = X
        self.Y = Y
        self.cv_folds = cv_folds
        self.cv_results = {}

    def detailed_cross_validation(self, n_components_range=range(1, 11)):
        """详细的交叉验证分析"""
        print("🔍 详细交叉验证分析")
        print("=" * 60)

        # 创建K折分割器
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        results = {}
        detailed_fold_results = {}

        for n_comp in n_components_range:
            print(f"\n📊 测试 {n_comp} 个主成分:")

            pls = PLSRegression(n_components=n_comp)

            # 存储每折的详细结果
            fold_scores = []
            fold_details = []

            fold_idx = 1
            for train_idx, val_idx in kf.split(self.X):
                # 分割数据
                X_train_fold = self.X[train_idx]
                X_val_fold = self.X[val_idx]
                Y_train_fold = self.Y[train_idx]
                Y_val_fold = self.Y[val_idx]

                # 训练和预测
                pls.fit(X_train_fold, Y_train_fold)
                Y_pred_fold = pls.predict(X_val_fold)

                # 计算性能指标
                r2_fold = r2_score(Y_val_fold, Y_pred_fold)
                rmse_fold = np.sqrt(mean_squared_error(Y_val_fold, Y_pred_fold))

                fold_scores.append(r2_fold)
                fold_details.append({
                    'fold': fold_idx,
                    'train_samples': len(train_idx),
                    'val_samples': len(val_idx),
                    'r2': r2_fold,
                    'rmse': rmse_fold
                })

                print(f"  Fold {fold_idx}: R²={r2_fold:.4f}, RMSE={rmse_fold:.5f}, "
                      f"Train/Val: {len(train_idx)}/{len(val_idx)}")

                fold_idx += 1

            # 计算统计信息
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            results[n_comp] = {
                'mean_r2': mean_score,
                'std_r2': std_score,
                'fold_scores': fold_scores,
                'fold_details': fold_details
            }

            print(f"  📈 平均 R²: {mean_score:.4f} ± {std_score:.4f}")

        self.cv_results = results
        return results

    def visualize_cv_results(self):
        """可视化交叉验证结果"""
        if not self.cv_results:
            print("❌ 请先运行 detailed_cross_validation()")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 准备数据
        n_components = list(self.cv_results.keys())
        mean_scores = [self.cv_results[n]['mean_r2'] for n in n_components]
        std_scores = [self.cv_results[n]['std_r2'] for n in n_components]

        # 图1: 交叉验证曲线
        ax1 = axes[0, 0]
        ax1.errorbar(n_components, mean_scores, yerr=std_scores,
                     marker='o', markersize=8, linewidth=2, capsize=5)
        ax1.fill_between(n_components,
                         np.array(mean_scores) - np.array(std_scores),
                         np.array(mean_scores) + np.array(std_scores),
                         alpha=0.3)

        # 标记最优点
        best_idx = np.argmax(mean_scores)
        best_n = n_components[best_idx]
        ax1.scatter(best_n, mean_scores[best_idx], color='red', s=100,
                    marker='*', label=f'Best: n={best_n}')

        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Cross-validation R²')
        ax1.set_title('Cross-validation Performance Curve')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 图2: 不同主成分数的得分分布
        ax2 = axes[0, 1]

        # 准备箱线图数据
        all_scores = []
        all_labels = []
        for n_comp in n_components[::2]:  # 每隔一个显示，避免过密
            scores = self.cv_results[n_comp]['fold_scores']
            all_scores.extend(scores)
            all_labels.extend([f'n={n_comp}'] * len(scores))

        # 创建DataFrame用于seaborn
        df_box = pd.DataFrame({'Components': all_labels, 'R² Score': all_scores})
        sns.boxplot(data=df_box, x='Components', y='R² Score', ax=ax2)
        ax2.set_title('R² Score Distribution by Components')
        ax2.tick_params(axis='x', rotation=45)

        # 图3: 每折详细表现（热力图）
        ax3 = axes[0, 2]

        # 构建热力图数据
        heatmap_data = []
        for n_comp in n_components:
            fold_scores = self.cv_results[n_comp]['fold_scores']
            heatmap_data.append(fold_scores)

        heatmap_data = np.array(heatmap_data)
        im = ax3.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')

        ax3.set_xticks(range(self.cv_folds))
        ax3.set_xticklabels([f'Fold {i + 1}' for i in range(self.cv_folds)])
        ax3.set_yticks(range(len(n_components)))
        ax3.set_yticklabels([f'n={n}' for n in n_components])
        ax3.set_title('R² Scores Heatmap\n(Components vs Folds)')

        # 添加数值标签
        for i in range(len(n_components)):
            for j in range(self.cv_folds):
                text = ax3.text(j, i, f'{heatmap_data[i, j]:.3f}',
                                ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=ax3, label='R² Score')

        # 图4: 方差分析
        ax4 = axes[1, 0]
        variances = [self.cv_results[n]['std_r2'] ** 2 for n in n_components]
        ax4.plot(n_components, variances, 'o-', linewidth=2, markersize=8)
        ax4.set_xlabel('Number of Components')
        ax4.set_ylabel('Variance of R² Scores')
        ax4.set_title('Model Stability Analysis')
        ax4.grid(True, alpha=0.3)

        # 图5: 训练集大小影响
        ax5 = axes[1, 1]

        # 选择最优主成分数进行分析
        best_n_comp = n_components[np.argmax(mean_scores)]
        fold_details = self.cv_results[best_n_comp]['fold_details']

        train_sizes = [detail['train_samples'] for detail in fold_details]
        val_scores = [detail['r2'] for detail in fold_details]

        ax5.scatter(train_sizes, val_scores, s=100, alpha=0.7)
        for i, detail in enumerate(fold_details):
            ax5.annotate(f"Fold {detail['fold']}",
                         (detail['train_samples'], detail['r2']),
                         xytext=(5, 5), textcoords='offset points')

        ax5.set_xlabel('Training Set Size')
        ax5.set_ylabel('Validation R²')
        ax5.set_title(f'Training Size vs Performance\n(n_components={best_n_comp})')
        ax5.grid(True, alpha=0.3)

        # 图6: 数据分割可视化
        ax6 = axes[1, 2]

        # 创建分割可视化
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        fold_data = []

        fold_idx = 1
        for train_idx, val_idx in kf.split(self.X):
            # 训练集
            fold_data.extend([(i, fold_idx, 'Train') for i in train_idx])
            # 验证集
            fold_data.extend([(i, fold_idx, 'Validation') for i in val_idx])
            fold_idx += 1

        df_split = pd.DataFrame(fold_data, columns=['Sample_Index', 'Fold', 'Type'])

        # 创建分割矩阵
        split_matrix = np.zeros((len(self.X), self.cv_folds))
        fold_idx = 0
        for train_idx, val_idx in kf.split(self.X):
            split_matrix[train_idx, fold_idx] = 1  # 训练集为1
            split_matrix[val_idx, fold_idx] = 2  # 验证集为2
            fold_idx += 1

        im2 = ax6.imshow(split_matrix.T, cmap='RdBu', aspect='auto')
        ax6.set_xlabel('Sample Index')
        ax6.set_ylabel('Fold Number')
        ax6.set_title('K-Fold Data Splitting Visualization')
        ax6.set_yticks(range(self.cv_folds))
        ax6.set_yticklabels([f'Fold {i + 1}' for i in range(self.cv_folds)])

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', label='Training Set'),
                           Patch(facecolor='red', label='Validation Set')]
        ax6.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig('data/figures/cross_validation_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 交叉验证分析图表已保存到: data/figures/cross_validation_analysis.png")
        plt.show()

    def generate_cv_report(self):
        """生成交叉验证详细报告"""
        if not self.cv_results:
            print("❌ 请先运行 detailed_cross_validation()")
            return

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"data/results/cross_validation_report_{timestamp}.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Cross-Validation Detailed Analysis Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Cross-validation Strategy: {self.cv_folds}-Fold\n")
            f.write(f"Total Samples: {len(self.X)}\n")
            f.write(f"Features: {self.X.shape[1]}\n")
            f.write(f"Target Variables: {self.Y.shape[1]}\n\n")

            f.write("1. Overall Results Summary\n")
            f.write("-" * 40 + "\n")

            best_n = max(self.cv_results.keys(),
                         key=lambda k: self.cv_results[k]['mean_r2'])
            best_score = self.cv_results[best_n]['mean_r2']
            best_std = self.cv_results[best_n]['std_r2']

            f.write(f"Best Configuration:\n")
            f.write(f"  Components: {best_n}\n")
            f.write(f"  Mean R²: {best_score:.6f}\n")
            f.write(f"  Std R²: {best_std:.6f}\n")
            f.write(
                f"  95% Confidence Interval: [{best_score - 1.96 * best_std:.6f}, {best_score + 1.96 * best_std:.6f}]\n\n")

            f.write("2. Detailed Results by Component Number\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Components':<12}{'Mean R²':<12}{'Std R²':<12}{'Min R²':<12}{'Max R²':<12}\n")
            f.write("-" * 60 + "\n")

            for n_comp in sorted(self.cv_results.keys()):
                result = self.cv_results[n_comp]
                mean_r2 = result['mean_r2']
                std_r2 = result['std_r2']
                min_r2 = min(result['fold_scores'])
                max_r2 = max(result['fold_scores'])

                f.write(f"{n_comp:<12}{mean_r2:<12.6f}{std_r2:<12.6f}{min_r2:<12.6f}{max_r2:<12.6f}\n")

            f.write("\n3. Fold-by-Fold Analysis (Best Configuration)\n")
            f.write("-" * 40 + "\n")

            best_details = self.cv_results[best_n]['fold_details']
            f.write(f"Configuration: {best_n} components\n\n")
            f.write(f"{'Fold':<6}{'Train Samples':<14}{'Val Samples':<12}{'R²':<12}{'RMSE':<12}\n")
            f.write("-" * 56 + "\n")

            for detail in best_details:
                f.write(f"{detail['fold']:<6}{detail['train_samples']:<14}{detail['val_samples']:<12}"
                        f"{detail['r2']:<12.6f}{detail['rmse']:<12.6f}\n")

            f.write("\n4. Statistical Analysis\n")
            f.write("-" * 40 + "\n")

            # 计算变异系数
            cv_coefficient = best_std / best_score * 100
            f.write(f"Coefficient of Variation: {cv_coefficient:.2f}%\n")

            # 稳定性评估
            if cv_coefficient < 5:
                stability = "Excellent"
            elif cv_coefficient < 10:
                stability = "Good"
            elif cv_coefficient < 15:
                stability = "Acceptable"
            else:
                stability = "Poor"

            f.write(f"Model Stability: {stability}\n")

            # 性能一致性
            fold_range = max(result['fold_scores']) - min(result['fold_scores'])
            f.write(f"Performance Range: {fold_range:.6f}\n")

            f.write("\n5. Recommendations\n")
            f.write("-" * 40 + "\n")

            if best_score > 0.9:
                f.write("✅ Excellent model performance, ready for deployment\n")
            elif best_score > 0.8:
                f.write("✅ Good model performance, suitable for most applications\n")
            elif best_score > 0.7:
                f.write("⚠️ Acceptable performance, consider feature engineering\n")
            else:
                f.write("❌ Poor performance, model needs significant improvement\n")

            if cv_coefficient < 10:
                f.write("✅ Model shows good stability across folds\n")
            else:
                f.write("⚠️ Model shows high variance, consider regularization\n")

        print(f"📋 交叉验证详细报告已保存到: {report_file}")


def analyze_cross_validation():
    """运行完整的交叉验证分析"""
    print("🔍 加载数据...")

    # 加载数据
    X = pd.read_csv("data/processed/X_dataset.csv").values
    Y = pd.read_csv("data/processed/Y_labels.csv").values

    print(f"数据规模: {X.shape[0]} 样本, {X.shape[1]} 特征")

    # 创建分析器
    cv_analyzer = CrossValidationAnalyzer(X, Y, cv_folds=5)

    # 运行详细交叉验证
    results = cv_analyzer.detailed_cross_validation()

    # 生成可视化
    cv_analyzer.visualize_cv_results()

    # 生成报告
    cv_analyzer.generate_cv_report()

    print("\n✅ 交叉验证分析完成！")
    return cv_analyzer


if __name__ == "__main__":
    # 确保目录存在
    import os

    os.makedirs("data/results", exist_ok=True)
    os.makedirs("data/figures", exist_ok=True)

    # 运行分析
    analyzer = analyze_cross_validation()