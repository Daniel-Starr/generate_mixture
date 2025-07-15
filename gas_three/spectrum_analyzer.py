# test_known_mixture.py
# 使用已知浓度的混合光谱测试模型检测性能

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import os
from datetime import datetime


class MixtureConcentrationTester:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.wavenumber_range = None
        self.model_ready = False
        self.gas_names = ['NO', 'NO2', 'SO2']

        # 已知的真实浓度
        self.true_concentrations = {
            'NO': 0.2,  # 20% (2/10)
            'NO2': 0.4,  # 40% (4/10)
            'SO2': 0.4  # 40% (4/10)
        }

    def load_trained_model(self):
        """加载已训练的模型"""
        try:
            print("📊 加载训练好的模型...")

            # 加载训练数据重新训练模型（如果没有保存的模型）
            X_train = pd.read_csv("X_dataset.csv")
            Y_train = pd.read_csv("Y_labels.csv")

            # 获取波数范围
            wavenumber_cols = [col for col in X_train.columns if 'cm-1' in col]
            self.wavenumber_range = [float(col.replace('cm-1', '')) for col in wavenumber_cols]
            self.wavenumber_range.sort()

            # 数据预处理
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train.values)
            #对X进行标准化处理(每个特征减去均值后除以标准差)  目的：消除不同波数点强度量纲差异，使所有特征处于相同尺度

            # 训练模型
            # 训练PLS模型

            # n_components=5 表示使用5个潜在变量(主成分)
            self.model = PLSRegression(n_components=5)
            # 指定潜在变量(主成分)数量为5
            # PLS通过寻找X和Y之间的最大协方差方向来构建潜在变量




            # X_train_scaled: 标准化后的光谱数据矩阵 (样本数 × 波数点)
            # Y_train.values: 浓度标签矩阵 (样本数 × 气体种类)
            self.model.fit(X_train_scaled, Y_train.values)
            """1. 初始化:
   X0 = X (标准化后的光谱数据)
   Y0 = Y (浓度矩阵)
   k = 0 (迭代次数)

2. 主成分提取(重复直到提取n_components个成分):
   a. 计算权重向量w:
        w = X_k' * Y_k / ||X_k' * Y_k||
        (最大化X和Y之间的协方差)
   
   b. 计算X得分向量t:
        t = X_k * w
        (X在w方向上的投影)
   
   c. 计算Y权重向量c:
        c = Y_k' * t / (t' * t)
        (Y在t方向上的权重)
   
   d. 计算Y得分向量u:
        u = Y_k * c
        (Y在c方向上的投影)
   
   e. 计算X载荷向量p:
        p = X_k' * t / (t' * t)
        (X与t的关系)
   
   f. 计算Y载荷向量q:
        q = Y_k' * u / (u' * u)
        (Y与u的关系)
   
   g. 更新残差矩阵:
        X_{k+1} = X_k - t * p'
        Y_{k+1} = Y_k - t * c'
        (减去当前主成分解释的部分)
   
   h. k = k + 1

3. 构建回归系数矩阵B:
        B = W(P'W)^{-1}C'
        (其中W是权重矩阵，P是X载荷矩阵，C是Y权重矩阵)"""



            self.model_ready = True
            print(f"✅ 模型加载成功")
            print(f"📊 模型支持波数范围: {min(self.wavenumber_range):.0f} - {max(self.wavenumber_range):.0f} cm⁻¹")
            print(f"📊 训练数据规模: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")

            return True

        except FileNotFoundError as e:
            print(f"❌ 找不到训练数据文件: {e}")
            print("请确保 X_dataset.csv 和 Y_labels.csv 存在")
            return False
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False

    def load_test_spectrum(self, file_path):
        """加载测试光谱数据"""
        try:
            print(f"\n📄 读取测试光谱: {file_path}")

            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                return None, None

            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 检查列名
            if 'wavenumber' not in df.columns or 'intensity' not in df.columns:
                print(f"❌ 文件格式错误，需要 'wavenumber' 和 'intensity' 列")
                print(f"实际列名: {list(df.columns)}")
                return None, None

            # 提取数据
            wavenumbers = df['wavenumber'].values
            intensities = df['intensity'].values

            # 移除无效数据
            valid_mask = np.isfinite(wavenumbers) & np.isfinite(intensities)
            wavenumbers_clean = wavenumbers[valid_mask]
            intensities_clean = intensities[valid_mask]

            print(f"✅ 成功读取测试光谱")
            print(f"📊 数据点数: {len(wavenumbers_clean)}")
            print(f"📊 波数范围: {wavenumbers_clean.min():.1f} - {wavenumbers_clean.max():.1f} cm⁻¹")
            print(f"📊 强度范围: {intensities_clean.min():.3e} - {intensities_clean.max():.3e}")

            return wavenumbers_clean, intensities_clean

        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            return None, None

    def preprocess_test_spectrum(self, wavenumbers, intensities):
        """预处理测试光谱到模型输入格式"""
        if not self.model_ready:
            print("❌ 模型未加载")
            return None

        print(f"\n🔧 预处理测试光谱...")

        # 检查波数范围兼容性
        test_min, test_max = wavenumbers.min(), wavenumbers.max()
        model_min, model_max = min(self.wavenumber_range), max(self.wavenumber_range)

        print(f"   测试光谱范围: {test_min:.1f} - {test_max:.1f} cm⁻¹")
        print(f"   模型需要范围: {model_min:.1f} - {model_max:.1f} cm⁻¹")

        # 计算覆盖度
        overlap_min = max(test_min, model_min)
        overlap_max = min(test_max, model_max)

        if overlap_min >= overlap_max:
            print("❌ 测试光谱与模型波数范围不匹配")
            return None

        coverage = (overlap_max - overlap_min) / (model_max - model_min)
        print(f"   波数覆盖度: {coverage:.1%}")

        if coverage < 0.8:
            print("⚠️ 警告: 波数覆盖度较低，预测可能不准确")

        # 插值到模型波数网格
        try:
            interp_func = interp1d(wavenumbers, intensities,
                                   kind='linear', bounds_error=False, fill_value=0)

            model_spectrum = interp_func(self.wavenumber_range)

            # 检查插值结果
            valid_points = np.sum(model_spectrum != 0)
            total_points = len(model_spectrum)

            print(f"   插值结果: {valid_points}/{total_points} 有效点 ({valid_points / total_points:.1%})")

            return model_spectrum.reshape(1, -1)

        except Exception as e:
            print(f"❌ 光谱预处理失败: {e}")
            return None

    def predict_concentrations(self, spectrum_data):
        """预测气体浓度"""
        if not self.model_ready or spectrum_data is None:
            return None

        try:
            print(f"\n🔍 进行浓度预测...")

            # 标准化输入数据  标准化输入数据 (使用训练时的scaler)
            spectrum_scaled = self.scaler.transform(spectrum_data)

            # 预测浓度
            raw_predictions = self.model.predict(spectrum_scaled)[0]

            print(f"   原始预测值: {raw_predictions}")

            # 处理预测结果
            # 确保非负
            predictions_positive = np.maximum(raw_predictions, 0)

            # 归一化到总和为1
            #"归一化到总和为1"是指将预测出的各个气体浓度值进行调整，使得所有气体的浓度预测值之和等于1（即100%）。
            # 这样做的目的是将预测结果表示为各气体在混合物中所占的比例。
            total = np.sum(predictions_positive)
            if total > 0:
                predictions_normalized = predictions_positive / total
            else:
                # 如果所有预测都是0，使用均匀分布
                predictions_normalized = np.array([1 / 3, 1 / 3, 1 / 3])
                print("⚠️ 所有预测值为0，使用均匀分布")

            print(f"   归一化预测: {predictions_normalized}")

            # 组织结果
            predicted_concentrations = {}
            for i, gas in enumerate(self.gas_names):
                predicted_concentrations[gas] = float(predictions_normalized[i])

            return predicted_concentrations

        except Exception as e:
            print(f"❌ 浓度预测失败: {e}")
            return None

    def calculate_errors(self, predicted_concentrations):
        """计算预测误差"""
        print(f"\n📊 计算预测误差...")

        errors = {}
        absolute_errors = []
        relative_errors = []

        print(f"{'气体':>6} {'真实值':>8} {'预测值':>8} {'绝对误差':>8} {'相对误差':>8}")
        print("-" * 50)
        # 遍历每种气体
        for gas in self.gas_names:
            # 获取真实值和预测值
            true_val = self.true_concentrations[gas]
            pred_val = predicted_concentrations[gas]
            # 计算绝对误差 = |预测值 - 真实值|
            abs_error = abs(pred_val - true_val)
            # 计算相对误差 = (|预测值 - 真实值| / 真实值) * 100%
            # 避免除以零错误（真实值>0时计算）
            rel_error = (abs_error / true_val) * 100 if true_val > 0 else 0

            errors[gas] = {
                'true': true_val,
                'predicted': pred_val,
                'absolute_error': abs_error,
                'relative_error': rel_error
            }
            # 收集误差用于整体统计
            absolute_errors.append(abs_error)
            relative_errors.append(rel_error)

            print(f"{gas:>6} {true_val:>8.3f} {pred_val:>8.3f} {abs_error:>8.3f} {rel_error:>7.1f}%")

        # 总体误差统计
        mean_abs_error = np.mean(absolute_errors)
        mean_rel_error = np.mean(relative_errors)
        max_rel_error = np.max(relative_errors)

        print("-" * 50)
        print(f"{'平均':>6} {'':>8} {'':>8} {mean_abs_error:>8.3f} {mean_rel_error:>7.1f}%")
        print(f"最大相对误差: {max_rel_error:.1f}%")

        # 性能评估
        if mean_rel_error < 5:
            performance = "🌟 优秀"
        elif mean_rel_error < 10:
            performance = "✅ 良好"
        elif mean_rel_error < 20:
            performance = "⚠️ 一般"
        else:
            performance = "❌ 较差"

        print(f"\n🎯 模型性能评估: {performance} (平均相对误差: {mean_rel_error:.1f}%)")

        return errors, mean_abs_error, mean_rel_error

    def visualize_results(self, wavenumbers, intensities, predicted_concentrations, errors):
        """可视化测试结果"""
        try:
            print(f"\n📊 生成结果可视化...")

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. 测试光谱
            axes[0, 0].plot(wavenumbers, intensities, 'blue', linewidth=1.5, alpha=0.8)
            axes[0, 0].set_title('Test Spectrum (Known 2:4:4 Mixture)', fontweight='bold')
            axes[0, 0].set_xlabel('Wavenumber (cm⁻¹)')
            axes[0, 0].set_ylabel('Intensity')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

            # 添加统计信息
            max_intensity = intensities.max()
            mean_intensity = intensities.mean()
            axes[0, 0].text(0.02, 0.98, f'Max: {max_intensity:.2e}\nMean: {mean_intensity:.2e}',
                            transform=axes[0, 0].transAxes,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
                            verticalalignment='top', fontsize=9)

            # 2. 真实值vs预测值对比
            gas_names = list(predicted_concentrations.keys())
            true_values = [self.true_concentrations[gas] for gas in gas_names]
            pred_values = [predicted_concentrations[gas] for gas in gas_names]

            x = np.arange(len(gas_names))
            width = 0.35

            bars1 = axes[0, 1].bar(x - width / 2, true_values, width,
                                   label='True', color='green', alpha=0.7)
            bars2 = axes[0, 1].bar(x + width / 2, pred_values, width,
                                   label='Predicted', color='orange', alpha=0.7)

            axes[0, 1].set_title('True vs Predicted Concentrations', fontweight='bold')
            axes[0, 1].set_xlabel('Gas Type')
            axes[0, 1].set_ylabel('Concentration')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(gas_names)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height,
                                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

            # 3. 误差分析
            rel_errors = [errors[gas]['relative_error'] for gas in gas_names]
            colors = ['red', 'blue', 'green']

            bars = axes[1, 0].bar(gas_names, rel_errors, color=colors, alpha=0.7)
            axes[1, 0].set_title('Relative Error by Gas', fontweight='bold')
            axes[1, 0].set_xlabel('Gas Type')
            axes[1, 0].set_ylabel('Relative Error (%)')
            axes[1, 0].grid(True, alpha=0.3)

            # 添加误差标签
            for bar, error in zip(bars, rel_errors):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height,
                                f'{error:.1f}%', ha='center', va='bottom', fontsize=9)

            # 4. 散点图 (真实值 vs 预测值)
            axes[1, 1].scatter(true_values, pred_values,
                               c=colors, s=100, alpha=0.7, edgecolors='black')

            # 添加完美预测线
            min_val = min(min(true_values), min(pred_values))
            max_val = max(max(true_values), max(pred_values))
            axes[1, 1].plot([min_val, max_val], [min_val, max_val],
                            'k--', alpha=0.5, linewidth=2, label='Perfect Prediction')

            # 标注气体名称
            for i, gas in enumerate(gas_names):
                axes[1, 1].annotate(gas, (true_values[i], pred_values[i]),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=10, fontweight='bold')

            axes[1, 1].set_title('True vs Predicted (Scatter Plot)', fontweight='bold')
            axes[1, 1].set_xlabel('True Concentration')
            axes[1, 1].set_ylabel('Predicted Concentration')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_aspect('equal', adjustable='box')

            # 添加R²信息
            from sklearn.metrics import r2_score
            r2 = r2_score(true_values, pred_values)
            axes[1, 1].text(0.05, 0.95, f'R² = {r2:.4f}',
                            transform=axes[1, 1].transAxes,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
                            verticalalignment='top', fontsize=10)

            plt.suptitle('Gas Concentration Detection Test Results\n'
                         'Known Mixture: NO:NO2:SO2 = 2:4:4 (20%:40%:40%)',
                         fontsize=14, fontweight='bold')

            plt.tight_layout()

            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f'concentration_test_results_{timestamp}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📊 结果图表已保存: {plot_filename}")

            plt.show()

        except Exception as e:
            print(f"⚠️ 可视化失败: {e}")

    def test_mixture_concentration(self, test_file_path):
        """完整的混合物浓度测试流程"""
        print("🔬 已知混合光谱的浓度检测测试")
        print("=" * 60)
        print(f"测试文件: {test_file_path}")
        print(f"已知真实浓度: NO={self.true_concentrations['NO']:.1%}, "
              f"NO2={self.true_concentrations['NO2']:.1%}, "
              f"SO2={self.true_concentrations['SO2']:.1%}")
        print("=" * 60)

        # 1. 加载模型
        if not self.load_trained_model():
            return None

        # 2. 读取测试光谱
        wavenumbers, intensities = self.load_test_spectrum(test_file_path)
        if wavenumbers is None:
            return None

        # 3. 预处理光谱
        spectrum_data = self.preprocess_test_spectrum(wavenumbers, intensities)
        if spectrum_data is None:
            return None

        # 4. 预测浓度
        predicted_concentrations = self.predict_concentrations(spectrum_data)
        if predicted_concentrations is None:
            return None

        # 5. 计算误差
        errors, mean_abs_error, mean_rel_error = self.calculate_errors(predicted_concentrations)

        # 6. 可视化结果
        self.visualize_results(wavenumbers, intensities, predicted_concentrations, errors)

        # 7. 保存详细结果
        self.save_test_results(predicted_concentrations, errors, mean_abs_error, mean_rel_error)

        return {
            'predicted_concentrations': predicted_concentrations,
            'errors': errors,
            'mean_absolute_error': mean_abs_error,
            'mean_relative_error': mean_rel_error
        }

    def save_test_results(self, predicted_concentrations, errors, mean_abs_error, mean_rel_error):
        """保存测试结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 创建结果DataFrame
            results_data = []
            for gas in self.gas_names:
                results_data.append({
                    'Gas': gas,
                    'True_Concentration': self.true_concentrations[gas],
                    'Predicted_Concentration': predicted_concentrations[gas],
                    'Absolute_Error': errors[gas]['absolute_error'],
                    'Relative_Error_Percent': errors[gas]['relative_error']
                })

            results_df = pd.DataFrame(results_data)

            # 保存CSV
            csv_filename = f'concentration_test_results_{timestamp}.csv'
            results_df.to_csv(csv_filename, index=False)

            # 保存详细报告
            report_filename = f'test_report_{timestamp}.txt'
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write("气体浓度检测测试报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"已知混合比例: NO:NO2:SO2 = 2:4:4\n\n")

                f.write("详细结果:\n")
                f.write("-" * 30 + "\n")
                for gas in self.gas_names:
                    f.write(f"{gas}:\n")
                    f.write(f"  真实浓度: {self.true_concentrations[gas]:.3f}\n")
                    f.write(f"  预测浓度: {predicted_concentrations[gas]:.3f}\n")
                    f.write(f"  绝对误差: {errors[gas]['absolute_error']:.3f}\n")
                    f.write(f"  相对误差: {errors[gas]['relative_error']:.1f}%\n\n")

                f.write(f"总体性能:\n")
                f.write(f"  平均绝对误差: {mean_abs_error:.3f}\n")
                f.write(f"  平均相对误差: {mean_rel_error:.1f}%\n")

            print(f"💾 测试结果已保存:")
            print(f"   📄 {csv_filename}")
            print(f"   📋 {report_filename}")

        except Exception as e:
            print(f"⚠️ 保存结果失败: {e}")


# 主函数
def test_known_mixture():
    """测试已知浓度的混合光谱"""
    tester = MixtureConcentrationTester()

    # 你的测试文件路径
    test_file = r"E:\generate_mixture\gas_three\test\mixed_spectrum_244_noisy_20250710_152143.csv"

    # 运行测试
    results = tester.test_mixture_concentration(test_file)

    if results:
        print(f"\n🎉 测试完成！")
        print(f"📊 结果摘要:")
        print(f"   平均相对误差: {results['mean_relative_error']:.1f}%")

        if results['mean_relative_error'] < 5:
            print(f"   🌟 模型性能优秀！")
        elif results['mean_relative_error'] < 10:
            print(f"   ✅ 模型性能良好")
        else:
            print(f"   ⚠️ 模型性能有待提升")
    else:
        print(f"\n❌ 测试失败")


if __name__ == "__main__":
    test_known_mixture()