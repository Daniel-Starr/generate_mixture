"""
气体光谱分析系统 - 优化版本
用于NO、NO2、SO2三种气体的定量分析
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import json

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """系统配置类"""
    # 数据路径
    data_dir: Path = Path("E:/generate_mixture/hitran_csv")
    output_dir: Path = Path("./output")

    # 气体配置
    gases: List[str] = None

    # 光谱参数
    wavenumber_step: float = 0.5  # 波数步长

    # 数据生成参数
    noise_levels: List[float] = None  # 多种噪声水平
    samples_per_ratio: int = 20

    # 模型参数
    max_components: int = 15
    cv_folds: int = 10
    test_size: float = 0.2

    # 可视化参数
    figure_dpi: int = 300

    def __post_init__(self):
        if self.gases is None:
            self.gases = ['NO', 'NO2', 'SO2']
        if self.noise_levels is None:
            self.noise_levels = [0.005, 0.01, 0.02]  # 0.5%, 1%, 2%噪声

        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        for subdir in ['data', 'models', 'figures', 'reports']:
            (self.output_dir / subdir).mkdir(exist_ok=True)


class SpectralDataProcessor:
    """光谱数据处理类"""

    def __init__(self, config: Config):
        self.config = config
        self.interpolated_data = None
        self.wavenumber_range = None

    def load_and_interpolate(self) -> pd.DataFrame:
        """加载并插值光谱数据"""
        logger.info("开始加载光谱数据...")

        gas_data = {}
        wavenumber_ranges = []

        # 加载各气体数据
        for gas in self.config.gases:
            file_path = self.config.data_dir / f"{gas}.csv"
            df = pd.read_csv(file_path)
            gas_data[gas] = df
            wavenumber_ranges.append((df['nu'].min(), df['nu'].max()))
            logger.info(f"{gas}: 波数范围 {df['nu'].min():.2f} - {df['nu'].max():.2f}")

        # 确定公共波数范围
        common_min = max(r[0] for r in wavenumber_ranges)
        common_max = min(r[1] for r in wavenumber_ranges)
        logger.info(f"公共波数范围: {common_min:.2f} - {common_max:.2f}")

        # 创建统一波数轴
        self.wavenumber_range = np.arange(
            np.ceil(common_min),
            np.floor(common_max) + self.config.wavenumber_step,
            self.config.wavenumber_step
        )

        # 插值
        interpolated_dict = {'wavenumber': self.wavenumber_range}

        for gas, df in gas_data.items():
            # 使用更高级的插值方法
            interp_func = interp1d(
                df['nu'], df['sw'],
                kind='cubic',  # 三次样条插值
                bounds_error=False,
                fill_value=0
            )
            interpolated_dict[gas] = interp_func(self.wavenumber_range)

        self.interpolated_data = pd.DataFrame(interpolated_dict)

        # 保存插值后的数据
        output_path = self.config.output_dir / 'data' / 'interpolated_spectra.csv'
        self.interpolated_data.to_csv(output_path, index=False)
        logger.info(f"插值数据已保存: {output_path}")

        return self.interpolated_data

    def plot_spectra(self):
        """绘制光谱图"""
        if self.interpolated_data is None:
            raise ValueError("请先运行 load_and_interpolate()")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        # 单独绘制每种气体
        for i, gas in enumerate(self.config.gases):
            ax = axes[i]
            ax.plot(self.interpolated_data['wavenumber'],
                    self.interpolated_data[gas],
                    label=gas, linewidth=1.5)
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Absorption Intensity')
            ax.set_title(f'{gas} Spectrum')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 绘制叠加光谱
        ax = axes[3]
        for gas in self.config.gases:
            ax.plot(self.interpolated_data['wavenumber'],
                    self.interpolated_data[gas],
                    label=gas, alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Absorption Intensity')
        ax.set_title('All Spectra Overlay')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        output_path = self.config.output_dir / 'figures' / 'individual_spectra.png'
        plt.savefig(output_path, dpi=self.config.figure_dpi)
        plt.close()
        logger.info(f"光谱图已保存: {output_path}")


class DatasetGenerator:
    """数据集生成器"""

    def __init__(self, config: Config, spectral_data: pd.DataFrame):
        self.config = config
        self.spectral_data = spectral_data
        self.X_data = None
        self.Y_data = None

    def generate_concentration_grid(self) -> np.ndarray:
        """生成浓度组合网格（优化的采样策略）"""
        # 使用更智能的采样策略
        concentrations = []

        # 1. 均匀网格采样
        for no in np.arange(0.1, 0.5, 0.05):
            for no2 in np.arange(0.1, 0.5, 0.05):
                so2 = 1.0 - no - no2
                if 0.1 <= so2 <= 0.8:  # 确保SO2也在合理范围
                    concentrations.append([no, no2, so2])

        # 2. 添加边界情况
        boundary_samples = [
            [0.8, 0.1, 0.1],  # NO为主
            [0.1, 0.8, 0.1],  # NO2为主
            [0.1, 0.1, 0.8],  # SO2为主
            [0.33, 0.33, 0.34],  # 均匀分布
        ]
        concentrations.extend(boundary_samples)

        # 3. 添加随机采样以增加多样性
        n_random = 50
        for _ in range(n_random):
            # 使用Dirichlet分布生成和为1的随机数
            random_conc = np.random.dirichlet([2, 2, 2])  # alpha参数控制分布
            concentrations.append(random_conc.tolist())

        return np.array(concentrations)

    def add_realistic_noise(self, spectrum: np.ndarray, noise_level: float) -> np.ndarray:
        """添加更真实的噪声模型"""
        # 1. 乘性噪声（与信号强度相关）
        multiplicative_noise = np.random.normal(1, noise_level, size=spectrum.shape)

        # 2. 加性噪声（基线噪声）
        additive_noise = np.random.normal(0, noise_level * np.mean(spectrum), size=spectrum.shape)

        # 3. 偶尔的尖峰噪声
        spike_prob = 0.001
        spike_mask = np.random.random(spectrum.shape) < spike_prob
        spike_noise = np.random.normal(0, 5 * noise_level * np.mean(spectrum), size=spectrum.shape)

        noisy_spectrum = spectrum * multiplicative_noise + additive_noise
        noisy_spectrum[spike_mask] += spike_noise[spike_mask]

        # 确保非负
        noisy_spectrum = np.maximum(noisy_spectrum, 0)

        return noisy_spectrum

    def generate_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """生成训练数据集"""
        logger.info("开始生成数据集...")

        concentrations = self.generate_concentration_grid()
        logger.info(f"生成了 {len(concentrations)} 种浓度组合")

        X_list = []
        Y_list = []
        metadata_list = []

        # 提取纯组分光谱
        pure_spectra = self.spectral_data[self.config.gases].values.T

        for conc in concentrations:
            for noise_level in self.config.noise_levels:
                for i in range(self.config.samples_per_ratio):
                    # 生成混合光谱
                    mixed_spectrum = np.dot(conc, pure_spectra)

                    # 添加噪声
                    noisy_spectrum = self.add_realistic_noise(mixed_spectrum, noise_level)

                    X_list.append(noisy_spectrum)
                    Y_list.append(conc)
                    metadata_list.append({
                        'noise_level': noise_level,
                        'sample_id': i,
                        'total_intensity': np.sum(noisy_spectrum)
                    })

        # 转换为DataFrame
        wavenumbers = self.spectral_data['wavenumber'].values
        self.X_data = pd.DataFrame(X_list, columns=[f'{w:.1f}cm-1' for w in wavenumbers])
        self.Y_data = pd.DataFrame(Y_list, columns=[f'{gas}_conc' for gas in self.config.gases])

        # 保存元数据
        metadata_df = pd.DataFrame(metadata_list)

        # 保存数据集
        self.X_data.to_csv(self.config.output_dir / 'data' / 'X_dataset.csv', index=False)
        self.Y_data.to_csv(self.config.output_dir / 'data' / 'Y_labels.csv', index=False)
        metadata_df.to_csv(self.config.output_dir / 'data' / 'dataset_metadata.csv', index=False)

        logger.info(f"数据集生成完成: {len(self.X_data)} 个样本")

        return self.X_data, self.Y_data


class AdvancedPLSModel:
    """高级PLS模型"""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.best_params = None
        self.performance_metrics = {}

    def optimize_hyperparameters(self, X_train: np.ndarray, Y_train: np.ndarray) -> Dict:
        """使用网格搜索优化超参数"""
        logger.info("开始超参数优化...")

        # 定义参数网格
        param_grid = {
            'n_components': range(2, min(self.config.max_components, X_train.shape[1] // 2)),
            'scale': [True, False],
            'max_iter': [500, 1000]
        }

        # 网格搜索
        grid_search = GridSearchCV(
            PLSRegression(),
            param_grid,
            cv=self.config.cv_folds,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, Y_train)

        self.best_params = grid_search.best_params_
        logger.info(f"最优参数: {self.best_params}")

        return self.best_params

    def train(self, X_train: np.ndarray, Y_train: np.ndarray,
              X_test: np.ndarray, Y_test: np.ndarray):
        """训练模型"""
        logger.info("开始训练PLS模型...")

        # 数据标准化
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        Y_train_scaled = self.scaler_Y.fit_transform(Y_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        Y_test_scaled = self.scaler_Y.transform(Y_test)

        # 优化超参数
        self.optimize_hyperparameters(X_train_scaled, Y_train_scaled)

        # 训练最终模型
        self.model = PLSRegression(**self.best_params)
        self.model.fit(X_train_scaled, Y_train_scaled)

        # 预测
        Y_pred_scaled = self.model.predict(X_test_scaled)
        Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)

        # 计算性能指标
        self.calculate_metrics(Y_test, Y_pred)

        # 保存模型
        self.save_model()

        return Y_pred

    def calculate_metrics(self, Y_true: np.ndarray, Y_pred: np.ndarray):
        """计算详细的性能指标"""
        gases = self.config.gases

        # 整体指标
        self.performance_metrics['overall'] = {
            'rmse': np.sqrt(mean_squared_error(Y_true, Y_pred)),
            'r2': r2_score(Y_true, Y_pred),
            'mae': mean_absolute_error(Y_true, Y_pred)
        }

        # 每种气体的指标
        for i, gas in enumerate(gases):
            self.performance_metrics[gas] = {
                'rmse': np.sqrt(mean_squared_error(Y_true[:, i], Y_pred[:, i])),
                'r2': r2_score(Y_true[:, i], Y_pred[:, i]),
                'mae': mean_absolute_error(Y_true[:, i], Y_pred[:, i]),
                'mape': np.mean(np.abs((Y_true[:, i] - Y_pred[:, i]) / Y_true[:, i])) * 100
            }

        # 保存性能报告
        self.save_performance_report()

    def save_performance_report(self):
        """保存性能报告"""
        report_path = self.config.output_dir / 'reports' / 'model_performance.json'

        report = {
            'best_params': self.best_params,
            'performance_metrics': self.performance_metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)

        logger.info(f"性能报告已保存: {report_path}")

        # 打印性能摘要
        print("\n=== 模型性能摘要 ===")
        print(f"整体 R²: {self.performance_metrics['overall']['r2']:.4f}")
        print(f"整体 RMSE: {self.performance_metrics['overall']['rmse']:.4f}")

        for gas in self.config.gases:
            metrics = self.performance_metrics[gas]
            print(f"\n{gas}:")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")

    def save_model(self):
        """保存模型和相关组件"""
        model_dir = self.config.output_dir / 'models'

        # 保存PLS模型
        joblib.dump(self.model, model_dir / 'pls_model.pkl')

        # 保存标准化器
        joblib.dump(self.scaler_X, model_dir / 'scaler_X.pkl')
        joblib.dump(self.scaler_Y, model_dir / 'scaler_Y.pkl')

        # 保存配置
        joblib.dump(self.config, model_dir / 'config.pkl')

        logger.info(f"模型已保存到: {model_dir}")

    def visualize_results(self, Y_test: np.ndarray, Y_pred: np.ndarray):
        """高级结果可视化"""
        gases = self.config.gases
        n_gases = len(gases)

        # 1. 预测vs真实散点图
        fig, axes = plt.subplots(1, n_gases, figsize=(5 * n_gases, 4))
        if n_gases == 1:
            axes = [axes]

        for i, (gas, ax) in enumerate(zip(gases, axes)):
            ax.scatter(Y_test[:, i], Y_pred[:, i], alpha=0.5, s=20)

            # 添加理想线
            min_val, max_val = Y_test[:, i].min(), Y_test[:, i].max()
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            # 添加统计信息
            r2 = self.performance_metrics[gas]['r2']
            rmse = self.performance_metrics[gas]['rmse']
            ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel(f'True {gas} Concentration')
            ax.set_ylabel(f'Predicted {gas} Concentration')
            ax.set_title(f'{gas} Prediction')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'figures' / 'prediction_scatter.png',
                    dpi=self.config.figure_dpi)
        plt.close()

        # 2. 残差分析
        fig, axes = plt.subplots(2, n_gases, figsize=(5 * n_gases, 8))

        for i, gas in enumerate(gases):
            residuals = Y_test[:, i] - Y_pred[:, i]

            # 残差vs预测值
            ax1 = axes[0, i] if n_gases > 1 else axes[0]
            ax1.scatter(Y_pred[:, i], residuals, alpha=0.5, s=20)
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel(f'Predicted {gas}')
            ax1.set_ylabel('Residuals')
            ax1.set_title(f'{gas} Residuals vs Predicted')
            ax1.grid(True, alpha=0.3)

            # 残差直方图
            ax2 = axes[1, i] if n_gases > 1 else axes[1]
            ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Residuals')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'{gas} Residual Distribution')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'figures' / 'residual_analysis.png',
                    dpi=self.config.figure_dpi)
        plt.close()

        # 3. 特征重要性（VIP分数）
        self.plot_vip_scores()

        logger.info("可视化结果已保存")

    def plot_vip_scores(self):
        """计算并绘制VIP（Variable Importance in Projection）分数"""
        # 计算VIP分数
        T = self.model.x_scores_
        W = self.model.x_weights_
        Q = self.model.y_loadings_

        p, h = W.shape
        VIP = np.zeros((p,))

        for i in range(p):
            weight = np.array([(W[i, j] / np.linalg.norm(W[:, j])) ** 2 for j in range(h)])
            VIP[i] = np.sqrt(
                p * np.sum(weight * np.sum(Q ** 2, axis=0) * np.sum(T ** 2, axis=0) / np.sum(np.sum(T ** 2, axis=0))))

        # 选择最重要的特征
        n_top = 50
        top_indices = np.argsort(VIP)[-n_top:]

        # 绘图
        plt.figure(figsize=(12, 6))
        plt.bar(range(n_top), VIP[top_indices])
        plt.axhline(y=1, color='r', linestyle='--', label='VIP = 1')
        plt.xlabel('Feature Index')
        plt.ylabel('VIP Score')
        plt.title('Top 50 Variable Importance in Projection (VIP) Scores')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'figures' / 'vip_scores.png',
                    dpi=self.config.figure_dpi)
        plt.close()


class GasAnalysisSystem:
    """气体分析系统主类"""

    def __init__(self, config: Config):
        self.config = config
        self.processor = SpectralDataProcessor(config)
        self.generator = None
        self.model = AdvancedPLSModel(config)

    def run_complete_analysis(self):
        """运行完整的分析流程"""
        logger.info("=== 开始气体光谱分析 ===")

        # 1. 数据预处理
        spectral_data = self.processor.load_and_interpolate()
        self.processor.plot_spectra()

        # 2. 生成数据集
        self.generator = DatasetGenerator(self.config, spectral_data)
        X_data, Y_data = self.generator.generate_dataset()

        # 3. 数据分割
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_data.values, Y_data.values,
            test_size=self.config.test_size,
            random_state=42
        )

        # 4. 训练模型
        Y_pred = self.model.train(X_train, Y_train, X_test, Y_test)

        # 5. 可视化结果
        self.model.visualize_results(Y_test, Y_pred)

        # 6. 保存测试结果
        self.save_test_results(Y_test, Y_pred)

        logger.info("=== 分析完成 ===")

    def save_test_results(self, Y_test: np.ndarray, Y_pred: np.ndarray):
        """保存测试结果"""
        gases = self.config.gases

        # 真实值
        Y_test_df = pd.DataFrame(Y_test, columns=[f'{gas}_true' for gas in gases])
        Y_test_df.to_csv(self.config.output_dir / 'data' / 'Y_test.csv', index=False)

        # 预测值
        Y_pred_df = pd.DataFrame(Y_pred, columns=[f'{gas}_pred' for gas in gases])
        Y_pred_df.to_csv(self.config.output_dir / 'data' / 'Y_pred.csv', index=False)

        # 对比结果
        comparison_df = pd.concat([Y_test_df, Y_pred_df], axis=1)

        # 添加误差列
        for gas in gases:
            comparison_df[f'{gas}_error'] = comparison_df[f'{gas}_true'] - comparison_df[f'{gas}_pred']
            comparison_df[f'{gas}_error_pct'] = np.abs(
                comparison_df[f'{gas}_error'] / comparison_df[f'{gas}_true']) * 100

        comparison_df.to_csv(self.config.output_dir / 'reports' / 'test_comparison.csv', index=False)

    def predict_unknown_sample(self, spectrum_path: str) -> pd.DataFrame:
        """预测未知样本"""
        # 加载模型
        model_dir = self.config.output_dir / 'models'
        pls_model = joblib.load(model_dir / 'pls_model.pkl')
        scaler_X = joblib.load(model_dir / 'scaler_X.pkl')
        scaler_Y = joblib.load(model_dir / 'scaler_Y.pkl')

        # 加载光谱
        spectrum = pd.read_csv(spectrum_path).values

        # 标准化
        spectrum_scaled = scaler_X.transform(spectrum.reshape(1, -1))

        # 预测
        prediction_scaled = pls_model.predict(spectrum_scaled)
        prediction = scaler_Y.inverse_transform(prediction_scaled)

        # 返回结果
        result_df = pd.DataFrame(
            prediction,
            columns=[f'{gas}_concentration' for gas in self.config.gases]
        )

        return result_df


# 主程序
def main():
    """主函数"""
    # 创建配置
    config = Config()

    # 创建并运行系统
    system = GasAnalysisSystem(config)
    system.run_complete_analysis()

    # 演示预测功能
    logger.info("\n=== 预测示例 ===")
    logger.info("模型已训练完成，可以使用以下代码预测未知样本：")
    logger.info("result = system.predict_unknown_sample('path_to_spectrum.csv')")


if __name__ == "__main__":
    main()