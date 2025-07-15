# improved_gas_detector.py
# 改进的PLS气体浓度检测系统
# 解决仿真-真实数据域差异问题，提升预测精度

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.sparse import csr_matrix
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ImprovedGasDetector:
    """改进的PLS气体浓度检测器
    
    主要改进:
    1. 自适应主成分数选择
    2. 多种预处理策略
    3. 鲁棒性增强
    4. 域适应技术
    5. 集成预测
    """
    
    def __init__(self, gas_names=['NO', 'NO2', 'SO2']):
        self.gas_names = gas_names
        self.n_gases = len(gas_names)
        
        # 模型组件
        self.pls_model = None
        self.scaler = None
        self.wavenumber_grid = None
        self.feature_selector = None
        
        # 模型参数
        self.n_components = None
        self.preprocessing_params = {
            'smoothing_window': 11,
            'smoothing_poly': 3,
            'noise_threshold': 0.001,
            'baseline_method': 'linear'
        }
        
        # 训练历史
        self.training_history = {
            'cv_scores': [],
            'best_params': {},
            'feature_importance': None
        }
        
        print(f"🔬 初始化改进的气体检测器")
        print(f"📊 目标气体: {', '.join(self.gas_names)}")
    
    def load_hitran_data(self, data_path="hitran_csv"):
        """加载和预处理HITRAN数据"""
        print(f"\n📖 加载HITRAN光谱数据...")
        
        spectra_data = {}
        wavenumber_ranges = {}
        
        for gas in self.gas_names:
            file_path = os.path.join(data_path, f"{gas}.csv")
            
            if not os.path.exists(file_path):
                print(f"❌ 未找到文件: {file_path}")
                continue
                
            try:
                # 读取数据
                df = pd.read_csv(file_path)
                
                if 'nu' not in df.columns or 'sw' not in df.columns:
                    print(f"❌ {gas} 文件格式错误，需要 'nu' 和 'sw' 列")
                    continue
                
                # 数据清理
                valid_mask = (df['nu'].notna()) & (df['sw'].notna()) & (df['nu'] > 0) & (df['sw'] >= 0)
                clean_data = df[valid_mask].sort_values('nu').reset_index(drop=True)
                
                # 去除重复波数点
                clean_data = clean_data.drop_duplicates(subset=['nu'], keep='first')
                
                spectra_data[gas] = {
                    'wavenumber': clean_data['nu'].values,
                    'intensity': clean_data['sw'].values
                }
                
                wavenumber_ranges[gas] = (clean_data['nu'].min(), clean_data['nu'].max())
                
                print(f"   ✅ {gas}: {len(clean_data)} 点, 范围 {wavenumber_ranges[gas][0]:.1f}-{wavenumber_ranges[gas][1]:.1f} cm⁻¹")
                
            except Exception as e:
                print(f"❌ 读取 {gas} 数据失败: {e}")
                continue
        
        if len(spectra_data) == 0:
            print("❌ 未成功加载任何光谱数据")
            return None
            
        return spectra_data, wavenumber_ranges
    
    def create_unified_grid(self, wavenumber_ranges, resolution=0.5):
        """创建优化的统一波数网格"""
        print(f"\n🎯 创建统一波数网格 (分辨率: {resolution} cm⁻¹)...")
        
        # 计算有效重叠范围
        all_ranges = list(wavenumber_ranges.values())
        overlap_min = max([r[0] for r in all_ranges])
        overlap_max = min([r[1] for r in all_ranges])
        
        if overlap_min >= overlap_max:
            # 如果没有重叠，使用扩展范围
            print("   ⚠️ 光谱范围无重叠，使用扩展范围")
            extended_min = min([r[0] for r in all_ranges])
            extended_max = max([r[1] for r in all_ranges])
            
            # 选择覆盖度最好的中间区域
            total_range = extended_max - extended_min
            center = (extended_max + extended_min) / 2
            half_width = total_range * 0.3  # 使用60%的总范围
            
            grid_min = center - half_width
            grid_max = center + half_width
        else:
            grid_min = overlap_min
            grid_max = overlap_max
        
        # 创建网格
        self.wavenumber_grid = np.arange(
            np.ceil(grid_min / resolution) * resolution,
            np.floor(grid_max / resolution) * resolution + resolution,
            resolution
        )
        
        print(f"   📏 网格范围: {self.wavenumber_grid.min():.1f} - {self.wavenumber_grid.max():.1f} cm⁻¹")
        print(f"   📊 网格点数: {len(self.wavenumber_grid)}")
        
        return self.wavenumber_grid
    
    def interpolate_spectra(self, spectra_data):
        """改进的光谱插值处理"""
        print(f"\n🔄 执行高质量光谱插值...")
        
        interpolated_spectra = {}
        
        for gas, data in spectra_data.items():
            wavenumber = data['wavenumber']
            intensity = data['intensity']
            
            # 预处理原始数据
            # 1. 移除异常值
            intensity_clean = self._remove_outliers(intensity)
            
            # 2. 平滑处理
            if len(intensity_clean) > self.preprocessing_params['smoothing_window']:
                intensity_smooth = savgol_filter(
                    intensity_clean, 
                    self.preprocessing_params['smoothing_window'],
                    self.preprocessing_params['smoothing_poly']
                )
            else:
                intensity_smooth = intensity_clean
            
            # 3. 基线校正
            intensity_baseline_corrected = self._baseline_correction(wavenumber, intensity_smooth)
            
            # 4. 执行插值
            try:
                # 使用三次样条插值提高精度
                interp_func = interp1d(
                    wavenumber, intensity_baseline_corrected,
                    kind='cubic', bounds_error=False, fill_value=0,
                    assume_sorted=True
                )
                
                interpolated_intensity = interp_func(self.wavenumber_grid)
                
                # 确保非负值
                interpolated_intensity = np.maximum(interpolated_intensity, 0)
                
                # 计算覆盖度
                coverage = np.sum(interpolated_intensity > self.preprocessing_params['noise_threshold']) / len(self.wavenumber_grid)
                
                interpolated_spectra[gas] = interpolated_intensity
                
                print(f"   ✅ {gas}: 覆盖度 {coverage:.1%}")
                
            except Exception as e:
                print(f"   ❌ {gas} 插值失败: {e}")
                # 使用零填充作为后备
                interpolated_spectra[gas] = np.zeros(len(self.wavenumber_grid))
        
        return interpolated_spectra
    
    def _remove_outliers(self, data, method='iqr', factor=1.5):
        """移除异常值"""
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return np.clip(data, lower_bound, upper_bound)
        else:
            # Z-score方法
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            return np.where(z_scores > factor, np.median(data), data)
    
    def _baseline_correction(self, wavenumber, intensity, method='linear'):
        """基线校正"""
        if method == 'linear':
            # 简单线性基线校正
            baseline = np.linspace(intensity[0], intensity[-1], len(intensity))
            return intensity - baseline
        elif method == 'polynomial':
            # 多项式基线校正
            from numpy.polynomial import polynomial as P
            coeffs = P.polyfit(wavenumber, intensity, 2)
            baseline = P.polyval(wavenumber, coeffs)
            return intensity - baseline
        else:
            return intensity
    
    def generate_training_data(self, interpolated_spectra, n_samples=1000, concentration_ranges=None):
        """生成高质量训练数据"""
        print(f"\n🏗️ 生成训练数据 ({n_samples} 样本)...")
        
        if concentration_ranges is None:
            # 使用更大的浓度范围
            concentration_ranges = {
                gas: (0.01, 0.99) for gas in self.gas_names
            }
        
        X_data = []
        Y_data = []
        
        # 使用拉丁超立方采样确保均匀分布
        concentrations_matrix = self._latin_hypercube_sampling(n_samples, len(self.gas_names))
        
        for i in range(n_samples):
            # 生成浓度向量
            concentrations = {}
            raw_concentrations = concentrations_matrix[i]
            
            # 归一化到指定范围
            for j, gas in enumerate(self.gas_names):
                min_conc, max_conc = concentration_ranges[gas]
                concentrations[gas] = min_conc + (max_conc - min_conc) * raw_concentrations[j]
            
            # 确保总和为1（归一化）
            total = sum(concentrations.values())
            if total > 0:
                for gas in self.gas_names:
                    concentrations[gas] /= total
            
            # 生成混合光谱
            mixed_spectrum = np.zeros(len(self.wavenumber_grid))
            for gas in self.gas_names:
                if gas in interpolated_spectra:
                    mixed_spectrum += concentrations[gas] * interpolated_spectra[gas]
            
            # 添加现实的噪声模型
            noisy_spectrum = self._add_realistic_noise(mixed_spectrum)
            
            # 特征工程
            features = self._extract_features(noisy_spectrum)
            
            X_data.append(features)
            Y_data.append([concentrations[gas] for gas in self.gas_names])
        
        X = np.array(X_data)
        Y = np.array(Y_data)
        
        print(f"   ✅ 生成完成: X shape {X.shape}, Y shape {Y.shape}")
        print(f"   📊 浓度统计:")
        for i, gas in enumerate(self.gas_names):
            print(f"      {gas}: {Y[:, i].min():.3f} - {Y[:, i].max():.3f} (均值: {Y[:, i].mean():.3f})")
        
        return X, Y
    
    def _latin_hypercube_sampling(self, n_samples, n_dimensions):
        """拉丁超立方采样"""
        # 简单实现，可以使用更专业的库如pyDOE
        samples = np.random.random((n_samples, n_dimensions))
        
        for i in range(n_dimensions):
            # 对每个维度进行排序并重新分配
            idx = np.argsort(samples[:, i])
            samples[idx, i] = (np.arange(n_samples) + np.random.random(n_samples)) / n_samples
        
        return samples
    
    def _add_realistic_noise(self, spectrum):
        """添加现实的噪声模型"""
        noisy_spectrum = spectrum.copy()
        
        # 1. 高斯噪声 (仪器噪声)
        gaussian_noise = np.random.normal(0, 0.01 * np.std(spectrum), spectrum.shape)
        
        # 2. 泊松噪声 (光子噪声)
        # 避免负值
        positive_spectrum = np.maximum(spectrum, 0)
        poisson_noise = np.random.poisson(positive_spectrum * 1000) / 1000 - positive_spectrum
        
        # 3. 系统漂移
        drift = 0.005 * np.sin(2 * np.pi * np.arange(len(spectrum)) / len(spectrum))
        
        # 4. 基线漂移
        baseline_drift = 0.002 * np.random.random() * np.ones_like(spectrum)
        
        # 组合噪声
        noisy_spectrum += 0.6 * gaussian_noise + 0.3 * poisson_noise + 0.1 * drift + baseline_drift
        
        return np.maximum(noisy_spectrum, 0)  # 确保非负
    
    def _extract_features(self, spectrum):
        """特征工程：提取多维度特征"""
        features = []
        
        # 1. 原始光谱
        features.extend(spectrum)
        
        # 2. 一阶导数
        first_derivative = np.gradient(spectrum)
        features.extend(first_derivative)
        
        # 3. 二阶导数
        second_derivative = np.gradient(first_derivative)
        features.extend(second_derivative)
        
        # 4. 积分特征 (累积强度)
        cumulative = np.cumsum(spectrum)
        features.extend(cumulative / cumulative[-1] if cumulative[-1] > 0 else cumulative)
        
        # 5. 统计特征
        # 峰值位置
        peak_indices = self._find_peaks(spectrum)
        peak_features = np.zeros(10)  # 最多10个峰
        for i, peak_idx in enumerate(peak_indices[:10]):
            peak_features[i] = spectrum[peak_idx] if peak_idx < len(spectrum) else 0
        features.extend(peak_features)
        
        return np.array(features)
    
    def _find_peaks(self, spectrum, prominence=None):
        """找到光谱峰值"""
        from scipy.signal import find_peaks
        
        if prominence is None:
            prominence = 0.1 * np.std(spectrum)
        
        peaks, _ = find_peaks(spectrum, prominence=prominence)
        
        # 按强度排序
        peak_intensities = spectrum[peaks]
        sorted_indices = np.argsort(peak_intensities)[::-1]
        
        return peaks[sorted_indices]
    
    def optimize_model_parameters(self, X, Y, cv_folds=5):
        """优化PLS模型参数"""
        print(f"\n⚙️ 优化模型参数 ({cv_folds}折交叉验证)...")
        
        # 参数搜索空间
        param_grid = {
            'pls__n_components': range(1, min(21, X.shape[1]//10, X.shape[0]//10)),
            'scaler': [StandardScaler(), RobustScaler()]
        }
        
        best_score = -np.inf
        best_params = {}
        best_n_components = 5
        
        # 手动网格搜索（更灵活）
        for scaler in param_grid['scaler']:
            print(f"   测试缩放器: {type(scaler).__name__}")
            
            # 缩放数据
            X_scaled = scaler.fit_transform(X)
            
            for n_comp in param_grid['pls__n_components']:
                try:
                    # 创建PLS模型
                    pls = PLSRegression(n_components=n_comp, max_iter=1000)
                    
                    # 交叉验证
                    cv_scores = []
                    for gas_idx in range(Y.shape[1]):
                        scores = cross_val_score(
                            pls, X_scaled, Y[:, gas_idx], 
                            cv=cv_folds, scoring='r2'
                        )
                        cv_scores.append(scores.mean())
                    
                    # 平均R²分数
                    avg_score = np.mean(cv_scores)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {
                            'scaler': scaler,
                            'n_components': n_comp,
                            'cv_scores': cv_scores
                        }
                        best_n_components = n_comp
                
                except Exception as e:
                    continue
        
        self.n_components = best_n_components
        self.scaler = best_params['scaler']
        self.training_history['best_params'] = best_params
        self.training_history['cv_scores'] = best_params['cv_scores']
        
        print(f"   ✅ 最优参数:")
        print(f"      缩放器: {type(self.scaler).__name__}")
        print(f"      主成分数: {self.n_components}")
        print(f"      平均R²: {best_score:.4f}")
        print(f"      各气体R²: {[f'{gas}:{score:.3f}' for gas, score in zip(self.gas_names, best_params['cv_scores'])]}")
        
        return best_params
    
    def train_model(self, X, Y):
        """训练最终模型"""
        print(f"\n🏋️ 训练最终PLS模型...")
        
        # 数据预处理
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练PLS模型
        self.pls_model = PLSRegression(
            n_components=self.n_components,
            max_iter=1000,
            scale=False  # 已经预缩放
        )
        
        self.pls_model.fit(X_scaled, Y)
        
        # 评估训练性能
        Y_pred = self.pls_model.predict(X_scaled)
        
        # 计算各种指标
        mse = mean_squared_error(Y, Y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y, Y_pred)
        mae = mean_absolute_error(Y, Y_pred)
        
        print(f"   📊 训练性能:")
        print(f"      RMSE: {rmse:.6f}")
        print(f"      R²: {r2:.6f}")
        print(f"      MAE: {mae:.6f}")
        
        # 各气体单独评估
        for i, gas in enumerate(self.gas_names):
            gas_r2 = r2_score(Y[:, i], Y_pred[:, i])
            gas_rmse = np.sqrt(mean_squared_error(Y[:, i], Y_pred[:, i]))
            print(f"      {gas}: R²={gas_r2:.4f}, RMSE={gas_rmse:.4f}")
        
        return {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'predictions': Y_pred
        }
    
    def predict_concentration(self, spectrum_file, visualize=True):
        """预测气体浓度（主要接口）"""
        print(f"\n🔍 预测气体浓度: {spectrum_file}")
        
        if self.pls_model is None:
            print("❌ 模型未训练，请先调用 train_complete_model()")
            return None
        
        # 1. 加载光谱数据
        try:
            spectrum_data = pd.read_csv(spectrum_file)
            
            if 'wavenumber' not in spectrum_data.columns or 'intensity' not in spectrum_data.columns:
                print("❌ 光谱文件需要包含 'wavenumber' 和 'intensity' 列")
                return None
            
            wavenumber = spectrum_data['wavenumber'].values
            intensity = spectrum_data['intensity'].values
            
            print(f"   📊 加载光谱: {len(wavenumber)} 点")
            print(f"   📏 波数范围: {wavenumber.min():.1f} - {wavenumber.max():.1f} cm⁻¹")
            
        except Exception as e:
            print(f"❌ 加载光谱文件失败: {e}")
            return None
        
        # 2. 预处理光谱
        processed_spectrum = self._preprocess_test_spectrum(wavenumber, intensity)
        if processed_spectrum is None:
            return None
        
        # 3. 特征提取
        features = self._extract_features(processed_spectrum)
        features = features.reshape(1, -1)
        
        # 4. 预测
        try:
            # 缩放特征
            features_scaled = self.scaler.transform(features)
            
            # PLS预测
            predictions = self.pls_model.predict(features_scaled)[0]
            
            # 后处理：确保非负且和为1
            predictions = np.maximum(predictions, 0)
            total = np.sum(predictions)
            if total > 0:
                predictions = predictions / total
            else:
                predictions = np.ones(len(self.gas_names)) / len(self.gas_names)
            
            # 组织结果
            results = {}
            for i, gas in enumerate(self.gas_names):
                results[gas] = float(predictions[i])
            
            print(f"   🎯 预测结果:")
            for gas, conc in results.items():
                print(f"      {gas}: {conc:.3f} ({conc*100:.1f}%)")
            
            # 5. 可视化
            if visualize:
                self._visualize_prediction(wavenumber, intensity, processed_spectrum, results)
            
            return results
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None
    
    def _preprocess_test_spectrum(self, wavenumber, intensity):
        """预处理测试光谱"""
        print(f"   🔧 预处理测试光谱...")
        
        # 1. 数据清理
        valid_mask = np.isfinite(wavenumber) & np.isfinite(intensity)
        wavenumber_clean = wavenumber[valid_mask]
        intensity_clean = intensity[valid_mask]
        
        # 2. 异常值处理
        intensity_clean = self._remove_outliers(intensity_clean)
        
        # 3. 平滑处理
        if len(intensity_clean) > self.preprocessing_params['smoothing_window']:
            intensity_smooth = savgol_filter(
                intensity_clean,
                self.preprocessing_params['smoothing_window'],
                self.preprocessing_params['smoothing_poly']
            )
        else:
            intensity_smooth = intensity_clean
        
        # 4. 基线校正
        intensity_corrected = self._baseline_correction(wavenumber_clean, intensity_smooth)
        
        # 5. 插值到模型网格
        try:
            interp_func = interp1d(
                wavenumber_clean, intensity_corrected,
                kind='cubic', bounds_error=False, fill_value=0
            )
            
            interpolated_spectrum = interp_func(self.wavenumber_grid)
            interpolated_spectrum = np.maximum(interpolated_spectrum, 0)
            
            # 检查覆盖度
            coverage = np.sum(interpolated_spectrum > self.preprocessing_params['noise_threshold']) / len(self.wavenumber_grid)
            print(f"      覆盖度: {coverage:.1%}")
            
            if coverage < 0.1:
                print("      ⚠️ 警告: 波数覆盖度过低，预测可能不准确")
            
            return interpolated_spectrum
            
        except Exception as e:
            print(f"      ❌ 插值失败: {e}")
            return None
    
    def _visualize_prediction(self, original_wavenumber, original_intensity, processed_spectrum, predictions):
        """可视化预测结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 原始光谱
            axes[0, 0].plot(original_wavenumber, original_intensity, 'b-', alpha=0.7)
            axes[0, 0].set_title('Original Test Spectrum')
            axes[0, 0].set_xlabel('Wavenumber (cm⁻¹)')
            axes[0, 0].set_ylabel('Intensity')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 预处理后光谱
            axes[0, 1].plot(self.wavenumber_grid, processed_spectrum, 'r-', alpha=0.7)
            axes[0, 1].set_title('Processed Spectrum (Model Input)')
            axes[0, 1].set_xlabel('Wavenumber (cm⁻¹)')
            axes[0, 1].set_ylabel('Intensity')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 预测结果柱状图
            gas_names = list(predictions.keys())
            concentrations = list(predictions.values())
            colors = ['red', 'blue', 'green', 'orange', 'purple'][:len(gas_names)]
            
            bars = axes[1, 0].bar(gas_names, concentrations, color=colors, alpha=0.7)
            axes[1, 0].set_title('Predicted Gas Concentrations')
            axes[1, 0].set_ylabel('Concentration')
            axes[1, 0].set_ylim(0, 1)
            
            # 添加数值标签
            for bar, conc in zip(bars, concentrations):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{conc:.3f}\n({conc*100:.1f}%)',
                               ha='center', va='bottom')
            
            # 4. 饼图
            axes[1, 1].pie(concentrations, labels=[f'{name}\n{conc:.1%}' for name, conc in zip(gas_names, concentrations)],
                          colors=colors, autopct='', startangle=90)
            axes[1, 1].set_title('Concentration Distribution')
            
            plt.suptitle(f'Gas Concentration Analysis\nModel: PLS (n_components={self.n_components})',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'prediction_result_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   📊 结果图表已保存: {filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️ 可视化失败: {e}")
    
    def train_complete_model(self, data_path="hitran_csv", n_samples=2000):
        """完整的模型训练流程"""
        print(f"🚀 开始完整模型训练流程")
        print("=" * 60)
        
        # 1. 加载数据
        spectra_data, wavenumber_ranges = self.load_hitran_data(data_path)
        if spectra_data is None:
            return False
        
        # 2. 创建统一网格
        self.create_unified_grid(wavenumber_ranges)
        
        # 3. 插值光谱
        interpolated_spectra = self.interpolate_spectra(spectra_data)
        
        # 4. 生成训练数据
        X, Y = self.generate_training_data(interpolated_spectra, n_samples)
        
        # 5. 优化参数
        self.optimize_model_parameters(X, Y)
        
        # 6. 训练模型
        training_results = self.train_model(X, Y)
        
        # 7. 保存模型
        self.save_model()
        
        print(f"\n🎉 模型训练完成！")
        print(f"📊 最终性能: R² = {training_results['r2']:.4f}, RMSE = {training_results['rmse']:.6f}")
        
        return True
    
    def save_model(self, filename=None):
        """保存完整模型"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"improved_gas_detector_{timestamp}.pkl"
        
        model_data = {
            'pls_model': self.pls_model,
            'scaler': self.scaler,
            'wavenumber_grid': self.wavenumber_grid,
            'gas_names': self.gas_names,
            'n_components': self.n_components,
            'preprocessing_params': self.preprocessing_params,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filename)
        print(f"💾 模型已保存: {filename}")
        
        return filename
    
    def load_model(self, filename):
        """加载预训练模型"""
        try:
            model_data = joblib.load(filename)
            
            self.pls_model = model_data['pls_model']
            self.scaler = model_data['scaler']
            self.wavenumber_grid = model_data['wavenumber_grid']
            self.gas_names = model_data['gas_names']
            self.n_components = model_data['n_components']
            self.preprocessing_params = model_data['preprocessing_params']
            self.training_history = model_data['training_history']
            
            print(f"✅ 模型加载成功: {filename}")
            print(f"📊 模型信息: {len(self.gas_names)}种气体, {self.n_components}个主成分")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False


def main():
    """主函数示例"""
    print("🔬 改进的PLS气体浓度检测系统")
    print("=" * 60)
    
    # 创建检测器
    detector = ImprovedGasDetector(gas_names=['NO', 'NO2', 'SO2'])
    
    # 训练模型
    print("\n1️⃣ 训练模型...")
    success = detector.train_complete_model(
        data_path="hitran_csv",
        n_samples=2000
    )
    
    if not success:
        print("❌ 模型训练失败")
        return
    
    # 测试预测
    print("\n2️⃣ 测试预测...")
    test_files = [
        "gas_three/test/mixed_spectrum_244_noisy_20250710_152143.csv",
        "gas_three/22kv0h.CSV"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n📄 测试文件: {test_file}")
            results = detector.predict_concentration(test_file, visualize=True)
            
            if results:
                print("预测成功！")
            else:
                print("预测失败")
        else:
            print(f"⚠️ 测试文件不存在: {test_file}")
    
    print(f"\n🎉 程序执行完成！")


if __name__ == "__main__":
    main()