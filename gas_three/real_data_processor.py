# real_data_processor.py
# 真实实验数据处理模块 - 处理用户提供的真实光谱数据

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

class RealDataProcessor:
    """
    真实数据处理器，提供：
    1. 多种格式数据读取
    2. 光谱预处理和质量检查
    3. 与仿真数据的对齐
    4. 数据质量评估
    """
    
    def __init__(self, reference_wavenumbers_file: str = "data/processed/interpolated_spectra.csv"):
        self.reference_wavenumbers_file = reference_wavenumbers_file
        self.reference_wavenumbers = None
        self.processed_data = {}
        self.quality_metrics = {}
        
        # 加载参考波数
        if os.path.exists(reference_wavenumbers_file):
            ref_df = pd.read_csv(reference_wavenumbers_file)
            self.reference_wavenumbers = ref_df['wavenumber'].values
            print(f"📐 参考波数范围: {self.reference_wavenumbers.min():.1f} - {self.reference_wavenumbers.max():.1f} cm⁻¹")
        
    def load_real_data(self, file_path: str, data_format: str = 'auto', 
                      wavenumber_col: str = 'wavenumber', 
                      intensity_col: str = 'intensity') -> pd.DataFrame:
        """
        加载真实光谱数据
        
        Parameters:
        - file_path: 数据文件路径
        - data_format: 'csv', 'txt', 'excel', 'auto'
        - wavenumber_col: 波数列名
        - intensity_col: 强度列名
        """
        print(f"📂 加载真实数据: {file_path}")
        
        # 自动检测文件格式
        if data_format == 'auto':
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                data_format = 'csv'
            elif ext in ['.txt', '.dat']:
                data_format = 'txt'
            elif ext in ['.xlsx', '.xls']:
                data_format = 'excel'
            else:
                raise ValueError(f"不支持的文件格式: {ext}")
        
        # 读取数据
        try:
            if data_format == 'csv':
                df = pd.read_csv(file_path)
            elif data_format == 'txt':
                # 尝试不同的分隔符
                for sep in ['\t', ' ', ',', ';']:
                    try:
                        df = pd.read_csv(file_path, sep=sep)
                        if df.shape[1] >= 2:
                            break
                    except:
                        continue
                else:
                    raise ValueError("无法解析TXT文件")
            elif data_format == 'excel':
                df = pd.read_excel(file_path)
            
            print(f"   📊 数据形状: {df.shape}")
            print(f"   📋 列名: {list(df.columns)}")
            
        except Exception as e:
            raise ValueError(f"读取文件失败: {str(e)}")
        
        # 验证必要的列
        if wavenumber_col not in df.columns:
            # 尝试自动识别波数列
            possible_wave_cols = ['wavenumber', 'wave', 'wn', 'cm-1', 'frequency']
            for col in possible_wave_cols:
                if col in df.columns.str.lower().values:
                    wavenumber_col = df.columns[df.columns.str.lower() == col][0]
                    break
            else:
                print(f"   ⚠️ 未找到波数列，使用第一列: {df.columns[0]}")
                wavenumber_col = df.columns[0]
        
        if intensity_col not in df.columns:
            # 尝试自动识别强度列
            possible_intensity_cols = ['intensity', 'absorbance', 'abs', 'transmission', 'signal']
            for col in possible_intensity_cols:
                if col in df.columns.str.lower().values:
                    intensity_col = df.columns[df.columns.str.lower() == col][0]
                    break
            else:
                # 使用除波数列外的第一列
                intensity_cols = [col for col in df.columns if col != wavenumber_col]
                if intensity_cols:
                    intensity_col = intensity_cols[0]
                    print(f"   ⚠️ 未找到强度列，使用: {intensity_col}")
                else:
                    raise ValueError("无法识别强度列")
        
        # 提取波数和强度数据
        wavenumbers = df[wavenumber_col].values
        intensities = df[intensity_col].values
        
        # 基本数据验证
        self._validate_spectral_data(wavenumbers, intensities)
        
        # 创建标准格式的DataFrame
        processed_df = pd.DataFrame({
            'wavenumber': wavenumbers,
            'intensity': intensities
        })
        
        # 按波数排序
        processed_df = processed_df.sort_values('wavenumber').reset_index(drop=True)
        
        print(f"   ✅ 数据加载成功")
        print(f"   📐 波数范围: {wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹")
        print(f"   📊 强度范围: {intensities.min():.2e} - {intensities.max():.2e}")
        
        return processed_df
    
    def _validate_spectral_data(self, wavenumbers: np.ndarray, intensities: np.ndarray):
        """验证光谱数据质量"""
        
        issues = []
        
        # 检查数据长度
        if len(wavenumbers) != len(intensities):
            raise ValueError("波数和强度数据长度不匹配")
        
        if len(wavenumbers) < 100:
            issues.append("数据点太少（<100）")
        
        # 检查NaN值
        if np.isnan(wavenumbers).any():
            issues.append("波数数据包含NaN值")
        if np.isnan(intensities).any():
            issues.append("强度数据包含NaN值")
        
        # 检查波数单调性
        if not np.all(np.diff(wavenumbers) > 0):
            issues.append("波数数据非单调递增")
        
        # 检查数据范围合理性
        if wavenumbers.min() < 0 or wavenumbers.max() > 10000:
            issues.append("波数范围异常")
        
        # 检查强度值
        if np.all(intensities == 0):
            issues.append("所有强度值为零")
        
        if issues:
            print(f"   ⚠️ 数据质量问题: {'; '.join(issues)}")
        else:
            print(f"   ✅ 数据质量检查通过")
    
    def preprocess_spectrum(self, df: pd.DataFrame, 
                          baseline_correction: bool = True,
                          smoothing: bool = True,
                          normalization: str = 'minmax') -> pd.DataFrame:
        """
        光谱预处理
        
        Parameters:
        - baseline_correction: 是否进行基线校正
        - smoothing: 是否平滑
        - normalization: 'minmax', 'std', 'area', None
        """
        print("🔧 光谱预处理...")
        
        wavenumbers = df['wavenumber'].values
        intensities = df['intensity'].values.copy()
        
        # 1. 基线校正
        if baseline_correction:
            print("   📈 基线校正...")
            # 使用多项式拟合去除基线
            baseline = self._estimate_baseline(wavenumbers, intensities)
            intensities = intensities - baseline
        
        # 2. 平滑处理
        if smoothing:
            print("   🌊 光谱平滑...")
            # 使用Savitzky-Golay滤波
            window_length = min(11, len(intensities) // 10 * 2 + 1)  # 确保为奇数
            if window_length >= 3:
                intensities = signal.savgol_filter(intensities, window_length, 3)
        
        # 3. 归一化
        if normalization:
            print(f"   📏 数据归一化 ({normalization})...")
            if normalization == 'minmax':
                intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
            elif normalization == 'std':
                intensities = (intensities - intensities.mean()) / intensities.std()
            elif normalization == 'area':
                intensities = intensities / np.trapz(np.abs(intensities), wavenumbers)
        
        # 创建预处理后的DataFrame
        processed_df = pd.DataFrame({
            'wavenumber': wavenumbers,
            'intensity': intensities
        })
        
        print("   ✅ 预处理完成")
        
        return processed_df
    
    def _estimate_baseline(self, wavenumbers: np.ndarray, intensities: np.ndarray, 
                          poly_order: int = 3) -> np.ndarray:
        """估计光谱基线"""
        
        # 使用多项式拟合估计基线
        # 首先找到可能的基线点（局部最小值）
        from scipy.signal import find_peaks
        
        # 反转信号找峰（原信号的谷）  
        inverted = -intensities
        peaks, _ = find_peaks(inverted, distance=len(intensities)//20)
        
        if len(peaks) < 3:
            # 如果峰太少，使用边界点和中间的几个点
            peaks = np.linspace(0, len(intensities)-1, 5).astype(int)
        
        # 多项式拟合基线点
        baseline_points = intensities[peaks]
        baseline_wave = wavenumbers[peaks]
        
        poly_coef = np.polyfit(baseline_wave, baseline_points, poly_order)
        baseline = np.polyval(poly_coef, wavenumbers)
        
        return baseline
    
    def align_with_reference(self, df: pd.DataFrame, 
                           interpolation_method: str = 'linear') -> pd.DataFrame:
        """
        将光谱数据对齐到参考波数轴
        """
        if self.reference_wavenumbers is None:
            print("   ⚠️ 没有参考波数，跳过对齐")
            return df
        
        print("🎯 与参考波数对齐...")
        
        wavenumbers = df['wavenumber'].values
        intensities = df['intensity'].values
        
        # 检查重叠范围
        overlap_min = max(wavenumbers.min(), self.reference_wavenumbers.min())
        overlap_max = min(wavenumbers.max(), self.reference_wavenumbers.max())
        
        if overlap_min >= overlap_max:
            raise ValueError("与参考波数没有重叠范围")
        
        print(f"   📐 重叠范围: {overlap_min:.1f} - {overlap_max:.1f} cm⁻¹")
        
        # 限制参考波数到重叠范围
        ref_mask = (self.reference_wavenumbers >= overlap_min) & (self.reference_wavenumbers <= overlap_max)
        ref_wavenumbers_aligned = self.reference_wavenumbers[ref_mask]
        
        # 插值到参考波数轴
        interpolator = interp1d(wavenumbers, intensities, 
                              kind=interpolation_method, 
                              bounds_error=False, fill_value=0)
        
        aligned_intensities = interpolator(ref_wavenumbers_aligned)
        
        # 创建对齐后的DataFrame
        aligned_df = pd.DataFrame({
            'wavenumber': ref_wavenumbers_aligned,
            'intensity': aligned_intensities
        })
        
        print(f"   ✅ 对齐完成，数据点数: {len(aligned_df)}")
        
        return aligned_df
    
    def assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """评估数据质量"""
        
        print("📋 数据质量评估...")
        
        wavenumbers = df['wavenumber'].values
        intensities = df['intensity'].values
        
        quality_metrics = {
            'n_points': len(df),
            'wavenumber_range': [float(wavenumbers.min()), float(wavenumbers.max())],
            'wavenumber_resolution': float(np.median(np.diff(wavenumbers))),
            'intensity_range': [float(intensities.min()), float(intensities.max())],
            'intensity_std': float(intensities.std()),
            'signal_to_noise_ratio': float(np.abs(intensities.mean()) / intensities.std()) if intensities.std() > 0 else np.inf,
            'baseline_stability': float(np.std(intensities[:10]) + np.std(intensities[-10:])) / 2,
            'spectral_coverage': 0.0  # 将在后面计算
        }
        
        # 计算光谱覆盖率（与参考的重叠程度）
        if self.reference_wavenumbers is not None:
            ref_min, ref_max = self.reference_wavenumbers.min(), self.reference_wavenumbers.max()
            data_min, data_max = wavenumbers.min(), wavenumbers.max()
            
            overlap_min = max(ref_min, data_min)
            overlap_max = min(ref_max, data_max)
            
            if overlap_min < overlap_max:
                ref_range = ref_max - ref_min
                overlap_range = overlap_max - overlap_min
                quality_metrics['spectral_coverage'] = float(overlap_range / ref_range)
        
        # 质量评级
        score = 0
        if quality_metrics['n_points'] >= 1000:
            score += 2
        elif quality_metrics['n_points'] >= 500:
            score += 1
        
        if quality_metrics['signal_to_noise_ratio'] >= 10:
            score += 2
        elif quality_metrics['signal_to_noise_ratio'] >= 5:
            score += 1
        
        if quality_metrics['spectral_coverage'] >= 0.8:
            score += 2
        elif quality_metrics['spectral_coverage'] >= 0.6:
            score += 1
        
        quality_levels = {0: 'Poor', 1: 'Poor', 2: 'Fair', 3: 'Fair', 4: 'Good', 5: 'Good', 6: 'Excellent'}
        quality_metrics['quality_score'] = score
        quality_metrics['quality_level'] = quality_levels[score]
        
        self.quality_metrics = quality_metrics
        
        print(f"   📊 数据点数: {quality_metrics['n_points']}")
        print(f"   📐 波数分辨率: {quality_metrics['wavenumber_resolution']:.2f} cm⁻¹")
        print(f"   📶 信噪比: {quality_metrics['signal_to_noise_ratio']:.2f}")
        print(f"   📏 光谱覆盖率: {quality_metrics['spectral_coverage']:.1%}")
        print(f"   🏆 质量评级: {quality_metrics['quality_level']} ({score}/6)")
        
        return quality_metrics
    
    def predict_with_model(self, df: pd.DataFrame, 
                          model_path: str = "data/models/enhanced_pls_model.pkl") -> Dict:
        """
        使用训练好的模型预测气体浓度
        """
        print("🔮 使用模型预测气体浓度...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        model = joblib.load(model_path)
        
        # 检查是否需要缩放器
        scaler_X_path = model_path.replace('enhanced_pls_model.pkl', 'scaler_X.pkl')
        scaler_Y_path = model_path.replace('enhanced_pls_model.pkl', 'scaler_Y.pkl')
        
        use_scaling = os.path.exists(scaler_X_path) and os.path.exists(scaler_Y_path)
        
        if use_scaling:
            scaler_X = joblib.load(scaler_X_path)
            scaler_Y = joblib.load(scaler_Y_path)
        
        # 准备预测数据
        X_real = df['intensity'].values.reshape(1, -1)  # 单个样本
        
        # 检查特征维度
        expected_features = model.n_features_in_
        actual_features = X_real.shape[1]
        
        if actual_features != expected_features:
            print(f"   ⚠️ 特征维度不匹配: 期望 {expected_features}, 实际 {actual_features}")
            
            if actual_features > expected_features:
                # 截取
                X_real = X_real[:, :expected_features]
                print(f"   ✂️ 截取到前 {expected_features} 个特征")
            else:
                # 填充零值
                padding = np.zeros((1, expected_features - actual_features))
                X_real = np.hstack([X_real, padding])
                print(f"   🔧 填充到 {expected_features} 个特征")
        
        # 预测
        if use_scaling:
            X_real_scaled = scaler_X.transform(X_real)
            Y_pred_scaled = model.predict(X_real_scaled)
            Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
        else:
            Y_pred = model.predict(X_real)
        
        # 确保浓度和为1且非负
        Y_pred = np.maximum(Y_pred, 0)
        Y_pred = Y_pred / Y_pred.sum()
        
        # 构建结果
        gas_names = ['NO', 'NO2', 'SO2']
        predictions = {
            'concentrations': {gas: float(Y_pred[0, i]) for i, gas in enumerate(gas_names)},
            'total_concentration': float(Y_pred.sum()),
            'prediction_confidence': self._estimate_confidence(model, X_real if not use_scaling else X_real_scaled),
        }
        
        print(f"   🎯 预测结果:")
        for gas, conc in predictions['concentrations'].items():
            print(f"      {gas}: {conc:.3f} ({conc*100:.1f}%)")
        print(f"   📊 预测置信度: {predictions['prediction_confidence']:.3f}")
        
        return predictions
    
    def _estimate_confidence(self, model, X: np.ndarray) -> float:
        """估计预测置信度（简化版）"""
        
        # 基于模型的方差解释和输入数据的相似度
        try:
            # 计算预测方差（如果模型支持）
            if hasattr(model, 'x_scores_'):
                # 计算输入在PLS空间中的投影强度
                scores = model.transform(X)
                max_score = np.max(np.abs(scores))
                
                # 基于分数的置信度（简化）
                confidence = min(1.0, max_score / 2.0)
                return float(confidence)
        except:
            pass
        
        # 默认中等置信度
        return 0.7
    
    def visualize_processing_steps(self, original_df: pd.DataFrame, 
                                 processed_df: pd.DataFrame,
                                 aligned_df: pd.DataFrame = None,
                                 save_path: str = "data/figures/real_data_processing.png"):
        """可视化处理步骤"""
        
        print("📊 生成处理步骤可视化...")
        
        n_plots = 3 if aligned_df is not None else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        # 原始数据
        axes[0].plot(original_df['wavenumber'], original_df['intensity'], 'b-', alpha=0.7, linewidth=1)
        axes[0].set_title('原始光谱数据')
        axes[0].set_xlabel('Wavenumber (cm⁻¹)')
        axes[0].set_ylabel('Intensity')
        axes[0].grid(True, alpha=0.3)
        
        # 预处理后数据
        axes[1].plot(processed_df['wavenumber'], processed_df['intensity'], 'g-', alpha=0.7, linewidth=1)
        axes[1].set_title('预处理后光谱数据')
        axes[1].set_xlabel('Wavenumber (cm⁻¹)')
        axes[1].set_ylabel('Processed Intensity')
        axes[1].grid(True, alpha=0.3)
        
        # 对齐后数据（如果有）
        if aligned_df is not None and len(axes) > 2:
            axes[2].plot(aligned_df['wavenumber'], aligned_df['intensity'], 'r-', alpha=0.7, linewidth=1)
            axes[2].set_title('对齐后光谱数据')
            axes[2].set_xlabel('Wavenumber (cm⁻¹)')
            axes[2].set_ylabel('Aligned Intensity')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 可视化已保存到: {save_path}")
    
    def save_processed_data(self, df: pd.DataFrame, 
                           save_path: str = "data/raw/processed_real_data.csv"):
        """保存处理后的数据"""
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        
        # 保存质量评估结果
        if self.quality_metrics:
            quality_path = save_path.replace('.csv', '_quality.json')
            with open(quality_path, 'w') as f:
                json.dump(self.quality_metrics, f, indent=2)
        
        print(f"   💾 处理后数据已保存到: {save_path}")

def process_real_data_pipeline(file_path: str, 
                             data_format: str = 'auto',
                             wavenumber_col: str = 'wavenumber',
                             intensity_col: str = 'intensity',
                             predict: bool = True) -> Dict:
    """
    真实数据处理完整流水线
    """
    print("🚀 开始真实数据处理流水线...")
    
    # 创建处理器
    processor = RealDataProcessor()
    
    # 1. 加载数据
    original_df = processor.load_real_data(file_path, data_format, wavenumber_col, intensity_col)
    
    # 2. 预处理
    processed_df = processor.preprocess_spectrum(original_df)
    
    # 3. 对齐
    aligned_df = processor.align_with_reference(processed_df)
    
    # 4. 质量评估
    quality_metrics = processor.assess_data_quality(aligned_df)
    
    # 5. 预测（如果要求）
    predictions = None
    if predict:
        try:
            predictions = processor.predict_with_model(aligned_df)
        except Exception as e:
            print(f"   ⚠️ 预测失败: {str(e)}")
            predictions = None
    
    # 6. 可视化
    processor.visualize_processing_steps(original_df, processed_df, aligned_df)
    
    # 7. 保存结果
    processor.save_processed_data(aligned_df)
    
    result = {
        'original_data': original_df,
        'processed_data': processed_df, 
        'aligned_data': aligned_df,
        'quality_metrics': quality_metrics,
        'predictions': predictions
    }
    
    print("🎉 真实数据处理完成！")
    
    return result

if __name__ == "__main__":
    # 示例用法
    print("💡 真实数据处理器已准备就绪")
    print("   使用方法: process_real_data_pipeline('your_data_file.csv')")
    print("   支持格式: CSV, TXT, Excel")
    print("   请确保数据包含波数和强度列")