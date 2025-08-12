# detect_spectrum.py
# 专门用于检测光谱数据中的气体浓度

import pandas as pd
import numpy as np
import joblib
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class SpectrumDetector:
    """光谱检测器 - 支持多种模型"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def load_enhanced_model(self):
        """加载Enhanced模型"""
        try:
            model_path = "data/models/enhanced_pls_model.pkl"
            scaler_X_path = "data/models/scaler_X.pkl"
            scaler_Y_path = "data/models/scaler_Y.pkl"
            
            if all(os.path.exists(p) for p in [model_path, scaler_X_path, scaler_Y_path]):
                self.models['enhanced'] = {
                    'model': joblib.load(model_path),
                    'scaler_X': joblib.load(scaler_X_path),
                    'scaler_Y': joblib.load(scaler_Y_path)
                }
                print("✓ Enhanced model loaded")
                return True
            else:
                print("✗ Enhanced model files not found")
                return False
        except Exception as e:
            print(f"✗ Enhanced model loading failed: {e}")
            return False
    
    def load_standard_model(self):
        """加载Standard模型"""
        try:
            model_path = "data/models/standard/standard_pls_model.pkl"
            scaler_X_path = "data/models/standard/standard_scaler_X.pkl"
            scaler_Y_path = "data/models/standard/standard_scaler_Y.pkl"
            
            if all(os.path.exists(p) for p in [model_path, scaler_X_path, scaler_Y_path]):
                self.models['standard'] = {
                    'model': joblib.load(model_path),
                    'scaler_X': joblib.load(scaler_X_path),
                    'scaler_Y': joblib.load(scaler_Y_path)
                }
                print("✓ Standard model loaded")
                return True
            else:
                print("✗ Standard model files not found")
                return False
        except Exception as e:
            print(f"✗ Standard model loading failed: {e}")
            return False
    
    def load_reference_wavenumbers(self):
        """加载参考波数轴"""
        # 尝试从不同位置加载参考波数
        possible_files = [
            "data/processed/interpolated_spectra.csv",
            "data/processed/standard_interpolated_spectra.csv"
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                ref_df = pd.read_csv(file_path)
                self.reference_wavenumbers = ref_df['wavenumber'].values
                print(f"✓ Reference wavenumbers loaded from {file_path}")
                print(f"  Range: {self.reference_wavenumbers.min():.1f} - {self.reference_wavenumbers.max():.1f} cm⁻¹")
                return True
        
        print("✗ Reference wavenumbers not found")
        return False
    
    def preprocess_spectrum(self, file_path):
        """预处理光谱数据"""
        print(f"Loading spectrum from: {file_path}")
        
        # 读取光谱数据
        df = pd.read_csv(file_path)
        
        # 检查数据格式
        if 'wavenumber' in df.columns and 'intensity' in df.columns:
            wavenumbers = df['wavenumber'].values
            intensities = df['intensity'].values
        elif len(df.columns) >= 2:
            # 假设第一列是波数，第二列是强度
            wavenumbers = df.iloc[:, 0].values
            intensities = df.iloc[:, 1].values
        else:
            raise ValueError("Cannot identify wavenumber and intensity columns")
        
        print(f"  Original data: {len(wavenumbers)} points")
        print(f"  Wavenumber range: {wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹")
        print(f"  Intensity range: {intensities.min():.2e} - {intensities.max():.2e}")
        
        # 插值到参考波数轴
        if hasattr(self, 'reference_wavenumbers'):
            print("  Interpolating to reference wavenumber axis...")
            
            # 找到重叠范围
            overlap_min = max(wavenumbers.min(), self.reference_wavenumbers.min())
            overlap_max = min(wavenumbers.max(), self.reference_wavenumbers.max())
            
            if overlap_min >= overlap_max:
                raise ValueError("No overlap with reference wavenumber range")
            
            print(f"  Overlap range: {overlap_min:.1f} - {overlap_max:.1f} cm⁻¹")
            
            # 限制参考波数到重叠范围
            ref_mask = (self.reference_wavenumbers >= overlap_min) & (self.reference_wavenumbers <= overlap_max)
            ref_wavenumbers_clipped = self.reference_wavenumbers[ref_mask]
            
            # 插值
            interpolator = interp1d(wavenumbers, intensities, 
                                  kind='linear', bounds_error=False, fill_value=0)
            interpolated_intensities = interpolator(ref_wavenumbers_clipped)
            
            print(f"  Interpolated data: {len(interpolated_intensities)} points")
            
            return interpolated_intensities
        else:
            # 直接使用原始数据
            print("  Using original data (no reference axis)")
            return intensities
    
    def predict_with_model(self, spectral_data, model_name):
        """使用指定模型进行预测"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model_info = self.models[model_name]
        model = model_info['model']
        scaler_X = model_info['scaler_X']
        scaler_Y = model_info['scaler_Y']
        
        # 确保数据形状正确
        if spectral_data.ndim == 1:
            spectral_data = spectral_data.reshape(1, -1)
        
        # 检查特征维度
        expected_features = model.n_features_in_
        actual_features = spectral_data.shape[1]
        
        if actual_features != expected_features:
            print(f"  Adjusting feature dimensions: {actual_features} → {expected_features}")
            
            if actual_features > expected_features:
                spectral_data = spectral_data[:, :expected_features]
            else:
                padding = np.zeros((spectral_data.shape[0], expected_features - actual_features))
                spectral_data = np.hstack([spectral_data, padding])
        
        # 标准化和预测
        X_scaled = scaler_X.transform(spectral_data)
        Y_pred_scaled = model.predict(X_scaled)
        Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
        
        # 确保浓度为正且和为1
        Y_pred = np.maximum(Y_pred, 0)
        Y_pred = Y_pred / Y_pred.sum(axis=1, keepdims=True)
        
        return Y_pred[0]  # 返回单个预测结果
    
    def detect_gases(self, file_path):
        """检测光谱中的气体浓度"""
        print("="*60)
        print("GAS CONCENTRATION DETECTION")
        print("="*60)
        print(f"Target file: {file_path}")
        print()
        
        # 预处理光谱
        try:
            spectral_data = self.preprocess_spectrum(file_path)
        except Exception as e:
            print(f"ERROR in preprocessing: {e}")
            return None
        
        print()
        print("DETECTION RESULTS:")
        print("="*40)
        
        results = {}
        gas_names = ['NO', 'NO2', 'SO2']
        
        # 使用所有可用模型进行预测
        for model_name in self.models.keys():
            try:
                print(f"\n{model_name.upper()} Model Prediction:")
                prediction = self.predict_with_model(spectral_data, model_name)
                
                results[model_name] = prediction
                
                for i, gas in enumerate(gas_names):
                    conc = prediction[i]
                    print(f"  {gas}: {conc:.4f} ({conc*100:.1f}%)")
                
                total = prediction.sum()
                print(f"  Total: {total:.4f}")
                
            except Exception as e:
                print(f"  ERROR with {model_name} model: {e}")
        
        # 如果有多个模型，计算平均值
        if len(results) > 1:
            print(f"\nAVERAGE Prediction (from {len(results)} models):")
            avg_prediction = np.mean(list(results.values()), axis=0)
            
            for i, gas in enumerate(gas_names):
                conc = avg_prediction[i]
                print(f"  {gas}: {conc:.4f} ({conc*100:.1f}%)")
        
        # 保存结果
        self.save_detection_results(file_path, results)
        
        return results
    
    def save_detection_results(self, input_file, results):
        """保存检测结果"""
        os.makedirs("data/results/detection", exist_ok=True)
        
        # 生成输出文件名
        input_filename = os.path.basename(input_file).replace('.csv', '')
        output_file = f"data/results/detection/{input_filename}_detection_results.csv"
        
        # 创建结果DataFrame
        gas_names = ['NO', 'NO2', 'SO2']
        result_data = []
        
        for model_name, prediction in results.items():
            row = {'Model': model_name}
            for i, gas in enumerate(gas_names):
                row[f'{gas}_concentration'] = prediction[i]
                row[f'{gas}_percentage'] = prediction[i] * 100
            result_data.append(row)
        
        # 如果有多个模型，添加平均值
        if len(results) > 1:
            avg_prediction = np.mean(list(results.values()), axis=0)
            row = {'Model': 'AVERAGE'}
            for i, gas in enumerate(gas_names):
                row[f'{gas}_concentration'] = avg_prediction[i]
                row[f'{gas}_percentage'] = avg_prediction[i] * 100
            result_data.append(row)
        
        results_df = pd.DataFrame(result_data)
        results_df.to_csv(output_file, index=False)
        
        print(f"\n📄 Results saved to: {output_file}")

def detect_spectrum_file(file_path):
    """检测单个光谱文件"""
    detector = SpectrumDetector()
    
    # 加载所有可用模型
    print("Loading available models...")
    detector.load_enhanced_model()
    detector.load_standard_model()
    
    if not detector.models:
        print("ERROR: No models available. Please build models first.")
        return None
    
    # 加载参考波数
    detector.load_reference_wavenumbers()
    
    # 执行检测
    results = detector.detect_gases(file_path)
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # 默认使用您指定的文件
        file_path = "E:/generate_mixture/gas_three/test/mixed_spectrum_244_clean_20250725_151622.csv"
    
    print("SPECTRUM DETECTION TOOL")
    print("="*40)
    print(f"Analyzing: {file_path}")
    print()
    
    if os.path.exists(file_path):
        results = detect_spectrum_file(file_path)
        
        if results:
            print("\n" + "="*60)
            print("DETECTION COMPLETED SUCCESSFULLY!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("DETECTION FAILED!")
            print("="*60)
    else:
        print(f"ERROR: File not found - {file_path}")