# build_standard_model.py
# 使用HITRAN标准气体数据构建'standard'模型

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from scipy.interpolate import interp1d
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt

class StandardGasModelBuilder:
    """
    标准气体模型构建器
    使用HITRAN数据构建three-gas标准预测模型
    """
    
    def __init__(self, hitran_path="E:/generate_mixture/hitran_csv"):
        self.hitran_path = hitran_path
        self.model_name = "standard"
        self.gas_names = ['NO', 'NO2', 'SO2']
        self.model_dir = "data/models/standard"
        self.results_dir = "data/results/standard"
        self.figures_dir = "data/figures/standard"
        
        # 创建目录
        for dir_path in [self.model_dir, self.results_dir, self.figures_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def step1_load_hitran_data(self):
        """步骤1：加载HITRAN标准气体数据"""
        print("STEP 1: Loading HITRAN Standard Gas Data")
        print("="*50)
        
        hitran_data = {}
        
        for gas in self.gas_names:
            file_path = os.path.join(self.hitran_path, f"{gas}.csv")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"HITRAN data not found: {file_path}")
            
            df = pd.read_csv(file_path)
            hitran_data[gas] = df
            
            print(f"  {gas}: {len(df)} spectral lines")
            print(f"       Wavenumber range: {df['nu'].min():.2f} - {df['nu'].max():.2f} cm⁻¹")
            print(f"       Intensity range: {df['sw'].min():.2e} - {df['sw'].max():.2e}")
        
        self.hitran_data = hitran_data
        print("✅ HITRAN data loaded successfully")
        return hitran_data
    
    def step2_preprocess_spectra(self):
        """步骤2：光谱数据预处理和插值"""
        print("\nSTEP 2: Preprocessing and Interpolating Spectra")
        print("="*50)
        
        # 找到共同的波数范围
        min_wavenumbers = []
        max_wavenumbers = []
        
        for gas, df in self.hitran_data.items():
            min_wavenumbers.append(df['nu'].min())
            max_wavenumbers.append(df['nu'].max())
        
        common_min = max(min_wavenumbers)  # 取最大的最小值
        common_max = min(max_wavenumbers)  # 取最小的最大值
        
        print(f"  Overlapping wavenumber range: {common_min:.2f} - {common_max:.2f} cm⁻¹")
        
        # 创建统一的波数轴（步长1 cm⁻¹）
        common_wavenumbers = np.arange(np.ceil(common_min), np.floor(common_max) + 1, 1)
        print(f"  Unified wavenumber axis: {len(common_wavenumbers)} points")
        
        # 插值到统一波数轴
        interpolated_spectra = {}
        interpolated_spectra['wavenumber'] = common_wavenumbers
        
        for gas, df in self.hitran_data.items():
            print(f"  Interpolating {gas} spectrum...")
            
            # 创建插值函数
            interpolator = interp1d(df['nu'].values, df['sw'].values, 
                                  kind='linear', bounds_error=False, fill_value=0)
            
            # 插值到统一波数轴
            interpolated_intensity = interpolator(common_wavenumbers)
            interpolated_spectra[gas] = interpolated_intensity
            
            print(f"    Non-zero points: {np.count_nonzero(interpolated_intensity)}")
        
        # 保存插值后的光谱
        interpolated_df = pd.DataFrame(interpolated_spectra)
        
        save_path = os.path.join("data/processed", f"{self.model_name}_interpolated_spectra.csv")
        interpolated_df.to_csv(save_path, index=False)
        
        self.interpolated_spectra = interpolated_df
        print(f"✅ Interpolated spectra saved to: {save_path}")
        
        return interpolated_df
    
    def step3_generate_training_data(self):
        """步骤3：生成标准模型训练数据"""
        print("\nSTEP 3: Generating Standard Model Training Data")
        print("="*50)
        
        # 从插值光谱中提取数据
        wavenumbers = self.interpolated_spectra['wavenumber'].values
        no_spectrum = self.interpolated_spectra['NO'].values
        no2_spectrum = self.interpolated_spectra['NO2'].values
        so2_spectrum = self.interpolated_spectra['SO2'].values
        
        print(f"  Spectral dimensions: {len(wavenumbers)} wavenumbers")
        
        # 生成浓度组合 - 标准建模使用更规范的组合
        print("  Generating concentration combinations...")
        
        # 标准建模：使用更密集但规律的浓度网格
        no_concentrations = np.arange(0.05, 0.7, 0.025)   # 5%-70%, 步长2.5%
        no2_concentrations = np.arange(0.05, 0.7, 0.025)  # 5%-70%, 步长2.5%
        
        X_data = []
        Y_labels = []
        
        # 标准建模参数
        samples_per_combination = 12  # 每个组合12个样本
        base_noise_level = 0.06       # 基础噪声6%
        spectral_noise_level = 0.04   # 光谱噪声4%
        nonlinear_strength = 0.10     # 非线性强度10%
        
        np.random.seed(2024)  # 设置标准建模种子
        
        valid_combinations = 0
        invalid_combinations = 0
        
        for no_conc in no_concentrations:
            for no2_conc in no2_concentrations:
                so2_conc = 1.0 - no_conc - no2_conc
                
                # 浓度约束：每种气体至少4%
                if so2_conc < 0.04 or no_conc < 0.04 or no2_conc < 0.04:
                    invalid_combinations += 1
                    continue
                
                # 避免极端浓度组合
                if max(no_conc, no2_conc, so2_conc) > 0.75:
                    invalid_combinations += 1
                    continue
                
                valid_combinations += 1
                
                # 为每个组合生成多个样本
                for sample_idx in range(samples_per_combination):
                    
                    # 基础线性混合
                    mixed_base = (no_conc * no_spectrum + 
                                 no2_conc * no2_spectrum + 
                                 so2_conc * so2_spectrum)
                    
                    # 添加标准建模的非线性效应
                    nonlinear_term = nonlinear_strength * (
                        no_conc * no2_conc * np.minimum(no_spectrum, no2_spectrum) +
                        no_conc * so2_conc * np.minimum(no_spectrum, so2_spectrum) +
                        no2_conc * so2_conc * np.minimum(no2_spectrum, so2_spectrum)
                    )
                    
                    mixed_spectrum = mixed_base + nonlinear_term
                    
                    # 标准建模噪声模型
                    # 1. 基础高斯噪声
                    gaussian_noise = np.random.normal(0, base_noise_level, len(mixed_spectrum))
                    
                    # 2. 光谱相关噪声
                    spectral_noise = np.zeros_like(mixed_spectrum)
                    for i in range(1, len(spectral_noise)):
                        spectral_noise[i] = (0.8 * spectral_noise[i-1] + 
                                           np.random.normal(0, spectral_noise_level))
                    
                    # 3. 应用噪声
                    noisy_spectrum = mixed_spectrum * (1 + gaussian_noise + spectral_noise)
                    
                    # 确保物理意义（非负）
                    noisy_spectrum = np.maximum(noisy_spectrum, 0)
                    
                    X_data.append(noisy_spectrum)
                    Y_labels.append([no_conc, no2_conc, so2_conc])
        
        print(f"  Valid combinations: {valid_combinations}")
        print(f"  Invalid combinations: {invalid_combinations}")
        print(f"  Total samples: {len(X_data)}")
        print(f"  Samples per combination: {samples_per_combination}")
        
        # 转换为DataFrame
        X_df = pd.DataFrame(X_data, columns=[f'{w:.1f}cm-1' for w in wavenumbers])
        Y_df = pd.DataFrame(Y_labels, columns=['NO_conc', 'NO2_conc', 'SO2_conc'])
        
        # 保存训练数据
        X_path = os.path.join("data/processed", f"{self.model_name}_X_dataset.csv")
        Y_path = os.path.join("data/processed", f"{self.model_name}_Y_labels.csv")
        
        X_df.to_csv(X_path, index=False)
        Y_df.to_csv(Y_path, index=False)
        
        # 保存数据生成参数
        generation_params = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(X_data),
            'valid_combinations': valid_combinations,
            'samples_per_combination': samples_per_combination,
            'base_noise_level': base_noise_level,
            'spectral_noise_level': spectral_noise_level,
            'nonlinear_strength': nonlinear_strength,
            'concentration_ranges': {
                'NO': [float(no_concentrations.min()), float(no_concentrations.max())],
                'NO2': [float(no2_concentrations.min()), float(no2_concentrations.max())]
            }
        }
        
        params_path = os.path.join(self.results_dir, "generation_parameters.json")
        with open(params_path, 'w') as f:
            json.dump(generation_params, f, indent=2)
        
        self.X_data = X_df
        self.Y_data = Y_df
        self.generation_params = generation_params
        
        print(f"✅ Training data saved:")
        print(f"    X: {X_path}")
        print(f"    Y: {Y_path}")
        print(f"    Parameters: {params_path}")
        
        return X_df, Y_df
    
    def step4_train_standard_model(self):
        """步骤4：训练标准模型"""
        print("\nSTEP 4: Training Standard Model")
        print("="*50)
        
        X = self.X_data.values
        Y = self.Y_data.values
        
        print(f"  Training data shape: X={X.shape}, Y={Y.shape}")
        
        # 数据分割 - 确保浓度组合不重叠
        print("  Splitting data by concentration combinations...")
        
        # 创建组合标识
        Y_temp = self.Y_data.copy()
        Y_temp['combination_id'] = Y_temp.groupby(['NO_conc', 'NO2_conc', 'SO2_conc']).ngroup()
        
        unique_combinations = Y_temp['combination_id'].unique()
        np.random.seed(42)
        shuffled_combinations = np.random.permutation(unique_combinations)
        
        # 分割比例
        n_total = len(unique_combinations)
        n_test = int(n_total * 0.2)
        n_val = int(n_total * 0.1)
        
        test_combinations = shuffled_combinations[:n_test]
        val_combinations = shuffled_combinations[n_test:n_test+n_val]
        train_combinations = shuffled_combinations[n_test+n_val:]
        
        # 创建分割掩码
        train_mask = Y_temp['combination_id'].isin(train_combinations)
        val_mask = Y_temp['combination_id'].isin(val_combinations)
        test_mask = Y_temp['combination_id'].isin(test_combinations)
        
        X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
        Y_train, Y_val, Y_test = Y[train_mask], Y[val_mask], Y[test_mask]
        
        print(f"  Train: {len(train_combinations)} combinations, {X_train.shape[0]} samples")
        print(f"  Val: {len(val_combinations)} combinations, {X_val.shape[0]} samples")
        print(f"  Test: {len(test_combinations)} combinations, {X_test.shape[0]} samples")
        
        # 数据标准化
        print("  Applying data standardization...")
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        Y_train_scaled = scaler_Y.fit_transform(Y_train)
        Y_val_scaled = scaler_Y.transform(Y_val)
        
        # 交叉验证选择最优组件数
        print("  Cross-validation for optimal components...")
        max_components = 15
        cv_scores = []
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for n_comp in range(1, max_components + 1):
            pls = PLSRegression(n_components=n_comp)
            
            # 计算多目标R²分数
            scores_per_target = []
            for target_idx in range(Y_train_scaled.shape[1]):
                scores = cross_val_score(pls, X_train_scaled, Y_train_scaled[:, target_idx], 
                                       cv=kfold, scoring='r2')
                scores_per_target.append(scores.mean())
            
            avg_score = np.mean(scores_per_target)
            cv_scores.append(avg_score)
            
            if n_comp <= 10:  # 只显示前10个
                print(f"    {n_comp} components: CV R² = {avg_score:.5f}")
        
        # 选择最优组件数
        best_n_components = np.argmax(cv_scores) + 1
        best_cv_score = cv_scores[best_n_components - 1]
        
        print(f"  ✅ Optimal components: {best_n_components} (CV R² = {best_cv_score:.5f})")
        
        # 训练最终标准模型
        print("  Training final standard model...")
        standard_model = PLSRegression(n_components=best_n_components)
        standard_model.fit(X_train_scaled, Y_train_scaled)
        
        # 预测和评估
        Y_train_pred_scaled = standard_model.predict(X_train_scaled)
        Y_val_pred_scaled = standard_model.predict(X_val_scaled)
        Y_test_pred_scaled = standard_model.predict(X_test_scaled)
        
        # 逆标准化
        Y_train_pred = scaler_Y.inverse_transform(Y_train_pred_scaled)
        Y_val_pred = scaler_Y.inverse_transform(Y_val_pred_scaled)
        Y_test_pred = scaler_Y.inverse_transform(Y_test_pred_scaled)
        
        # 计算性能指标
        performance = {
            'best_n_components': int(best_n_components),
            'best_cv_score': float(best_cv_score),
            'train_rmse': float(np.sqrt(mean_squared_error(Y_train, Y_train_pred))),
            'train_r2': float(r2_score(Y_train, Y_train_pred)),
            'val_rmse': float(np.sqrt(mean_squared_error(Y_val, Y_val_pred))),
            'val_r2': float(r2_score(Y_val, Y_val_pred)),
            'test_rmse': float(np.sqrt(mean_squared_error(Y_test, Y_test_pred))),
            'test_r2': float(r2_score(Y_test, Y_test_pred))
        }
        
        print(f"  📊 Standard Model Performance:")
        print(f"      Train: RMSE={performance['train_rmse']:.5f}, R²={performance['train_r2']:.5f}")
        print(f"      Val:   RMSE={performance['val_rmse']:.5f}, R²={performance['val_r2']:.5f}")
        print(f"      Test:  RMSE={performance['test_rmse']:.5f}, R²={performance['test_r2']:.5f}")
        
        # 保存模型和相关文件
        self.standard_model = standard_model
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.performance = performance
        self.test_data = (X_test, Y_test, Y_test_pred)
        
        print("✅ Standard model training completed")
        
        return standard_model, performance
    
    def step5_save_and_validate(self):
        """步骤5：保存和验证标准模型"""
        print("\nSTEP 5: Saving and Validating Standard Model")
        print("="*50)
        
        # 保存模型文件
        model_path = os.path.join(self.model_dir, "standard_pls_model.pkl")
        scaler_X_path = os.path.join(self.model_dir, "standard_scaler_X.pkl")
        scaler_Y_path = os.path.join(self.model_dir, "standard_scaler_Y.pkl")
        
        joblib.dump(self.standard_model, model_path)
        joblib.dump(self.scaler_X, scaler_X_path)
        joblib.dump(self.scaler_Y, scaler_Y_path)
        
        print(f"  ✅ Model saved: {model_path}")
        print(f"  ✅ X Scaler saved: {scaler_X_path}")
        print(f"  ✅ Y Scaler saved: {scaler_Y_path}")
        
        # 保存性能结果
        results_path = os.path.join(self.results_dir, "model_performance.json")
        
        full_results = {
            'model_info': {
                'name': self.model_name,
                'type': 'PLSRegression',
                'components': self.performance['best_n_components'],
                'input_features': int(self.standard_model.n_features_in_),
                'output_targets': len(self.gas_names),
                'gas_names': self.gas_names
            },
            'performance': self.performance,
            'training_info': self.generation_params
        }
        
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"  ✅ Results saved: {results_path}")
        
        # 生成预测结果可视化
        self._create_prediction_plots()
        
        # 验证模型
        self._validate_model()
        
        print("  ✅ Standard model validation completed")
        
        return model_path, results_path
    
    def _create_prediction_plots(self):
        """创建预测结果可视化"""
        X_test, Y_test, Y_test_pred = self.test_data
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, gas_name in enumerate(self.gas_names):
            y_true = Y_test[:, i]
            y_pred = Y_test_pred[:, i]
            
            axes[i].scatter(y_true, y_pred, alpha=0.6, s=30)
            
            # 完美预测线
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            # 性能指标
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            axes[i].set_xlabel(f'True {gas_name} Concentration')
            axes[i].set_ylabel(f'Predicted {gas_name} Concentration')
            axes[i].set_title(f'{gas_name}: R² = {r2:.4f}, RMSE = {rmse:.4f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.figures_dir, "standard_model_predictions.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Prediction plots saved: {plot_path}")
    
    def _validate_model(self):
        """验证模型功能"""
        print("  🔍 Validating model functionality...")
        
        # 检查模型是否可以正常加载和预测
        try:
            # 重新加载模型
            model_path = os.path.join(self.model_dir, "standard_pls_model.pkl")
            loaded_model = joblib.load(model_path)
            
            # 创建测试样本
            test_sample = np.random.random((1, self.standard_model.n_features_in_))
            prediction = loaded_model.predict(test_sample)
            
            print(f"    Model loading: ✅")
            print(f"    Prediction shape: {prediction.shape}")
            print(f"    Output range: [{prediction.min():.3f}, {prediction.max():.3f}]")
            
        except Exception as e:
            print(f"    Validation failed: {str(e)}")
    
    def build_complete_standard_model(self):
        """构建完整的标准模型"""
        print("🏗️  BUILDING STANDARD GAS PREDICTION MODEL")
        print("="*60)
        print(f"Model Name: {self.model_name}")
        print(f"Gas Types: {', '.join(self.gas_names)}")
        print(f"Data Source: HITRAN Database")
        print("="*60)
        
        try:
            # 执行所有步骤
            self.step1_load_hitran_data()
            self.step2_preprocess_spectra()
            self.step3_generate_training_data()
            self.step4_train_standard_model()
            model_path, results_path = self.step5_save_and_validate()
            
            print("\n" + "="*60)
            print("🎉 STANDARD MODEL BUILD COMPLETE!")
            print("="*60)
            print(f"📋 Model Summary:")
            print(f"   Name: {self.model_name}")
            print(f"   Components: {self.performance['best_n_components']}")
            print(f"   Test R²: {self.performance['test_r2']:.5f}")
            print(f"   Test RMSE: {self.performance['test_rmse']:.5f}")
            print(f"   Model File: {model_path}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ STANDARD MODEL BUILD FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    builder = StandardGasModelBuilder()
    success = builder.build_complete_standard_model()
    
    if success:
        print("\n✅ Your 'standard' model is ready to use!")
        print("   Use it with: python predict_with_standard.py")
    else:
        print("\n❌ Standard model build failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()