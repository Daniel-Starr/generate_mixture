# run_enhanced_pipeline.py
# 运行增强版气体光谱分析流水线

import os
import sys
import argparse
import time
from datetime import datetime

def run_complete_pipeline(regenerate_data=True, retrain_model=True):
    """
    运行完整的增强流水线
    
    Parameters:
    - regenerate_data: 是否重新生成仿真数据
    - retrain_model: 是否重新训练模型
    """
    
    print("=" * 60)
    print("🚀 增强版气体光谱分析系统")
    print("   基于PLS回归的三气体(NO, NO2, SO2)浓度预测")
    print("=" * 60)
    
    start_time = time.time()
    
    # 确保必要的目录存在
    directories = [
        "data/processed", "data/models", "data/figures", 
        "data/raw", "data/results"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        
    print("📁 目录结构已准备就绪")
    
    try:
        # 步骤1: 数据预处理（如果需要）
        if not os.path.exists("data/processed/interpolated_spectra.csv"):
            print("\n" + "="*50)
            print("📊 步骤1: 光谱数据预处理")
            print("="*50)
            
            import subprocess
            result = subprocess.run([sys.executable, "01_preprocess.py（三气体版）.py"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ 预处理失败: {result.stderr}")
                return False
            print("✅ 光谱数据预处理完成")
        
        # 步骤2: 生成增强数据集
        if regenerate_data or not os.path.exists("data/processed/X_dataset.csv"):
            print("\n" + "="*50)
            print("🧬 步骤2: 生成增强混合数据集")
            print("="*50)
            
            import subprocess
            result = subprocess.run([sys.executable, "03_generate_dataset.py（三气体版）.py"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ 数据生成失败: {result.stderr}")
                return False
            else:
                print(result.stdout)
            print("✅ 增强混合数据集生成完成")
        
        # 步骤3: 智能数据分割
        print("\n" + "="*50)
        print("🎯 步骤3: 智能数据分割")
        print("="*50)
        
        from data_splitter import apply_intelligent_split
        split_result = apply_intelligent_split()
        if split_result is None:
            print("❌ 数据分割失败")
            return False
        print("✅ 智能数据分割完成")
        
        # 步骤4: 增强模型训练
        if retrain_model or not os.path.exists("data/models/enhanced_pls_model.pkl"):
            print("\n" + "="*50)
            print("🧠 步骤4: 增强模型训练")
            print("="*50)
            
            from enhanced_model_trainer import main as train_main
            trainer, results = train_main()
            
            if trainer is None:
                print("❌ 模型训练失败")
                return False
            print("✅ 增强模型训练完成")
        
        # 步骤5: 模型性能分析
        print("\n" + "="*50)
        print("📈 步骤5: 模型性能分析")
        print("="*50)
        
        analyze_enhanced_performance()
        print("✅ 性能分析完成")
        
        # 步骤6: 生成综合报告
        print("\n" + "="*50)
        print("📋 步骤6: 生成综合报告")
        print("="*50)
        
        generate_comprehensive_report()
        print("✅ 综合报告生成完成")
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("🎉 增强流水线执行完成!")
        print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
        print("="*60)
        
        # 显示关键结果
        display_key_results()
        
        return True
    
    except Exception as e:
        print(f"❌ 流水线执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def analyze_enhanced_performance():
    """分析增强模型的性能"""
    
    import pandas as pd
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    # 检查必要文件
    required_files = [
        "data/processed/X_test.csv",
        "data/processed/Y_test.csv", 
        "data/models/enhanced_training_results.json"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"   ⚠️ 缺少文件: {file_path}")
            return
    
    # 加载数据和结果
    X_test = pd.read_csv("data/processed/X_test.csv")
    Y_test = pd.read_csv("data/processed/Y_test.csv")
    
    with open("data/models/enhanced_training_results.json", 'r') as f:
        results = json.load(f)
    
    # 加载模型进行测试集预测
    import joblib
    model = joblib.load("data/models/enhanced_pls_model.pkl")
    
    if results['use_scaling']:
        scaler_X = joblib.load("data/models/scaler_X.pkl")
        scaler_Y = joblib.load("data/models/scaler_Y.pkl")
        
        X_test_scaled = scaler_X.transform(X_test)
        Y_test_pred_scaled = model.predict(X_test_scaled)
        Y_test_pred = scaler_Y.inverse_transform(Y_test_pred_scaled)
    else:
        Y_test_pred = model.predict(X_test.values)
    
    # 计算测试集性能
    test_metrics = {
        'test_rmse': float(np.sqrt(mean_squared_error(Y_test.values, Y_test_pred))),
        'test_mae': float(mean_absolute_error(Y_test.values, Y_test_pred)),
        'test_r2': float(r2_score(Y_test.values, Y_test_pred))
    }
    
    # 每个气体的性能
    gas_names = ['NO', 'NO2', 'SO2']
    for i, gas in enumerate(gas_names):
        test_metrics[f'test_{gas}_rmse'] = float(np.sqrt(mean_squared_error(Y_test.values[:, i], Y_test_pred[:, i])))
        test_metrics[f'test_{gas}_r2'] = float(r2_score(Y_test.values[:, i], Y_test_pred[:, i]))
    
    print("   📊 测试集性能:")
    print(f"      整体 - RMSE: {test_metrics['test_rmse']:.5f}, R²: {test_metrics['test_r2']:.5f}")
    for gas in gas_names:
        print(f"      {gas} - RMSE: {test_metrics[f'test_{gas}_rmse']:.5f}, R²: {test_metrics[f'test_{gas}_r2']:.5f}")
    
    # 与原始简单模型比较
    comparison = {
        'enhanced_model': test_metrics,
        'improvement_over_original': {
            'rmse_improvement': 'N/A',  # 需要原始结果进行比较
            'r2_improvement': 'N/A'
        }
    }
    
    # 保存性能分析结果
    with open("data/results/enhanced_performance_analysis.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"   💾 性能分析已保存到: data/results/enhanced_performance_analysis.json")

def generate_comprehensive_report():
    """生成综合分析报告"""
    
    import json
    import pandas as pd
    from datetime import datetime
    
    report_content = []
    report_content.append("# 增强版气体光谱分析系统 - 综合报告")
    report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("")
    
    # 1. 系统概述
    report_content.append("## 1. 系统概述")
    report_content.append("本系统是一个基于PLS回归的三气体(NO, NO2, SO2)浓度预测系统，具备以下改进:")
    report_content.append("- ✅ 解决了数据泄露问题")
    report_content.append("- ✅ 增加了数据复杂性和真实性")
    report_content.append("- ✅ 实现了严格的交叉验证")
    report_content.append("- ✅ 添加了多层次噪声模型")
    report_content.append("- ✅ 包含了非线性混合效应")
    report_content.append("")
    
    # 2. 数据生成参数
    if os.path.exists("data/processed/generation_params.json"):
        with open("data/processed/generation_params.json", 'r') as f:
            gen_params = json.load(f)
        
        report_content.append("## 2. 数据生成参数")
        report_content.append(f"- 总样本数: {gen_params.get('total_samples', 'N/A')}")
        report_content.append(f"- 唯一浓度组合: {gen_params.get('unique_combinations', 'N/A')}")
        report_content.append(f"- 基础噪声水平: {gen_params.get('base_noise_level', 'N/A')}")
        report_content.append(f"- 非线性强度: {gen_params.get('nonlinear_strength', 'N/A')}")
        report_content.append("")
    
    # 3. 数据分割信息
    if os.path.exists("data/processed/split_info.json"):
        with open("data/processed/split_info.json", 'r') as f:
            split_info = json.load(f)
        
        report_content.append("## 3. 数据分割策略")
        report_content.append(f"- 策略: {split_info.get('strategy', 'N/A')}")
        report_content.append(f"- 训练集: {split_info.get('train_samples', 'N/A')} 样本, {split_info.get('train_combinations', 'N/A')} 组合")
        report_content.append(f"- 验证集: {split_info.get('val_samples', 'N/A')} 样本, {split_info.get('val_combinations', 'N/A')} 组合")
        report_content.append(f"- 测试集: {split_info.get('test_samples', 'N/A')} 样本, {split_info.get('test_combinations', 'N/A')} 组合")
        report_content.append("- ✅ 确保浓度组合无重叠，避免数据泄露")
        report_content.append("")
    
    # 4. 模型性能
    if os.path.exists("data/models/enhanced_training_results.json"):
        with open("data/models/enhanced_training_results.json", 'r') as f:
            model_results = json.load(f)
        
        report_content.append("## 4. 模型性能")
        report_content.append(f"- 最优主成分数: {model_results.get('best_n_components', 'N/A')}")
        
        cv_results = model_results.get('cv_results', {})
        report_content.append(f"- 交叉验证R²: {cv_results.get('best_cv_score', 'N/A'):.5f} ± {cv_results.get('best_cv_std', 'N/A'):.5f}")
        
        val_results = model_results.get('validation_results', {})
        report_content.append(f"- 验证集RMSE: {val_results.get('validation_rmse', 'N/A'):.5f}")
        report_content.append(f"- 验证集R²: {val_results.get('validation_r2', 'N/A'):.5f}")
        report_content.append("")
    
    # 5. 改进效果
    report_content.append("## 5. 关键改进")
    report_content.append("### 5.1 数据泄露问题解决")
    report_content.append("- 原问题: 训练测试集有27/30个相同浓度组合")
    report_content.append("- 解决方案: 基于浓度组合的智能分割，确保零重叠")
    report_content.append("")
    
    report_content.append("### 5.2 数据复杂性增强")
    report_content.append("- 原问题: 仅30个组合，1%简单噪声")
    report_content.append("- 解决方案: 大幅增加组合数，多层次噪声模型")
    report_content.append("")
    
    report_content.append("### 5.3 模型验证严格化")
    report_content.append("- 原问题: 简单的random_state分割")
    report_content.append("- 解决方案: 严格交叉验证+独立验证集")
    report_content.append("")
    
    # 6. 使用指南
    report_content.append("## 6. 使用指南")
    report_content.append("### 6.1 处理真实数据")
    report_content.append("```python")
    report_content.append("from real_data_processor import process_real_data_pipeline")
    report_content.append("result = process_real_data_pipeline('your_data.csv')")
    report_content.append("```")
    report_content.append("")
    
    report_content.append("### 6.2 重新训练模型")
    report_content.append("```python")
    report_content.append("python run_enhanced_pipeline.py --retrain")
    report_content.append("```")
    report_content.append("")
    
    # 保存报告
    report_path = "data/results/comprehensive_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print(f"   📋 综合报告已保存到: {report_path}")

def display_key_results():
    """显示关键结果摘要"""
    
    print("\n" + "🔍 关键结果摘要:")
    
    # 数据统计
    if os.path.exists("data/processed/generation_params.json"):
        import json
        with open("data/processed/generation_params.json", 'r') as f:
            params = json.load(f)
        print(f"   📊 数据集: {params.get('total_samples', 'N/A')} 样本, {params.get('unique_combinations', 'N/A')} 唯一组合")
    
    # 模型性能
    if os.path.exists("data/results/enhanced_performance_analysis.json"):
        import json
        with open("data/results/enhanced_performance_analysis.json", 'r') as f:
            perf = json.load(f)
        
        enhanced = perf.get('enhanced_model', {})
        print(f"   🎯 测试集性能: RMSE = {enhanced.get('test_rmse', 'N/A'):.5f}, R² = {enhanced.get('test_r2', 'N/A'):.5f}")
    
    # 文件位置
    print("\n   📁 主要输出文件:")
    important_files = [
        ("模型文件", "data/models/enhanced_pls_model.pkl"),
        ("性能分析", "data/results/enhanced_performance_analysis.json"),
        ("综合报告", "data/results/comprehensive_report.md"),
        ("数据分割可视化", "data/figures/data_split_visualization.png"),
        ("模型性能可视化", "data/figures/enhanced_prediction_results.png")
    ]
    
    for name, path in important_files:
        if os.path.exists(path):
            print(f"      ✅ {name}: {path}")
        else:
            print(f"      ❌ {name}: {path} (缺失)")

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='运行增强版气体光谱分析流水线')
    parser.add_argument('--regenerate-data', action='store_true', 
                       help='重新生成仿真数据')
    parser.add_argument('--retrain', action='store_true',
                       help='重新训练模型')
    
    args = parser.parse_args()
    
    success = run_complete_pipeline(
        regenerate_data=args.regenerate_data,
        retrain_model=args.retrain
    )
    
    if success:
        print("\n💡 下一步建议:")
        print("   1. 查看 data/results/comprehensive_report.md 了解详细结果")
        print("   2. 使用 real_data_processor.process_real_data_pipeline() 处理您的真实数据")
        print("   3. 根据需要调整模型参数或数据生成参数")
        
        return 0
    else:
        print("\n❌ 流水线执行失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())