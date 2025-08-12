# performance_benchmark.py
# 全面的性能基准测试脚本

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题


def benchmark_default_pls():
    """基准测试默认PLS模型"""
    print("🔍 默认PLS模型性能测试")
    print("=" * 50)

    # 加载数据
    X = pd.read_csv("data/processed/X_dataset.csv").values
    Y = pd.read_csv("data/processed/Y_labels.csv").values

    print(f"数据集规模: {X.shape[0]} 样本, {X.shape[1]} 特征")

    # 记录训练时间
    start_time = time.time()

    # 拆分数据
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 训练模型
    pls = PLSRegression(n_components=5)
    pls.fit(X_train, Y_train)

    # 预测
    Y_pred = pls.predict(X_test)

    training_time = time.time() - start_time

    # 计算性能指标
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    r2 = r2_score(Y_test, Y_pred)

    # 各气体单独评估
    gas_names = ['NO', 'NO2', 'SO2']
    individual_results = {}

    for i, gas in enumerate(gas_names):
        rmse_gas = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred[:, i]))
        r2_gas = r2_score(Y_test[:, i], Y_pred[:, i])
        individual_results[gas] = {'rmse': rmse_gas, 'r2': r2_gas}

    # 输出结果
    print(f"\n📊 总体性能:")
    print(f"   RMSE: {rmse:.5f}")
    print(f"   R²: {r2:.5f}")
    print(f"   训练时间: {training_time:.3f}秒")

    print(f"\n📊 各气体表现:")
    for gas, metrics in individual_results.items():
        print(f"   {gas}: RMSE={metrics['rmse']:.5f}, R²={metrics['r2']:.5f}")

    return {
        'rmse': rmse,
        'r2': r2,
        'training_time': training_time,
        'individual_results': individual_results,
        'model': pls
    }


def benchmark_adaptive_pls():
    """基准测试自适应PLS模型"""
    print("\n🔍 自适应PLS模型性能测试")
    print("=" * 50)

    # 加载数据
    X = pd.read_csv("data/processed/X_dataset.csv").values
    Y = pd.read_csv("data/processed/Y_labels.csv").values

    start_time = time.time()

    # 拆分数据
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # 交叉验证选择最优主成分数
    max_components = 10
    best_score = -np.inf
    best_n = 1
    cv_scores = {}

    print("🔍 正在优化主成分数...")
    for n in range(1, max_components + 1):
        pls = PLSRegression(n_components=n)
        scores = cross_val_score(pls, X_train, Y_train[:, 0], cv=5, scoring="r2")
        mean_score = scores.mean()
        cv_scores[n] = mean_score

        print(f"   n={n}: CV-R² = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_n = n

    print(f"✅ 最优主成分数: {best_n}, CV-R² = {best_score:.4f}")

    # 使用最优参数训练最终模型
    pls_final = PLSRegression(n_components=best_n)
    pls_final.fit(X_train, Y_train)
    Y_pred = pls_final.predict(X_test)

    training_time = time.time() - start_time

    # 计算性能指标
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    r2 = r2_score(Y_test, Y_pred)

    # 各气体单独评估
    gas_names = ['NO', 'NO2', 'SO2']
    individual_results = {}

    for i, gas in enumerate(gas_names):
        rmse_gas = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred[:, i]))
        r2_gas = r2_score(Y_test[:, i], Y_pred[:, i])
        individual_results[gas] = {'rmse': rmse_gas, 'r2': r2_gas}

    # 输出结果
    print(f"\n📊 总体性能:")
    print(f"   RMSE: {rmse:.5f}")
    print(f"   R²: {r2:.5f}")
    print(f"   训练时间: {training_time:.3f}秒")
    print(f"   最优主成分数: {best_n}")

    print(f"\n📊 各气体表现:")
    for gas, metrics in individual_results.items():
        print(f"   {gas}: RMSE={metrics['rmse']:.5f}, R²={metrics['r2']:.5f}")

    return {
        'rmse': rmse,
        'r2': r2,
        'training_time': training_time,
        'best_n_components': best_n,
        'cv_scores': cv_scores,
        'individual_results': individual_results,
        'model': pls_final
    }


def compare_models(default_results, adaptive_results):
    """对比两个模型的性能"""
    print("\n📊 模型性能对比")
    print("=" * 50)

    # 创建对比表格
    comparison_data = {
        '指标': ['RMSE', 'R²', '训练时间(秒)'],
        '默认PLS': [
            f"{default_results['rmse']:.5f}",
            f"{default_results['r2']:.5f}",
            f"{default_results['training_time']:.3f}"
        ],
        '自适应PLS': [
            f"{adaptive_results['rmse']:.5f}",
            f"{adaptive_results['r2']:.5f}",
            f"{adaptive_results['training_time']:.3f}"
        ]
    }

    # 计算改进程度
    rmse_improvement = (default_results['rmse'] - adaptive_results['rmse']) / default_results['rmse'] * 100
    r2_improvement = (adaptive_results['r2'] - default_results['r2']) / default_results['r2'] * 100
    time_ratio = adaptive_results['training_time'] / default_results['training_time']

    comparison_data['改进程度'] = [
        f"{rmse_improvement:+.1f}%",
        f"{r2_improvement:+.1f}%",
        f"{time_ratio:.1f}倍"
    ]

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # 保存结果
    comparison_df.to_csv("data/results/model_comparison.csv", index=False)

    return comparison_df


def detailed_analysis():
    """详细分析和可视化"""
    print("\n📊 开始详细性能分析...")

    # 运行基准测试
    default_results = benchmark_default_pls()
    adaptive_results = benchmark_adaptive_pls()

    # 对比分析
    comparison = compare_models(default_results, adaptive_results)

    # 创建可视化
    create_performance_plots(default_results, adaptive_results)

    # 生成报告
    generate_report(default_results, adaptive_results, comparison)

    print("\n✅ 性能分析完成！")
    print("📁 结果文件保存在 data/results/ 目录")


def create_performance_plots(default_results, adaptive_results):
    """创建性能对比图表"""
    plt.figure(figsize=(15, 10))

    # 子图1：整体性能对比
    plt.subplot(2, 3, 1)
    models = ['Default PLS', 'Adaptive PLS']
    rmse_values = [default_results['rmse'], adaptive_results['rmse']]
    r2_values = [default_results['r2'], adaptive_results['r2']]

    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width / 2, rmse_values, width, label='RMSE', alpha=0.7)
    plt.bar(x + width / 2, r2_values, width, label='R²', alpha=0.7)
    plt.xlabel('Model Type')
    plt.ylabel('Performance Metric')
    plt.title('Overall Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：各气体RMSE对比
    plt.subplot(2, 3, 2)
    gas_names = ['NO', 'NO2', 'SO2']
    default_rmse = [default_results['individual_results'][gas]['rmse'] for gas in gas_names]
    adaptive_rmse = [adaptive_results['individual_results'][gas]['rmse'] for gas in gas_names]

    x = np.arange(len(gas_names))
    plt.bar(x - width / 2, default_rmse, width, label='Default PLS', alpha=0.7)
    plt.bar(x + width / 2, adaptive_rmse, width, label='Adaptive PLS', alpha=0.7)
    plt.xlabel('Gas Type')
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison by Gas')
    plt.xticks(x, gas_names)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图3：各气体R²对比
    plt.subplot(2, 3, 3)
    default_r2 = [default_results['individual_results'][gas]['r2'] for gas in gas_names]
    adaptive_r2 = [adaptive_results['individual_results'][gas]['r2'] for gas in gas_names]

    plt.bar(x - width / 2, default_r2, width, label='Default PLS', alpha=0.7)
    plt.bar(x + width / 2, adaptive_r2, width, label='Adaptive PLS', alpha=0.7)
    plt.xlabel('Gas Type')
    plt.ylabel('R²')
    plt.title('R² Comparison by Gas')
    plt.xticks(x, gas_names)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图4：训练时间对比
    plt.subplot(2, 3, 4)
    times = [default_results['training_time'], adaptive_results['training_time']]
    colors = ['skyblue', 'lightcoral']
    plt.bar(models, times, color=colors, alpha=0.7)
    plt.xlabel('Model Type')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.grid(True, alpha=0.3)

    # 添加数值标签
    for i, time in enumerate(times):
        plt.text(i, time + max(times) * 0.02, f'{time:.3f}s',
                 ha='center', va='bottom', fontweight='bold')

    # 子图5：交叉验证曲线（仅自适应模型）
    plt.subplot(2, 3, 5)
    if 'cv_scores' in adaptive_results:
        components = list(adaptive_results['cv_scores'].keys())
        scores = list(adaptive_results['cv_scores'].values())

        plt.plot(components, scores, 'o-', linewidth=2, markersize=8)
        plt.axvline(x=adaptive_results['best_n_components'],
                    color='red', linestyle='--', alpha=0.7,
                    label=f'Optimal n={adaptive_results["best_n_components"]}')
        plt.xlabel('Number of Components')
        plt.ylabel('Cross-validation R²')
        plt.title('Component Number Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 子图6：性能雷达图
    plt.subplot(2, 3, 6, projection='polar')

    # 准备雷达图数据
    categories = ['RMSE\n(lower better)', 'R²\n(higher better)', 'Speed\n(faster better)']

    # 归一化数据 (0-1之间)
    default_rmse_norm = 1 - default_results['rmse'] / max(default_results['rmse'], adaptive_results['rmse'])
    adaptive_rmse_norm = 1 - adaptive_results['rmse'] / max(default_results['rmse'], adaptive_results['rmse'])

    default_r2_norm = default_results['r2']
    adaptive_r2_norm = adaptive_results['r2']

    default_speed_norm = 1 / (1 + default_results['training_time'])
    adaptive_speed_norm = 1 / (1 + adaptive_results['training_time'])

    default_values = [default_rmse_norm, default_r2_norm, default_speed_norm]
    adaptive_values = [adaptive_rmse_norm, adaptive_r2_norm, adaptive_speed_norm]

    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合

    default_values += [default_values[0]]
    adaptive_values += [adaptive_values[0]]

    plt.plot(angles, default_values, 'o-', linewidth=2, label='Default PLS')
    plt.plot(angles, adaptive_values, 'o-', linewidth=2, label='Adaptive PLS')
    plt.fill(angles, default_values, alpha=0.25)
    plt.fill(angles, adaptive_values, alpha=0.25)

    plt.xticks(angles[:-1], categories)
    plt.ylim(0, 1)
    plt.title('Comprehensive Performance Radar')
    plt.legend()

    plt.tight_layout()
    plt.savefig('data/figures/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Performance comparison chart saved to: data/figures/performance_comparison.png")
    plt.show()


def generate_report(default_results, adaptive_results, comparison):
    """生成详细的性能报告"""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"data/results/performance_report_{timestamp}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Gas Spectrum Analysis System Performance Test Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("1. Overall Performance Comparison\n")
        f.write("-" * 30 + "\n")
        f.write(f"Default PLS Model:\n")
        f.write(f"  RMSE: {default_results['rmse']:.5f}\n")
        f.write(f"  R²: {default_results['r2']:.5f}\n")
        f.write(f"  Training Time: {default_results['training_time']:.3f}s\n\n")

        f.write(f"Adaptive PLS Model:\n")
        f.write(f"  RMSE: {adaptive_results['rmse']:.5f}\n")
        f.write(f"  R²: {adaptive_results['r2']:.5f}\n")
        f.write(f"  Training Time: {adaptive_results['training_time']:.3f}s\n")
        f.write(f"  Optimal Components: {adaptive_results['best_n_components']}\n\n")

        f.write("2. Individual Gas Performance\n")
        f.write("-" * 30 + "\n")
        for gas in ['NO', 'NO2', 'SO2']:
            f.write(f"{gas}:\n")
            f.write(f"  Default PLS: RMSE={default_results['individual_results'][gas]['rmse']:.5f}, "
                    f"R²={default_results['individual_results'][gas]['r2']:.5f}\n")
            f.write(f"  Adaptive PLS: RMSE={adaptive_results['individual_results'][gas]['rmse']:.5f}, "
                    f"R²={adaptive_results['individual_results'][gas]['r2']:.5f}\n\n")

        f.write("3. Performance Improvement Analysis\n")
        f.write("-" * 30 + "\n")
        rmse_improvement = (default_results['rmse'] - adaptive_results['rmse']) / default_results['rmse'] * 100
        r2_improvement = (adaptive_results['r2'] - default_results['r2']) / default_results['r2'] * 100
        f.write(f"RMSE Improvement: {rmse_improvement:.1f}%\n")
        f.write(f"R² Improvement: {r2_improvement:.1f}%\n")
        f.write(
            f"Training Time Increase: {adaptive_results['training_time'] / default_results['training_time']:.1f}x\n\n")

        f.write("4. Conclusions and Recommendations\n")
        f.write("-" * 30 + "\n")
        if rmse_improvement > 10:
            f.write("✅ Adaptive PLS significantly outperforms default model, recommended for deployment\n")
        elif rmse_improvement > 5:
            f.write("✅ Adaptive PLS slightly better than default model, choose based on requirements\n")
        else:
            f.write("⚠️ Similar performance, recommend faster default model\n")

    print(f"📋 Detailed report saved to: {report_file}")


if __name__ == "__main__":
    # 确保目录存在
    import os

    os.makedirs("data/results", exist_ok=True)
    os.makedirs("data/figures", exist_ok=True)

    # 运行完整分析
    detailed_analysis()