# vip_analyzer.py
# VIP (Variable Importance in Projection) 分析工具
# 专门用于分析PLS模型中最重要的光谱波段

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Dict, List, Tuple

# 导入字体配置
try:
    from font_config import setup_chinese_fonts, get_title_text
    setup_chinese_fonts()
except ImportError:
    # 如果没有font_config，使用简单的字体设置
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    def get_title_text(chinese_text, english_text=None):
        return english_text if english_text else chinese_text

class VIPAnalyzer:
    """
    VIP分析器，专门用于分析PLS模型的变量重要性
    
    功能：
    1. 计算VIP分数
    2. 识别重要波段
    3. 可视化分析结果
    4. 生成分析报告
    """
    
    def __init__(self, model_path: str = "data/models/enhanced_pls_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.vip_scores = None
        self.wavenumbers = None
        
    def load_model(self):
        """加载PLS模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件未找到: {self.model_path}")
            
        self.model = joblib.load(self.model_path)
        print(f"✅ 已加载PLS模型: {self.model.n_components} 个主成分")
        
    def load_wavenumbers(self, spectra_path: str = "data/processed/interpolated_spectra.csv"):
        """加载波数信息"""
        try:
            df = pd.read_csv(spectra_path)
            self.wavenumbers = df['wavenumber'].values
            print(f"✅ 已加载波数信息: {len(self.wavenumbers)} 个波数点")
        except:
            print("⚠️  无法加载波数信息，将使用索引代替")
            if self.model:
                self.wavenumbers = np.arange(self.model.n_features_in_)
            else:
                self.wavenumbers = None
    
    def calculate_vip_scores(self) -> np.ndarray:
        """
        计算VIP (Variable Importance in Projection) 分数
        
        VIP分数公式: VIP_j = sqrt(p * sum(a_j,h^2 * SS_h) / sum(SS_h))
        
        Returns:
            np.ndarray: 每个变量的VIP分数
        """
        if self.model is None:
            raise ValueError("请先加载模型")
            
        # 获取模型参数
        x_loadings = self.model.x_loadings_  # shape: (n_features, n_components)
        y_scores = self.model.y_scores_      # shape: (n_samples, n_components)
        
        # 计算每个主成分解释的Y方差
        ss_components = np.var(y_scores, axis=0, ddof=1)
        
        # 计算VIP分数
        n_features = x_loadings.shape[0]
        n_components = x_loadings.shape[1]
        
        vip_scores = np.zeros(n_features)
        total_ss = np.sum(ss_components)
        
        if total_ss > 0:
            for j in range(n_features):
                weighted_loadings_sq = np.sum((x_loadings[j, :] ** 2) * ss_components)
                vip_scores[j] = np.sqrt(n_features * weighted_loadings_sq / total_ss)
        
        self.vip_scores = vip_scores
        return vip_scores
    
    def get_important_variables(self, threshold: float = 1.0) -> Dict:
        """
        获取重要变量信息
        
        Args:
            threshold: VIP分数阈值，默认1.0
            
        Returns:
            Dict: 重要变量的统计信息
        """
        if self.vip_scores is None:
            self.calculate_vip_scores()
            
        important_mask = self.vip_scores > threshold
        important_indices = np.where(important_mask)[0]
        
        result = {
            'threshold': threshold,
            'total_variables': len(self.vip_scores),
            'important_count': len(important_indices),
            'important_percentage': len(important_indices) / len(self.vip_scores) * 100,
            'important_indices': important_indices.tolist(),
            'vip_mean': float(self.vip_scores.mean()),
            'vip_std': float(self.vip_scores.std()),
            'vip_max': float(self.vip_scores.max()),
            'vip_min': float(self.vip_scores.min())
        }
        
        if self.wavenumbers is not None:
            result['important_wavenumbers'] = self.wavenumbers[important_indices].tolist()
            result['top_10_wavenumbers'] = self.get_top_wavenumbers(10)
            
        return result
    
    def get_top_wavenumbers(self, n: int = 10) -> List[Dict]:
        """获取VIP分数最高的n个波数"""
        if self.vip_scores is None or self.wavenumbers is None:
            return []
            
        top_indices = np.argsort(self.vip_scores)[-n:][::-1]
        
        return [
            {
                'rank': i + 1,
                'wavenumber': float(self.wavenumbers[idx]),
                'vip_score': float(self.vip_scores[idx]),
                'index': int(idx)
            }
            for i, idx in enumerate(top_indices)
        ]
    
    def visualize_vip_analysis(self, save_dir: str = "data/figures/"):
        """生成VIP分析的完整可视化"""
        if self.vip_scores is None:
            self.calculate_vip_scores()
            
        os.makedirs(save_dir, exist_ok=True)
        
        # 使用波数或索引
        x_axis = self.wavenumbers if self.wavenumbers is not None else np.arange(len(self.vip_scores))
        # 使用数学模式显示上标，避免字体问题
        x_label = 'Wavenumber (cm$^{-1}$)' if self.wavenumbers is not None else 'Variable Index'
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. VIP分数全谱图
        axes[0, 0].plot(x_axis, self.vip_scores, 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='VIP = 1.0 (重要性阈值)')
        
        # 填充重要区域
        important_mask = self.vip_scores > 1.0
        if np.any(important_mask):
            axes[0, 0].fill_between(x_axis, self.vip_scores, 1.0, 
                                   where=important_mask, color='red', alpha=0.3,
                                   label=f'重要波段 (n={np.sum(important_mask)})')
        
        axes[0, 0].set_xlabel(x_label)
        axes[0, 0].set_ylabel('VIP Score')
        axes[0, 0].set_title(get_title_text('Variable Importance in Projection (VIP) 全谱分析', 
                                           'Variable Importance in Projection (VIP) Analysis'))
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 重要波段详细图
        important_indices = np.where(self.vip_scores > 1.0)[0]
        if len(important_indices) > 0:
            axes[0, 1].stem(x_axis[important_indices], self.vip_scores[important_indices], 
                           basefmt=' ', linefmt='g-', markerfmt='go')
            axes[0, 1].set_xlabel(x_label)
            axes[0, 1].set_ylabel('VIP Score')
            axes[0, 1].set_title(get_title_text(f'重要波段分析 (VIP > 1.0, 共{len(important_indices)}个)',
                                               f'Important Variables (VIP > 1.0, n={len(important_indices)})'))
            axes[0, 1].grid(True, alpha=0.3)
            
            # 标注最重要的波段
            if len(important_indices) <= 20:  # 如果重要波段不多，全部标注
                for idx in important_indices:
                    axes[0, 1].annotate(f'{x_axis[idx]:.0f}', 
                                       (x_axis[idx], self.vip_scores[idx]),
                                       xytext=(0, 10), textcoords='offset points',
                                       fontsize=8, ha='center')
        else:
            axes[0, 1].text(0.5, 0.5, get_title_text('没有VIP>1.0的变量\n(可能需要调整阈值)', 
                                                     'No variables with VIP>1.0\n(consider adjusting threshold)'), 
                           transform=axes[0, 1].transAxes, ha='center', va='center')
            axes[0, 1].set_title(get_title_text('重要波段分析', 'Important Variables Analysis'))
        
        # 3. VIP分数分布直方图
        axes[1, 0].hist(self.vip_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='VIP = 1.0')
        axes[1, 0].axvline(x=self.vip_scores.mean(), color='green', linestyle='-', linewidth=2, 
                          label=f'Mean = {self.vip_scores.mean():.3f}')
        axes[1, 0].set_xlabel('VIP Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(get_title_text('VIP分数分布直方图', 'VIP Score Distribution'))
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 累积重要性分析
        sorted_indices = np.argsort(self.vip_scores)[::-1]
        cumulative_importance = np.cumsum(self.vip_scores[sorted_indices])
        cumulative_percentage = cumulative_importance / cumulative_importance[-1] * 100
        
        axes[1, 1].plot(range(1, len(cumulative_percentage) + 1), cumulative_percentage, 'b-')
        axes[1, 1].axhline(y=80, color='red', linestyle='--', label='80% Importance')
        axes[1, 1].axhline(y=95, color='orange', linestyle='--', label='95% Importance')
        
        # 找到达到特定重要性的变量数
        for percentage, color in [(80, 'red'), (95, 'orange')]:
            idx = np.where(cumulative_percentage >= percentage)[0]
            if len(idx) > 0:
                axes[1, 1].axvline(x=idx[0]+1, color=color, linestyle=':', alpha=0.7)
                axes[1, 1].text(idx[0]+1, percentage+2, f'{idx[0]+1} vars', 
                               rotation=90, va='bottom', fontsize=9)
        
        axes[1, 1].set_xlabel('Variable Count (sorted by importance)')
        axes[1, 1].set_ylabel('Cumulative Importance (%)')
        axes[1, 1].set_title(get_title_text('累积变量重要性分析', 'Cumulative Variable Importance'))
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/vip_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 VIP可视化分析已保存到: {save_dir}/vip_comprehensive_analysis.png")
    
    def generate_report(self, save_path: str = "data/results/vip_analysis_report.txt"):
        """生成VIP分析报告"""
        if self.vip_scores is None:
            self.calculate_vip_scores()
            
        important_info = self.get_important_variables()
        top_wavenumbers = self.get_top_wavenumbers(20)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("VIP (Variable Importance in Projection) 分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("📊 整体统计信息:\n")
            f.write(f"  总变量数: {important_info['total_variables']}\n")
            f.write(f"  重要变量数 (VIP > 1.0): {important_info['important_count']}\n")
            f.write(f"  重要变量比例: {important_info['important_percentage']:.2f}%\n")
            f.write(f"  VIP分数统计: {important_info['vip_mean']:.3f} ± {important_info['vip_std']:.3f}\n")
            f.write(f"  VIP分数范围: [{important_info['vip_min']:.3f}, {important_info['vip_max']:.3f}]\n\n")
            
            if top_wavenumbers:
                f.write("🎯 Top 20 最重要波段:\n")
                f.write("排名  波数(cm⁻¹)    VIP分数    变量索引\n")
                f.write("-" * 40 + "\n")
                for item in top_wavenumbers:
                    f.write(f"{item['rank']:2d}    {item['wavenumber']:8.1f}    {item['vip_score']:7.4f}    {item['index']:6d}\n")
                f.write("\n")
            
            if 'important_wavenumbers' in important_info:
                f.write("📍 所有重要波段 (VIP > 1.0):\n")
                important_waves = important_info['important_wavenumbers']
                if important_waves:
                    # 按波数排序
                    important_waves_sorted = sorted(important_waves)
                    for i, wave in enumerate(important_waves_sorted):
                        if i % 10 == 0:
                            f.write("\n")
                        f.write(f"{wave:8.1f}")
                    f.write(f"\n\n总计: {len(important_waves)} 个重要波段\n\n")
                else:
                    f.write("  没有VIP > 1.0的波段\n\n")
            
            f.write("🔬 光谱解释建议:\n")
            f.write("  1. VIP > 1.0 的波段被认为对预测具有重要贡献\n")
            f.write("  2. 这些波段可能对应气体分子的特征吸收峰\n")
            f.write("  3. 可以重点关注VIP分数最高的波段进行光谱解释\n")
            f.write("  4. 建议结合化学知识分析重要波段的物理意义\n\n")
            
            f.write("📈 模型优化建议:\n")
            if important_info['important_count'] > 0:
                f.write(f"  1. 可考虑仅使用{important_info['important_count']}个重要变量构建简化模型\n")
                f.write("  2. 重要波段可用于特征选择和降维\n")
                f.write("  3. 建议验证简化模型的预测性能\n")
            else:
                f.write("  1. 没有明显的重要变量，可能需要:\n")
                f.write("     - 检查模型训练质量\n")
                f.write("     - 调整VIP阈值\n")
                f.write("     - 增加主成分数量\n")
        
        print(f"📄 VIP分析报告已保存到: {save_path}")
    
    def save_results(self, save_dir: str = "data/results/"):
        """保存所有VIP分析结果"""
        if self.vip_scores is None:
            self.calculate_vip_scores()
            
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存详细的VIP数据
        x_axis = self.wavenumbers if self.wavenumbers is not None else np.arange(len(self.vip_scores))
        
        vip_df = pd.DataFrame({
            'variable_index': range(len(self.vip_scores)),
            'wavenumber': x_axis,
            'vip_score': self.vip_scores,
            'is_important': self.vip_scores > 1.0,
            'importance_rank': len(self.vip_scores) - np.argsort(np.argsort(self.vip_scores))
        })
        
        vip_df = vip_df.sort_values('vip_score', ascending=False)
        vip_df.to_csv(f"{save_dir}/vip_detailed_results.csv", index=False)
        
        print(f"💾 VIP详细结果已保存到: {save_dir}/vip_detailed_results.csv")

def main():
    """主函数：执行完整的VIP分析"""
    print("🔬 VIP (Variable Importance in Projection) 分析工具")
    print("=" * 60)
    
    # 创建分析器
    analyzer = VIPAnalyzer()
    
    try:
        # 加载模型和数据
        analyzer.load_model()
        analyzer.load_wavenumbers()
        
        # 计算VIP分数
        print("\n📊 正在计算VIP分数...")
        vip_scores = analyzer.calculate_vip_scores()
        
        # 获取重要变量信息
        important_info = analyzer.get_important_variables()
        print(f"\n✅ VIP分析完成:")
        print(f"   重要变量数 (VIP > 1.0): {important_info['important_count']} / {important_info['total_variables']}")
        print(f"   重要变量比例: {important_info['important_percentage']:.2f}%")
        print(f"   VIP分数统计: {important_info['vip_mean']:.3f} ± {important_info['vip_std']:.3f}")
        
        # 显示最重要的波段
        if 'top_10_wavenumbers' in important_info:
            print(f"\n🎯 Top 10 最重要波段:")
            for item in important_info['top_10_wavenumbers']:
                print(f"   {item['rank']:2d}. {item['wavenumber']:8.1f} cm⁻¹ (VIP = {item['vip_score']:.4f})")
        
        # 生成可视化和报告
        print(f"\n📊 正在生成分析结果...")
        analyzer.visualize_vip_analysis()
        analyzer.generate_report()
        analyzer.save_results()
        
        print(f"\n🎉 VIP分析完成！请查看:")
        print(f"   - 可视化图表: data/figures/vip_comprehensive_analysis.png")
        print(f"   - 分析报告: data/results/vip_analysis_report.txt")
        print(f"   - 详细数据: data/results/vip_detailed_results.csv")
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        print("请确保:")
        print("1. 模型文件存在: data/models/enhanced_pls_model.pkl")
        print("2. 光谱文件存在: data/processed/interpolated_spectra.csv")
        print("3. 已完成模型训练")

if __name__ == "__main__":
    main()