"""
气体浓度预测器 - 用于实际测量
快速预测未知混合气体中NO、NO2、SO2的浓度
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Union, List, Dict
import json
from datetime import datetime


class GasConcentrationPredictor:
    """气体浓度预测器"""

    def __init__(self, model_dir: str = "./output/models"):
        """
        初始化预测器

        Parameters:
        -----------
        model_dir : str
            模型文件所在目录
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler_X = None
        self.scaler_Y = None
        self.config = None
        self.is_loaded = False

        # 加载模型
        self.load_models()

    def load_models(self):
        """加载训练好的模型和配置"""
        try:
            self.model = joblib.load(self.model_dir / 'pls_model.pkl')
            self.scaler_X = joblib.load(self.model_dir / 'scaler_X.pkl')
            self.scaler_Y = joblib.load(self.model_dir / 'scaler_Y.pkl')
            self.config = joblib.load(self.model_dir / 'config.pkl')
            self.is_loaded = True
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print(f"请确保模型文件在 {self.model_dir} 目录中")

    def predict_single(self, spectrum_file: str) -> Dict:
        """
        预测单个光谱文件

        Parameters:
        -----------
        spectrum_file : str
            光谱数据文件路径（CSV格式）

        Returns:
        --------
        Dict : 包含预测结果和置信度的字典
        """
        if not self.is_loaded:
            raise RuntimeError("模型未成功加载")

        # 读取光谱数据
        if isinstance(spectrum_file, str):
            spectrum_df = pd.read_csv(spectrum_file)
        else:
            spectrum_df = spectrum_file

        # 提取光谱值（假设第一列是波数，第二列是强度）
        if spectrum_df.shape[1] == 2:
            spectrum = spectrum_df.iloc[:, 1].values
        else:
            spectrum = spectrum_df.values.flatten()

        # 确保维度正确
        spectrum = spectrum.reshape(1, -1)

        # 标准化
        spectrum_scaled = self.scaler_X.transform(spectrum)

        # 预测
        prediction_scaled = self.model.predict(spectrum_scaled)
        prediction = self.scaler_Y.inverse_transform(prediction_scaled)

        # 计算预测置信度（基于训练时的R²）
        # 这里简化处理，实际应用中可以使用更复杂的不确定性估计
        confidence = self._estimate_confidence(spectrum_scaled)

        # 整理结果
        result = {
            'timestamp': datetime.now().isoformat(),
            'file': spectrum_file if isinstance(spectrum_file, str) else 'direct_input',
            'concentrations': {
                'NO': float(prediction[0, 0]),
                'NO2': float(prediction[0, 1]),
                'SO2': float(prediction[0, 2])
            },
            'confidence': confidence,
            'total': float(np.sum(prediction[0])),
            'status': 'success'
        }

        # 检查结果合理性
        result = self._validate_results(result)

        return result

    def predict_batch(self, spectrum_files: List[str]) -> pd.DataFrame:
        """
        批量预测多个光谱文件

        Parameters:
        -----------
        spectrum_files : List[str]
            光谱文件路径列表

        Returns:
        --------
        pd.DataFrame : 包含所有预测结果的DataFrame
        """
        results = []

        for i, file in enumerate(spectrum_files):
            print(f"处理文件 {i + 1}/{len(spectrum_files)}: {file}")
            try:
                result = self.predict_single(file)
                result['index'] = i
                results.append(result)
            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
                results.append({
                    'index': i,
                    'file': file,
                    'status': 'failed',
                    'error': str(e)
                })

        # 转换为DataFrame
        df_results = pd.DataFrame(results)

        # 展开浓度数据
        if 'concentrations' in df_results.columns:
            conc_df = pd.json_normalize(df_results['concentrations'])
            conc_df.columns = [f'conc_{col}' for col in conc_df.columns]
            df_results = pd.concat([df_results, conc_df], axis=1)

        return df_results

    def _estimate_confidence(self, spectrum_scaled: np.ndarray) -> float:
        """
        估计预测置信度

        基于光谱与训练数据的相似度
        """
        # 简化的置信度估计
        # 实际应用中可以使用：
        # 1. 预测区间
        # 2. 蒙特卡洛dropout
        # 3. 集成模型的方差

        # 这里使用一个简单的启发式方法
        # 基于标准化后光谱的范围
        spectrum_range = np.max(np.abs(spectrum_scaled))

        if spectrum_range < 3:  # 在3个标准差内
            confidence = 0.95
        elif spectrum_range < 5:
            confidence = 0.85
        else:
            confidence = 0.70

        return confidence

    def _validate_results(self, result: Dict) -> Dict:
        """验证结果的合理性"""
        concentrations = result['concentrations']
        total = sum(concentrations.values())

        # 检查总和是否接近1
        if abs(total - 1.0) > 0.1:
            result['warning'] = f'浓度总和为 {total:.2f}，偏离1.0较大'
            result['confidence'] *= 0.8

        # 检查是否有负值
        for gas, conc in concentrations.items():
            if conc < 0:
                concentrations[gas] = 0
                result['warning'] = f'{gas} 浓度为负，已修正为0'
                result['confidence'] *= 0.7
            elif conc > 1:
                concentrations[gas] = 1
                result['warning'] = f'{gas} 浓度超过1，已修正为1'
                result['confidence'] *= 0.7

        # 重新归一化
        total = sum(concentrations.values())
        if total > 0:
            for gas in concentrations:
                concentrations[gas] /= total

        result['concentrations'] = concentrations
        result['total'] = sum(concentrations.values())

        return result

    def visualize_prediction(self, spectrum_file: str, save_path: str = None):
        """
        可视化预测结果

        Parameters:
        -----------
        spectrum_file : str
            光谱文件路径
        save_path : str
            保存图片的路径（可选）
        """
        # 预测
        result = self.predict_single(spectrum_file)

        # 读取光谱用于显示
        spectrum_df = pd.read_csv(spectrum_file)

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 绘制光谱
        ax1.plot(spectrum_df.iloc[:, 0], spectrum_df.iloc[:, 1], 'b-', linewidth=1.5)
        ax1.set_xlabel('Wavenumber (cm⁻¹)')
        ax1.set_ylabel('Absorption Intensity')
        ax1.set_title('Input Spectrum')
        ax1.grid(True, alpha=0.3)

        # 绘制浓度饼图
        concentrations = result['concentrations']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        wedges, texts, autotexts = ax2.pie(
            concentrations.values(),
            labels=concentrations.keys(),
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )

        ax2.set_title(f'Predicted Gas Concentrations\n(Confidence: {result["confidence"]:.0%})')

        # 美化饼图
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存: {save_path}")

        plt.show()

        return result

    def generate_report(self, result: Union[Dict, pd.DataFrame],
                        report_path: str = "prediction_report.html"):
        """
        生成预测报告

        Parameters:
        -----------
        result : Dict or pd.DataFrame
            单个预测结果或批量预测结果
        report_path : str
            报告保存路径
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>气体浓度预测报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .warning { color: #ff6b6b; font-weight: bold; }
                .success { color: #4CAF50; }
                .summary { background-color: #e8f4f8; padding: 15px; margin: 20px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>气体浓度分析报告</h1>
            <p>生成时间: {timestamp}</p>
        """

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content = html_content.format(timestamp=timestamp)

        if isinstance(result, dict):
            # 单个结果
            html_content += self._format_single_result(result)
        else:
            # 批量结果
            html_content += self._format_batch_results(result)

        html_content += """
        </body>
        </html>
        """

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"报告已生成: {report_path}")

    def _format_single_result(self, result: Dict) -> str:
        """格式化单个结果"""
        html = f"""
        <div class="summary">
            <h2>预测结果</h2>
            <p><strong>文件:</strong> {result['file']}</p>
            <p><strong>状态:</strong> <span class="success">{result['status']}</span></p>
            <p><strong>置信度:</strong> {result['confidence']:.1%}</p>
        </div>

        <table>
            <tr>
                <th>气体</th>
                <th>浓度 (%)</th>
            </tr>
        """

        for gas, conc in result['concentrations'].items():
            html += f"""
            <tr>
                <td>{gas}</td>
                <td>{conc * 100:.2f}</td>
            </tr>
            """

        html += f"""
            <tr>
                <td><strong>总和</strong></td>
                <td><strong>{result['total'] * 100:.2f}</strong></td>
            </tr>
        </table>
        """

        if 'warning' in result:
            html += f'<p class="warning">⚠️ 警告: {result["warning"]}</p>'

        return html

    def _format_batch_results(self, results: pd.DataFrame) -> str:
        """格式化批量结果"""
        # 统计信息
        n_success = len(results[results['status'] == 'success'])
        n_total = len(results)

        html = f"""
        <div class="summary">
            <h2>批量预测摘要</h2>
            <p><strong>总文件数:</strong> {n_total}</p>
            <p><strong>成功预测:</strong> {n_success}</p>
            <p><strong>失败:</strong> {n_total - n_success}</p>
        </div>

        <table>
            <tr>
                <th>序号</th>
                <th>文件</th>
                <th>NO (%)</th>
                <th>NO2 (%)</th>
                <th>SO2 (%)</th>
                <th>置信度</th>
                <th>状态</th>
            </tr>
        """

        for _, row in results.iterrows():
            if row['status'] == 'success':
                html += f"""
                <tr>
                    <td>{row['index']}</td>
                    <td>{Path(row['file']).name}</td>
                    <td>{row.get('conc_NO', 0) * 100:.2f}</td>
                    <td>{row.get('conc_NO2', 0) * 100:.2f}</td>
                    <td>{row.get('conc_SO2', 0) * 100:.2f}</td>
                    <td>{row.get('confidence', 0):.1%}</td>
                    <td class="success">成功</td>
                </tr>
                """
            else:
                html += f"""
                <tr>
                    <td>{row['index']}</td>
                    <td>{Path(row['file']).name}</td>
                    <td colspan="4">{row.get('error', 'Unknown error')}</td>
                    <td class="warning">失败</td>
                </tr>
                """

        html += "</table>"

        # 添加统计图表
        if n_success > 0:
            success_data = results[results['status'] == 'success']
            avg_no = success_data['conc_NO'].mean() * 100
            avg_no2 = success_data['conc_NO2'].mean() * 100
            avg_so2 = success_data['conc_SO2'].mean() * 100

            html += f"""
            <div class="summary">
                <h3>平均浓度</h3>
                <p>NO: {avg_no:.2f}%</p>
                <p>NO2: {avg_no2:.2f}%</p>
                <p>SO2: {avg_so2:.2f}%</p>
            </div>
            """

        return html


# 使用示例
def example_usage():
    """演示如何使用预测器"""

    print("=== 气体浓度预测器使用示例 ===\n")

    # 1. 初始化预测器
    predictor = GasConcentrationPredictor(model_dir="./output/models")

    # 2. 单个文件预测
    print("1. 单个文件预测:")
    result = predictor.predict_single("unknown_spectrum.csv")
    print(f"   NO:  {result['concentrations']['NO'] * 100:.2f}%")
    print(f"   NO2: {result['concentrations']['NO2'] * 100:.2f}%")
    print(f"   SO2: {result['concentrations']['SO2'] * 100:.2f}%")
    print(f"   置信度: {result['confidence']:.1%}\n")

    # 3. 批量预测
    print("2. 批量文件预测:")
    files = ["spectrum1.csv", "spectrum2.csv", "spectrum3.csv"]
    batch_results = predictor.predict_batch(files)
    print(batch_results[['file', 'conc_NO', 'conc_NO2', 'conc_SO2', 'confidence']])

    # 4. 可视化
    print("\n3. 生成可视化:")
    predictor.visualize_prediction("unknown_spectrum.csv", "prediction_result.png")

    # 5. 生成报告
    print("\n4. 生成HTML报告:")
    predictor.generate_report(result, "single_prediction_report.html")
    predictor.generate_report(batch_results, "batch_prediction_report.html")


if __name__ == "__main__":
    example_usage()