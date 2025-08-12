# font_config.py
# 统一的matplotlib中文字体配置

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import warnings

def setup_chinese_fonts():
    """设置matplotlib的中文字体支持"""
    
    # 常见中文字体列表（按优先级排序）
    chinese_fonts = [
        'SimHei',           # 黑体 (Windows)
        'Microsoft YaHei',  # 微软雅黑 (Windows)
        'PingFang SC',      # 苹方 (macOS)
        'Hiragino Sans GB', # 冬青黑体 (macOS)  
        'WenQuanYi Micro Hei', # 文泉驿微米黑 (Linux)
        'Droid Sans Fallback', # Android字体
        'Arial Unicode MS',    # Unicode字体
        'DejaVu Sans'          # 默认字体
    ]
    
    # 检查系统中可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 找到第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    # 配置matplotlib - 混合字体以支持中文和特殊符号
    if selected_font:
        # 使用中文字体 + DejaVu Sans（支持上标下标）的组合
        plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans', 'Arial Unicode MS'] + chinese_fonts
        print(f"✅ 已设置中文字体: {selected_font} (混合DejaVu Sans支持特殊符号)")
    else:
        # 如果没找到中文字体，使用英文标题
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        print("⚠️  未找到中文字体，将使用英文标题")
        warnings.warn("建议安装中文字体以获得更好的显示效果")
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置数学文本字体，确保上标下标正常显示
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    plt.rcParams['mathtext.default'] = 'regular'
    
    return selected_font is not None

def get_title_text(chinese_text: str, english_text: str = None) -> str:
    """根据字体支持情况返回合适的标题文本"""
    # 简单检查：如果当前字体支持中文，返回中文，否则返回英文
    current_font = plt.rcParams['font.sans-serif'][0]
    
    if current_font in ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', 'WenQuanYi Micro Hei']:
        return chinese_text
    elif english_text:
        return english_text
    else:
        # 如果没有提供英文文本，返回简化的中文（去掉特殊符号）
        return chinese_text.replace('（', '(').replace('）', ')').replace('，', ', ')

# 自动设置字体
setup_chinese_fonts()