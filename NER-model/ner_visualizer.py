#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本名称：ner_visualizer.py
说明：此脚本提供了命名实体识别结果的可视化功能，可以将识别结果以彩色标注的方式展示，
     并支持导出为HTML格式，方便在浏览器中查看。
依赖：matplotlib, seaborn, pandas, numpy
使用方法：
    python ner_visualizer.py --text "北京大学位于北京市海淀区" --labels "B-ORG I-ORG I-ORG I-ORG O O B-LOC I-LOC I-LOC B-LOC I-LOC I-LOC"
    python ner_visualizer.py --input entities.json --output visualization.html
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from matplotlib.font_manager import FontProperties
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置中文字体
def set_chinese_font():
    """
    设置支持中文显示的字体
    """
    try:
        # 尝试使用系统中可能存在的中文字体
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # Windows黑体
            'C:/Windows/Fonts/simsun.ttc',   # Windows宋体
            'C:/Windows/Fonts/msyh.ttc',     # Windows微软雅黑
            '/System/Library/Fonts/PingFang.ttc',  # macOS
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'  # Linux
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                return FontProperties(fname=font_path)
        
        # 如果找不到中文字体，使用默认字体
        return FontProperties()
    except Exception as e:
        logger.warning(f"设置中文字体时出错: {e}")
        return FontProperties()

# 生成实体类型的颜色映射
def generate_entity_colors(entity_types):
    """
    为不同的实体类型生成不同的颜色
    """
    # 使用seaborn的颜色调色板
    palette = sns.color_palette("husl", len(entity_types))
    
    # 转换为十六进制颜色代码
    colors = {}
    for i, entity_type in enumerate(entity_types):
        colors[entity_type] = mcolors.rgb2hex(palette[i])
    
    return colors

# 可视化单个文本的命名实体
def visualize_entities(text, labels, output_path=None, title="命名实体识别结果"):
    """
    可视化文本中的命名实体
    
    参数:
        text: 文本字符串
        labels: 标签列表，与文本等长
        output_path: 输出图像的路径，如果为None则显示图像
        title: 图像标题
    """
    if len(text) != len(labels):
        raise ValueError(f"文本长度({len(text)})与标签长度({len(labels)})不匹配")
    
    # 提取实体类型
    entity_types = set()
    for label in labels:
        if label != 'O':
            entity_type = label[2:]  # 去掉B-/I-/E-/S-前缀
            entity_types.add(entity_type)
    
    # 生成颜色映射
    colors = generate_entity_colors(entity_types)
    
    # 创建图像
    plt.figure(figsize=(max(12, len(text) * 0.5), 4))
    
    # 设置中文字体
    font_prop = set_chinese_font()
    
    # 绘制文本字符
    for i, (char, label) in enumerate(zip(text, labels)):
        if label == 'O':
            # 非实体，使用黑色
            plt.text(i, 0.5, char, fontproperties=font_prop, fontsize=16, ha='center')
        else:
            # 实体，使用对应颜色
            entity_type = label[2:]  # 去掉B-/I-/E-/S-前缀
            color = colors[entity_type]
            
            # 绘制背景色
            plt.fill_between([i-0.4, i+0.4], 0.1, 0.9, color=color, alpha=0.3)
            
            # 绘制文本
            plt.text(i, 0.5, char, fontproperties=font_prop, fontsize=16, ha='center')
            
            # 在首个字符上方标注实体类型
            if label.startswith('B-') or label.startswith('S-'):
                plt.text(i, 0.9, entity_type, fontproperties=font_prop, fontsize=10, ha='center')
    
    # 设置图像属性
    plt.xlim(-1, len(text))
    plt.ylim(0, 1)
    plt.title(title, fontproperties=font_prop, fontsize=18)
    plt.axis('off')
    
    # 添加图例
    legend_elements = []
    for entity_type, color in colors.items():
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.3, label=entity_type))
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(legend_elements))
    
    plt.tight_layout()
    
    # 保存或显示图像
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        logger.info(f"可视化结果已保存到: {output_path}")
    else:
        plt.show()

# 生成HTML格式的可视化结果
def generate_html(texts, labels_list, entities_list, output_path):
    """
    生成HTML格式的可视化结果
    
    参数:
        texts: 文本列表
        labels_list: 标签列表的列表
        entities_list: 实体列表的列表
        output_path: 输出HTML文件的路径
    """
    # 提取所有实体类型
    all_entity_types = set()
    for labels in labels_list:
        for label in labels:
            if label != 'O':
                entity_type = label[2:]  # 去掉B-/I-/E-/S-前缀
                all_entity_types.add(entity_type)
    
    # 生成颜色映射
    colors = generate_entity_colors(all_entity_types)
    
    # 生成HTML内容
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>命名实体识别结果可视化</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { margin-bottom: 30px; }
            .text { font-size: 18px; line-height: 1.8; }
            .entity { padding: 2px 0; border-radius: 3px; }
            .entity-label { font-size: 12px; font-weight: bold; margin-left: 2px; }
            .entity-table { border-collapse: collapse; width: 100%; margin-top: 10px; }
            .entity-table th, .entity-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            .entity-table th { background-color: #f2f2f2; }
            h1, h2 { color: #333; }
        </style>
    </head>
    <body>
        <h1>命名实体识别结果可视化</h1>
    """
    
    # 添加图例
    html_content += """<div style="margin: 20px 0;">
        <h3>实体类型图例:</h3>
        <div style="display: flex; flex-wrap: wrap;">
    """
    
    for entity_type, color in colors.items():
        html_content += f"""<div style="margin-right: 15px; margin-bottom: 5px;">
            <span style="background-color: {color}; padding: 2px 5px; border-radius: 3px;">{entity_type}</span>
        </div>
        """
    
    html_content += """</div></div>"""
    
    # 添加每个文本的可视化结果
    for i, (text, labels, entities) in enumerate(zip(texts, labels_list, entities_list)):
        html_content += f"""<div class="container">
            <h2>文本 {i+1}</h2>
            <div class="text">
        """
        
        # 处理文本和标签
        i = 0
        while i < len(text):
            if labels[i] == 'O':
                # 非实体
                html_content += text[i]
                i += 1
            else:
                # 实体
                entity_type = labels[i][2:]  # 去掉B-/I-/E-/S-前缀
                color = colors[entity_type]
                
                # 找到实体的结束位置
                start = i
                while i < len(text) and labels[i] != 'O' and labels[i][2:] == entity_type:
                    i += 1
                end = i
                
                entity_text = text[start:end]
                html_content += f"""<span class="entity" style="background-color: {color};">{entity_text}
                    <span class="entity-label">{entity_type}</span></span>"""
        
        html_content += """</div>
            <h3>识别到的实体:</h3>
            <table class="entity-table">
                <tr><th>实体</th><th>类型</th></tr>
        """
        
        # 添加实体表格
        for entity, entity_type in entities:
            color = colors[entity_type]
            html_content += f"""<tr>
                <td><span style="background-color: {color}; padding: 2px 5px;">{entity}</span></td>
                <td>{entity_type}</td>
            </tr>
            """
        
        html_content += """</table>
        </div>
        <hr>
        """
    
    html_content += """</body>
    </html>
    """
    
    # 写入HTML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML可视化结果已保存到: {output_path}")

# 从JSON文件加载实体识别结果
def load_entities_from_json(file_path):
    """
    从JSON文件加载实体识别结果
    
    参数:
        file_path: JSON文件路径
        
    返回:
        texts: 文本列表
        entities_list: 实体列表的列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    entities_list = []
    
    for item in data:
        texts.append(item['text'])
        entities = [(entity['entity'], entity['type']) for entity in item['entities']]
        entities_list.append(entities)
    
    return texts, entities_list

# 从实体列表重建标签序列
def reconstruct_labels(text, entities):
    """
    从实体列表重建标签序列
    
    参数:
        text: 文本字符串
        entities: 实体列表，每个元素为(实体文本, 实体类型)元组
        
    返回:
        labels: 标签列表，与文本等长
    """
    labels = ['O'] * len(text)
    
    for entity, entity_type in entities:
        # 查找实体在文本中的所有位置
        start = 0
        while True:
            start = text.find(entity, start)
            if start == -1:
                break
            
            # 标注实体
            if len(entity) == 1:
                labels[start] = f'S-{entity_type}'
            else:
                labels[start] = f'B-{entity_type}'
                for i in range(start + 1, start + len(entity) - 1):
                    labels[i] = f'I-{entity_type}'
                labels[start + len(entity) - 1] = f'E-{entity_type}'
            
            start += len(entity)
    
    return labels

# 主函数
def main():
    parser = argparse.ArgumentParser(description="命名实体识别结果可视化工具")
    
    # 输入参数组
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="要可视化的文本")
    input_group.add_argument("--input", help="包含实体识别结果的JSON文件路径")
    
    # 其他参数
    parser.add_argument("--labels", help="空格分隔的标签序列，与文本等长")
    parser.add_argument("--output", help="输出文件路径，支持.png, .jpg, .pdf或.html格式")
    
    args = parser.parse_args()
    
    # 处理单个文本
    if args.text:
        if not args.labels:
            parser.error("使用--text参数时，必须提供--labels参数")
        
        text = args.text
        labels = args.labels.split()
        
        if len(text) != len(labels):
            parser.error(f"文本长度({len(text)})与标签长度({len(labels)})不匹配")
        
        # 提取实体
        entities = []
        entity = ""
        entity_type = ""
        
        for i, (char, label) in enumerate(zip(text, labels)):
            if label.startswith('B-'):
                if entity:
                    entities.append((entity, entity_type))
                entity = char
                entity_type = label[2:]
            elif label.startswith('I-') and entity and entity_type == label[2:]:
                entity += char
            elif label.startswith('E-') and entity and entity_type == label[2:]:
                entity += char
                entities.append((entity, entity_type))
                entity = ""
                entity_type = ""
            elif label.startswith('S-'):
                if entity:
                    entities.append((entity, entity_type))
                entities.append((char, label[2:]))
                entity = ""
                entity_type = ""
            elif label == 'O':
                if entity:
                    entities.append((entity, entity_type))
                    entity = ""
                    entity_type = ""
        
        # 处理最后一个实体
        if entity:
            entities.append((entity, entity_type))
        
        # 输出可视化结果
        if args.output:
            if args.output.endswith('.html'):
                generate_html([text], [labels], [entities], args.output)
            else:
                visualize_entities(text, labels, args.output)
        else:
            visualize_entities(text, labels)
    
    # 处理JSON文件
    elif args.input:
        texts, entities_list = load_entities_from_json(args.input)
        
        # 重建标签序列
        labels_list = []
        for text, entities in zip(texts, entities_list):
            labels = reconstruct_labels(text, entities)
            labels_list.append(labels)
        
        # 输出可视化结果
        if args.output:
            if args.output.endswith('.html'):
                generate_html(texts, labels_list, entities_list, args.output)
            else:
                # 如果不是HTML格式，只可视化第一个文本
                if texts:
                    visualize_entities(texts[0], labels_list[0], args.output)
                    logger.warning("非HTML格式只能可视化第一个文本，其他文本被忽略")
        else:
            # 交互式显示每个文本的可视化结果
            for i, (text, labels) in enumerate(zip(texts, labels_list)):
                visualize_entities(text, labels, title=f"文本 {i+1}")

if __name__ == "__main__":
    main()