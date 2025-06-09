import re
import csv
from collections import defaultdict
import hashlib
import argparse
import os
try:
    from graphviz import Digraph
    VISUAL = True
except:
    print('graphviz not installed, skip visualize')
    VISUAL = False

class NoteParser:
    def __init__(self):
        self.concepts = defaultdict(dict)
        self.relations = []
        self.current_concept = None

    def _generate_id(self, name):
        """生成唯一标识符"""
        return hashlib.md5(name.encode()).hexdigest()[:8]

    def parse_line(self, line):
        """解析单行Markdown内容
        ::param line: 单行Markdown内容
        结构：
        ## 【核心概念】
        - 概念名 @ 概念类型
        ## 【关系】
        - 概念名 --> 概念名 : 关系类型
        """
        # 检测核心概念
        if match := re.match(r'## 【核心概念】', line):
            self.current_concept = None
            return
        
        # 概念定义（支持多级缩进）
        if match := re.match(r'(-+)\s*(.*?)\s*@(\w+)', line):
            indent = len(match.group(1))
            name = match.group(2).strip()
            ctype = match.group(3)
            
            # 生成概念ID
            cid = self._generate_id(name)
            self.concepts[cid] = {
                'id': cid,
                'name': name,
                'type': ctype,
                'parent': self.current_concept,
                'props': {},
                '_o_name_': name
            }
            
            # 更新当前概念层级
            if indent == 1:
                self.current_concept = cid

        # 关系提取
        # 匹配格式：- 源概念 --> 目标概念 : 关系类型 [属性键=属性值 ...]
        # 考虑到关系类型和属性可能混合，使用更精确的匹配
        if match := re.match(r'^-+\s*(.*?)\s*-->\s*(.*?)\s*:\s*([^\s]+)(?:\s+(.*))?$', line):
            source = match.group(1).strip()
            target = match.group(2).strip()
            rel_type = match.group(3).strip()
            props_str = match.group(4) # 属性字符串，可能为None

            # 去除md符号(-)及空格
            source = re.sub(r'^[\s-]+', '', source)

            # 如果rel_type为f"类型={_type_}"，则令rel_type = _type_
            if match := re.match(r'类型\s*=\s*(.+)', rel_type):
                rel_type = match.group(1)

            rel_props = {}
            if props_str:
                # 提取关系的属性，支持键=值和键="值"格式
                prop_matches = re.finditer(r'(\w+)\s*=\s*([^\s=]+|"[^"]+")', props_str)
                for p_match in prop_matches:
                    key, value = p_match.groups()
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    rel_props[key.strip()] = value.strip()

            relation = {
                'source': self._generate_id(source),
                'target': self._generate_id(target),
                'type': rel_type,
                '_o_source_': source,
                '_o_target_': target
            }
            relation.update(rel_props)  # 添加属性到关系字典
            self.relations.append(relation)

        # 概念属性提取（支持多个键值对和带引号的值）
        # 确保只在当前概念存在时才尝试提取属性
        if self.current_concept and '=' in line:
            # 使用更健壮的属性解析模式，只匹配行尾或下一个键值对前的属性
            prop_matches = re.finditer(r'(\w+)\s*=\s*([^\s=]+|"[^"]+")', line)
            for match in prop_matches:
                key, value = match.groups()
                # 处理带引号的值
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                self.concepts[self.current_concept]['props'][key.strip()] = value.strip()

    def export_csv(self, node_path, rel_path):
        """动态生成节点和关系CSV"""
        # 收集所有属性字段
        all_props = set()
        for c in self.concepts.values():
            all_props.update(c['props'].keys())
        
        fieldnames = [':ID', 'name', 'type', 'parent'] + [f'{prop}:LABEL' for prop in sorted(all_props)]
        
        with open(node_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for c in self.concepts.values():
                row = {
                    ':ID': c['id'],
                    'name': c['name'],
                    'type': c['type'], 
                    'parent': c['parent'] or ''
                }
                for prop_key, prop_value in c['props'].items():
                    row[f'{prop_key}:LABEL'] = prop_value
                writer.writerow(row)

        # 新增关系文件处理
        with open(rel_path, 'w', newline='', encoding='utf-8') as f:
            # 收集关系属性字段
            rel_props = set()
            for rel in self.relations:
                # 排除固定字段，只收集额外属性
                for k in rel.keys():
                    if k not in ['source', 'target', 'type', '_o_source_', '_o_target_']:
                        rel_props.add(k)
            
            # 关系文件头
            rel_fields = [':START_ID', ':END_ID', ':TYPE', '_o_source_', '_o_target_'] + [f'{prop}:LABEL' for prop in sorted(rel_props)]
            writer = csv.DictWriter(f, fieldnames=rel_fields)
            writer.writeheader()
            
            for rel in self.relations:
                row = {
                    ':START_ID': rel['source'],
                    ':END_ID': rel['target'],
                    ':TYPE': rel.get('type', 'RELATED'),
                    '_o_source_': rel.get('_o_source_', ''),
                    '_o_target_': rel.get('_o_target_', '')
                }
                # 添加额外属性，键名加上 :LABEL 后缀
                for k, v in rel.items():
                    if k not in ['source', 'target', 'type', '_o_source_', '_o_target_']:
                        row[f'{k}:LABEL'] = v
                writer.writerow(row)

    def visualize(self):
        if not VISUAL: return
        dot = Digraph()
        for c in self.concepts.values():
            dot.node(c['id'], c['name'])
        for r in self.relations:
            dot.edge(r['source'], r['target'], label=r['type'])
        dot.render('knowledge_preview')

# 使用示例
if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', '--input', type=str, help='input file')
    psr.add_argument('-o', '--output', type=str, help='output dir', default='output')
    psr.add_argument('-v', '--visualize', action='store_true', help='visualize the knowledge graph')
    args = psr.parse_args()

    def main(input=args.input):
        parser = NoteParser()
        
        with open(input, 'r', encoding='utf-8') as f:
            for line in f:
                parser.parse_line(line.strip())
        
        input_filename = os.path.splitext(os.path.basename(input))[0]
        parser.export_csv(os.path.join(args.output, input_filename + '_concepts.csv'), 
                        os.path.join(args.output, input_filename + '_relations.csv'))
        
        if args.visualize:
            parser.visualize()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.input is None:
        psr.print_help()
        exit(1)
    elif not os.path.exists(args.input):
        print(f"\033[33m输入路径不存在:\033[0m {args.input}")
        exit(1)
    elif os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.endswith('.md'):
                print(f"\033[32m处理文件:\033[0m {file}")
                main(os.path.join(os.path.abspath(args.input), file))
    elif os.path.isfile(args.input):
        main()