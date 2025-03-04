import re
import csv
from collections import defaultdict
import hashlib
from graphviz import Digraph
import argparse
import os

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
        if '-->' in line:
            parts = re.split(r'\s*-->\s*|\s*:\s*', line)
            if len(parts) >= 2:
                source = parts[0].strip()
                target = parts[1].strip()
                rel_type = parts[2].strip() if len(parts)>2 else 'RELATED'
                if rematch := re.match(r'类型\s*=\s*(.+)', rel_type):
                    rel_type = rematch.group(1)

                # 去除md符号(-)及空格
                source = re.sub(r'^[\s-]+', '', source)
                
                self.relations.append({
                    'source': self._generate_id(source),
                    'target': self._generate_id(target),
                    'type': rel_type,
                    '_o_source_': source,
                    '_o_target_': target
                })

        # 属性提取（支持多个键值对）
        if '=' in line:
            # 分割多个属性（支持空格分隔）
            pairs = re.findall(r'(\w+)\s*=\s*([^=]+)(?=\s+\w+=|$)', line)
            for key, value in pairs:
                self.concepts[self.current_concept]['props'][key.strip()] = value.strip()

    def export_csv(self, node_path, rel_path):
        """动态生成节点和关系CSV"""
        # 收集所有属性字段
        all_props = set()
        for c in self.concepts.values():
            all_props.update(c['props'].keys())
        
        # 节点文件头
        fieldnames = [':ID', 'name', 'type', 'parent'] + sorted(all_props)
        
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
                row.update(c['props'])
                writer.writerow(row)

        # 新增关系文件处理
        with open(rel_path, 'w', newline='', encoding='utf-8') as f:
            # 收集关系属性字段
            rel_props = set()
            for rel in self.relations:
                rel_props.update(rel.keys())
            rel_props -= {'source', 'target', 'type'}  # 排除固定字段
            
            # 关系文件头
            rel_fields = [':START_ID', ':END_ID', 'type'] + sorted(rel_props)
            writer = csv.DictWriter(f, fieldnames=rel_fields)
            writer.writeheader()
            
            for rel in self.relations:
                row = {
                    ':START_ID': rel['source'],
                    ':END_ID': rel['target'],
                    'type': rel.get('type', 'RELATED')
                }
                # 添加额外属性
                row.update({k: v for k, v in rel.items() 
                          if k not in ['source', 'target', 'type']})
                writer.writerow(row)

    def visualize(self):
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

    parser = NoteParser()
    
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            parser.parse_line(line.strip())
    
    parser.export_csv(os.path.join(args.output, args.input.split('.')[0]+'_concepts.csv'), os.path.join(args.output, args.input.split('.')[0]+'_relations.csv')) 
    if args.visualize:
        parser.visualize()