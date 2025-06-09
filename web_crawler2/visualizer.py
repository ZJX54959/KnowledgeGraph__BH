import json
import networkx as nx
import matplotlib.pyplot as plt
import os

class KnowledgeGraphVisualizer:
    def __init__(self, json_path='KnowledgeGraph/knowledge_graph.json'):
        # 确保json_path是绝对路径
        if not os.path.isabs(json_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, json_path)

        self.json_path = json_path
        self.graph = nx.DiGraph()

    def load_graph_data(self):
        if not os.path.exists(self.json_path):
            print(f"错误：知识图谱文件未找到：{self.json_path}")
            return False
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.data = data
            return True
        except Exception as e:
            print(f"加载知识图谱数据失败：{e}")
            return False

    def build_networkx_graph(self):
        if not hasattr(self, 'data'):
            print("请先加载知识图谱数据。")
            return

        # 添加节点
        for i, node_data in enumerate(self.data):
            self.graph.add_node(i, name=node_data.get('name', f'Node {i}'), url=node_data.get('URL', ''))

        # 添加边
        for i, node_data in enumerate(self.data):
            # 处理子节点关系
            for child in node_data.get('children', []):
                child_index = child.get('index')
                if child_index is not None and child_index < len(self.data):
                    self.graph.add_edge(i, child_index, relation='child_of')
            # 处理父节点关系（如果需要，也可以从父节点反向添加）
            # for parent in node_data.get('parents', []):
            #     parent_index = parent.get('index')
            #     if parent_index is not None and parent_index < len(self.data):
            #         self.graph.add_edge(parent_index, i, relation='parent_of')

    def visualize_graph(self):
        if self.graph.number_of_nodes() == 0:
            print("图谱为空，无法可视化。")
            return
        
        num_nodes = self.graph.number_of_nodes()
        
        # 根据节点数量动态调整参数
        figsize_base = 12
        figsize_val = max(figsize_base, num_nodes / 5) # 每5个节点增加1英寸
        node_size_base = 3000
        node_size_val = max(500, node_size_base / (1 + num_nodes / 20)) # 节点越多，节点越小
        font_size_base = 10
        font_size_val = max(5, font_size_base / (1 + num_nodes / 50)) # 节点越多，字体越小
        k_val = 0.5 / (1 + num_nodes / 100)**0.5 # 节点越多，k越小，节点间斥力越大，尝试分散
        iterations_val = 50 + int(num_nodes / 10) # 节点越多，迭代次数越多

        plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
        
        plt.figure(figsize=(figsize_val, figsize_val))
        
        # 尝试使用 'graphviz_layout' 如果 pygraphviz 安装了
        try:
            import pygraphviz
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog='neato') # neato, dot, fdp, circo, twopi
            print("使用 Graphviz (neato) 布局")
        except ImportError:
            print("未找到 pygraphviz，将使用 spring_layout。如果节点过多，布局可能拥挤。")
            print("可以尝试安装 pygraphviz 以获得更好的布局效果: pip install pygraphviz")
            pos = nx.spring_layout(self.graph, k=k_val, iterations=iterations_val, seed=42) 

        # 绘制节点
        nx.draw_networkx_nodes(self.graph, pos, node_color='skyblue', node_size=node_size_val, alpha=0.9)

        # 绘制边
        nx.draw_networkx_edges(self.graph, pos, edgelist=self.graph.edges(),
                               width=1, alpha=0.7, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=10)

        # 绘制节点标签
        node_labels = {i: self.graph.nodes[i]['name'] for i in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=font_size_val, font_weight='bold')

        plt.title("知识图谱可视化", fontsize=font_size_val + 4)
        plt.axis('off')  # 关闭坐标轴
        plt.tight_layout() # 调整布局以适应图像
        plt.show()

    def print_tree_view(self, max_depth=float('inf'), max_children=float('inf')):
        if not hasattr(self, 'data') or not self.data:
            print("知识图谱数据为空，无法生成树状视图。")
            return

        node_map = {node['URL']: node for node in self.data if 'URL' in node}
        children_map = {node['URL']: [] for node in self.data if 'URL' in node}
        all_urls = set(node['URL'] for node in self.data if 'URL' in node)

        for node_data in self.data:
            current_url = node_data.get('URL')
            if current_url:
                for child_info in node_data.get('children', []):
                    child_index = child_info.get('index')
                    if child_index is not None and child_index < len(self.data):
                        child_url = self.data[child_index].get('URL')
                        if child_url and current_url != child_url:
                            children_map[current_url].append(child_url)

        is_child_of_any = set()
        for url, children_list in children_map.items():
            for child_url in children_list:
                is_child_of_any.add(child_url)

        root_urls = [url for url in all_urls if url not in is_child_of_any]

        if not root_urls:
            print("未找到明确的根节点，可能存在循环或图谱结构复杂。将尝试从第一个节点开始打印。")
            if self.data:
                first_node_url = self.data[0].get('URL')
                if first_node_url:
                    root_urls = [first_node_url]
                else:
                    print("第一个节点没有URL，无法开始树状打印。")
                    return
            else:
                return

        def print_node_recursive(url, prefix, is_last, current_depth, visited_urls):
            if url in visited_urls or current_depth > max_depth:
                return
            visited_urls.add(url)

            node = node_map.get(url)
            if node:
                connector = "└─" if is_last else "├─"
                print(f"{prefix}{connector} \033[32m{node['name']}\033[0m \033[2m({node['URL']})\033[0m")
                
                children_to_print = children_map.get(url, [])[:max_children]
                for i, child_url in enumerate(children_to_print):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    print_node_recursive(child_url, new_prefix, i == len(children_to_print) - 1, current_depth + 1, visited_urls.copy())
                if len(children_map.get(url, [])) > max_children:
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    print(f"{new_prefix}└─ ... (更多子节点)")

        print("\n知识图谱树状视图：")
        visited_urls = set()
        for i, root_url in enumerate(root_urls):
            print_node_recursive(root_url, "", i == len(root_urls) - 1, 1, visited_urls.copy())

import argparse

# 示例使用
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="知识图谱可视化工具")
    parser.add_argument('--json_path', type=str, default='KnowledgeGraph/knowledge_graph.json', help='知识图谱JSON文件路径')
    parser.add_argument('--depth', type=int, default=3, help='树状图显示的最大深度')
    parser.add_argument('--nodes', type=int, default=6, help='同一父级下最多显示的子节点数量')
    parser.add_argument('--no-graph', action='store_true', help='不显示图形化界面')
    parser.add_argument('--no-tree', action='store_true', help='不打印树状视图')

    args = parser.parse_args()

    visualizer = KnowledgeGraphVisualizer(json_path=args.json_path)
    if visualizer.load_graph_data():
        if not args.no_graph or not args.no_tree:
            visualizer.build_networkx_graph()
        
        if not args.no_graph:
            visualizer.visualize_graph()
        
        if not args.no_tree:
            visualizer.print_tree_view(max_depth=args.depth, max_children=args.nodes)