# Construct Knowledge Graph from HTML Elements
from ast import arg
import os
import re
import json
from bs4 import BeautifulSoup
from web_crawler import WebCrawler
from knowledge_extractor import KnowledgeExtractor
import argparse

class KnowledgeGraphConstructor:
    def __init__(self, base_url="", save_dir='KnowledgeGraph'):
        """
        初始化知识图谱构造器
        
        Args:
            base_url: 起始URL
            save_dir: 保存目录
        """
        self.base_url = base_url
        self.crawler = WebCrawler(save_dir='Temp')
        self.extractor = KnowledgeExtractor()
        
        # 确保save_dir是绝对路径
        if not os.path.isabs(save_dir):
            # 获取当前脚本所在目录的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(current_dir, save_dir)
            
        self.save_dir = save_dir
        
        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 知识图谱
        self.knowledge_graph = []
        
        # 已爬取的URL集合，避免重复爬取
        self.crawled_urls = set()
    
    def extract_links(self, html_path):
        """
        从HTML文件中提取百科链接
        
        Args:
            html_path: HTML文件路径
            
        Returns:
            list: 链接列表，每个元素为(链接文本, URL)
        """
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取百科内容区域的链接
            links = []
            content_div = soup.find('div', class_='mainContent_MGLNI')
            
            if content_div:
                # 查找所有链接，包括内联链接
                # a_tags = content_div.find_all('a', class_=['innerLink_KLXyc', 'lemma_inlink'], href=True)
                a_tags = content_div.find_all('a', class_=['innerLink_KLXyc', 'lemma_inlink'], href=True)
                
                for a in a_tags:
                    href = a.get('href')
                    text = a.get_text().strip()
                    
                    # 只保留百科内部链接
                    if href.startswith('/item/'):
                        full_url = f"https://baike.baidu.com{href.split('?')[0]}"  # 去除查询参数
                        links.append((text, full_url))
            
            return links
        except Exception as e:
            print(f"提取链接失败: {str(e)}")
            return []
    
    def create_ontology(self, url):
        """
        从URL创建本体
        
        Args:
            url: 百科URL
            
        Returns:
            dict: 本体字典
        """
        # 添加到已爬取集合，避免重复爬取
        self.crawled_urls.add(url)
        
        # 检查是否有缓存文件
        filename = re.sub(r'[\\/:*?"<>|]', '_', url)
        filename = filename.replace('http_', '').replace('https_', '')
        filename = filename[:100] if len(filename) > 100 else filename
        cache_file = os.path.join(self.crawler.save_dir, f"{filename}.html")
        if os.path.exists(cache_file):
            print(f"使用缓存文件: {cache_file}")
            success, html_path = True, cache_file
        else:
            # 爬取页面
            success, html_path = self.crawler.crawl(url)
            
        if not success:
            print(f"爬取页面失败: {html_path}")
            return None
        
        # 提取百科信息
        info = self.extractor.extract_baike_info(html_path)
        
        if "error" in info:
            print(f"提取信息失败: {info['error']}")
            return None
        
        # 提取链接
        links = self.extract_links(html_path)
        
        # 创建本体
        ontology = {
            "name": info["title"],
            "contents": info["summary"],
            "children": [],
            "parents": [],
            "URL": url,
            "basic_info": info["basic_info"],
            "catalog": info["catalog"],
            "content_paragraphs": info["content"],
            "references": info["references"]
        }
        
        return ontology, links
    
    def build_graph(self, start_url, max_depth=2, max_nodes=100):
        """
        构建知识图谱
        
        Args:
            start_url: 起始URL
            max_depth: 最大深度
            max_nodes: 最大节点数
            
        Returns:
            list: 知识图谱
        """
        # 重置状态
        self.knowledge_graph = []
        self.crawled_urls = set()
        
        # 使用BFS构建图
        queue = [(start_url, 0)]  # (URL, 深度)
        url_to_index = {}  # URL到索引的映射
        
        while queue and len(self.knowledge_graph) < max_nodes:
            url, depth = queue.pop(0)
            
            if depth > max_depth:
                continue
            
            print(f"正在处理: {url} (深度: {depth})")
            result = self.create_ontology(url)
            
            if result is None:
                continue
                
            ontology, links = result
            print(f"links: {links}")
            
            # 检查当前URL是否已经存在于图中（可能是占位符）
            if url in url_to_index:
                current_index = url_to_index[url]
                # 合并占位符信息与新提取的本体信息
                # 确保children和parents列表不重复添加
                existing_children = self.knowledge_graph[current_index].get("children", [])
                existing_parents = self.knowledge_graph[current_index].get("parents", [])

                # 更新本体信息，保留已有的children和parents
                self.knowledge_graph[current_index].update(ontology)
                self.knowledge_graph[current_index]["children"] = list(set(tuple(sorted(d.items())) for d in existing_children + ontology["children"]))
                self.knowledge_graph[current_index]["parents"] = list(set(tuple(sorted(d.items())) for d in existing_parents + ontology["parents"]))

            else:
                # 添加到图中
                current_index = len(self.knowledge_graph)
                url_to_index[url] = current_index
                self.knowledge_graph.append(ontology)
            
            # 处理链接
            if depth < max_depth:
                for link_text, link_url in links:
                    # 如果链接已经在图中，建立关系
                    if link_url in url_to_index:
                        child_index = url_to_index[link_url]
                        
                        # 添加父子关系
                        self.knowledge_graph[current_index]["children"].append({
                            "name": link_text,
                            "index": child_index
                        })
                        
                        self.knowledge_graph[child_index]["parents"].append({
                            "name": ontology["name"],
                            "index": current_index
                        })
                    else:
                        # 如果链接未被处理过，添加到队列中，并为它在图中创建一个占位符
                        if link_url not in self.crawled_urls:
                            # 标记为已爬取，避免重复添加到队列
                            self.crawled_urls.add(link_url)
                            queue.append((link_url, depth + 1))

                            # 为未来的子节点创建一个占位符，并建立父子关系
                            placeholder_index = len(self.knowledge_graph)
                            self.knowledge_graph.append({
                                "name": link_text, # 暂时使用链接文本作为名称
                                "URL": link_url,
                                "children": [],
                                "parents": [{
                                    "name": ontology["name"],
                                    "index": current_index
                                }]
                            })
                            url_to_index[link_url] = placeholder_index

                            self.knowledge_graph[current_index]["children"].append({
                                "name": link_text,
                                "index": placeholder_index
                            })
                        # 无论链接是否已在队列中或已处理，如果它在url_to_index中，都尝试建立关系
                        elif link_url in url_to_index:
                            child_index = url_to_index[link_url]
                            # 避免重复添加父子关系
                            if {"name": link_text, "index": child_index} not in self.knowledge_graph[current_index]["children"]:
                                self.knowledge_graph[current_index]["children"].append({
                                    "name": link_text,
                                    "index": child_index
                                })
                            # 确保父节点关系也正确建立
                            if {"name": ontology["name"], "index": current_index} not in self.knowledge_graph[child_index]["parents"]:
                                self.knowledge_graph[child_index]["parents"].append({
                                    "name": ontology["name"],
                                    "index": current_index
                                })

                    
                    print(f"\033[32m已处理: \033[0m{link_url}")
                
            print(f"depth: {depth}")
        
        return self.knowledge_graph
    
    def save_graph(self, output_path=None):
        """
        保存知识图谱
        
        Args:
            output_path: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            str: 保存的文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.save_dir, "knowledge_graph.json")
        # 确保输出路径是绝对路径
        if not os.path.isabs(output_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(current_dir, output_path)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_graph, f, ensure_ascii=False, indent=4)
            print(f"知识图谱已保存至: {output_path}")
            return output_path
        except Exception as e:
            print(f"保存知识图谱失败: {str(e)}")
            return None

# 使用示例
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--base_url", type=str, default="https://baike.baidu.com/item/知识图谱/8120012", help="起始URL")
        parser.add_argument('--max_depth', type=int, default=2, help='最大深度')
        parser.add_argument('--max_nodes', type=int, default=1, help='最大节点数')
        args = parser.parse_args()
        # 创建知识图谱构造器
        constructor = KnowledgeGraphConstructor()
        
        # 构建知识图谱
        start_url = args.base_url or "https://baike.baidu.com/item/知识图谱/8120012"
        print(f"开始构建知识图谱，起始URL: {start_url}")
        graph = constructor.build_graph(start_url, max_depth=args.max_depth, max_nodes=args.max_nodes)
        
        # 保存知识图谱
        if graph:
            constructor.save_graph()
        
            # 打印统计信息
            print(f"\n知识图谱构建完成，共包含\033[93m {len(graph)} \033[0m个节点")
            for i, node in enumerate(graph):
                print(f"节点 {i+1}: {node['name']} - 子节点数: {len(node['children'])} - 父节点数: {len(node['parents'])}")
        else:
            print("知识图谱构建失败，没有生成节点。")

    except Exception as e:
        print(f"\033[33m脚本执行过程中发生错误: {e}\033[0m")
        import traceback
        traceback.print_exc()
