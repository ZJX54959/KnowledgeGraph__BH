# extract elements from .html
import re
import json
import os
from bs4 import BeautifulSoup
from lxml import etree
import pandas as pd

class KnowledgeExtractor:
    def __init__(self):
        """初始化知识提取器"""
        pass
    
    def extract_all_links(self, html_path):
        """从HTML文件中提取所有超链接的文本和URL
        
        Args:
            html_path: HTML文件路径
            
        Returns:
            list: 包含字典的列表，每个字典包含'text'和'url'键
        """
        success, soup_or_error, _ = self.load_html(html_path)
        
        if not success:
            print(f"加载HTML文件失败: {soup_or_error}")
            return []
        
        soup = soup_or_error
        links = []
        for a_tag in soup.find_all('a', href=True):
            text = a_tag.get_text().strip()
            url = a_tag['href']
            if text and url:
                links.append({'text': text, 'url': url})
        return links

    def load_html(self, html_path):
        """加载HTML文件
        
        Args:
            html_path: HTML文件路径
            
        Returns:
            tuple: (是否成功, BeautifulSoup对象或错误信息, lxml的etree对象或None)
        """
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # 检查是否是百度安全验证页面
            if self._is_verification_page(html_content):
                return False, "该页面是百度安全验证页面，无法提取内容", None
            
            # 创建BeautifulSoup对象
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 创建lxml的etree对象，用于xpath查询
            html_tree = etree.HTML(html_content)
            
            return True, soup, html_tree
            
        except Exception as e:
            return False, f"加载HTML文件失败: {str(e)}", None
    
    def _is_verification_page(self, html_content):
        """检查是否是百度安全验证页面
        
        Args:
            html_content: HTML内容
            
        Returns:
            bool: 是否是验证页面
        """
        verification_patterns = [
            '百度安全验证',
            '请输入验证码',
            'security_verify',
            '安全验证中心'
        ]
        
        for pattern in verification_patterns:
            if pattern in html_content:
                return True
        return False
    
    def extract_by_xpath(self, html_tree, xpath_expr):
        """使用xpath提取元素
        
        Args:
            html_tree: lxml的etree对象
            xpath_expr: xpath表达式
            
        Returns:
            list: 提取的元素列表
        """
        try:
            elements = html_tree.xpath(xpath_expr)
            return elements
        except Exception as e:
            print(f"xpath提取失败: {str(e)}")
            return []
    
    def extract_by_css(self, soup, css_selector):
        """使用CSS选择器提取元素
        
        Args:
            soup: BeautifulSoup对象
            css_selector: CSS选择器
            
        Returns:
            list: 提取的元素列表
        """
        try:
            elements = soup.select(css_selector)
            return elements
        except Exception as e:
            print(f"CSS选择器提取失败: {str(e)}")
            return []
    
    def extract_by_tag(self, soup, tag_name, attrs=None):
        """使用标签名和属性提取元素
        
        Args:
            soup: BeautifulSoup对象
            tag_name: 标签名
            attrs: 属性字典，可选
            
        Returns:
            list: 提取的元素列表
        """
        try:
            if attrs:
                elements = soup.find_all(tag_name, attrs=attrs)
            else:
                elements = soup.find_all(tag_name)
            return elements
        except Exception as e:
            print(f"标签提取失败: {str(e)}")
            return []
    
    def extract_baike_info(self, html_path):
        """提取百度百科页面的基本信息
        
        Args:
            html_path: HTML文件路径
            
        Returns:
            dict: 提取的信息字典
        """
        success, soup_or_error, html_tree = self.load_html(html_path)
        
        if not success:
            return {"error": soup_or_error}
        
        soup = soup_or_error
        
        # 提取标题
        title = ""
        # title_elem = soup.find('h1', class_='lemmaWgt-lemmaTitle-title')
        title_elem = soup.find('h1', class_='lemmaTitle_iuBlp J-lemma-title')
        if title_elem:
            title = title_elem.get_text().strip()
        
        # 提取摘要
        summary = ""
        # summary_elem = soup.find('div', class_='lemma-summary')
        summary_elem = soup.find('div', class_='lemmaSummary_dhg1F J-summary')
        if summary_elem:
            summary = summary_elem.get_text().strip()
        
        # 提取基本信息表格
        basic_info = {}
        # basic_info_elem = soup.find('div', class_='basic-info')
        basic_info_elem = soup.find('div', class_='basicInfo_tLQSv J-basic-info')
        if basic_info_elem:
            # name_elems = basic_info_elem.find_all('dt', class_='basicInfo-item name')
            name_elems = basic_info_elem.find_all('dt', class_='basicInfoItem_iG0fH itemName_RXMP4')
            # value_elems = basic_info_elem.find_all('dd', class_='basicInfo-item value')
            value_elems = basic_info_elem.find_all('dd', class_='basicInfoItem_iG0fH itemValue_oIfsW')
            
            for name_elem, value_elem in zip(name_elems, value_elems):
                name = name_elem.get_text().strip()
                value = value_elem.get_text().strip()
                basic_info[name] = value
        
        # 提取目录
        catalog = []
        catalog_elems = soup.select('.catalogList_dUefQ li')
        for elem in catalog_elems:
            # 获取目录级别
            level = 'level1' if 'level1' in elem.get('class', []) else 'level2'
            
            # 提取目录文本
            text_elem = elem.select_one('.catalogText_rCNjq a')
            if text_elem:
                text = text_elem.get_text().strip()
                # 添加缩进表示层级关系
                if level == 'level2':
                    text = '    ' + text
                catalog.append(text)
        
        # 提取正文段落
        content_paragraphs = []
        # content_elems = soup.select('.para')
        content_elems = soup.select('.para_WzwJ3')
        for elem in content_elems:
            text = elem.get_text().strip()
            if text:  # 只添加非空段落
                content_paragraphs.append(text)
        
        # 提取参考资料
        references = []
        reference_elems = soup.select('.referenceItem_Z9PAD.J-ref-item')
        for elem in reference_elems:
            # 提取参考资料标题和链接
            ref_link = elem.select_one('.refLink_Pcdfd')
            if ref_link:
                ref_text = ref_link.get_text().strip()
                ref_url = ref_link.get('href', '')
                if ref_url and not ref_url.startswith('http'):
                    ref_url = f"https://baike.baidu.com{ref_url}"
                
                # 提取来源和日期信息
                source = elem.select_one('span:nth-of-type(1)')
                date = elem.select_one('span:nth-of-type(2)')
                
                ref_info = {
                    'title': ref_text,
                    'url': ref_url,
                    'source': source.get_text().strip() if source else '',
                    'date': date.get_text().strip() if date else ''
                }
                references.append(ref_info)
        
        # 组装结果
        result = {
            "title": title,
            "summary": summary,
            "basic_info": basic_info,
            "catalog": catalog,
            "content": content_paragraphs,
            "references": references
        }
        
        return result
    
    def save_to_json(self, data, output_path):
        """将提取的数据保存为JSON文件
        
        Args:
            data: 要保存的数据
            output_path: 输出文件路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            print(f"保存JSON文件失败: {str(e)}")
            return False
    
    def save_to_csv(self, data, output_path):
        """将提取的数据保存为CSV文件
        
        Args:
            data: 要保存的数据（字典或列表）
            output_path: 输出文件路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            # 如果是嵌套字典，需要先展平
            if isinstance(data, dict) and any(isinstance(v, dict) for v in data.values()):
                flattened_data = {}
                for key, value in data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            flattened_data[f"{key}_{sub_key}"] = sub_value
                    else:
                        flattened_data[key] = value
                data = flattened_data
            
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            return True
        except Exception as e:
            print(f"保存CSV文件失败: {str(e)}")
            return False

# 使用示例
if __name__ == "__main__":
    from web_crawler import WebCrawler
    
    # 获取当前脚本所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(current_dir, 'Temp')
    
    # 确保Temp目录存在
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # 爬取页面
    crawler = WebCrawler(save_dir=temp_dir)
    url = "https://baike.baidu.com/item/知识图谱/8120012"
    success, result = crawler.crawl(url)
    
    if success:
        print(f"页面爬取成功，保存至: {result}")
        
        # 提取知识
        extractor = KnowledgeExtractor()
        info = extractor.extract_baike_info(result)
        
        # 保存为JSON
        json_path = os.path.join(temp_dir, 'python_info.json')
        extractor.save_to_json(info, json_path)
        print(f"知识提取成功，保存至: {json_path}")
        
        # 打印部分信息
        print(f"\n标题: {info['title']}")
        print(f"\n摘要: {info['summary'][:200]}...")
        print(f"\n基本信息条目数: {len(info['basic_info'])}")
        print(f"\n目录条目数: {len(info['catalog'])}")
        print(f"\n正文段落数: {len(info['content'])}")
    else:
        print(f"页面爬取失败: {result}")