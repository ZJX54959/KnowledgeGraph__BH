# return .html
import requests
import time
import random
import os
import re
import logging
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

class WebCrawler:
    def __init__(self, save_dir='Temp', use_selenium=True, manual_verify=True):
        """初始化爬虫类
        
        Args:
            save_dir: 保存HTML文件的目录
            use_selenium: 是否使用selenium模拟浏览器
            manual_verify: 是否在遇到验证页面时允许用户手动验证
        """
        self.ua = UserAgent()
        
        # 确保save_dir是绝对路径
        if not os.path.isabs(save_dir):
            # 获取当前脚本所在目录的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(current_dir, save_dir)
            
        self.save_dir = save_dir
        self.use_selenium = use_selenium
        self.manual_verify = manual_verify
        self.driver = None
        
        # 配置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('WebCrawler')
        
        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 常见的浏览器列表，用于生成更真实的User-Agent
        self.browsers = [
            'chrome',
            'firefox',
            'safari',
            'edge',
            'opera'
        ]
    
    def get_random_headers(self):
        """生成随机请求头，确保使用电脑端UA"""
        # 随机选择一个浏览器类型
        browser = random.choice(self.browsers)
        # 获取对应浏览器的随机UA，确保是电脑端UA
        user_agent = self.ua[browser]
        
        # 确保UA不包含移动设备相关字符串
        if 'mobile' in user_agent.lower() or 'android' in user_agent.lower() or 'iphone' in user_agent.lower():
            # 如果是移动设备UA，则使用默认的Chrome桌面版UA
            user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        
        # 随机生成一些常见的请求头参数
        accept_encodings = ['gzip, deflate, br', 'gzip, deflate', 'br', 'gzip']
        accept_languages = [
            'zh-CN,zh;q=0.9,en;q=0.8',
            'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'en-US,en;q=0.9,zh-CN;q=0.8',
            'zh-CN;q=0.9,en;q=0.8'
        ]
        
        # 随机生成referer，模拟从搜索引擎或其他页面跳转
        referers = [
            'https://www.baidu.com/s?wd=python',
            'https://www.google.com/search?q=python',
            'https://cn.bing.com/search?q=python',
            'https://www.sogou.com/web?query=python',
            'https://baike.baidu.com/',
            None  # 有时不带referer
        ]
        
        # 基础请求头
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': random.choice(accept_encodings),
            'Accept-Language': random.choice(accept_languages),
            'Connection': random.choice(['keep-alive', 'close']),
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': random.choice(['max-age=0', 'no-cache', 'no-store']),
            'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': random.choice(['same-origin', 'same-site', 'cross-site', 'none']),
            'Sec-Fetch-User': '?1',
            'force_refresh': '1'
        }
        
        # 随机添加referer
        referer = random.choice(referers)
        if referer:
            headers['Referer'] = referer
            
        # 随机添加一些额外的请求头，增加真实性
        if random.random() > 0.5:
            headers['DNT'] = '1'  # Do Not Track
        
        if random.random() > 0.7:
            viewport_widths = [1366, 1440, 1536, 1920, 2560]
            viewport_heights = [768, 900, 864, 1080, 1440]
            headers['Viewport-Width'] = str(random.choice(viewport_widths))
            headers['Viewport-Height'] = str(random.choice(viewport_heights))
        
        return headers
    
    def is_verification_page(self, html_content):
        """检查是否是百度安全验证页面
        
        Args:
            html_content: HTML内容
            
        Returns:
            bool: 是否是验证页面
        """
        # 文本模式匹配
        verification_patterns = [
            '百度安全验证',
            '请输入验证码',
            # 'security_verify',
            # '安全验证中心',
            # '验证码',
            # '人机验证',
            # '安全检查',
            '请完成下方验证',
            '您的访问出现异常',
            '访问异常',
            '请控制访问频率'
        ]
        
        for pattern in verification_patterns:
            if pattern in html_content:
                return True
        
        # 检查页面标题
        # title_patterns = ['<title>百度安全验证</title>', '<title>安全验证</title>', '<title>百度一下，你就知道</title>']
        # for pattern in title_patterns:
        #     if pattern in html_content:
        #         return True
                
        # 检查是否包含验证相关的URL
        # url_patterns = ['wappass.baidu.com', 'verify', 'captcha', 'antispam']
        # for pattern in url_patterns:
        #     if pattern in html_content:
        #         return True
                
        # 检查页面内容长度，验证页面通常较短
        # if len(html_content) < 5000 and ('百度' in html_content or 'baidu' in html_content):
        #     # 进一步检查是否缺少正常百科页面的特征
        #     if 'lemmaWgt-lemmaTitle-title' not in html_content and 'lemma-summary' not in html_content:
        #         return True
                
        return False
    
    def fetch_page(self, url, max_retries=3, retry_delay=3):
        """获取网页内容
        
        Args:
            url: 要爬取的URL
            max_retries: 最大重试次数
            retry_delay: 重试延迟(秒)
            
        Returns:
            tuple: (是否成功, HTML内容或错误信息)
        """
        # 创建一个Session对象来维持cookies
        session = requests.Session()
        
        # 模拟正常浏览器行为：先访问百度首页获取cookies
        try:
            print("先访问百度首页获取cookies...")
            baidu_url = "https://www.baidu.com"
            headers = self.get_random_headers()
            session.get(baidu_url, headers=headers, timeout=10)
            time.sleep(random.uniform(1, 3))
        except Exception as e:
            print(f"访问百度首页失败: {str(e)}")
            # 继续执行，不影响主要流程
        
        for attempt in range(max_retries):
            try:
                # 随机延迟，避免频繁请求，增加延迟时间范围
                sleep_time = random.uniform(5, 15)
                print(f"等待 {sleep_time:.2f} 秒后发送请求...")
                time.sleep(sleep_time)
                
                # 获取随机请求头
                headers = self.get_random_headers()
                
                # 构建请求参数
                request_kwargs = {
                    'headers': headers,
                    'timeout': 20,
                    'allow_redirects': True,
                }
                
                # 随机决定是否使用代理（如果有代理的话）
                # 注意：这里没有实际使用代理，如果需要，可以添加代理服务
                # if random.random() > 0.7 and self.proxies:
                #     proxy = random.choice(self.proxies)
                #     request_kwargs['proxies'] = {'http': proxy, 'https': proxy}
                #     print(f"使用代理: {proxy}")
                
                # 发送请求，使用session保持cookies
                print(f"正在请求URL: {url}")
                response = session.get(url, **request_kwargs)
                
                # 检查状态码
                if response.status_code != 200:
                    print(f"请求返回非200状态码: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 2))  # 增加更长的延迟
                        continue
                    else:
                        return False, f"达到最大重试次数，状态码: {response.status_code}"
                        
                # 获取实际URL（处理重定向）
                final_url = response.url
                if final_url != url:
                    print(f"请求被重定向到: {final_url}")
                    
                # 获取响应内容
                html_content = response.text
                
                # 检查是否是验证页面
                if self.is_verification_page(response.text):
                    print(f"遇到安全验证页面，尝试重试 ({attempt+1}/{max_retries})")
                    time.sleep(retry_delay * (attempt + 1))  # 增加延迟时间
                    continue
                
                return True, response.text
                
            except requests.exceptions.RequestException as e:
                print(f"请求失败 ({attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    return False, f"达到最大重试次数，无法获取页面: {str(e)}"
        
        return False, "达到最大重试次数，可能被反爬机制拦截"
    
    def save_html(self, html_content, url, filename=None):
        """保存HTML内容到文件
        
        Args:
            html_content: HTML内容
            url: 原始URL
            filename: 文件名，如果为None则从URL生成
            
        Returns:
            str: 保存的文件路径
        """
        if filename is None:
            # 从URL生成文件名
            filename = re.sub(r'[\\/:*?"<>|]', '_', url)
            filename = filename.replace('http_', '').replace('https_', '')
            filename = filename[:100] if len(filename) > 100 else filename  # 限制文件名长度
            filename = f"{filename}.html"
        
        file_path = os.path.join(self.save_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return file_path
    
    def init_selenium(self):
        """初始化Selenium WebDriver"""
        try:
            # 配置Chrome选项
            chrome_options = Options()
            
            # 如果需要手动验证，则不使用无头模式
            if not self.manual_verify:
                chrome_options.add_argument('--headless')  # 无头模式
                
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            
            # 获取随机UA并确保是电脑端UA
            user_agent = self.ua.random
            if 'mobile' in user_agent.lower() or 'android' in user_agent.lower() or 'iphone' in user_agent.lower():
                # 如果是移动设备UA，则使用默认的Chrome桌面版UA
                user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
            
            chrome_options.add_argument(f'user-agent={user_agent}')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            
            # 添加实验性选项，绕过检测
            chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # 初始化WebDriver
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # 设置页面加载超时
            self.driver.set_page_load_timeout(30)
            
            # 执行JavaScript来绕过WebDriver检测
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"初始化Selenium失败: {str(e)}")
            return False
    
    def close_selenium(self):
        """关闭Selenium WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                self.logger.error(f"关闭Selenium失败: {str(e)}")
            finally:
                self.driver = None
    
    def fetch_with_selenium(self, url, max_retries=3):
        """使用Selenium获取网页内容
        
        Args:
            url: 要爬取的URL
            max_retries: 最大重试次数
            
        Returns:
            tuple: (是否成功, HTML内容或错误信息)
        """
        if not self.driver and not self.init_selenium():
            return False, "无法初始化Selenium WebDriver"
        
        for attempt in range(max_retries):
            try:
                # 随机延迟
                sleep_time = random.uniform(3, 8)
                self.logger.info(f"等待 {sleep_time:.2f} 秒后使用Selenium访问页面...")
                time.sleep(sleep_time)
                
                # 访问URL
                self.logger.info(f"正在使用Selenium访问: {url}")
                self.driver.get(url)
                
                # 等待页面加载完成
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # 随机滚动页面，模拟真实用户行为
                self.logger.info("模拟页面滚动...")
                for _ in range(random.randint(3, 8)):
                    scroll_amount = random.randint(300, 700)
                    self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                    time.sleep(random.uniform(0.5, 2))
                
                # 获取页面内容
                html_content = self.driver.page_source
                
                # 检查是否是验证页面
                if self.is_verification_page(html_content):
                    self.logger.warning(f"遇到安全验证页面，尝试重试 ({attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(random.uniform(5, 10))
                        continue
                    else:
                        if self.manual_verify:
                            self.logger.info("已达到最大重试次数，浏览器将保持打开状态，请手动完成验证")
                            self.logger.info("请在浏览器中完成验证后，按回车键继续...")
                            
                            # 等待用户手动完成验证
                            try:
                                input("请在浏览器中完成验证后，按回车键继续...")
                                
                                # 验证完成后，重新获取页面内容
                                self.logger.info("正在重新获取页面内容...")
                                html_content = self.driver.page_source
                                
                                # 检查是否仍然是验证页面
                                if self.is_verification_page(html_content):
                                    self.logger.warning("验证似乎未成功完成，页面仍然是验证页面")
                                    return False, "验证未成功完成"
                                else:
                                    self.logger.info("验证成功完成，已获取到页面内容")
                                    return True, html_content
                            except Exception as e:
                                self.logger.error(f"等待验证过程中发生错误: {str(e)}")
                                return False, f"等待验证过程中发生错误: {str(e)}"
                        else:
                            return False, "达到最大重试次数，仍然遇到安全验证页面"
                
                return True, html_content
                
            except TimeoutException:
                self.logger.warning(f"页面加载超时 ({attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    # 刷新页面重试
                    try:
                        self.driver.refresh()
                    except:
                        pass
                    time.sleep(random.uniform(5, 10))
                else:
                    return False, "页面加载超时，达到最大重试次数"
                    
            except WebDriverException as e:
                self.logger.error(f"Selenium错误 ({attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    # 尝试重新初始化WebDriver
                    self.close_selenium()
                    if not self.init_selenium():
                        return False, "无法重新初始化Selenium WebDriver"
                    time.sleep(random.uniform(5, 10))
                else:
                    return False, f"Selenium错误，达到最大重试次数: {str(e)}"
                    
            except Exception as e:
                self.logger.error(f"未知错误 ({attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(5, 10))
                else:
                    return False, f"未知错误，达到最大重试次数: {str(e)}"
        
        return False, "达到最大重试次数，无法获取页面"
    
    def crawl(self, url, filename=None):
        """爬取页面并保存
        
        Args:
            url: 要爬取的URL
            filename: 保存的文件名，如果为None则从URL生成
            
        Returns:
            tuple: (是否成功, 文件路径或错误信息)
        """
        try:
            # 优先使用Selenium方式获取页面
            if self.use_selenium:
                self.logger.info("使用Selenium方式获取页面...")
                success, result = self.fetch_with_selenium(url)
                if success:
                    self.logger.info("Selenium获取页面成功")
                else:
                    self.logger.warning(f"Selenium获取页面失败: {result}，尝试使用requests方式...")
                    success, result = self.fetch_page(url)
            else:
                self.logger.info("使用requests方式获取页面...")
                success, result = self.fetch_page(url)
            
            if success:
                file_path = self.save_html(result, url, filename)
                return True, file_path
            else:
                return False, result
        finally:
            # 确保关闭Selenium WebDriver
            if self.use_selenium:
                self.close_selenium()

# 使用示例
if __name__ == "__main__":
    crawler = WebCrawler()
    url = "https://baike.baidu.com/item/知识图谱/8120012"
    success, result = crawler.crawl(url)
    
    if success:
        print(f"页面爬取成功，保存至: {result}")
    else:
        print(f"页面爬取失败: {result}")