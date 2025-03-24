import pdfplumber
import fitz
from pathlib import Path
from typing import Union, Callable, Optional
import logging
import re
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 新增配置常量
IMAGE_PLACEHOLDER = "[图片]"
FORMULA_PLACEHOLDER = "[公式]"
MATH_SYMBOLS = r'[\∑∏∫∮√∞∠∥≈≠≡≤≥±×÷→⇌⇔∨∧¬∃∀]'  # 常见数学符号正则
DIR_MARKER = {
    "\uf06e": ("●", 0),  # 实心圆圈，0级缩进
    "\uf075": ("◆", 1)   # 实心菱形，1级缩进
}

def extract_pdf(
    input_path: Union[str, Path],
    output_dir: Union[str, Path] = None,
    engine: str = "pdfplumber",
    image_handler: Optional[Callable] = None,  # 新增图片处理回调
    formula_handler: Optional[Callable] = None  # 新增公式处理回调
) -> str:
    """
    提取PDF文本内容（支持批量处理）
    
    参数：
    input_path: PDF文件路径或包含PDF的目录路径
    output_dir: 文本输出目录（None则不保存文件）
    engine: 使用的解析引擎（pdfplumber 或 pymupdf）
    
    返回：
    提取的文本内容（批量处理时返回空字符串）
    """
    # 创建输出目录
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理单个文件
    if Path(input_path).is_file():
        return _process_single_file(Path(input_path), output_dir, engine, image_handler, formula_handler)
    
    # 处理目录
    elif Path(input_path).is_dir():
        _process_directory(Path(input_path), output_dir, engine, image_handler, formula_handler)
        return ""
    
    else:
        raise FileNotFoundError(f"路径不存在: {input_path}")

def _process_single_file(file_path: Path, output_dir: Path, engine: str, image_handler, formula_handler) -> str:
    try:
        if engine == "pdfplumber":
            text = _extract_with_pdfplumber(file_path, image_handler, formula_handler)
        elif engine == "pymupdf":
            text = _extract_with_pymupdf(file_path, image_handler, formula_handler)
        else:
            raise ValueError(f"不支持的引擎: {engine}")
        
        if output_dir:
            output_path = output_dir / f"{file_path.stem}.txt"
            output_path.write_text(text, encoding="utf-8")
            logger.info(f"文件已保存: {output_path}")
        
        return text
    
    except Exception as e:
        logger.error(f"处理文件 {file_path.name} 失败: {str(e)}")
        raise

def _process_directory(dir_path: Path, output_dir: Path, engine: str, image_handler, formula_handler):
    for pdf_file in dir_path.glob("*.pdf"):
        try:
            _process_single_file(pdf_file, output_dir, engine, image_handler, formula_handler)
        except Exception as e:
            logger.error(f"跳过文件 {pdf_file.name}: {str(e)}")
            continue

def _extract_with_pdfplumber(file_path: Path, image_handler, formula_handler) -> str:
    full_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # 图片检测
            if page.images:
                full_text.append(_handle_images(page, image_handler))
                
            # 文本提取与公式处理
            text = page.extract_text(layout=True, x_tolerance=1, y_tolerance=1)
            processed_text = _process_formulas(text, formula_handler)
            full_text.append(processed_text)
            
            # 增强元数据获取
            chars = page.chars
            if chars:
                # 获取字体信息（示例）
                math_fonts = {'Cambria Math', 'STIX'}
                font_names = {c['fontname'] for c in chars}
                has_math_font = not math_fonts.isdisjoint(font_names)
                
                # 获取位置信息（示例）
                x0 = min(c['x0'] for c in chars)
                x1 = max(c['x1'] for c in chars)
                is_centered = abs((x0 + x1)/2 - page.width/2) < 20
    return "\n".join(full_text)

def _extract_with_pymupdf(file_path: Path, image_handler, formula_handler) -> str:
    full_text = []
    doc = fitz.open(file_path)
    for page in doc:
        # 图片检测
        if page.get_images():
            full_text.append(_handle_images(page, image_handler))
            
        # 文本提取与公式处理
        text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        processed_text = _process_formulas(text, formula_handler)
        full_text.append(processed_text)
        
        # 增强元数据获取
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0 and 'font' in b:  # 文本块
                font = b['font']
                is_math_font = "Math" in font
                # 计算块居中程度
                b_center = (b['bbox'][0] + b['bbox'][2])/2
                page_center = page.rect.width/2
                is_centered = abs(b_center - page_center) < 20
    return "\n".join(full_text)

# 新增处理函数 ---------------------------------------------------
def _handle_images(page, custom_handler: Optional[Callable] = None) -> str:
    """处理图片逻辑，预留处理接口"""
    if custom_handler:  # 如果用户提供自定义处理方式
        return custom_handler(page)
    return IMAGE_PLACEHOLDER + "\n"  # 默认替换为[图片]

def _process_formulas(text: str, custom_handler: Optional[Callable] = None) -> str:
    """公式处理管道"""
    if custom_handler:  # 优先使用自定义处理
        return custom_handler(text)
    
    # 默认处理流程
    processed = _convert_latex(text)  # 尝试转换LaTeX
    processed = _replace_math_symbols(processed)  # 替换未识别公式
    processed = _replace_dir_markers(processed)  # 新增目录标记处理
    return processed

def _convert_latex(text: str) -> str:
    """尝试识别数学公式"""
    # 匹配简单数学表达式（如$...$格式）
    latex_pattern = r'\$(.*?)\$'
    return re.sub(latex_pattern, r'\(\1\)', text)  # 转换为LaTeX格式

def _replace_math_symbols(text: str) -> str:
    # 增强版公式检测
    if _is_math_formula(text):
        return FORMULA_PLACEHOLDER
    return text

def _is_math_formula(text: str) -> bool:
    """综合判断是否为公式"""
    # 特征1：数学符号密度
    symbol_count = len(re.findall(MATH_SYMBOLS, text))
    
    # 特征2：特殊结构模式
    patterns = [
        r'\w_{.+}',    # 下标
        r'\w^{.+}',    # 上标
        r'\\[a-zA-Z]+' # LaTeX命令
    ]
    
    # 特征3：排版特征（需解析引擎支持）
    is_centered = False  # 需要从page对象获取坐标信息
    has_math_font = False # 需要从文本属性获取字体信息
    
    return (
        (symbol_count / len(text) > 0.1) or
        any(re.search(p, text) for p in patterns) or
        (is_centered and has_math_font)
    )

def _replace_dir_markers(text: str) -> str:
    """替换目录标记为Markdown列表格式"""
    lines = []
    for line in text.split('\n'):
        for marker, (symbol, indent) in DIR_MARKER.items():
            if marker in line:
                # 替换符号并添加缩进
                line = line.replace(marker, '    ' * indent + f'- {symbol} ')
                break
        lines.append(line)
    return '\n'.join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF文本提取工具")
    parser.add_argument("-i", "--input", help="输入PDF文件或目录")
    parser.add_argument("-o", "--output", help="输出目录", default="output")
    parser.add_argument("-e", "--engine", help="使用引擎", choices=["pdfplumber", "pymupdf"], default="pymupdf")
    parser.add_argument("-im", "--image", help="处理图片", action="store_true")
    parser.add_argument("-f", "--formula", help="处理公式", action="store_true")
    args = parser.parse_args()
    extract_pdf(args.input, args.output, args.engine, args.image, args.formula)

