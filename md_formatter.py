import re
from pathlib import Path
from typing import Union, Callable, Optional
import logging
import argparse

# 保持与PDF提取脚本一致的日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认替换规则
DEFAULT_IMAGE_PATTERNS = [
    r'!\[.*?\]\(.*?\)',        # 标准Markdown图片
    r'<img.*?src=".*?".*?>',   # HTML图片标签
    r'\\begin{figure}.*?\\end{figure}'  # LaTeX图片环境
]

DEFAULT_FORMULA_PATTERNS = [
    r'\$\$.*?\$\$',            # 块公式
    r'\$.*?\$',                # 行内公式
    r'\\begin{equation}.*?\\end{equation}',
    r'\\\[.*?\\\]'             # LaTeX块公式
]

def process_md(
    input_path: Union[str, Path],
    output_dir: Union[str, Path] = None,
    image_handler: Optional[Callable] = None,
    formula_handler: Optional[Callable] = None
) -> None:
    """
    处理Markdown文件中的图片和公式
    
    参数：
    input_path: MD文件路径或包含MD的目录路径
    output_dir: 输出目录（None则覆盖原文件）
    image_handler: 自定义图片处理函数
    formula_handler: 自定义公式处理函数
    """
    # ... 与PDF提取脚本类似的目录处理逻辑 ...
    input_path = Path(input_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        _process_single_md(input_path, output_dir, image_handler, formula_handler)
    elif input_path.is_dir():
        for md_file in input_path.glob("**/*.md"):
            try:
                _process_single_md(md_file, output_dir, image_handler, formula_handler)
            except Exception as e:
                logger.error(f"处理文件失败 {md_file}: {str(e)}")
                continue
    else:
        raise FileNotFoundError(f"路径不存在: {input_path}")

def _process_single_md(file_path: Path, output_dir: Optional[Path], image_handler, formula_handler) -> None:
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # 处理图片
        processed = _replace_patterns(
            content,
            patterns=DEFAULT_IMAGE_PATTERNS,
            placeholder="[图片]",
            custom_handler=image_handler
        )
        
        # 处理公式
        processed = _replace_patterns(
            processed,
            patterns=DEFAULT_FORMULA_PATTERNS,
            placeholder="[公式]",
            custom_handler=formula_handler
        )
        
        # 保存文件
        if output_dir:
            output_path = output_dir / file_path.name
        else:
            output_path = file_path
            
        output_path.write_text(processed, encoding='utf-8')
        logger.info(f"文件已处理: {output_path}")
        
    except Exception as e:
        logger.error(f"处理失败 {file_path.name}: {str(e)}")
        raise

def _replace_patterns(
    text: str,
    patterns: list,
    placeholder: str,
    custom_handler: Optional[Callable] = None
) -> str:
    """通用替换逻辑"""
    def _default_replacer(match):
        return f"\n{placeholder}\n" if match.group().count('\n') > 1 else placeholder
    
    handler = custom_handler or _default_replacer
    
    combined_pattern = '|'.join(f'({p})' for p in patterns)
    return re.sub(
        combined_pattern,
        lambda m: handler(m),
        text,
        flags=re.DOTALL
    )

if __name__ == "__main__":
    # 保持与PDF提取脚本一致的命令行参数风格
    parser = argparse.ArgumentParser(description="Markdown内容格式化工具")
    parser.add_argument("-i", "--input", required=True, help="输入MD文件或目录")
    parser.add_argument("-o", "--output", help="输出目录（默认覆盖原文件）")
    args = parser.parse_args()
    
    process_md(
        input_path=args.input,
        output_dir=args.output
    ) 