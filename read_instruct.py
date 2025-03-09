import os
import json
from openai import OpenAI
import argparse
import re
from pdf_extractor import extract_pdf

def construct_client(api_key):
    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
    )

    api_keys = [
        os.getenv('OPENAI_API_KEY'),
        os.getenv('DASH_SCOPE_API_KEY'),
    ]

    if not client.api_key:
        if api_keys:
            client.api_key = api_keys[1]
        else:
            raise ValueError("API key is not set")
    
    return client

def read_file(data_dir, input_file, split=4096):
    file_path = os.path.join(data_dir, input_file)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    
    if file_path.endswith('.md'):
        return split_md_content(data, split)
    else:  # 非markdown文件简单分割
        return [data[i:i+split] for i in range(0, len(data), split)]

def split_md_content(content, max_length=4096, tolerance=200):
    """
    智能分割Markdown内容，支持结构感知和容忍量控制
    优先级：标题 > 列表 > 代码块/表格 > 段落
    """
    # 结构匹配正则（扩展支持更多Markdown元素）
    pattern = re.compile(r'''
        (\n\#{1,6}\s+.*?)(?=\n\#|$)       # 标题（1-6级）
        |(\n\*{3,})                     # 分割线
        |(\n```.*?\n```)                # 代码块
        |(\n\|.*?\|)                    # 表格行
        |(\n[-*+] .*?(?:\n[ \t]+.*?)*) # 无缩进列表
        |(\n\d+\. .*?(?:\n[ \t]+.*?)*) # 有序列表
        |(\n[ \t]+[-*+] .*?)           # 缩进列表
        |(\n\n+)                        # 空行
    ''', re.DOTALL|re.VERBOSE)

    blocks = []
    last_pos = 0
    current_hierarchy = []

    for match in pattern.finditer(content):
        end = match.end()
        chunk = content[last_pos:end]
        chunk_len = len(chunk.encode('utf-8'))
        
        # 结构类型检测
        struct_type = next((i for i, g in enumerate(match.groups()) if g), None)
        priority = get_priority(struct_type, match.group())
        
        # 当前块长度检查（带容忍量）
        if chunk_len > (max_length + tolerance):
            # 在容忍范围内寻找最佳分割点
            split_pos = find_optimal_split(
                content, 
                last_pos, 
                end, 
                max_length, 
                tolerance, 
                current_hierarchy
            )
            if split_pos > last_pos:
                blocks.append(content[last_pos:split_pos])
                last_pos = split_pos
                current_hierarchy = update_hierarchy(content[last_pos:split_pos])
        
        # 维护层级结构
        if priority is not None:
            current_hierarchy = update_hierarchy(current_hierarchy, priority)

    # 处理剩余内容  
    if last_pos < len(content):
        blocks.append(content[last_pos:])
    return blocks

def get_priority(struct_type, text):
    """ 获取结构优先级 """
    if struct_type == 0:  # 标题
        level = text.count('#')
        return level - 1  # 一级标题0，二级1...六级5
    elif struct_type in (1,2,3):  # 分割线/代码块/表格
        return 6
    elif struct_type in (4,5):    # 无缩进列表
        return 7
    elif struct_type == 6:        # 缩进列表
        indent = len(text) - len(text.lstrip(' \t'))
        return 8 + indent//2      # 每级缩进+1
    return None

def find_optimal_split(content, start, end, max_len, tolerance, hierarchy):
    """ 在容忍范围内寻找最佳分割点 """
    window_start = max(start, end - max_len - tolerance)
    best_split = end
    best_priority = float('inf')
    
    # 反向扫描寻找最佳分割点
    for match in reversed(list(pattern.finditer(content, window_start, end))):
        pos = match.start()
        priority = get_priority(*match)
        
        # 层级检查（避免非正序层级）
        if hierarchy and priority < hierarchy[-1]:
            continue
            
        if priority < best_priority:
            best_priority = priority
            best_split = pos
            
    return best_split if (end - best_split) <= max_len else end

def push_instruct(data, prompt_template = 'convert.pmpt'):
    # 读取prompt内容
    with open(prompt_template, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    # 构建请求的prompt
    prompt = f"{prompt_template}\n\n请开始判断以下数据:\n\n{data}"
    return prompt

def get_response(prompt, model='llama3.3-70b-instruct', client=None, split=4096):
    # 调用OpenAI API
    response = client.chat.completions.create(
        # model="gpt-4o-mini",
        # model="shenzhi-wang/Llama3-8B-Chinese-Chat-GGUF-8bit", 
        model=model, 
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=split,
        temperature=0
    )
    return response

def save_response(output_file, response): 
    # 处理API响应
    output = response.choices[0].message.content.strip()

    with open(output_file, 'a', encoding='utf-8') as out_file:
        # out_file.write(f"文件: {filename}\n")
        out_file.write(output + "\n")
    print("数据筛选完成，结果已保存到", output_file)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input', type=str, default='KG第1讲_note.md')
    args.add_argument('-o', '--output', type=str, default='output.md')
    args.add_argument('-p', '--prompt', type=str, default='convert.pmpt')
    args.add_argument('-m', '--model', type=str, default='llama3.3-70b-instruct')
    args.add_argument('-a', '--api_key', type=str, default=os.getenv('DASH_SCOPE_API_KEY'), help='Custom API key, default is DASH_SCOPE_API_KEY0')
    args.add_argument('-s', '--split', type=int, default=65536, help='Split size, default is 65536')
    args = args.parse_args()

    client = construct_client(args.api_key)
    data_chunks = read_file('.', args.input, args.split)  # 获取分块数据
    
    full_output = []
    for chunk in data_chunks:
        prompt = push_instruct(chunk, args.prompt)
        response = get_response(prompt, args.model, client, args.split)
        full_output.append(response.choices[0].message.content.strip())
    
    # 合并结果并保存
    with open(args.output, 'w', encoding='utf-8') as out_file:
        out_file.write('\n\n'.join(full_output))
    print("数据筛选完成，结果已保存到", args.output)

