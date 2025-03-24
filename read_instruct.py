import os
import json
from openai import OpenAI
import argparse
import re
from pdf_extractor import extract_pdf

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
    

    blocks = []
    last_pos = 0
    current_hierarchy = []

    for match in pattern.finditer(content):
        end = match.end()
        chunk = content[last_pos:end]
        chunk_len = len(chunk.encode('utf-8'))
        
        # 结构类型检测
        struct_type = next((i for i, g in enumerate(match.groups()) if g), None)
        priority = get_priority(match)
        
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
                current_hierarchy = update_hierarchy(current_hierarchy, priority)
        
        # 维护层级结构
        if priority is not None:
            current_hierarchy = update_hierarchy(current_hierarchy, priority)

    # 处理剩余内容  
    if last_pos < len(content):
        blocks.append(content[last_pos:])
    return blocks

def get_priority(match):
    if not match:
        return -1
        
    # 根据匹配组获取具体内容
    matched_text = match.group(0)
    
    # 定义优先级规则
    if re.match(r'^#{1,6}\s', matched_text):  # 标题
        return 3
    elif re.match(r'^[*-]\s', matched_text):  # 列表项
        return 2
    elif re.match(r'^\n{2,}', matched_text):  # 空行
        return 1
    else:
        return 0

def find_optimal_split(content, start, end, max_len, tolerance, hierarchy):
    """ 在容忍范围内寻找最佳分割点 """
    window_start = max(start, end - max_len - tolerance)
    best_split = end
    best_priority = float('inf')
    
    # 反向扫描寻找最佳分割点
    for match in reversed(list(pattern.finditer(content, window_start, end))):
        pos = match.start()
        priority = get_priority(match)
        
        # 层级检查（避免非正序层级）
        if hierarchy and priority < hierarchy[-1]:
            continue
            
        if priority < best_priority:
            best_priority = priority
            best_split = pos
            
    return best_split if (end - best_split) <= max_len else end

def get_response(prompt, model='llama3.3-70b-instruct', client=None, split=4096, context_aware=False, history=[]):
    """ 修改后的请求函数，支持上下文保持 """
    messages = []
    
    if context_aware and history:
        # 携带历史对话上下文
        messages = history.copy()
    else:
        # 传统模式：单次提示
        messages = [{
            "role": "user",
            "content": prompt
        }]
    
    response = client.chat.completions.create(
        model=model, 
        messages=messages,
        max_tokens=split,
        temperature=0.1
    )
    return response

def push_instruct(data, prompt_template='convert.pmpt', context_mode=False):
    """ 修改后的指令生成函数 """
    with open(prompt_template, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    if context_mode:
        # 上下文模式：返回系统提示和当前数据分离
        return {
            "system": system_prompt,
            "user_data": data
        }
    else:
        # 传统模式：拼接提示词
        return f"{system_prompt}\n\n请开始判断以下数据:\n\n{data}"

def save_response(output_file, response): 
    # 处理API响应
    output = response.choices[0].message.content.strip()

    with open(output_file, 'a', encoding='utf-8') as out_file:
        # out_file.write(f"文件: {filename}\n")
        out_file.write(output + "\n")
    print("数据筛选完成，结果已保存到", output_file)

def update_hierarchy(current_hierarchy, new_priority=None):
    """ 更新标题层级结构 
    参数：
        current_hierarchy: 当前层级栈（如 [0,1] 表示二级标题下的三级标题）
        new_priority: 新遇到元素的优先级（来自get_priority的返回值）
    返回：
        更新后的层级栈
    """
    if new_priority is None:
        return []  # 重置层级
    
    # 标题类型处理（0-5是标题优先级）
    if new_priority <= 5:
        # 根据标题级别调整层级栈
        target_level = new_priority
        
        # 找到最近的父级或同级标题
        while current_hierarchy:
            if target_level <= current_hierarchy[-1]:
                current_hierarchy.pop()
            else:
                break
                
        # 添加当前层级（确保不超过标题级别）
        if not current_hierarchy or target_level == current_hierarchy[-1] + 1:
            current_hierarchy.append(target_level)
            
    return current_hierarchy.copy()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input', type=str, default='KG第1讲_note.md')
    args.add_argument('-o', '--output', type=str, default='output.md')
    args.add_argument('-p', '--prompt', type=str, default='convert.pmpt')
    args.add_argument('-m', '--model', type=str, default='qwen-turbo')
    args.add_argument('-a', '--api_key', type=str, default=os.getenv('DASH_SCOPE_API_KEY'), help='Custom API key, default is DASH_SCOPE_API_KEY0')
    args.add_argument('-s', '--split', type=int, default=65536, help='Split size, default is 65536')
    args.add_argument('-c', '--context_aware', action='store_true', help='Enable context-aware processing')
    args = args.parse_args()

    client = construct_client(args.api_key)
    data_chunks = read_file('.', args.input, args.split)
    
    full_output = []
    chat_history = []  # 新增：维护对话历史
    
    # 初始化系统提示
    if args.context_aware:
        system_prompt = push_instruct("", args.prompt, context_mode=True)["system"]
        chat_history.append({"role": "system", "content": system_prompt})

    for chunk in data_chunks:
        if args.context_aware:
            # 上下文模式：分离系统提示和用户数据
            prompt_data = push_instruct(chunk, args.prompt, context_mode=True)
            chat_history.append({"role": "user", "content": prompt_data["user_data"]})
        else:
            # 传统模式
            prompt = push_instruct(chunk, args.prompt)
        
        response = get_response(
            prompt=prompt if not args.context_aware else None,
            model=args.model,
            client=client,
            split=args.split,
            context_aware=args.context_aware,
            history=chat_history
        )
        
        # 保存回复并维护历史
        reply = response.choices[0].message.content.strip()
        full_output.append(reply)
        
        if args.context_aware:
            chat_history.append({"role": "assistant", "content": reply})
            # 保持历史长度（防止token超限）
            if len(chat_history) > 5:  # 保留最近5轮对话
                chat_history = [chat_history[0]] + chat_history[-4:]

        output_dir = os.path.dirname(args.output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 合并结果并保存
        with open(args.output, 'w', encoding='utf-8') as out_file:
            out_file.write('\n\n'.join(full_output))
        
        # 打印进度
        print(f"[{len(full_output)/len(data_chunks) * 100}%] Already processed {len(full_output) * args.split} tokens")
    print("数据筛选完成，结果已保存到", args.output)

