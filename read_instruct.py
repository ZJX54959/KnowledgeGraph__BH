import os
import json
from openai import OpenAI
import argparse

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

def read_file(data_dir, input_file):
    # 定义数据目录和输出目录
    file_path = os.path.join(data_dir, input_file)
    # 读取数据集
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

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
    args.add_argument('-s', '--split', type=int, default=4096, help='Split size, default is 4096')
    args = args.parse_args()

    client = construct_client(args.api_key)
    data = read_file('.', args.input)
    prompt = push_instruct(data, args.prompt)
    response = get_response(prompt, args.model, client, args.split)
    save_response(args.output, response)

