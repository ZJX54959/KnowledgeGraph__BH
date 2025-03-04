import os
import json
from openai import OpenAI

# 设置OpenAI客户端
# client = OpenAI(
#     base_url="https://api.openai.com/v1",
#     api_key=os.getenv("OPENAI_API_KEY"),
# )
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-0af42d408e7d4a81be9aea3d136faea9",
)

# 读取prompt_v1.txt内容
with open('prompt_v2.txt', 'r', encoding='utf-8') as f:
    prompt_template = f.read()

# 定义数据目录和输出目录
data_dir = 'huatuo26M-testdatasets'
output_file = 'filtered_cancer_data_huatuo.txt'
filename = 'test_datasets.json'
file_path = os.path.join(data_dir, filename)

# 读取数据集
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 分次处理数据
batch_size = 16  # 假设每次处理4096个token
total_data = len(data)
max_tokens_per_batch = 4096  # 设置每个批次的最大token数

for i in range(0, total_data, batch_size):
    # 为每个数据项添加id
    batch_data = [{"id": idx, **item} for idx, item in enumerate(data[i:i + batch_size], start=i)]
    
    # 估算每条数据的token数
    def estimate_tokens(text):
        return len(text) // 2  # 简单估算：假设每个token平均由4个字符组成

    # 检查并调整批次数据以满足token限制
    while True:
        batch_text = json.dumps(batch_data, ensure_ascii=False)
        total_tokens = estimate_tokens(batch_text)
        
        if total_tokens <= max_tokens_per_batch:
            break
        
        # 找到占用token数最多的数据并移除
        max_token_data = max(batch_data, key=lambda x: estimate_tokens(json.dumps(x, ensure_ascii=False)))
        batch_data.remove(max_token_data)
        print(f"剔除数据ID: {max_token_data['id']}，当前总token: {total_tokens}")

    # 构建请求的prompt
    prompt = f"{prompt_template}\n\n请开始判断以下数据:\n\n{batch_text}"

    # 调用OpenAI API
    response = client.chat.completions.create(
        # model="gpt-4o-mini",
        # model="shenzhi-wang/Llama3-8B-Chinese-Chat-GGUF-8bit", 
        model="llama3.3-70b-instruct", 
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        # max_tokens=1024,
        temperature=0
    )

    # 处理API响应
    output = response.choices[0].message.content.strip()

    # 解析输出并筛选相关数据
    with open(output_file, 'a', encoding='utf-8') as out_file:
        # out_file.write(f"文件: {filename}\n")
        out_file.write(output + "\n")
        # out_file.write("---\n")

    # 输出当前完成进度
    progress = (i + batch_size) / total_data * 100
    print(f"当前进度: {progress:.2f}%")

print("数据筛选完成，结果已保存到", output_file)
