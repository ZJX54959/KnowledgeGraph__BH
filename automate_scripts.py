import subprocess
import os
import argparse
import re

processed_files = [
    "第2讲-知识表示-2025春季.pdf",
    "第1讲-知识图谱概述-2025春季.pdf",
]

def run_script(command, verbose=False):
    """运行指定命令并打印进度"""
    if verbose:
        print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)
    if verbose:
        print("Command completed.")

if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', '--input', type=str, help='input file directory', default="D:\\新建文件夹\\知识图谱")
    psr.add_argument('-o', '--output', type=str, help='output dir', default='output')
    psr.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
    psr.add_argument('-re', '--regex', type=str, help='regex pattern to match input files', default='.*')
    psr.add_argument('-r', '--recursive', action='store_true', help='recursive mode')
    args = psr.parse_args()

    # 获取输入目录中的文件列表
    input_files = [f for f in os.listdir(args.input) if re.match(args.regex, f) and f not in processed_files]

    for file in input_files:
        base_name = os.path.splitext(file)[0]
        
        # 自动化运行所有命令
        commands = [
            ['python', 'd:\\Documents\\Coding\\KnowledgeGraph\\pdf_extractor.py', '-i', os.path.join(args.input, file), '-o', os.path.join(args.output, 'pdf_ext')],
            ['python', 'd:\\Documents\\Coding\\KnowledgeGraph\\read_instruct.py', '-i', os.path.join(args.output, 'pdf_ext', f'{base_name}.txt'), '-s', '4096', '-o', os.path.join(args.output, 'lectures_ext', base_name + '.md')],
            ['python', 'd:\\Documents\\Coding\\KnowledgeGraph\\md2csv.py', '-i', os.path.join(args.output, 'lectures_ext', base_name + '.md')]
        ]

        for command in commands:
            run_script(command, args.verbose)