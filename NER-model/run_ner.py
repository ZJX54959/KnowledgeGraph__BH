#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本名称：run_ner.py
说明：此脚本提供了一个简单的命令行界面，用于训练和测试BiLSTM+CRF命名实体识别模型。
依赖：bilstm_crf_ner.py
使用方法：
    python run_ner.py train  # 训练模型
    python run_ner.py predict  # 预测文本
    python run_ner.py evaluate  # 评估模型
"""

import os
import sys
import argparse
import logging
import json
from bilstm_crf_ner import main as ner_main

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model():
    """
    训练NER模型的交互式界面
    """
    print("\n===== 训练BiLSTM+CRF命名实体识别模型 =====\n")
    
    # 设置默认值
    data_path = input(f"请输入训练数据路径 (默认: ./data/ner_sample.txt): ") or "./data/ner_sample.txt"
    model_path = input(f"请输入模型保存路径 (默认: ./models/ner_model.pt): ") or "./models/ner_model.pt"
    batch_size = input(f"请输入批次大小 (默认: 32): ") or "32"
    epochs = input(f"请输入训练轮数 (默认: 10): ") or "10"
    lr = input(f"请输入学习率 (默认: 0.001): ") or "0.001"
    embedding_dim = input(f"请输入嵌入层维度 (默认: 128): ") or "128"
    hidden_dim = input(f"请输入隐藏层维度 (默认: 256): ") or "256"
    num_layers = input(f"请输入LSTM层数 (默认: 2): ") or "2"
    dropout = input(f"请输入Dropout比例 (默认: 0.1): ") or "0.1"
    split_ratio = input(f"请输入验证集比例 (默认: 0.1): ") or "0.1"
    
    # 构建命令行参数
    cmd_args = [
        "train",
        "--data_path", data_path,
        "--model_path", model_path,
        "--batch_size", batch_size,
        "--epochs", epochs,
        "--lr", lr,
        "--embedding_dim", embedding_dim,
        "--hidden_dim", hidden_dim,
        "--num_layers", num_layers,
        "--dropout", dropout,
        "--split_ratio", split_ratio
    ]
    
    # 检查数据路径是否存在
    if not os.path.exists(data_path):
        create_sample = input(f"数据路径 {data_path} 不存在，是否创建示例数据? (y/n): ").lower()
        if create_sample == 'y':
            cmd_args.append("--create_sample")
        else:
            print("训练取消。")
            return
    
    # 确保模型目录存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 调用NER模型的main函数
    sys.argv = ["bilstm_crf_ner.py"] + cmd_args
    ner_main()

def predict_text():
    """
    预测文本的交互式界面
    """
    print("\n===== 使用BiLSTM+CRF模型进行命名实体识别 =====\n")
    
    # 设置默认值
    model_path = input(f"请输入模型路径 (默认: ./models/ner_model.pt): ") or "./models/ner_model.pt"
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在。请先训练模型。")
        return
    
    # 选择预测模式
    mode = input("请选择预测模式 (1: 输入文本, 2: 输入文件路径): ")
    
    if mode == "1":
        # 文本预测模式
        while True:
            text = input("\n请输入要识别的文本 (输入'q'退出): ")
            if text.lower() == 'q':
                break
            
            if not text.strip():
                continue
            
            # 构建命令行参数
            cmd_args = ["predict", "--text", text, "--model_path", model_path]
            
            # 调用NER模型的main函数
            sys.argv = ["bilstm_crf_ner.py"] + cmd_args
            ner_main()
    
    elif mode == "2":
        # 文件预测模式
        file_path = input("请输入文本文件路径: ")
        
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在。")
            return
        
        # 构建命令行参数
        cmd_args = ["predict", "--file_path", file_path, "--model_path", model_path]
        
        # 调用NER模型的main函数
        sys.argv = ["bilstm_crf_ner.py"] + cmd_args
        ner_main()
    
    else:
        print("无效的选择。")

def evaluate_model():
    """
    评估模型的交互式界面
    """
    print("\n===== 评估BiLSTM+CRF命名实体识别模型 =====\n")
    
    # 设置默认值
    model_path = input(f"请输入模型路径 (默认: ./models/ner_model.pt): ") or "./models/ner_model.pt"
    test_path = input(f"请输入测试数据路径 (默认: ./data/ner_sample.json): ") or "./data/ner_sample.json"
    batch_size = input(f"请输入批次大小 (默认: 32): ") or "32"
    
    # 检查模型和测试数据是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在。请先训练模型。")
        return
    
    if not os.path.exists(test_path):
        print(f"错误: 测试数据 {test_path} 不存在。")
        return
    
    # 构建命令行参数 - 使用预测模式处理文件
    cmd_args = ["evaluate", "--data_path", test_path, "--model_path", model_path, "--batch_size", batch_size]
    
    # 调用NER模型的main函数
    sys.argv = ["bilstm_crf_ner.py"] + cmd_args
    ner_main()
    

def main():
    parser = argparse.ArgumentParser(description="BiLSTM+CRF命名实体识别模型交互式界面")
    parser.add_argument("action", nargs="?", choices=["train", "predict", "evaluate"], 
                        help="执行的操作: train(训练), predict(预测), evaluate(评估)")
    
    args = parser.parse_args()
    
    if args.action == "train":
        train_model()
    elif args.action == "predict":
        predict_text()
    elif args.action == "evaluate":
        evaluate_model()
    else:
        # 如果没有提供操作，显示交互式菜单
        while True:
            print("\n===== BiLSTM+CRF命名实体识别模型 =====\n")
            print("1. 训练模型")
            print("2. 预测文本")
            print("3. 评估模型")
            print("4. 退出")
            
            choice = input("\n请选择操作 (1-4): ")
            
            if choice == "1":
                train_model()
            elif choice == "2":
                predict_text()
            elif choice == "3":
                evaluate_model()
            elif choice == "4":
                print("\n感谢使用！再见！")
                break
            else:
                print("无效的选择，请重试。")

if __name__ == "__main__":
    main()