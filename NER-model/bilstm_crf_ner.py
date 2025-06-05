#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本名称：bilstm_crf_ner.py
说明：本脚本实现了基于BiLSTM+CRF的命名实体识别模型，用于从文本中抽取实体，
     可以作为知识图谱构建的前置步骤。该模型使用双向LSTM进行特征提取，
     并使用条件随机场(CRF)进行序列标注，以识别文本中的命名实体。
依赖：torch, numpy, sklearn, tqdm
使用方法：
    python bilstm_crf_ner.py train --data_path <训练数据路径> --model_path <模型保存路径>
    python bilstm_crf_ner.py predict --text "待识别的文本" --model_path <模型路径>
例如：
    python bilstm_crf_ner.py train --data_path ./data/ner_data.txt --model_path ./models/ner_model.pt
    python bilstm_crf_ner.py predict --text "北京大学位于北京市海淀区" --model_path ./models/ner_model.pt
"""

from this import d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
import json
import re
from tqdm import tqdm
import pickle
from sklearn.metrics import precision_recall_fscore_support
import logging
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed()

# 定义标签映射（BIOES标注方案）
# B-X: 实体X的开始
# I-X: 实体X的中间
# E-X: 实体X的结束
# S-X: 单个字符的实体X
# O: 非实体
class LabelVocab:
    def __init__(self):
        self.label2id = {'O': 0}
        self.id2label = {0: 'O'}
        self.num_labels = 1
    
    def add_label(self, label):
        if label not in self.label2id:
            self.label2id[label] = self.num_labels
            self.id2label[self.num_labels] = label
            self.num_labels += 1
    
    def convert_label_to_id(self, label):
        return self.label2id.get(label, 0)  # 默认返回O的ID
    
    def convert_id_to_label(self, idx):
        return self.id2label.get(idx, 'O')  # 默认返回O
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'label2id': self.label2id,
                'id2label': self.id2label,
                'num_labels': self.num_labels
            }, f)
    
    def load(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.label2id = data['label2id']
            self.id2label = data['id2label']
            self.num_labels = data['num_labels']

# 词汇表类，用于将文本转换为ID
class Vocab:
    def __init__(self, min_freq=1, special_tokens=['[PAD]', '[UNK]']):
        self.word2id = {}
        self.id2word = {}
        self.word_freq = {}
        self.min_freq = min_freq
        
        # 添加特殊标记
        for token in special_tokens:
            self.add_word(token)
        
        self.pad_token_id = self.word2id['[PAD]']
        self.unk_token_id = self.word2id['[UNK]']
    
    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = len(self.word2id)
            self.id2word[len(self.id2word)] = word
            self.word_freq[word] = 1
        else:
            self.word_freq[word] += 1
    
    def build_vocab(self, texts):
        # 统计词频
        for text in texts:
            for char in text:
                self.add_word(char)
        
        # 根据最小频率过滤词汇
        self.word2id = {'[PAD]': 0, '[UNK]': 1}
        self.id2word = {0: '[PAD]', 1: '[UNK]'}
        idx = 2
        
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in ['[PAD]', '[UNK]']:
                self.word2id[word] = idx
                self.id2word[idx] = word
                idx += 1
    
    def convert_word_to_id(self, word):
        return self.word2id.get(word, self.unk_token_id)
    
    def convert_id_to_word(self, idx):
        return self.id2word.get(idx, '[UNK]')
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'word2id': self.word2id,
                'id2word': self.id2word,
                'pad_token_id': self.pad_token_id,
                'unk_token_id': self.unk_token_id
            }, f)
    
    def load(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.word2id = data['word2id']
            self.id2word = data['id2word']
            self.pad_token_id = data['pad_token_id']
            self.unk_token_id = data['unk_token_id']

# 数据集类
class NERDataset(Dataset):
    def __init__(self, texts, labels, vocab, label_vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label_vocab = label_vocab
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 将文本和标签转换为ID
        text_ids = [self.vocab.convert_word_to_id(char) for char in text]
        label_ids = [self.label_vocab.convert_label_to_id(l) for l in label]
        
        return {
            'text_ids': torch.tensor(text_ids),
            'label_ids': torch.tensor(label_ids),
            'text': text,
            'label': label
        }

# 数据处理函数
def collate_fn(batch):
    max_len = max([len(item['text_ids']) for item in batch])
    
    text_ids = []
    label_ids = []
    attention_mask = []
    texts = []
    labels = []
    
    for item in batch:
        text_id = item['text_ids']
        label_id = item['label_ids']
        padding_len = max_len - len(text_id)
        
        # 填充
        text_id = torch.cat([text_id, torch.zeros(padding_len, dtype=torch.long)])
        label_id = torch.cat([label_id, torch.zeros(padding_len, dtype=torch.long)])
        mask = torch.cat([torch.ones(len(item['text_ids']), dtype=torch.long), torch.zeros(padding_len, dtype=torch.long)])
        
        text_ids.append(text_id)
        label_ids.append(label_id)
        attention_mask.append(mask)
        texts.append(item['text'])
        labels.append(item['label'])
    
    return {
        'text_ids': torch.stack(text_ids),
        'label_ids': torch.stack(label_ids),
        'attention_mask': torch.stack(attention_mask),
        'texts': texts,
        'labels': labels
    }

# 加载数据函数
def load_data(file_path):
    """
    加载NER数据，支持多种格式：
    1. CoNLL格式：每行一个字符和标签，句子之间用空行分隔
    2. JSON格式：包含text和labels字段的JSON对象列表
    """
    texts = []
    labels = []
    
    # 根据文件扩展名判断格式
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                texts.append(item['text'])
                labels.append(item['labels'])
    else:  # 默认为CoNLL格式
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            text = []
            label = []
            
            for line in lines:
                line = line.strip()
                if line == '':
                    if text:
                        texts.append(''.join(text))
                        labels.append(label)
                        text = []
                        label = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        text.append(parts[0])
                        label.append(parts[-1])
            
            if text:  # 处理最后一个句子
                texts.append(''.join(text))
                labels.append(label)
    
    return texts, labels

# 创建示例数据
def create_sample_data(output_path):
    """
    创建示例NER数据用于测试
    """
    sample_data = [
        {
            "text": "北京大学位于北京市海淀区",
            "labels": ["B-ORG", "I-ORG", "I-ORG", "O", "O", "B-LOC", "I-LOC", "I-LOC", "B-LOC", "I-LOC", "I-LOC"]
        },
        {
            "text": "李明在清华大学读书",
            "labels": ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-ORG", "O", "O"]
        },
        {
            "text": "今天上海的天气真不错",
            "labels": ["O", "O", "B-LOC", "I-LOC", "O", "O", "O", "O", "O", "O"]
        },
        {
            "text": "苹果公司发布了新款iPhone手机",
            "labels": ["B-ORG", "I-ORG", "I-ORG", "O", "O", "O", "O", "B-PRO", "I-PRO", "I-PRO", "I-PRO", "I-PRO", "I-PRO"]
        }
    ]
    
    # 转换为CoNLL格式
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            text = item["text"]
            labels = item["labels"]
            for char, label in zip(text, labels):
                f.write(f"{char} {label}\n")
            f.write("\n")
    
    logger.info(f"已创建示例数据：{output_path}")

def txt_to_json(txt_file_path, json_file_path):
    """
    将CoNLL格式的txt文件转换为JSON格式
    """
    data = []
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        text = []
        labels = []
        
        for line in lines:
            line = line.strip()
            if line == '':
                if text:
                    data.append({
                        "text": ''.join(text),
                        "labels": labels
                    })
                    text = []
                    labels = []
            else:
                parts = line.split()
                if len(parts) == 2:
                    text.append(parts[0])
                    labels.append(parts[1])
        
        if text:  # 处理最后一个句子
            data.append({
                "text": ''.join(text),
                "labels": labels
            })
    
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return data

# BiLSTM-CRF模型定义
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_labels, dropout=0.1):
        super(BiLSTM_CRF, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        
        # 字符嵌入层
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2, 
            num_layers=num_layers, 
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 线性层，将LSTM的输出映射到标签空间
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
        
        # dropout层
        self.dropout = nn.Dropout(dropout)
        
        # CRF层的转移矩阵
        # transitions[i, j]表示从标签j转移到标签i的分数
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        
        # 特殊的开始和结束标签的转移分数
        self.start_transitions = nn.Parameter(torch.randn(num_labels))
        self.end_transitions = nn.Parameter(torch.randn(num_labels))
    
    def _forward_alg(self, feats, mask):
        """
        计算所有可能路径的分数总和（对数空间）
        """
        batch_size, seq_len, num_labels = feats.size()
        
        # 初始化alpha，包含开始标签的转移分数
        alpha = self.start_transitions.view(1, num_labels) + feats[:, 0]
        
        for i in range(1, seq_len):
            emit_score = feats[:, i].view(batch_size, 1, num_labels)
            trans_score = self.transitions.view(1, num_labels, num_labels)
            
            # 前一步的alpha + 转移分数 + 发射分数
            alpha_t = alpha.view(batch_size, num_labels, 1) + trans_score + emit_score
            
            # 对所有可能的前一个标签求log_sum_exp
            mask_i = mask[:, i].view(batch_size, 1)
            alpha_next = torch.logsumexp(alpha_t, dim=1) * mask_i + alpha * (1 - mask_i)
            
            alpha = alpha_next
        
        # 加上结束标签的转移分数
        alpha = alpha + self.end_transitions.view(1, num_labels)
        
        # 对所有可能的结束标签求log_sum_exp
        return torch.logsumexp(alpha, dim=1)
    
    def _score_sentence(self, feats, tags, mask):
        """
        计算给定标签序列的分数
        """
        batch_size, seq_len, num_labels = feats.size()
        
        # 初始分数：开始标签的转移分数 + 第一个标签的发射分数
        score = self.start_transitions[tags[:, 0]]
        score += feats[torch.arange(batch_size), 0, tags[:, 0]]
        
        for i in range(1, seq_len):
            # 只在mask=1的位置计算分数
            mask_i = mask[:, i]
            
            # 转移分数：从前一个标签到当前标签
            score += self.transitions[tags[:, i], tags[:, i-1]] * mask_i
            
            # 发射分数：当前位置的特征生成当前标签的分数
            score += feats[torch.arange(batch_size), i, tags[:, i]] * mask_i
        
        # 找到序列的最后一个有效位置
        seq_ends = mask.sum(dim=1).long() - 1
        
        # 加上从最后一个标签到结束标签的转移分数
        last_tags = tags[torch.arange(batch_size), seq_ends]
        score += self.end_transitions[last_tags]
        
        return score
    
    def _viterbi_decode(self, feats, mask):
        """
        使用维特比算法解码最佳标签序列
        """
        batch_size, seq_len, num_labels = feats.size()
        
        # 初始化viterbi变量和反向指针
        viterbi = self.start_transitions.view(1, num_labels) + feats[:, 0]
        backpointers = torch.zeros(batch_size, seq_len, num_labels, dtype=torch.long, device=feats.device)
        
        for i in range(1, seq_len):
            # 前一步的viterbi分数 + 转移分数 + 当前发射分数
            next_tag_var = viterbi.view(batch_size, num_labels, 1) + self.transitions.view(1, num_labels, num_labels)
            best_tag_id = next_tag_var.argmax(dim=1)
            backpointers[:, i] = best_tag_id
            
            # 更新viterbi分数
            viterbi_scores = next_tag_var.gather(1, best_tag_id.unsqueeze(1)).squeeze(1)
            viterbi_scores += feats[:, i]
            
            # 只在mask=1的位置更新viterbi
            mask_i = mask[:, i].view(batch_size, 1)
            viterbi = viterbi_scores * mask_i + viterbi * (1 - mask_i)
        
        # 转移到结束标签
        viterbi += self.end_transitions.view(1, num_labels)
        
        # 找到最佳结束标签
        best_tag_id = viterbi.argmax(dim=1)
        best_path_scores = viterbi.gather(1, best_tag_id.unsqueeze(1)).squeeze(1)
        
        # 回溯找到最佳路径
        best_paths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=feats.device)
        best_paths[:, -1] = best_tag_id
        
        for i in range(seq_len - 1, 0, -1):
            best_tag_id = backpointers[:, i].gather(1, best_tag_id.unsqueeze(1)).squeeze(1)
            best_paths[:, i-1] = best_tag_id
        
        return best_path_scores, best_paths
    
    def neg_log_likelihood(self, emissions, tags, mask=None):
        """
        计算负对数似然损失
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.float)
        
        # 前向算法计算所有路径的分数
        forward_score = self._forward_alg(emissions, mask)
        
        # 计算真实路径的分数
        gold_score = self._score_sentence(emissions, tags, mask)
        
        # 损失 = 所有路径分数 - 真实路径分数
        return (forward_score - gold_score).mean()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        前向传播
        """
        # 获取字符嵌入
        embeds = self.word_embeds(input_ids)
        embeds = self.dropout(embeds)
        
        # BiLSTM层
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        
        # 线性层映射到标签空间
        emissions = self.hidden2tag(lstm_out)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.float)
        
        if labels is not None:
            # 训练模式，计算损失
            loss = self.neg_log_likelihood(emissions, labels, attention_mask)
            # 解码最佳路径
            _, tag_seq = self._viterbi_decode(emissions, attention_mask)
            return loss, tag_seq
        else:
            # 预测模式，只解码
            _, tag_seq = self._viterbi_decode(emissions, attention_mask)
            return tag_seq

# 训练函数
def train(model, train_dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        # 将数据移到设备上
        input_ids = batch['text_ids'].to(device)
        label_ids = batch['label_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播
        loss, _ = model(input_ids, attention_mask, label_ids)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(train_dataloader)

# 评估函数
def evaluate(model, eval_dataloader, label_vocab, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # 将数据移到设备上
            input_ids = batch['text_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 前向传播
            tag_seq = model(input_ids, attention_mask)
            
            # 将预测结果和真实标签转换为列表
            for i, (pred, true, mask) in enumerate(zip(tag_seq, label_ids, attention_mask)):
                length = mask.sum().item()
                pred = pred[:length].cpu().numpy()
                true = true[:length].cpu().numpy()
                
                # 将ID转换为标签
                pred_labels = [label_vocab.convert_id_to_label(p) for p in pred]
                true_labels = [label_vocab.convert_id_to_label(t) for t in true]
                
                all_predictions.extend(pred_labels)
                all_labels.extend(true_labels)
    
    # 计算评估指标
    labels = list(set([label for label in label_vocab.label2id.keys() if label != 'O']))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, labels=labels, average='weighted'
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 预测函数
def predict(model, text, vocab, label_vocab, device):
    model.eval()
    
    # 将文本转换为ID
    text_ids = [vocab.convert_word_to_id(char) for char in text]
    input_ids = torch.tensor([text_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(device)
    
    # 预测
    with torch.no_grad():
        tag_seq = model(input_ids, attention_mask)
    
    # 将预测结果转换为标签
    pred_labels = [label_vocab.convert_id_to_label(p) for p in tag_seq[0].cpu().numpy()]
    
    # 提取实体
    entities = []
    entity = ""
    entity_type = ""
    
    for i, (char, label) in enumerate(zip(text, pred_labels)):
        if label.startswith('B-'):
            if entity:
                entities.append((entity, entity_type))
            entity = char
            entity_type = label[2:]
        elif label.startswith('I-') and entity and entity_type == label[2:]:
            entity += char
        elif label.startswith('E-') and entity and entity_type == label[2:]:
            entity += char
            entities.append((entity, entity_type))
            entity = ""
            entity_type = ""
        elif label.startswith('S-'):
            if entity:
                entities.append((entity, entity_type))
            entities.append((char, label[2:]))
            entity = ""
            entity_type = ""
        elif label == 'O':
            if entity:
                entities.append((entity, entity_type))
                entity = ""
                entity_type = ""
    
    # 处理最后一个实体
    if entity:
        entities.append((entity, entity_type))
    
    return entities, pred_labels

# 保存模型
def save_model(model, vocab, label_vocab, model_path):
    # 创建目录
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), model_path)
    
    # 保存词汇表和标签映射
    vocab_path = os.path.join(os.path.dirname(model_path), 'vocab.pkl')
    label_vocab_path = os.path.join(os.path.dirname(model_path), 'label_vocab.pkl')
    
    vocab.save(vocab_path)
    label_vocab.save(label_vocab_path)
    
    logger.info(f"模型已保存到：{model_path}")

# 加载模型
def load_model(model_path, device):
    # 获取模型目录
    model_dir = os.path.dirname(model_path)
    
    # 加载词汇表和标签映射
    vocab = Vocab()
    label_vocab = LabelVocab()
    
    vocab_path = os.path.join(model_dir, 'vocab.pkl')
    label_vocab_path = os.path.join(model_dir, 'label_vocab.pkl')
    
    vocab.load(vocab_path)
    label_vocab.load(label_vocab_path)
    
    # 创建模型
    model = BiLSTM_CRF(
        vocab_size=len(vocab.word2id),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_labels=label_vocab.num_labels,
        dropout=0.1
    )
    
    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    return model, vocab, label_vocab

# 主函数
def main():
    parser = argparse.ArgumentParser(description="BiLSTM-CRF命名实体识别模型")
    subparsers = parser.add_subparsers(dest="mode", help="运行模式")
    
    # 训练模式参数
    train_parser = subparsers.add_parser("train", help="训练模式")
    train_parser.add_argument("--data_path", required=True, help="训练数据路径")
    train_parser.add_argument("--model_path", default="./models/ner_model.pt", help="模型保存路径")
    train_parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    train_parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    train_parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    train_parser.add_argument("--embedding_dim", type=int, default=128, help="嵌入层维度")
    train_parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    train_parser.add_argument("--num_layers", type=int, default=2, help="LSTM层数")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout比例")
    train_parser.add_argument("--split_ratio", type=float, default=0.1, help="验证集比例")
    train_parser.add_argument("--create_sample", action="store_true", help="是否创建示例数据")
    
    # 预测模式参数
    predict_parser = subparsers.add_parser("predict", help="预测模式")
    predict_parser.add_argument("--text", help="待识别的文本")
    predict_parser.add_argument("--model_path", required=True, help="模型路径")
    predict_parser.add_argument("--file_path", help="待识别的文本文件路径")

    evaluate_parser = subparsers.add_parser("evaluate", help="评估模式")
    evaluate_parser.add_argument("--data_path", required=True, help="评估数据路径")
    evaluate_parser.add_argument("--model_path", required=True, help="模型路径")
    evaluate_parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 根据模式执行相应操作
    if args.mode == "train":
        # 创建示例数据（如果需要）
        if args.create_sample:
            os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
            create_sample_data(args.data_path)
        
        # 加载数据
        logger.info(f"加载训练数据: {args.data_path}")
        texts, labels = load_data(args.data_path)
        logger.info(f"加载了 {len(texts)} 条训练数据")
        
        # 构建词汇表和标签映射
        vocab = Vocab(min_freq=1)
        vocab.build_vocab(texts)
        logger.info(f"词汇表大小: {len(vocab.word2id)}")
        
        label_vocab = LabelVocab()
        for label_seq in labels:
            for label in label_seq:
                label_vocab.add_label(label)
        logger.info(f"标签集大小: {label_vocab.num_labels}")
        logger.info(f"标签集: {list(label_vocab.label2id.keys())}")
        
        # 划分训练集和验证集
        indices = list(range(len(texts)))
        np.random.shuffle(indices)
        split = int(len(indices) * args.split_ratio)
        train_indices = indices[split:]
        val_indices = indices[:split]
        
        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_texts = [texts[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        
        logger.info(f"训练集大小: {len(train_texts)}, 验证集大小: {len(val_texts)}")
        
        # 创建数据集和数据加载器
        train_dataset = NERDataset(train_texts, train_labels, vocab, label_vocab)
        val_dataset = NERDataset(val_texts, val_labels, vocab, label_vocab)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # 创建模型
        model = BiLSTM_CRF(
            vocab_size=len(vocab.word2id),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_labels=label_vocab.num_labels,
            dropout=args.dropout
        )
        model.to(device)
        
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # 训练模型
        logger.info("开始训练...")
        best_f1 = 0.0
        for epoch in range(1, args.epochs + 1):
            # 训练
            train_loss = train(model, train_dataloader, optimizer, device, epoch)
            logger.info(f"Epoch {epoch}/{args.epochs}, 训练损失: {train_loss:.4f}")
            
            # 评估
            metrics = evaluate(model, val_dataloader, label_vocab, device)
            logger.info(f"Epoch {epoch}/{args.epochs}, 验证集评估结果:")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1: {metrics['f1']:.4f}")
            
            # 保存最佳模型
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                save_model(model, vocab, label_vocab, args.model_path)
                logger.info(f"保存最佳模型，F1: {best_f1:.4f}")
        
        logger.info("训练完成!")
    
    elif args.mode == "predict":
        # 加载模型
        logger.info(f"加载模型: {args.model_path}")
        model, vocab, label_vocab = load_model(args.model_path, device)
        
        if args.text:
            # 预测单个文本
            entities, labels = predict(model, args.text, vocab, label_vocab, device)
            
            # 输出结果
            logger.info("识别结果:")
            for entity, entity_type in entities:
                logger.info(f"  实体: {entity}, 类型: {entity_type}")
            
            # 可视化标注
            visual_result = ""
            for char, label in zip(args.text, labels):
                if label != 'O':
                    visual_result += f"[{char}:{label}] "
                else:
                    visual_result += char
            logger.info(f"标注结果: {visual_result}")
        
        elif args.file_path:
            # 预测文件中的文本
            with open(args.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            results = []
            for line in tqdm(lines, desc="处理文件"):
                line = line.strip()
                if line:
                    entities, _ = predict(model, line, vocab, label_vocab, device)
                    results.append({
                        "text": line,
                        "entities": [{
                            "entity": entity,
                            "type": entity_type
                        } for entity, entity_type in entities]
                    })
            
            # 保存结果
            output_path = os.path.splitext(args.file_path)[0] + "_entities.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"处理完成，结果已保存到: {output_path}")
        
        else:
            logger.error("请提供待识别的文本或文件路径")
    
    elif args.mode == "evaluate":
        # 评估模型
        logger.info(f"加载模型: {args.model_path}")
        model, vocab, label_vocab = load_model(args.model_path, device)

        # 加载测试数据
        logger.info(f"加载测试数据: {args.data_path}")
        texts, labels = load_data(args.data_path)
        logger.info(f"加载了 {len(texts)} 条测试数据")

        # 创建数据集和数据加载器
        test_dataset = NERDataset(texts, labels, vocab, label_vocab)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False, 
            collate_fn=collate_fn
        )
        # 评估模型
        metrics = evaluate(model, test_dataloader, label_vocab, device)
        logger.info("测试集评估结果:")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")

        # 输出评估结果的位置
        # output_path = os.path.splitext(args.data_path)[0] + "_entities.json"
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     json.dump(metrics, f, ensure_ascii=False, indent=2)
        # print(f"\n评估结果已保存到: {output_path}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()