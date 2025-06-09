# 命名实体识别数据目录

此目录用于存放命名实体识别(NER)的训练数据和测试数据。

## 数据格式

支持以下两种格式：

1. **CoNLL格式**：每行一个字符和标签，句子之间用空行分隔
   ```
   北 B-ORG
   京 I-ORG
   大 I-ORG
   学 I-ORG
   位 O
   于 O
   北 B-LOC
   京 I-LOC
   市 I-LOC
   
   李 B-PER
   明 I-PER
   在 O
   清 B-ORG
   华 I-ORG
   大 I-ORG
   学 I-ORG
   ```

2. **JSON格式**：包含text和labels字段的JSON对象列表
   ```json
   [
     {
       "text": "北京大学位于北京市",
       "labels": ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O", "B-LOC", "I-LOC", "I-LOC"]
     },
     {
       "text": "李明在清华大学",
       "labels": ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG"]
     }
   ]
   ```

## 标签说明

使用BIOES标注方案：
- B-X: 实体X的开始
- I-X: 实体X的中间
- E-X: 实体X的结束
- S-X: 单个字符的实体X
- O: 非实体

## 实体类型

常见的实体类型包括：
- PER: 人名
- ORG: 组织机构名
- LOC: 地名
- TIME: 时间
- DATE: 日期
- PRO: 产品名