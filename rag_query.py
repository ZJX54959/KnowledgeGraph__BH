#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import ast

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGQuery:
    def __init__(self, concepts_embedding_path: Path = None, relations_embedding_path: Path = None, 
                 concepts_data_path: Path = None, relations_data_path: Path = None, 
                 input_dir: Path = None,
                 model_name: str = 'paraphrase-MiniLM-L6-v2'):
        
        self.model = SentenceTransformer(model_name)

        self.concepts_embeddings = pd.DataFrame()
        self.relations_embeddings = pd.DataFrame()
        self.concepts_data = pd.DataFrame()
        self.relations_data = pd.DataFrame()

        if input_dir:
            # 批量加载概念Embedding文件
            concept_embedding_files = list(input_dir.glob('*_concepts_embeddings.csv'))
            if concept_embedding_files:
                logger.info(f"从 {input_dir} 发现 {len(concept_embedding_files)} 个概念Embedding文件。")
                self.concepts_embeddings = pd.concat([self._load_embeddings_single(f, 'concept_name') for f in concept_embedding_files], ignore_index=True)
            else:
                logger.warning(f"在目录 {input_dir} 中未找到任何概念Embedding文件。")

            # 批量加载关系Embedding文件
            relation_embedding_files = list(input_dir.glob('*_relations_embeddings.csv'))
            if relation_embedding_files:
                logger.info(f"从 {input_dir} 发现 {len(relation_embedding_files)} 个关系Embedding文件。")
                self.relations_embeddings = pd.concat([self._load_embeddings_single(f, 'relation_phrase') for f in relation_embedding_files], ignore_index=True)
            else:
                logger.warning(f"在目录 {input_dir} 中未找到任何关系Embedding文件。")

            # 批量加载概念数据文件
            concept_data_files = list(input_dir.glob('*_concepts.csv'))
            if concept_data_files:
                logger.info(f"从 {input_dir} 发现 {len(concept_data_files)} 个概念数据文件。")
                self.concepts_data = pd.concat([self._load_data_single(f) for f in concept_data_files], ignore_index=True)
            else:
                logger.warning(f"在目录 {input_dir} 中未找到任何概念数据文件。")

            # 批量加载关系数据文件
            relation_data_files = list(input_dir.glob('*_relations.csv'))
            if relation_data_files:
                logger.info(f"从 {input_dir} 发现 {len(relation_data_files)} 个关系数据文件。")
                self.relations_data = pd.concat([self._load_data_single(f) for f in relation_data_files], ignore_index=True)
            else:
                logger.warning(f"在目录 {input_dir} 中未找到任何关系数据文件。")
        else:
            # 如果没有input_dir，则使用单独指定的文件路径
            self.concepts_embeddings = self._load_embeddings_single(concepts_embedding_path, 'concept_name') if concepts_embedding_path else pd.DataFrame()
            self.relations_embeddings = self._load_embeddings_single(relations_embedding_path, 'relation_phrase') if relations_embedding_path else pd.DataFrame()
            self.concepts_data = self._load_data_single(concepts_data_path) if concepts_data_path else pd.DataFrame()
            self.relations_data = self._load_data_single(relations_data_path) if relations_data_path else pd.DataFrame()

        logger.info("RAGQuery 初始化完成。")

    def _load_embeddings_single(self, path: Path, text_column: str):
        if path.exists():
            df = pd.read_csv(path)
            # 假设 'embedding' 列存储的是字符串形式的numpy数组，需要转换
            df['embedding'] = df['embedding'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)
            # 过滤掉无法转换的行
            df = df[df['embedding'].apply(lambda x: isinstance(x, np.ndarray))]
            logger.info(f"从 {path} 加载了 {len(df)} 条Embedding数据。")
            return df
        else:
            logger.warning(f"Embedding文件 {path} 不存在。")
            return pd.DataFrame()

    def _load_data_single(self, path: Path):
        if path.exists():
            df = pd.read_csv(path)
            logger.info(f"从 {path} 加载了 {len(df)} 条原始数据。")
            return df
        else:
            logger.warning(f"数据文件 {path} 不存在。")
            return pd.DataFrame()

    def _search_similar(self, query_embedding: np.ndarray, embeddings_df: pd.DataFrame, top_k: int = 5):
        if embeddings_df.empty or 'embedding' not in embeddings_df.columns or embeddings_df['embedding'].isnull().all():
            return []
        
        # 计算余弦相似度
        # 确保所有embedding都是numpy数组且维度一致
        valid_embeddings = [e for e in embeddings_df['embedding'] if isinstance(e, np.ndarray) and e.size > 0]
        if not valid_embeddings:
            return []

        embeddings_matrix = np.vstack(valid_embeddings)
        
        # 归一化以计算余弦相似度
        query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_matrix_norm = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1)[:, np.newaxis]
        
        similarities = np.dot(embeddings_matrix_norm, query_embedding_norm)
        
        # 获取相似度最高的索引
        top_k_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_k_indices:
            # 找到原始DataFrame中对应的行
            original_row_index = embeddings_df.index[embeddings_df['embedding'].apply(lambda x: np.array_equal(x, valid_embeddings[idx]))].tolist()[0]
            results.append({
                'text': embeddings_df.loc[original_row_index, embeddings_df.columns[0]], # 第一个非embedding列作为文本
                'similarity': similarities[idx],
                'data': embeddings_df.loc[original_row_index].to_dict() # 包含所有原始数据
            })
        return results

    def query(self, user_query: str, top_k: int = 3):
        logger.info(f"处理查询: '{user_query}'")
        query_embedding = self.model.encode(user_query)

        # 搜索相似的概念
        similar_concepts = self._search_similar(query_embedding, self.concepts_embeddings, top_k)
        logger.info(f"找到 {len(similar_concepts)} 个相似概念。")

        # 搜索相似的关系
        similar_relations = self._search_similar(query_embedding, self.relations_embeddings, top_k)
        logger.info(f"找到 {len(similar_relations)} 个相似关系。")

        # 结合检索到的信息
        retrieved_info = ""
        if similar_concepts:
            retrieved_info += "相关概念：\n"
            for concept in similar_concepts:
                # 从原始概念数据中查找更详细的信息
                concept_name = concept['text']
                detail = self.concepts_data[self.concepts_data['name'] == concept_name].to_dict(orient='records')
                retrieved_info += f"- {concept_name} (相似度: {concept['similarity']:.4f})\n"
                if detail: retrieved_info += f"  详细信息: {detail[0]}\n"

        if similar_relations:
            retrieved_info += "\n相关关系：\n"
            for relation in similar_relations:
                # 从原始关系数据中查找更详细的信息
                relation_phrase = relation['text']
                # 关系短语可能需要解析回source, relation, target来匹配原始数据
                # 这是一个简化的例子，实际可能需要更复杂的匹配逻辑
                detail = self.relations_data[self.relations_data.apply(lambda row: f"{row['_o_source_']} {row[':TYPE']} {row['_o_target_']}" == relation_phrase, axis=1)].to_dict(orient='records')
                retrieved_info += f"- {relation_phrase} (相似度: {relation['similarity']:.4f})\n"
                if detail: retrieved_info += f"  详细信息: {detail[0]}\n"

        # TODO: 使用LLM生成答案
        # 这里只是一个占位符，实际需要集成一个LLM API
        if retrieved_info:
            response = f"根据您的查询和检索到的知识：\n{retrieved_info}\n\n请问您还需要了解什么？"
        else:
            response = "抱歉，未能找到与您查询相关的信息。"
        
        return response

def main():
    parser = argparse.ArgumentParser(description="基于RAG的知识图谱查询脚本。")
    parser.add_argument('--input_dir', type=str, help="包含Embedding和原始数据文件的目录。如果指定，将尝试加载默认文件。")
    parser.add_argument('--concepts_embedding_csv', type=str, help="概念Embedding CSV文件的路径。")
    parser.add_argument('--relations_embedding_csv', type=str, help="关系Embedding CSV文件的路径。")
    parser.add_argument('--concepts_data_csv', type=str, help="原始概念数据CSV文件的路径。")
    parser.add_argument('--relations_data_csv', type=str, help="原始关系数据CSV文件的路径。")
    parser.add_argument('--query', type=str, help="要查询的问题。如果未提供，将进入交互模式。")

    args = parser.parse_args()

    # 检查是否提供了足够的参数
    if not args.input_dir and not (args.concepts_embedding_csv and args.relations_embedding_csv and args.concepts_data_csv and args.relations_data_csv):
        logger.error("请至少提供 --input_dir 或所有单独的CSV文件路径参数。")
        return

    rag_query_instance = RAGQuery(
        concepts_embedding_path=Path(args.concepts_embedding_csv) if args.concepts_embedding_csv else None,
        relations_embedding_path=Path(args.relations_embedding_csv) if args.relations_embedding_csv else None,
        concepts_data_path=Path(args.concepts_data_csv) if args.concepts_data_csv else None,
        relations_data_path=Path(args.relations_data_csv) if args.relations_data_csv else None,
        input_dir=Path(args.input_dir) if args.input_dir else None
    )

    if args.query:
        response = rag_query_instance.query(args.query)
        print(response)
    else:
        print("进入交互式查询模式 (输入 'exit' 退出)。")
        while True:
            user_input = input("您的查询: ")
            if user_input.lower() == 'exit':
                break
            response = rag_query_instance.query(user_input)
            print(response)

if __name__ == '__main__':
    main()