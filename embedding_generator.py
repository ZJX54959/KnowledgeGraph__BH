#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_embeddings(concepts_csv_path: Path, relations_csv_path: Path, output_dir: Path, base_name: str):
    """
    生成知识图谱概念和关系的Embedding。
    """
    logger.info(f"开始生成Embedding，概念文件: {concepts_csv_path}, 关系文件: {relations_csv_path}")

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载Embedding模型
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2') # 示例模型

    # 处理概念Embedding
    if concepts_csv_path.exists():
        concepts_df = pd.read_csv(concepts_csv_path)
        if 'name' in concepts_df.columns:
            concept_names = concepts_df['name'].tolist()
            logger.info(f"发现 {len(concept_names)} 个概念进行Embedding。")
            # 生成概念Embedding
            concept_embeddings = model.encode(concept_names)
            
            # 保存概念Embedding
            concepts_embedding_path = output_dir / f"{base_name}_concepts_embeddings.csv"
            pd.DataFrame({'concept_name': concept_names, 'embedding': concept_embeddings.tolist()}).to_csv(concepts_embedding_path, index=False)
            logger.info(f"概念Embedding已保存到: {concepts_embedding_path}")
            # 复制原始概念CSV文件到输出目录
            import shutil
            shutil.copy(concepts_csv_path, output_dir / concepts_csv_path.name)
        else:
            logger.warning(f"概念CSV文件 {concepts_csv_path} 中未找到 'name' 列。")
    else:
        logger.warning(f"概念CSV文件 {concepts_csv_path} 不存在，跳过概念Embedding生成。")

    # 处理关系Embedding
    if relations_csv_path.exists():
        relations_df = pd.read_csv(relations_csv_path)
        # 假设关系由 ':START_ID', ':TYPE', ':END_ID' 列定义
        if all(col in relations_df.columns for col in ['_o_source_', ':TYPE', '_o_target_']):
            relation_phrases = []
            for _, row in relations_df.iterrows():
                # 构建关系短语，例如 "source_id TYPE target_id"
                relation_phrases.append(f"{row['_o_source_']} {row[':TYPE']} {row['_o_target_']}")
            
            logger.info(f"发现 {len(relation_phrases)} 个关系进行Embedding。")
            # 生成关系Embedding
            relation_embeddings = model.encode(relation_phrases)

            # 保存关系Embedding
            relations_embedding_path = output_dir / f"{base_name}_relations_embeddings.csv"
            pd.DataFrame({'relation_phrase': relation_phrases, 'embedding': relation_embeddings.tolist()}).to_csv(relations_embedding_path, index=False)
            logger.info(f"关系Embedding已保存到: {relations_embedding_path}")
            # 复制原始关系CSV文件到输出目录
            import shutil
            shutil.copy(relations_csv_path, output_dir / relations_csv_path.name)
        else:
            logger.warning(f"关系CSV文件 {relations_csv_path} 中未找到 '_o_source_', ':TYPE' 或 '_o_target_' 列。")
    else:
        logger.warning(f"关系CSV文件 {relations_csv_path} 不存在，跳过关系Embedding生成。")

    logger.info("Embedding生成完成。")

def main():
    parser = argparse.ArgumentParser(description="批量生成知识图谱概念和关系的Embedding。")
    parser.add_argument('--input_dir', type=str, required=True, help="包含概念和关系CSV文件的输入目录。")
    parser.add_argument('--output_dir', type=str, default="./embeddings", help="保存Embedding的输出目录。")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        logger.error(f"输入路径 {input_dir} 不是一个有效的目录。")
        return

    # 查找所有以 '_concepts.csv' 结尾的文件
    concept_files = sorted(list(input_dir.glob('*_concepts.csv')))

    if not concept_files:
        logger.warning(f"在目录 {input_dir} 中未找到任何以 '_concepts.csv' 结尾的概念文件。")
        return

    for concepts_csv_path in concept_files:
        # 根据概念文件找到对应的关系文件
        base_name = concepts_csv_path.stem.replace('_concepts', '')
        relations_csv_path = input_dir / f"{base_name}_relations.csv"

        if relations_csv_path.exists():
            logger.info(f"正在处理文件对：{concepts_csv_path.name} 和 {relations_csv_path.name}")
            generate_embeddings(concepts_csv_path, relations_csv_path, output_dir, base_name)
        else:
            logger.warning(f"未找到与 {concepts_csv_path.name} 对应的关系文件 {relations_csv_path.name}，跳过。")

if __name__ == '__main__':
    main()