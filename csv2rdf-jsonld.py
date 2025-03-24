import re
import csv
import json
import argparse

OntologyIRI = 'http://www.semanticweb.org/dell/ontologies/2025/2/untitled-ontology-8'

# 定义哪些 type 值被认为是类
class_types = {"概念", "元素", "类别", "模型", "组件"}

def concepts(input_csv):
    # 读取 CSV 文件，确保使用 utf-8 编码
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # 构建 :ID 到 name 的映射，用于确定父节点引用
    id_to_name = {row[":ID"].strip(): row["name"].strip() for row in rows if row[":ID"].strip()}

    output = []

    # 第一个元素为本体定义
    ontology = {
        "@id": OntologyIRI,
        "@type": ["http://www.w3.org/2002/07/owl#Ontology"]
    }
    output.append(ontology)

    for row in rows:
        # 忽略 :ID 为空的行
        if not row[":ID"].strip():
            continue

        entity = {}

        # 使用实体名称作为 URI 的 fragment，将空格替换为下划线
        fragment = row["name"].strip().replace(" ", "_")
        entity["@id"] = f"{OntologyIRI}#{fragment}"

        # 根据 type 字段判断实体类型
        typ = row["type"].strip()
        if typ in class_types:
            entity["@type"] = ["http://www.w3.org/2002/07/owl#Class"]
        else:
            entity["@type"] = ["http://www.w3.org/2002/07/owl#NamedIndividual"]

        # 添加标签
        entity["http://www.w3.org/2000/01/rdf-schema#label"] = [row["name"].strip()]

        # 处理父节点关系（CSV 中 parent 字段存储的是父节点的 :ID）
        parent_id = row["parent"].strip()
        if parent_id:
            parent_name = id_to_name.get(parent_id, "")
            if parent_name:
                parent_fragment = parent_name.replace(" ", "_")
                parent_uri = f"{OntologyIRI}#{parent_fragment}"
                # 如果实体为类，则用 rdfs:subClassOf 表示父子关系，否则用 rdf:type 指定所属类
                if "http://www.w3.org/2002/07/owl#Class" in entity["@type"]:
                    entity["http://www.w3.org/2000/01/rdf-schema#subClassOf"] = [{"@id": parent_uri}]
                else:
                    entity["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"] = [{"@id": parent_uri}]

        # 将除 :ID、name、type、parent 之外的其他非空字段作为附加属性加入
        for key in row:
            if key not in {":ID", "name", "type", "parent"}:
                value = row[key].strip()
                if value:
                    # 使用本体 IRI 加上字段名作为属性 IRI
                    prop = f"{OntologyIRI}#{key}"
                    entity[prop] = [value]

        output.append(entity)

    # # 写入 JSON-LD 输出文件
    # with open(output_jsonld, "w", encoding='utf-8') as f:
    #     json.dump(output, f, ensure_ascii=False, indent=2)
    return output

def relations(input_csv):
    # 读取 CSV 文件，确保使用 utf-8 编码
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    output = []

    for row in rows:
        # 忽略 :START_ID 为空的行
        ... # To be Done
    
    return output

def main(input_concepts, input_relations, output_jsonld):
    concepts_data = concepts(input_concepts)
    relations_data = relations(input_relations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 CSV 文件转换为符合 Protégé 标准的 JSON-LD 格式")
    parser.add_argument("-ic", "--input_concepts", help="输入 Concepts CSV 文件路径")
    parser.add_argument("-ir", "--input_relations", help="输入 Relations CSV 文件路径")
    parser.add_argument("output_jsonld", help="输出 JSON-LD 文件路径")
    args = parser.parse_args()
    
    main(args.input_concepts, args.input_relations, args.output_jsonld)