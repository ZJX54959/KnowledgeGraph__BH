import re
import csv
import json
import argparse
import hashlib

OntologyIRI = 'http://www.semanticweb.org/dell/ontologies/2025/2/untitled-ontology-8'

# 定义哪些 type 值被认为是类
class_types = {"概念", "元素", "类别", "模型", "组件"}

def _generate_id(name):
    """生成唯一标识符"""
    return hashlib.md5(name.encode()).hexdigest()[:8]

def concepts(input_csv):
    # 读取 CSV 文件，确保使用 utf-8 编码
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # 构建 :ID 到 name 的映射，用于确定父节点引用
    id_to_name = {row[":ID"].strip(): row["name"].strip() for row in rows if row[":ID"].strip()}
    
    output = []
    id_to_uri = {}  # 记录每个概念的 URI 映射

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

        # 使用 _generate_id 生成唯一标识符
        entity_id = _generate_id(row["name"].strip())
        uri = f"{OntologyIRI}#{entity_id}"
        id_to_uri[row[":ID"].strip()] = uri

        entity = {}
        entity["@id"] = uri

        # 根据 type 字段判断实体类型
        typ = row["type"].strip()
        if typ in class_types:
            entity["@type"] = ["http://www.w3.org/2002/07/owl#Class"]
        else:
            entity["@type"] = ["http://www.w3.org/2002/07/owl#NamedIndividual"]

        # 添加标签
        entity["http://www.w3.org/2000/01/rdf-schema#label"] = [row["name"].strip()]

        # 添加注释 (rdfs:comment)
        if "描述:LABEL" in row and row["描述:LABEL"].strip():
            entity["http://www.w3.org/2000/01/rdf-schema#comment"] = [row["描述:LABEL"].strip()]

        # 处理父节点关系（CSV 中 parent 字段存储的是父节点的 :ID）
        parent_id = row["parent"].strip()
        if parent_id:
            parent_name = id_to_name.get(parent_id, "")
            if parent_name:
                parent_entity_id = _generate_id(parent_name)
                parent_uri = f"{OntologyIRI}#{parent_entity_id}"
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
                    prop = f"{OntologyIRI}#{key}"
                    entity[prop] = [value]
        output.append(entity)

    return output, id_to_uri

def relations(input_csv, id_to_uri):
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    output = []
    for row in rows:
        start_id = row[":START_ID"].strip()
        end_id = row[":END_ID"].strip()
        if not start_id or not end_id:
            continue

        subject_uri = id_to_uri.get(start_id)
        object_uri = id_to_uri.get(end_id)
        if not subject_uri or not object_uri:
            continue

        pred_fragment = row["type"].strip().replace(" ", "_")
        predicate = f"{OntologyIRI}#{pred_fragment}"

        output.append({
            "subject": subject_uri,
            "predicate": predicate,
            "object": object_uri
        })
    return output

def main(input_concepts, input_relations, output_jsonld, load_jsonld=None):
    if load_jsonld:
        with open(load_jsonld, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    new_concepts, id_to_uri = concepts(input_concepts)
    new_relations = relations(input_relations, id_to_uri)

    # 合并 concepts: 如果存在相同 @id 则进行合并，否则添加
    existing_ids = {entity.get("@id") for entity in existing_data if "@id" in entity}
    merged = existing_data.copy()
    for concept in new_concepts:
        if concept.get("@id") not in existing_ids:
            merged.append(concept)
        else:
            for entity in merged:
                if entity.get("@id") == concept.get("@id"):
                    for key, value in concept.items():
                        if key not in entity:
                            entity[key] = value
                        elif isinstance(entity[key], list) and isinstance(value, list):
                            for v in value:
                                if v not in entity[key]:
                                    entity[key].append(v)
                    break

    # 整合 relations 到对应概念
    for rel in new_relations:
        subject_uri = rel["subject"]
        predicate = rel["predicate"]
        object_uri = rel["object"]
        for entity in merged:
            if entity.get("@id") == subject_uri:
                if predicate in entity:
                    exists = any(item.get("@id") == object_uri for item in entity[predicate])
                    if not exists:
                        entity[predicate].append({"@id": object_uri})
                else:
                    entity[predicate] = [{"@id": object_uri}]
                break

    with open(output_jsonld, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 CSV 文件转换为符合 Protégé 标准的 JSON-LD 格式")
    parser.add_argument("-i", "--input", help="输入 CSV 文件或包含 CSV 文件的目录路径")
    parser.add_argument("-ic", "--input_concepts", help="输入 Concepts CSV 文件路径 (当 -i 为文件时使用)")
    parser.add_argument("-ir", "--input_relations", help="输入 Relations CSV 文件路径 (当 -i 为文件时使用)")
    parser.add_argument("-o", "--output_jsonld", help="输出 JSON-LD 文件路径")
    parser.add_argument("--ontology_iri", help="Ontology IRI")
    parser.add_argument("-l", "--load", help="Load JSON-LD file path")
    args = parser.parse_args()

    OntologyIRI = args.ontology_iri or OntologyIRI

    if args.input is None and (args.input_concepts is None or args.input_relations is None):
        parser.print_help()
        exit(1)
    elif args.input and not os.path.exists(args.input):
        print(f"\033[33m输入路径不存在:\033[0m {args.input}")
        exit(1)
    elif args.input and os.path.isdir(args.input):
        concept_files = {}
        relation_files = {}
        for file in os.listdir(args.input):
            if file.endswith('_concepts.csv'):
                prefix = file.replace('_concepts.csv', '')
                concept_files[prefix] = os.path.join(args.input, file)
            elif file.endswith('_relations.csv'):
                prefix = file.replace('_relations.csv', '')
                relation_files[prefix] = os.path.join(args.input, file)

        for prefix in concept_files:
            if prefix in relation_files:
                input_concepts_path = concept_files[prefix]
                input_relations_path = relation_files[prefix]
                output_jsonld_path = os.path.join(args.input, f"{prefix}.jsonld")
                print(f"\033[32m处理文件对:\033[0m {input_concepts_path} 和 {input_relations_path}")
                main(input_concepts_path, input_relations_path, output_jsonld_path, load_jsonld=args.load)
            else:
                print(f"\033[33m警告: 找不到与 {prefix}_concepts.csv 匹配的 relations 文件。跳过此概念文件。\033[0m")
    elif args.input and os.path.isfile(args.input):
        # 如果 -i 参数指向一个文件，则需要同时提供 -ic 和 -ir
        if args.input_concepts and args.input_relations:
            main(args.input_concepts, args.input_relations, args.output_jsonld, load_jsonld=args.load)
        else:
            print("\033[31m错误: 当 -i 参数指向文件时，必须同时提供 -ic 和 -ir 参数。\033[0m")
            parser.print_help()
            exit(1)
    else:
        # 如果没有提供 -i 参数，则使用 -ic 和 -ir 参数
        if args.input_concepts and args.input_relations:
            main(args.input_concepts, args.input_relations, args.output_jsonld, load_jsonld=args.load)
        else:
            parser.print_help()
            exit(1)