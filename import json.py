import json
from rdflib import Graph, Literal, Namespace, RDF, URIRef

def json_to_rdf(json_data):
    # 创建一个RDF图
    g = Graph()

    # 定义命名空间
    EX = Namespace("http://www.semanticweb.org/dell/ontologies/2025/2/untitled-ontology-6#")

    # 将JSON数据转换为RDF三元组
    lemma_uri = URIRef(EX[f"lemma/{json_data['lemmaId']}"])
    g.add((lemma_uri, RDF.type, EX.Lemma))
    g.add((lemma_uri, EX.title, Literal(json_data['lemmaTitle'])))
    g.add((lemma_uri, EX.description, Literal(json_data['lemmaDesc'])))

    # 处理嵌套结构
    ext_data = json_data.get("extData", {})
    classify = ext_data.get("classify", [])
    for item in classify:
        class_uri = URIRef(EX[f"class/{item['id']}"])
        g.add((lemma_uri, EX.hasClass, class_uri))
        g.add((class_uri, EX.name, Literal(item['name'])))

    # 处理其他属性和关系
    for key, value in json_data.items():
        if isinstance(value, (str, int, float)):
            g.add((lemma_uri, EX[key], Literal(value)))
        elif isinstance(value, list):
            for elem in value:
                if isinstance(elem, dict):
                    for sub_key, sub_value in elem.items():
                        sub_uri = URIRef(EX[f"{sub_key}/{sub_value}"])
                        g.add((lemma_uri, EX[key], sub_uri))

    # 返回RDF/XML格式
    return g.serialize(format='xml')

# 读取JSON文件
with open("d:\\Documents\\Coding\\KnowledgeGraph\\New 0.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 转换为RDF/XML
rdf_xml = json_to_rdf(json_data)
print(rdf_xml)

with open("d:\\Documents\\Coding\\KnowledgeGraph\\New 0.xml", "w", encoding="utf-8") as f:
    f.write(rdf_xml)