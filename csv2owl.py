import csv
import argparse
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL

# Define a default ontology IRI, can be overridden by command line argument
DEFAULT_ONTOLOGY_IRI = 'http://www.example.org/ontology#'

# Define which types are considered classes
CLASS_TYPES = {"概念", "元素", "类别", "模型", "组件", "现象", "类型", "目标", "方法", "算法", "属性", "操作", "主题", "任务", "组织"}

def concepts_to_owl(input_csv, ontology_iri):
    g = Graph()
    EX = Namespace(ontology_iri)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("ex", EX)

    # Add ontology declaration
    g.add((URIRef(ontology_iri), RDF.type, OWL.Ontology))

    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    id_to_uri = {}

    for row in rows:
        if not row.get(':ID') or not row.get('name'):
            print(f"Skipping row due to missing ID or name: {row}")
            continue

        entity_id = row[':ID'].strip()
        entity_name = row['name'].strip()
        entity_uri = EX[entity_id]
        id_to_uri[entity_id] = entity_uri

        entity_type_str = row.get('type:LABLE', '').strip()

        if entity_type_str in CLASS_TYPES:
            g.add((entity_uri, RDF.type, OWL.Class))
        else:
            # If not a class, it's an individual. We need to assign it to a class.
            # For now, let's make it an individual of owl:Thing or a generic class if parent is not specified
            # or if parent is also an individual (which is not ideal for OWL structure)
            g.add((entity_uri, RDF.type, OWL.NamedIndividual))
            # Attempt to assign to a class based on parent if parent is a class
            parent_id = row.get('parent', '').strip()
            if parent_id and parent_id in id_to_uri:
                parent_uri = id_to_uri[parent_id]
                # Check if parent is a class
                is_parent_class = False
                for s, p, o in g.triples((parent_uri, RDF.type, OWL.Class)):
                    is_parent_class = True
                    break
                if is_parent_class:
                    g.add((entity_uri, RDF.type, parent_uri))
                # else: # Parent is an individual, this case needs careful handling or a default class
                #     g.add((entity_uri, RDF.type, OWL.Thing)) # Fallback
            # else:
            #     g.add((entity_uri, RDF.type, OWL.Thing)) # Fallback if no parent or parent not processed

        g.add((entity_uri, RDFS.label, Literal(entity_name)))

        parent_id = row.get('parent', '').strip()
        if parent_id and parent_id in id_to_uri:
            parent_uri = id_to_uri[parent_id]
            if entity_type_str in CLASS_TYPES:
                 # Check if parent is also a class before adding subClassOf
                is_parent_class_for_subclass = False
                for s,p,o in g.triples((parent_uri, RDF.type, OWL.Class)):
                    is_parent_class_for_subclass = True
                    break
                if is_parent_class_for_subclass:
                    g.add((entity_uri, RDFS.subClassOf, parent_uri))
            # For individuals, the type assertion to parent class is handled above

        # Add other properties
        for key, value in row.items():
            if key not in [':ID', 'name', 'type:LABLE', 'parent'] and value and value.strip():
                prop_uri = EX[key.replace(':', '_').replace(' ', '_')]
                # Check if property exists, if not, define it as AnnotationProperty or DatatypeProperty/ObjectProperty
                # For simplicity, let's assume they are annotation properties or datatype properties for now
                is_prop_defined = any(g.triples((prop_uri, RDF.type, None)))
                if not is_prop_defined:
                    g.add((prop_uri, RDF.type, OWL.AnnotationProperty)) # Default to AnnotationProperty
                g.add((entity_uri, prop_uri, Literal(value.strip())))

    return g, id_to_uri

def relations_to_owl(g, input_csv, id_to_uri, ontology_iri):
    EX = Namespace(ontology_iri)
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            start_id = row.get(':START_ID', '').strip()
            end_id = row.get(':END_ID', '').strip()
            rel_type = row.get(':TYPE', '').strip()

            if not start_id or not end_id or not rel_type:
                print(f"Skipping relation due to missing fields: {row}")
                continue

            subject_uri = id_to_uri.get(start_id)
            object_uri = id_to_uri.get(end_id)

            if not subject_uri or not object_uri:
                print(f"Skipping relation due to unknown subject/object ID: {row}")
                continue

            # Sanitize relation type to be a valid URI fragment
            predicate_local_name = rel_type.replace(' ', '_')
            predicate_uri = EX[predicate_local_name]

            # Define the property if it doesn't exist
            # For simplicity, define as owl:ObjectProperty. Domain/range can be added for more rigor.
            is_prop_defined = any(g.triples((predicate_uri, RDF.type, None)))
            if not is_prop_defined:
                g.add((predicate_uri, RDF.type, OWL.ObjectProperty))
                # Optionally add rdfs:label for the property
                g.add((predicate_uri, RDFS.label, Literal(rel_type)))

            g.add((subject_uri, predicate_uri, object_uri))
    return g

def main():
    parser = argparse.ArgumentParser(description='Convert CSV files to OWL RDF/XML format.')
    parser.add_argument('-ic', '--input_concepts', required=True, help='Input Concepts CSV file path.')
    parser.add_argument('-ir', '--input_relations', required=True, help='Input Relations CSV file path.')
    parser.add_argument('-o', '--output_owl', required=True, help='Output OWL RDF/XML file path.')
    parser.add_argument('--ontology_iri', default=DEFAULT_ONTOLOGY_IRI, help=f'Ontology IRI (default: {DEFAULT_ONTOLOGY_IRI}).')

    args = parser.parse_args()

    ontology_iri = args.ontology_iri
    if not ontology_iri.endswith('#') and not ontology_iri.endswith('/'):
        ontology_iri += '#'

    graph, id_to_uri_map = concepts_to_owl(args.input_concepts, ontology_iri)
    graph = relations_to_owl(graph, args.input_relations, id_to_uri_map, ontology_iri)

    graph.serialize(destination=args.output_owl, format='xml')
    print(f"OWL file generated at {args.output_owl}")

if __name__ == '__main__':
    main()