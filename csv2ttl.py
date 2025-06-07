import csv
import argparse
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Define a default ontology IRI, can be overridden by command line argument
DEFAULT_ONTOLOGY_IRI = 'http://www.example.org/ontology#'
DEFAULT_BASE_IRI = 'http://www.example.org/data/' # For individuals and other data

# Define which types are considered classes
CLASS_TYPES = {"概念", "元素", "类别", "模型", "组件", "现象", "类型", "目标", "方法", "算法", "属性", "操作", "主题", "任务", "组织"}

def concepts_to_ttl(input_csv, ontology_iri, base_iri):
    g = Graph()
    EX_ONT = Namespace(ontology_iri)
    EX_DATA = Namespace(base_iri)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("ont", EX_ONT) # Ontology prefix
    g.bind("data", EX_DATA) # Data prefix
    g.bind("", EX_DATA) # Default namespace for data if not specified

    # Add ontology declaration (optional in TTL but good practice for context)
    # g.add((URIRef(ontology_iri), RDF.type, OWL.Ontology))

    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    id_to_uri = {}

    for row in rows:
        if not row.get(':ID') or not row.get('name'):
            print(f"Skipping row due to missing ID or name: {row}")
            continue

        entity_id_str = row[':ID'].strip()
        entity_name = row['name'].strip()
        
        # Decide if the URI should be in ontology or data namespace
        entity_type_str = row.get('type:LABLE', '').strip()
        if entity_type_str in CLASS_TYPES:
            entity_uri = EX_ONT[entity_id_str] # Classes in ontology namespace
        else:
            entity_uri = EX_DATA[entity_id_str] # Individuals in data namespace
            
        id_to_uri[entity_id_str] = entity_uri

        if entity_type_str in CLASS_TYPES:
            g.add((entity_uri, RDF.type, OWL.Class))
        else:
            g.add((entity_uri, RDF.type, OWL.NamedIndividual))
            # Try to assign to a class based on parent if parent is a class
            parent_id = row.get('parent', '').strip()
            if parent_id and parent_id in id_to_uri:
                parent_uri_for_type = id_to_uri[parent_id]
                # Check if this parent_uri is actually a class (defined in ontology namespace)
                if str(parent_uri_for_type).startswith(ontology_iri):
                     g.add((entity_uri, RDF.type, parent_uri_for_type))
                # else: # Parent is an individual or in data namespace, not suitable as a class type here
                #    pass # Or assign to a default class like owl:Thing if needed

        g.add((entity_uri, RDFS.label, Literal(entity_name)))

        parent_id = row.get('parent', '').strip()
        if parent_id and parent_id in id_to_uri:
            parent_uri = id_to_uri[parent_id]
            if entity_type_str in CLASS_TYPES:
                # Ensure parent is also a class (in ontology namespace) for rdfs:subClassOf
                if str(parent_uri).startswith(ontology_iri):
                    g.add((entity_uri, RDFS.subClassOf, parent_uri))
            # For individuals, rdf:type to parent class is handled above.
            # If a different relationship to parent individual is needed, it would be a specific property.

        # Add other properties
        for key, value in row.items():
            if key not in [':ID', 'name', 'type:LABLE', 'parent'] and value and value.strip():
                prop_local_name = key.replace(':', '_').replace(' ', '_')
                prop_uri = EX_ONT[prop_local_name] # Properties defined in ontology namespace
                
                # Define property type if not already defined (simplified)
                is_prop_defined = any(g.triples((prop_uri, RDF.type, None)))
                if not is_prop_defined:
                    # Simple assumption: if value looks like a URI (e.g. from id_to_uri), it's an ObjectProperty
                    # This is a heuristic and might need refinement based on actual data patterns
                    is_object_prop = False
                    # A more robust check would be to see if 'value' corresponds to an ID in id_to_uri
                    # For now, we'll default to AnnotationProperty or DatatypeProperty
                    g.add((prop_uri, RDF.type, OWL.AnnotationProperty)) # Default
                    g.add((prop_uri, RDFS.label, Literal(key)))

                g.add((entity_uri, prop_uri, Literal(value.strip())))

    return g, id_to_uri

def relations_to_ttl(g, input_csv, id_to_uri, ontology_iri, base_iri):
    EX_ONT = Namespace(ontology_iri)
    # EX_DATA = Namespace(base_iri) # Not strictly needed here if subjects/objects are already correct

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

            predicate_local_name = rel_type.replace(' ', '_')
            predicate_uri = EX_ONT[predicate_local_name] # Relations/Properties in ontology namespace

            is_prop_defined = any(g.triples((predicate_uri, RDF.type, None)))
            if not is_prop_defined:
                g.add((predicate_uri, RDF.type, OWL.ObjectProperty))
                g.add((predicate_uri, RDFS.label, Literal(rel_type)))
                # Could add rdfs:domain and rdfs:range if class information of subject/object is available and reliable

            g.add((subject_uri, predicate_uri, object_uri))
    return g

def main():
    parser = argparse.ArgumentParser(description='Convert CSV files to Turtle (TTL) format.')
    parser.add_argument('-ic', '--input_concepts', required=True, help='Input Concepts CSV file path.')
    parser.add_argument('-ir', '--input_relations', required=True, help='Input Relations CSV file path.')
    parser.add_argument('-o', '--output_ttl', required=True, help='Output TTL file path.')
    parser.add_argument('--ontology_iri', default=DEFAULT_ONTOLOGY_IRI, help=f'Ontology IRI (default: {DEFAULT_ONTOLOGY_IRI}). Used for classes and properties.')
    parser.add_argument('--base_iri', default=DEFAULT_BASE_IRI, help=f'Base IRI for data/individuals (default: {DEFAULT_BASE_IRI}).')

    args = parser.parse_args()

    ontology_iri = args.ontology_iri
    if not ontology_iri.endswith('#') and not ontology_iri.endswith('/'):
        ontology_iri += '#'
    
    base_iri = args.base_iri
    if not base_iri.endswith('#') and not base_iri.endswith('/'):
        base_iri += '/' # Usually ends with / for data

    graph, id_to_uri_map = concepts_to_ttl(args.input_concepts, ontology_iri, base_iri)
    graph = relations_to_ttl(graph, args.input_relations, id_to_uri_map, ontology_iri, base_iri)

    graph.serialize(destination=args.output_ttl, format='turtle')
    print(f"TTL file generated at {args.output_ttl}")

if __name__ == '__main__':
    main()