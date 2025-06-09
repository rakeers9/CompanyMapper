import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["OPENAI_API_KEY"] = "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"

ALLOWED_RELATIONSHIP_TYPES = [
    "Supplies", "Manufactures", "Designs", "Distributes", "Operates_In", 
    "Owns", "Acquires", "Partners_With", "Licenses_From", "Regulated_By", 
    "Complies_With", "Uses_Material", "Generates_Emission", "Has_Initiative", 
    "Invests_In", "Provides_Service", "Faces_Litigation", "Receives_Certification", 
    "Reports_On", "Impacts"
]


file_paths = [
        "/Users/sreekargudipati/Coding Projects/CompanyMapper/documents/apple_10K_2024_annualreport.pdf",
        "/Users/sreekargudipati/Coding Projects/CompanyMapper/documents/Apple_Environmental_Progress_Report_2024.pdf",
        "/Users/sreekargudipati/Coding Projects/CompanyMapper/documents/Apple-Supply-Chain-2025-Progress-Report.pdf"
]


def load_documents(file_paths: List[str]):
    all_docs = []
    for file_path in file_paths:
        doc_type = file_path.split("/")[-1].split(".")[-1]
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        for doc in docs:
            doc.metadata['source'] = file_path
            
        all_docs.extend(docs)
    return all_docs



# documents = load_documents(file_paths)
# print(f"Loaded {len(documents)} documents.")
# print(f"First document: {documents[5].page_content[:100]}...")


def split_documents(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
    )
    return text_splitter.split_documents(documents)

# chunks = split_documents(documents)
# print(f"Split into {len(chunks)} chunks.")
# print(f"First chunk: {chunks[0].page_content[:100]}...")


def extract_entities_and_relationships(chunk: Document):
    print(f"Processing chunk with metadata: {chunk.metadata}")
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=os.environ["OPENAI_API_KEY"])
    print("LLM initialized")
    prompt_template = """
        You are an expert in extracting structured information about Apple Inc. from documents.
        
        Extract entities and their relationships from the following text. Focus on:
        1. Apple and its business units
        2. Suppliers and manufacturing partners
        3. Products and components
        4. Geographic locations of operations
        5. Materials used in products
        6. Environmental initiatives
        7. Regulations and compliance
        

        PRIORITY FOCUS - Extract these high-value relationships:
        1. Specific suppliers and what they supply to Apple (use "Supplies")
        2. Materials used in products (use "Uses_Material") 
        3. Manufacturing relationships (use "Manufactures")
        4. Key operational locations (use "Operates_In")

        ENTITY GUIDELINES:
        - For products: Use product LINES (iPhone, iPad, Mac) not specific models
        - For suppliers: Extract SPECIFIC company names, not generic terms
        - For materials: Extract specific materials (aluminum, rare earth elements, etc.)
        - For locations: Focus on major operational centers and manufacturing hubs

        AVOID:
        - Generic entities like "Various Suppliers" or "Various Components"
        - Over-detailed product variants (iPad Pro vs just iPad)
        - Internal Apple organizational relationships (use "Designs" sparingly)

        
        IMPORTANT: Only use these specific relationship types:
        - Supplies: When entity A supplies to entity B
        - Manufactures: When entity A manufactures entity B
        - Designs: When entity A designs entity B
        - Distributes: When entity A distributes entity B
        - Operates_In: When entity A operates in location B
        - Owns: When entity A owns entity B
        - Acquires: When entity A acquires entity B
        - Partners_With: When entity A partners with entity B
        - Licenses_From: When entity A licenses from entity B
        - Regulated_By: When entity A is regulated by entity B
        - Complies_With: When entity A complies with regulation/standard B
        - Uses_Material: When entity A uses material B
        - Generates_Emission: When entity A generates emission B
        - Has_Initiative: When entity A has environmental initiative B
        - Invests_In: When entity A invests in entity B
        - Provides_Service: When entity A provides service B
        - Faces_Litigation: When entity A faces litigation B
        - Receives_Certification: When entity A receives certification B
        - Reports_On: When entity A reports on topic B
        - Impacts: When entity A impacts entity B
        
        Do NOT create relationships with any other types. 
        Discard any relationships that do not match these types.
        
        TEXT:
        {text}
        
        SOURCE:
        {source}
        
        OUTPUT FORMAT:
        Return a JSON object with these arrays:
        1. "entities": List of entities found with their types and properties
        2. "relationships": List of relationships between entities with their types and properties
        
        Example:
        {{
            "entities": [
                {{"id": "apple", "type": "Company", "name": "Apple Inc.", "properties": {{"description": "Technology company"}}}},
                {{"id": "foxconn", "type": "Supplier", "name": "Foxconn", "properties": {{"description": "Manufacturing partner", "risk_level": "Medium"}}}}
            ],
            "relationships": [
                {{"source": "foxconn", "target": "apple", "type": "Supplies", "properties": {{"details": "Assembles iPhones", "dependency_level": "Critical"}}}}
            ]
        }}
        
        Return ONLY valid JSON:
        """
        
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    print("Chain initialized")

    result = chain.invoke({"text": chunk.page_content, "source": chunk.metadata["source"]}).content
    print(f"LLM output: {result}")
    
    try:
        extracted_data = json.loads(result)
        
        if not isinstance(extracted_data, dict):
            print(f"Error: Expected a dictionary, got {type(extracted_data)}")
            return {"entities": [], "relationships": []}
            
        if "entities" not in extracted_data or "relationships" not in extracted_data:
            print(f"Error: Missing 'entities' or 'relationships' keys in the output")
            return {"entities": [], "relationships": []}
            
        valid_entities = []
        for entity in extracted_data.get("entities", []):
            if not isinstance(entity, dict):
                print(f"Warning: Expected entity to be a dict, got {type(entity)}")
                continue
                
            if "id" not in entity or "type" not in entity or "name" not in entity:
                print(f"Warning: Entity missing required fields: {entity}")
                continue
                
            if "properties" not in entity or not isinstance(entity["properties"], dict):
                entity["properties"] = {}
                
            valid_entities.append(entity)
            
        valid_relationships = []
        filtered_count = 0
        
        for rel in extracted_data.get("relationships", []):
            if not isinstance(rel, dict):
                print(f"Warning: Expected relationship to be a dict, got {type(rel)}")
                filtered_count += 1
                continue
            
            # if "source" not in rel or "target" not in rel or "type" not in rel:
            #     print(f"Warning: Relationship missing required fields: {rel}")
            #     continue
            
            if rel["type"] not in ALLOWED_RELATIONSHIP_TYPES:
                print(f"Warning: Invalid relationship type '{rel['type']}' in {rel}")
                filtered_count += 1
                continue
                
                
            if "properties" not in rel or not isinstance(rel["properties"], dict):
                rel["properties"] = {}
                
            valid_relationships.append(rel)
        
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} relationships due to invalid structure or disallowed types")
        
        print(f"Final count: {len(valid_entities)} entities, {len(valid_relationships)} relationships")
        return {
            "entities": valid_entities,
            "relationships": valid_relationships
        }
        
    except json.JSONDecodeError:
        print(f"Error parsing LLM output as JSON: {result}")
        return {"entities": [], "relationships": []}

    
def process_all_chunks(chunks: List[Document]):
    print(f"Processing {len(chunks)} chunks")
    relationship_type_counts = {}
    all_entities = {}
    all_relationships = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        extracted_data = extract_entities_and_relationships(chunk)
        
        print(f"Found {len(extracted_data['entities'])} entities and {len(extracted_data['relationships'])} relationships")

        
        for entity in extracted_data["entities"]:
            if not isinstance(entity, dict) or "id" not in entity:
                print(f"Warning: Skipping invalid entity: {entity}")
                continue
                
            entity_id = entity["id"]
            if entity_id not in all_entities:
                all_entities[entity_id] = entity
            else:
                if "properties" not in entity:
                    entity["properties"] = {}
                if "properties" not in all_entities[entity_id]:
                    all_entities[entity_id]["properties"] = {}
                all_entities[entity_id]["properties"].update(entity["properties"])
        
        for rel in extracted_data["relationships"]:
            all_relationships.append(rel)
            # if not isinstance(rel, dict) or "source" not in rel or "target" not in rel:
            #     print(f"Warning: Skipping invalid relationship: {rel}")
            #     continue
            
            rel_type = rel.get("type", "Unknown")
            relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1
        
        # TO STOP IT TAKING TOO LONG                    
        if i >= 30:
            print("Processed 30 chunks, stopping for now.")
            break
    
    entities_list = list(all_entities.values())
    
    result = {
        "entities": entities_list,
        "relationships": all_relationships
    }
    
    print(f"\nFinished processing. Found {len(entities_list)} unique entities and {len(all_relationships)} relationships.")
    print("\nRelationship types found:")
    for rel_type, count in sorted(relationship_type_counts.items(), key=lambda x: x[1], reverse=True):
        # Mark if this type is allowed or not
        status = "✓ ALLOWED" if rel_type in ALLOWED_RELATIONSHIP_TYPES else "✗ SHOULD NOT APPEAR"
        print(f"  {rel_type}: {count} ({status})")
    
    # Check if any disallowed types slipped through
    disallowed_found = [rt for rt in relationship_type_counts.keys() if rt not in ALLOWED_RELATIONSHIP_TYPES]
    if disallowed_found:
        print(f"\nWARNING: Found {len(disallowed_found)} disallowed relationship types: {disallowed_found}")
        print("This might indicate the LLM is not following instructions properly.")
    else:
        print(f"\n✓ SUCCESS: All relationship types are from the allowed list!")
        
    return result
        
        
        
# knowledge_graph = process_all_chunks(chunks)
# print(knowledge_graph['entities'][8])

def create_networkx_graph(knowledge_graph: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()
    
    for entity in knowledge_graph["entities"]:
        attributes = {
            "type": entity["type"],
            "name": entity["name"],
        }
        
        for prop_key, prop_value in entity["properties"].items():
            attributes[f"prop_{prop_key}"] = prop_value
            
        if "source" in entity:
            attributes["source"] = entity["source"]
            
        G.add_node(entity["id"], **attributes)
    
    for rel in knowledge_graph["relationships"]:
        if rel["source"] not in G or rel["target"] not in G:
            continue
            
        rel_key = (rel["source"], rel["target"], rel["type"])
        
        attributes = {
            "type": rel["type"],
        }
        
        for prop_key, prop_value in rel["properties"].items():
            attributes[f"prop_{prop_key}"] = prop_value
            
        G.add_edge(rel["source"], rel["target"], **attributes)
    
    return G


def validate_graph_relationships(G):
    """
    Validate that the NetworkX graph only contains allowed relationship types
    """
    print("\n=== FINAL GRAPH VALIDATION ===")
    
    graph_rel_types = {}
    for _,_, attrs in G.edges(data=True):
        rel_type = attrs.get('type', 'Unknown')
        graph_rel_types[rel_type] = graph_rel_types.get(rel_type, 0) + 1
        
    print("Relationship types in final graph:")
    all_allowed = True
    
    for rel_type, count in sorted(graph_rel_types.items(), key=lambda x: x[1], reverse=True):
        if rel_type in ALLOWED_RELATIONSHIP_TYPES:
            print(f"  ✓ {rel_type}: {count}")
        else:
            print(f"  ✗ {rel_type}: {count} (SHOULD NOT BE PRESENT)")
            all_allowed = False
    
    if all_allowed:
        print(f"\n✓ SUCCESS: All {sum(graph_rel_types.values())} relationships use allowed types!")
    else:
        print(f"\n✗ WARNING: Some relationships use disallowed types!")
    
    return all_allowed
        
    
    
    
# G = create_networkx_graph(knowledge_graph)
# validate_graph_relationships(G)

# print(f"Knowledge graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# print(G.nodes(data=True))