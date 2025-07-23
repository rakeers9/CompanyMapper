# knowledge_graph.py

import os
import json
from dotenv import load_dotenv
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
from collections import defaultdict

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

ROOT_ID  = "apple"
CAT_REL = "Belongs_To"

MACRO_CATEGORIES = {
    "cat::Product"    : "Products",
    "cat::Supplier"   : "Suppliers",
    "cat::Component"  : "Components",
    "cat::Material"   : "Materials",
    "cat::Location"   : "Locations",
    "cat::Employee"   : "Employees",
    "cat::Service"    : "Services",
    "cat::Initiative" : "Initiatives"
}

GROUP_KEYS = {
    "Product":   ["family", "name"],
    "Supplier":  ["country", "industry"],
    "Component": ["component_type", "family"],
    "Material":  ["material_class"],
    "Employee":  ["department", "role"],
    "Service":   ["service_class"],
    "Initiative":["initiative_type"],
    "Location":  [],
}

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
        #doc_type = file_path.split("/")[-1].split(".")[-1]
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
        
        #TODO:Pandas db
        
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    print("Chain initialized")

    result = chain.invoke({"text": chunk.page_content, "source": chunk.metadata["source"]}).content
    print(f"LLM output: {result}")
    
    try:
        extracted_data = json.loads(result) #could add loop to try to do it multiple times
        
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
        # if i >= 15:
        #     print("Processed 15 chunks, stopping for now.")
        #     break
    
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
           
    
def save_networkx_graph(G: nx.DiGraph, output_file: str):
    """Save the NetworkX graph to multiple formats"""
    import os
    import pickle
    
    # Save as pickle for Streamlit compatibility
    with open(f"{output_file}.gpickle", 'wb') as f:
        pickle.dump(G, f)
    
    # Save as GEXF (backup format)
    nx.write_gexf(G, f"{output_file}.gexf")
    
    # Save as JSON for inspection
    graph_data = {
        "nodes": [{"id": node, **G.nodes[node]} for node in G.nodes()],
        "links": [{"source": u, "target": v, **G[u][v]} for u, v in G.edges()]
    }
    with open(f"{output_file}.json", 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"Graph saved in multiple formats with base name: {output_file}")

# def ensure_hierarchy(G: nx.DiGraph) -> nx.DiGraph:
#     # ---------- first ring ----------
#     for cat_id, label in MACRO_CATEGORIES.items():
#         if cat_id not in G:
#             G.add_node(cat_id, type="Category", name=label)
#         if not G.has_edge(ROOT_ID, cat_id):
#             G.add_edge(ROOT_ID, cat_id, type=CAT_REL)

#     # ---------- build buckets for potential sub-groups ----------
#     buckets = defaultdict(list)

#     for n, attrs in list(G.nodes(data=True)):
#         node_type = attrs.get("type")           # may be None on old nodes
#         if n == ROOT_ID or node_type == "Category":
#             continue

#         # fall-back to catch-all if type missing
#         cat_id = f"cat::{node_type}" if node_type else "cat::Other"
#         if cat_id not in MACRO_CATEGORIES:
#             cat_id = "cat::Other"

#         # choose first grouping key that exists
#         group_name = None
#         for key in GROUP_KEYS.get(node_type, []):
#             val = attrs.get(key)
#             if val:
#                 group_name = val.split()[0] if key == "name" else val
#                 break

#         if group_name:
#             buckets[(cat_id, group_name)].append(n)
#         else:
#             G.add_edge(cat_id, n, type=CAT_REL)

#     # ---------- create sub-category nodes ----------
#     for (cat_id, group), members in buckets.items():
#         if len(members) == 1:
#             G.add_edge(cat_id, members[0], type=CAT_REL)
#             continue

#         sub_id = f"{cat_id}::{group.lower().replace(' ', '_')}"
#         if sub_id not in G:
#             G.add_node(sub_id, name=group, type="Subcategory")
#             G.add_edge(cat_id, sub_id, type=CAT_REL)
#         for m in members:
#             G.add_edge(sub_id, m, type=CAT_REL)

#     return G



# def ensure_hierarchy(G: nx.DiGraph) -> nx.DiGraph:
#     """
#     Ensure the graph has a proper hierarchy with categories and subcategories
#     Fixed to handle nodes without 'type' attribute gracefully
#     """
#     # First, ensure ROOT_ID exists
#     if ROOT_ID not in G:
#         G.add_node(ROOT_ID, type="Company", name="Apple Inc.")
    
#     # Create category nodes and connect them to root
#     for cat_id, label in MACRO_CATEGORIES.items():
#         if cat_id not in G:
#             G.add_node(cat_id, type="Category", name=label)
#         if not G.has_edge(ROOT_ID, cat_id):
#             G.add_edge(ROOT_ID, cat_id, type=CAT_REL)
            
#     buckets = defaultdict(list)
    
#     # Process all nodes to organize them into categories
#     for n, attrs in list(G.nodes(data=True)):
#         # Skip root and category nodes
#         if n == ROOT_ID or n.startswith("cat::") or attrs.get("type") == "Category":
#             continue
        
#         # Get node type, default to "Other" if missing
#         node_type = attrs.get('type', 'Other')
        
#         # Find the appropriate category
#         cat_id = f"cat::{node_type}"
#         if cat_id not in MACRO_CATEGORIES:
#             # If the node type doesn't have a predefined category, put it in "Other"
#             cat_id = "cat::Other"
#             # Also ensure the "Other" category exists
#             if cat_id not in MACRO_CATEGORIES:
#                 MACRO_CATEGORIES[cat_id] = "Other"
#                 if cat_id not in G:
#                     G.add_node(cat_id, type="Category", name="Other")
#                 if not G.has_edge(ROOT_ID, cat_id):
#                     G.add_edge(ROOT_ID, cat_id, type=CAT_REL)
        
#         # Try to find a grouping key for subcategories
#         group_name = None
#         group_keys = GROUP_KEYS.get(node_type, [])
        
#         for key in group_keys:
#             val = attrs.get(key)
#             if val:
#                 group_name = val.split()[0] if key == "name" else val
#                 break
        
#         # If we found a group name, add to buckets for subcategory creation
#         if group_name:
#             buckets[(cat_id, group_name)].append(n)
#         else:
#             # Direct connection to category
#             if not G.has_edge(cat_id, n):
#                 G.add_edge(cat_id, n, type=CAT_REL)
    
#     # Create subcategories for grouped items
#     for (cat_id, group), members in buckets.items():
#         if len(members) == 1:
#             # If only one member, connect directly to category
#             if not G.has_edge(cat_id, members[0]):
#                 G.add_edge(cat_id, members[0], type=CAT_REL)
#         else:
#             # Create subcategory for multiple members
#             sub_id = f"{cat_id}::{group.lower().replace(' ', '_')}"
#             if sub_id not in G:
#                 G.add_node(sub_id, name=group, type="Subcategory")
#                 G.add_edge(cat_id, sub_id, type=CAT_REL)
            
#             # Connect all members to the subcategory
#             for m in members:
#                 if not G.has_edge(sub_id, m):
#                     G.add_edge(sub_id, m, type=CAT_REL)
    
#     return G


def validate_and_fix_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Validate and fix common issues in the graph
    """
    print("Validating and fixing graph...")
    
    # Fix nodes without required attributes
    nodes_fixed = 0
    for node_id, attrs in list(G.nodes(data=True)):
        if 'type' not in attrs:
            # Try to infer type from node ID or set default
            if node_id == ROOT_ID:
                attrs['type'] = 'Company'
            elif node_id.startswith('cat::'):
                attrs['type'] = 'Category'
            else:
                attrs['type'] = 'Unknown'
            nodes_fixed += 1
            
        if 'name' not in attrs:
            attrs['name'] = node_id.replace('_', ' ').title()
            nodes_fixed += 1
    
    if nodes_fixed > 0:
        print(f"Fixed {nodes_fixed} node attributes")
    
    # Validate relationship types
    edges_fixed = 0
    for u, v, attrs in list(G.edges(data=True)):
        if 'type' not in attrs:
            attrs['type'] = 'Unknown'
            edges_fixed += 1
    
    if edges_fixed > 0:
        print(f"Fixed {edges_fixed} edge attributes")
    
    print("Graph validation complete")
    return G


def ensure_hierarchy(G: nx.DiGraph) -> nx.DiGraph:
    """
    Ensure the graph has a proper solar system hierarchy:
    - Root (Apple Inc) at center
    - Only category nodes connected to root
    - All entities connected to their respective categories
    """
    print("Building solar system hierarchy...")
    
    # First, ensure ROOT_ID exists
    if ROOT_ID not in G:
        G.add_node(ROOT_ID, type="Company", name="Apple Inc.")
    
    # Create category nodes
    for cat_id, label in MACRO_CATEGORIES.items():
        if cat_id not in G:
            G.add_node(cat_id, type="Category", name=label)
    
    edges_to_remove = []
    for successor in list(G.successors(ROOT_ID)):
        edges_to_remove.append((ROOT_ID, successor))
    for predecessor in list(G.predecessors(ROOT_ID)):
        edges_to_remove.append((predecessor, ROOT_ID))
    
    for edge in edges_to_remove:
        if G.has_edge(edge[0], edge[1]):
            G.remove_edge(edge[0], edge[1])
    
    print(f"Removed {len(edges_to_remove)} edges from root node")
    
    for cat_id in MACRO_CATEGORIES.keys():
        if not G.has_edge(ROOT_ID, cat_id):
            G.add_edge(ROOT_ID, cat_id, type=CAT_REL)
    
    buckets = defaultdict(list)
    direct_category_assignments = defaultdict(list)
    
    for n, attrs in list(G.nodes(data=True)):
        # Skip root, category nodes, and subcategory nodes
        if (n == ROOT_ID or 
            n.startswith("cat::") or 
            attrs.get("type") == "Category" or 
            attrs.get("type") == "Subcategory"):
            continue
        
        # Get node type, default to "Other" if missing
        node_type = attrs.get('type', 'Other')
        
        # Map node type to category
        cat_id = f"cat::{node_type}"
        if cat_id not in MACRO_CATEGORIES:
            cat_id = "cat::Other"
            # Ensure "Other" category exists
            if cat_id not in MACRO_CATEGORIES:
                MACRO_CATEGORIES[cat_id] = "Other"
                if cat_id not in G:
                    G.add_node(cat_id, type="Category", name="Other")
                if not G.has_edge(ROOT_ID, cat_id):
                    G.add_edge(ROOT_ID, cat_id, type=CAT_REL)
        
        # Try to find a grouping key for subcategories
        group_name = None
        group_keys = GROUP_KEYS.get(node_type, [])
        
        for key in group_keys:
            val = attrs.get(key)
            if val:
                group_name = val.split()[0] if key == "name" else val
                break
        
        # If we found a group name, add to buckets for subcategory creation
        if group_name:
            buckets[(cat_id, group_name)].append(n)
        else:
            # Direct assignment to category
            direct_category_assignments[cat_id].append(n)
    

    entities_to_clean = []
    for bucket_list in buckets.values():
        entities_to_clean.extend(bucket_list)
    for direct_list in direct_category_assignments.values():
        entities_to_clean.extend(direct_list)
    
    edges_cleaned = 0
    for entity in entities_to_clean:
        for cat_id in list(MACRO_CATEGORIES.keys()) + [ROOT_ID]:
            if G.has_edge(entity, cat_id):
                G.remove_edge(entity, cat_id)
                edges_cleaned += 1
            if G.has_edge(cat_id, entity):
                G.remove_edge(cat_id, entity)
                edges_cleaned += 1
    
    print(f"Cleaned {edges_cleaned} inappropriate category/root connections")
    
    for cat_id, entities in direct_category_assignments.items():
        for entity in entities:
            if not G.has_edge(cat_id, entity):
                G.add_edge(cat_id, entity, type=CAT_REL)
    
    subcategories_created = 0
    for (cat_id, group), members in buckets.items():
        if len(members) == 1:
            # If only one member, connect directly to category
            if not G.has_edge(cat_id, members[0]):
                G.add_edge(cat_id, members[0], type=CAT_REL)
        else:
            # Create subcategory for multiple members
            sub_id = f"{cat_id}::{group.lower().replace(' ', '_')}"
            if sub_id not in G:
                G.add_node(sub_id, name=group, type="Subcategory")
                subcategories_created += 1
            
            # Connect subcategory to main category
            if not G.has_edge(cat_id, sub_id):
                G.add_edge(cat_id, sub_id, type=CAT_REL)
            
            # Connect all members to the subcategory
            for m in members:
                if not G.has_edge(sub_id, m):
                    G.add_edge(sub_id, m, type=CAT_REL)
    
    print(f"Created {subcategories_created} subcategories")
    
    root_connections = list(G.successors(ROOT_ID))
    non_category_connections = [
        conn for conn in root_connections 
        if not conn.startswith("cat::") and G.nodes[conn].get("type") != "Category"
    ]
    
    if non_category_connections:
        print(f"WARNING: Found {len(non_category_connections)} non-category connections to root: {non_category_connections}")
        # Remove these inappropriate connections
        for conn in non_category_connections:
            G.remove_edge(ROOT_ID, conn)
        print("Removed inappropriate root connections")
    
    print(f"Hierarchy complete: Root -> {len(list(G.successors(ROOT_ID)))} categories -> entities")
    return G


def validate_hierarchy_structure(G: nx.DiGraph) -> bool:
    """
    Validate that the graph has the correct solar system structure
    """
    print("\n=== HIERARCHY VALIDATION ===")
    
    # Check 1: Root exists
    if ROOT_ID not in G:
        print("❌ Root node missing")
        return False
    
    # Check 2: Root only connected to categories
    root_successors = list(G.successors(ROOT_ID))
    non_category_successors = [
        s for s in root_successors 
        if not s.startswith("cat::") and G.nodes[s].get("type") != "Category"
    ]
    
    if non_category_successors:
        print(f"❌ Root connected to non-categories: {non_category_successors}")
        return False
    else:
        print(f"✅ Root connected only to {len(root_successors)} categories")
    
    # Check 3: All categories connected to root
    category_nodes = [n for n in G.nodes() if n.startswith("cat::") or G.nodes[n].get("type") == "Category"]
    unconnected_categories = [cat for cat in category_nodes if not G.has_edge(ROOT_ID, cat)]
    
    if unconnected_categories:
        print(f"❌ Categories not connected to root: {unconnected_categories}")
        return False
    else:
        print(f"✅ All {len(category_nodes)} categories connected to root")
    
    # Check 4: No entities directly connected to root
    all_entities = [
        n for n in G.nodes() 
        if (n != ROOT_ID and 
            not n.startswith("cat::") and 
            G.nodes[n].get("type") not in ["Category", "Subcategory"])
    ]
    
    entities_connected_to_root = [
        e for e in all_entities 
        if G.has_edge(ROOT_ID, e) or G.has_edge(e, ROOT_ID)
    ]
    
    if entities_connected_to_root:
        print(f"❌ Entities directly connected to root: {entities_connected_to_root}")
        return False
    else:
        print(f"✅ No entities directly connected to root (found {len(all_entities)} entities)")
    
    print("✅ Hierarchy structure is valid!")
    return True


if __name__ == "__main__":
    print("Starting Apple Knowledge Graph creation...")
    
    documents = load_documents(file_paths)
    print(f"Loaded {len(documents)} documents.")
    
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    
    knowledge_graph = process_all_chunks(chunks)
    
    G = create_networkx_graph(knowledge_graph)

    G = ensure_hierarchy(G)
    
    validate_graph_relationships(G)
    
    save_networkx_graph(G, "apple_knowledge_graph")
    