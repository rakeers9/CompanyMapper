import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

#APIKEY: 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
os.environ["OPENAI_API_KEY"] = "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"

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
        
        
def split_documents(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
    )
    return text_splitter.split_documents(documents)

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
        for rel in extracted_data.get("relationships", []):
            if not isinstance(rel, dict):
                print(f"Warning: Expected relationship to be a dict, got {type(rel)}")
                continue
                
            if "source" not in rel or "target" not in rel or "type" not in rel:
                print(f"Warning: Relationship missing required fields: {rel}")
                continue
                
            if "properties" not in rel or not isinstance(rel["properties"], dict):
                rel["properties"] = {}
                
            valid_relationships.append(rel)
            
        return {
            "entities": valid_entities,
            "relationships": valid_relationships
        }
        
    except json.JSONDecodeError:
        print(f"Error parsing LLM output as JSON: {result}")
        return {"entities": [], "relationships": []}

    
def process_all_chunks(chunks: List[Document]):
    print(f"Processing {len(chunks)} chunks")
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
            if not isinstance(rel, dict) or "source" not in rel or "target" not in rel:
                print(f"Warning: Skipping invalid relationship: {rel}")
                continue
                
            all_relationships.append(rel)
    
    entities_list = list(all_entities.values())
    
    return {
        "entities": entities_list,
        "relationships": all_relationships
    }
        

def create_networkx_graph(knowledge_graph: Dict[str, Any]) -> nx.DiGraph:
    """
    Create a NetworkX directed graph from the knowledge graph data.
    """
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

# Main execution
if __name__ == "__main__":
    # File paths to documents
    file_paths = [
        "/Users/sreekargudipati/Coding Projects/CompanyMapper/documents/apple_10K_2024_annualreport.pdf",
        "/Users/sreekargudipati/Coding Projects/CompanyMapper/documents/Apple_Environmental_Progress_Report_2024.pdf",
        "/Users/sreekargudipati/Coding Projects/CompanyMapper/documents/Apple-Supply-Chain-2025-Progress-Report.pdf"
    ]
    
    documents = load_documents(file_paths)
    chunks = split_documents(documents)
    knowledge_graph = process_all_chunks(chunks)
    
    with open("apple_extracted_data.json", "w") as f:
        json.dump(knowledge_graph, f, indent=2)
    
    G = create_networkx_graph(knowledge_graph)
    
    print(f"Knowledge graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    