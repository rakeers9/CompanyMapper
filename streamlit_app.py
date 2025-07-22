# streamlit_app.py - Complete Enhanced Streamlit App with Research Agent
"""
Complete Apple Knowledge Graph System with AI Research Agent
- Solar system knowledge graph visualization
- Real-time news monitoring and processing
- AI-powered relevance detection with LangGraph
- Automatic research agent for high-priority articles
- Executive impact summaries and recommendations
"""

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import pickle
import math
from datetime import datetime, timedelta
import time
from pathlib import Path
from typing import List, Dict, Any, Generator
import os

# Import knowledge graph functions
from knowledge_graph import (
    ensure_hierarchy, 
    validate_and_fix_graph, 
    validate_hierarchy_structure
)

# Import news modules
try:
    from news_pipeline import NewsPipeline, NewsDatabase
    from relevance_detection import RelevanceMonitor, EntityMatcher
    from research_agent import ResearchAgent, ResearchDatabase, ResearchResult
    from langchain_openai import OpenAIEmbeddings
    NEWS_MODULES_AVAILABLE = True
except ImportError:
    NEWS_MODULES_AVAILABLE = False

ROOT_ID  = "apple"
CAT_REL = "Belongs_To"

# Set page config
st.set_page_config(
    page_title="Apple Knowledge Graph System",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for graph data
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'news_updates' not in st.session_state:
    st.session_state.news_updates = []

# Initialize session state for news
if 'news_pipeline' not in st.session_state:
    st.session_state.news_pipeline = None
if 'relevance_monitor' not in st.session_state:
    st.session_state.relevance_monitor = None
if 'last_news_update' not in st.session_state:
    st.session_state.last_news_update = None

# Initialize session state for research agent
if 'research_agent' not in st.session_state:
    st.session_state.research_agent = None
if 'research_database' not in st.session_state:
    st.session_state.research_database = None
if 'research_results' not in st.session_state:
    st.session_state.research_results = []

class GraphManager:
    """Manages graph operations with persistence and real-time updates"""
    
    def __init__(self, graph_file="apple_knowledge_graph.gpickle"):
        self.graph_file = graph_file
        
    def load_graph(self):
        """Load graph from file or create sample if not found"""
        import pickle
        
        try:
            # Try to load the graph using pickle directly (not nx.read_gpickle)
            if Path(self.graph_file).exists():
                with open(self.graph_file, 'rb') as f:
                    G = pickle.load(f)
                st.success(f"✅ Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            else:
                G = self.create_sample_graph()
                st.info("📝 Graph file not found - created sample graph. Upload your graph file via the sidebar.")
            return G
        except Exception as e:
            st.error(f"❌ Error loading graph: {e}")
            st.info("📝 Creating sample graph instead")
            return self.create_sample_graph()
            
    def save_graph(self, G):
        """Save graph to file"""
        try:
            nx.write_gpickle(G, self.graph_file)
            st.session_state.last_update = datetime.now()
            return True
        except Exception as e:
            st.error(f"Error saving graph: {e}")
            return False
        
    def create_sample_graph(self):
        """Create sample graph for demonstration"""
        G = nx.DiGraph()
        
        # Sample nodes with more detailed attributes
        nodes = [
            ("apple", {
                "name": "Apple Inc.", 
                "type": "Company", 
                "prop_description": "Technology company",
                "prop_risk_level": "Low",
                "created": datetime.now().isoformat()
            }),
            ("iphone", {
                "name": "iPhone", 
                "type": "Product", 
                "prop_description": "Smartphone product line",
                "created": datetime.now().isoformat()
            }),
            ("foxconn", {
                "name": "Foxconn", 
                "type": "Supplier", 
                "prop_description": "Manufacturing partner",
                "prop_risk_level": "Medium",
                "created": datetime.now().isoformat()
            }),
            ("china", {
                "name": "China", 
                "type": "Location", 
                "prop_description": "Manufacturing location",
                "created": datetime.now().isoformat()
            }),
        ]
        
        for node_id, attrs in nodes:
            G.add_node(node_id, **attrs)
        
        # Sample edges
        edges = [
            ("apple", "iphone", {"type": "Designs", "created": datetime.now().isoformat()}),
            ("foxconn", "apple", {"type": "Supplies", "created": datetime.now().isoformat()}),
            ("foxconn", "china", {"type": "Operates_In", "created": datetime.now().isoformat()}),
        ]
        
        for source, target, attrs in edges:
            G.add_edge(source, target, **attrs)
        
        return G

    def add_node(self, G, node_id, name, node_type, description="", **kwargs):
        """Add a new node to the graph"""
        if node_id in G:
            return False, f"Node {node_id} already exists"
        
        attrs = {
            "name": name,
            "type": node_type,
            "prop_description": description,
            "created": datetime.now().isoformat(),
            **kwargs
        }
        
        G.add_node(node_id, **attrs)
        return True, f"Added node: {name}"
    
    def add_edge(self, G, source, target, relationship_type, **kwargs):
        """Add a new edge to the graph"""
        if source not in G or target not in G:
            return False, "Both source and target nodes must exist"
        
        if G.has_edge(source, target):
            return False, f"Edge from {source} to {target} already exists"
        
        attrs = {
            "type": relationship_type,
            "created": datetime.now().isoformat(),
            **kwargs
        }
        
        G.add_edge(source, target, **attrs)
        return True, f"Added relationship: {source} → {target} ({relationship_type})"
    
    def remove_node(self, G, node_id):
        """Remove a node from the graph"""
        if node_id not in G:
            return False, f"Node {node_id} does not exist"
        
        G.remove_node(node_id)
        return True, f"Removed node: {node_id}"
    
    def remove_edge(self, G, source, target):
        """Remove an edge from the graph"""
        if not G.has_edge(source, target):
            return False, f"Edge from {source} to {target} does not exist"
        
        G.remove_edge(source, target)
        return True, f"Removed edge: {source} → {target}"

# Allowed relationship types
ALLOWED_RELATIONSHIP_TYPES = [
    "Supplies", "Manufactures", "Designs", "Distributes", "Operates_In", 
    "Owns", "Acquires", "Partners_With", "Licenses_From", "Regulated_By", 
    "Complies_With", "Uses_Material", "Generates_Emission", "Has_Initiative", 
    "Invests_In", "Provides_Service", "Faces_Litigation", "Receives_Certification", 
    "Reports_On", "Impacts"
]

def create_solar_system_layout(G, root_id=ROOT_ID):
    """Create a solar system layout with root at center and categories in orbit"""
    pos = {}
    
    if root_id not in G:
        # Fallback to spring layout if no root
        return nx.spring_layout(G, k=1, iterations=50)
    
    # Step 1: Place root at center
    pos[root_id] = (0, 0)
    
    # Step 2: Find categories (direct children of root)
    categories = [n for n in G.successors(root_id) if n.startswith("cat::")]
    
    if not categories:
        # Fallback if no categories found
        return nx.spring_layout(G, k=1, iterations=50)
    
    # Step 3: Arrange categories in a circle around root
    category_radius = 3
    angle_step = 2 * math.pi / len(categories)
    
    for i, cat in enumerate(categories):
        angle = i * angle_step
        x = category_radius * math.cos(angle)
        y = category_radius * math.sin(angle)
        pos[cat] = (x, y)
    
    # Step 4: For each category, arrange its children
    entity_radius = 6  # Distance from center for entities
    subcategory_radius = 4.5  # Distance for subcategories
    
    for i, cat in enumerate(categories):
        cat_angle = i * angle_step
        
        # Get direct children of this category
        children = list(G.successors(cat))
        
        if not children:
            continue
        
        # Separate subcategories from direct entities
        subcategories = [n for n in children if G.nodes[n].get("type") == "Subcategory"]
        direct_entities = [n for n in children if G.nodes[n].get("type") != "Subcategory"]
        
        # Calculate the angular span for this category's children
        total_children = len(subcategories) + len(direct_entities)
        
        # Add some spacing between category groups
        angular_span = min(angle_step * 0.8, math.pi / 2)  # Don't let it get too wide
        
        if total_children == 1:
            # Single child: place it directly radially outward from category
            child_angle = cat_angle
            child = children[0]
            if G.nodes[child].get("type") == "Subcategory":
                r = subcategory_radius
            else:
                r = entity_radius
            pos[child] = (r * math.cos(child_angle), r * math.sin(child_angle))
        else:
            # Multiple children: spread them in an arc
            start_angle = cat_angle - angular_span / 2
            angle_per_child = angular_span / (total_children - 1) if total_children > 1 else 0
            
            # First place subcategories
            child_index = 0
            for subcat in subcategories:
                child_angle = start_angle + child_index * angle_per_child
                x = subcategory_radius * math.cos(child_angle)
                y = subcategory_radius * math.sin(child_angle)
                pos[subcat] = (x, y)
                child_index += 1
                
                # Place children of subcategory
                subcat_children = list(G.successors(subcat))
                if subcat_children:
                    # Arrange subcategory children in a small arc around the subcategory
                    sub_angular_span = angular_span * 0.3  # Smaller span for subcategory children
                    sub_start_angle = child_angle - sub_angular_span / 2
                    sub_angle_per_child = (sub_angular_span / (len(subcat_children) - 1) 
                                         if len(subcat_children) > 1 else 0)
                    
                    for j, sub_child in enumerate(subcat_children):
                        sub_child_angle = sub_start_angle + j * sub_angle_per_child
                        sub_x = entity_radius * math.cos(sub_child_angle)
                        sub_y = entity_radius * math.sin(sub_child_angle)
                        pos[sub_child] = (sub_x, sub_y)
            
            # Then place direct entities
            for entity in direct_entities:
                child_angle = start_angle + child_index * angle_per_child
                x = entity_radius * math.cos(child_angle)
                y = entity_radius * math.sin(child_angle)
                pos[entity] = (x, y)
                child_index += 1
    
    # Step 5: Handle any remaining nodes not yet positioned
    unpositioned = [n for n in G.nodes() if n not in pos]
    if unpositioned:
        # Place them in outer orbit
        outer_radius = 8
        angle_step_outer = 2 * math.pi / len(unpositioned)
        for i, node in enumerate(unpositioned):
            angle = i * angle_step_outer
            x = outer_radius * math.cos(angle)
            y = outer_radius * math.sin(angle)
            pos[node] = (x, y)
    
    return pos

def create_network_plot(G, selected_types=None, search_term="", highlight_recent=False):
    """Create interactive network plot with solar system layout"""
    
    if G.number_of_nodes() == 0:
        return go.Figure().add_annotation(text="No nodes in graph", x=0.5, y=0.5, showarrow=False)
    
    # Filter graph
    filtered_nodes = list(G.nodes())
    if selected_types and "All" not in selected_types:
        filtered_nodes = [n for n in G.nodes() 
                         if G.nodes[n].get('type', 'Unknown') in selected_types]
    
    if search_term:
        filtered_nodes = [n for n in filtered_nodes 
                         if search_term.lower() in G.nodes[n].get('name', n).lower()]
    
    subG = G.subgraph(filtered_nodes)
    
    if len(subG.nodes()) == 0:
        return go.Figure().add_annotation(text="No nodes match filters", x=0.5, y=0.5, showarrow=False)
    
    # Use solar system layout
    pos = create_solar_system_layout(subG, ROOT_ID)
    
    # Create edges
    edge_x, edge_y = [], []
    for edge in subG.edges(data=True):
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    # Edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Create node traces by type with different styling for each level
    node_traces = []
    
    # Define colors and sizes for different node types
    type_styling = {
        'Company': {'color': '#FFD700', 'size': 40, 'symbol': 'star'},  # Gold star for root
        'Category': {'color': '#FF6B6B', 'size': 25, 'symbol': 'circle'},  # Red circles for categories
        'Subcategory': {'color': '#4ECDC4', 'size': 20, 'symbol': 'circle'},  # Teal for subcategories
        'Product': {'color': '#45B7D1', 'size': 15, 'symbol': 'circle'},  # Blue for products
        'Supplier': {'color': '#96CEB4', 'size': 15, 'symbol': 'square'},  # Green squares for suppliers
        'Material': {'color': '#FFEAA7', 'size': 15, 'symbol': 'diamond'},  # Yellow diamonds for materials
        'Location': {'color': '#DDA0DD', 'size': 15, 'symbol': 'triangle-up'},  # Purple triangles for locations
        'Unknown': {'color': '#95A5A6', 'size': 12, 'symbol': 'circle'},  # Gray for unknown
    }
    
    # Group nodes by type
    nodes_by_type = {}
    for n in subG.nodes():
        node_type = subG.nodes[n].get('type', 'Unknown')
        if node_type not in nodes_by_type:
            nodes_by_type[node_type] = []
        nodes_by_type[node_type].append(n)
    
    for node_type, type_nodes in nodes_by_type.items():
        if not type_nodes:
            continue
            
        styling = type_styling.get(node_type, type_styling['Unknown'])
        
        node_x = [pos[n][0] for n in type_nodes if n in pos]
        node_y = [pos[n][1] for n in type_nodes if n in pos]
        node_text = [subG.nodes[n].get('name', n) for n in type_nodes if n in pos]
        
        # Check for recent nodes
        node_sizes = []
        node_colors = []
        for n in type_nodes:
            if n not in pos:
                continue
                
            is_recent = False
            if highlight_recent and 'created' in subG.nodes[n]:
                try:
                    created_time = datetime.fromisoformat(subG.nodes[n]['created'])
                    is_recent = (datetime.now() - created_time).seconds < 3600
                except:
                    pass
            
            node_sizes.append(styling['size'] + 5 if is_recent else styling['size'])
            node_colors.append('red' if is_recent else styling['color'])
        
        # Create hover info
        node_info = []
        for n in type_nodes:
            if n not in pos:
                continue
            info = f"<b>{subG.nodes[n].get('name', n)}</b><br>"
            info += f"Type: {subG.nodes[n].get('type', 'Unknown')}<br>"
            info += f"Connections: {len(list(subG.neighbors(n)))}<br>"
            info += f"Description: {subG.nodes[n].get('prop_description', 'N/A')}<br>"
            if 'created' in subG.nodes[n]:
                info += f"Created: {subG.nodes[n]['created'][:19]}"
            node_info.append(info)
        
        # Create trace for this node type
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_info,
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                symbol=styling['symbol'],
                line=dict(width=2, color='white')
            ),
            name=node_type,
            textfont=dict(size=8, color='white')
        )
        node_traces.append(node_trace)
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title={
                'text': f'Apple Knowledge Graph - Solar System View ({len(subG.nodes())} nodes, {len(subG.edges())} edges)',
                'font': {'size': 16},
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[dict(
                text="🌟 Apple Inc at center | 🔴 Red = Recently added | Hover for details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='black',  # Dark background for space theme
            paper_bgcolor='black'
        )
    )
    
    return fig

def display_graph_editor(G, graph_manager):
    """Display graph editing interface"""
    st.subheader("✏️ Graph Editor")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Add Node", "Add Relationship", "Remove Items", "Import Data"])
    
    with tab1:
        st.write("**Add New Node**")
        col1, col2 = st.columns(2)
        
        with col1:
            new_node_id = st.text_input("Node ID (unique identifier):", key="new_node_id")
            new_node_name = st.text_input("Display Name:", key="new_node_name")
            new_node_type = st.selectbox("Node Type:", 
                options=["Company", "Product", "Supplier", "Location", "Component", "Service", "Other"],
                key="new_node_type")
        
        with col2:
            new_node_desc = st.text_area("Description:", key="new_node_desc")
            new_node_risk = st.selectbox("Risk Level:", 
                options=["Low", "Medium", "High", "Unknown"], 
                key="new_node_risk")
        
        if st.button("Add Node", type="primary"):
            if new_node_id and new_node_name:
                success, message = graph_manager.add_node(
                    G, new_node_id, new_node_name, new_node_type, 
                    new_node_desc, prop_risk_level=new_node_risk
                )
                if success:
                    graph_manager.save_graph(G)
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please provide both Node ID and Display Name")
    
    with tab2:
        st.write("**Add New Relationship**")
        
        if G.number_of_nodes() >= 2:
            nodes = list(G.nodes())
            node_names = {n: G.nodes[n].get('name', n) for n in nodes}
            
            col1, col2 = st.columns(2)
            
            with col1:
                source_node = st.selectbox(
                    "Source Node:", 
                    options=nodes,
                    format_func=lambda x: f"{node_names[x]} ({x})",
                    key="rel_source"
                )
                
                target_node = st.selectbox(
                    "Target Node:", 
                    options=nodes,
                    format_func=lambda x: f"{node_names[x]} ({x})",
                    key="rel_target"
                )
            
            with col2:
                rel_type = st.selectbox(
                    "Relationship Type:", 
                    options=ALLOWED_RELATIONSHIP_TYPES,
                    key="rel_type"
                )
                
                rel_strength = st.selectbox(
                    "Relationship Strength:",
                    options=["Critical", "Important", "Minor", "Unknown"],
                    key="rel_strength"
                )
            
            if st.button("Add Relationship", type="primary"):
                if source_node != target_node:
                    success, message = graph_manager.add_edge(
                        G, source_node, target_node, rel_type, 
                        prop_strength=rel_strength
                    )
                    if success:
                        graph_manager.save_graph(G)
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Source and target nodes must be different")
        else:
            st.info("Add at least 2 nodes before creating relationships")
    
    with tab3:
        st.write("**Remove Nodes or Relationships**")
        
        if G.number_of_nodes() > 0:
            removal_type = st.radio("What to remove:", ["Node", "Relationship"])
            
            if removal_type == "Node":
                nodes = list(G.nodes())
                node_names = {n: G.nodes[n].get('name', n) for n in nodes}
                
                node_to_remove = st.selectbox(
                    "Select node to remove:",
                    options=nodes,
                    format_func=lambda x: f"{node_names[x]} ({x})"
                )
                
                if st.button("Remove Node", type="secondary"):
                    success, message = graph_manager.remove_node(G, node_to_remove)
                    if success:
                        graph_manager.save_graph(G)
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            
            else:  # Remove relationship
                edges = list(G.edges())
                if edges:
                    edge_options = [(u, v) for u, v in edges]
                    edge_labels = [f"{G.nodes[u].get('name', u)} → {G.nodes[v].get('name', v)} ({G[u][v].get('type', 'Unknown')})" 
                                 for u, v in edges]
                    
                    selected_idx = st.selectbox(
                        "Select relationship to remove:",
                        options=range(len(edge_options)),
                        format_func=lambda x: edge_labels[x]
                    )
                    
                    if st.button("Remove Relationship", type="secondary"):
                        source, target = edge_options[selected_idx]
                        success, message = graph_manager.remove_edge(G, source, target)
                        if success:
                            graph_manager.save_graph(G)
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                else:
                    st.info("No relationships to remove")
        else:
            st.info("No nodes in graph")
    
    with tab4:
        st.write("**Import Data**")
        st.info("🚧 Ready for Milestone 4: News data import will be added here")
        
        # Placeholder for future news import functionality
        uploaded_file = st.file_uploader("Upload JSON data (for future use):", type=['json'])
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                st.json(data)
                st.info("File uploaded successfully - integration coming in Milestone 4")
            except:
                st.error("Invalid JSON file")

# Enhanced streaming functions
def get_streaming_articles(pipeline, max_articles: int) -> Generator[Dict[str, Any], None, None]:
    """Generator that yields articles one by one for streaming processing"""
    
    # Fetch articles from all sources
    print("Fetching articles for streaming...")
    
    # Get from RSS feeds
    rss_articles = pipeline.rss_client.fetch_all_feeds()
    all_articles = rss_articles
    
    # Get from NewsAPI if available
    if pipeline.newsapi_client:
        from datetime import timedelta
        from_date = datetime.now() - timedelta(days=3)
        
        for query in pipeline.apple_queries[:3]:  # Limit queries
            newsapi_articles = pipeline.newsapi_client.fetch_everything(
                query=query,
                from_date=from_date,
                page_size=20
            )
            
            # Convert NewsAPI format
            for article in newsapi_articles:
                formatted_article = {
                    "title": article.get("title", ""),
                    "content": article.get("description", "") or article.get("content", ""),
                    "url": article.get("url", ""),
                    "published_date": article.get("publishedAt", ""),
                    "author": article.get("author"),
                    "source": article.get("source", {}).get("name", "NewsAPI")
                }
                all_articles.append(formatted_article)
    
    # Deduplicate
    seen_urls = set()
    unique_articles = []
    
    for article in all_articles:
        url = article.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)
    
    # Yield articles one by one
    for article in unique_articles[:max_articles]:
        yield article

def display_streaming_article_with_research(
    container, 
    article, 
    relevance_match, 
    research_result,
    count: int, 
    show_low_relevance: bool, 
    stored_successfully: bool
):
    """Enhanced article display with research results"""
    
    # Skip low relevance articles if not showing them
    if not show_low_relevance and relevance_match.relevance_score < 2.0:
        return
    
    with container:
        # Determine styling based on research and relevance
        if research_result:
            if research_result.urgency_level == "Critical":
                border_color = "#ff0000"
                icon = "🚨"
            elif research_result.urgency_level == "High":
                border_color = "#ff4444"
                icon = "🔥"
            else:
                border_color = "#ff8800"
                icon = "🔬"
        elif relevance_match.relevance_category == "HIGH":
            border_color = "#ff4444"
            icon = "🔥"
        elif relevance_match.relevance_category == "MEDIUM":
            border_color = "#ffaa00"
            icon = "🟡"
        else:
            border_color = "#888888"
            icon = "⚪"
        
        # Create expandable article display with research
        title_suffix = f" - RESEARCH COMPLETE ({research_result.urgency_level})" if research_result else ""
        expanded = research_result is not None or relevance_match.relevance_category == "HIGH"
        
        with st.expander(
            f"{icon} Article #{count}: {article.title[:60]}...{title_suffix} (Score: {relevance_match.relevance_score:.2f})", 
            expanded=expanded
        ):
            
            # Main article info
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Title:** {article.title}")
                st.write(f"**Source:** {article.source}")
                st.write(f"**URL:** {article.url}")
                
                if article.content:
                    st.write(f"**Content Preview:** {article.content[:200]}...")
                
                if relevance_match.graph_entities:
                    st.write(f"**Matched Entities:** {', '.join(relevance_match.graph_entities[:5])}")
                
                if relevance_match.keyword_matches:
                    st.write(f"**Keywords:** {', '.join(relevance_match.keyword_matches[:5])}")
            
            with col2:
                st.metric("Relevance Score", f"{relevance_match.relevance_score:.2f}")
                st.metric("Category", relevance_match.relevance_category)
                st.metric("Entities Found", len(relevance_match.graph_entities))
                
                if research_result:
                    st.success("🔬 Research Complete")
                    st.metric("Research Time", f"{research_result.research_duration_seconds:.1f}s")
                elif relevance_match.processed_by_agent:
                    st.info("🔬 Research Triggered")
                
                if stored_successfully:
                    st.success("💾 Stored in DB")
                else:
                    st.warning("⚠️ Storage Failed")
                
                # Sentiment info
                if hasattr(article, 'sentiment_score') and article.sentiment_score:
                    sentiment_emoji = "😊" if article.sentiment_score > 0.1 else "😐" if article.sentiment_score > -0.1 else "😟"
                    st.write(f"**Sentiment:** {sentiment_emoji} {article.sentiment_score:.2f}")
            
            # Research Results Section
            if research_result:
                st.markdown("---")
                st.subheader("🔬 Research Analysis Results")
                
                # Executive Summary
                st.write("**Executive Summary:**")
                st.info(research_result.executive_summary)
                
                # Impact Analysis in columns
                impact_col1, impact_col2, impact_col3 = st.columns(3)
                
                with impact_col1:
                    st.write("**💰 Financial Impact**")
                    fin_impact = research_result.financial_impact
                    if fin_impact:
                        st.write(f"• Likelihood: {fin_impact.get('likelihood', 'Unknown')}")
                        st.write(f"• Severity: {fin_impact.get('severity', 'Unknown')}")
                        st.write(f"• Timeline: {fin_impact.get('timeline', 'Unknown')}")
                        if fin_impact.get('details'):
                            st.write(f"• Details: {fin_impact['details'][:100]}...")
                
                with impact_col2:
                    st.write("**⚙️ Operational Impact**")
                    op_impact = research_result.operational_impact
                    if op_impact:
                        st.write(f"• Likelihood: {op_impact.get('likelihood', 'Unknown')}")
                        st.write(f"• Severity: {op_impact.get('severity', 'Unknown')}")
                        st.write(f"• Timeline: {op_impact.get('timeline', 'Unknown')}")
                        if op_impact.get('details'):
                            st.write(f"• Details: {op_impact['details'][:100]}...")
                
                with impact_col3:
                    st.write("**🎭 Reputational Impact**")
                    rep_impact = research_result.reputational_impact
                    if rep_impact:
                        st.write(f"• Likelihood: {rep_impact.get('likelihood', 'Unknown')}")
                        st.write(f"• Severity: {rep_impact.get('severity', 'Unknown')}")
                        st.write(f"• Timeline: {rep_impact.get('timeline', 'Unknown')}")
                        if rep_impact.get('details'):
                            st.write(f"• Details: {rep_impact['details'][:100]}...")
                
                # Key Takeaways and Actions
                if research_result.key_takeaways or research_result.recommended_actions:
                    action_col1, action_col2 = st.columns(2)
                    
                    with action_col1:
                        if research_result.key_takeaways:
                            st.write("**📋 Key Takeaways:**")
                            for takeaway in research_result.key_takeaways[:3]:
                                st.write(f"• {takeaway}")
                    
                    with action_col2:
                        if research_result.recommended_actions:
                            st.write("**🎯 Recommended Actions:**")
                            for action in research_result.recommended_actions[:3]:
                                st.write(f"• {action}")
                
                # Stock Context (if available)
                if research_result.stock_price_context and research_result.stock_price_context.get('current_price'):
                    stock_data = research_result.stock_price_context
                    st.write("**📈 Stock Context:**")
                    change_color = "green" if stock_data.get('change', 0) >= 0 else "red"
                    st.markdown(
                        f"AAPL: ${stock_data.get('current_price', 'N/A')} "
                        f"<span style='color: {change_color}'>({stock_data.get('change_percent', 0):.2f}%)</span>",
                        unsafe_allow_html=True
                    )
        
        # Add separator
        st.markdown("---")

def display_streaming_error(container, article_data, count: int, error_message: str):
    """Display an error in the streaming interface"""
    
    with container:
        with st.expander(f"❌ Article #{count}: ERROR processing {article_data.get('title', 'Unknown')[:50]}..."):
            st.error(f"**Error:** {error_message}")
            st.write(f"**Title:** {article_data.get('title', 'Unknown')}")
            st.write(f"**Source:** {article_data.get('source', 'Unknown')}")
            st.write(f"**URL:** {article_data.get('url', 'Unknown')}")
        
        st.markdown("---")

def run_enhanced_streaming_pipeline(
    G, 
    max_articles: int, 
    days_back: int, 
    min_relevance: float, 
    openai_api_key: str, 
    show_low_relevance: bool, 
    auto_scroll: bool, 
    stream_delay: float,
    enable_research: bool = True,
    alpha_vantage_key: str = None
):
    """Enhanced streaming pipeline with real-time research"""
    
    # Initialize systems if needed
    if not st.session_state.news_pipeline:
        st.session_state.news_pipeline = NewsPipeline(
            openai_api_key=openai_api_key,
            db_path="./apple_news_db"
        )
    
    if not st.session_state.relevance_monitor:
        st.session_state.relevance_monitor = RelevanceMonitor(
            openai_api_key=openai_api_key,
            news_db_path="./apple_news_db"
        )
    
    # Initialize research agent
    if enable_research and not st.session_state.research_agent:
        st.session_state.research_agent = ResearchAgent(
            openai_api_key=openai_api_key,
            alpha_vantage_key=alpha_vantage_key
        )
        st.session_state.research_database = ResearchDatabase("./research_results.json")
    
    # Create containers
    st.markdown("---")
    st.header("🌊 Enhanced Live News Stream with AI Research")
    
    status_container = st.container()
    progress_container = st.container()
    stream_container = st.container()
    
    with status_container:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Initialize metrics
        total_processed = st.empty()
        high_relevance_count = st.empty()
        research_triggered_count = st.empty()
        research_completed_count = st.empty()
        processing_rate = st.empty()
        
        with col1:
            total_processed.metric("📊 Processed", "0")
        with col2:
            high_relevance_count.metric("🔥 High Relevance", "0")
        with col3:
            research_triggered_count.metric("🔬 Research Triggered", "0")
        with col4:
            research_completed_count.metric("✅ Research Complete", "0")
        with col5:
            processing_rate.metric("⚡ Articles/min", "0")
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Start streaming process
    start_time = datetime.now()
    processed_count = 0
    high_relevance_articles = 0
    research_triggered_articles = 0
    research_completed_articles = 0
    
    try:
        # Get the streaming generator
        article_stream = get_streaming_articles(st.session_state.news_pipeline, max_articles)
        
        # Process articles one by one
        for article_data in article_stream:
            processed_count += 1
            
            # Update progress
            progress = min(processed_count / max_articles, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing article {processed_count}/{max_articles}: {article_data['title'][:50]}...")
            
            research_result = None
            
            try:
                # Process the article
                processed_article = st.session_state.news_pipeline.processor.process_article(article_data)
                stored_successfully = st.session_state.news_pipeline.database.add_article(processed_article)
                
                # Create embedding if missing
                if not processed_article.embedding:
                    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                    processed_article.embedding = embeddings.embed_query(processed_article.cleaned_content)
                
                # Get graph entities
                matcher = EntityMatcher(openai_api_key)
                graph_entities = matcher.extract_graph_entities(G)
                
                # Process through relevance workflow
                relevance_match = st.session_state.relevance_monitor.workflow.process_article(
                    processed_article, graph_entities
                )
                
                # Update relevance monitor
                if relevance_match.relevance_score >= min_relevance:
                    st.session_state.relevance_monitor.relevance_matches.append(relevance_match)
                    high_relevance_articles += 1
                
                # Trigger research if enabled and conditions met
                if (enable_research and 
                    st.session_state.research_agent and 
                    relevance_match.processed_by_agent and
                    relevance_match.relevance_score >= 4.0):
                    
                    research_triggered_articles += 1
                    
                    # Update status to show research in progress
                    status_text.text(f"🔬 RESEARCHING: {processed_article.title[:50]}...")
                    
                    # Perform research immediately
                    research_result = st.session_state.research_agent.research_article(
                        processed_article, relevance_match
                    )
                    
                    # Save research result
                    st.session_state.research_database.save_result(research_result)
                    st.session_state.research_results.append(research_result)
                    research_completed_articles += 1
                    
                    # Update the relevance match with research info
                    relevance_match.agent_analysis = research_result.executive_summary
                
                # Display article in stream (with research results if available)
                display_streaming_article_with_research(
                    stream_container, 
                    processed_article, 
                    relevance_match, 
                    research_result,
                    processed_count, 
                    show_low_relevance, 
                    stored_successfully
                )
                
            except Exception as e:
                # Display error in stream
                display_streaming_error(stream_container, article_data, processed_count, str(e))
            
            # Update metrics
            elapsed_time = (datetime.now() - start_time).total_seconds()
            rate = (processed_count / elapsed_time) * 60 if elapsed_time > 0 else 0
            
            total_processed.metric("📊 Processed", str(processed_count))
            high_relevance_count.metric("🔥 High Relevance", str(high_relevance_articles))
            research_triggered_count.metric("🔬 Research Triggered", str(research_triggered_articles))
            research_completed_count.metric("✅ Research Complete", str(research_completed_articles))
            processing_rate.metric("⚡ Articles/min", f"{rate:.1f}")
            
            # Auto-scroll to bottom if enabled
            if auto_scroll:
                st.markdown('<div id="bottom"></div>', unsafe_allow_html=True)
                st.markdown('<script>document.getElementById("bottom").scrollIntoView();</script>', 
                           unsafe_allow_html=True)
            
            # Add delay between articles
            time.sleep(stream_delay)
            
            # Check if user wants to stop
            if processed_count >= max_articles:
                break
    
    except Exception as e:
        st.error(f"Streaming pipeline error: {e}")
    
    finally:
        # Save all data
        st.session_state.relevance_monitor._save_relevance_matches()
        
        # Final status
        progress_bar.progress(1.0)
        status_text.text(f"✅ Streaming complete! Processed {processed_count} articles")
        
        # Update session state
        st.session_state.last_news_update = datetime.now()
        
        # Success message with research stats
        if research_completed_articles > 0:
            st.success(
                f"🎉 Enhanced streaming pipeline completed! "
                f"Processed {processed_count} articles with {high_relevance_articles} high-relevance matches "
                f"and completed {research_completed_articles} research analyses."
            )
        else:
            st.success(
                f"🎉 Streaming pipeline completed! "
                f"Processed {processed_count} articles with {high_relevance_articles} high-relevance matches."
            )

# News pipeline functions
def run_news_pipeline(max_articles: int, openai_api_key: str):
    """Run the news pipeline"""
    with st.spinner(f"Fetching and processing up to {max_articles} articles..."):
        try:
            # Initialize pipeline if needed
            if not st.session_state.news_pipeline:
                st.session_state.news_pipeline = NewsPipeline(
                    openai_api_key=openai_api_key,
                    db_path="./apple_news_db"
                )
            
            # Run pipeline
            result = st.session_state.news_pipeline.run_pipeline(max_articles)
            
            # Update session state
            st.session_state.last_news_update = datetime.now()
            
            # Display results
            st.success(f"✅ Pipeline complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Articles Fetched", result['articles_fetched'])
            with col2:
                st.metric("Articles Processed", result['articles_processed'])
            with col3:
                st.metric("Success Rate", f"{result['success_rate']:.1%}")
            
            st.info(f"Processing time: {result['duration_seconds']:.1f} seconds")
            
        except Exception as e:
            st.error(f"Error running pipeline: {e}")

def analyze_relevance(G, days_back: int, min_relevance: float, openai_api_key: str):
    """Analyze news relevance"""
    with st.spinner(f"Analyzing relevance for articles from last {days_back} days..."):
        try:
            # Initialize monitor if needed
            if not st.session_state.relevance_monitor:
                st.session_state.relevance_monitor = RelevanceMonitor(
                    openai_api_key=openai_api_key,
                    news_db_path="./apple_news_db"
                )
            
            # Process recent news
            new_matches = st.session_state.relevance_monitor.process_recent_news(
                G, 
                days_back=days_back, 
                min_relevance_score=min_relevance
            )
            
            # Display results
            st.success(f"✅ Relevance analysis complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("New High-Relevance Matches", len(new_matches))
            with col2:
                high_priority = sum(1 for m in new_matches if m.relevance_category == "HIGH")
                st.metric("High Priority", high_priority)
            with col3:
                research_triggered = sum(1 for m in new_matches if m.processed_by_agent)
                st.metric("Research Triggered", research_triggered)
            
            # Show top matches
            if new_matches:
                st.subheader("🔥 New High-Relevance Matches")
                for match in new_matches[:3]:
                    with st.expander(f"{match.news_title} (Score: {match.relevance_score:.2f})"):
                        st.write(f"**Entities:** {', '.join(match.graph_entities)}")
                        st.write(f"**Preview:** {match.news_content_preview}")
                        st.markdown(f"[Read Article]({match.news_url})")
            
        except Exception as e:
            st.error(f"Error analyzing relevance: {e}")

def display_news_overview():
    """Display news system overview and status"""
    st.subheader("📊 System Overview")
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    
    # Get database stats
    try:
        if st.session_state.news_pipeline:
            total_articles = st.session_state.news_pipeline.database.get_article_count()
        else:
            # Try to get count directly
            if NEWS_MODULES_AVAILABLE:
                db = NewsDatabase("./apple_news_db")
                total_articles = db.get_article_count()
            else:
                total_articles = 0
    except:
        total_articles = 0
    
    try:
        if st.session_state.relevance_monitor:
            total_matches = len(st.session_state.relevance_monitor.relevance_matches)
            high_relevance = len(st.session_state.relevance_monitor.get_matches_by_category("HIGH"))
        else:
            total_matches = 0
            high_relevance = 0
    except:
        total_matches = 0
        high_relevance = 0
    
    with col1:
        st.metric("📰 Total Articles", total_articles)
    
    with col2:
        st.metric("🎯 Relevance Matches", total_matches)
    
    with col3:
        st.metric("🔥 High Priority", high_relevance)
    
    with col4:
        last_update = st.session_state.get('last_news_update', 'Never')
        if isinstance(last_update, datetime):
            last_update = last_update.strftime("%H:%M")
        elif isinstance(last_update, str) and last_update != 'Never':
            try:
                dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                last_update = dt.strftime("%H:%M")
            except:
                last_update = "Recent"
        st.metric("🕐 Last Update", last_update)
    
    # Recent activity
    st.subheader("🕐 Recent Activity")
    
    if st.session_state.relevance_monitor:
        try:
            recent_matches = sorted(
                st.session_state.relevance_monitor.relevance_matches,
                key=lambda x: x.created_at if isinstance(x.created_at, datetime) else datetime.fromisoformat(x.created_at),
                reverse=True
            )[:5]
            
            if recent_matches:
                for match in recent_matches:
                    with st.expander(f"🎯 {match.news_title[:80]}... ({match.relevance_category})"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Score:** {match.relevance_score:.2f}")
                            st.write(f"**Entities:** {', '.join(match.graph_entities[:3])}")
                            st.write(f"**Preview:** {match.news_content_preview}")
                            
                        with col2:
                            st.write(f"**Category:** {match.relevance_category}")
                            
                            try:
                                if isinstance(match.created_at, datetime):
                                    time_str = match.created_at.strftime('%H:%M')
                                elif isinstance(match.created_at, str):
                                    dt = datetime.fromisoformat(match.created_at.replace('Z', '+00:00'))
                                    time_str = dt.strftime('%H:%M')
                                else:
                                    time_str = "Unknown"
                            except:
                                time_str = "Unknown"
                            
                            st.write(f"**Time:** {time_str}")
                            
                            if st.button("🔗 View Article", key=f"view_{match.news_article_id}"):
                                st.markdown(f"[Open Article]({match.news_url})")
            else:
                st.info("No recent activity. Run the news pipeline to see updates here.")
        except Exception as e:
            st.error(f"Error loading recent activity: {e}")
            st.info("This might be due to data format issues. Try reinitializing the news system.")
    else:
        st.info("Initialize the news system to see recent activity.")

def display_enhanced_pipeline_control(G, openai_api_key: str):
    """Enhanced pipeline control with research options"""
    st.subheader("🔄 Enhanced News Pipeline Control")
    
    # API Key configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Pipeline Settings**")
        max_articles = st.slider("Max articles to process:", 10, 200, 50)
        min_relevance = st.slider("Minimum relevance score:", 0.0, 10.0, 2.0, 0.5)
        days_back = st.selectbox("Look back (days):", [1, 3, 7, 14], index=2)
    
    with col2:
        st.write("**Research Settings**")
        enable_research = st.checkbox("Enable AI Research Agent", value=True, help="Automatically research high-priority articles")
        alpha_vantage_key = st.text_input("Alpha Vantage API Key (optional):", type="password", help="For real financial data")
        research_threshold = st.slider("Research trigger score:", 3.0, 10.0, 4.0, 0.5)
    
    # Streaming options
    st.write("**Streaming Options**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_low_relevance = st.checkbox("Show low relevance", value=True)
    with col2:
        auto_scroll = st.checkbox("Auto-scroll", value=True)
    with col3:
        stream_delay = st.slider("Delay (s):", 0.1, 3.0, 1.0, 0.1)
    with col4:
        show_research_details = st.checkbox("Expand research", value=True)
    
    # Pipeline actions
    st.write("**Actions**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Run News Pipeline", type="primary"):
            run_news_pipeline(max_articles, openai_api_key)
    
    with col2:
        if st.button("🎯 Analyze Relevance"):
            if G is not None:
                analyze_relevance(G, days_back, min_relevance, openai_api_key)
            else:
                st.error("Knowledge graph not available")
    
    with col3:
        if st.button("🌊🔬 Run Enhanced Stream", type="primary"):
            if G is not None:
                run_enhanced_streaming_pipeline(
                    G, max_articles, days_back, min_relevance, openai_api_key, 
                    show_low_relevance, auto_scroll, stream_delay, enable_research, alpha_vantage_key
                )
            else:
                st.error("Knowledge graph must be initialized")

def display_relevance_matches(G):
    """Display relevance matches interface"""
    st.subheader("🎯 Relevance Matches")
    
    if not st.session_state.relevance_monitor:
        st.info("Initialize the news system to see relevance matches.")
        return
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category_filter = st.selectbox(
            "Filter by category:",
            ["All", "HIGH", "MEDIUM", "LOW"]
        )
    
    with col2:
        entity_filter = st.text_input("Filter by entity:")
    
    with col3:
        sort_by = st.selectbox(
            "Sort by:",
            ["Relevance Score", "Date", "Title"]
        )
    
    # Get and filter matches
    try:
        all_matches = st.session_state.relevance_monitor.relevance_matches
        
        # Apply filters
        filtered_matches = all_matches
        
        if category_filter != "All":
            filtered_matches = [m for m in filtered_matches if m.relevance_category == category_filter]
        
        if entity_filter:
            filtered_matches = [
                m for m in filtered_matches 
                if entity_filter.lower() in " ".join(m.graph_entities).lower()
            ]
        
        # Sort matches with datetime handling
        if sort_by == "Relevance Score":
            filtered_matches.sort(key=lambda x: x.relevance_score, reverse=True)
        elif sort_by == "Date":
            def get_datetime(match):
                if isinstance(match.created_at, datetime):
                    return match.created_at
                elif isinstance(match.created_at, str):
                    try:
                        return datetime.fromisoformat(match.created_at.replace('Z', '+00:00'))
                    except:
                        return datetime.now()
                else:
                    return datetime.now()
            
            filtered_matches.sort(key=get_datetime, reverse=True)
        else:  # Title
            filtered_matches.sort(key=lambda x: x.news_title)
        
        # Display matches
        st.write(f"Found {len(filtered_matches)} matches")
        
        for i, match in enumerate(filtered_matches[:20]):  # Limit to 20 for performance
            # Create expandable match display
            score_color = "🔴" if match.relevance_category == "HIGH" else "🟡" if match.relevance_category == "MEDIUM" else "🟢"
            
            with st.expander(f"{score_color} {match.news_title} (Score: {match.relevance_score:.2f})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Content Preview:**")
                    st.write(match.news_content_preview)
                    
                    st.write(f"**Matched Entities:**")
                    if match.graph_entities:
                        entity_text = ", ".join(match.graph_entities[:5])
                        st.write(entity_text)
                    
                    # Show entity match details
                    if match.entity_matches:
                        with st.expander("📊 Entity Match Details"):
                            match_df = pd.DataFrame(match.entity_matches)
                            if not match_df.empty:
                                st.dataframe(
                                    match_df[['entity_name', 'entity_type', 'combined_score', 'semantic_similarity']],
                                    use_container_width=True
                                )
                
                with col2:
                    st.write(f"**Category:** {match.relevance_category}")
                    st.write(f"**Score:** {match.relevance_score:.2f}")
                    
                    # Handle datetime display safely
                    try:
                        if isinstance(match.created_at, datetime):
                            date_str = match.created_at.strftime('%Y-%m-%d %H:%M')
                        elif isinstance(match.created_at, str):
                            dt = datetime.fromisoformat(match.created_at.replace('Z', '+00:00'))
                            date_str = dt.strftime('%Y-%m-%d %H:%M')
                        else:
                            date_str = "Unknown"
                    except:
                        date_str = "Unknown"
                    
                    st.write(f"**Date:** {date_str}")
                    st.write(f"**Research:** {'Yes' if match.processed_by_agent else 'No'}")
                    
                    if st.button("🔗 Open Article", key=f"open_{match.news_article_id}"):
                        st.markdown(f"[Open in new tab]({match.news_url})")
        
        # Pagination for large result sets
        if len(filtered_matches) > 20:
            st.info(f"Showing first 20 of {len(filtered_matches)} matches. Use filters to narrow results.")
    
    except Exception as e:
        st.error(f"Error displaying matches: {e}")
        st.info("Try reinitializing the news system if this error persists.")

def display_research_dashboard():
    """Display research results dashboard"""
    st.subheader("🔬 Research Results Dashboard")
    
    if not st.session_state.research_database:
        st.info("No research results available. Run the enhanced streaming pipeline to generate research.")
        return
    
    # Get recent research results
    recent_results = st.session_state.research_database.get_recent_results(20)
    
    if not recent_results:
        st.info("No research results found. Try running the enhanced streaming pipeline with research enabled.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Research", len(recent_results))
    
    with col2:
        critical_count = sum(1 for r in recent_results if r.urgency_level == "Critical")
        st.metric("Critical Issues", critical_count)
    
    with col3:
        high_count = sum(1 for r in recent_results if r.urgency_level == "High")
        st.metric("High Priority", high_count)
    
    with col4:
        avg_time = np.mean([r.research_duration_seconds for r in recent_results if r.research_duration_seconds])
        st.metric("Avg Research Time", f"{avg_time:.1f}s")
    
    # Filter and sort options
    col1, col2 = st.columns(2)
    
    with col1:
        urgency_filter = st.selectbox("Filter by urgency:", ["All", "Critical", "High", "Medium", "Low"])
    
    with col2:
        sort_by = st.selectbox("Sort by:", ["Date", "Urgency", "Research Time"])
    
    # Filter results
    filtered_results = recent_results
    if urgency_filter != "All":
        filtered_results = [r for r in filtered_results if r.urgency_level == urgency_filter]
    
    # Sort results
    if sort_by == "Date":
        filtered_results.sort(key=lambda x: x.research_triggered_at, reverse=True)
    elif sort_by == "Urgency":
        urgency_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        filtered_results.sort(key=lambda x: urgency_order.get(x.urgency_level, 0), reverse=True)
    elif sort_by == "Research Time":
        filtered_results.sort(key=lambda x: x.research_duration_seconds or 0, reverse=True)
    
    # Display results
    st.write(f"**Research Results ({len(filtered_results)} shown)**")
    
    for i, result in enumerate(filtered_results[:10]):  # Show first 10
        urgency_emoji = {
            "Critical": "🚨",
            "High": "🔥", 
            "Medium": "🟡",
            "Low": "🟢"
        }.get(result.urgency_level, "⚪")
        
        with st.expander(f"{urgency_emoji} {result.article_title[:80]}... ({result.urgency_level})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Executive Summary:**")
                st.info(result.executive_summary[:300] + "..." if len(result.executive_summary) > 300 else result.executive_summary)
                
                if result.key_takeaways:
                    st.write("**Key Takeaways:**")
                    for takeaway in result.key_takeaways[:3]:
                        st.write(f"• {takeaway}")
                
                if result.recommended_actions:
                    st.write("**Recommended Actions:**")
                    for action in result.recommended_actions[:3]:
                        st.write(f"• {action}")
            
            with col2:
                st.metric("Urgency Level", result.urgency_level)
                
                if result.research_duration_seconds:
                    st.metric("Research Time", f"{result.research_duration_seconds:.1f}s")
                
                if isinstance(result.research_triggered_at, datetime):
                    time_str = result.research_triggered_at.strftime('%Y-%m-%d %H:%M')
                else:
                    time_str = "Unknown"
                st.write(f"**Time:** {time_str}")
                
                if st.button("🔗 View Article", key=f"research_view_{result.article_id}"):
                    st.markdown(f"[Open Article]({result.article_url})")
                
                # Impact indicators
                impacts = []
                if result.financial_impact and result.financial_impact.get('severity') in ['Major', 'Moderate']:
                    impacts.append("💰 Financial")
                if result.operational_impact and result.operational_impact.get('severity') in ['Major', 'Moderate']:
                    impacts.append("⚙️ Operational")
                if result.reputational_impact and result.reputational_impact.get('severity') in ['Major', 'Moderate']:
                    impacts.append("🎭 Reputational")
                
                if impacts:
                    st.write(f"**Impacts:** {', '.join(impacts)}")

def display_news_analytics():
    """Display news analytics and insights"""
    st.subheader("📈 News Analytics")
    
    if not st.session_state.relevance_monitor:
        st.info("Initialize the news system to see analytics.")
        return
    
    try:
        matches = st.session_state.relevance_monitor.relevance_matches
        
        if not matches:
            st.info("No data available. Run the news pipeline first.")
            return
        
        # Convert to DataFrame for analysis with datetime handling
        match_data = []
        for match in matches:
            # Handle datetime conversion
            try:
                if isinstance(match.created_at, datetime):
                    date_obj = match.created_at
                elif isinstance(match.created_at, str):
                    date_obj = datetime.fromisoformat(match.created_at.replace('Z', '+00:00'))
                else:
                    date_obj = datetime.now()
            except:
                date_obj = datetime.now()
            
            match_data.append({
                'title': match.news_title,
                'relevance_score': match.relevance_score,
                'category': match.relevance_category,
                'entity_count': len(match.graph_entities),
                'semantic_similarity': match.semantic_similarity,
                'date': date_obj,
                'research_triggered': match.processed_by_agent
            })
        
        df = pd.DataFrame(match_data)
        
        # Analytics visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Relevance score distribution
            fig = px.histogram(
                df, 
                x='relevance_score', 
                title="Relevance Score Distribution",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Category breakdown
            category_counts = df['category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Articles by Relevance Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Time series of articles
            daily_counts = df.groupby(df['date'].dt.date).size().reset_index()
            daily_counts.columns = ['date', 'count']
            
            fig = px.line(
                daily_counts,
                x='date',
                y='count',
                title="Articles Processed Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Entity involvement
            entity_counts = df['entity_count'].value_counts().sort_index()
            fig = px.bar(
                x=entity_counts.index,
                y=entity_counts.values,
                title="Distribution of Entity Matches per Article"
            )
            fig.update_xaxis(title="Number of Entities Matched")
            fig.update_yaxis(title="Number of Articles")
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = df['relevance_score'].mean()
            st.metric("Average Relevance Score", f"{avg_score:.2f}")
        
        with col2:
            high_relevance_pct = (df['category'] == 'HIGH').mean() * 100
            st.metric("High Relevance %", f"{high_relevance_pct:.1f}%")
        
        with col3:
            research_triggered_pct = df['research_triggered'].mean() * 100
            st.metric("Research Triggered %", f"{research_triggered_pct:.1f}%")
        
        with col4:
            avg_entities = df['entity_count'].mean()
            st.metric("Avg Entities per Article", f"{avg_entities:.1f}")
    
    except Exception as e:
        st.error(f"Error generating analytics: {e}")
        st.info("Try reinitializing the news system if this error persists.")

def display_enhanced_news_dashboard(G, openai_api_key: str):
    """Enhanced news dashboard with research integration"""
    st.header("📰 Enhanced News Monitoring Dashboard with AI Research")
    
    if not NEWS_MODULES_AVAILABLE:
        st.error("News pipeline modules are not available. Please install the required dependencies.")
        return
    
    # API Key configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        newsapi_key = st.text_input(
            "NewsAPI Key (optional):",
            type="password",
            help="Optional: Provides access to more news sources"
        )
    
    with col2:
        st.markdown("### 🔧 Enhanced Setup")
        if st.button("Initialize Enhanced System"):
            with st.spinner("Initializing enhanced news system..."):
                try:
                    # Initialize pipeline
                    st.session_state.news_pipeline = NewsPipeline(
                        openai_api_key=openai_api_key,
                        newsapi_key=newsapi_key if newsapi_key else None,
                        db_path="./apple_news_db"
                    )
                    
                    # Initialize relevance monitor
                    st.session_state.relevance_monitor = RelevanceMonitor(
                        openai_api_key=openai_api_key,
                        news_db_path="./apple_news_db"
                    )
                    
                    # Initialize research agent
                    st.session_state.research_agent = ResearchAgent(
                        openai_api_key=openai_api_key
                    )
                    
                    # Initialize research database
                    st.session_state.research_database = ResearchDatabase("./research_results.json")
                    
                    st.success("✅ Enhanced news system with AI research initialized!")
                    
                except Exception as e:
                    st.error(f"Error initializing enhanced system: {e}")
    
    # Enhanced dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🌊🔬 Enhanced Pipeline", 
        "🎯 Relevance Matches", 
        "🔬 Research Results",
        "📈 Analytics"
    ])
    
    with tab1:
        display_news_overview()
    
    with tab2:
        display_enhanced_pipeline_control(G, openai_api_key)
    
    with tab3:
        display_relevance_matches(G)
    
    with tab4:
        display_research_dashboard()
    
    with tab5:
        display_news_analytics()

def main():
    """Enhanced main Streamlit application with research integration"""
    
    # Initialize graph manager
    graph_manager = GraphManager()
    
    # Load graph
    if st.session_state.graph is None:
        with st.spinner("Loading graph..."):
            st.session_state.graph = graph_manager.load_graph()
    
    G = st.session_state.graph

    # Process hierarchy with validation
    try:
        with st.spinner("Building solar system hierarchy..."):
            G = validate_and_fix_graph(G)  # Fix any missing attributes
            G = ensure_hierarchy(G)  # Create solar system structure
            st.session_state.graph = G  # Update session state with fixed graph
        
        # Validate the hierarchy structure
        hierarchy_valid = validate_hierarchy_structure(G)
        if hierarchy_valid:
            st.success("✅ Solar system hierarchy successfully created!")
        else:
            st.warning("⚠️ Hierarchy structure may have issues - check console output")
            
    except Exception as e:
        st.error(f"Error processing graph hierarchy: {e}")
        st.info("Graph will be displayed without hierarchical organization")
        
    # Main title
    st.title("🍎 Apple Knowledge Graph System")
    st.markdown("**Solar System View with AI-Powered News Monitoring & Research - Complete Company Impact Analysis**")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Hierarchy validation section
    st.sidebar.subheader("🌟 Hierarchy Status")
    if st.sidebar.button("Validate Hierarchy"):
        with st.spinner("Validating hierarchy..."):
            is_valid = validate_hierarchy_structure(G)
            if is_valid:
                st.sidebar.success("✅ Hierarchy is valid!")
            else:
                st.sidebar.error("❌ Hierarchy has issues")
    
    if st.sidebar.button("Rebuild Hierarchy"):
        with st.spinner("Rebuilding solar system structure..."):
            G = validate_and_fix_graph(G)
            G = ensure_hierarchy(G)
            st.session_state.graph = G
            graph_manager.save_graph(G)
            st.sidebar.success("✅ Hierarchy rebuilt!")
            st.rerun()
    
    # File operations
    st.sidebar.subheader("📁 File Operations")
    uploaded_file = st.sidebar.file_uploader("Upload graph file:", type=['gpickle', 'gexf'])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.gpickle'):
                # Use pickle directly for .gpickle files
                G = pickle.load(uploaded_file)
            elif uploaded_file.name.endswith('.gexf'):
                G = nx.read_gexf(uploaded_file)
                
            G = validate_and_fix_graph(G)
            G = ensure_hierarchy(G)
            st.session_state.graph = G
            st.sidebar.success("Graph loaded and processed successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    
    if st.sidebar.button("Save Graph"):
        if graph_manager.save_graph(G):
            st.sidebar.success("Graph saved!")
    
    # Visualization controls
    st.sidebar.subheader("🎨 Visualization")
    all_types = ["All"] + list(set(G.nodes[n].get('type', 'Unknown') for n in G.nodes()))
    selected_types = st.sidebar.multiselect(
        "Filter by type:", 
        options=all_types, 
        default=["All"]
    )
    
    search_term = st.sidebar.text_input("Search nodes:")
    highlight_recent = st.sidebar.checkbox("Highlight recent additions", value=True)
    
    # Display mode with enhanced options
    display_mode = st.sidebar.selectbox(
        "Display Mode:",
        ["Solar System View", "Editor", "Analytics", "Hierarchy Debug", "📰🔬 Enhanced News Monitoring"]
    )
    
    # Main content area
    if display_mode == "Solar System View":
        st.header("🌟 Solar System Network Visualization")
        
        # Quick hierarchy stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Nodes", G.number_of_nodes())
        with col2:
            categories = [n for n in G.nodes() if n.startswith("cat::")]
            st.metric("Categories", len(categories))
        with col3:
            entities = [n for n in G.nodes() if not n.startswith("cat::") and 
                       G.nodes[n].get("type") not in ["Category", "Company"]]
            st.metric("Entities", len(entities))
        with col4:
            root_connections = len(list(G.successors(ROOT_ID)))
            st.metric("Root Connections", root_connections)
        
        # Network plot with solar system layout
        fig = create_network_plot(G, selected_types, search_term, highlight_recent)
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend explanation
        st.info("🌟 **Solar System Legend:** Gold star = Apple Inc (center) | Red circles = Categories | Other shapes = Entities by type")
    
    elif display_mode == "Editor":
        display_graph_editor(G, graph_manager)
        
        # Mini graph view
        st.subheader("📊 Current Graph Structure")
        mini_fig = create_network_plot(G, highlight_recent=True)
        st.plotly_chart(mini_fig, use_container_width=True)
    
    elif display_mode == "Analytics":
        st.header("📈 Graph Analytics")
        
        # Basic analytics
        if G.number_of_nodes() > 0:
            # Node type distribution
            node_types = [G.nodes[n].get('type', 'Unknown') for n in G.nodes()]
            type_counts = pd.Series(node_types).value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(values=type_counts.values, names=type_counts.index,
                            title="Node Type Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Relationship analysis
                if G.number_of_edges() > 0:
                    rel_types = [G[u][v].get('type', 'Unknown') for u, v in G.edges()]
                    rel_counts = pd.Series(rel_types).value_counts()
                    
                    fig = px.bar(x=rel_counts.index, y=rel_counts.values,
                               title="Relationship Type Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Hierarchy Analysis
            st.subheader("🏗️ Hierarchy Analysis")
            
            # Root analysis
            root_successors = list(G.successors(ROOT_ID))
            st.write(f"**Root ({ROOT_ID}) is connected to:** {', '.join(root_successors)}")
            
            # Category analysis
            for cat in [n for n in root_successors if n.startswith("cat::")]:
                cat_children = list(G.successors(cat))
                st.write(f"**{G.nodes[cat].get('name', cat)}:** {len(cat_children)} items")
                if len(cat_children) <= 10:  # Show details for smaller categories
                    child_names = [G.nodes[child].get('name', child) for child in cat_children]
                    st.write(f"  └── {', '.join(child_names)}")
        else:
            st.info("Add some nodes to see analytics")
    
    elif display_mode == "Hierarchy Debug":
        st.header("🔍 Hierarchy Debug Information")
        
        # Detailed hierarchy analysis
        st.subheader("Root Node Analysis")
        if ROOT_ID in G:
            root_data = G.nodes[ROOT_ID]
            st.json(dict(root_data))
            
            root_out = list(G.successors(ROOT_ID))
            root_in = list(G.predecessors(ROOT_ID))
            
            st.write(f"**Outgoing connections ({len(root_out)}):** {root_out}")
            st.write(f"**Incoming connections ({len(root_in)}):** {root_in}")
        else:
            st.error(f"Root node '{ROOT_ID}' not found!")
        
        # Category analysis
        st.subheader("Category Analysis")
        categories = [n for n in G.nodes() if n.startswith("cat::") or G.nodes[n].get("type") == "Category"]
        
        for cat in categories:
            with st.expander(f"Category: {G.nodes[cat].get('name', cat)}"):
                cat_data = G.nodes[cat]
                st.json(dict(cat_data))
                
                children = list(G.successors(cat))
                parents = list(G.predecessors(cat))
                
                st.write(f"**Parents:** {parents}")
                st.write(f"**Children ({len(children)}):** {children}")
        
        # Problem detection
        st.subheader("⚠️ Potential Issues")
        
        # Entities connected to root
        entities_to_root = []
        for n in G.nodes():
            if (n != ROOT_ID and 
                not n.startswith("cat::") and 
                G.nodes[n].get("type") not in ["Category", "Subcategory"] and
                (G.has_edge(ROOT_ID, n) or G.has_edge(n, ROOT_ID))):
                entities_to_root.append(n)
        
        if entities_to_root:
            st.error(f"❌ {len(entities_to_root)} entities connected directly to root: {entities_to_root}")
        else:
            st.success("✅ No entities directly connected to root")
        
        # Orphaned nodes
        orphaned = []
        for n in G.nodes():
            if (n != ROOT_ID and 
                not n.startswith("cat::") and 
                G.nodes[n].get("type") not in ["Category"] and
                len(list(G.predecessors(n))) == 0):
                orphaned.append(n)
        
        if orphaned:
            st.warning(f"⚠️ {len(orphaned)} orphaned nodes: {orphaned}")
        else:
            st.success("✅ No orphaned nodes found")
    
    elif display_mode == "📰🔬 Enhanced News Monitoring":
        # Get OpenAI API key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("Please set OPENAI_API_KEY environment variable to use enhanced news monitoring.")
        else:
            display_enhanced_news_dashboard(G, openai_api_key)
    
    # Enhanced footer with milestone 6 completion
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 System Status")
    st.sidebar.markdown("""
    **✅ Completed:**
    - ⭐ Solar system hierarchy structure
    - 🎨 Interactive graph visualization  
    - ✏️ Real-time graph editing
    - 💾 Data persistence
    - 🔍 Filter and search capabilities
    - 📰 News pipeline integration
    - 🎯 Relevance detection & LangGraph
    - 📊 Analytics dashboard
    - 🔬 **AI Research Agent (Milestone 6)**
    - 🌊 **Real-time streaming with research**
    - 📋 **Executive impact summaries**
    
    **🎉 Project Complete!**
    - All 6 milestones implemented
    - Full AI-augmented company impact analysis
    - Real-time news monitoring with research
    """)

if __name__ == "__main__":
    main()