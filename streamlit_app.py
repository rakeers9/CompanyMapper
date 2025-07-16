#streamlit_app.py

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
import time
from pathlib import Path
from knowledge_graph import ensure_hierarchy, validate_and_fix_graph

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
                "type": "GeographicLocation", 
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

# Allowed relationship types (from your Milestone 2)
ALLOWED_RELATIONSHIP_TYPES = [
    "Supplies", "Manufactures", "Designs", "Distributes", "Operates_In", 
    "Owns", "Acquires", "Partners_With", "Licenses_From", "Regulated_By", 
    "Complies_With", "Uses_Material", "Generates_Emission", "Has_Initiative", 
    "Invests_In", "Provides_Service", "Faces_Litigation", "Receives_Certification", 
    "Reports_On", "Impacts"
]

# def create_network_plot(G, selected_types=None, search_term="", highlight_recent=False):
#     """Create interactive network plot with highlighting options"""
    
#     if G.number_of_nodes() == 0:
#         return go.Figure().add_annotation(text="No nodes in graph", x=0.5, y=0.5, showarrow=False)
    
#     # Filter graph
#     filtered_nodes = list(G.nodes())
#     if selected_types and "All" not in selected_types:
#         filtered_nodes = [n for n in G.nodes() 
#                          if G.nodes[n].get('type', 'Unknown') in selected_types]
    
#     if search_term:
#         filtered_nodes = [n for n in filtered_nodes 
#                          if search_term.lower() in G.nodes[n].get('name', n).lower()]
    
#     subG = G.subgraph(filtered_nodes)
    
#     if len(subG.nodes()) == 0:
#         return go.Figure().add_annotation(text="No nodes match filters", x=0.5, y=0.5, showarrow=False)
    
#     # # Calculate layout
#     # pos = nx.spring_layout(subG, k=1, iterations=50)
    
#     # # Create edges
#     edge_x, edge_y = [], []
    
#     try:
#         from networkx.drawing.nx_agraph import graphviz_layout
#         pos = graphviz_layout(subG, prog="dot", root=ROOT_ID)
#     except Exception:
#         # fallback simple layered layout
#         levels, queue = {}, [(ROOT_ID, 0)]
#         while queue:
#             n, d = queue.pop(0)
#             levels[n] = d
#             for child in subG.successors(n):
#                 if child not in levels:
#                     queue.append((child, d + 1))
#         pos = {}
#         for depth in set(levels.values()):
#             nodes = [k for k, v in levels.items() if v == depth]
#             for i, node in enumerate(nodes):
#                 pos[node] = (i, -depth)
    
#     for edge in subG.edges(data=True):
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_x.extend([x0, x1, None])
#         edge_y.extend([y0, y1, None])
    
#     # Edge trace
#     edge_trace = go.Scatter(
#         x=edge_x, y=edge_y,
#         line=dict(width=2, color='#888'),
#         hoverinfo='none',
#         mode='lines',
#         showlegend=False
#     )
    
#     # Node traces by type
#     node_traces = []
#     unique_types = list(set(subG.nodes[n].get('type', 'Unknown') for n in subG.nodes()))
#     colors = px.colors.qualitative.Set3[:len(unique_types)]
    
#     for i, node_type in enumerate(unique_types):
#         type_nodes = [n for n in subG.nodes() 
#                      if subG.nodes[n].get('type', 'Unknown') == node_type]
        
#         node_x = [pos[n][0] for n in type_nodes]
#         node_y = [pos[n][1] for n in type_nodes]
#         node_text = [subG.nodes[n].get('name', n) for n in type_nodes]
        
#         # Check for recent nodes
#         node_sizes = []
#         node_colors = []
#         for n in type_nodes:
#             is_recent = False
#             if highlight_recent and 'created' in subG.nodes[n]:
#                 try:
#                     created_time = datetime.fromisoformat(subG.nodes[n]['created'])
#                     is_recent = (datetime.now() - created_time).seconds < 3600
#                 except:
#                     pass
            
#             node_sizes.append(25 if is_recent else 20)
#             node_colors.append('red' if is_recent else colors[i])
        
#         node_info = []
#         for n in type_nodes:
#             info = f"<b>{subG.nodes[n].get('name', n)}</b><br>"
#             info += f"Type: {subG.nodes[n].get('type', 'Unknown')}<br>"
#             info += f"Connections: {len(list(subG.neighbors(n)))}<br>"
#             info += f"Description: {subG.nodes[n].get('prop_description', 'N/A')}<br>"
#             if 'created' in subG.nodes[n]:
#                 info += f"Created: {subG.nodes[n]['created'][:19]}"
#             node_info.append(info)
        
#         node_trace = go.Scatter(
#             x=node_x, y=node_y,
#             mode='markers+text',
#             hoverinfo='text',
#             hovertext=node_info,
#             text=node_text,
#             textposition="middle center",
#             marker=dict(
#                 size=node_sizes,
#                 color=node_colors,
#                 line=dict(width=2, color='white')
#             ),
#             name=node_type,
#             textfont=dict(size=8, color='white')
#         )
#         node_traces.append(node_trace)
    
#     # Create figure with FIXED layout parameters
#     fig = go.Figure(
#         data=[edge_trace] + node_traces,
#         layout=go.Layout(
#             title={
#                 'text': f'Apple Knowledge Graph ({len(subG.nodes())} nodes, {len(subG.edges())} edges)',
#                 'font': {'size': 16},
#                 'x': 0.5,
#                 'xanchor': 'center'
#             },
#             showlegend=True,
#             hovermode='closest',
#             margin=dict(b=20,l=5,r=5,t=40),
#             annotations=[dict(
#                 text="🔴 Red nodes = Recently added | Hover for details",
#                 showarrow=False,
#                 xref="paper", yref="paper",
#                 x=0.005, y=-0.002,
#                 xanchor='left', yanchor='bottom',
#                 font=dict(color='gray', size=12)
#             )],
#             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
#         )
#     )
    
#     return fig


import math
import numpy as np

def create_solar_system_layout(G, root_id=ROOT_ID):
    """
    Create a solar system layout with root at center and categories in orbit
    """
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
                options=["Company", "Product", "Supplier", "GeographicLocation", "Component", "Service", "Other"],
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


def main():
    """Main Streamlit application"""
    
    # Initialize graph manager
    graph_manager = GraphManager()
    
    # Load graph
    if st.session_state.graph is None:
        with st.spinner("Loading graph..."):
            st.session_state.graph = graph_manager.load_graph()
    
    G = st.session_state.graph

    # Process hierarchy with validation
    try:
        from knowledge_graph import validate_and_fix_graph, ensure_hierarchy, validate_hierarchy_structure
        
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
    st.markdown("**Solar System View - Interactive graph with Apple Inc at the center**")
    
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
    
    # Display mode
    display_mode = st.sidebar.selectbox(
        "Display Mode:",
        ["Solar System View", "Editor", "Analytics", "Hierarchy Debug"]
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
    
    # Footer with next steps
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Ready for Next Milestones")
    st.sidebar.markdown("""
    **✅ Completed:**
    - ⭐ Solar system hierarchy structure
    - 🎨 Interactive graph visualization  
    - ✏️ Real-time graph editing
    - 💾 Data persistence
    - 🔍 Filter and search capabilities
    
    **🚧 Ready to add:**
    - 📰 News pipeline integration
    - 🤖 LangGraph workflows  
    - 🔬 Research agent integration
    """)

if __name__ == "__main__":
    main()