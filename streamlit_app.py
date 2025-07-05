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

def create_network_plot(G, selected_types=None, search_term="", highlight_recent=False):
    """Create interactive network plot with highlighting options"""
    
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
    
    # Calculate layout
    pos = nx.spring_layout(subG, k=1, iterations=50)
    
    # Create edges
    edge_x, edge_y = [], []
    
    for edge in subG.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Node traces by type
    node_traces = []
    unique_types = list(set(subG.nodes[n].get('type', 'Unknown') for n in subG.nodes()))
    colors = px.colors.qualitative.Set3[:len(unique_types)]
    
    for i, node_type in enumerate(unique_types):
        type_nodes = [n for n in subG.nodes() 
                     if subG.nodes[n].get('type', 'Unknown') == node_type]
        
        node_x = [pos[n][0] for n in type_nodes]
        node_y = [pos[n][1] for n in type_nodes]
        node_text = [subG.nodes[n].get('name', n) for n in type_nodes]
        
        # Check for recent nodes
        node_sizes = []
        node_colors = []
        for n in type_nodes:
            is_recent = False
            if highlight_recent and 'created' in subG.nodes[n]:
                try:
                    created_time = datetime.fromisoformat(subG.nodes[n]['created'])
                    is_recent = (datetime.now() - created_time).seconds < 3600
                except:
                    pass
            
            node_sizes.append(25 if is_recent else 20)
            node_colors.append('red' if is_recent else colors[i])
        
        node_info = []
        for n in type_nodes:
            info = f"<b>{subG.nodes[n].get('name', n)}</b><br>"
            info += f"Type: {subG.nodes[n].get('type', 'Unknown')}<br>"
            info += f"Connections: {len(list(subG.neighbors(n)))}<br>"
            info += f"Description: {subG.nodes[n].get('prop_description', 'N/A')}<br>"
            if 'created' in subG.nodes[n]:
                info += f"Created: {subG.nodes[n]['created'][:19]}"
            node_info.append(info)
        
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
                line=dict(width=2, color='white')
            ),
            name=node_type,
            textfont=dict(size=8, color='white')
        )
        node_traces.append(node_trace)
    
    # Create figure with FIXED layout parameters
    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title={
                'text': f'Apple Knowledge Graph ({len(subG.nodes())} nodes, {len(subG.edges())} edges)',
                'font': {'size': 16},
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[dict(
                text="🔴 Red nodes = Recently added | Hover for details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
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
    
    # Main title
    st.title("🍎 Apple Knowledge Graph System")
    st.markdown("**Interactive graph editor with real-time updates - Ready for next milestones!**")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
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
            st.session_state.graph = G
            st.sidebar.success("Graph loaded successfully!")
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
        ["Graph View", "Editor", "Analytics"]
    )
    
    # Main content area
    if display_mode == "Graph View":
        st.header("🕸️ Network Visualization")
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", G.number_of_nodes())
        with col2:
            st.metric("Edges", G.number_of_edges())
        with col3:
            density = nx.density(G) if G.number_of_nodes() > 0 else 0
            st.metric("Density", f"{density:.3f}")
        with col4:
            recent_count = sum(1 for n in G.nodes(data=True) 
                             if 'created' in n[1] and 
                             (datetime.now() - datetime.fromisoformat(n[1]['created'])).seconds < 3600)
            st.metric("Recent Nodes", recent_count)
        
        # Network plot
        fig = create_network_plot(G, selected_types, search_term, highlight_recent)
        st.plotly_chart(fig, use_container_width=True)
    
    elif display_mode == "Editor":
        display_graph_editor(G, graph_manager)
        
        # Mini graph view
        st.subheader("📊 Current Graph")
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
        else:
            st.info("Add some nodes to see analytics")
    
    # Footer with next steps
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Ready for Next Milestones")
    st.sidebar.markdown("""
    **✅ Completed:**
    - Interactive graph visualization
    - Real-time graph editing
    - Data persistence
    - Filter and search capabilities
    
    **🚧 Ready to add:**
    - News pipeline integration
    - LangGraph workflows  
    - Research agent integration
    """)

if __name__ == "__main__":
    main()