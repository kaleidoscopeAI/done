# visualization/visualizer.py
import logging
from typing import Dict, List, Optional, Any
import time

# Potential visualization libraries (install separately if used)
try:
    import plotly.graph_objects as go
    import networkx as nx # Needed if visualizing the memory graph
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly or NetworkX not found. Visualization module will be limited.")

logger = logging.getLogger(__name__)

class SystemVisualizer:
    """
    Handles visualization of the AI system's state, including nodes,
    energy levels, and potentially the memory graph.
    (Conceptual: Actual implementation depends heavily on desired output format).
    """
    def __init__(self, node_manager, memory_graph):
        self.node_manager = node_manager
        self.memory_graph = memory_graph
        self.last_viz_time = None
        logger.info(f"SystemVisualizer initialized. Plotly available: {PLOTLY_AVAILABLE}")

    def generate_node_visualization(self, output_format="html", filename="node_visualization.html") -> Optional[str]:
        """
        Generates a visualization of the current nodes and their states.

        Args:
            output_format (str): 'html', 'json', 'png', etc. (depends on library)
            filename (str): Path to save the visualization file (for html/png).

        Returns:
            Optional[str]: File path if saved, JSON string, or None on failure.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Cannot generate Plotly visualization: Plotly library not available.")
            return None

        logger.info("Generating node state visualization...")
        nodes = self.node_manager.get_all_nodes()
        if not nodes:
            logger.info("No nodes to visualize.")
            return None

        # --- Example using Plotly 3D Scatter Plot ---
        node_ids = [n.id for n in nodes]
        node_types = [n.node_type for n in nodes]
        # Simple spatial layout (e.g., random or based on type) - replace with better layout if needed
        positions = self._get_node_positions(nodes)
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        z_coords = [p[2] for p in positions]

        # Color nodes by type, size by energy
        unique_types = sorted(list(set(node_types)))
        color_map = {ntype: i for i, ntype in enumerate(unique_types)}
        node_colors = [color_map[ntype] for ntype in node_types]
        node_sizes = [max(5, n.state.energy * 0.1) for n in nodes] # Scale energy for size

        hover_texts = [
            f"ID: {n.id}<br>Type: {n.node_type}<br>Status: {n.state.status}<br>Energy: {n.state.energy:.2f}<br>Capabilities: {n.capabilities}"
            for n in nodes
        ]

        fig = go.Figure(data=[go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis', # Choose a colorscale
                opacity=0.8,
                colorbar=dict(title='Node Type Index', tickvals=list(color_map.values()), ticktext=list(color_map.keys()))
            ),
            text=hover_texts,
            hoverinfo='text'
        )])

        fig.update_layout(
            title='AI System Node Visualization',
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            margin=dict(l=0, r=0, b=0, t=40) # Adjust margins
        )

        self.last_viz_time = time.time()

        # --- Output Handling ---
        if output_format == "html":
            try:
                fig.write_html(filename)
                logger.info(f"Node visualization saved to {filename}")
                return filename
            except Exception as e:
                logger.error(f"Failed to save visualization HTML to {filename}: {e}", exc_info=True)
                return None
        elif output_format == "json":
             return fig.to_json()
        # Add other formats (png, etc.) if needed, might require 'kaleido' package: pip install kaleido
        # elif output_format == "png":
        #     try:
        #         fig.write_image(filename)
        #         logger.info(f"Node visualization saved to {filename}")
        #         return filename
        #     except Exception as e: ...

        else:
            logger.warning(f"Unsupported visualization output format: {output_format}")
            return None

    def _get_node_positions(self, nodes: List[Any]) -> List[Tuple[float, float, float]]:
        """Placeholder for calculating 3D positions for nodes."""
        # Simple random layout for now
        positions = []
        for _ in nodes:
            positions.append((random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)))
        return positions

    def visualize_memory_graph(self, output_format="html", filename="memory_graph.html"):
        """Generates a visualization of the memory graph (conceptual)."""
        if not PLOTLY_AVAILABLE or not nx:
            logger.warning("Cannot visualize memory graph: Plotly or NetworkX not available.")
            return None

        G = self.memory_graph.graph
        if G.number_of_nodes() == 0:
            logger.info("Memory graph is empty, nothing to visualize.")
            return None

        logger.info(f"Generating memory graph visualization (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})...")

        # --- NetworkX + Plotly Graph Visualization ---
        # 1. Calculate node positions using a NetworkX layout algorithm
        pos = nx.spring_layout(G, dim=3) # 3D layout

        # 2. Extract edge coordinates
        edge_x, edge_y, edge_z = [], [], []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None]) # None separates line segments
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        # 3. Create Plotly edge trace
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        # 4. Extract node coordinates and attributes
        node_x, node_y, node_z = [], [], []
        node_text = []
        node_colors = []
        node_sizes = []
        color_map = {'InputData': 'red', 'TextNode': 'blue', 'VisualNode': 'green',
                     'Entity': 'orange', 'Summary': 'purple', 'Pattern': 'cyan', 'default': 'grey'}

        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_info = G.nodes[node]
            text = f"ID: {node}<br>" + "<br>".join(f"{k}: {v}" for k, v in node_info.items())
            node_text.append(text)
            node_type = node_info.get('type', 'default')
            node_colors.append(color_map.get(node_type, color_map['default']))
            # Size based on degree or specific attribute?
            node_sizes.append(5 + G.degree(node)) # Example size based on connections

        # 5. Create Plotly node trace
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line_width=1),
            text=node_text
        )

        # 6. Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Memory Graph Visualization',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[dict(showarrow=False, text=" ", xref="paper", yref="paper", x=0.005, y=-0.002)],
                            scene=dict(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                       zaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        ))

        # --- Output Handling ---
        if output_format == "html":
            try:
                fig.write_html(filename)
                logger.info(f"Memory graph visualization saved to {filename}")
                return filename
            except Exception as e:
                logger.error(f"Failed to save memory graph HTML to {filename}: {e}", exc_info=True)
                return None
        # Add other formats if needed

        else:
            logger.warning(f"Unsupported visualization output format: {output_format}")
            return None


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Need dummy NodeManager and MemoryGraph for testing
    class MockNodeManager:
         def get_all_nodes(self):
             # Create a few mock nodes
             class MockNode:
                 def __init__(self, id, type, energy):
                     self.id = id; self.node_type = type;
                     self.state = type('obj', (object,), {'energy': energy, 'status': 'idle'})()
                     self.capabilities = {}
             return [MockNode("n1", "text", 80), MockNode("n2", "visual", 120), MockNode("n3", "text", 50)]

    class MockMemoryGraph:
         def __init__(self):
             self.graph = nx.DiGraph()
             self.graph.add_node("d1", type="InputData")
             self.graph.add_node("n1", type="TextNode")
             self.graph.add_node("e1", type="Entity", name="AI")
             self.graph.add_edge("n1", "d1", type="processed")
             self.graph.add_edge("n1", "e1", type="extracted")

    print("Testing SystemVisualizer...")
    if not PLOTLY_AVAILABLE:
         print("Plotly not available, skipping visualization generation.")
    else:
         node_mgr = MockNodeManager()
         mem_graph = MockMemoryGraph()
         visualizer = SystemVisualizer(node_mgr, mem_graph)

         # Ensure viz directory exists relative to this script
         viz_dir = os.path.join(os.path.dirname(__file__), '..', 'visualization_output')
         os.makedirs(viz_dir, exist_ok=True)
         node_viz_path = os.path.join(viz_dir, 'test_node_viz.html')
         mem_viz_path = os.path.join(viz_dir, 'test_mem_graph_viz.html')


         print(f"Attempting to generate node visualization to {node_viz_path}")
         visualizer.generate_node_visualization(filename=node_viz_path)

         print(f"\nAttempting to generate memory graph visualization to {mem_viz_path}")
         visualizer.visualize_memory_graph(filename=mem_viz_path)

         print("\nVisualization generation attempted. Check the 'visualization_output' directory.")
