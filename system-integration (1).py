import numpy as np
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt

class KaleidoscopeSystem:
    """
    Integrates all components of the Kaleidoscope AI system.
    """
    def __init__(self):
        self.data_processor = DataProcessor()
        self.kaleidoscope_engine = KaleidoscopeEngine(num_banks=5, bank_capacity=3)
        self.cluster_manager = ClusterManager()
        
    def process_input_data(self, raw_data: List[Dict[str, Any]]) -> None:
        """
        Processes a batch of input data through the entire system.
        """
        print("Starting Kaleidoscope AI Processing Pipeline...")
        print("-" * 50)
        
        # Step 1: Data Standardization
        standardized_data = []
        for data_item in raw_data:
            std_data = self.data_processor.process(
                data_item['content'],
                data_item['type']
            )
            standardized_data.append(std_data)
            print(f"Standardized {data_item['type']} data: {std_data.metadata}")
        
        # Step 2: Initial Knowledge Processing
        print("\nProcessing through Kaleidoscope Engine...")
        for data in standardized_data:
            self.kaleidoscope_engine.add_insight(data)
        
        # Step 3: Generate Initial Insights
        insights = self.kaleidoscope_engine.generate_insights()
        print(f"Generated {len(insights)} initial insights")
        
        # Step 4: Create Domain Nodes
        print("\nCreating Domain-Specific Nodes...")
        for insight in insights:
            domain_node = self.cluster_manager.create_domain_node(insight)
            print(f"Created node in domain: {domain_node.domain}")
        
        # Step 5: Establish Node Connections
        self._establish_node_connections()
        
        # Step 6: Update Clusters
        print("\nForming Knowledge Clusters...")
        self.cluster_manager.update_clusters()
        
        # Step 7: Generate System Status
        self._print_system_status()
        
        # Step 8: Visualize Results
        self._visualize_system_state()

    def _establish_node_connections(self):
        """
        Establishes connections between nodes based on similarity.
        """
        nodes = list(self.cluster_manager.domain_nodes.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                similarity = self.cluster_manager.calculate_node_similarity(
                    nodes[i],
                    nodes[j]
                )
                if similarity > self.cluster_manager.similarity_threshold:
                    self.cluster_manager.add_connection(
                        nodes[i],
                        nodes[j],
                        weight=similarity
                    )

    def _print_system_status(self):
        """
        Prints current system status and cluster information.
        """
        print("\nSystem Status:")
        print("-" * 50)
        
        # Kaleidoscope Engine Status
        engine_status = self.kaleidoscope_engine.get_status()
        print("\nKaleidoscope Engine:")
        print(f"Active Memory Banks: {engine_status['total_insights']}")
        
        # Cluster Status
        cluster_summary = self.cluster_manager.get_cluster_summary()
        print(f"\nCluster Summary:")
        print(f"Total Nodes: {cluster_summary['total_nodes']}")
        print(f"Total Clusters: {cluster_summary['total_clusters']}")
        
        for cluster_id, info in cluster_summary['clusters'].items():
            print(f"\nCluster: {cluster_id}")
            print(f"Domain: {info['domain']}")
            print(f"Size: {info['size']} nodes")
            print(f"Average Confidence: {info['average_confidence']:.2f}")

    def _visualize_system_state(self):
        """
        Creates a visualization of the current system state.
        """
        plt.figure(figsize=(12, 8))
        
        # Get the visualization graph
        viz_graph = self.cluster_manager.visualize_clusters()
        
        # Create layout
        pos = nx.spring_layout(viz_graph)
        
        # Draw nodes
        domains = set(nx.get_node_attributes(viz_graph, 'domain').values())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(domains)))
        domain_color_map = dict(zip(domains, colors))
        
        for domain in domains:
            domain_nodes = [
                node for node, attr in viz_graph.nodes(data=True)
                if attr.get('domain') == domain
            ]
            nx.draw_networkx_nodes(
                viz_graph,
                pos,
                nodelist=domain_nodes,
                node_color=[domain_color_map[domain]],
                node_size=500,
                alpha=0.6,
                label=domain
            )
        
        # Draw edges
        nx.draw_networkx_edges(
            viz_graph,
            pos,
            alpha=0.5,
            edge_color='gray'
        )
        
        # Add labels
        labels = {
            node: f"{attr['domain'][:3]}{i}"
            for i, (node, attr) in enumerate(viz_graph.nodes(data=True))
        }
        nx.draw_networkx_labels(viz_graph, pos, labels, font_size=8)
        
        plt.title("Kaleidoscope AI Knowledge Clusters")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()

def demonstrate_system():
    """
    Demonstrates the complete Kaleidoscope AI system with sample data.
    """
    # Sample input data
    sample_data = [
        {
            'type': 'text',
            'content': 'The cell membrane consists of a phospholipid bilayer with embedded proteins.'
        },
        {
            'type': 'text',
            'content': 'Membrane proteins facilitate transport of molecules across the cell membrane.'
        },
        {
            'type': 'numerical',
            'content': [23.5, 45.2, 67.8]  # Example membrane thickness measurements
        },
        {
            'type': 'text',
            'content': 'Chemical compounds can affect membrane permeability through various mechanisms.'
        },
        {
            'type': 'structured',
            'content': {
                'membrane_components': ['phospholipids', 'proteins', 'cholesterol'],
                'properties': {'fluidity': 0.75, 'permeability': 0.45}
            }
        }
    ]
    
    # Initialize and run system
    system = KaleidoscopeSystem()
    system.process_input_data(sample_data)

if __name__ == "__main__":
    demonstrate_system()
