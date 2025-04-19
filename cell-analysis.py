def demonstrate_cell_analysis():
    # Initialize the system
    system = KaleidoscopeSystem()
    
    # Complex cellular data representing different aspects of cell biology
    cell_data = [
        # Cell Membrane Components and Function
        {
            'type': 'text',
            'content': """
            The cell membrane is a fluid mosaic of phospholipids, proteins, and cholesterol.
            Transmembrane proteins form channels and pumps for selective ion transport.
            Glycoproteins on the surface serve as recognition sites for cell signaling.
            The membrane maintains homeostasis through regulated substance exchange.
            """
        },
        {
            'type': 'numerical',
            'content': {
                'membrane_thickness': [7.5, 8.0, 7.8],  # nanometers
                'protein_density': [25000, 28000, 27500],  # proteins per μm²
                'lipid_ratio': [0.65, 0.25, 0.10]  # phospholipids, proteins, cholesterol
            }
        },
        
        # Mitochondrial Structure and Function
        {
            'type': 'text',
            'content': """
            Mitochondria have an outer membrane and a highly folded inner membrane forming cristae.
            The electron transport chain in cristae generates ATP through oxidative phosphorylation.
            Matrix enzymes catalyze the citric acid cycle and fatty acid oxidation.
            Mitochondrial DNA encodes essential proteins for energy production.
            """
        },
        {
            'type': 'numerical',
            'content': {
                'atp_production': [30, 32, 34],  # ATP molecules per glucose
                'membrane_potential': [-140, -160, -150],  # millivolts
                'oxygen_consumption': [0.2, 0.25, 0.22]  # μmol O₂/min/mg protein
            }
        },
        
        # Endoplasmic Reticulum and Protein Synthesis
        {
            'type': 'text',
            'content': """
            Rough ER is studded with ribosomes for protein synthesis.
            Smooth ER functions in lipid synthesis and calcium storage.
            Newly synthesized proteins undergo folding and modification in the ER lumen.
            Protein quality control ensures proper folding before transport to Golgi.
            """
        },
        {
            'type': 'structured',
            'content': {
                'protein_synthesis_rate': {'mean': 6.0, 'std': 0.5},  # proteins per second
                'calcium_concentration': {'er_lumen': 2.0, 'cytosol': 0.0001},  # mM
                'chaperone_proteins': ['BiP', 'calnexin', 'calreticulin', 'PDI'],
                'post_translational_modifications': ['glycosylation', 'disulfide_bonds', 'phosphorylation']
            }
        },
        
        # Nucleus and Gene Expression
        {
            'type': 'text',
            'content': """
            The nuclear envelope consists of inner and outer membranes with nuclear pores.
            Chromatin organization regulates gene accessibility and expression.
            Transcription factors bind specific DNA sequences to control gene activation.
            mRNA processing includes 5' capping, splicing, and 3' polyadenylation.
            """
        },
        {
            'type': 'structured',
            'content': {
                'nuclear_pore_density': 2500,  # pores per nucleus
                'chromatin_states': {
                    'euchromatin': 'active',
                    'heterochromatin': 'inactive'
                },
                'gene_expression_factors': [
                    'RNA_polymerase_II',
                    'general_transcription_factors',
                    'enhancers',
                    'silencers'
                ]
            }
        },
        
        # Cellular Transport and Vesicles
        {
            'type': 'text',
            'content': """
            Vesicular transport moves proteins between organelles.
            COPII vesicles transport from ER to Golgi.
            COPI vesicles mediate retrograde Golgi to ER transport.
            Clathrin-coated vesicles facilitate endocytosis at the plasma membrane.
            """
        },
        {
            'type': 'structured',
            'content': {
                'vesicle_types': {
                    'COPII': 'anterograde',
                    'COPI': 'retrograde',
                    'clathrin': 'endocytic',
                    'secretory': 'exocytic'
                },
                'transport_rates': {
                    'er_to_golgi': 400,  # vesicles/min
                    'golgi_to_pm': 300,  # vesicles/min
                    'endocytosis': 250   # vesicles/min
                }
            }
        }
    ]
    
    # Process the complex cell data
    print("\nProcessing Complex Cell Biology Data...")
    system.process_input_data(cell_data)
    
    # Generate detailed analysis of the formed knowledge structure
    cluster_summary = system.cluster_manager.get_cluster_summary()
    
    print("\nDetailed Analysis of Knowledge Structure:")
    print("-" * 50)
    
    for cluster_id, info in cluster_summary['clusters'].items():
        print(f"\nCluster: {cluster_id}")
        print(f"Domain: {info['domain']}")
        print(f"Number of Nodes: {info['size']}")
        print(f"Confidence Level: {info['average_confidence']:.2f}")
        
        # Get nodes in this cluster
        nodes = info['nodes']
        print("\nKey Concepts and Relationships:")
        for node in nodes:
            node_data = system.cluster_manager.domain_nodes[node['id']]
            print(f"- {node_data.knowledge.metadata.get('primary_concept', 'Unknown')}")
            print(f"  Connections: {len(node_data.connections)}")
    
    # Visualize the knowledge network
    system._visualize_system_state()
    
    return system

if __name__ == "__main__":
    cell_system = demonstrate_cell_analysis()
