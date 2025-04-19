import requests
import pandas as pd
from datetime import datetime

class LiveTest:
    def __init__(self):
        self.uniprot_api = "https://rest.uniprot.org/uniprotkb/search?query=locations:(location:membrane)&format=json"
        self.kaleidoscope = KaleidoscopeSystem()
        
    def run_membrane_analysis(self):
        print("Starting Live Membrane Protein Analysis")
        start_time = datetime.now()
        
        # Fetch real membrane protein data
        response = requests.get(self.uniprot_api)
        if response.status_code != 200:
            raise Exception("Failed to fetch protein data")
            
        protein_data = response.json()
        
        # Process through Kaleidoscope
        processed_data = []
        for protein in protein_data['results']:
            std_data = StandardizedData(
                raw_data={
                    'id': protein['primaryAccession'],
                    'name': protein.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
                    'function': [f.get('value') for f in protein.get('comments', []) if f.get('commentType') == 'FUNCTION'],
                    'locations': [l.get('value') for l in protein.get('comments', []) if l.get('commentType') == 'SUBCELLULAR LOCATION']
                },
                data_type='protein',
                metadata={
                    'source': 'UniProt',
                    'timestamp': datetime.now().isoformat()
                }
            )
            processed_data.append(std_data)
            
        # Feed data into system
        self.kaleidoscope.process_input_data(processed_data)
        
        # Get results
        cluster_summary = self.kaleidoscope.cluster_manager.get_cluster_summary()
        
        print(f"\nAnalysis Complete in {datetime.now() - start_time}")
        print(f"Proteins Processed: {len(processed_data)}")
        print(f"Clusters Formed: {cluster_summary['total_clusters']}")
        
        return cluster_summary

if __name__ == "__main__":
    test = LiveTest()
    results = test.run_membrane_analysis()
    print("\nDetailed Results:")
    for cluster_id, info in results['clusters'].items():
        print(f"\nCluster: {cluster_id}")
        print(f"Size: {info['size']} nodes")
        print(f"Average Confidence: {info['average_confidence']:.2f}")
