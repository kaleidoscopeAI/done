import urllib.request
import json
from datetime import datetime

class ImmediateTest:
    def __init__(self):
        self.uniprot_url = "https://rest.uniprot.org/uniprotkb/search?query=locations:(location:membrane)&format=json"
        self.kaleidoscope = KaleidoscopeSystem()
        
    def run_test(self):
        print("Fetching live membrane protein data...")
        
        with urllib.request.urlopen(self.uniprot_url) as response:
            protein_data = json.loads(response.read())
            
        print(f"Processing {len(protein_data['results'])} proteins...")
        
        for protein in protein_data['results']:
            self.kaleidoscope.process_input_data({
                'type': 'protein',
                'content': protein
            })
            
        return self.kaleidoscope.cluster_manager.get_cluster_summary()

test = ImmediateTest()
results = test.run_test()
