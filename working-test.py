from kaleidoscope_ai.core.base_node import BaseNode
from kaleidoscope_ai.engines.kaleidoscope_engine import KaleidoscopeEngine
import urllib.request
import json
from datetime import datetime

# Initialize core components
engine = KaleidoscopeEngine(num_gears=5, gear_memory_threshold=10)
node = BaseNode()

# Fetch and process real data
print("Fetching membrane protein data...")
url = "https://rest.uniprot.org/uniprotkb/search?query=locations:(location:membrane)&format=json"

with urllib.request.urlopen(url) as response:
    data = json.loads(response.read())
    print(f"Processing {len(data['results'])} proteins...")
    
    for protein in data['results']:
        # Convert protein data to standardized format
        std_data = {
            'data_type': 'protein',
            'raw_data': protein,
            'metadata': {
                'accession': protein['primaryAccession'],
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Process through node
        processed_data = node.process_data_chunk(std_data)
        
        # Feed into Kaleidoscope engine
        if processed_data:
            engine.add_insight(processed_data)

# Generate insights
insights = engine.generate_insights()
print(f"\nGenerated {len(insights)} insights")

# Display results
print("\nCluster Analysis:")
status = engine.get_status()
print(f"Active Memory Banks: {status['current_bank']}")
print(f"Total Insights Processed: {status['total_insights']}")
