from flask import Flask, send_file
from flask_socketio import SocketIO, emit
from supernode_core import SuperNodeCore, encode_data, decode_data
import threading
import time
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize SuperNode
node = SuperNodeCore()
node.start()

# Simulated network of nodes (for visualization)
nodes = [{'id': node.id, 'vector': node.state.vector[:3], 'energy': node.state.energy,
          'resonance': node.state.resonance, 'stability': node.state.stability, 'connections': []}]

def update_nodes():
    """Periodically update node states for clients."""
    while node.running:
        nodes[0].update({
            'vector': node.state.vector[:3],  # Use first 3 dims for 3D visualization
            'energy': node.state.energy,
            'resonance': node.state.resonance,
            'stability': node.state.stability
        })
        socketio.emit('node_update', {'nodes': nodes})
        time.sleep(1)

# Start background thread for node updates
threading.Thread(target=update_nodes, daemon=True).start()

@app.route('/')
def index():
    return send_file('index.html')

@socketio.on('connect')
def handle_connect():
    emit('node_update', {'nodes': nodes})

@socketio.on('process_text')
def handle_process_text(data):
    text = data.get('text', '')
    output, resonance = node.process_text(text)
    decoded = decode_data(output)
    emit('process_result', {'decoded': decoded, 'resonance': resonance})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
