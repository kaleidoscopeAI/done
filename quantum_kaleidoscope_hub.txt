#!/usr/bin/env python3
"""
Quantum Kaleidoscope Orchestration Hub
======================================
Middleware layer integrating the Quantum Kaleidoscope backend (SuperNodeCore)
with an immersive 3D visualization frontend.

Usage:
    python quantum_kaleidoscope_hub.py --port 8080 --data-dir ./data

Requirements:
    - Python 3.8+
    - Flask, Flask-SocketIO, numpy, networkx, python-dotenv, eventlet, colorlog
    - supernode_core.py (from previous context)
"""

import os
import sys
import time
import json
import uuid
import math
import argparse
import threading
import random
import logging
import subprocess
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Setup virtual environment if not present
VENV_PATH = ".venv"
if not Path(VENV_PATH).exists():
    try:
        import venv
        print(f"Creating virtual environment at {VENV_PATH}...")
        venv.create(VENV_PATH, with_pip=True)
        python_path = os.path.join(VENV_PATH, 'Scripts' if os.name == 'nt' else 'bin', 'python')
        pip_path = os.path.join(VENV_PATH, 'Scripts' if os.name == 'nt' else 'bin', 'pip')
        print("Installing required packages...")
        subprocess.check_call([pip_path, 'install', 'flask', 'flask-socketio', 'numpy', 'networkx', 'python-dotenv', 'eventlet', 'colorlog'])
        print("Starting with virtual environment...")
        os.execv(python_path, [python_path] + sys.argv)
    except Exception as e:
        print(f"Failed to create virtual environment: {e}")
        print("Continuing with system Python...")

# Import dependencies
try:
    from flask import Flask, send_file
    from flask_socketio import SocketIO, emit
    import networkx as nx
    import colorlog
    from supernode_core import SuperNodeCore, encode_data, decode_data
except ImportError as e:
    print(f"Required package missing: {e}")
    print("Please install: pip install flask flask-socketio numpy networkx python-dotenv eventlet colorlog")
    sys.exit(1)

# Setup logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    log_colors={'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white'}
))
logger = colorlog.getLogger("QuantumHub")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Data structures
@dataclass
class QuantumNode:
    id: str
    position: Tuple[float, float, float]
    energy: float
    stability: float
    connections: Dict[str, float] = field(default_factory=dict)
    features: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    creation_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    color: str = "#4488ff"
    size: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "position": list(self.position), "energy": self.energy, "stability": self.stability,
            "connections": self.connections, "features": self.features[:20], "metadata": self.metadata,
            "visible": self.visible, "creation_time": self.creation_time, "last_update": self.last_update,
            "color": self.color, "size": self.size
        }

@dataclass
class QuantumConnection:
    source: str
    target: str
    strength: float = 0.5
    type: str = "standard"
    metadata: Dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    creation_time: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "source": self.source, "target": self.target, "strength": self.strength, "type": self.type,
            "metadata": self.metadata, "visible": self.visible, "creation_time": self.creation_time
        }

@dataclass
class SimulationState:
    nodes: Dict[str, QuantumNode] = field(default_factory=dict)
    connections: List[QuantumConnection] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "connections": [conn.to_dict() for conn in self.connections],
            "events": self.events[-50:], "insights": self.insights[-20:], "metrics": self.metrics,
            "timestamp": self.timestamp
        }

class QuantumOrchestrator:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state = SimulationState()
        self.state_lock = threading.RLock()
        self.running = True
        self.simulation_thread = None
        self.network_graph = nx.Graph()
        self.metrics = {"fps": 0, "node_count": 0, "connection_count": 0, "update_time_ms": 0, "start_time": time.time()}
        
        # Initialize SuperNodeCore
        self.supernode = SuperNodeCore()
        self.supernode.start()
        self._initialize_from_supernode()

    def _initialize_from_supernode(self):
        """Initialize state from SuperNodeCore"""
        logger.info("Initializing from SuperNodeCore")
        seed_node = QuantumNode(
            id=self.supernode.id,
            position=(0, 0, 0),
            energy=self.supernode.state.energy,
            stability=self.supernode.state.stability,
            features=self.supernode.state.vector[:32].tolist(),
            metadata={"type": "seed", "description": "SuperNode origin"}
        )
        with self.state_lock:
            self.state.nodes[seed_node.id] = seed_node
            self.state.events.append({"type": "system", "message": "Initialized from SuperNode", "timestamp": time.time()})
            self.state.metrics.update({
                "node_count": 1, "connection_count": 0,
                "avg_energy": seed_node.energy, "avg_connections": 0, "simulation_time": 0
            })

    def start(self):
        logger.info("Starting Quantum Orchestrator")
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()

    def stop(self):
        logger.info("Stopping Quantum Orchestrator")
        self.running = False
        self.supernode.stop()
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)

    def get_current_state(self) -> Dict[str, Any]:
        with self.state_lock:
            return self.state.to_dict()

    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not metadata:
            metadata = {}
        metadata.update({"timestamp": time.time(), "source": "user_input", "text_sample": text[:100] + ("..." if len(text) > 100 else "")})
        logger.info(f"Processing text input ({len(text)} characters)")

        with self.state_lock:
            # Process text with SuperNodeCore
            output, resonance = self.supernode.process_text(text)
            decoded = decode_data(output)
            node_id = str(uuid.uuid4())
            position = (random.uniform(-15, 15), random.uniform(-15, 15), random.uniform(-5, 5))
            node = QuantumNode(
                id=node_id, position=position, energy=self.supernode.state.energy, stability=self.supernode.state.stability,
                features=self.supernode.state.vector[:32].tolist(), metadata=metadata, color=self._generate_color_from_text(text)
            )
            self.state.nodes[node_id] = node

            # Connect to seed node if similarity is high
            seed_node = self.state.nodes[self.supernode.id]
            similarity = self._calculate_similarity(seed_node.features, node.features)
            if similarity > 0.6:
                conn = QuantumConnection(source=seed_node.id, target=node_id, strength=similarity, type="semantic")
                self.state.connections.append(conn)
                seed_node.connections[node_id] = similarity
                node.connections[seed_node.id] = similarity

            # Generate insights
            insights = self._generate_insights_from_text(text, node_id, [(seed_node.id, similarity)] if similarity > 0.6 else [])
            self.state.insights.extend(insights)
            self.state.events.append({"type": "node_creation", "node_id": node_id, "message": "Created node from text", "timestamp": time.time()})
            self.state.timestamp = time.time()

            return {"node_id": node_id, "position": list(position), "decoded": decoded, "resonance": resonance, "insights": insights}

    def _calculate_similarity(self, features1: List[float], features2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(features1, features2))
        mag1 = math.sqrt(sum(a * a for a in features1))
        mag2 = math.sqrt(sum(b * b for b in features2))
        return dot_product / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0

    def _generate_color_from_text(self, text: str) -> str:
        import hashlib
        hash_obj = hashlib.md5(text.encode('utf-8'))
        color = "#" + hash_obj.hexdigest()[:6]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        if r + g + b < 300:
            r, g, b = min(255, r + 100), min(255, g + 100), min(255, b + 100)
            color = f"#{r:02x}{g:02x}{b:02x}"
        return color

    def _generate_insights_from_text(self, text: str, node_id: str, related_nodes: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        words = text.lower().split()
        complexity = min(1.0, sum(len(w) for w in words) / max(1, len(words)) / 10)
        insights = [{
            "type": "content_analysis", "title": "Text Processed",
            "content": f"Processed text with {len(words)} words, complexity {complexity:.2f}.",
            "confidence": 0.85, "timestamp": time.time()
        }]
        if related_nodes:
            top_id, similarity = related_nodes[0]
            insights.append({
                "type": "relationship_analysis", "title": "Semantic Connection",
                "content": f"Node {node_id} linked to {top_id} with similarity {similarity:.2f}.",
                "confidence": similarity, "timestamp": time.time()
            })
        return insights

    def _simulation_loop(self):
        logger.info("Starting simulation loop")
        last_metrics_update = time.time()
        frame_count = 0

        while self.running:
            start_time = time.time()
            with self.state_lock:
                # Update seed node from SuperNodeCore
                seed_node = self.state.nodes[self.supernode.id]
                seed_node.energy = self.supernode.state.energy
                seed_node.stability = self.supernode.state.stability
                seed_node.features = self.supernode.state.vector[:32].tolist()
                seed_node.last_update = time.time()

                # Simple force-directed layout
                forces = {node_id: [0.0, 0.0, 0.0] for node_id in self.state.nodes}
                for i, (n1_id, n1) in enumerate(self.state.nodes.items()):
                    for n2_id, n2 in list(self.state.nodes.items())[i+1:]:
                        dx, dy, dz = n2.position[0] - n1.position[0], n2.position[1] - n1.position[1], n2.position[2] - n1.position[2]
                        dist = max(math.sqrt(dx*dx + dy*dy + dz*dz), 1.0)
                        force = 5.0 / dist  # Repulsion
                        forces[n1_id] = [f - d * force / dist for f, d in zip(forces[n1_id], (dx, dy, dz))]
                        forces[n2_id] = [f + d * force / dist for f, d in zip(forces[n2_id], (dx, dy, dz))]

                for conn in self.state.connections:
                    n1, n2 = self.state.nodes[conn.source], self.state.nodes[conn.target]
                    dx, dy, dz = n2.position[0] - n1.position[0], n2.position[1] - n1.position[1], n2.position[2] - n1.position[2]
                    dist = max(math.sqrt(dx*dx + dy*dy + dz*dz), 0.1)
                    force = dist * conn.strength * 3.0  # Attraction
                    forces[conn.source] = [f + d * force / dist for f, d in zip(forces[conn.source], (dx, dy, dz))]
                    forces[conn.target] = [f - d * force / dist for f, d in zip(forces[conn.target], (dx, dy, dz))]

                for node_id, force in forces.items():
                    node = self.state.nodes[node_id]
                    mag = math.sqrt(sum(f * f for f in force))
                    if mag > 5.0:
                        force = [f * 5.0 / mag for f in force]
                    node.position = tuple(p + f * node.stability * 0.8 for p, f in zip(node.position, force))

                # Update metrics
                self.state.metrics.update({
                    "node_count": len(self.state.nodes), "connection_count": len(self.state.connections),
                    "avg_energy": np.mean([n.energy for n in self.state.nodes.values()]),
                    "avg_connections": np.mean([len(n.connections) for n in self.state.nodes.values()]),
                    "simulation_time": time.time() - self.metrics["start_time"]
                })

            # Update FPS
            frame_count += 1
            if time.time() - last_metrics_update > 1.0:
                self.metrics["fps"] = frame_count / (time.time() - last_metrics_update)
                self.metrics["update_time_ms"] = (time.time() - start_time) * 1000
                frame_count = 0
                last_metrics_update = time.time()

            elapsed = time.time() - start_time
            if elapsed < 0.033:  # ~30 FPS
                time.sleep(0.033 - elapsed)

# Flask Server
class QuantumKaleidoscopeServer:
    def __init__(self, orchestrator: QuantumOrchestrator, port: int = 8080):
        self.orchestrator = orchestrator
        self.port = port
        self.app = Flask(__name__, static_folder=None, template_folder=None)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode="eventlet")
        self._register_routes()
        self._register_socketio_handlers()

    def _register_routes(self):
        @self.app.route('/')
        def index():
            return send_file('index.html')

    def _register_socketio_handlers(self):
        @self.socketio.on('connect')
        def handle_connect():
            emit('state_update', self.orchestrator.get_current_state())

        @self.socketio.on('process_text')
        def handle_process_text(data):
            text = data.get('text', '')
            if text:
                result = self.orchestrator.process_text(text)
                emit('process_result', result)

        @self.socketio.on('get_state')
        def handle_get_state():
            emit('state_update', self.orchestrator.get_current_state())

    def run(self):
        logger.info(f"Starting server on port {self.port}")
        self.socketio.run(self.app, host='0.0.0.0', port=self.port)

def main():
    parser = argparse.ArgumentParser(description="Quantum Kaleidoscope Orchestration Hub")
    parser.add_argument('--port', type=int, default=8080, help="Port to run the server on")
    parser.add_argument('--data-dir', type=str, default="./data", help="Directory for data storage")
    args = parser.parse_args()

    orchestrator = QuantumOrchestrator(data_dir=args.data_dir)
    orchestrator.start()
    server = QuantumKaleidoscopeServer(orchestrator, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        orchestrator.stop()

if __name__ == '__main__':
    main()
