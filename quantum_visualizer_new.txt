#!/usr/bin/env python3
"""
Quantum Visualizer (New)
========================

A simplified web-based 3D visualization for the Quantum Kaleidoscope system using Flask.
"""

import sys
import logging
from flask import Flask, render_template_string, jsonify
import requests
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("QuantumVisualizerNew")

app = Flask(__name__)

# HTML template with Three.js
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Kaleidoscope Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        body { margin: 0; background: #0a0a1a; }
        #canvas { width: 100%; height: 100vh; }
    </style>
</head>
<body>
    <div id="canvas"></div>
    <script>
        try {
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer();
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.getElementById('canvas').appendChild(renderer.domElement);
            
            camera.position.z = 30;
            const nodesGroup = new THREE.Group();
            scene.add(nodesGroup);
            const connectionsGroup = new THREE.Group();
            scene.add(connectionsGroup);
            
            scene.add(new THREE.AmbientLight(0x404040));
            scene.add(new THREE.DirectionalLight(0xffffff, 0.8));
            
            function fetchData() {
                fetch('/data')
                    .then(response => response.json())
                    .then(data => {
                        nodesGroup.clear();
                        connectionsGroup.clear();
                        
                        data.nodes.forEach(node => {
                            const geometry = new THREE.SphereGeometry(0.5 + node.energy, 16, 16);
                            const material = new THREE.MeshPhongMaterial({ color: 0x4488ff });
                            const mesh = new THREE.Mesh(geometry, material);
                            mesh.position.set(node.position[0], node.position[1], node.position[2]);
                            nodesGroup.add(mesh);
                        });
                        
                        data.connections.forEach(conn => {
                            const source = data.nodes.find(n => n.id === conn.source);
                            const target = data.nodes.find(n => n.id === conn.target);
                            if (source && target) {
                                const geometry = new THREE.BufferGeometry().setFromPoints([
                                    new THREE.Vector3(source.position[0], source.position[1], source.position[2]),
                                    new THREE.Vector3(target.position[0], target.position[1], target.position[2])
                                ]);
                                const material = new THREE.LineBasicMaterial({ color: 0xaaaaff, opacity: conn.strength, transparent: true });
                                connectionsGroup.add(new THREE.Line(geometry, material));
                            }
                        });
                    })
                    .catch(error => console.error('Fetch error:', error));
            }
            
            function animate() {
                requestAnimationFrame(animate);
                nodesGroup.rotation.y += 0.001;
                connectionsGroup.rotation.y += 0.001;
                renderer.render(scene, camera);
            }
            
            fetchData();
            setInterval(fetchData, 5000);
            animate();
            
            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });
        } catch (e) {
            console.error('JavaScript error:', e);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/data')
def get_data():
    try:
        api_url = f"http://localhost:{app.config['api_port']}/api/visualization"
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return jsonify({"nodes": [], "connections": []})  # Fallback data

def main():
    parser = argparse.ArgumentParser(description="Quantum Visualizer (New)")
    parser.add_argument("--api-port", type=int, default=8000, help="Main system API port")
    parser.add_argument("--port", type=int, default=8081, help="Visualizer server port")
    
    args = parser.parse_args()
    
    app.config['api_port'] = args.api_port
    logger.info(f"Starting Visualizer on port {args.port}, connecting to API at {args.api_port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)

if __name__ == "__main__":
    main()
