#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Server for Quantum Kaleidoscope
===================================
Provides HTTP API for interfacing with the Quantum Kaleidoscope engine.
"""

import sys
import logging
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required. Install with 'pip install numpy'")
    sys.exit(1)

try:
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
except ImportError:
    print("Error: Flask dependencies are required. Install with 'pip install flask flask-cors'")
    sys.exit(1)

import os
import json
import time
import threading

# --- Import QuantumKaleidoscope back at the top ---
try:
    # This should resolve correctly now because run_system.py is the entry point
    from quantum_kaleidoscope import QuantumKaleidoscope
    _QuantumKaleidoscope_imported = True
except ImportError:
    print("ERROR in api_server.py: Failed to import QuantumKaleidoscope.")
    QuantumKaleidoscope = None # Define as None to allow script loading but fail later checks
    _QuantumKaleidoscope_imported = False
# --- End Import Move ---


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("KaleidoscopeAPI")

# Create the Flask app
static_folder_path = os.path.join(os.path.dirname(__file__), "static")
if not os.path.isdir(static_folder_path):
    logger.warning(f"Static folder not found at: {static_folder_path}. Frontend might not work.")

app = Flask(__name__, static_folder=static_folder_path)
CORS(app)

# Global reference
kaleidoscope_system: Optional[QuantumKaleidoscope] = None # Use type hint

# ===================================
# Flask Routes (Unchanged from previous version)
# ===================================
@app.route("/")
def home():
    """Serve the main HTML page"""
    index_path = os.path.join(app.static_folder, "index.html")
    if not os.path.isfile(index_path):
        logger.error(f"index.html not found in static folder: {app.static_folder}")
        return jsonify({"error": "Frontend not found. 'index.html' missing from static folder."}), 404
    return send_from_directory(app.static_folder, "index.html")

@app.route("/js/<path:filename>")
def serve_js(filename):
    """Serve files from the js subfolder"""
    js_dir = os.path.join(app.static_folder, 'js')
    if not os.path.isdir(js_dir):
         logger.error(f"Static/js folder not found at: {js_dir}. Frontend might not work.")
         return jsonify({"error": "static/js folder missing"}), 404
    if ".." in filename or filename.startswith("/"):
        logger.warning(f"Attempted path traversal: {filename}")
        return jsonify({"error": "Invalid path"}), 400
    return send_from_directory(js_dir, filename)

@app.route("/api/status", methods=["GET"])
def get_status():
    """Get system status"""
    if not kaleidoscope_system:
        logger.warning("API call to /api/status before system was initialized.")
        return jsonify({"error": "System not initialized"}), 503
    try:
        status = kaleidoscope_system.get_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve system status"}), 500

@app.route("/api/visualization", methods=["GET"])
def get_visualization_data():
    """Get visualization data"""
    if not kaleidoscope_system:
        logger.warning("API call to /api/visualization before system was initialized.")
        return jsonify({"error": "System not initialized"}), 503
    try:
        viz_data = kaleidoscope_system.get_visualization_data()
        try:
            if 'nodes' in viz_data:
                 for node in viz_data.get('nodes', []):
                      if 'position' in node and isinstance(node['position'], np.ndarray):
                           node['position'] = node['position'].tolist()
        except Exception as e:
            logger.warning(f"Error converting numpy arrays in viz_data: {e}")
        return jsonify(viz_data)
    except Exception as e:
        logger.error(f"Error getting visualization data: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve visualization data"}), 500

@app.route("/api/process/text", methods=["POST"])
def process_text():
    """Process text input"""
    if not kaleidoscope_system: return jsonify({"error": "System not initialized"}), 503
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 415
    data = request.json
    if not data or "text" not in data: return jsonify({"error": "Missing 'text' in request body"}), 400
    text = data["text"]; metadata = data.get("metadata", {})
    if not isinstance(text, str): return jsonify({"error": "'text' must be a string"}), 400
    if not isinstance(metadata, dict): return jsonify({"error": "'metadata' must be an object"}), 400
    try:
        logger.info(f"Processing text input: {text[:100]}...")
        result = kaleidoscope_system.process_text(text, metadata); logger.info("Text processing complete.")
        return jsonify(result)
    except Exception as e: logger.error(f"Error processing text: {e}", exc_info=True); return jsonify({"error": "Failed to process text input"}), 500

@app.route("/api/nodes/create", methods=["POST"])
def create_node():
    """Create a new node"""
    if not kaleidoscope_system: return jsonify({"error": "System not initialized"}), 503
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 415
    data = request.json
    if not data or "features" not in data: return jsonify({"error": "Missing 'features' in request body"}), 400
    try:
        features_list = data["features"]
        if not isinstance(features_list, list): raise ValueError("'features' must be a list of numbers")
        features = np.array(features_list, dtype=float)
        position = None
        if "position" in data:
            position_list = data.get("position")
            if position_list is not None:
                if not isinstance(position_list, list): raise ValueError("'position' must be a list of numbers")
                position = np.array(position_list, dtype=float)
        energy = float(data.get("energy", 0.5)); stability = float(data.get("stability", 0.8))
        node_id = kaleidoscope_system.engine.create_node(features=features, position=position, energy=energy, stability=stability)
        logger.info(f"Created node: {node_id}")
        return jsonify({"node_id": node_id}), 201
    except (ValueError, TypeError) as e: logger.warning(f"Invalid data for node creation: {e}"); return jsonify({"error": f"Invalid data format: {e}"}), 400
    except AttributeError: logger.error("System object missing 'engine' attribute or 'create_node' method."); return jsonify({"error": "Internal server configuration error"}), 500
    except Exception as e: logger.error(f"Error creating node: {e}", exc_info=True); return jsonify({"error": "Failed to create node"}), 500

@app.route("/api/nodes/connect", methods=["POST"])
def connect_nodes():
    """Connect two nodes"""
    if not kaleidoscope_system: return jsonify({"error": "System not initialized"}), 503
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 415
    data = request.json
    if not data or "node1_id" not in data or "node2_id" not in data: return jsonify({"error": "Missing 'node1_id' or 'node2_id' in request body"}), 400
    try:
        node1_id = str(data["node1_id"]); node2_id = str(data["node2_id"]); strength = None
        if "strength" in data:
             strength_val = data.get("strength")
             if strength_val is not None: strength = float(strength_val)
        result = kaleidoscope_system.engine.connect_nodes(node1_id, node2_id, strength)
        if result: logger.info(f"Connected nodes {node1_id} and {node2_id}"); return jsonify({"success": True})
        else: logger.warning(f"Failed connecting nodes {node1_id} and {node2_id} (e.g., invalid IDs)"); return jsonify({"success": False, "message": "Failed to connect nodes (e.g., invalid IDs)"}), 400
    except (ValueError, TypeError) as e: logger.warning(f"Invalid data for node connection: {e}"); return jsonify({"error": f"Invalid data format: {e}"}), 400
    except AttributeError: logger.error("System object missing 'engine' attribute or 'connect_nodes' method."); return jsonify({"error": "Internal server configuration error"}), 500
    except Exception as e: logger.error(f"Error connecting nodes: {e}", exc_info=True); return jsonify({"error": "Failed to connect nodes"}), 500

@app.route("/api/auto-generation/start", methods=["POST"])
def start_auto_generation():
    """Start auto-generation"""
    if not kaleidoscope_system: return jsonify({"error": "System not initialized"}), 503
    data = request.json or {}
    try:
        interval = float(data.get("interval", 5.0))
        if interval <= 0: raise ValueError("Interval must be positive")
        kaleidoscope_system.start_auto_generation(interval=interval)
        return jsonify({"success": True, "message": f"Auto-generation started/resumed with interval {interval}s"})
    except (ValueError, TypeError) as e: logger.warning(f"Invalid interval for auto-generation: {e}"); return jsonify({"error": f"Invalid interval value: {e}"}), 400
    except AttributeError: logger.error("System object missing 'start_auto_generation' method."); return jsonify({"error": "Internal server configuration error"}), 500
    except Exception as e: logger.error(f"Error starting auto-generation: {e}", exc_info=True); return jsonify({"error": "Failed to start auto-generation"}), 500

@app.route("/api/auto-generation/stop", methods=["POST"])
def stop_auto_generation():
    """Stop auto-generation"""
    if not kaleidoscope_system: return jsonify({"error": "System not initialized"}), 503
    try:
        kaleidoscope_system.stop_auto_generation()
        return jsonify({"success": True, "message": "Auto-generation stopped"})
    except AttributeError: logger.error("System object missing 'stop_auto_generation' method."); return jsonify({"error": "Internal server configuration error"}), 500
    except Exception as e: logger.error(f"Error stopping auto-generation: {e}", exc_info=True); return jsonify({"error": "Failed to stop auto-generation"}), 500

@app.route("/api/simulate", methods=["POST"])
def run_simulation():
    """Run simulation steps"""
    if not kaleidoscope_system: return jsonify({"error": "System not initialized"}), 503
    data = request.json or {}
    try:
        steps = int(data.get("steps", 1))
        if steps <= 0: raise ValueError("Steps must be positive")
        if steps > 1000: raise ValueError("Number of steps exceeds limit (1000)")
        logger.info(f"Running {steps} simulation step(s)...")
        for i in range(steps): kaleidoscope_system.engine.run_simulation_step()
        logger.info(f"Finished {steps} simulation step(s).")
        return jsonify({"success": True, "message": f"Ran {steps} simulation steps"})
    except (ValueError, TypeError) as e: logger.warning(f"Invalid steps for simulation: {e}"); return jsonify({"error": f"Invalid steps value: {e}"}), 400
    except AttributeError: logger.error("System object missing 'engine' attribute or 'run_simulation_step' method."); return jsonify({"error": "Internal server configuration error"}), 500
    except Exception as e: logger.error(f"Error running simulation: {e}", exc_info=True); return jsonify({"error": "Failed to run simulation"}), 500

# ===================================
# Server Start Function
# ===================================

# Use the imported class for type hint now
def start_server(system: QuantumKaleidoscope, host: str = "0.0.0.0", port: int = 8000):
    """Start the API server"""
    global kaleidoscope_system

    # --- Debug Logs Removed ---

    # Assign the passed system to the global variable *first*
    kaleidoscope_system = system

    # Check if QuantumKaleidoscope class was imported successfully at the top
    if not _QuantumKaleidoscope_imported or QuantumKaleidoscope is None:
        logger.critical("QuantumKaleidoscope class was not imported correctly. Cannot start server.")
        return

    # Perform the check using the class imported at the top level
    if not kaleidoscope_system or not isinstance(kaleidoscope_system, QuantumKaleidoscope):
         # This check should pass now with the new entry point structure
         logger.critical(
             f"CRITICAL: Invalid system object passed to start_server. "
             f"Type received: {type(kaleidoscope_system)}. Expected {QuantumKaleidoscope}. Aborting server start."
         )
         kaleidoscope_system = None # Ensure global is None if check fails
         return

    logger.info("Quantum Kaleidoscope system object validated and initialized for API.")

    # Start Flask app
    logger.info(f"Starting Flask API server on http://{host}:{port}")
    try:
        # For production, use a proper WSGI server:
        # from waitress import serve
        # serve(app, host=host, port=port)
        app.run(host=host, port=port, debug=False, threaded=True) # threaded=True for basic concurrency
    except OSError as e:
         if "Address already in use" in str(e):
              logger.error(f"Cannot start server: Port {port} is already in use.")
         else:
              logger.error(f"OS error starting Flask server: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to start Flask server: {e}", exc_info=True)

# ===================================
# Direct Execution (Test Mode - Optional)
# ===================================
if __name__ == "__main__":
    import argparse

    if not _QuantumKaleidoscope_imported or QuantumKaleidoscope is None:
        print("\nERROR: Cannot start server in test mode because QuantumKaleidoscope class could not be imported.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Quantum Kaleidoscope API Server (Test Mode)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--dimension", type=int, default=128, help="Dimension for test system")
    args = parser.parse_args()

    print(f"\n--- Running {__file__} directly (Test Mode) ---")
    print("Initializing Quantum Kaleidoscope system (Test Instance)...")
    try:
        test_system = QuantumKaleidoscope(dimension=args.dimension)
        print("Test system created successfully.")
    except Exception as e:
        print(f"\nERROR: Failed to create QuantumKaleidoscope instance in test mode: {e}")
        sys.exit(1)

    print(f"Attempting to start API server directly on http://{args.host}:{args.port}")
    start_server(test_system, host=args.host, port=args.port)
    print("--- API Server Test Mode Ended ---")
