#!/usr/bin/env python3
"""
Setup script for Kaleidoscope AI Frontend
Creates the frontend directory and saves the HTML file
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Setup")

def main():
    # Create frontend directory if it doesn't exist
    frontend_dir = os.path.join(os.getcwd(), "frontend")
    os.makedirs(frontend_dir, exist_ok=True)
    logger.info(f"Frontend directory: {frontend_dir}")
    
    # Path to index.html
    index_path = os.path.join(frontend_dir, "index.html")
    
    # Write the HTML content to the file
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(FRONTEND_HTML)
        
    logger.info(f"Frontend HTML saved to {index_path} ({len(FRONTEND_HTML)} bytes)")
    logger.info("Setup complete! Now run 'python kaleidoscope.py' to start the platform")

# The HTML content will be defined below
FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kaleidoscope AI - Quantum Visualization</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary: #3a0ca3;
            --secondary: #4cc9f0;
            --accent: #f72585;
            --dark: #101020;
            --light: #ffffff;
            --success: #06d6a0;
            --warning: #ffd166;
            --danger: #ef476f;
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Arial', sans-serif;
        }
        body {
            overflow: hidden;
            background: radial-gradient(circle at center, #1a1a3a 0%, #000020 100%);
        }
        #visualization {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
        }
        #particle-layer {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 2;
            pointer-events: none;
            opacity: 0.7;
        }
        #ui-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 3;
            pointer-events: none;
        }
        .ui-element {
            pointer-events: auto;
        }
        #control-panel {
            position: fixed;
            top: 20px;
            right: 20px;
            width: 300px;
            background: rgba(16, 16, 32, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid var(--secondary);
            padding: 15px;
            color: var(--light);
            z-index: 10;
            box-shadow: 0 0 20px rgba(76, 201, 240, 0.3);
            transform: translateX(320px);
            transition: transform 0.3s ease;
        }
        #control-panel.visible {
            transform: translateX(0);
        }
        #control-panel h3 {
            margin-bottom: 15px;
            color: var(--secondary);
            font-size: 16px;
            border-bottom: 1px solid rgba(76, 201, 240, 0.3);
            padding-bottom: 8px;
        }
        .control-row {
            margin-bottom: 12px;
        }
        .control-row label {
            display: block;
            margin-bottom: 5px;
            font-size: 14px;
            color: rgba(255, 255, 255, 0.8);
        }
        .slider-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .slider-container input {
            flex: 1;
        }
        .slider-container .value {
            width: 40px;
            text-align: center;
            font-size: 12px;
            color: var(--secondary);
        }
        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 5px;
            background: rgba(76, 201, 240, 0.2);
            border-radius: 3px;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            height: 15px;
            width: 15px;
            border-radius: 50%;
            background: var(--secondary);
            cursor: pointer;
        }
        .button-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 15px;
        }
        .ctrl-btn {
            background: rgba(58, 12, 163, 0.4);
            border: 1px solid var(--primary);
            color: var(--light);
            padding: 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        .ctrl-btn:hover {
            background: var(--primary);
        }
        .ctrl-btn.active {
            background: var(--primary);
            border-color: var(--secondary);
        }
        .colorbox {
            width: 20px;
            height: 20px;
            display: inline-block;
            border-radius: 3px;
            cursor: pointer;
            border: 2px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.2s;
        }
        .colorbox:hover {
            transform: scale(1.1);
            border-color: white;
        }
        .colorbox.active {
            border-color: white;
        }
        .colors-row {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }
        #panel-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(16, 16, 32, 0.8);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 11;
            border: 1px solid var(--secondary);
            color: var(--secondary);
            box-shadow: 0 0 10px rgba(76, 201, 240, 0.3);
        }
        #stats-panel {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(16, 16, 32, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid var(--secondary);
            padding: 10px 15px;
            color: var(--light);
            font-family: monospace;
            font-size: 12px;
            z-index: 10;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            gap: 15px;
            margin-bottom: 4px;
        }
        .stat-value {
            color: var(--secondary);
        }
        #toast {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%) translateY(100px);
            background: rgba(16, 16, 32, 0.9);
            border-left: 3px solid var(--secondary);
            color: var(--light);
            padding: 12px 20px;
            border-radius: 5px;
            font-size: 14px;
            transition: transform 0.3s ease;
            z-index: 100;
        }
        #toast.visible {
            transform: translateX(-50%) translateY(0);
        }
        #loading {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at center, #1a1a3a 0%, #000020 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            transition: opacity 0.5s ease;
        }
        #loading h2 {
            color: var(--light);
            margin-bottom: 20px;
            font-size: 24px;
        }
        .loading-cube-container {
            width: 100px;
            height: 100px;
            perspective: 800px;
            margin-bottom: 30px;
        }
        .loading-cube {
            width: 100%;
            height: 100%;
            position: relative;
            transform-style: preserve-3d;
            transform: translateZ(-50px);
            animation: loading-rotate 3s infinite linear;
        }
        .loading-face {
            position: absolute;
            width: 100px;
            height: 100px;
            border: 2px solid var(--secondary);
            background: rgba(76, 201, 240, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--secondary);
            font-size: 24px;
        }
        .loading-face:nth-child(1) { transform: rotateY(0deg) translateZ(50px); }
        .loading-face:nth-child(2) { transform: rotateY(90deg) translateZ(50px); }
        .loading-face:nth-child(3) { transform: rotateY(180deg) translateZ(50px); }
        .loading-face:nth-child(4) { transform: rotateY(-90deg) translateZ(50px); }
        .loading-face:nth-child(5) { transform: rotateX(90deg) translateZ(50px); }
        .loading-face:nth-child(6) { transform: rotateX(-90deg) translateZ(50px); }
        @keyframes loading-rotate {
            0% { transform: translateZ(-50px) rotateX(0deg) rotateY(0deg); }
            100% { transform: translateZ(-50px) rotateX(360deg) rotateY(360deg); }
        }
        #progress-bar {
            width: 300px;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 20px;
        }
        #progress-fill {
            height: 100%;
            width: 0;
            background: linear-gradient(90deg, var(--accent), var(--secondary));
            transition: width 0.5s ease;
        }
        #loading-text {
            color: rgba(255, 255, 255, 0.7);
            font-size: 14px;
            margin-top: 10px;
            font-family: monospace;
        }
        #context-menu {
            position: absolute;
            background: rgba(16, 16, 32, 0.9);
            border: 1px solid var(--secondary);
            border-radius: 5px;
            padding: 5px 0;
            min-width: 150px;
            z-index: 100;
            display: none;
        }
        .context-item {
            padding: 8px 12px;
            cursor: pointer;
            font-size: 14px;
            color: var(--light);
        }
        .context-item:hover {
            background: rgba(76, 201, 240, 0.2);
        }
        .context-divider {
            height: 1px;
            background: rgba(255, 255, 255, 0.1);
            margin: 5px 0;
        }
        #modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }
        #modal.visible {
            opacity: 1;
            pointer-events: auto;
        }
        .modal-content {
            background: rgba(26, 26, 46, 0.95);
            border: 1px solid var(--secondary);
            border-radius: 10px;
            padding: 20px;
            width: 90%;
            max-width: 500px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 1px solid rgba(76, 201, 240, 0.3);
            padding-bottom: 10px;
        }
        .modal-title {
            color: var(--secondary);
            font-size: 18px;
        }
        .modal-close {
            color: var(--light);
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
        }
        .modal-body {
            color: var(--light);
            margin-bottom: 20px;
            max-height: 60vh;
            overflow-y: auto;
        }
        .modal-footer {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
        }
        .modal-btn {
            background: rgba(58, 12, 163, 0.4);
            border: 1px solid var(--primary);
            color: var(--light);
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .modal-btn:hover {
            background: var(--primary);
        }
        .modal-btn.primary {
            background: var(--primary);
            border-color: var(--secondary);
        }
        .modal-btn.primary:hover {
            background: var(--secondary);
            color: var(--dark);
        }
        
        /* Ollama Chat Interface */
        #chat-panel {
            position: fixed;
            top: 20px;
            left: 20px;
            width: 350px;
            height: 70%;
            background: rgba(16, 16, 32, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid var(--secondary);
            display: flex;
            flex-direction: column;
            color: var(--light);
            z-index: 10;
            box-shadow: 0 0 20px rgba(76, 201, 240, 0.3);
            transform: translateX(-370px);
            transition: transform 0.3s ease;
        }
        
        #chat-panel.visible {
            transform: translateX(0);
        }
        
        #chat-toggle {
            position: fixed;
            top: 20px;
            left: 20px;
            background: rgba(16, 16, 32, 0.8);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 11;
            border: 1px solid var(--secondary);
            color: var(--secondary);
            box-shadow: 0 0 10px rgba(76, 201, 240, 0.3);
        }
        
        #chat-header {
            padding: 15px;
            border-bottom: 1px solid rgba(76, 201, 240, 0.3);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        #chat-header h3 {
            color: var(--secondary);
            font-size: 16px;
            margin: 0;
        }
        
        #model-select {
            background: rgba(16, 16, 32, 0.6);
            border: 1px solid rgba(76, 201, 240, 0.5);
            color: var(--light);
            padding: 5px;
            border-radius: 3px;
        }
        
        #chat-body {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .chat-message {
            max-width: 85%;
            padding: 10px 15px;
            border-radius: 12px;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        
        .chat-message.user {
            background: rgba(58, 12, 163, 0.6);
            align-self: flex-end;
            border-bottom-right-radius: 0;
        }
        
        .chat-message.ai {
            background: rgba(76, 201, 240, 0.3);
            align-self: flex-start;
            border-bottom-left-radius: 0;
        }
        
        #chat-input-container {
            padding: 15px;
            border-top: 1px solid rgba(76, 201, 240, 0.3);
        }
        
        #chat-form {
            display: flex;
            gap: 10px;
        }
        
        #chat-input {
            flex: 1;
            background: rgba(16, 16, 32, 0.6);
            border: 1px solid rgba(76, 201, 240, 0.5);
            color: var(--light);
            padding: 10px 15px;
            border-radius: 20px;
            outline: none;
        }
        
        #chat-input:focus {
            border-color: var(--secondary);
        }
        
        #chat-submit {
            background: var(--secondary);
            color: var(--dark);
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
        }
        
        .typing-indicator {
            display: inline-flex;
            align-items: center;
            padding: 8px 15px;
            background: rgba(76, 201, 240, 0.2);
            border-radius: 12px;
            margin-top: 5px;
        }
        
        .typing-indicator span {
            height: 8px;
            width: 8px;
            border-radius: 50%;
            background: var(--secondary);
            margin: 0 2px;
            display: inline-block;
            animation: typing-bounce 1.3s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing-bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        .node-info {
            position: absolute;
            background: rgba(16, 16, 32, 0.9);
            border: 1px solid var(--secondary);
            padding: 5px 10px;
            border-radius: 4px;
            color: var(--light);
            font-size: 12px;
            z-index: 15;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
        }

        @media (max-width: 768px) {
            #control-panel, #chat-panel {
                width: 280px;
            }
            .button-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div id="loading">
        <div class="loading-cube-container">
            <div class="loading-cube">
                <div class="loading-face"><i class="fas fa-atom"></i></div>
                <div class="loading-face"><i class="fas fa-brain"></i></div>
                <div class="loading-face"><i class="fas fa-project-diagram"></i></div>
                <div class="loading-face"><i class="fas fa-cube"></i></div>
                <div class="loading-face"><i class="fas fa-network-wired"></i></div>
                <div class="loading-face"><i class="fas fa-microchip"></i></div>
            </div>
        </div>
        <h2>Initializing Quantum Visualization</h2>
        <div id="progress-bar">
            <div id="progress-fill"></div>
        </div>
        <div id="loading-text">Loading core components...</div>
    </div>
    
    <canvas id="visualization"></canvas>
    <canvas id="particle-layer"></canvas>
    
    <div id="ui-overlay">
        <!-- Chat Panel Toggle -->
        <div id="chat-toggle" class="ui-element">
            <i class="fas fa-comment"></i>
        </div>
        
        <!-- Chat Panel -->
        <div id="chat-panel" class="ui-element">
            <div id="chat-header">
                <h3>Ollama AI Chat</h3>
                <div>
                    <select id="model-select">
                        <option value="llama2">Llama2</option>
                        <option value="mistral">Mistral</option>
                        <option value="gemma">Gemma</option>
                    </select>
                </div>
            </div>
            <div id="chat-body">
                <div class="chat-message ai">
                    Hello! I'm your AI assistant. Ask me anything or give me a topic to visualize in the quantum thought space.
                </div>
            </div>
            <div id="chat-input-container">
                <form id="chat-form">
                    <input type="text" id="chat-input" placeholder="Type your message..." autocomplete="off">
                    <button type="submit" id="chat-submit">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </form>
            </div>
        </div>
        
        <!-- Control Panel Toggle -->
        <div id="panel-toggle" class="ui-element">
            <i class="fas fa-cog"></i>
        </div>
        
        <!-- Control Panel -->
        <div id="control-panel" class="ui-element">
            <h3>Thought Space Controls</h3>
            
            <div class="control-row">
                <label>Mind Space Size</label>
                <div class="slider-container">
                    <input type="range" id="size-slider" min="5" max="30" value="15">
                    <div class="value" id="size-value">15</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Concept Density</label>
                <div class="slider-container">
                    <input type="range" id="density-slider" min="1" max="20" value="8">
                    <div class="value" id="density-value">8</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Neural Energy</label>
                <div class="slider-container">
                    <input type="range" id="energy-slider" min="0" max="100" value="60">
                    <div class="value" id="energy-value">60</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Association Strength</label>
                <div class="slider-container">
                    <input type="range" id="connection-slider" min="1" max="15" value="5">
                    <div class="value" id="connection-value">5</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Thought Flow Speed</label>
                <div class="slider-container">
                    <input type="range" id="rotation-slider" min="0" max="10" value="2">
                    <div class="value" id="rotation-value">2</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Concept Type</label>
                <div class="colors-row">
                    <div class="colorbox active" data-color="#f72585" data-type="memory" style="background-color: #f72585;" title="Memory"></div>
                    <div class="colorbox" data-color="#4cc9f0" data-type="logic" style="background-color: #4cc9f0;" title="Logic"></div>
                    <div class="colorbox" data-color="#7209b7" data-type="creative" style="background-color: #7209b7;" title="Creative"></div>
                    <div class="colorbox" data-color="#06d6a0" data-type="sensory" style="background-color: #06d6a0;" title="Sensory"></div>
                    <div class="colorbox" data-color="#ffd166" data-type="emotional" style="background-color: #ffd166;" title="Emotional"></div>
                </div>
            </div>
            
            <div class="button-grid">
                <button class="ctrl-btn" id="reset-btn">Reset Thoughts</button>
                <button class="ctrl-btn" id="add-nodes-btn">Add Concepts</button>
                <button class="ctrl-btn" id="entangle-btn">Connect Ideas</button>
                <button class="ctrl-btn" id="explosion-btn">Insight Burst</button>
                <button class="ctrl-btn" id="wireframe-btn">Toggle Framework</button>
                <button class="ctrl-btn" id="glow-btn">Toggle Activation</button>
                <button class="ctrl-btn" id="save-btn">Save Image</button>
                <button class="ctrl-btn" id="fullscreen-btn">Fullscreen</button>
            </div>
        </div>
        
        <!-- Stats Panel -->
        <div id="stats-panel" class="ui-element">
            <div class="stat-row">
                <div class="stat-label">FPS:</div>
                <div class="stat-value" id="fps-value">0</div>
            </div>
            <div class="stat-row">
                <div class="stat-label">Concepts:</div>
                <div class="stat-value" id="nodes-value">0</div>
            </div>
            <div class="stat-row">
                <div class="stat-label">Connections:</div>
                <div class="stat-value" id="connections-value">0</div>
            </div>
            <div class="stat-row">
                <div class="stat-label">Energy:</div>
                <div class="stat-value" id="energy-stat-value">60%</div>
            </div>
            <div class="stat-row">
                <div class="stat-label">Model:</div>
                <div class="stat-value" id="model-stat-value">llama2</div>
            </div>
        </div>
    </div>
    
    <div id="toast"></div>
    <div id="node-info" class="node-info"></div>
    
    <div id="context-menu" class="ui-element">
        <div class="context-item" id="ctx-add-node">Add Concept Here</div>
        <div class="context-item" id="ctx-clear-area">Clear Nearby Concepts</div>
        <div class="context-divider"></div>
        <div class="context-item" id="ctx-explode-from-here">Insight Burst Here</div>
        <div class="context-item" id="ctx-create-cluster">Create Thought Cluster</div>
        <div class="context-item" id="ctx-query-concept">Query This Concept</div>
    </div>
    
    <div id="modal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title">Quantum Visualization</div>
                <button class="modal-close">&times;</button>
            </div>
            <div class="modal-body" id="modal-body">
                Modal content here
            </div>
            <div class="modal-footer">
                <button class="modal-btn" id="modal-cancel">Cancel</button>
                <button class="modal-btn primary" id="modal-confirm">Confirm</button>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (() => {
        // Core system state
        const state = {
            cube: {
                size: 15,
                density: 8,
                rotationSpeed: 2,
                wireframe: true,
                color: 0x4cc9f0
            },
            nodes: {
                color: 0xf72585,
                glowEnabled: true,
                connectionThreshold: 5,
                entangled: false
            },
            energy: 60,
            running: true,
            mousePosition: { x: 0, y: 0 },
            mouseWorldPosition: new THREE.Vector3(),
            fpsSamples: [],
            showContextMenu: false,
            contextMenuPosition: { x: 0, y: 0 },
            selectedNode: null,
            explosion: {
                active: false,
                position: new THREE.Vector3(),
                timer: 0,
                duration: 3000, // ms
                strength: 1
            },
            lastClickTime: 0,
            isDragging: false,
            previousMousePosition: { x: 0, y: 0 },
            // AI-related properties
            ai: {
                connected: false,
                model: "llama2",
                thinking: false,
                history: [],
                conceptMap: new Map(),
                nodeTypes: {
                    memory: { color: 0xf72585, description: "Memory concept" },
                    logic: { color: 0x4cc9f0, description: "Logic concept" },
                    creative: { color: 0x7209b7, description: "Creative concept" },
                    sensory: { color: 0x06d6a0, description: "Sensory concept" },
                    emotional: { color: 0xffd166, description: "Emotional concept" }
                },
                currentNodeType: "memory",
                socket: null
            }
        };
        
        // Collection of 3D objects
        const objects = {
            nodes: [],
            connections: [],
            cube: null,
            cubeWireframe: null,
            raycaster: new THREE.Raycaster(),
            scene: null,
            camera: null,
            renderer: null,
            particleCanvas: null,
            particleContext: null,
            particles: [],
            clock: new THREE.Clock(),
            pointLight: null
        };
        
        // UI interactions and state
        const ui = {
            controlPanelVisible: false,
            chatPanelVisible: false,
            sliders: {},
            values: {},
            buttons: {},
            toastTimeout: null,
            modalCallback: null,
            colorBoxes: [],
            time: 0
        };
        
        /**
         * Initialize the WebSocket connection to the Kaleidoscope AI backend
         */
        function connectToBridge() {
            try {
                // Get the host from current window location
                const host = window.location.hostname || 'localhost';
                const port = 8700; // Default bridge port
                
                const wsUrl = `ws://${host}:${port}`;
                console.log(`Connecting to bridge at: ${wsUrl}`);
                
                state.ai.socket = new WebSocket(wsUrl);
                
                // Connection opened
                state.ai.socket.addEventListener('open', (event) => {
                    console.log('Connected to Kaleidoscope Bridge');
                    
                    // Send initialization message
                    state.ai.socket.send(JSON.stringify({
                        type: 'init',
                        client_type: 'frontend',
                        version: '1.0.0'
                    }));
                    
                    state.ai.connected = true;
                    showToast('Connected to Kaleidoscope Bridge');
                });
                
                // Listen for messages
                state.ai.socket.addEventListener('message', (event) => {
                    try {
                        const message = JSON.parse(event.data);
                        processBridgeMessage(message);
                    } catch (e) {
                        console.error('Error processing bridge message:', e);
                    }
                });
                
                // Connection closed
                state.ai.socket.addEventListener('close', (event) => {
                    console.log('Disconnected from Kaleidoscope Bridge');
                    state.ai.connected = false;
                    showToast('Disconnected from bridge, reconnecting...', 'warning');
                    
                    // Try to reconnect after 5 seconds
                    setTimeout(connectToBridge, 5000);
                });
                
                // Connection error
                state.ai.socket.addEventListener('error', (event) => {
                    console.error('WebSocket error:', event);
                    state.ai.connected = false;
                    
                    // Don't attempt to reconnect here - the close handler will do it
                });
                
            } catch (error) {
                console.error('Error connecting to bridge:', error);
                showToast('Bridge connection error - will retry soon', 'error');
                
                // Try to reconnect after 5 seconds
                setTimeout(connectToBridge, 5000);
            }
        }
        
        /**
         * Process messages from the Kaleidoscope Bridge
         */
        function processBridgeMessage(message) {
            console.log('Bridge message:', message);
            
            if (!message || !message.type) {
                console.error('Invalid message format:', message);
                return;
            }
            
            const type = message.type;
            const data = message.data || {};
            
            switch (type) {
                case 'state':
                    // Update visualization with state from server
                    updateVisualizationFromState(data);
                    break;
                    
                case 'update':
                    // Handle incremental updates
                    updateVisualizationFromState(data);
                    break;
                    
                case 'chat_response':
                    // Handle AI response
                    if (data.text) {
                        addAIChatMessage(data.text);
                        
                        // Extract key concepts from AI response to visualize
                        const concepts = extractConcepts(data.text);
                        visualizeConcepts(concepts);
                    }
                    
                    // Set AI thinking to false
                    state.ai.thinking = false;
                    removeTypingIndicator();
                    break;
                    
                case 'pong':
                    // Server is alive (response to ping)
                    break;
                    
                default:
                    console.log('Unhandled message type:', type);
            }
        }
        
        /**
         * Extract key concepts from AI text response
         */
        function extractConcepts(text) {
            // Simple extraction based on capitalized words and phrases
            // In a real app this would use NLP or more sophisticated parsing
            
            const concepts = [];
            const lines = text.split(/[.!?\\n]/);
            
            // Find capitalized words and phrases that might be concepts
            const conceptRegex = /\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b/g;
            const matches = text.match(conceptRegex) || [];
            
            // Add unique matches to concepts
            const uniqueMatches = [...new Set(matches.map(m => m.toLowerCase()))];
            
            uniqueMatches.forEach(match => {
                if (match.length > 3) { // Filter out short words
                    concepts.push({
                        text: match,
                        importance: Math.random() * 0.5 + 0.5, // randomize importance
                        type: getRandomNodeType()
                    });
                }
            });
            
            // If we didn't find enough concepts, extract some important words
            if (concepts.length < 5) {
                const words = text.split(/\s+/);
                const importantWords = words.filter(word => 
                    word.length > 5 && 
                    !word.match(/^(the|and|that|this|with|from|your|their|have|been|would|could|should|will|into|about)$/i)
                );
                
                // Add unique important words
                const uniqueWords = [...new Set(importantWords.map(w => w.toLowerCase()))];
                
                for (let i = 0; i < Math.min(5, uniqueWords.length); i++) {
                    const word = uniqueWords[i].replace(/[.,;:!?()]/g, '');
                    if (word.length > 5) {
                        concepts.push({
                            text: word,
                            importance: Math.random() * 0.4 + 0.3,
                            type: getRandomNodeType()
                        });
                    }
                }
            }
            
            return concepts.slice(0, 10); // Limit to 10 concepts
        }
        
        /**
         * Get a random node type for a concept
         */
        function getRandomNodeType() {
            const types = Object.keys(state.ai.nodeTypes);
            return types[Math.floor(Math.random() * types.length)];
        }
        
        /**
         * Visualize concepts by creating nodes
         */
        function visualizeConcepts(concepts) {
            if (concepts.length === 0) return;
            
            // Create an explosion effect for new thoughts
            createExplosionEffect(new THREE.Vector3(0, 0, 0), 1.0);
            
            // Calculate positions in a spherical pattern
            const radius = state.cube.size * 0.4;
            
            concepts.forEach((concept, index) => {
                const angle = (index / concepts.length) * Math.PI * 2;
                const height = (Math.random() - 0.5) * radius;
                
                // Create spiral pattern
                const x = Math.cos(angle) * radius * 0.8;
                const y = height;
                const z = Math.sin(angle) * radius * 0.8;
                
                const position = new THREE.Vector3(x, y, z);
                
                // Get color from concept type
                const typeInfo = state.ai.nodeTypes[concept.type];
                state.nodes.color = typeInfo.color;
                
                // Create node
                const node = addNode(position);
                
                // Store concept data
                node.userData.concept = concept;
                node.userData.label = concept.text;
                node.userData.type = concept.type;
                
                // Scale based on importance
                const scale = 0.8 + (concept.importance * 0.7);
                node.scale.set(scale, scale, scale);
                
                // Map concept to node
                state.ai.conceptMap.set(concept.text, node);
            });
            
            // Create connections between concepts
            setTimeout(connectConcepts, 500);
            
            // Update stats display
            updateStats();
            
            function connectConcepts() {
                // Connect related concepts
                for (let i = 0; i < concepts.length - 1; i++) {
                    const node1 = state.ai.conceptMap.get(concepts[i].text);
                    const node2 = state.ai.conceptMap.get(concepts[i+1].text);
                    
                    if (node1 && node2) {
                        createConnection(node1, node2);
                    }
                }
                
                // Add some random connections for more interesting structure
                for (let i = 0; i < concepts.length; i++) {
                    const node1 = state.ai.conceptMap.get(concepts[i].text);
                    if (!node1) continue;
                    
                    // Connect to 1-2 random other nodes
                    const numConnections = Math.floor(Math.random() * 2) + 1;
                    
                    for (let j = 0; j < numConnections; j++) {
                        const randomIndex = Math.floor(Math.random() * concepts.length);
                        if (randomIndex !== i) {
                            const node2 = state.ai.conceptMap.get(concepts[randomIndex].text);
                            if (node2) {
                                createConnection(node1, node2);
                            }
                        }
                    }
                }
                
                // Update stats after connections
                updateStats();
            }
        }
        
        /**
         * Send a chat message to Ollama
         */
        async function sendChatMessage(message) {
            if (!message.trim()) return;
            
            // Add user message to chat
            addUserChatMessage(message);
            
            // Show typing indicator
            showTypingIndicator();
            
            // Set AI thinking state
            state.ai.thinking = true;
            
            try {
                // If connected to bridge, send via WebSocket
                if (state.ai.connected && state.ai.socket && state.ai.socket.readyState === WebSocket.OPEN) {
                    state.ai.socket.send(JSON.stringify({
                        type: 'chat_message',
                        model: state.ai.model,
                        message: message
                    }));
                } else {
                    // Direct API call to Ollama if no bridge connection
                    try {
                        const response = await fetch('http://localhost:11434/api/generate', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                model: state.ai.model,
                                prompt: message,
                                stream: false
                            })
                        });
                        
                        const data = await response.json();
                        
                        if (data && data.response) {
                            // Add AI response to chat
                            addAIChatMessage(data.response);
                            
                            // Extract concepts from response
                            const concepts = extractConcepts(data.response);
                            visualizeConcepts(concepts);
                        } else {
                            showToast('Failed to get response from Ollama', 'error');
                            addAIChatMessage('Sorry, I encountered an error processing your request.');
                        }
                    } catch (error) {
                        console.error('Ollama API error:', error);
                        showToast('Error connecting to Ollama API', 'error');
                        addAIChatMessage("I'm having trouble connecting to the Ollama service. Make sure it's running on your system.");
                    }
                    
                    // Set AI thinking to false
                    state.ai.thinking = false;
                    removeTypingIndicator();
                }
            } catch (error) {
                console.error('Error sending chat message:', error);
                showToast('Error communicating with AI', 'error');
                state.ai.thinking = false;
                removeTypingIndicator();
            }
        }
        
        /**
         * Add a user message to the chat
         */
        function addUserChatMessage(message) {
            const chatBody = document.getElementById('chat-body');
            const messageElement = document.createElement('div');
            messageElement.className = 'chat-message user';
            messageElement.textContent = message;
            chatBody.appendChild(messageElement);
            
            // Scroll to bottom
            chatBody.scrollTop = chatBody.scrollHeight;
            
            // Add to history
            state.ai.history.push({ role: 'user', content: message });
        }
        
        /**
         * Add an AI message to the chat
         */
        function addAIChatMessage(message) {
            const chatBody = document.getElementById('chat-body');
            const messageElement = document.createElement('div');
            messageElement.className = 'chat-message ai';
            messageElement.textContent = message;
            chatBody.appendChild(messageElement);
            
            // Scroll to bottom
            chatBody.scrollTop = chatBody.scrollHeight;
            
            // Add to history
            state.ai.history.push({ role: 'assistant', content: message });
        }
        
        /**
         * Show typing indicator in chat
         */
        function showTypingIndicator() {
            const chatBody = document.getElementById('chat-body');
            const typingElement = document.createElement('div');
            typingElement.className = 'typing-indicator';
            typingElement.id = 'typing-indicator';
            typingElement.innerHTML = '<span></span><span></span><span></span>';
            chatBody.appendChild(typingElement);
            
            // Scroll to bottom
            chatBody.scrollTop = chatBody.scrollHeight;
        }
        
        /**
         * Remove typing indicator from chat
         */
        function removeTypingIndicator() {
            const typingIndicator = document.getElementById('typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }
        
        /**
         * Update the visualization based on state from the server
         */
        function updateVisualizationFromState(data) {
            if (!data) return;
            
            // Update energy level if provided
            if (data.energy_level !== undefined) {
                state.energy = data.energy_level * 100;
                ui.sliders.energy.value = state.energy;
                ui.values.energy.textContent = Math.round(state.energy);
                document.getElementById('energy-stat-value').textContent = `${Math.round(state.energy)}%`;
                
                // Update point light intensity
                if (objects.pointLight) {
                    objects.pointLight.intensity = state.energy / 50;
                }
            }
            
            // Update complexity/density if provided
            if (data.complexity !== undefined) {
                state.cube.density = Math.max(1, Math.round(data.complexity * 20));
                ui.sliders.density.value = state.cube.density;
                ui.values.density.textContent = state.cube.density;
            }
            
            // Update nodes if provided
            if (data.nodes && data.nodes.length > 0) {
                // Clear existing nodes
                clearNodes();
                
                // Create new nodes from data
                data.nodes.forEach(nodeData => {
                    const position = new THREE.Vector3(
                        nodeData.position.x * state.cube.size/2,
                        nodeData.position.y * state.cube.size/2,
                        nodeData.position.z * state.cube.size/2
                    );
                    
                    // Create node and set properties
                    const node = addNode(position);
                    
                    if (nodeData.id) {
                        node.userData.id = nodeData.id;
                    }
                    
                    if (nodeData.type) {
                        const typeInfo = state.ai.nodeTypes[nodeData.type] || state.ai.nodeTypes.memory;
                        node.material.color.set(typeInfo.color);
                        node.material.emissive.set(typeInfo.color);
                        node.userData.type = nodeData.type;
                        node.userData.originalColor = typeInfo.color;
                    }
                    
                    if (nodeData.energy !== undefined) {
                        node.userData.energy = nodeData.energy;
                    }
                });
                
                // Create connections if provided
                if (data.connections && data.connections.length > 0) {
                    // Find nodes by ID and connect them
                    data.connections.forEach(conn => {
                        const node1 = objects.nodes.find(n => n.userData.id === conn.source);
                        const node2 = objects.nodes.find(n => n.userData.id === conn.target);
                        
                        if (node1 && node2) {
                            createConnection(node1, node2);
                        }
                    });
                }
                
                // Update stats
                updateStats();
            }
        }
        
        /**
         * Initialize the 3D visualization system
         */
        function initVisualization() {
            try {
                // Setup main ThreeJS scene
                objects.scene = new THREE.Scene();
                objects.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                objects.camera.position.z = 30;
                
                // Create renderer with antialiasing and transparency
                objects.renderer = new THREE.WebGLRenderer({
                    canvas: document.getElementById('visualization'),
                    antialias: true,
                    alpha: true
                });
                objects.renderer.setSize(window.innerWidth, window.innerHeight);
                objects.renderer.setClearColor(0x000000, 0);
                objects.renderer.setPixelRatio(window.devicePixelRatio);
                
                // Setup particle canvas
                objects.particleCanvas = document.getElementById('particle-layer');
                objects.particleContext = objects.particleCanvas.getContext('2d');
                objects.particleCanvas.width = window.innerWidth;
                objects.particleCanvas.height = window.innerHeight;
                
                // Setup lighting
                setupLighting();
                
                // Create initial cube
                createQuantumCube();
                
                // Setup event handlers
                setupEventHandlers();
                
                // Connect to WebSocket bridge
                connectToBridge();
                
                // Begin the animation loop
                animate();
                
                showToast("Kaleidoscope AI visualization initialized");
                
                // Show panels
                document.getElementById('control-panel').classList.add('visible');
                ui.controlPanelVisible = true;
                document.getElementById('chat-panel').classList.add('visible');
                ui.chatPanelVisible = true;
                
                // Update model display
                document.getElementById('model-stat-value').textContent = state.ai.model;
            } catch (error) {
                console.error("Error initializing visualization:", error);
                showToast("Error initializing visualization", "error");
            }
        }
        
        /**
         * Setup lighting for the scene
         */
        function setupLighting() {
            // Ambient light
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
            objects.scene.add(ambientLight);
            
            // Directional light
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
            directionalLight.position.set(5, 5, 5);
            objects.scene.add(directionalLight);
            
            // Point light at center
            const pointLight = new THREE.PointLight(0x4cc9f0, 1, 100);
            pointLight.position.set(0, 0, 0);
            objects.scene.add(pointLight);
            
            // Store the point light for energy adjustments
            objects.pointLight = pointLight;
        }
        
        /**
         * Create the quantum cube and nodes
         */
        function createQuantumCube() {
            try {
                // Remove existing cube and nodes
                if (objects.cube) objects.scene.remove(objects.cube);
                if (objects.cubeWireframe) objects.scene.remove(objects.cubeWireframe);
                clearNodes();
                
                // Create cube geometry
                const geometry = new THREE.BoxGeometry(
                    state.cube.size,
                    state.cube.size,
                    state.cube.size
                );
                
                // Create wireframe
                const wireframeGeometry = new THREE.EdgesGeometry(geometry);
                const wireframeMaterial = new THREE.LineBasicMaterial({
                    color: state.cube.color,
                    transparent: true,
                    opacity: 0.7
                });
                
                objects.cubeWireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
                objects.cubeWireframe.visible = state.cube.wireframe;
                objects.scene.add(objects.cubeWireframe);
                
                // Create nodes
                createNodes();
                
                // Update UI stats
                updateStats();
            } catch (error) {
                console.error("Error creating quantum cube:", error);
                showToast("Error creating quantum structure", "error");
            }
        }
        
        /**
         * Create quantum nodes inside the cube
         */
        function createNodes() {
            try {
                const nodeCount = Math.pow(state.cube.density, 2);
                const halfSize = state.cube.size / 2 * 0.8; // 80% of half-size to keep inside
                
                for (let i = 0; i < nodeCount; i++) {
                    // Random position within cube bounds
                    const position = new THREE.Vector3(
                        (Math.random() * 2 - 1) * halfSize,
                        (Math.random() * 2 - 1) * halfSize,
                        (Math.random() * 2 - 1) * halfSize
                    );
                    
                    // Create the node with random type
                    const types = Object.keys(state.ai.nodeTypes);
                    const randType = types[Math.floor(Math.random() * types.length)];
                    const typeInfo = state.ai.nodeTypes[randType];
                    
                    // Set node color
                    state.nodes.color = typeInfo.color;
                    
                    // Create the node
                    const node = addNode(position);
                    node.userData.type = randType;
                    node.userData.originalColor = typeInfo.color;
                }
                
                showToast(`Created ${nodeCount} quantum nodes`);
            } catch (error) {
                console.error("Error creating nodes:", error);
                showToast("Error creating quantum nodes", "error");
            }
        }
        
        /**
         * Add a single node at the specified position
         */
        function addNode(position) {
            try {
                // Create sphere geometry for the node
                const geometry = new THREE.SphereGeometry(0.4, 16, 16);
                
                // Create material with physical properties
                const material = new THREE.MeshPhysicalMaterial({
                    color: state.nodes.color,
                    metalness: 0.8,
                    roughness: 0.2,
                    emissive: state.nodes.color,
                    emissiveIntensity: state.nodes.glowEnabled ? 0.5 : 0,
                    transparent: true,
                    opacity: 0.9
                });
                
                // Create the node mesh
                const node = new THREE.Mesh(geometry, material);
                node.position.copy(position);
                
                // Generate a unique ID for the node
                node.userData.id = `node-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                
                // Add physics properties to the node
                node.userData = {
                    ...node.userData,
                    velocity: new THREE.Vector3(
                        (Math.random() - 0.5) * 0.05,
                        (Math.random() - 0.5) * 0.05,
                        (Math.random() - 0.5) * 0.05
                    ),
                    energy: Math.random() * 0.5 + 0.5,
                    connections: [],
                    entangled: false,
                    originalColor: state.nodes.color,
                    phase: Math.random() * Math.PI * 2
                };
                
                // Add to scene and track
                objects.scene.add(node);
                objects.nodes.push(node);
                
                return node;
            } catch (error) {
                console.error("Error adding node:", error);
                return null;
            }
        }
        
        /**
         * Create connections between nodes based on distance
         */
        function entangleNodes() {
            try {
                // Remove existing connections
                clearConnections();
                
                // Reset node connection properties
                objects.nodes.forEach(node => {
                    node.userData.connections = [];
                    node.userData.entangled = false;
                });
                
                // Create connections based on distance
                for (let i = 0; i < objects.nodes.length; i++) {
                    for (let j = i + 1; j < objects.nodes.length; j++) {
                        const node1 = objects.nodes[i];
                        const node2 = objects.nodes[j];
                        
                        const distance = node1.position.distanceTo(node2.position);
                        
                        if (distance < state.nodes.connectionThreshold) {
                            createConnection(node1, node2);
                        }
                    }
                }
                
                state.nodes.entangled = objects.connections.length > 0;
                showToast(`Created ${objects.connections.length} thought connections`);
                updateStats();
            } catch (error) {
                console.error("Error entangling nodes:", error);
                showToast("Error creating thought connections", "error");
            }
        }
        
        /**
         * Clear all connections
         */
        function clearConnections() {
            objects.connections.forEach(conn => {
                objects.scene.remove(conn);
            });
            objects.connections = [];
        }
        
        /**
         * Create a connection between two nodes
         */
        function createConnection(node1, node2) {
            try {
                // Create line between nodes
                const points = [
                    node1.position.clone(),
                    node2.position.clone()
                ];
                
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                
                const material = new THREE.LineBasicMaterial({
                    color: state.cube.color,
                    transparent: true,
                    opacity: 0.5
                });
                
                const connection = new THREE.Line(geometry, material);
                
                // Store connection references
                connection.userData = {
                    node1: node1,
                    node2: node2
                };
                
                node1.userData.connections.push(connection);
                node2.userData.connections.push(connection);
                node1.userData.entangled = true;
                node2.userData.entangled = true;
                
                objects.scene.add(connection);
                objects.connections.push(connection);
                
                return connection;
            } catch (error) {
                console.error("Error creating connection:", error);
                return null;
            }
        }
        
        /**
         * Clear all nodes from the scene
         */
        function clearNodes() {
            try {
                // Remove connections
                clearConnections();
                
                // Remove nodes
                objects.nodes.forEach(node => {
                    objects.scene.remove(node);
                });
                objects.nodes = [];
                
                // Clear concept map
                state.ai.conceptMap.clear();
                
                updateStats();
            } catch (error) {
                console.error("Error clearing nodes:", error);
            }
        }
        
        /**
         * Create an explosion effect at a specific position
         */
        function createExplosionEffect(position, strength = 1) {
            try {
                // Set explosion state
                state.explosion.active = true;
                state.explosion.position.copy(position);
                state.explosion.timer = 0;
                state.explosion.strength = strength;
                
                // Create particle burst
                const particleCount = 50 * strength;
                
                for (let i = 0; i < particleCount; i++) {
                    const size = Math.random() * 0.3 + 0.1;
                    
                    // Random direction from center point
                    const direction = new THREE.Vector3(
                        Math.random() * 2 - 1,
                        Math.random() * 2 - 1,
                        Math.random() * 2 - 1
                    ).normalize();
                    
                    // Random speed
                    const speed = (Math.random() * 0.5 + 0.5) * strength;
                    
                    // Particle position starts at explosion center
                    const particle = {
                        position: position.clone(),
                        velocity: direction.clone().multiplyScalar(speed),
                        size: size,
                        color: state.nodes.color,
                        opacity: 1,
                        life: 1, // 0-1 lifecycle
                        decay: 0.01 + Math.random() * 0.02 // How fast it decays per frame
                    };
                    
                    objects.particles.push(particle);
                }
                
                // If connected to bridge, notify about burst
                if (state.ai.connected && state.ai.socket && state.ai.socket.readyState === WebSocket.OPEN) {
                    state.ai.socket.send(JSON.stringify({
                        type: 'control',
                        command: 'burst',
                        parameters: {
                            strength: strength,
                            position: [position.x, position.y, position.z]
                        }
                    }));
                }
                
                showToast("Neural insight burst initiated");
            } catch (error) {
                console.error("Error creating explosion:", error);
            }
        }
        
        /**
         * Draw 2D particles on the particle canvas
         */
        function drawPartic