#!/usr/bin/env python3
"""
Quantum-Inspired Neural Network Synchronization System
Server Node Implementation

This server implements a WebSocket-based distributed quantum-inspired neural network
that achieves emergent intelligence through non-local correlations between nodes.
"""

import asyncio
import websockets
import numpy as np
import json
import logging
import time
import uuid
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from collections import deque
import threading
import hashlib
from typing import Dict, List, Tuple, Set, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("quantum-server")

# Quantum simulation constants
HILBERT_SPACE_DIM = 32  # Dimensionality of the quantum state space
ENTANGLEMENT_THRESHOLD = 0.75  # Threshold for considering nodes entangled
MAX_COHERENCE_TIME = 120  # Maximum time (seconds) a quantum state remains coherent
DECOHERENCE_RATE = 0.05  # Rate at which quantum states decohere
TELEPORTATION_PROBABILITY = 0.3  # Probability of successful quantum teleportation

class QuantumState:
    """Represents a quantum state in the distributed system"""
    
    def __init__(self, dimension: int = HILBERT_SPACE_DIM):
        """Initialize a random quantum state"""
        self.dimension = dimension
        # Initialize to a random pure state
        state_vector = np.random.normal(0, 1, dimension) + 1j * np.random.normal(0, 1, dimension)
        self.state = state_vector / np.linalg.norm(state_vector)
        self.creation_time = time.time()
        self.fidelity = 1.0  # Initial fidelity is perfect
        self.entangled_with = set()  # IDs of entangled nodes
    
    def evolve(self, hamiltonian: np.ndarray, dt: float) -> None:
        """Evolve quantum state according to Hamiltonian dynamics"""
        # U = exp(-i*H*dt)
        # Use eigendecomposition for stability in evolution
        evolution_operator = np.exp(-1j * hamiltonian * dt)
        self.state = evolution_operator @ self.state
        # Renormalize to handle numerical errors
        self.state = self.state / np.linalg.norm(self.state)
    
    def apply_noise(self, dt: float) -> None:
        """Apply decoherence noise to the quantum state"""
        # Calculate elapsed time and apply decoherence
        elapsed_time = time.time() - self.creation_time
        decoherence = np.exp(-DECOHERENCE_RATE * elapsed_time)
        
        # Mix with random state based on decoherence
        if decoherence < 1.0:
            random_state = np.random.normal(0, 1, self.dimension) + 1j * np.random.normal(0, 1, self.dimension)
            random_state = random_state / np.linalg.norm(random_state)
            
            # Apply partial decoherence
            self.state = decoherence * self.state + np.sqrt(1 - decoherence**2) * random_state
            self.state = self.state / np.linalg.norm(self.state)
            
            # Update fidelity
            self.fidelity = decoherence
    
    def measure(self, observable: np.ndarray) -> float:
        """Measure an observable on the quantum state"""
        # <ψ|O|ψ>
        expectation = np.real(np.conjugate(self.state) @ observable @ self.state)
        return expectation
    
    def entangle_with(self, other_id: str) -> None:
        """Mark this state as entangled with another node"""
        self.entangled_with.add(other_id)
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the quantum state for transmission"""
        return {
            "real": self.state.real.tolist(),
            "imag": self.state.imag.tolist(),
            "fidelity": self.fidelity,
            "creation_time": self.creation_time,
            "entangled_with": list(self.entangled_with)
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'QuantumState':
        """Deserialize a quantum state from received data"""
        state = cls(dimension=len(data["real"]))
        state.state = np.array(data["real"]) + 1j * np.array(data["imag"])
        state.fidelity = data["fidelity"]
        state.creation_time = data["creation_time"]
        state.entangled_with = set(data["entangled_with"])
        return state

class QuantumNetworkNode:
    """Represents a node in the quantum network"""
    
    def __init__(self, node_id: str, name: str):
        self.id = node_id
        self.name = name
        self.state = QuantumState()
        self.creation_time = time.time()
        self.last_interaction = time.time()
        self.connection = None  # WebSocket connection
        self.is_active = True
        self.data_buffers = {}  # Buffer for quantum teleportation data
        self.capabilities = set()  # Node capabilities
        self.metrics = {
            "processed_messages": 0,
            "quantum_operations": 0,
            "entanglement_count": 0,
            "teleportation_attempts": 0,
            "teleportation_successes": 0
        }
    
    def update_activity(self) -> None:
        """Update node activity timestamp"""
        self.last_interaction = time.time()
    
    def is_stale(self, threshold: float = 60.0) -> bool:
        """Check if node connection is stale"""
        return (time.time() - self.last_interaction) > threshold

class QuantumNeuralNetwork:
    """Implements a quantum-inspired neural network for distributed intelligence"""
    
    def __init__(self, dimensions: Tuple[int, int] = (16, 16)):
        """Initialize the quantum neural network"""
        self.input_dim, self.output_dim = dimensions
        
        # Quantum-inspired weights with complex values
        self.weights = np.random.normal(0, 0.1, (self.output_dim, self.input_dim)) + \
                       1j * np.random.normal(0, 0.1, (self.output_dim, self.input_dim))
        
        # Activation thresholds
        self.thresholds = np.random.normal(0, 0.1, self.output_dim)
        
        # Learning parameters
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.previous_gradient = np.zeros_like(self.weights)
        
        # Memory buffer for experience replay
        self.memory_buffer = deque(maxlen=1000)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the quantum neural network"""
        # Ensure input is complex
        if not np.iscomplexobj(inputs):
            inputs = inputs.astype(complex)
        
        # Apply weights with quantum interference
        activations = self.weights @ inputs
        
        # Apply non-linear activation (quantum-inspired sigmoid)
        outputs = 1 / (1 + np.exp(-np.abs(activations) + self.thresholds.reshape(-1, 1)))
        
        # Apply phase information
        phase = np.angle(activations)
        outputs = outputs * np.exp(1j * phase)
        
        return outputs
    
    def backward(self, inputs: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """Backward pass for learning"""
        # Calculate error
        error = targets - outputs
        
        # Calculate gradient with respect to weights
        gradient = -error @ np.conjugate(inputs).T
        
        # Apply momentum
        update = self.learning_rate * gradient + self.momentum * self.previous_gradient
        self.weights -= update
        self.previous_gradient = update
        
        # Update thresholds
        threshold_gradient = -np.mean(error, axis=1).real
        self.thresholds -= self.learning_rate * threshold_gradient
    
    def train_batch(self, batch_size: int = 32) -> float:
        """Train on a batch of experiences from memory"""
        if len(self.memory_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        batch = np.random.choice(len(self.memory_buffer), batch_size, replace=False)
        batch_loss = 0.0
        
        for idx in batch:
            inputs, targets = self.memory_buffer[idx]
            outputs = self.forward(inputs)
            self.backward(inputs, outputs, targets)
            batch_loss += np.mean(np.abs(targets - outputs)**2)
        
        return batch_loss / batch_size
    
    def add_experience(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Add experience to memory buffer"""
        self.memory_buffer.append((inputs, targets))

class QuantumNetworkServer:
    """WebSocket server that manages the quantum network"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.nodes = {}  # node_id -> QuantumNetworkNode
        self.connections = {}  # connection -> node_id
        self.network_graph = nx.Graph()
        self.neural_network = QuantumNeuralNetwork()
        
        # Global quantum state
        self.global_state = QuantumState(dimension=HILBERT_SPACE_DIM * 2)
        
        # Observables for measurements
        self.observables = self._generate_observables()
        
        # Background tasks
        self.tasks = set()
        self.shutdown_event = asyncio.Event()
    
    def _generate_observables(self) -> Dict[str, np.ndarray]:
        """Generate quantum observables for measurements"""
        dim = HILBERT_SPACE_DIM
        
        # Pauli matrices (tensor product with identity matrices)
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        identity = np.eye(2)
        
        observables = {}
        
        # Generate observables for different measurements
        for i in range(dim // 2):
            # Create Pauli X observable acting on the i-th qubit
            obs_x = np.eye(1)
            for j in range(dim // 2):
                if j == i:
                    obs_x = np.kron(obs_x, sigma_x)
                else:
                    obs_x = np.kron(obs_x, identity)
            observables[f"X_{i}"] = obs_x
            
            # Create Pauli Z observable acting on the i-th qubit
            obs_z = np.eye(1)
            for j in range(dim // 2):
                if j == i:
                    obs_z = np.kron(obs_z, sigma_z)
                else:
                    obs_z = np.kron(obs_z, identity)
            observables[f"Z_{i}"] = obs_z
            
            # Create entanglement observable between i and (i+1)%dim qubits
            if i < (dim // 2) - 1:
                obs_ent = np.eye(1)
                for j in range(dim // 2):
                    if j == i:
                        obs_ent = np.kron(obs_ent, sigma_x)
                    elif j == (i + 1):
                        obs_ent = np.kron(obs_ent, sigma_x)
                    else:
                        obs_ent = np.kron(obs_ent, identity)
                observables[f"Ent_{i}_{i+1}"] = obs_ent
        
        # Global observable for network-wide measurements
        observables["Global"] = np.random.normal(0, 1, (dim, dim))
        observables["Global"] = (observables["Global"] + observables["Global"].T.conj()) / 2  # Make Hermitian
        
        return observables
    
    def register_node(self, node_id: str, name: str, connection) -> QuantumNetworkNode:
        """Register a new node in the network"""
        if node_id in self.nodes:
            # Node already exists, update connection
            node = self.nodes[node_id]
            node.connection = connection
            node.update_activity()
            node.is_active = True
        else:
            # Create new node
            node = QuantumNetworkNode(node_id, name)
            node.connection = connection
            self.nodes[node_id] = node
            self.network_graph.add_node(node_id, name=name)
        
        self.connections[connection] = node_id
        logger.info(f"Node registered: {name} ({node_id})")
        return node
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the network"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.is_active = False
            
            # Remove the connection mapping
            if node.connection in self.connections:
                del self.connections[node.connection]
            
            # Don't delete the node yet, mark it as inactive
            logger.info(f"Node marked inactive: {node.name} ({node_id})")
    
    def get_node_by_connection(self, connection) -> Optional[QuantumNetworkNode]:
        """Get a node by its WebSocket connection"""
        node_id = self.connections.get(connection)
        return self.nodes.get(node_id)
    
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[Set[str]] = None) -> None:
        """Broadcast a message to all active nodes"""
        exclude = exclude or set()
        message_json = json.dumps(message)
        
        for node_id, node in self.nodes.items():
            if node.is_active and node_id not in exclude and node.connection is not None:
                try:
                    await node.connection.send(message_json)
                except Exception as e:
                    logger.error(f"Error broadcasting to {node.name} ({node_id}): {e}")
                    node.is_active = False
    
    def calculate_entanglement(self, node1_id: str, node2_id: str) -> float:
        """Calculate entanglement between two nodes"""
        if node1_id not in self.nodes or node2_id not in self.nodes:
            return 0.0
        
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        
        # Calculate quantum fidelity between states as measure of entanglement
        state1 = node1.state.state
        state2 = node2.state.state
        
        fidelity = np.abs(np.vdot(state1, state2))**2
        return fidelity
    
    def update_network_graph(self) -> None:
        """Update network graph based on quantum entanglement"""
        # Clear existing edges
        self.network_graph.clear_edges()
        
        # Calculate entanglement between all active node pairs
        active_nodes = [node_id for node_id, node in self.nodes.items() if node.is_active]
        
        for i, node1_id in enumerate(active_nodes):
            for node2_id in active_nodes[i+1:]:
                entanglement = self.calculate_entanglement(node1_id, node2_id)
                
                if entanglement > ENTANGLEMENT_THRESHOLD:
                    # Nodes are entangled, add edge
                    self.network_graph.add_edge(node1_id, node2_id, weight=entanglement)
                    
                    # Update node entanglement records
                    self.nodes[node1_id].state.entangle_with(node2_id)
                    self.nodes[node2_id].state.entangle_with(node1_id)
                    
                    # Update metrics
                    self.nodes[node1_id].metrics["entanglement_count"] += 1
                    self.nodes[node2_id].metrics["entanglement_count"] += 1
    
    def create_hamiltonian(self) -> np.ndarray:
        """Create a Hamiltonian based on the network state"""
        dim = HILBERT_SPACE_DIM
        
        # Start with a random Hermitian matrix as base Hamiltonian
        H_base = np.random.normal(0, 1, (dim, dim)) + 1j * np.random.normal(0, 1, (dim, dim))
        H_base = (H_base + H_base.conj().T) / 2  # Make Hermitian
        
        # Add terms based on network connectivity
        H_network = np.zeros((dim, dim), dtype=complex)
        
        # Map nodes to indices in the Hamiltonian
        active_nodes = [node_id for node_id, node in self.nodes.items() if node.is_active]
        node_indices = {node_id: i % dim for i, node_id in enumerate(active_nodes)}
        
        # Add interaction terms for entangled nodes
        for node1_id, node2_id, edge_data in self.network_graph.edges(data=True):
            if node1_id in node_indices and node2_id in node_indices:
                i = node_indices[node1_id]
                j = node_indices[node2_id]
                entanglement = edge_data.get('weight', 0.5)
                
                # Add interaction term
                interaction_strength = -entanglement  # Negative for attraction
                
                # Simple interaction: -J|i⟩⟨j| - J|j⟩⟨i|
                H_network[i, j] += interaction_strength
                H_network[j, i] += interaction_strength
        
        # Combine base and network Hamiltonians
        H = 0.7 * H_base + 0.3 * H_network
        
        # Ensure Hermitian
        H = (H + H.conj().T) / 2
        
        return H
    
    async def process_message(self, node_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a message from a node"""
        if node_id not in self.nodes:
            return {"type": "error", "error": "Node not registered"}
        
        node = self.nodes[node_id]
        node.update_activity()
        node.metrics["processed_messages"] += 1
        
        msg_type = message.get("type", "")
        
        # Handle different message types
        if msg_type == "ping":
            return {"type": "pong", "timestamp": time.time()}
        
        elif msg_type == "quantum_state_update":
            # Update node's quantum state
            if "state" in message:
                try:
                    node.state = QuantumState.deserialize(message["state"])
                    node.metrics["quantum_operations"] += 1
                    return {"type": "quantum_state_ack"}
                except Exception as e:
                    logger.error(f"Error deserializing quantum state: {e}")
                    return {"type": "error", "error": "Invalid quantum state data"}
        
        elif msg_type == "entanglement_request":
            # Request entanglement with another node
            target_id = message.get("target_id")
            if target_id in self.nodes and self.nodes[target_id].is_active:
                target_node = self.nodes[target_id]
                
                # Create entangled states
                self._create_entangled_pair(node, target_node)
                
                # Notify both nodes
                entanglement = self.calculate_entanglement(node_id, target_id)
                
                await self.broadcast(
                    {
                        "type": "entanglement_created", 
                        "node1_id": node_id, 
                        "node2_id": target_id,
                        "entanglement": entanglement
                    },
                    exclude={node_id, target_id}
                )
                
                return {
                    "type": "entanglement_success", 
                    "target_id": target_id,
                    "entanglement": entanglement
                }
            else:
                return {"type": "error", "error": "Target node not available"}
        
        elif msg_type == "teleport_data":
            # Quantum teleportation of data
            target_id = message.get("target_id")
            data = message.get("data")
            
            if target_id in self.nodes and self.nodes[target_id].is_active and data:
                target_node = self.nodes[target_id]
                
                # Simulate quantum teleportation
                success = self._simulate_teleportation(node, target_node, data)
                node.metrics["teleportation_attempts"] += 1
                
                if success:
                    node.metrics["teleportation_successes"] += 1
                    return {"type": "teleport_success", "target_id": target_id}
                else:
                    return {"type": "teleport_failure", "reason": "Teleportation failed"}
            else:
                return {"type": "error", "error": "Teleportation target not available"}
        
        elif msg_type == "neural_net_train":
            # Train the quantum neural network
            inputs = np.array(message.get("inputs", []))
            targets = np.array(message.get("targets", []))
            
            if inputs.size > 0 and targets.size > 0:
                # Add to experience buffer
                self.neural_network.add_experience(inputs, targets)
                
                # Train on a batch
                loss = self.neural_network.train_batch()
                
                return {"type": "neural_net_result", "loss": float(loss)}
            else:
                return {"type": "error", "error": "Invalid training data"}
        
        elif msg_type == "neural_net_predict":
            # Make prediction with neural network
            inputs = np.array(message.get("inputs", []))
            
            if inputs.size > 0:
                # Forward pass
                outputs = self.neural_network.forward(inputs)
                
                return {
                    "type": "neural_net_prediction", 
                    "outputs_real": outputs.real.tolist(),
                    "outputs_imag": outputs.imag.tolist()
                }
            else:
                return {"type": "error", "error": "Invalid input data"}
        
        elif msg_type == "get_network_stats":
            # Return network statistics
            return {
                "type": "network_stats",
                "active_nodes": sum(1 for node in self.nodes.values() if node.is_active),
                "total_nodes": len(self.nodes),
                "entangled_pairs": self.network_graph.number_of_edges(),
                "global_state_fidelity": self.global_state.fidelity
            }
        
        else:
            logger.warning(f"Unknown message type: {msg_type}")
            return {"type": "error", "error": "Unknown message type"}
    
    def _create_entangled_pair(self, node1: QuantumNetworkNode, node2: QuantumNetworkNode) -> None:
        """Create an entangled state pair between two nodes"""
        # Create a maximally entangled Bell state
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        
        dim = HILBERT_SPACE_DIM
        
        # Create basis state |00⟩
        state00 = np.zeros(dim, dtype=complex)
        state00[0] = 1.0
        
        # Create basis state |11⟩
        state11 = np.zeros(dim, dtype=complex)
        state11[dim-1] = 1.0
        
        # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        bell_state = (state00 + state11) / np.sqrt(2)
        
        # Assign to both nodes with slight variations to simulate noise
        noise1 = np.random.normal(0, 0.05, dim) + 1j * np.random.normal(0, 0.05, dim)
        noise2 = np.random.normal(0, 0.05, dim) + 1j * np.random.normal(0, 0.05, dim)
        
        state1 = bell_state + noise1
        state2 = bell_state + noise2
        
        # Normalize
        state1 = state1 / np.linalg.norm(state1)
        state2 = state2 / np.linalg.norm(state2)
        
        # Set states
        node1.state.state = state1
        node2.state.state = state2
        
        # Mark as entangled
        node1.state.entangle_with(node2.id)
        node2.state.entangle_with(node1.id)
    
    def _simulate_teleportation(self, sender: QuantumNetworkNode, receiver: QuantumNetworkNode, data: Any) -> bool:
        """Simulate quantum teleportation of data"""
        # Check if nodes are entangled
        entanglement = self.calculate_entanglement(sender.id, receiver.id)
        
        if entanglement < ENTANGLEMENT_THRESHOLD:
            return False
        
        # Success probability based on entanglement and teleportation probability
        success_prob = entanglement * TELEPORTATION_PROBABILITY
        
        if np.random.random() < success_prob:
            # Teleportation succeeds
            # In real quantum teleportation, the quantum state is transferred, not classical data
            # Here we simulate by adding the data to receiver's buffer
            
            # Hash the data to simulate a quantum fingerprint
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, sort_keys=True)
            else:
                data_str = str(data)
            
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()
            
            # Store in receiver's buffer
            timestamp = time.time()
            receiver.data_buffers[data_hash] = {
                "data": data,
                "sender_id": sender.id,
                "timestamp": timestamp,
                "fidelity": entanglement
            }
            
            # Consume the entanglement (in quantum teleportation, entanglement is used up)
            sender.state.entangled_with.remove(receiver.id)
            receiver.state.entangled_with.remove(sender.id)
            
            return True
        else:
            # Teleportation fails
            return False
    
    async def simulation_loop(self) -> None:
        """Background task for quantum simulation"""
        logger.info("Starting quantum simulation loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Update active nodes
                active_nodes = [node for node in self.nodes.values() if node.is_active]
                
                if not active_nodes:
                    # No active nodes, wait and continue
                    await asyncio.sleep(1)
                    continue
                
                # Update network graph based on entanglement
                self.update_network_graph()
                
                # Create Hamiltonian based on network state
                hamiltonian = self.create_hamiltonian()
                
                # Evolve quantum states for each node
                for node in active_nodes:
                    # Apply quantum evolution
                    node.state.evolve(hamiltonian, dt=0.1)
                    
                    # Apply decoherence
                    node.state.apply_noise(dt=0.1)
                    
                    # Increment operation counter
                    node.metrics["quantum_operations"] += 1
                
                # Evolve global state
                self.global_state.evolve(hamiltonian, dt=0.1)
                self.global_state.apply_noise(dt=0.1)
                
                # Broadcast global state measurement results periodically
                if np.random.random() < 0.1:  # 10% chance each cycle
                    # Measure global observable
                    global_measurement = self.global_state.measure(self.observables["Global"])
                    
                    await self.broadcast({
                        "type": "global_measurement",
                        "observable": "Global",
                        "value": float(global_measurement),
                        "fidelity": float(self.global_state.fidelity)
                    })
                
                # Sleep to control simulation rate
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                await asyncio.sleep(1)
    
    async def cleanup_loop(self) -> None:
        """Background task for cleaning up inactive nodes"""
        logger.info("Starting cleanup loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Identify stale nodes
                stale_nodes = []
                
                for node_id, node in self.nodes.items():
                    if node.is_stale():
                        stale_nodes.append(node_id)
                
                # Remove stale nodes
                for node_id in stale_nodes:
                    if node_id in self.nodes:
                        logger.info(f"Removing stale node: {self.nodes[node_id].name} ({node_id})")
                        self.remove_node(node_id)
                
                # Clean up data buffers
                max_buffer_age = 600  # 10 minutes
                current_time = time.time()
                
                for node in self.nodes.values():
                    old_keys = []
                    
                    for data_hash, buffer_data in node.data_buffers.items():
                        if current_time - buffer_data["timestamp"] > max_buffer_age:
                            old_keys.append(data_hash)
                    
                    for key in old_keys:
                        del node.data_buffers[key]
                
                # Wait before next cleanup
                await asyncio.sleep(30)
                
    async def handle_connection(self, websocket, path):
        """Handle a WebSocket connection"""
        try:
            # Get initial message for identification
            init_message = await websocket.recv()
            init_data = json.loads(init_message)
            
            # Validate initial message
            if init_data.get("type") != "register":
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "First message must be registration"
                }))
                return
            
            # Extract node info
            node_id = init_data.get("node_id", str(uuid.uuid4()))
            node_name = init_data.get("name", f"Node-{node_id[:8]}")
            
            # Register node
            node = self.register_node(node_id, node_name, websocket)
            
            # Send welcome message with network state
            await websocket.send(json.dumps({
                "type": "welcome",
                "node_id": node_id,
                "network_size": len(self.nodes),
                "active_nodes": sum(1 for n in self.nodes.values() if n.is_active),
                "global_state_fidelity": self.global_state.fidelity
            }))
            
            # Process messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.process_message(node_id, data)
                    if response:
                        await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "Invalid JSON"
                    }))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": f"Internal error: {str(e)}"
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Connection handler error: {e}")
        finally:
            # Get node and remove it when connection is closed
            node = self.get_node_by_connection(websocket)
            if node:
                self.remove_node(node.id)
    
    async def run(self):
        """Start the WebSocket server"""
        logger.info(f"Starting Quantum Network Server on {self.host}:{self.port}")
        
        # Start background tasks
        simulation_task = asyncio.create_task(self.simulation_loop())
        cleanup_task = asyncio.create_task(self.cleanup_loop())
        
        self.tasks.add(simulation_task)
        self.tasks.add(cleanup_task)
        
        # Start WebSocket server
        async with websockets.serve(self.handle_connection, self.host, self.port):
            await self.shutdown_event.wait()
        
        # Cancel background tasks
        for task in self.tasks:
            task.cancel()
        
        logger.info("Server shutting down")
    
    def shut_down(self):
        """Shut down the server"""
        self.shutdown_event.set()