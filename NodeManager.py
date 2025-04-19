import numpy as np
import logging
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from kaleidoscope_ai.nodes.BaseNode import BaseNode
from kaleidoscope_ai.bridge import bridge
from kaleidoscope_ai.bridge.file_system_bridge import get_file_bridge, FileType

logger = logging.getLogger(__name__)

class NodeManager:
    def __init__(self, initial_seed: Optional[Dict] = None):
        """Initialize the NodeManager with optional initial seed data."""
        self.nodes = {}  # Dictionary of node_id -> BaseNode
        self.next_node_id = 1
        self.c_nodes = {}  # Track C-side node pointers: node_id -> c_node_ptr
        
        # Get file system bridge instance
        self.file_bridge = get_file_bridge()
        
        # Initialize with seed if provided
        if initial_seed:
            self._initialize_from_seed(initial_seed)
        else:
            # Try to load from stored state first
            if not self._load_state_from_file():
                # If no stored state, create a default seed
                self._create_default_seed()
        
        # Register callback for C-side events
        bridge.register_callback('node_created', self._handle_c_node_created)
        bridge.register_callback('nodes_connected', self._handle_c_nodes_connected)
        bridge.register_callback('memory_updated', self._handle_c_memory_updated)
        
        # Register for file change notifications
        self.file_bridge.add_file_watcher(FileType.NODE_DATA, self._handle_node_file_change)
    
    def _initialize_from_seed(self, seed_data: Dict):
        """Initialize nodes from provided seed data."""
        for node_data in seed_data.get("nodes", []):
            node_id = node_data.get("id", f"node_{self.next_node_id}")
            node_type = node_data.get("type", "standard")
            position = node_data.get("position", [0.0, 0.0, 0.0])
            attributes = node_data.get("attributes", {})
            
            # Create node on both Python and C sides
            node = BaseNode(node_id, node_type, position, attributes)
            self.nodes[node_id] = node
            
            # Create corresponding C node
            c_node_type = self._map_node_type_to_c(node_type)
            c_node_ptr = self._create_c_node(int(uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF), c_node_type)
            if c_node_ptr:
                self.c_nodes[node_id] = c_node_ptr
            
            self.next_node_id += 1
        
        # Create connections after all nodes exist
        for node_data in seed_data.get("nodes", []):
            node_id = node_data.get("id")
            if node_id in self.nodes:
                for conn_data in node_data.get("connections", []):
                    target_id = conn_data.get("target")
                    strength = conn_data.get("strength", 1.0)
                    conn_type = conn_data.get("type", "standard")
                    
                    if target_id in self.nodes:
                        # Create Python-side connection
                        self.nodes[node_id].add_connection(target_id, strength, conn_type)
                        
                        # Create C-side connection if both nodes exist in C
                        if node_id in self.c_nodes and target_id in self.c_nodes:
                            self._connect_c_nodes(node_id, target_id, strength)
    
    def _create_default_seed(self):
        """Create a minimal default configuration of nodes."""
        # Create a central node
        central_node = BaseNode(f"node_{self.next_node_id}", "core", [0.0, 0.0, 0.0])
        self.nodes[central_node.id] = central_node
        
        # Create corresponding C node
        c_node_ptr = self._create_c_node(int(uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF), 1) # 1 = core type
        if c_node_ptr:
            self.c_nodes[central_node.id] = c_node_ptr
        
        self.next_node_id += 1
        
        # Create a few surrounding nodes
        for i in range(5):
            angle = (i / 5) * 2 * np.pi
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            z = 0.0
            
            node = BaseNode(f"node_{self.next_node_id}", "standard", [x, y, z])
            self.nodes[node.id] = node
            
            # Create corresponding C node
            c_node_ptr = self._create_c_node(int(uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF), 0) # 0 = standard type
            if c_node_ptr:
                self.c_nodes[node.id] = c_node_ptr
            
            # Connect to central node on Python side
            node.add_connection(central_node.id)
            central_node.add_connection(node.id)
            
            # Connect on C side
            if node.id in self.c_nodes and central_node.id in self.c_nodes:
                self._connect_c_nodes(node.id, central_node.id, 1.0)
            
            self.next_node_id += 1
    
    def _map_node_type_to_c(self, node_type: str) -> int:
        """Map Python node type to C node type integer."""
        type_map = {
            "standard": 0,
            "core": 1,
            "memory": 2,
            "processing": 3,
            "input": 4,
            "output": 5
        }
        return type_map.get(node_type.lower(), 0)
    
    def _create_c_node(self, c_node_id: int, c_node_type: int) -> int:
        """Create a node in the C system and return its pointer."""
        try:
            c_node_ptr = bridge.lib.bridge_create_node(c_node_id, c_node_type)
            return c_node_ptr
        except Exception as e:
            logger.error(f"Failed to create C node: {e}")
            return 0
    
    def _connect_c_nodes(self, source_id: str, target_id: str, strength: float) -> bool:
        """Connect two nodes in the C system."""
        try:
            if source_id in self.c_nodes and target_id in self.c_nodes:
                result = bridge.lib.bridge_connect_nodes(
                    self.c_nodes[source_id], 
                    self.c_nodes[target_id], 
                    strength
                )
                return result == 1
            return False
        except Exception as e:
            logger.error(f"Failed to connect C nodes: {e}")
            return False
    
    def create_node(self, node_type: str = "standard", position: Optional[List[float]] = None, 
                   attributes: Optional[Dict] = None) -> BaseNode:
        """Create a new node with given properties in both Python and C systems."""
        node_id = f"node_{self.next_node_id}"
        node = BaseNode(node_id, node_type, position, attributes)
        self.nodes[node_id] = node
        
        # Create corresponding C node
        c_node_type = self._map_node_type_to_c(node_type)
        c_node_id = int(uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF)
        c_node_ptr = self._create_c_node(c_node_id, c_node_type)
        if c_node_ptr:
            self.c_nodes[node_id] = c_node_ptr
        
        # Save node to unified file system
        self._save_node_to_file(node)
        
        self.next_node_id += 1
        return node
    
    def _save_node_to_file(self, node: BaseNode) -> bool:
        """Save a node to the unified file system."""
        try:
            # Create serializable representation of the node
            node_data = {
                "id": node.id,
                "type": node.type,
                "energy_level": node.energy_level,
                "position": node.position.tolist() if hasattr(node.position, "tolist") else node.position,
                "attributes": node.attributes,
                "connections": [
                    {
                        "target_id": conn.target_id,
                        "strength": conn.strength,
                        "type": conn.type,
                        "stress": conn.stress,
                        "age": conn.age
                    } for conn in node.connections
                ],
                "stress_tensor": node.stress_tensor.tolist() if hasattr(node.stress_tensor, "tolist") else [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                "resonance_frequency": node.resonance_frequency,
                "adaptation_rate": node.adaptation_rate,
                "age": node.age
            }
            
            # Save to file system, notify C side but not Python (we already have the data)
            filename = f"{node.id}.json"
            return self.file_bridge.write_json(FileType.NODE_DATA, filename, node_data, notify_python=False)
        except Exception as e:
            logger.error(f"Error saving node {node.id} to file: {e}")
            return False
    
    def _load_node_from_file(self, node_id: str) -> Optional[BaseNode]:
        """Load a node from the unified file system."""
        try:
            filename = f"{node_id}.json"
            node_data = self.file_bridge.read_json(FileType.NODE_DATA, filename)
            
            if node_data:
                # Create BaseNode from data
                position = node_data.get("position", [0.0, 0.0, 0.0])
                attributes = node_data.get("attributes", {})
                node_type = node_data.get("type", "standard")
                
                node = BaseNode(node_id, node_type, position, attributes)
                node.energy_level = node_data.get("energy_level", 1.0)
                node.resonance_frequency = node_data.get("resonance_frequency", 1.0)
                node.adaptation_rate = node_data.get("adaptation_rate", 0.05)
                node.age = node_data.get("age", 0)
                
                # Convert stress tensor back to numpy array
                stress_tensor_data = node_data.get("stress_tensor", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
                node.stress_tensor = np.array(stress_tensor_data)
                
                # Add connections (placeholder - actual connections added after all nodes are loaded)
                self.pending_connections[node_id] = node_data.get("connections", [])
                
                return node
            
            return None
        except Exception as e:
            logger.error(f"Error loading node {node_id} from file: {e}")
            return None
    
    def _load_state_from_file(self) -> bool:
        """Load the entire node network state from files."""
        try:
            # Clear existing nodes and connections
            self.nodes = {}
            self.c_nodes = {}
            self.pending_connections = {}  # Store connections to create after all nodes loaded
            
            # Get list of node data files
            node_files = self.file_bridge.list_files(FileType.NODE_DATA)
            if not node_files:
                logger.info("No saved node files found to load")
                return False
            
            # Load nodes from files
            max_id_num = 0
            for filename in node_files:
                if filename.endswith(".json"):
                    node_id = filename[:-5]  # Remove .json extension
                    node = self._load_node_from_file(node_id)
                    
                    if node:
                        self.nodes[node_id] = node
                        
                        # Track highest node ID for next_node_id
                        if node_id.startswith("node_"):
                            try:
                                id_num = int(node_id[5:])
                                max_id_num = max(max_id_num, id_num)
                            except ValueError:
                                pass
                        
                        # Create corresponding C node
                        c_node_type = self._map_node_type_to_c(node.type)
                        c_node_id = int(uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF)
                        c_node_ptr = self._create_c_node(c_node_id, c_node_type)
                        if c_node_ptr:
                            self.c_nodes[node_id] = c_node_ptr
            
            # Set next_node_id to one more than the highest found
            self.next_node_id = max_id_num + 1
            
            # Create connections now that all nodes exist
            for node_id, connections in self.pending_connections.items():
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    
                    for conn_data in connections:
                        target_id = conn_data.get("target_id")
                        if target_id in self.nodes:
                            strength = conn_data.get("strength", 1.0)
                            conn_type = conn_data.get("type", "standard")
                            
                            # Create connection in Python
                            connection = node.add_connection(target_id, strength, conn_type)
                            
                            # Set additional properties if available
                            connection.stress = conn_data.get("stress", 0.0)
                            connection.age = conn_data.get("age", 0)
                            
                            # Create connection in C
                            if node_id in self.c_nodes and target_id in self.c_nodes:
                                self._connect_c_nodes(node_id, target_id, strength)
            
            # Clean up temporary storage
            self.pending_connections = {}
            
            logger.info(f"Loaded {len(self.nodes)} nodes from file system")
            return len(self.nodes) > 0
        
        except Exception as e:
            logger.error(f"Error loading node state from files: {e}")
            return False
    
    def save_state(self) -> bool:
        """Save the entire node network state to files."""
        try:
            success_count = 0
            error_count = 0
            
            # Save each node to its own file
            for node_id, node in self.nodes.items():
                if self._save_node_to_file(node):
                    success_count += 1
                else:
                    error_count += 1
            
            logger.info(f"Saved node state to files: {success_count} successful, {error_count} failed")
            return error_count == 0
        
        except Exception as e:
            logger.error(f"Error saving node state to files: {e}")
            return False
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its connections from both Python and C systems."""
        if node_id in self.nodes:
            # Remove connections to this node from all other nodes
            for other_node in self.nodes.values():
                other_node.remove_connection(node_id)
            
            # Remove from Python side
            del self.nodes[node_id]
            
            # Remove from C side if it exists
            if node_id in self.c_nodes:
                try:
                    # Call C function to remove node if available
                    if hasattr(bridge.lib, 'bridge_remove_node'):
                        bridge.lib.bridge_remove_node(self.c_nodes[node_id])
                    del self.c_nodes[node_id]
                except Exception as e:
                    logger.error(f"Error removing C node: {e}")
            
            # Remove from file system
            try:
                filename = f"{node_id}.json"
                self.file_bridge.delete_file(FileType.NODE_DATA, filename)
            except Exception as e:
                logger.error(f"Error deleting node file for {node_id}: {e}")
            
            return True
        return False
    
    def synchronize_with_c(self):
        """Synchronize state between Python and C systems."""
        logger.info("Synchronizing Python and C node systems")
        
        # Track successful/failed sync operations
        success_count = 0
        error_count = 0
        
        # Synchronize all nodes to the C system
        for node_id, node in self.nodes.items():
            if node_id in self.c_nodes:
                try:
                    # Convert node state to JSON for C system
                    node_data = json.dumps({
                        "id": node.id,
                        "type": node.type, 
                        "energy": node.energy_level,
                        "connections": len(node.connections),
                        "position": node.position.tolist() if hasattr(node.position, "tolist") else node.position,
                        "stress": np.trace(node.stress_tensor) if hasattr(node.stress_tensor, "tolist") else 0,
                        "attributes": node.attributes
                    })
                    
                    # Update C-side memory graph
                    if hasattr(bridge.lib, 'bridge_update_memory_graph'):
                        # Use the actual C node ID instead of generating a new UUID
                        c_node_id = self.c_nodes[node_id]
                        bridge.lib.bridge_update_memory_graph(c_node_id, node_data.encode('utf-8'))
                        success_count += 1
                        
                except Exception as e:
                    logger.error(f"Error synchronizing node {node_id}: {e}")
                    error_count += 1
        
        # Also fetch any updates from C to Python
        try:
            if hasattr(bridge.lib, 'bridge_get_updated_nodes'):
                updated_nodes_data = bridge.lib.bridge_get_updated_nodes()
                if updated_nodes_data:
                    self._process_c_node_updates(updated_nodes_data)
        except Exception as e:
            logger.error(f"Error getting updates from C system: {e}")
            error_count += 1
        
        # Save the current state to disk
        self.save_state()
        
        logger.info(f"Synchronization completed: {success_count} successful, {error_count} failed")
    
    def _handle_node_file_change(self, filename: str, is_delete: bool) -> None:
        """Handle notifications about node file changes."""
        try:
            if is_delete:
                # File was deleted, check if node needs to be removed from memory
                node_id = filename[:-5] if filename.endswith(".json") else filename
                if node_id in self.nodes:
                    # Only remove from memory if it was externally deleted
                    logger.info(f"Node file {filename} was deleted externally, removing from memory")
                    self.nodes.pop(node_id, None)
                    self.c_nodes.pop(node_id, None)
            else:
                # File was created or updated
                node_id = filename[:-5] if filename.endswith(".json") else filename
                
                # Don't reload if we already have this node (prevents circular updates)
                if node_id not in self.nodes:
                    logger.info(f"Loading new or updated node from file: {filename}")
                    node = self._load_node_from_file(node_id)
                    if node:
                        self.nodes[node_id] = node
        except Exception as e:
            logger.error(f"Error handling node file change for {filename}: {e}")
    
    # C-side event handlers
    def _handle_c_node_created(self, event_name: str, data: str):
        """Handle node creation events from C side."""
        try:
            event_data = json.loads(data)
            logger.debug(f"C node created: {event_data}")
            # Additional processing as needed
        except Exception as e:
            logger.error(f"Error handling C node creation event: {e}")
    
    def _handle_c_nodes_connected(self, event_name: str, data: str):
        """Handle node connection events from C side."""
        try:
            event_data = json.loads(data)
            logger.debug(f"C nodes connected: {event_data}")
            # Additional processing as needed
        except Exception as e:
            logger.error(f"Error handling C node connection event: {e}")
    
    def _handle_c_memory_updated(self, event_name: str, data: str):
        """Handle memory update events from C side."""
        try:
            event_data = json.loads(data)
            logger.debug(f"C memory updated: {event_data}")
            # Additional processing as needed
        except Exception as e:
            logger.error(f"Error handling C memory update event: {e}")