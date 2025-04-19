import asyncio
import logging
import time
from typing import Dict, List, Any, Optional

from kaleidoscope_ai.core.NodeManager import NodeManager
from kaleidoscope_ai.core.PerspectiveManager import PerspectiveManager
from kaleidoscope_ai.core.laws import GrowthLaws
from kaleidoscope_ai.processors.GPTProcessor import GPTProcessor
from kaleidoscope_ai.network.websocket_client import CubeStateWebSocketClient

logger = logging.getLogger(__name__)

class AI_Core:
    def __init__(self, initial_seed: Optional[Dict] = None, config: Optional[Dict] = None):
        """Initialize the AI Core with optional initial seed and configuration."""
        self.config = config or {}
        self.node_manager = NodeManager(initial_seed)
        self.perspective_manager = PerspectiveManager(self.node_manager)
        self.growth_laws = GrowthLaws(self.node_manager, self.perspective_manager)
        self.gpt_processor = GPTProcessor()
        self.websocket_client = CubeStateWebSocketClient(
            uri=self.config.get("websocket_uri", "ws://localhost:8765")
        )
        self.running = False
        self.simulation_step = 0
        self.last_update_time = time.time()
        
    def get_current_simulation_state(self) -> Dict:
        """
        Collect the current state of the simulation to be sent to the quantum bridge.
        This includes node positions, connections, types, and other relevant data.
        """
        # Get all nodes and their properties
        nodes = self.node_manager.get_all_nodes()
        node_data = []
        
        for node in nodes:
            # Extract node properties
            node_info = {
                "id": node.id,
                "type": node.type,
                "position": node.position.tolist() if hasattr(node.position, "tolist") else node.position,
                "connections": [conn.target_id for conn in node.connections],
                "attributes": node.attributes,
                "stress_tensor": node.stress_tensor.tolist() if hasattr(node.stress_tensor, "tolist") else node.stress_tensor,
                "energy_level": node.energy_level,
            }
            node_data.append(node_info)
        
        # Get global network properties
        network_properties = {
            "total_energy": self.node_manager.calculate_total_energy(),
            "network_tension": self.perspective_manager.calculate_string_tension(),
            "stability_index": self.growth_laws.calculate_network_stability(),
            "growth_rate": self.growth_laws.current_growth_rate,
        }
        
        # Combine everything into a single state object
        simulation_state = {
            "step": self.simulation_step,
            "timestamp": time.time(),
            "nodes": node_data,
            "network": network_properties,
        }
        
        return simulation_state

    async def run_simulation_and_send_updates(self):
        """Run the main simulation loop and send updates to the quantum bridge."""
        self.running = True
        
        # Connect to the WebSocket server
        await self.websocket_client.connect()
        
        try:
            while self.running:
                # Run a single simulation step
                self.step_simulation()
                
                # Get the current simulation state
                cube_state = self.get_current_simulation_state()
                
                # Send the state to the quantum bridge
                await self.websocket_client.send_cube_state(cube_state)
                
                # Sleep to maintain desired simulation speed
                await asyncio.sleep(self.config.get("simulation_interval", 0.1))
        
        except Exception as e:
            logger.error(f"Error in simulation loop: {str(e)}")
        finally:
            # Close the WebSocket connection when done
            await self.websocket_client.close()
            self.running = False
    
    def step_simulation(self):
        """Run a single step of the simulation."""
        # Update the simulation step counter
        self.simulation_step += 1
        
        # Apply growth laws to evolve the network
        self.growth_laws.apply_laws()
        
        # Update perspectives based on the new network state
        self.perspective_manager.update_perspectives()
        
        # Log statistics periodically
        current_time = time.time()
        if current_time - self.last_update_time >= self.config.get("log_interval", 5.0):
            self.log_statistics()
            self.last_update_time = current_time
    
    def log_statistics(self):
        """Log statistics about the current simulation state."""
        stats = {
            "step": self.simulation_step,
            "node_count": self.node_manager.get_node_count(),
            "total_energy": self.node_manager.calculate_total_energy(),
            "network_tension": self.perspective_manager.calculate_string_tension(),
        }
        logger.info(f"Simulation stats: {stats}")
    
    def stop(self):
        """Stop the simulation loop."""
        self.running = False