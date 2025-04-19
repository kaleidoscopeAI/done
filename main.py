#!/usr/bin/env python3
"""
Kaleidoscope AI System - Main Entry Point

This is the main entry point for the Kaleidoscope AI system. It initializes all
components, establishes connections between Python and C subsystems, and provides
a command-line interface for interacting with the system.

Features:
- Initializes the bridge between Python and C components
- Sets up file system for data sharing
- Creates and manages the node network and memory graph
- Provides visualization through the cube visualizer
- Supports interactive commands and tasks
"""

import os
import sys
import json
import argparse
import asyncio
import logging
import signal
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('kaleidoscope.log')
    ]
)
logger = logging.getLogger("KaleidoscopeAI")

# Import Kaleidoscope components
from kaleidoscope_ai.bridge.c_bridge import bridge
from kaleidoscope_ai.bridge.file_system_bridge import get_file_bridge, FileType
from kaleidoscope_ai.core.NodeManager import NodeManager
from kaleidoscope_ai.core.PerspectiveManager import PerspectiveManager
from kaleidoscope_ai.core.IntegratedSystem import get_integrated_system
from kaleidoscope_ai.visualization.cube_visualizer import get_visualizer

class KaleidoscopeSystem:
    """
    Main class for the Kaleidoscope AI system.
    
    This class coordinates all components of the system, handling initialization,
    shutdown, and user commands.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Kaleidoscope AI system.
        
        Args:
            config_path: Path to configuration file (JSON)
        """
        self.config = self._load_config(config_path)
        self.running = False
        self.initialized = False
        self.tasks = []
        self.integrated_system = None
        self.node_manager = None
        self.perspective_manager = None
        self.visualizer = None
        self.file_bridge = None
        
        # Create shared directories
        shared_dir = Path(self.config.get("paths", {}).get("shared_dir", "./shared"))
        shared_dir.mkdir(exist_ok=True)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Kaleidoscope AI system initialized")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file, with defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        # Default configuration
        config = {
            "paths": {
                "shared_dir": "./shared",
                "lib_dir": "./lib",
                "visualization_dir": "./visualizations"
            },
            "simulation": {
                "interval": 0.1,
                "sync_interval": 1.0
            },
            "system": {
                "max_tasks": 100,
                "max_nodes": 1000,
                "debug": False,
                "auto_visualize": True
            },
            "visualization": {
                "update_interval": 1.0,
                "node_size_factor": 10,
                "edge_width_factor": 3
            }
        }
        
        # Load from file if specified
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                
                # Merge configurations
                self._deep_update(config, file_config)
                logger.info(f"Loaded configuration from {config_path}")
            
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
        
        return config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """
        Deep update a nested dictionary.
        
        Args:
            base_dict: Dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    async def initialize(self) -> bool:
        """
        Initialize all components of the system.
        
        Returns:
            Success flag
        """
        if self.initialized:
            logger.warning("System already initialized")
            return True
        
        logger.info("Initializing Kaleidoscope AI system...")
        
        try:
            # Initialize C bridge
            lib_path = self.config.get("paths", {}).get("lib_path")
            if lib_path:
                lib_path = os.path.abspath(lib_path)
                if os.path.exists(lib_path):
                    logger.info(f"Initializing C bridge with library at {lib_path}")
                    bridge_initialized = bridge.init_bridge(lib_path)
                    if not bridge_initialized:
                        logger.warning("Failed to initialize C bridge, continuing in Python-only mode")
                else:
                    logger.warning(f"C library not found at {lib_path}, continuing in Python-only mode")
            
            # Initialize file system bridge
            shared_dir = self.config.get("paths", {}).get("shared_dir", "./shared")
            self.file_bridge = get_file_bridge(shared_dir=shared_dir)
            
            # Initialize node manager
            self.node_manager = NodeManager()
            
            # Initialize perspective manager
            self.perspective_manager = PerspectiveManager(self.node_manager)
            
            # Initialize visualizer
            self.visualizer = get_visualizer()
            vis_config = self.config.get("visualization", {})
            if "node_size_factor" in vis_config:
                self.visualizer.node_size_factor = vis_config["node_size_factor"]
            if "edge_width_factor" in vis_config:
                self.visualizer.edge_width_factor = vis_config["edge_width_factor"]
            if "update_interval" in vis_config:
                self.visualizer.update_interval = vis_config["update_interval"]
            if "output_dir" in vis_config:
                self.visualizer.output_dir = Path(vis_config["output_dir"])
                self.visualizer.output_dir.mkdir(exist_ok=True)
            
            # Initialize integrated system
            self.integrated_system = get_integrated_system(self.config)
            if not self.integrated_system:
                logger.error("Failed to initialize integrated system")
                return False
            
            # Start the integrated system
            await self.integrated_system.start()
            
            self.initialized = True
            logger.info("Kaleidoscope AI system initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing system: {e}", exc_info=True)
            return False
    
    async def start(self) -> bool:
        """
        Start the Kaleidoscope AI system.
        
        Returns:
            Success flag
        """
        if self.running:
            logger.warning("System already running")
            return True
        
        if not self.initialized:
            success = await self.initialize()
            if not success:
                logger.error("Failed to initialize system")
                return False
        
        logger.info("Starting Kaleidoscope AI system...")
        self.running = True
        
        # Start background task for system operations
        asyncio.create_task(self._system_loop())
        
        logger.info("Kaleidoscope AI system started")
        return True
    
    async def _system_loop(self) -> None:
        """Background task for system operations"""
        while self.running:
            try:
                # Update perspectives
                self.perspective_manager.update_perspectives()
                
                # Update visualizations if auto-visualize is enabled
                if self.config.get("system", {}).get("auto_visualize", True):
                    await self._update_visualizations()
                
                # Process tasks
                for task in list(self.tasks):
                    # Process task...
                    # For demonstration, we'll just remove it
                    self.tasks.remove(task)
            
            except Exception as e:
                logger.error(f"Error in system loop: {e}")
            
            # Sleep for a bit
            await asyncio.sleep(1.0)
    
    async def _update_visualizations(self) -> None:
        """Update visualizations"""
        try:
            # Prepare node data for visualization
            nodes = {}
            connections = []
            
            for node in self.node_manager.get_all_nodes():
                # Convert node to dictionary for visualization
                nodes[node.id] = {
                    'id': node.id,
                    'type': node.type,
                    'energy_level': node.energy_level,
                    'position': node.position.tolist() if hasattr(node.position, 'tolist') else node.position,
                    'attributes': node.attributes
                }
                
                # Add connections
                for conn in node.connections:
                    connections.append({
                        'source': node.id,
                        'target': conn.target_id,
                        'strength': conn.strength
                    })
            
            # Get cube data from the integrated system
            cube_data = self.integrated_system.get_cube_data() if self.integrated_system else None
            
            # Get emotional state from the integrated system
            emotional_state = self.integrated_system.get_emotional_state() if self.integrated_system else None
            
            # Update visualizations
            output_files = await self.visualizer.update_visualization(
                nodes, connections, cube_data, emotional_state
            )
            
            if output_files:
                logger.debug(f"Updated visualizations: {', '.join(output_files.keys())}")
        
        except Exception as e:
            logger.error(f"Error updating visualizations: {e}")
    
    async def shutdown(self) -> None:
        """Shut down the Kaleidoscope AI system."""
        if not self.running:
            logger.warning("System not running")
            return
        
        logger.info("Shutting down Kaleidoscope AI system...")
        self.running = False
        
        # Shut down integrated system
        if self.integrated_system:
            await self.integrated_system.shutdown()
        
        # Shut down file system bridge
        if self.file_bridge:
            self.file_bridge.shutdown()
        
        # Shut down C bridge
        if bridge.initialized:
            bridge.shutdown()
        
        logger.info("Kaleidoscope AI system shut down")
    
    def _signal_handler(self, sig, frame) -> None:
        """Handle signals for graceful shutdown"""
        logger.info(f"Received signal {sig}, shutting down...")
        # Schedule shutdown
        if asyncio.get_event_loop().is_running():
            asyncio.create_task(self.shutdown())
        else:
            # If not in an event loop, create one
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.shutdown())
            loop.close()
        
        # Exit after a short delay to allow cleanup
        time.sleep(1)
        sys.exit(0)
    
    def add_task(self, task: Dict[str, Any], priority: int = 1) -> bool:
        """
        Add a task to the system.
        
        Args:
            task: Task description
            priority: Task priority (1-5, 5 is highest)
            
        Returns:
            Success flag
        """
        if not self.running:
            logger.warning("System not running, cannot add task")
            return False
        
        # Add the task
        self.tasks.append({
            "task_data": task,
            "priority": priority,
            "created_at": time.time()
        })
        
        # Forward to integrated system if applicable
        if self.integrated_system:
            return self.integrated_system.add_task(task, priority)
        
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            Status dictionary
        """
        status = {
            "running": self.running,
            "initialized": self.initialized,
            "tasks_pending": len(self.tasks)
        }
        
        # Add node manager status
        if self.node_manager:
            status["nodes"] = {
                "count": len(self.node_manager.get_all_nodes()),
                "types": {}
            }
            
            # Count nodes by type
            for node in self.node_manager.get_all_nodes():
                node_type = node.type
                if node_type not in status["nodes"]["types"]:
                    status["nodes"]["types"][node_type] = 0
                status["nodes"]["types"][node_type] += 1
        
        # Add integrated system status
        if self.integrated_system:
            integrated_status = self.integrated_system.get_system_status()
            if integrated_status:
                status["integrated_system"] = integrated_status
        
        return status
    
    def generate_dashboard(self) -> str:
        """
        Generate an interactive dashboard for the current system state.
        
        Returns:
            Path to the generated dashboard HTML file
        """
        if not self.visualizer:
            logger.warning("Visualizer not initialized")
            return ""
        
        try:
            # Prepare node data for visualization
            nodes = {}
            connections = []
            
            for node in self.node_manager.get_all_nodes():
                # Convert node to dictionary for visualization
                nodes[node.id] = {
                    'id': node.id,
                    'type': node.type,
                    'energy_level': node.energy_level,
                    'position': node.position.tolist() if hasattr(node.position, 'tolist') else node.position,
                    'attributes': node.attributes
                }
                
                # Add connections
                for conn in node.connections:
                    connections.append({
                        'source': node.id,
                        'target': conn.target_id,
                        'strength': conn.strength
                    })
            
            # Get cube data from the integrated system
            cube_data = self.integrated_system.get_cube_data() if self.integrated_system else None
            
            # Get emotional state from the integrated system
            emotional_state = self.integrated_system.get_emotional_state() if self.integrated_system else None
            
            # Create dashboard
            dashboard_path = self.visualizer.create_interactive_dashboard(
                nodes, connections, cube_data, emotional_state
            )
            
            logger.info(f"Generated interactive dashboard at {dashboard_path}")
            return dashboard_path
        
        except Exception as e:
            logger.error(f"Error generating dashboard: {e}")
            return ""
    
    async def run_tests(self) -> bool:
        """
        Run system tests to verify functionality.
        
        Returns:
            Success flag
        """
        # Import the system test module
        from kaleidoscope_ai.system_test import run_all_tests
        
        # Run all tests
        return await run_all_tests()


async def interactive_mode(system: KaleidoscopeSystem) -> None:
    """
    Run the system in interactive mode.
    
    Args:
        system: Initialized KaleidoscopeSystem instance
    """
    print("\nKaleidoscope AI Interactive Mode")
    print("Type 'help' for available commands, 'exit' to quit.\n")
    
    while True:
        try:
            # Get user input
            cmd = input("kaleidoscope> ").strip()
            
            if not cmd:
                continue
            
            # Process command
            if cmd in ("exit", "quit"):
                # Exit interactive mode
                break
            
            elif cmd == "help":
                # Show help
                print("\nAvailable commands:")
                print("  help               Show this help message")
                print("  exit, quit         Exit interactive mode")
                print("  status             Show system status")
                print("  start              Start the system")
                print("  stop               Stop the system")
                print("  restart            Restart the system")
                print("  dashboard          Generate interactive dashboard")
                print("  test               Run system tests")
                print("  create <type>      Create a node of the specified type")
                print("  connect <id1> <id2> <strength>  Connect two nodes")
                print("  visualize          Update visualizations")
                print("\n")
            
            elif cmd == "status":
                # Show system status
                status = system.get_system_status()
                print("\nSystem Status:")
                print(json.dumps(status, indent=2))
                print("\n")
            
            elif cmd == "start":
                # Start the system
                await system.start()
                print("System started")
            
            elif cmd == "stop":
                # Stop the system
                await system.shutdown()
                print("System stopped")
            
            elif cmd == "restart":
                # Restart the system
                await system.shutdown()
                await system.start()
                print("System restarted")
            
            elif cmd == "dashboard":
                # Generate dashboard
                dashboard_path = system.generate_dashboard()
                if dashboard_path:
                    print(f"Dashboard generated at {dashboard_path}")
                else:
                    print("Failed to generate dashboard")
            
            elif cmd == "test":
                # Run tests
                print("Running system tests...")
                success = await system.run_tests()
                if success:
                    print("All tests passed!")
                else:
                    print("Some tests failed. Check the logs for details.")
            
            elif cmd.startswith("create "):
                # Create a node
                parts = cmd.split(" ", 1)
                if len(parts) < 2:
                    print("Error: Missing node type")
                    continue
                
                node_type = parts[1].strip()
                if not system.node_manager:
                    print("Error: Node manager not initialized")
                    continue
                
                node = system.node_manager.create_node(node_type=node_type)
                if node:
                    print(f"Created node with ID {node.id}")
                else:
                    print("Failed to create node")
            
            elif cmd.startswith("connect "):
                # Connect nodes
                parts = cmd.split(" ")
                if len(parts) < 4:
                    print("Error: Usage: connect <id1> <id2> <strength>")
                    continue
                
                try:
                    source_id = parts[1]
                    target_id = parts[2]
                    strength = float(parts[3])
                    
                    if not system.node_manager:
                        print("Error: Node manager not initialized")
                        continue
                    
                    source_node = system.node_manager.get_node(source_id)
                    if not source_node:
                        print(f"Error: Source node {source_id} not found")
                        continue
                    
                    success = source_node.add_connection(target_id, strength)
                    if success:
                        print(f"Connected nodes {source_id} -> {target_id} with strength {strength}")
                    else:
                        print("Failed to connect nodes")
                
                except ValueError:
                    print("Error: Strength must be a number")
                except Exception as e:
                    print(f"Error: {e}")
            
            elif cmd == "visualize":
                # Update visualizations
                await system._update_visualizations()
                print("Visualizations updated")
            
            else:
                # Unknown command
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            # Exit on Ctrl+C
            break
        
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nExiting interactive mode.\n")


def print_banner():
    """Print a banner for the system"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║   ██╗  ██╗ █████╗ ██╗     ███████╗██╗██████╗  ██████╗ ███████╗ ██████╗ ██████╗ ██████╗ ███████╗ ║
    ║   ██║ ██╔╝██╔══██╗██║     ██╔════╝██║██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██╔════╝ ║
    ║   █████╔╝ ███████║██║     █████╗  ██║██║  ██║██║   ██║███████╗██║     ██║     ██████╔╝█████╗   ║
    ║   ██╔═██╗ ██╔══██║██║     ██╔══╝  ██║██║  ██║██║   ██║╚════██║██║     ██║     ██╔═══╝ ██╔══╝   ║
    ║   ██║  ██╗██║  ██║███████╗███████╗██║██████╔╝╚██████╔╝███████║╚██████╗╚██████╗██║     ███████╗ ║
    ║   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═════╝  ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝╚═╝     ╚══════╝ ║
    ║                                                                                                 ║
    ║                        ARTIFICIAL INTELLIGENCE SYSTEM                                           ║
    ║                                                                                                 ║
    ╚═════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


async def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Kaleidoscope AI System')
    parser.add_argument('-c', '--config', help='Path to configuration file')
    parser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('-t', '--test', action='store_true', help='Run system tests')
    parser.add_argument('-d', '--dashboard', action='store_true', help='Generate dashboard')
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Initialize system
    system = KaleidoscopeSystem(config_path=args.config)
    
    # Start the system
    await system.start()
    
    try:
        # Run tests if requested
        if args.test:
            print("\nRunning system tests...")
            success = await system.run_tests()
            if success:
                print("All tests passed!")
            else:
                print("Some tests failed. Check the logs for details.")
        
        # Generate dashboard if requested
        if args.dashboard:
            print("\nGenerating dashboard...")
            dashboard_path = system.generate_dashboard()
            if dashboard_path:
                print(f"Dashboard generated at {dashboard_path}")
            else:
                print("Failed to generate dashboard")
        
        # Run in interactive mode if requested
        if args.interactive:
            await interactive_mode(system)
        elif not args.test and not args.dashboard:
            # If not running tests, dashboard, or interactive mode, just run indefinitely
            print("\nSystem running. Press Ctrl+C to stop.\n")
            while system.running:
                await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down...\n")
    
    finally:
        # Shut down the system
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())