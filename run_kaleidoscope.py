#!/usr/bin/env python3
"""
Kaleidoscope AI System - Runner Script

This script provides a simple way to run the integrated Kaleidoscope AI system.
It connects the core Resonance System with NodeManager, PerspectiveManager,
and other components through the integration layer.
"""

import os
import sys
import argparse
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure script directory is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('kaleidoscope_runner.log')
    ]
)
logger = logging.getLogger("KaleidoscopeRunner")

# Import the integration layer
try:
    from integration import get_integration_layer, run_integration, IntegrationLayer
except ImportError as e:
    logger.error(f"Error importing integration layer: {e}")
    print(f"Error: Unable to import integration layer: {e}")
    print("Please ensure the integration.py file is in the same directory or in your Python path.")
    sys.exit(1)

async def run_kaleidoscope(args: argparse.Namespace) -> None:
    """
    Run the Kaleidoscope AI system with the specified configuration.
    
    Args:
        args: Command-line arguments
    """
    try:
        logger.info("Starting Kaleidoscope AI system...")
        
        # Get the integration layer
        integration = get_integration_layer(args.config)
        
        # Start the integration layer
        await integration.start()
        
        # If interactive mode is enabled
        if args.interactive:
            await run_interactive_mode(integration)
        else:
            # Run until interrupted
            print("\nKaleidoscope AI system is running. Press Ctrl+C to stop.\n")
            while integration.running:
                await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down...")
    
    except Exception as e:
        logger.error(f"Error running Kaleidoscope AI system: {e}", exc_info=True)
        print(f"Error: {e}")
    
    finally:
        # Ensure clean shutdown
        if 'integration' in locals():
            await integration.shutdown()
        logger.info("Kaleidoscope AI system shut down")

async def run_interactive_mode(integration: IntegrationLayer) -> None:
    """
    Run the system in interactive mode, accepting user commands.
    
    Args:
        integration: IntegrationLayer instance
    """
    print("\nKaleidoscope AI Interactive Mode")
    print("Type 'help' for available commands, 'exit' to quit.\n")
    
    while integration.running:
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
                print("  help                Show this help message")
                print("  exit, quit          Exit interactive mode")
                print("  status              Show system status")
                print("  sync                Synchronize systems")
                print("  save                Save system state")
                print("  create <type>       Create a node of specified type")
                print("  connect <id1> <id2> Connect two nodes")
                print("\n")
            
            elif cmd == "status":
                # Show system status
                status = integration.get_status()
                print("\nSystem Status:")
                print(json.dumps(status, indent=2))
                print("\n")
            
            elif cmd == "sync":
                # Synchronize systems
                integration.synchronize_systems()
                print("Systems synchronized")
            
            elif cmd == "save":
                # Save system state
                await integration._save_state()
                print("System state saved")
            
            elif cmd.startswith("create "):
                # Create a node
                parts = cmd.split(" ", 1)
                if len(parts) < 2:
                    print("Error: Missing node type")
                    continue
                
                node_type = parts[1].strip()
                position = [0.0, 0.0, 0.0]  # Default position
                
                # Create node in NodeManager
                node = integration.node_manager.create_node(
                    node_type=node_type,
                    position=position
                )
                
                if node:
                    node_id = node['id'] if isinstance(node, dict) else node.id
                    print(f"Created node with ID {node_id}")
                    # Force synchronization
                    integration.synchronize_systems()
                else:
                    print("Failed to create node")
            
            elif cmd.startswith("connect "):
                # Connect nodes
                parts = cmd.split(" ")
                if len(parts) < 3:
                    print("Error: Usage: connect <id1> <id2> [strength]")
                    continue
                
                try:
                    source_id = parts[1]
                    target_id = parts[2]
                    strength = float(parts[3]) if len(parts) > 3 else 1.0
                    
                    source_node = integration.node_manager.get_node(source_id)
                    if not source_node:
                        print(f"Error: Source node {source_id} not found")
                        continue
                    
                    # Handle both dict and object nodes
                    if isinstance(source_node, dict):
                        # Dictionary node - add connection to connections list
                        connection = {"target_id": target_id, "strength": strength}
                        if "connections" not in source_node:
                            source_node["connections"] = []
                        source_node["connections"].append(connection)
                        success = True
                    else:
                        # Object node - use add_connection method
                        success = source_node.add_connection(target_id, strength)
                    
                    if success:
                        print(f"Connected nodes {source_id} -> {target_id} with strength {strength}")
                        # Force synchronization
                        integration.synchronize_systems()
                    else:
                        print("Failed to connect nodes")
                
                except ValueError:
                    print("Error: Strength must be a number")
                except Exception as e:
                    print(f"Error: {e}")
            
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

def print_banner() -> None:
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
    
                            INTEGRATED RESONANCE SYSTEM
    """
    print(banner)

def main() -> None:
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Kaleidoscope AI System Runner')
    parser.add_argument('-c', '--config', help='Path to configuration file')
    parser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Print banner
    print_banner()
    
    # Run the system
    asyncio.run(run_kaleidoscope(args))

if __name__ == "__main__":
    main()