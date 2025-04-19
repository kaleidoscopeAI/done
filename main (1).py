#!/usr/bin/env python3
import asyncio
import logging
import argparse
import os
import signal
import subprocess
from typing import Optional, Dict, Any

from kaleidoscope_ai.core.AI_Core import AI_Core

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kaleidoscope_ai.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Kaleidoscope AI Simulation')
    
    parser.add_argument('--websocket-uri', 
                        default='ws://localhost:8765',
                        help='WebSocket URI for quantum-bridge server')
    
    parser.add_argument('--start-bridge', 
                        action='store_true',
                        help='Start the quantum-bridge.py server automatically')
    
    parser.add_argument('--simulation-interval', 
                        type=float, 
                        default=0.1,
                        help='Time interval between simulation steps (seconds)')
    
    parser.add_argument('--log-interval', 
                        type=float, 
                        default=5.0,
                        help='Time interval between logging statistics (seconds)')
    
    return parser.parse_args()

def start_quantum_bridge() -> Optional[subprocess.Popen]:
    """Start the quantum-bridge.py server as a separate process."""
    bridge_path = os.path.join(os.path.dirname(__file__), 'quantum-bridge.py')
    
    if not os.path.exists(bridge_path):
        logger.error(f"quantum-bridge.py not found at {bridge_path}")
        return None
    
    try:
        # Start the bridge server
        process = subprocess.Popen(['python3', bridge_path])
        logger.info(f"Started quantum-bridge.py (PID: {process.pid})")
        return process
    except Exception as e:
        logger.error(f"Failed to start quantum-bridge.py: {str(e)}")
        return None

async def main():
    """Main entry point for the Kaleidoscope AI simulation."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configuration for the AI Core
    config = {
        "websocket_uri": args.websocket_uri,
        "simulation_interval": args.simulation_interval,
        "log_interval": args.log_interval,
    }
    
    # Start the quantum-bridge.py server if requested
    bridge_process = None
    if args.start_bridge:
        bridge_process = start_quantum_bridge()
    
    # Create and initialize the AI Core
    logger.info("Initializing Kaleidoscope AI Core...")
    ai_core = AI_Core(config=config)
    
    # Set up signal handling for graceful shutdown
    loop = asyncio.get_event_loop()
    
    def shutdown_handler():
        logger.info("Shutting down Kaleidoscope AI...")
        ai_core.stop()
        
        # Terminate the bridge process if we started it
        if bridge_process:
            logger.info("Terminating quantum-bridge.py...")
            bridge_process.terminate()
    
    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)
    
    try:
        # Run the simulation
        logger.info("Starting Kaleidoscope AI simulation...")
        await ai_core.run_simulation_and_send_updates()
    finally:
        # Clean up resources
        shutdown_handler()
        logger.info("Kaleidoscope AI shutdown complete.")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())