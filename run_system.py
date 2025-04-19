#!/usr/bin/env python3
"""
Main entry point for the Quantum Kaleidoscope system.
Initializes the system and starts the API server if requested.
"""

import argparse
import os
import sys
import logging
import time # Added for sleep in non-server mode

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s %(name)s: %(message)s",
    # Use current date format
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("RunSystem")

# --- Import Core System ---
try:
    # Assuming quantum_kaleidoscope.py is in the same directory
    from quantum_kaleidoscope import QuantumKaleidoscope
    log.info("Successfully imported QuantumKaleidoscope.")
except ImportError as e:
    log.error(f"Failed to import QuantumKaleidoscope: {e}", exc_info=True)
    log.error("Make sure quantum_kaleidoscope.py is in the same directory or Python path.")
    sys.exit(1)
except Exception as e:
    log.error(f"An unexpected error occurred during QuantumKaleidoscope import: {e}", exc_info=True)
    sys.exit(1)


# --- Import API Server ---
# Import attempt needed to check if server *can* be started
try:
    from api_server import start_server
    log.info("Successfully imported start_server from api_server.")
    api_server_available = True
except ImportError:
    log.warning("Could not import start_server from api_server. API functionality unavailable.")
    log.warning("Make sure api_server.py is in the same directory or Python path.")
    start_server = None # Define as None if import fails
    api_server_available = False
except Exception as e:
    log.error(f"An unexpected error occurred during api_server import: {e}", exc_info=True)
    start_server = None
    api_server_available = False

def main():
    parser = argparse.ArgumentParser(description="Run Quantum Kaleidoscope System")
    parser.add_argument("--dimension", type=int, default=384, help="Vector dimension for the core engine")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for storing data")
    parser.add_argument("--auto-gen", action="store_true", help="Enable automatic node creation and simulation")
    parser.add_argument("--interval", type=float, default=5.0, help="Interval (seconds) for auto-generation")
    parser.add_argument("--input", type=str, help="Path to an input text file to process on startup (optional)")
    parser.add_argument("--server", action="store_true", help="Start the HTTP API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address for the API server")
    parser.add_argument("--port", type=int, default=8000, help="Port number for the API server")

    args = parser.parse_args()

    log.info("=" * 30)
    log.info("Initializing Quantum Kaleidoscope System...")
    log.info(f"Dimension: {args.dimension}")
    log.info(f"Data Directory: {args.data_dir}")
    log.info(f"Auto-Generation: {'Enabled' if args.auto_gen else 'Disabled'} (Interval: {args.interval}s)")
    log.info(f"API Server Requested: {'Yes' if args.server else 'No'}")
    if args.server:
        log.info(f"API Server Config: Host={args.host}, Port={args.port}")
    log.info("=" * 30)


    try:
        # Create the main system instance
        system = QuantumKaleidoscope(dimension=args.dimension, data_dir=args.data_dir)
        log.info(f"Quantum Kaleidoscope system instance created successfully. ID: {id(system)}")
    except Exception as e:
        log.error(f"FATAL: Failed to create QuantumKaleidoscope instance: {e}", exc_info=True)
        sys.exit(1)

    # Process input file if provided
    if args.input:
        if os.path.exists(args.input):
            try:
                with open(args.input, 'r', encoding='utf-8') as f: # Added encoding
                    text = f.read()
                log.info(f"Processing input file: {args.input} ({len(text)} characters)")
                result = system.process_text(text) # Assuming process_text returns a dict
                log.info(f"File processing complete. Result ID: {result.get('processing_id', 'N/A')}")
                log.info(f"-> Patterns: {result.get('pattern_count', 0)}, Insights: {result.get('insight_count', 0)}, Perspectives: {result.get('perspective_count', 0)}")
            except FileNotFoundError:
                 log.warning(f"Input file specified but not found: {args.input}")
            except Exception as e:
                log.error(f"Error processing input file {args.input}: {e}", exc_info=True)
        else:
            log.warning(f"Input file path does not exist: {args.input}")

    # Start auto-generation if requested
    if args.auto_gen:
        try:
            # start_auto_generation should log its own status
            system.start_auto_generation(interval=args.interval)
        except AttributeError:
            log.error("System object does not support 'start_auto_generation'. Cannot enable.")
        except Exception as e:
            log.error(f"Failed to start auto-generation: {e}", exc_info=True)

    # Start server if requested AND if it was imported successfully
    if args.server:
        if api_server_available and start_server is not None:
            log.info(f"Attempting to start API server on {args.host}:{args.port}...")
            try:
                # This function call will typically block the main thread until the server stops (e.g., Ctrl+C)
                start_server(system, host=args.host, port=args.port)
                log.info("API server has shut down.")
            except Exception as e:
                log.error(f"An error occurred while running the API server: {e}", exc_info=True)
        else:
            log.error("Cannot start server: --server flag was used, but API server code failed to import.")
            # Exit if server was requested but unavailable? Or just warn? Let's warn and potentially exit.
            log.warning("Proceeding without API server.")
            # If only server was requested, maybe exit here?
            # sys.exit(1) # Uncomment if server failure should stop everything

    # If server wasn't requested or failed, but other tasks might run (like auto-gen)
    if not args.server or (args.server and not api_server_available):
         if args.auto_gen and system.auto_gen_active: # Check if auto-gen actually started
              log.info("Auto-generation is active. Keeping main thread alive. Press Ctrl+C to exit.")
              try:
                   while True:
                        # Keep main thread alive while background threads (like auto-gen) run
                        time.sleep(3600) # Sleep for a long time
              except KeyboardInterrupt:
                   log.info("Ctrl+C received. Stopping system...")
         else:
              log.info("No server requested and no long-running tasks active. Exiting.")

    # Cleanup / Shutdown procedures (if any)
    if args.auto_gen:
         try:
             system.stop_auto_generation() # Ensure background thread is stopped on exit
         except Exception as e:
             log.warning(f"Error during auto-generation stop on exit: {e}")

    log.info("System shutdown complete.")


if __name__ == "__main__":
    main()
