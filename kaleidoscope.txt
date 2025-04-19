#!/usr/bin/env python3
"""
Kaleidoscope AI - Main Entry Point
==================================
Command-line interface for the Kaleidoscope AI Software Ingestion & Mimicry System.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any

# Import the core module
from kaleidoscope_core import KaleidoscopeCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for Kaleidoscope AI CLI"""
    parser = argparse.ArgumentParser(description="Kaleidoscope AI - Software Ingestion & Mimicry System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest and analyze software")
    ingest_parser.add_argument("--file", "-f", required=True, help="Path to software file to ingest")
    ingest_parser.add_argument("--work-dir", "-w", help="Working directory", default="kaleidoscope_workdir")
    
    # Mimic command
    mimic_parser = subparsers.add_parser("mimic", help="Generate mimicked software")
    mimic_parser.add_argument("--spec", "-s", required=True, help="Path to specification file")
    mimic_parser.add_argument("--language", "-l", required=True, 
                             choices=["python", "javascript", "cpp"], 
                             help="Target language for mimicked software")
    mimic_parser.add_argument("--work-dir", "-w", help="Working directory", default="kaleidoscope_workdir")
    
    # Web interface command
    web_parser = subparsers.add_parser("web", help="Start web interface")
    web_parser.add_argument("--host", help="Host address to bind", default="127.0.0.1")
    web_parser.add_argument("--port", "-p", type=int, help="Port to listen on", default=5000)
    web_parser.add_argument("--work-dir", "-w", help="Working directory", default="kaleidoscope_workdir")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Initialize Kaleidoscope
        kaleidoscope = KaleidoscopeCore(work_dir=args.work_dir)
        
        # Execute command
        if args.command == "ingest":
            result = kaleidoscope.ingest_software(args.file)
            logger.info(f"Ingestion completed with status: {result['status']}")
            
            if result["status"] == "completed":
                logger.info(f"Generated {len(result['spec_files'])} specification files")
                logger.info(f"Generated {len(result['reconstructed_files'])} reconstructed files")
                
        elif args.command == "mimic":
            spec_files = [args.spec]
            result = kaleidoscope.mimic_software(spec_files, args.language)
            logger.info(f"Mimicry completed with status: {result['status']}")
            
            if result["status"] == "completed":
                logger.info(f"Generated {len(result['mimicked_files'])} mimicked files")
                logger.info(f"Output directory: {result['mimicked_dir']}")
                
        elif args.command == "web":
            # Import web module only when needed
            from kaleidoscope_web import start_server
            start_server(work_dir=args.work_dir, host=args.host, port=args.port)
            
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
