#!/usr/bin/env python3
"""
Unified Configuration System for Kaleidoscope AI

This module provides a unified configuration system that can be used by both
the Python and C components of the Kaleidoscope AI system. It handles loading
configuration from files, environment variables, and defaults.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "system": {
        "name": "Kaleidoscope AI",
        "version": "1.0.0",
        "log_level": "INFO",
        "max_tasks": 1000,
        "dimension": 256,
        "memory_limit": "8g",
        "gpu_available": False
    },
    "paths": {
        "c_library": "/home/jg/Desktop/ai_system/libkaleidoscope.so",
        "data_dir": "./data",
        "log_dir": "./logs",
        "output_dir": "./output",
        "shared_dir": "./shared"
    },
    "integration": {
        "simulation_interval": 0.1,
        "sync_interval": 5.0,
        "websocket_port": 8765,
        "bridge_logging": True
    },
    "features": {
        "enable_growth_laws": True,
        "enable_perspective_management": True,
        "enable_quantum_bridge": False,
        "persistence_enabled": True,
        "distributed_execution": False
    },
    "network": {
        "initial_nodes": 10,
        "connection_strength": 1.0,
        "stress_threshold": 5.0
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file, environment variables, and defaults
    
    Args:
        config_path: Path to config file (JSON)
        
    Returns:
        Configuration dictionary
    """
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Try to load from file if specified
    if config_path:
        try:
            config_path = Path(config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                
                # Deep merge with defaults
                config = deep_merge_dicts(config, file_config)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    
    # Check for environment variables
    env_prefix = "KALEIDOSCOPE_"
    for env_var, value in os.environ.items():
        if env_var.startswith(env_prefix):
            # Convert to config key
            key_path = env_var[len(env_prefix):].lower().split('_')
            
            # Navigate to the right place in the config
            curr = config
            for part in key_path[:-1]:
                if part not in curr:
                    curr[part] = {}
                curr = curr[part]
            
            # Set the value (try to convert to appropriate type)
            try:
                # Try parsing as JSON first (for structured data)
                parsed_value = json.loads(value)
                curr[key_path[-1]] = parsed_value
            except json.JSONDecodeError:
                # Otherwise use the string value
                curr[key_path[-1]] = value
    
    # Ensure all required directories exist
    for key, path in config["paths"].items():
        if key.endswith('_dir'):  # Only create directories, not files
            os.makedirs(path, exist_ok=True)
    
    return config

def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries
    
    Args:
        dict1: First dictionary (base)
        dict2: Second dictionary (overrides)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_dicts(result[key], value)
        else:
            # Override or add simple values
            result[key] = value
    
    return result

def get_c_config_string(config: Dict[str, Any]) -> str:
    """
    Convert configuration to a C-compatible string format
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration string in a format parsable by C code
    """
    # Flatten the configuration for C side
    flat_config = {}
    
    def flatten_dict(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                flatten_dict(v, f"{prefix}{k}.")
            else:
                flat_config[f"{prefix}{k}"] = v
    
    flatten_dict(config)
    
    # Convert to string format that C can parse
    config_str = ""
    for key, value in flat_config.items():
        if isinstance(value, bool):
            value_str = "1" if value else "0"
        elif isinstance(value, (int, float)):
            value_str = str(value)
        else:
            value_str = f'"{value}"'
        
        config_str += f"{key}={value_str};"
    
    return config_str

# Singleton configuration
_config = None

def get_config(config_path: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
    """
    Get the singleton configuration instance
    
    Args:
        config_path: Optional path to configuration file
        force_reload: Whether to force reload even if already loaded
        
    Returns:
        Configuration dictionary
    """
    global _config
    if _config is None or force_reload:
        _config = load_config(config_path)
    return _config