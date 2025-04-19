"""
Bridge for C/Python interoperability using the libkaleidoscope.so shared library.
"""
import ctypes
import os
import sys
import logging
import json  # Import json for parsing
from pathlib import Path
import traceback
from typing import Any, Callable, Optional, Tuple, List, Dict

# Define callback type for Python side
EVENT_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p)

class BridgeError(Exception):
    """Exception raised for errors in the BridgeInterface."""
    pass

class BridgeInterface:
    """Bridge interface for interacting with the C libkaleidoscope.so library."""

    def __init__(self, library_path: Optional[str] = None):
        """Initialize the bridge interface."""
        self.logger = logging.getLogger("kaleidoscope.bridge")
        self._lib = None
        self._callback_refs = {}  # Keep references to callback objects
        self._load_library(library_path)
        self._initialize_c_system()  # Automatically initialize C system

    def _load_library(self, library_path: Optional[str] = None):
        """Load the C shared library."""
        try:
            lib_name = "libkaleidoscope.so"
            if library_path and os.path.exists(library_path):
                lib_full_path = library_path
            else:
                # Search in common locations relative to this file and system paths
                possible_paths = [
                    os.path.dirname(__file__),  # Directory containing this bridge file
                    str(Path(__file__).parent.parent.parent),  # Project root (assuming standard structure)
                    "/usr/local/lib",
                    "/usr/lib",
                    "."  # Current working directory
                ]
                found_path = None
                for path in possible_paths:
                    test_path = os.path.join(path, lib_name)
                    if os.path.exists(test_path):
                        found_path = test_path
                        break
                if not found_path:
                    # Try loading by name only, relying on system linker paths (LD_LIBRARY_PATH)
                    try:
                        self._lib = ctypes.CDLL(lib_name)
                        self.logger.info(f"Loaded library '{lib_name}' using system linker.")
                        self._setup_functions()
                        return
                    except OSError:
                        raise BridgeError(f"Could not find or load {lib_name} in standard paths or via system linker.")

                lib_full_path = found_path

            self._lib = ctypes.CDLL(lib_full_path)
            self._setup_functions()
            self.logger.info(f"Loaded library from {lib_full_path}")

        except Exception as e:
            error_msg = f"Error loading library: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise BridgeError(error_msg) from e

    def _setup_functions(self):
        """Set up the function prototypes from the C library based on bridge_adapter.h."""
        if not self._lib:
            raise BridgeError("Library not loaded")

        # Define common argtypes and restypes
        self._lib.initialize_system.argtypes = []
        self._lib.initialize_system.restype = ctypes.c_int

        self._lib.shutdown_system.argtypes = []
        self._lib.shutdown_system.restype = ctypes.c_int

        self._lib.configure_system.argtypes = [ctypes.c_char_p]
        self._lib.configure_system.restype = ctypes.c_int

        self._lib.bridge_create_node.argtypes = [ctypes.c_uint64, ctypes.c_int]
        self._lib.bridge_create_node.restype = ctypes.c_uint64  # Returns node ID

        self._lib.bridge_connect_nodes.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_double]
        self._lib.bridge_connect_nodes.restype = ctypes.c_int  # Returns 1 on success, 0 on failure

        self._lib.bridge_update_memory_graph.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
        self._lib.bridge_update_memory_graph.restype = ctypes.c_int  # Returns 1 on success, 0 on failure

        self._lib.bridge_get_updated_nodes.argtypes = []
        self._lib.bridge_get_updated_nodes.restype = ctypes.c_char_p  # Returns JSON string or NULL

        self._lib.bridge_remove_node.argtypes = [ctypes.c_uint64]
        self._lib.bridge_remove_node.restype = ctypes.c_int  # Returns 1 on success, 0 on failure

        # Task Manager functions
        self._lib.init_task_manager.argtypes = [ctypes.c_int]
        self._lib.init_task_manager.restype = ctypes.c_void_p  # Returns TaskManager pointer or NULL

        self._lib.add_task.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_char_p]
        self._lib.add_task.restype = ctypes.c_int  # Returns 1 on success, 0 on failure

        self._lib.get_next_task.argtypes = [ctypes.c_void_p]
        self._lib.get_next_task.restype = ctypes.c_char_p  # Returns JSON string or NULL

        self._lib.complete_task.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.complete_task.restype = ctypes.c_int  # Returns 1 on success, 0 on failure

        # Callback registration
        self._lib.register_callback.argtypes = [ctypes.c_char_p, EVENT_CALLBACK]
        self._lib.register_callback.restype = ctypes.c_int  # Returns 1 on success, 0 on failure

        # Error handling
        self._lib.bridge_get_last_error.argtypes = []
        self._lib.bridge_get_last_error.restype = ctypes.c_int

        self._lib.bridge_get_last_error_message.argtypes = []
        self._lib.bridge_get_last_error_message.restype = ctypes.c_char_p  # Returns error string

    def _initialize_c_system(self):
        """Initialize the C system via the bridge."""
        if not self._lib:
            raise BridgeError("Library not loaded")
        self.logger.info("Initializing C system via bridge...")
        result = self._lib.initialize_system()
        if result != 0:  # BRIDGE_ERROR_NONE is 0
            error_msg = f"Failed to initialize C system: {self.get_last_error_message()} (Code: {result})"
            self.logger.error(error_msg)
            raise BridgeError(error_msg)
        self.logger.info("C system initialized successfully.")

    def shutdown(self):
        """Shutdown the C system via the bridge."""
        if not self._lib:
            self.logger.warning("Attempted to shutdown bridge, but library not loaded.")
            return False
        self.logger.info("Shutting down C system via bridge...")
        result = self._lib.shutdown_system()
        if result == 0:  # BRIDGE_ERROR_NONE
            self.logger.info("C system shutdown successful.")
            return True
        else:
            error_msg = f"Failed to shutdown C system: {self.get_last_error_message()} (Code: {result})"
            self.logger.error(error_msg)
            # Don't raise an exception on shutdown failure, just log it.
            return False

    def configure(self, config_str: str) -> bool:
        """Configure the C system."""
        if not self._lib:
            raise BridgeError("Library not loaded")
        if not isinstance(config_str, str):
            raise TypeError("config_str must be a string")

        config_bytes = config_str.encode('utf-8')
        result = self._lib.configure_system(ctypes.c_char_p(config_bytes))
        if result != 0:  # BRIDGE_ERROR_NONE
            error_msg = f"Failed to configure system: {self.get_last_error_message()} (Code: {result})"
            self.logger.error(error_msg)
            return False
        return True

    def _id_to_uint64(self, node_id: Any) -> int:
        """Convert various ID types to uint64, hashing strings if necessary."""
        if isinstance(node_id, int):
            return node_id
        elif isinstance(node_id, str):
            try:
                # Try direct conversion first
                return int(node_id)
            except ValueError:
                # If not an integer string, hash it
                h = hash(node_id)
                return h & 0xFFFFFFFFFFFFFFFF  # Ensure it fits in uint64
        else:
            raise TypeError(f"Unsupported node ID type: {type(node_id)}")

    def create_node(self, node_id: Any, node_type: Any) -> Optional[int]:
        """
        Create a node in the C system.
        Args:
            node_id: Node ID (int or str). If str, will be hashed if not numeric.
            node_type: Node type (int or str). If str, mapped to int.
        Returns:
            The actual node ID (uint64) assigned by C, or None on failure.
        """
        if not self._lib:
            raise BridgeError("Library not loaded")

        node_id_int = self._id_to_uint64(node_id)

        # Convert node_type to int if it's a string
        if isinstance(node_type, str):
            type_map = {"standard": 0, "core": 1, "memory": 2, "processing": 3, "custom": 4}  # Example mapping
            node_type_int = type_map.get(node_type.lower(), 0)  # Default to 0
        elif isinstance(node_type, int):
            node_type_int = node_type
        else:
            raise TypeError(f"Unsupported node type: {type(node_type)}")

        result_id = self._lib.bridge_create_node(ctypes.c_uint64(node_id_int), ctypes.c_int(node_type_int))

        if result_id == 0:
            error_msg = f"Failed to create node (requested ID: {node_id_int}): {self.get_last_error_message()}"
            self.logger.error(error_msg)
            return None
        else:
            self.logger.info(f"Bridge created node with ID: {result_id}")
            return result_id  # Return the actual ID from C

    def connect_nodes(self, node1_id: Any, node2_id: Any, strength: float = 1.0) -> bool:
        """Connect two nodes in the C system."""
        if not self._lib:
            raise BridgeError("Library not loaded")

        node1_id_int = self._id_to_uint64(node1_id)
        node2_id_int = self._id_to_uint64(node2_id)

        result = self._lib.bridge_connect_nodes(
            ctypes.c_uint64(node1_id_int),
            ctypes.c_uint64(node2_id_int),
            ctypes.c_double(strength)
        )

        if result == 0:  # Failure is 0
            error_msg = f"Failed to connect nodes {node1_id_int} and {node2_id_int}: {self.get_last_error_message()}"
            self.logger.error(error_msg)
            return False
        return True

    def update_node(self, node_id: Any, data: Dict[str, Any]) -> bool:
        """Update a node in the C system with JSON data."""
        if not self._lib:
            raise BridgeError("Library not loaded")
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")

        node_id_int = self._id_to_uint64(node_id)
        data_json = json.dumps(data)
        data_bytes = data_json.encode('utf-8')

        result = self._lib.bridge_update_memory_graph(
            ctypes.c_uint64(node_id_int),
            ctypes.c_char_p(data_bytes)
        )

        if result == 0:  # Failure is 0
            error_msg = f"Failed to update node {node_id_int}: {self.get_last_error_message()}"
            self.logger.error(error_msg)
            return False
        return True

    def delete_node(self, node_id: Any) -> bool:
        """Remove a node from the C system."""
        if not self._lib:
            raise BridgeError("Library not loaded")

        node_id_int = self._id_to_uint64(node_id)
        result = self._lib.bridge_remove_node(ctypes.c_uint64(node_id_int))

        if result == 0:  # Failure is 0
            error_msg = f"Failed to delete node {node_id_int}: {self.get_last_error_message()}"
            self.logger.error(error_msg)
            return False
        return True

    def get_updated_nodes(self) -> List[Dict[str, Any]]:
        """Get all updated nodes from the C system as a list of dictionaries."""
        if not self._lib:
            raise BridgeError("Library not loaded")

        result_ptr = self._lib.bridge_get_updated_nodes()
        if not result_ptr:
            # Could be an error or just no updated nodes
            last_error_code = self.get_last_error_code()
            if last_error_code != 0:  # BRIDGE_ERROR_NONE
                error_msg = f"Error getting updated nodes: {self.get_last_error_message()} (Code: {last_error_code})"
                self.logger.error(error_msg)
            return []  # Return empty list if NULL or error

        try:
            nodes_json = ctypes.string_at(result_ptr).decode('utf-8')
            if nodes_json == "null":  # Handle JSON null if no tasks
                return []
            nodes_list = json.loads(nodes_json)
            if not isinstance(nodes_list, list):
                self.logger.error(f"Expected a list from bridge_get_updated_nodes, got {type(nodes_list)}")
                return []
            return nodes_list
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse updated nodes JSON: {nodes_json[:100]}...")
            return []
        except Exception as e:
            self.logger.error(f"Error processing updated nodes: {e}")
            return []

    def register_event_callback(self, event_name: str, callback: Callable[[str, str], None]):
        """Register a Python callback for C events."""
        if not self._lib:
            raise BridgeError("Library not loaded")
        if not callable(callback):
            raise TypeError("callback must be callable")

        event_name_bytes = event_name.encode('utf-8')

        # Create ctypes callback wrapper
        c_callback = EVENT_CALLBACK(callback)

        # Store a reference to prevent garbage collection
        self._callback_refs[event_name] = c_callback

        result = self._lib.register_callback(ctypes.c_char_p(event_name_bytes), c_callback)

        if result == 0:  # Failure is 0
            # Remove reference if registration failed
            del self._callback_refs[event_name]
            error_msg = f"Failed to register callback for '{event_name}': {self.get_last_error_message()}"
            self.logger.error(error_msg)
            return False
        self.logger.info(f"Registered callback for event: {event_name}")
        return True

    def get_last_error_code(self) -> int:
        """Get the last error code from the C system."""
        if not self._lib:
            return -1  # Indicate library not loaded
        return self._lib.bridge_get_last_error()

    def get_last_error_message(self) -> str:
        """Get the last error message from the C system."""
        if not self._lib:
            return "Bridge library not loaded"

        error_msg_ptr = self._lib.bridge_get_last_error_message()
        if error_msg_ptr:
            try:
                return ctypes.string_at(error_msg_ptr).decode('utf-8')
            except Exception as e:
                return f"Error decoding error message: {e}"
        else:
            # Check error code if message pointer is NULL
            error_code = self.get_last_error_code()
            if error_code == 0:  # BRIDGE_ERROR_NONE
                return "No error"
            else:
                return f"Unknown error (Code: {error_code}, Message unavailable)"

    def __del__(self):
        """Ensure C system is shut down when the bridge object is destroyed."""
        try:
            self.shutdown()
        except Exception as e:
            # Log error during shutdown in destructor, but don't raise
            self.logger.error(f"Error during bridge shutdown in destructor: {e}")

# Example Usage (optional, can be removed or placed in a separate script)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        bridge = BridgeInterface()

        # Configure
        bridge.configure("paths.shared_dir=./ai_shared_data")

        # Define a callback
        def my_event_handler(event_name_ptr, data_ptr):
            event_name = event_name_ptr.decode('utf-8')
            data = data_ptr.decode('utf-8')
            logger.info(f"Python received event: {event_name}, Data: {data}")

        # Register callback
        bridge.register_event_callback("node_created", my_event_handler)
        bridge.register_event_callback("*", my_event_handler)  # Catch all events

        # Node operations
        node1_id = bridge.create_node("node_A", "standard")
        node2_id = bridge.create_node(1001, 1)  # Use int type

        if node1_id and node2_id:
            bridge.connect_nodes(node1_id, node2_id, 0.75)
            bridge.update_node(node1_id, {"name": "Alpha", "value": 123})
            bridge.update_node(node2_id, {"name": "Beta", "status": "active"})

        # Get updates
        updated = bridge.get_updated_nodes()
        logger.info(f"Updated nodes: {updated}")

        # Delete node
        if node1_id:
            bridge.delete_node(node1_id)

        # Get updates again
        updated = bridge.get_updated_nodes()
        logger.info(f"Updated nodes after delete: {updated}")

    except BridgeError as e:
        logger.error(f"Bridge operation failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Bridge shutdown is handled by __del__
        logger.info("Example finished.")