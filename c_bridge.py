#!/usr/bin/env python3
"""
C Bridge for Kaleidoscope AI

This module provides a bridge between Python and C components of Kaleidoscope AI.
It allows for bidirectional communication and shared data access between the Python
and C sides of the system through dynamic library loading.
"""

import os
import sys
import json
import logging
import ctypes
import threading
from typing import Dict, Any, Optional, Callable, List, Tuple
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Callback function type for events from C to Python
C_CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p)

class CBridge:
    """Bridge class for C-Python communication"""
    
    def __init__(self):
        """Initialize the bridge without loading a library yet"""
        self.lib = None
        self.lib_path = None
        self.callbacks = {}
        self.callback_lock = threading.RLock()
        self.event_handlers = {}
        
        # Keep a reference to the callback function to prevent garbage collection
        self._c_callback_obj = None
    
    def init_bridge(self, lib_path: str) -> bool:
        """
        Initialize the bridge by loading the C library
        
        Args:
            lib_path: Path to the shared library (.so/.dll)
            
        Returns:
            Success flag
        """
        try:
            # Store the path for future reference
            self.lib_path = lib_path
            
            if not os.path.exists(lib_path):
                logger.error(f"Library not found at {lib_path}")
                return False
            
            # Load the library
            self.lib = ctypes.cdll.LoadLibrary(lib_path)
            
            # Define function prototypes
            self._define_function_prototypes()
            
            # Register the global callback
            self._register_global_callback()
            
            logger.info(f"C bridge initialized with library: {lib_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize C bridge: {e}", exc_info=True)
            self.lib = None
            return False
    
    def _define_function_prototypes(self):
        """Define prototypes for all C functions"""
        if not self.lib:
            return
        
        # Core system functions
        self.lib.initialize_system.argtypes = [ctypes.c_int]
        self.lib.initialize_system.restype = ctypes.c_int
        
        self.lib.configure_system.argtypes = [ctypes.c_char_p]
        self.lib.configure_system.restype = ctypes.c_int
        
        self.lib.register_event_callback.argtypes = [C_CALLBACK_TYPE]
        self.lib.register_event_callback.restype = ctypes.c_int
        
        self.lib.shutdown_system.argtypes = []
        self.lib.shutdown_system.restype = ctypes.c_int
        
        # Task manager functions
        self.lib.create_task_manager.argtypes = [ctypes.c_int]
        self.lib.create_task_manager.restype = ctypes.c_void_p
        
        self.lib.add_task.argtypes = [
            ctypes.c_void_p,  # task_manager
            ctypes.c_int,     # task_id
            ctypes.c_int,     # priority
            ctypes.c_int,     # assigned_node_id
            ctypes.c_char_p   # task_data
        ]
        self.lib.add_task.restype = ctypes.c_int
        
        self.lib.get_next_task.argtypes = [ctypes.c_void_p]
        self.lib.get_next_task.restype = ctypes.c_char_p
        
        self.lib.complete_task.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.complete_task.restype = ctypes.c_int
        
        self.lib.destroy_task_manager.argtypes = [ctypes.c_void_p]
        self.lib.destroy_task_manager.restype = None
        
        # File system notification functions
        self.lib.notify_file_change.argtypes = [
            ctypes.c_char_p,  # file_type
            ctypes.c_char_p,  # filename
            ctypes.c_int      # event_type (1=create/update, 2=delete)
        ]
        self.lib.notify_file_change.restype = ctypes.c_int
        
        # Node management functions
        self.lib.create_node.argtypes = [
            ctypes.c_int,     # node_type
            ctypes.c_float,   # pos_x
            ctypes.c_float,   # pos_y
            ctypes.c_float,   # pos_z
            ctypes.c_char_p   # attributes (JSON string)
        ]
        self.lib.create_node.restype = ctypes.c_int
        
        self.lib.connect_nodes.argtypes = [
            ctypes.c_int,     # source_id
            ctypes.c_int,     # target_id
            ctypes.c_float    # strength
        ]
        self.lib.connect_nodes.restype = ctypes.c_int
        
        self.lib.get_node_data.argtypes = [ctypes.c_int]
        self.lib.get_node_data.restype = ctypes.c_char_p
        
        self.lib.update_node_position.argtypes = [
            ctypes.c_int,     # node_id
            ctypes.c_float,   # pos_x
            ctypes.c_float,   # pos_y
            ctypes.c_float    # pos_z
        ]
        self.lib.update_node_position.restype = ctypes.c_int
        
        self.lib.get_all_nodes.argtypes = []
        self.lib.get_all_nodes.restype = ctypes.c_char_p
        
        self.lib.remove_node.argtypes = [ctypes.c_int]
        self.lib.remove_node.restype = ctypes.c_int
        
        logger.debug("Function prototypes defined for C library")
    
    def _register_global_callback(self):
        """Register the global callback function with C code"""
        if not self.lib or not hasattr(self.lib, 'register_event_callback'):
            return
        
        # Define the Python callback function that will handle events from C
        def event_callback(event_name_bytes, data_bytes):
            try:
                # Convert bytes to strings
                event_name = event_name_bytes.decode('utf-8')
                data = data_bytes.decode('utf-8')
                
                # Call registered handlers for this event
                with self.callback_lock:
                    handlers = self.event_handlers.get(event_name, [])
                
                for handler in handlers:
                    try:
                        handler(event_name, data)
                    except Exception as e:
                        logger.error(f"Error in event handler for {event_name}: {e}", exc_info=True)
            
            except Exception as e:
                logger.error(f"Error in global event callback: {e}", exc_info=True)
        
        # Create a C-compatible callback function
        self._c_callback_obj = C_CALLBACK_TYPE(event_callback)
        
        # Register with the C library
        result = self.lib.register_event_callback(self._c_callback_obj)
        
        if result != 0:
            logger.error(f"Failed to register global callback, error code: {result}")
    
    def register_callback(self, event_name: str, callback: Callable[[str, str], None]) -> bool:
        """
        Register a callback for a specific event type
        
        Args:
            event_name: Name of the event to listen for
            callback: Function to call when event occurs (takes event_name, data)
            
        Returns:
            Success flag
        """
        with self.callback_lock:
            if event_name not in self.event_handlers:
                self.event_handlers[event_name] = []
            
            self.event_handlers[event_name].append(callback)
            logger.debug(f"Registered callback for event: {event_name}")
        
        return True
    
    def unregister_callback(self, event_name: str, callback: Callable[[str, str], None]) -> bool:
        """
        Unregister a previously registered callback
        
        Args:
            event_name: Name of the event
            callback: The callback function to remove
            
        Returns:
            Success flag
        """
        with self.callback_lock:
            if event_name in self.event_handlers:
                try:
                    self.event_handlers[event_name].remove(callback)
                    logger.debug(f"Unregistered callback for event: {event_name}")
                    return True
                except ValueError:
                    logger.warning(f"Callback not found for event: {event_name}")
        
        return False
    
    def notify_file_change(self, file_type: str, filename: str, event_type: int = 1) -> bool:
        """
        Notify C code about a file change
        
        Args:
            file_type: Type of file (from FileType enum value)
            filename: Name of the file
            event_type: 1=create/update, 2=delete
            
        Returns:
            Success flag
        """
        if not self.lib or not hasattr(self.lib, 'notify_file_change'):
            logger.error("Cannot notify file change: Library not loaded or function not available")
            return False
        
        try:
            result = self.lib.notify_file_change(
                file_type.encode('utf-8'),
                filename.encode('utf-8'),
                event_type
            )
            
            if result != 0:
                logger.error(f"Error notifying file change, code: {result}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error calling notify_file_change: {e}", exc_info=True)
            return False
    
    def init_task_manager(self, max_tasks: int) -> Optional[int]:
        """
        Initialize the task manager in C code
        
        Args:
            max_tasks: Maximum number of tasks
            
        Returns:
            Task manager pointer (as int) or None on failure
        """
        if not self.lib or not hasattr(self.lib, 'create_task_manager'):
            logger.error("Cannot create task manager: Library not loaded or function not available")
            return None
        
        try:
            task_manager_ptr = self.lib.create_task_manager(max_tasks)
            
            if not task_manager_ptr:
                logger.error("Failed to create task manager")
                return None
            
            return task_manager_ptr
            
        except Exception as e:
            logger.error(f"Error calling create_task_manager: {e}", exc_info=True)
            return None
    
    def add_task(self, task_manager_ptr: int, task_id: int, priority: int, 
                assigned_node_id: int, task_data: str) -> bool:
        """
        Add a task to the task manager
        
        Args:
            task_manager_ptr: Task manager pointer
            task_id: Task ID
            priority: Priority (higher = more important)
            assigned_node_id: ID of the node assigned to the task (0 = unassigned)
            task_data: Task data as JSON string
            
        Returns:
            Success flag
        """
        if not self.lib or not hasattr(self.lib, 'add_task'):
            logger.error("Cannot add task: Library not loaded or function not available")
            return False
        
        try:
            result = self.lib.add_task(
                task_manager_ptr,
                task_id,
                priority,
                assigned_node_id,
                task_data.encode('utf-8') if task_data else b''
            )
            
            if result != 0:
                logger.error(f"Error adding task, code: {result}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error calling add_task: {e}", exc_info=True)
            return False
    
    def get_next_task(self, task_manager_ptr: int) -> Optional[str]:
        """
        Get the next task from the task manager
        
        Args:
            task_manager_ptr: Task manager pointer
            
        Returns:
            Task data as JSON string or None if no tasks
        """
        if not self.lib or not hasattr(self.lib, 'get_next_task'):
            logger.error("Cannot get next task: Library not loaded or function not available")
            return None
        
        try:
            task_data = self.lib.get_next_task(task_manager_ptr)
            
            if not task_data:
                return None
            
            return task_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error calling get_next_task: {e}", exc_info=True)
            return None
    
    def complete_task(self, task_manager_ptr: int, task_id: int) -> bool:
        """
        Mark a task as complete
        
        Args:
            task_manager_ptr: Task manager pointer
            task_id: Task ID
            
        Returns:
            Success flag
        """
        if not self.lib or not hasattr(self.lib, 'complete_task'):
            logger.error("Cannot complete task: Library not loaded or function not available")
            return False
        
        try:
            result = self.lib.complete_task(task_manager_ptr, task_id)
            
            if result != 0:
                logger.error(f"Error completing task, code: {result}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error calling complete_task: {e}", exc_info=True)
            return False
    
    def create_node(self, node_type: int, position: List[float], attributes: Dict[str, Any]) -> Optional[int]:
        """
        Create a node in the C system
        
        Args:
            node_type: Type of node (0=standard, 1=core, 2=capability)
            position: [x, y, z] position
            attributes: Node attributes
            
        Returns:
            Node ID or None on failure
        """
        if not self.lib or not hasattr(self.lib, 'create_node'):
            logger.error("Cannot create node: Library not loaded or function not available")
            return None
        
        try:
            attributes_json = json.dumps(attributes)
            
            node_id = self.lib.create_node(
                node_type,
                position[0], position[1], position[2],
                attributes_json.encode('utf-8')
            )
            
            if node_id <= 0:
                logger.error(f"Error creating node, code: {node_id}")
                return None
            
            return node_id
            
        except Exception as e:
            logger.error(f"Error calling create_node: {e}", exc_info=True)
            return None
    
    def connect_nodes(self, source_id: int, target_id: int, strength: float) -> bool:
        """
        Connect two nodes in the C system
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            strength: Connection strength
            
        Returns:
            Success flag
        """
        if not self.lib or not hasattr(self.lib, 'connect_nodes'):
            logger.error("Cannot connect nodes: Library not loaded or function not available")
            return False
        
        try:
            result = self.lib.connect_nodes(source_id, target_id, strength)
            
            if result != 0:
                logger.error(f"Error connecting nodes, code: {result}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error calling connect_nodes: {e}", exc_info=True)
            return False
    
    def get_node_data(self, node_id: int) -> Optional[Dict[str, Any]]:
        """
        Get node data from the C system
        
        Args:
            node_id: Node ID
            
        Returns:
            Node data dictionary or None on failure
        """
        if not self.lib or not hasattr(self.lib, 'get_node_data'):
            logger.error("Cannot get node data: Library not loaded or function not available")
            return None
        
        try:
            node_data = self.lib.get_node_data(node_id)
            
            if not node_data:
                return None
            
            return json.loads(node_data.decode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error calling get_node_data: {e}", exc_info=True)
            return None
    
    def update_node_position(self, node_id: int, position: List[float]) -> bool:
        """
        Update a node's position in the C system
        
        Args:
            node_id: Node ID
            position: [x, y, z] position
            
        Returns:
            Success flag
        """
        if not self.lib or not hasattr(self.lib, 'update_node_position'):
            logger.error("Cannot update node position: Library not loaded or function not available")
            return False
        
        try:
            result = self.lib.update_node_position(
                node_id,
                position[0], position[1], position[2]
            )
            
            if result != 0:
                logger.error(f"Error updating node position, code: {result}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error calling update_node_position: {e}", exc_info=True)
            return False
    
    def get_all_nodes(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get all nodes from the C system
        
        Returns:
            List of node dictionaries or None on failure
        """
        if not self.lib or not hasattr(self.lib, 'get_all_nodes'):
            logger.error("Cannot get all nodes: Library not loaded or function not available")
            return None
        
        try:
            nodes_json = self.lib.get_all_nodes()
            
            if not nodes_json:
                return []
            
            return json.loads(nodes_json.decode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error calling get_all_nodes: {e}", exc_info=True)
            return None
    
    def remove_node(self, node_id: int) -> bool:
        """
        Remove a node from the C system
        
        Args:
            node_id: Node ID
            
        Returns:
            Success flag
        """
        if not self.lib or not hasattr(self.lib, 'remove_node'):
            logger.error("Cannot remove node: Library not loaded or function not available")
            return False
        
        try:
            result = self.lib.remove_node(node_id)
            
            if result != 0:
                logger.error(f"Error removing node, code: {result}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error calling remove_node: {e}", exc_info=True)
            return False
    
    def shutdown(self) -> bool:
        """
        Shutdown the C system
        
        Returns:
            Success flag
        """
        if not self.lib or not hasattr(self.lib, 'shutdown_system'):
            logger.warning("Cannot shutdown: Library not loaded or function not available")
            return False
        
        try:
            result = self.lib.shutdown_system()
            
            if result != 0:
                logger.error(f"Error shutting down system, code: {result}")
                return False
            
            logger.info("C system shut down successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error calling shutdown_system: {e}", exc_info=True)
            return False
        finally:
            # Remove reference to the library
            self.lib = None


# Create a singleton instance
bridge = CBridge()