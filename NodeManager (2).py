import json
import logging
import traceback
from typing import Dict, Any, Optional, Tuple, List

# Import the corrected bridge interface
from kaleidoscope_ai.bridge.c_bridge import BridgeInterface, BridgeError

class NodeError(Exception):
    """Exception raised for errors in the NodeManager."""
    pass

class NodeManager:
    """
    Manages nodes in the Kaleidoscope AI system using the C bridge.
    Provides a Pythonic interface for creating, updating, deleting,
    and retrieving nodes, interacting with the C backend via the bridge.
    """

    def __init__(self, bridge: BridgeInterface):
        """
        Initialize the NodeManager.
        Args:
            bridge: An initialized instance of the BridgeInterface.
        """
        if not isinstance(bridge, BridgeInterface):
            raise TypeError("NodeManager requires an initialized BridgeInterface instance.")
        self.bridge = bridge
        # Local cache of nodes - might become inconsistent if C side changes nodes directly.
        # Consider fetching from C or relying solely on C state if consistency is critical.
        self.nodes: Dict[int, Dict[str, Any]] = {} # Store nodes by their uint64 ID from C
        self.logger = logging.getLogger("kaleidoscope.NodeManager")
        self.logger.info("NodeManager initialized.")
        # Optionally, fetch initial node state from C if supported by the bridge
        # self._sync_nodes_from_bridge()

    def _sync_nodes_from_bridge(self):
        """(Optional) Sync local node cache with the C backend."""
        self.logger.warning("_sync_nodes_from_bridge not implemented. Local cache may be stale.")
        # Placeholder: If bridge had a 'get_all_nodes' function:
        # try:
        #     all_nodes_data = self.bridge.get_all_nodes() # Assuming this returns list of dicts
        #     self.nodes = {node_data['node_id']: node_data for node_data in all_nodes_data}
        #     self.logger.info(f"Synced {len(self.nodes)} nodes from C bridge.")
        # except BridgeError as e:
        #     self.logger.error(f"Failed to sync nodes from bridge: {e}")

    def create_node(self, node_id: Any, node_type: Any, properties: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Create a new node in the system via the C bridge.

        Args:
            node_id: Requested unique identifier for the node (int or str).
            node_type: Type of node to create (int or str).
            properties: Optional dictionary of node properties to set initially.

        Returns:
            The actual integer node ID assigned by the C backend, or None on failure.

        Raises:
            NodeError: If the node creation fails.
        """
        self.logger.debug(f"Requesting node creation: id={node_id}, type={node_type}, props={properties}")
        try:
            # Create node in C backend first
            c_node_id = self.bridge.create_node(node_id, node_type)

            if c_node_id is None:
                # Error already logged by bridge
                return None # Creation failed in C

            # If properties are provided, update the newly created node
            if properties:
                if not self.update_node(c_node_id, properties):
                    # Update failed, log warning, but node was created
                    self.logger.warning(f"Node {c_node_id} created, but initial property update failed.")
                    # Decide whether to delete the node or keep it partially created
                    # self.delete_node(c_node_id) # Option: Rollback by deleting
                    # return None

            # Update local cache (use the ID returned by C)
            self.nodes[c_node_id] = {
                "type": node_type, # Store original requested type for reference
                "properties": properties if properties else {}
            }
            self.logger.info(f"Successfully created node {c_node_id} (requested id: {node_id})")
            return c_node_id

        except BridgeError as e:
            error_msg = f"Bridge error creating node (requested id: {node_id}): {e}"
            self.logger.error(error_msg)
            raise NodeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error creating node (requested id: {node_id}): {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise NodeError(error_msg) from e

    def update_node(self, node_id: Any, properties: Dict[str, Any]) -> bool:
        """
        Update node properties via the C bridge.

        Args:
            node_id: ID of the node to update (int or str).
            properties: Dictionary of properties to update/add.

        Returns:
            bool: True if update was successful in the C backend.

        Raises:
            NodeError: If the update fails.
        """
        self.logger.debug(f"Requesting node update: id={node_id}, props={properties}")
        if not isinstance(properties, dict):
            raise TypeError("properties must be a dictionary")

        try:
            # Update node in C backend
            success = self.bridge.update_node(node_id, properties)

            if success:
                # Update local cache if the node exists there
                # Need the actual uint64 ID used by the bridge
                try:
                    c_node_id = self.bridge._id_to_uint64(node_id) # Use bridge's internal conversion
                    if c_node_id in self.nodes:
                        self.nodes[c_node_id]["properties"].update(properties)
                    else:
                        # If not in cache, maybe fetch it or just log
                        self.logger.warning(f"Node {c_node_id} updated in C, but not found in local cache.")
                        # Option: Add to cache after successful update
                        # self.nodes[c_node_id] = {"type": "unknown", "properties": properties}
                except TypeError:
                     self.logger.error(f"Could not convert node_id {node_id} for cache update.")

                self.logger.info(f"Successfully updated node {node_id}")
            else:
                # Error already logged by bridge
                pass # Failed to update in C

            return success
        except BridgeError as e:
            error_msg = f"Bridge error updating node {node_id}: {e}"
            self.logger.error(error_msg)
            raise NodeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error updating node {node_id}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise NodeError(error_msg) from e

    def get_node(self, node_id: Any) -> Optional[Dict[str, Any]]:
        """
        Get a node's data from the local cache.
        Note: This might be stale if the C side modifies nodes independently.

        Args:
            node_id: ID of the node to retrieve (int or str).

        Returns:
            Dict or None: Node data if found in cache, None otherwise.

        Raises:
            NodeError: If an error occurs during retrieval.
        """
        self.logger.debug(f"Requesting node get: id={node_id}")
        try:
            c_node_id = self.bridge._id_to_uint64(node_id) # Convert to consistent ID format
            node_data = self.nodes.get(c_node_id)
            if node_data:
                 self.logger.debug(f"Found node {c_node_id} in cache.")
            else:
                 self.logger.debug(f"Node {c_node_id} not found in cache.")
                 # Optionally try fetching from C bridge if a get_node function exists
                 # node_data = self.bridge.get_node_data(c_node_id)
                 # if node_data: self.nodes[c_node_id] = node_data # Update cache
            return node_data
        except BridgeError as e:
             error_msg = f"Bridge error during get_node for {node_id}: {e}"
             self.logger.error(error_msg)
             raise NodeError(error_msg) from e
        except TypeError as e:
             error_msg = f"Invalid node ID type for get_node {node_id}: {e}"
             self.logger.error(error_msg)
             raise NodeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error retrieving node {node_id}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise NodeError(error_msg) from e

    def delete_node(self, node_id: Any) -> bool:
        """
        Delete a node via the C bridge.

        Args:
            node_id: ID of the node to delete (int or str).

        Returns:
            bool: True if deletion was successful in the C backend.

        Raises:
            NodeError: If the deletion fails.
        """
        self.logger.debug(f"Requesting node delete: id={node_id}")
        try:
            # Delete node in C backend
            success = self.bridge.delete_node(node_id)

            if success:
                # Remove from local cache if it exists
                try:
                    c_node_id = self.bridge._id_to_uint64(node_id) # Convert ID
                    if c_node_id in self.nodes:
                        del self.nodes[c_node_id]
                except TypeError:
                     self.logger.error(f"Could not convert node_id {node_id} for cache removal.")
                except KeyError:
                     pass # Node wasn't in cache anyway

                self.logger.info(f"Successfully deleted node {node_id}")
            else:
                # Error already logged by bridge
                pass # Failed to delete in C

            return success
        except BridgeError as e:
            error_msg = f"Bridge error deleting node {node_id}: {e}"
            self.logger.error(error_msg)
            raise NodeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error deleting node {node_id}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise NodeError(error_msg) from e

    def connect_nodes(self, source_id: Any, target_id: Any, strength: float = 1.0) -> bool:
        """
        Connect two nodes via the C bridge.

        Args:
            source_id: ID of the source node (int or str).
            target_id: ID of the target node (int or str).
            strength: Strength of the connection (default: 1.0).

        Returns:
            bool: True if connection was successful in the C backend.

        Raises:
            NodeError: If the connection fails.
        """
        self.logger.debug(f"Requesting node connection: {source_id} -> {target_id} (strength={strength})")
        try:
            success = self.bridge.connect_nodes(source_id, target_id, strength)
            if success:
                self.logger.info(f"Successfully connected nodes {source_id} -> {target_id}")
            else:
                # Error logged by bridge
                pass
            return success
        except BridgeError as e:
            error_msg = f"Bridge error connecting nodes {source_id} -> {target_id}: {e}"
            self.logger.error(error_msg)
            raise NodeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error connecting nodes {source_id} -> {target_id}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise NodeError(error_msg) from e

    def get_updated_nodes(self) -> List[Dict[str, Any]]:
        """
        Retrieve a list of nodes that have been updated in the C backend.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing an updated node.
                                  Returns an empty list if no updates or an error occurs.
        """
        self.logger.debug("Requesting updated nodes from bridge.")
        try:
            updated_nodes_data = self.bridge.get_updated_nodes()
            # Update local cache based on received updates
            for node_data in updated_nodes_data:
                node_id = node_data.get('node_id')
                if node_id is not None:
                    if node_id in self.nodes:
                        # Update existing cached node (partially, based on available data)
                        self.nodes[node_id]['properties'].update({
                            'energy': node_data.get('energy'),
                            'last_update': node_data.get('timestamp')
                        })
                        # Type might also change, update if necessary
                        # self.nodes[node_id]['type'] = node_data.get('type', self.nodes[node_id]['type'])
                    else:
                        # Add newly seen node to cache (might be incomplete)
                        self.nodes[node_id] = {
                            'type': node_data.get('type', 'unknown'),
                            'properties': {
                                'energy': node_data.get('energy'),
                                'last_update': node_data.get('timestamp')
                            }
                        }
            self.logger.debug(f"Received {len(updated_nodes_data)} updated nodes.")
            return updated_nodes_data
        except BridgeError as e:
            self.logger.error(f"Bridge error getting updated nodes: {e}")
            return [] # Return empty list on error
        except Exception as e:
            self.logger.error(f"Unexpected error getting updated nodes: {e}")
            self.logger.debug(traceback.format_exc())
            return []

    def list_nodes(self) -> List[int]:
         """List node IDs currently in the local cache."""
         return list(self.nodes.keys())

# Example Usage (using the NodeManager)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Initialize the bridge first
        bridge = BridgeInterface()

        # Pass the bridge to the NodeManager
        node_manager = NodeManager(bridge)

        # Create nodes using NodeManager
        node_a_id = node_manager.create_node("node_A", "standard", {"color": "red", "size": 10})
        node_b_id = node_manager.create_node("1001", "core", {"status": "init"})

        if node_a_id and node_b_id:
            logger.info(f"Created node A: {node_a_id}, Node B: {node_b_id}")

            # Connect nodes
            node_manager.connect_nodes(node_a_id, node_b_id, 0.8)

            # Update node
            node_manager.update_node(node_a_id, {"size": 12, "label": "Alpha Node"})

            # Get node data (from cache)
            node_a_data = node_manager.get_node(node_a_id)
            logger.info(f"Node A data (cache): {node_a_data}")

            # Get updated nodes (from C)
            updates = node_manager.get_updated_nodes()
            logger.info(f"Updated nodes from C: {updates}")

            # List nodes in cache
            logger.info(f"Nodes in cache: {node_manager.list_nodes()}")

            # Delete node
            node_manager.delete_node(node_a_id)
            logger.info(f"Nodes in cache after delete: {node_manager.list_nodes()}")

        else:
            logger.error("Failed to create initial nodes.")

    except NodeError as e:
        logger.error(f"NodeManager operation failed: {e}")
    except BridgeError as e:
        logger.error(f"Bridge operation failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Bridge shutdown is handled by its destructor via NodeManager holding a reference
        logger.info("NodeManager example finished.")