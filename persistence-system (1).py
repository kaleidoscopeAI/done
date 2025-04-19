import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
import networkx as nx
from pathlib import Path
import pickle
import hashlib
import logging

class KnowledgePersistence:
    """Manages persistence and recovery of system knowledge and state."""
    
    def __init__(self, database_path: str = "kaleidoscope.db"):
        self.database_path = database_path
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self._initialize_database()
        self.logger = logging.getLogger(__name__)

    def _initialize_database(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # Create tables for different types of data
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge_nodes (
                node_id TEXT PRIMARY KEY,
                data BLOB,
                metadata TEXT,
                timestamp TEXT,
                version INTEGER
            );

            CREATE TABLE IF NOT EXISTS relationships (
                source_id TEXT,
                target_id TEXT,
                relationship_type TEXT,
                weight REAL,
                timestamp TEXT,
                FOREIGN KEY (source_id) REFERENCES knowledge_nodes (node_id),
                FOREIGN KEY (target_id) REFERENCES knowledge_nodes (node_id)
            );

            CREATE TABLE IF NOT EXISTS system_state (
                state_id TEXT PRIMARY KEY,
                state_type TEXT,
                state_data BLOB,
                timestamp TEXT
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                description TEXT,
                state_hash TEXT,
                timestamp TEXT
            );
        """)

        conn.commit()
        conn.close()

    def save_knowledge_state(self, network: nx.DiGraph):
        """Persists current knowledge network state."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()

        try:
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")

            # Save nodes
            for node_id, data in network.nodes(data=True):
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO knowledge_nodes 
                    (node_id, data, metadata, timestamp, version)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        node_id,
                        pickle.dumps(data.get('data')),
                        json.dumps(data.get('metadata', {})),
                        timestamp,
                        data.get('version', 1)
                    )
                )

            # Save relationships
            cursor.execute("DELETE FROM relationships")  # Clear existing relationships
            for source, target, data in network.edges(data=True):
                cursor.execute(
                    """
                    INSERT INTO relationships 
                    (source_id, target_id, relationship_type, weight, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        source,
                        target,
                        data.get('type', 'default'),
                        data.get('weight', 1.0),
                        timestamp
                    )
                )

            conn.commit()
            self.logger.info(f"Successfully saved knowledge state at {timestamp}")

        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error saving knowledge state: {str(e)}")
            raise

        finally:
            conn.close()

    def load_knowledge_state(self) -> nx.DiGraph:
        """Recovers knowledge network from persistent storage."""
        network = nx.DiGraph()
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        try:
            # Load nodes
            cursor.execute("SELECT node_id, data, metadata, version FROM knowledge_nodes")
            for node_id, data_blob, metadata_str, version in cursor.fetchall():
                network.add_node(
                    node_id,
                    data=pickle.loads(data_blob),
                    metadata=json.loads(metadata_str),
                    version=version
                )

            # Load relationships
            cursor.execute(
                """
                SELECT source_id, target_id, relationship_type, weight 
                FROM relationships
                """
            )
            for source, target, rel_type, weight in cursor.fetchall():
                network.add_edge(
                    source,
                    target,
                    type=rel_type,
                    weight=weight
                )

            self.logger.info("Successfully loaded knowledge state")
            return network

        except Exception as e:
            self.logger.error(f"Error loading knowledge state: {str(e)}")
            raise

        finally:
            conn.close()

    def create_checkpoint(self, network: nx.DiGraph, description: str = ""):
        """Creates a checkpoint of current system state."""
        checkpoint_id = hashlib.sha256(
            datetime.now().isoformat().encode()
        ).hexdigest()[:16]
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
        
        try:
            # Serialize network state
            state_data = {
                'network': network,
                'timestamp': datetime.now().isoformat(),
                'description': description
            }
            
            # Save checkpoint file
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state_data, f)

            # Record checkpoint in database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO checkpoints 
                (checkpoint_id, description, state_hash, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    description,
                    self._calculate_state_hash(network),
                    datetime.now().isoformat()
                )
            )
            
            conn.commit()
            conn.close()

            self.logger.info(f"Created checkpoint: {checkpoint_id}")
            return checkpoint_id

        except Exception as e:
            self.logger.error(f"Error creating checkpoint: {str(e)}")
            raise

    def restore_from_checkpoint(self, checkpoint_id: str) -> Optional[nx.DiGraph]:
        """Restores system state from a checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
        
        if not checkpoint_path.exists():
            self.logger.error(f"Checkpoint not found: {checkpoint_id}")
            return None

        try:
            with open(checkpoint_path, 'rb') as f:
                state_data = pickle.load(f)
                
            self.logger.info(f"Restored from checkpoint: {checkpoint_id}")
            return state_data['network']

        except Exception as e:
            self.logger.error(f"Error restoring from checkpoint: {str(e)}")
            return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Lists all available checkpoints."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT checkpoint_id, description, timestamp 
            FROM checkpoints 
            ORDER BY timestamp DESC
            """
        )
        
        checkpoints = [
            {
                'id': row[0],
                'description': row[1],
                'timestamp': row[2]
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        return checkpoints

    def _calculate_state_hash(self, network: nx.DiGraph) -> str:
        """Calculates a hash of the current network state."""
        state_repr = (
            f"{sorted(network.nodes(data=True))}:"
            f"{sorted(network.edges(data=True))}"
        )
        return hashlib.sha256(state_repr.encode()).hexdigest()

    def cleanup_old_checkpoints(self, max_checkpoints: int = 10):
        """Removes old checkpoints beyond the maximum limit."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > max_checkpoints:
            checkpoints_to_remove = checkpoints[max_checkpoints:]
            
            for checkpoint in checkpoints_to_remove:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint['id']}.pkl"
                
                try:
                    # Remove checkpoint file
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()

                    # Remove from database
                    conn = sqlite3.connect(self.database_path)
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM checkpoints WHERE checkpoint_id = ?",
                        (checkpoint['id'],)
                    )
                    conn.commit()
                    conn.close()

                except Exception as e:
                    self.logger.error(f"Error removing checkpoint {checkpoint['id']}: {str(e)}")

    def get_state_diff(self, checkpoint1_id: str, checkpoint2_id: str) -> Dict[str, Any]:
        """Calculates differences between two checkpoints."""
        network1 = self.restore_from_checkpoint(checkpoint1_id)
        network2 = self.restore_from_checkpoint(checkpoint2_id)
        
        if not network1 or not network2:
            return {}

        diff = {
            'nodes': {
                'added': list(set(network2.nodes()) - set(network1.nodes())),
                'removed': list(set(network1.nodes()) - set(network2.nodes()))
            },
            'edges': {
                'added': list(set(network2.edges()) - set(network1.edges())),
                'removed': list(set(network1.edges()) - set(network2.edges()))
            }
        }

        # Calculate changed node attributes
        diff['node_changes'] = {}
        for node in set(network1.nodes()) & set(network2.nodes()):
            attrs1 = network1.nodes[node]
            attrs2 = network2.nodes[node]
            if attrs1 != attrs2:
                diff['node_changes'][node] = {
                    'before': attrs1,
                    'after': attrs2
                }

        return diff
