#!/usr/bin/env python3
"""
supernode_manager.py

Main interface and management system for SuperNode Collective Intelligence.
Entry point for all SuperNode operations with data flow control.
"""

import numpy as np
import torch
import logging
import asyncio
import json
import time
import os
import threading
import sys
import argparse
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Callable
from dataclasses import dataclass, field, asdict
import uuid
import concurrent.futures
from enum import Enum, auto

# Import from other components
from supernode_core import (
    SuperNodeCore, SuperNodeDNA, SuperNodeState, 
    encode_data, decode_data, ResonanceMode
)
from supernode_processor import (
    SuperNodeProcessor, Pattern, Insight, Speculation, Perspective,
    PatternType, InsightType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SuperNodeManager")

# Define constants
DEFAULT_DIMENSION = 1024
DEFAULT_CONCURRENCY = 4
DEFAULT_TIMEOUT = 300  # seconds
DEFAULT_RESONANCE_MODE = ResonanceMode.HYBRID

class SuperNodeStatus(Enum):
    """Status states for a SuperNode"""
    INITIALIZING = auto()
    ACTIVE = auto()
    PROCESSING = auto()
    IDLE = auto()
    MERGING = auto()
    ERROR = auto()
    TERMINATED = auto()

@dataclass
class SuperNodeConfig:
    """Configuration for a SuperNode instance"""
    id: str
    dimension: int = DEFAULT_DIMENSION
    resonance_mode: ResonanceMode = DEFAULT_RESONANCE_MODE
    max_patterns: int = 100000
    max_insights: int = 10000
    max_perspectives: int = 1000
    speculation_depth: int = 3
    enable_persistence: bool = False
    persistence_path: str = "./data"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingResult:
    """Results from a processing operation"""
    id: str
    input_id: str
    node_id: str
    timestamp: float
    duration: float
    pattern_count: int
    insight_count: int
    speculation_count: int
    perspective_count: int
    patterns: List[str]
    insights: List[str]
    perspectives: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingResult':
        """Create from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ProcessingResult':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))

@dataclass
class SuperNodeInput:
    """Input data for processing"""
    id: str
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "id": self.id,
            "data": self.data.tolist(),
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SuperNodeInput':
        """Create from dictionary"""
        input_data = np.array(data["data"])
        return cls(
            id=data["id"],
            data=input_data,
            metadata=data["metadata"],
            timestamp=data["timestamp"]
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SuperNodeInput':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))

class SuperNodeInstance:
    """
    Management wrapper for a SuperNode instance.
    Handles lifecycle, persistence, and operations.
    """
    def __init__(self, config: SuperNodeConfig):
        self.config = config
        self.id = config.id
        self.status = SuperNodeStatus.INITIALIZING
        self.creation_time = time.time()
        self.last_activity = self.creation_time
        
        # Initialize components
        self.core = SuperNodeCore(
            dimension=config.dimension, 
            resonance_mode=config.resonance_mode
        )
        self.processor = SuperNodeProcessor(self.core)
        
        # Processing history
        self.processing_results = {}  # result_id -> ProcessingResult
        self.inputs = {}  # input_id -> SuperNodeInput
        
        # Configure processor
        self.processor.speculation_depth = config.speculation_depth
        
        # Thread lock
        self.lock = threading.RLock()
        
        # Persistence
        self.enable_persistence = config.enable_persistence
        if self.enable_persistence:
            self.persistence_path = config.persistence_path
            # Create directory if it doesn't exist
            os.makedirs(self.persistence_path, exist_ok=True)
            
        self.logger = logging.getLogger(f"SuperNodeInstance_{self.id}")
        self.logger.info(f"Initialized SuperNode instance with ID: {self.id}")
    
    def start(self) -> None:
        """Start the SuperNode instance"""
        with self.lock:
            if self.status != SuperNodeStatus.INITIALIZING:
                self.logger.warning(f"Cannot start SuperNode in state: {self.status}")
                return
                
            # Start core evolution
            self.core.start()
            
            # Update status
            self.status = SuperNodeStatus.IDLE
            self.last_activity = time.time()
            
            self.logger.info(f"SuperNode {self.id} started successfully")
    
    def stop(self) -> None:
        """Stop the SuperNode instance"""
        with self.lock:
            if self.status == SuperNodeStatus.TERMINATED:
                self.logger.warning(f"SuperNode {self.id} already terminated")
                return
                
            # Stop core evolution
            self.core.stop()
            
            # Save state if persistence enabled
            if self.enable_persistence:
                self._save_state()
                
            # Update status
            self.status = SuperNodeStatus.TERMINATED
            self.last_activity = time.time()
            
            self.logger.info(f"SuperNode {self.id} stopped successfully")
    
    def process_data(self, input_data: SuperNodeInput) -> ProcessingResult:
        """
        Process input data through the SuperNode.
        
        Args:
            input_data: Input data object
            
        Returns:
            Processing result object
        """
        with self.lock:
            # Check status
            if self.status not in [SuperNodeStatus.IDLE, SuperNodeStatus.ACTIVE]:
                self.logger.warning(f"Cannot process data in state: {self.status}")
                raise RuntimeError(f"SuperNode {self.id} is not ready for processing")
                
            # Update status
            self.status = SuperNodeStatus.PROCESSING
            self.last_activity = time.time()
            
            # Store input
            self.inputs[input_data.id] = input_data
            
            # Process data
            start_time = time.time()
            result = None
            error = None
            
            try:
                # Process through processor
                proc_result = self.processor.process_data(
                    input_data.data, 
                    input_data.metadata
                )
                
                # Create result object
                result_id = f"result_{uuid.uuid4().hex}"
                result = ProcessingResult(
                    id=result_id,
                    input_id=input_data.id,
                    node_id=self.id,
                    timestamp=time.time(),
                    duration=time.time() - start_time,
                    pattern_count=proc_result.get("pattern_count", 0),
                    insight_count=proc_result.get("insight_count", 0),
                    speculation_count=proc_result.get("speculation_count", 0),
                    perspective_count=proc_result.get("perspective_count", 0),
                    patterns=proc_result.get("patterns", []),
                    insights=proc_result.get("insights", []),
                    perspectives=proc_result.get("perspectives", []),
                    metadata=input_data.metadata
                )
                
                # Store result
                self.processing_results[result_id] = result
                
                # Persist state if enabled
                if self.enable_persistence:
                    self._save_result(result)
                    
            except Exception as e:
                self.logger.error(f"Error processing data: {e}")
                error = e
                
            # Update status
            self.status = SuperNodeStatus.ACTIVE if error is None else SuperNodeStatus.ERROR
            self.last_activity = time.time()
            
            if error:
                raise RuntimeError(f"Error processing data: {error}")
                
            return result
    
    def process_text(self, text: str, metadata: Dict[str, Any] = None) -> ProcessingResult:
        """
        Process text input through the SuperNode.
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            Processing result object
        """
        if metadata is None:
            metadata = {}
            
        # Update metadata
        metadata["content_type"] = "text"
        metadata["text_length"] = len(text)
        
        # Encode text to vector
        data = encode_data(text)
        
        # Create input object
        input_id = f"input_{uuid.uuid4().hex}"
        input_data = SuperNodeInput(
            id=input_id,
            data=data,
            metadata=metadata
        )
        
        # Process data
        return self.process_data(input_data)
    
    def get_insights(self) -> List[Dict[str, Any]]:
        """Get all insights from the processor"""
        with self.lock:
            return self.processor.get_all_insights()
    
    def get_perspectives(self) -> List[Dict[str, Any]]:
        """Get all perspectives from the processor"""
        with self.lock:
            return self.processor.get_all_perspectives()
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations based on insights and perspectives"""
        with self.lock:
            return self.processor.generate_recommendations()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the SuperNode"""
        with self.lock:
            core_status = self.core.get_status()
            
            return {
                "id": self.id,
                "status": self.status.name,
                "creation_time": self.creation_time,
                "last_activity": self.last_activity,
                "uptime": time.time() - self.creation_time,
                "core_status": core_status,
                "processing_count": len(self.processing_results),
                "input_count": len(self.inputs),
                "pattern_count": len(self.processor.patterns),
                "insight_count": len(self.processor.insights),
                "perspective_count": len(self.processor.perspectives)
            }
    
    def generate_output(self) -> str:
        """
        Generate output text based on current state.
        
        Returns:
            Generated text output
        """
        with self.lock:
            # Get output vector from core
            output_vector = self.core.generate_output()
            
            # Decode to text
            text = decode_data(output_vector)
            
            return text
    
    def extract_knowledge(self) -> Dict[str, Any]:
        """
        Extract structured knowledge from current state.
        
        Returns:
            Dictionary with structured knowledge
        """
        with self.lock:
            # Get top perspectives
            perspectives = self.processor.get_all_perspectives()
            top_perspectives = sorted(perspectives, key=lambda p: p.get("impact", 0), reverse=True)[:5]
            
            # Get top insights
            insights = self.processor.get_all_insights()
            top_insights = sorted(insights, key=lambda i: i.get("importance", 0), reverse=True)[:10]
            
            # Extract patterns from top insights
            knowledge = {
                "perspectives": top_perspectives,
                "insights": top_insights,
                "core_status": self.core.get_status(),
                "timestamp": time.time()
            }
            
            return knowledge
    
    def merge_with(self, other: 'SuperNodeInstance') -> 'SuperNodeInstance':
        """
        Merge this SuperNode with another SuperNode.
        
        Args:
            other: Another SuperNode instance to merge with
            
        Returns:
            New SuperNode instance resulting from the merge
        """
        with self.lock, other.lock:
            # Update status
            self.status = SuperNodeStatus.MERGING
            other.status = SuperNodeStatus.MERGING
            
            # Create new config for merged node
            merged_config = SuperNodeConfig(
                id=f"merged_{uuid.uuid4().hex}",
                dimension=self.config.dimension,
                resonance_mode=self.config.resonance_mode,
                max_patterns=max(self.config.max_patterns, other.config.max_patterns),
                max_insights=max(self.config.max_insights, other.config.max_insights),
                max_perspectives=max(self.config.max_perspectives, other.config.max_perspectives),
                speculation_depth=max(self.config.speculation_depth, other.config.speculation_depth),
                enable_persistence=self.config.enable_persistence or other.config.enable_persistence,
                persistence_path=self.config.persistence_path,
                metadata={
                    "merged_from": [self.id, other.id],
                    "merge_time": time.time()
                }
            )
            
            # Create new instance with merged config
            merged = SuperNodeInstance(merged_config)
            
            # Merge cores
            merged.core = self.core.merge_with(other.core)
            
            # Merge processors
            merged.processor = self.processor.merge_with(other.processor)
            
            # Start merged node
            merged.start()
            
            # Update status
            self.status = SuperNodeStatus.ACTIVE
            other.status = SuperNodeStatus.ACTIVE
            
            return merged
    
    def _save_state(self) -> None:
        """Save state to persistence store"""
        if not self.enable_persistence:
            return
            
        try:
            # Create state directory if it doesn't exist
            node_dir = os.path.join(self.persistence_path, self.id)
            os.makedirs(node_dir, exist_ok=True)
            
            # Save core state
            core_status = self.core.get_status()
            with open(os.path.join(node_dir, "core_status.json"), "w") as f:
                json.dump(core_status, f)
                
            # Save processor state (insights, perspectives)
            insights = self.processor.get_all_insights()
            with open(os.path.join(node_dir, "insights.json"), "w") as f:
                json.dump(insights, f)
                
            perspectives = self.processor.get_all_perspectives()
            with open(os.path.join(node_dir, "perspectives.json"), "w") as f:
                json.dump(perspectives, f)
                
            # Save config
            with open(os.path.join(node_dir, "config.json"), "w") as f:
                json.dump(asdict(self.config), f)
                
            # Save instance status
            status = self.get_status()
            with open(os.path.join(node_dir, "status.json"), "w") as f:
                # Convert enum to string
                status["status"] = status["status"].name if isinstance(status["status"], SuperNodeStatus) else status["status"]
                json.dump(status, f)
                
            self.logger.info(f"Saved state for SuperNode {self.id}")
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def _save_result(self, result: ProcessingResult) -> None:
        """Save processing result to persistence store"""
        if not self.enable_persistence:
            return
            
        try:
            # Create results directory if it doesn't exist
            results_dir = os.path.join(self.persistence_path, self.id, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save result
            with open(os.path.join(results_dir, f"{result.id}.json"), "w") as f:
                json.dump(result.to_dict(), f)
                
            self.logger.info(f"Saved result {result.id} for SuperNode {self.id}")
            
        except Exception as e:
            self.logger.error(f"Error saving result: {e}")
    
    @classmethod
    def load(cls, node_id: str, persistence_path: str) -> Optional['SuperNodeInstance']:
        """
        Load SuperNode instance from persistence store.
        
        Args:
            node_id: ID of the SuperNode to load
            persistence_path: Path to persistence store
            
        Returns:
            Loaded SuperNode instance or None if not found
        """
        node_dir = os.path.join(persistence_path, node_id)
        
        # Check if directory exists
        if not os.path.isdir(node_dir):
            logger.warning(f"SuperNode {node_id} not found in persistence store")
            return None
            
        try:
            # Load config
            with open(os.path.join(node_dir, "config.json"), "r") as f:
                config_dict = json.load(f)
                
            # Create config
            config = SuperNodeConfig(**config_dict)
            
            # Create instance
            instance = cls(config)
            
            # Start instance
            instance.start()
            
            # Load processing results
            results_dir = os.path.join(node_dir, "results")
            if os.path.isdir(results_dir):
                for filename in os.listdir(results_dir):
                    if filename.endswith(".json"):
                        with open(os.path.join(results_dir, filename), "r") as f:
                            result_dict = json.load(f)
                            result = ProcessingResult.from_dict(result_dict)
                            instance.processing_results[result.id] = result
            
            logger.info(f"Loaded SuperNode {node_id} from persistence store")
            
            return instance
            
        except Exception as e:
            logger.error(f"Error loading SuperNode {node_id}: {e}")
            return None

class SuperNodeManager:
    """
    Manager for multiple SuperNode instances.
    Handles creation, management, and inter-node operations.
    """
    def __init__(self, 
                 base_persistence_path: str = "./data",
                 max_concurrent_tasks: int = DEFAULT_CONCURRENCY):
        self.nodes = {}  # node_id -> SuperNodeInstance
        self.base_persistence_path = base_persistence_path
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Create persistence directory if it doesn't exist
        os.makedirs(base_persistence_path, exist_ok=True)
        
        # Thread pool for concurrent operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # Thread lock
        self.lock = threading.RLock()
        
        # Processing queue
        self.processing_queue = []
        self.queue_lock = threading.Lock()
        self.queue_event = threading.Event()
        
        # Start queue processing thread
        self.queue_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.queue_thread.start()
        
        self.logger = logging.getLogger("SuperNodeManager")
        self.logger.info(f"Initialized SuperNodeManager with base path: {base_persistence_path}")
    
    def create_node(self, 
                   config: Optional[SuperNodeConfig] = None,
                   dimension: int = DEFAULT_DIMENSION,
                   resonance_mode: ResonanceMode = DEFAULT_RESONANCE_MODE,
                   enable_persistence: bool = True) -> str:
        """
        Create a new SuperNode instance.
        
        Args:
            config: Optional configuration object
            dimension: Vector dimension if no config provided
            resonance_mode: Resonance mode if no config provided
            enable_persistence: Enable state persistence if no config provided
            
        Returns:
            ID of the created SuperNode
        """
        with self.lock:
            # Create config if not provided
            if config is None:
                node_id = f"node_{uuid.uuid4().hex}"
                persistence_path = os.path.join(self.base_persistence_path, node_id)
                
                config = SuperNodeConfig(
                    id=node_id,
                    dimension=dimension,
                    resonance_mode=resonance_mode,
                    enable_persistence=enable_persistence,
                    persistence_path=persistence_path
                )
            
            # Create instance
            instance = SuperNodeInstance(config)
            
            # Start instance
            instance.start()
            
            # Store instance
            self.nodes[instance.id] = instance
            
            self.logger.info(f"Created SuperNode with ID: {instance.id}")
            
            return instance.id
    
    def delete_node(self, node_id: str) -> bool:
        """
        Delete a SuperNode instance.
        
        Args:
            node_id: ID of the SuperNode to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if node_id not in self.nodes:
                self.logger.warning(f"SuperNode {node_id} not found")
                return False
                
            # Stop instance
            self.nodes[node_id].stop()
            
            # Remove from nodes dict
            del self.nodes[node_id]
            
            self.logger.info(f"Deleted SuperNode {node_id}")
            
            return True
    
    def get_node(self, node_id: str) -> Optional[SuperNodeInstance]:
        """
        Get a SuperNode instance by ID.
        
        Args:
            node_id: ID of the SuperNode to get
            
        Returns:
            SuperNode instance or None if not found
        """
        with self.lock:
            return self.nodes.get(node_id)
    
    def list_nodes(self) -> List[str]:
        """
        List all SuperNode IDs.
        
        Returns:
            List of SuperNode IDs
        """
        with self.lock:
            return list(self.nodes.keys())
    
    def get_node_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a SuperNode.
        
        Args:
            node_id: ID of the SuperNode
            
        Returns:
            Status dictionary or None if not found
        """
        with self.lock:
            node = self.nodes.get(node_id)
            if node is None:
                return None
                
            return node.get_status()
    
    def process_data(self, 
                    node_id: str, 
                    input_data: SuperNodeInput) -> Optional[ProcessingResult]:
        """
        Process data through a SuperNode.
        
        Args:
            node_id: ID of the SuperNode to use
            input_data: Input data object
            
        Returns:
            Processing result or None if node not found
        """
        with self.lock:
            node = self.nodes.get(node_id)
            if node is None:
                self.logger.warning(f"SuperNode {node_id} not found")
                return None
                
            # Process data
            return node.process_data(input_data)
    
    def process_text(self, 
                    node_id: str, 
                    text: str, 
                    metadata: Dict[str, Any] = None) -> Optional[ProcessingResult]:
        """
        Process text through a SuperNode.
        
        Args:
            node_id: ID of the SuperNode to use
            text: Input text
            metadata: Optional metadata
            
        Returns:
            Processing result or None if node not found
        """
        with self.lock:
            node = self.nodes.get(node_id)
            if node is None:
                self.logger.warning(f"SuperNode {node_id} not found")
                return None
                
            # Process text
            return node.process_text(text, metadata)
    
    def process_data_async(self, 
                          node_id: str, 
                          input_data: SuperNodeInput) -> str:
        """
        Process data asynchronously through a SuperNode.
        
        Args:
            node_id: ID of the SuperNode to use
            input_data: Input data object
            
        Returns:
            Task ID for the queued task
        """
        with self.lock:
            node = self.nodes.get(node_id)
            if node is None:
                self.logger.warning(f"SuperNode {node_id} not found")
                raise ValueError(f"SuperNode {node_id} not found")
                
            # Create task ID
            task_id = f"task_{uuid.uuid4().hex}"
            
            # Add to queue
            with self.queue_lock:
                self.processing_queue.append({
                    "task_id": task_id,
                    "node_id": node_id,
                    "input_data": input_data,
                    "timestamp": time.time(),
                    "result": None,
                    "error": None,
                    "status": "queued"
                })
                
                # Signal queue event
                self.queue_event.set()
                
            self.logger.info(f"Queued task {task_id} for SuperNode {node_id}")
            
            return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of an asynchronous task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task status dictionary or None if not found
        """
        with self.queue_lock:
            for task in self.processing_queue:
                if task["task_id"] == task_id:
                    return {
                        "task_id": task["task_id"],
                        "node_id": task["node_id"],
                        "timestamp": task["timestamp"],
                        "status": task["status"],
                        "result": task["result"].to_dict() if task["result"] else None,
                        "error": str(task["error"]) if task["error"] else None
                    }
                    
            return None
    
    def merge_nodes(self, node_ids: List[str]) -> Optional[str]:
        """
        Merge multiple SuperNodes into a new SuperNode.
        
        Args:
            node_ids: List of SuperNode IDs to merge
            
        Returns:
            ID of the merged SuperNode or None if error
        """
        with self.lock:
            # Check all nodes exist
            nodes = []
            for node_id in node_ids:
                node = self.nodes.get(node_id)
                if node is None:
                    self.logger.warning(f"SuperNode {node_id} not found")
                    return None
                    
                nodes.append(node)
                
            if len(nodes) < 2:
                self.logger.warning("Need at least two SuperNodes to merge")
                return None
                
            # Merge nodes
            merged = nodes[0]
            for node in nodes[1:]:
                merged = merged.merge_with(node)
                
            # Store merged node
            self.nodes[merged.id] = merged
            
            self.logger.info(f"Merged {len(nodes)} SuperNodes into {merged.id}")
            
            return merged.id
    
    def save_all(self) -> None:
        """Save state of all SuperNode instances"""
        with self.lock:
            for node in self.nodes.values():
                if node.enable_persistence:
                    try:
                        node._save_state()
                    except Exception as e:
                        self.logger.error(f"Error saving state for SuperNode {node.id}: {e}")
            
            self.logger.info("Saved state of all SuperNodes")
    
    def load_node(self, node_id: str) -> Optional[str]:
        """
        Load a SuperNode from persistence store.
        
        Args:
            node_id: ID of the SuperNode to load
            
        Returns:
            ID of the loaded SuperNode or None if error
        """
        with self.lock:
            # Check if node already loaded
            if node_id in self.nodes:
                self.logger.warning(f"SuperNode {node_id} already loaded")
                return node_id
                
            # Load node
            instance = SuperNodeInstance.load(node_id, self.base_persistence_path)
            if instance is None:
                return None
                
            # Store instance
            self.nodes[instance.id] = instance
            
            return instance.id
    
    def load_all_nodes(self) -> List[str]:
        """
        Load all SuperNodes from persistence store.
        
        Returns:
            List of loaded SuperNode IDs
        """
        # Get all directories in base path
        node_dirs = []
        for item in os.listdir(self.base_persistence_path):
            full_path = os.path.join(self.base_persistence_path, item)
            if os.path.isdir(full_path):
                node_dirs.append(item)
                
        # Load each node
        loaded_nodes = []
        for node_id in node_dirs:
            loaded_id = self.load_node(node_id)
            if loaded_id:
                loaded_nodes.append(loaded_id)
                
        self.logger.info(f"Loaded {len(loaded_nodes)} SuperNodes")
        
        return loaded_nodes
    
    def _process_queue(self) -> None:
        """Background thread to process queued tasks"""
        while True:
            # Wait for queue event
            self.queue_event.wait()
            
            # Process queue
            with self.queue_lock:
                # Clear event
                self.queue_event.clear()
                
                # Get all queued tasks
                tasks = [t for t in self.processing_queue if t["status"] == "queued"]
                
                # Sort by timestamp
                tasks.sort(key=lambda t: t["timestamp"])
                
            # Process tasks (up to max concurrent tasks)
            futures = []
            for task in tasks[:self.max_concurrent_tasks]:
                with self.queue_lock:
                    # Update status
                    for t in self.processing_queue:
                        if t["task_id"] == task["task_id"]:
                            t["status"] = "processing"
                            break
                
                # Submit task
                future = self.executor.submit(
                    self._process_task,
                    task["task_id"],
                    task["node_id"],
                    task["input_data"]
                )
                futures.append((task["task_id"], future))
                
            # Wait for tasks to complete
            for task_id, future in futures:
                try:
                    result = future.result(timeout=DEFAULT_TIMEOUT)
                    
                    # Update task
                    with self.queue_lock:
                        for task in self.processing_queue:
                            if task["task_id"] == task_id:
                                task["result"] = result
                                task["status"] = "completed"
                                break
                                
                except Exception as e:
                    self.logger.error(f"Error processing task {task_id}: {e}")
                    
                    # Update task
                    with self.queue_lock:
                        for task in self.processing_queue:
                            if task["task_id"] == task_id:
                                task["error"] = e
                                task["status"] = "error"
                                break
                                
            # Clean up old completed tasks
            with self.queue_lock:
                # Keep only tasks less than 1 hour old
                current_time = time.time()
                self.processing_queue = [
                    t for t in self.processing_queue
                    if t["status"] in ["queued", "processing"] or
                    (current_time - t["timestamp"]) < 3600
                ]
                
                # Signal event if there are still queued tasks
                if any(t["status"] == "queued" for t in self.processing_queue):
                    self.queue_event.set()
                    
            # Sleep briefly
            time.sleep(0.1)
    
    def _process_task(self, 
                     task_id: str, 
                     node_id: str, 
                     input_data: SuperNodeInput) -> ProcessingResult:
        """Process a queued task"""
        self.logger.info(f"Processing task {task_id} for SuperNode {node_id}")
        
        # Get node
        with self.lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise ValueError(f"SuperNode {node_id} not found")
                
        # Process data
        result = node.process_data(input_data)
        
        self.logger.info(f"Completed task {task_id} for SuperNode {node_id}")
        
        return result

def create_text_processor(dimension: int = 1024, 
                         enable_persistence: bool = True, 
                         persistence_path: str = "./data") -> SuperNodeManager:
    """
    Create a SuperNodeManager configured for text processing.
    
    Args:
        dimension: Vector dimension
        enable_persistence: Enable state persistence
        persistence_path: Path to persistence store
        
    Returns:
        Configured SuperNodeManager
    """
    # Create manager
    manager = SuperNodeManager(base_persistence_path=persistence_path)
    
    # Create initial node
    config = SuperNodeConfig(
        id=f"text_processor_{uuid.uuid4().hex}",
        dimension=dimension,
        resonance_mode=ResonanceMode.HYBRID,
        enable_persistence=enable_persistence,
        persistence_path=persistence_path
    )
    
    node_id = manager.create_node(config=config)
    
    return manager, node_id

def process_file(file_path: str, manager: SuperNodeManager, node_id: str) -> ProcessingResult:
    """
    Process a file through a SuperNode.
    
    Args:
        file_path: Path to the file
        manager: SuperNodeManager instance
        node_id: ID of the SuperNode to use
        
    Returns:
        Processing result
    """
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Extract metadata
    metadata = {
        "filename": os.path.basename(file_path),
        "file_path": file_path,
        "file_size": os.path.getsize(file_path),
        "content_type": "text"
    }
    
    # Process content
    return manager.process_text(node_id, content, metadata)

# Main entry point for command-line usage
def main():
    parser = argparse.ArgumentParser(description='SuperNode Collective Intelligence System')
    parser.add_argument('--action', choices=['create', 'process', 'status', 'extract', 'merge'], 
                        default='create', help='Action to perform')
    parser.add_argument('--input', type=str, help='Input file or directory')
    parser.add_argument('--node-id', type=str, help='SuperNode ID')
    parser.add_argument('--node-ids', type=str, help='Comma-separated list of SuperNode IDs to merge')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--persistence', type=str, default='./data', help='Persistence directory')
    parser.add_argument('--dimension', type=int, default=1024, help='Vector dimension')
    parser.add_argument('--no-persistence', action='store_true', help='Disable persistence')
    
    args = parser.parse_args()
    
    # Create manager
    manager = SuperNodeManager(base_persistence_path=args.persistence)
    
    if args.action == 'create':
        # Create a new SuperNode
        config = SuperNodeConfig(
            id=f"node_{uuid.uuid4().hex}",
            dimension=args.dimension,
            enable_persistence=not args.no_persistence,
            persistence_path=args.persistence
        )
        
        node_id = manager.create_node(config=config)
        print(f"Created SuperNode: {node_id}")
        
    elif args.action == 'process':
        # Process input file or directory
        if not args.input:
            print("Error: --input required for 'process' action")
            return 1
            
        if not args.node_id:
            print("Error: --node-id required for 'process' action")
            return 1
            
        if os.path.isfile(args.input):
            # Process single file
            result = process_file(args.input, manager, args.node_id)
            print(f"Processed file: {args.input}")
            print(f"Result: {result.id}")
            print(f"Patterns: {result.pattern_count}")
            print(f"Insights: {result.insight_count}")
            print(f"Perspectives: {result.perspective_count}")
            
        elif os.path.isdir(args.input):
            # Process all files in directory
            results = []
            for root, _, files in os.walk(args.input):
                for filename in files:
                    if filename.endswith('.txt') or filename.endswith('.py') or filename.endswith('.c'):
                        file_path = os.path.join(root, filename)
                        try:
                            result = process_file(file_path, manager, args.node_id)
                            results.append((file_path, result))
                            print(f"Processed file: {file_path}")
                        except Exception as e:
                            print(f"Error processing file {file_path}: {e}")
                            
            # Print summary
            print(f"Processed {len(results)} files")
            
        else:
            print(f"Error: Input path not found: {args.input}")
            return 1
            
    elif args.action == 'status':
        # Get status of a SuperNode
        if not args.node_id:
            # List all nodes
            node_ids = manager.list_nodes()
            print(f"Available SuperNodes: {len(node_ids)}")
            for node_id in node_ids:
                status = manager.get_node_status(node_id)
                print(f"- {node_id}: {status['status']}")
        else:
            # Get status of specific node
            status = manager.get_node_status(args.node_id)
            if status:
                print(f"SuperNode: {args.node_id}")
                print(f"Status: {status['status']}")
                print(f"Uptime: {status['uptime']:.2f} seconds")
                print(f"Patterns: {status['pattern_count']}")
                print(f"Insights: {status['insight_count']}")
                print(f"Perspectives: {status['perspective_count']}")
                print("Core status:")
                for key, value in status['core_status'].items():
                    print(f"- {key}: {value}")
            else:
                print(f"SuperNode not found: {args.node_id}")
                return 1
                
    elif args.action == 'extract':
        # Extract knowledge from a SuperNode
        if not args.node_id:
            print("Error: --node-id required for 'extract' action")
            return 1
            
        # Get node
        node = manager.get_node(args.node_id)
        if not node:
            print(f"SuperNode not found: {args.node_id}")
            return 1
            
        # Extract knowledge
        knowledge = node.extract_knowledge()
        
        # Output knowledge
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(knowledge, f, indent=2)
            print(f"Knowledge written to: {args.output}")
        else:
            print("Extracted Knowledge:")
            print(f"Perspectives: {len(knowledge['perspectives'])}")
            print(f"Insights: {len(knowledge['insights'])}")
            
            # Print top perspectives
            print("\nTop Perspectives:")
            for i, perspective in enumerate(knowledge['perspectives'][:3]):
                print(f"{i+1}. {perspective['description']} (Impact: {perspective['impact']:.2f})")
                
            # Print top insights
            print("\nTop Insights:")
            for i, insight in enumerate(knowledge['insights'][:5]):
                print(f"{i+1}. {insight['description']} (Importance: {insight['importance']:.2f})")
                
    elif args.action == 'merge':
        # Merge SuperNodes
        if not args.node_ids:
            print("Error: --node-ids required for 'merge' action")
            return 1
            
        # Parse node IDs
        node_ids = args.node_ids.split(',')
        if len(node_ids) < 2:
            print("Error: At least two SuperNode IDs required for merge")
            return 1
            
        # Merge nodes
        merged_id = manager.merge_nodes(node_ids)
        if merged_id:
            print(f"Merged SuperNodes: {', '.join(node_ids)}")
            print(f"Merged SuperNode ID: {merged_id}")
        else:
            print("Error merging SuperNodes")
            return 1
            
    return 0

if __name__ == "__main__":
    sys.exit(main())
