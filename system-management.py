import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
from datetime import datetime
import logging

@dataclass
class SystemConfig:
    """System configuration settings."""
    # Memory Management
    memory_threshold: float = 0.8
    cache_size: int = 1000
    cleanup_interval: int = 3600  # seconds
    
    # Processing Settings
    batch_size: int = 100
    max_threads: int = 4
    processing_timeout: int = 300  # seconds
    
    # Learning Parameters
    learning_rate: float = 0.1
    exploration_rate: float = 0.2
    pattern_threshold: float = 0.7
    
    # Network Settings
    max_connections: int = 1000
    connection_timeout: int = 30  # seconds
    retry_attempts: int = 3
    
    # Storage Settings
    checkpoint_interval: int = 1800  # seconds
    max_checkpoints: int = 10
    storage_path: str = "data"

class SystemManager:
    """Manages system configuration and runtime behavior."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = SystemConfig()
        self.config_path = config_path or "config/system.yaml"
        self.components = {}
        self.running_tasks = {}
        self.task_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration if exists
        if Path(self.config_path).exists():
            self.load_config()
            
    def load_config(self):
        """Loads system configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Update configuration
            for key, value in config_data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    
            self.logger.info("Successfully loaded system configuration")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
            
    def save_config(self):
        """Saves current configuration to file."""
        try:
            config_dir = Path(self.config_path).parent
            config_dir.mkdir(exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(asdict(self.config), f)
                
            self.logger.info("Successfully saved system configuration")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise
            
    def register_component(self, name: str, component: Any):
        """Registers a system component."""
        self.components[name] = {
            'instance': component,
            'status': 'registered',
            'registered_at': datetime.now().isoformat()
        }
        
    def start_component(self, name: str):
        """Starts a registered component."""
        if name not in self.components:
            raise ValueError(f"Component not registered: {name}")
            
        component = self.components[name]['instance']
        
        try:
            if hasattr(component, 'start'):
                component.start()
            self.components[name]['status'] = 'running'
            self.components[name]['started_at'] = datetime.now().isoformat()
            
            self.logger.info(f"Started component: {name}")
            
        except Exception as e:
            self.components[name]['status'] = 'error'
            self.components[name]['error'] = str(e)
            self.logger.error(f"Error starting component {name}: {str(e)}")
            raise
            
    def stop_component(self, name: str):
        """Stops a running component."""
        if name not in self.components:
            raise ValueError(f"Component not registered: {name}")
            
        component = self.components[name]['instance']
        
        try:
            if hasattr(component, 'stop'):
                component.stop()
            self.components[name]['status'] = 'stopped'
            self.components[name]['stopped_at'] = datetime.now().isoformat()
            
            self.logger.info(f"Stopped component: {name}")
            
        except Exception as e:
            self.components[name]['status'] = 'error'
            self.components[name]['error'] = str(e)
            self.logger.error(f"Error stopping component {name}: {str(e)}")
            raise
            
    def get_component_status(self, name: str) -> Dict[str, Any]:
        """Gets status of a component."""
        if name not in self.components:
            raise ValueError(f"Component not registered: {name}")
            
        return self.components[name]
        
    def get_system_status(self) -> Dict[str, Any]:
        """Gets overall system status."""
        return {
            'components': {
                name: info['status']
                for name, info in self.components.items()
            },
            'tasks': len(self.running_tasks),
            'config': asdict(self.config)
        }
        
    def update_config(self, updates: Dict[str, Any]):
        """Updates system configuration."""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated configuration: {key} = {value}")
                
        # Save updated configuration
        self.save_config()
        
        # Notify components of configuration change
        self._notify_config_change(updates)
        
    def _notify_config_change(self, updates: Dict[str, Any]):
        """Notifies components of configuration changes."""
        for name, component in self.components.items():
            if hasattr(component['instance'], 'handle_config_update'):
                try:
                    component['instance'].handle_config_update(updates)
                except Exception as e:
                    self.logger.error(
                        f"Error updating configuration for component {name}: {str(e)}"
                    )
                    
    def register_task(self, task_id: str, task_info: Dict[str, Any]):
        """Registers a running task."""
        with self.task_lock:
            self.running_tasks[task_id] = {
                'info': task_info,
                'started_at': datetime.now().isoformat(),
                'status': 'running'
            }
            
    def update_task_status(self, task_id: str, status: str, result: Any = None):
        """Updates status of a running task."""
        with self.task_lock:
            if task_id in self.running_tasks:
                self.running_tasks[task_id].update({
                    'status': status,
                    'completed_at': datetime.now().isoformat(),
                    'result': result
                })
                
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Gets status of a task."""
        return self.running_tasks.get(task_id, {})
        
    def cleanup_tasks(self):
        """Cleans up completed tasks."""
        with self.task_lock:
            completed_tasks = [
                task_id for task_id, info in self.running_tasks.items()
                if info['status'] in ['completed', 'failed']
            ]
            
            for task_id in completed_tasks:
                del self.running_tasks[task_id]

class ComponentManager:
    """Manages individual system components."""
    
    def __init__(self, system_manager: SystemManager):
        self.system_manager = system_manager
        self.component_configs = {}
        
    def create_component(self, component_type: str, config: Dict[str, Any]) -> Any:
        """Creates a new component instance."""
        if component_type not in COMPONENT_REGISTRY:
            raise ValueError(f"Unknown component type: {component_type}")
            
        # Create component instance
        component_class = COMPONENT_REGISTRY[component_type]
        component = component_class(**config)
        
        # Register with system manager
        self.system_manager.register_component(
            f"{component_type}_{len(self.component_configs)}",
            component
        )
        
        # Store configuration
        self.component_configs[component] = config
        
        return component
        
    def configure_component(self, component: Any, config: Dict[str, Any]):
        """Configures a component."""
        if component not in self.component_configs:
            raise ValueError("Unknown component")
            
        # Update configuration
        self.component_configs[component].update(config)
        
        # Apply configuration to component
        if hasattr(component, 'configure'):
            component.configure(config)
            
    def get_component_config(self, component: Any) -> Dict[str, Any]:
        """Gets configuration of a component."""
        return self.component_configs.get(component, {})

# Registry of available component types
COMPONENT_REGISTRY = {
    'memory_bank': MemoryBank,
    'knowledge_graph': KnowledgeGraph,
    'pattern_recognizer': PatternRecognizer,
    'insight_generator': InsightGenerator
}
